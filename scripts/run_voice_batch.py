"""Process a directory of WAV files through the live ZentroMart voice agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from conversation_agent import RAGConversationResponder
from conversation_runner import ConversationResult
from rag import RAGAnswer, answer_with_evidence
from retriever import EvidenceRetriever
from vector_index import DEFAULT_INDEX_DIR, load_vector_index
from voice_adapter import (
    GnaniSTT,
    GnaniTTS,
    VoiceConversationSession,
    pcm_bytes_to_stt_frames,
)


class BatchInputError(ValueError):
    """Invalid input path or audio format for the batch runner."""


def wav_file_to_stt_frames(audio_path: Path) -> tuple[bytes, ...]:
    """Decode PCM WAV and convert it to mono, 16 kHz, signed 16-bit STT frames."""
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            compression = wav_file.getcomptype()
            frame_count = wav_file.getnframes()
            raw_audio = wav_file.readframes(frame_count)
    except (wave.Error, EOFError, OSError) as error:
        raise BatchInputError(f"Cannot decode WAV audio ({type(error).__name__}).") from None

    if compression != "NONE":
        raise BatchInputError("Only uncompressed PCM WAV files are supported.")
    if channels not in {1, 2}:
        raise BatchInputError("WAV audio must have one or two channels.")
    if sample_rate < 1:
        raise BatchInputError("WAV sample rate must be greater than zero.")
    if sample_width not in {1, 2, 3, 4}:
        raise BatchInputError("WAV audio must use 8-, 16-, 24-, or 32-bit PCM.")
    expected_size = frame_count * channels * sample_width
    if frame_count == 0 or len(raw_audio) != expected_size:
        raise BatchInputError("WAV audio is empty or incomplete.")

    if sample_width == 1:
        samples = (np.frombuffer(raw_audio, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif sample_width == 2:
        samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.int32)
    elif sample_width == 3:
        bytes_24 = np.frombuffer(raw_audio, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_24[:, 0].astype(np.int32)
            | (bytes_24[:, 1].astype(np.int32) << 8)
            | (bytes_24[:, 2].astype(np.int32) << 16)
        )
        samples = ((values ^ 0x800000) - 0x800000) >> 8
    else:
        samples = np.frombuffer(raw_audio, dtype="<i4").astype(np.int64) >> 16

    samples = samples.reshape(-1, channels).astype(np.float64)
    mono = samples.mean(axis=1)
    if sample_rate != 16_000:
        try:
            from scipy.signal import resample_poly
        except ImportError:
            raise BatchInputError(
                "Resampling needs the existing SciPy dependency; install project "
                "requirements before using non-16-kHz WAV input."
            ) from None
        import math

        divisor = math.gcd(sample_rate, 16_000)
        mono = resample_poly(
            mono,
            16_000 // divisor,
            sample_rate // divisor,
        )

    pcm_data = np.clip(np.rint(mono), -32768, 32767).astype("<i2").tobytes()
    try:
        return pcm_bytes_to_stt_frames(
            pcm_data,
            sample_rate=16_000,
            channels=1,
            sample_width=2,
        )
    except (TypeError, ValueError) as error:
        raise BatchInputError(f"Could not prepare WAV audio for STT: {error}") from None


def _unique_run_directory(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        name = f"voice_batch_{timestamp}_{suffix:03d}"
        candidate = output_root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise OSError("Could not create a unique batch output directory.")


def _output_name(relative_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_path.stem).strip("._-")
    stem = (stem or "audio")[:48]
    digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def _validate_directories(
    input_directory: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    input_root = Path(input_directory).resolve()
    output_root = Path(output_directory).resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise BatchInputError(f"Input directory does not exist: {input_root}")
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise BatchInputError("Output directory must not be inside the input directory.")
    return input_root, output_root


def _serialize_evidence(hit: object) -> dict[str, object]:
    metadata = getattr(hit, "metadata")
    return {
        "evidence_id": getattr(hit, "evidence_id"),
        "text": getattr(hit, "text"),
        "source_text": getattr(hit, "source_text"),
        "score": getattr(hit, "score"),
        "metadata": metadata,
    }


def _capture_retrieved_evidence(
    answerer: Callable[..., RAGAnswer],
    evidence_capture: list[dict[str, object]],
) -> Callable[..., RAGAnswer]:
    def capture(*args: object, **kwargs: object) -> RAGAnswer:
        answer = answerer(*args, **kwargs)
        evidence_capture.extend(
            _serialize_evidence(hit) for hit in answer.evidence
        )
        return answer

    return capture


def _safe_error(error: Exception) -> str:
    return re.sub(r"(?<!\d)(?:\d[\s-]*){4,6}(?!\d)", "[redacted]", str(error))


SessionFactory = Callable[
    [str, str, list[dict[str, object]]],
    VoiceConversationSession,
]


def run_batch(
    input_directory: Path,
    output_directory: Path,
    *,
    session_factory: SessionFactory,
    language: str = "en",
) -> tuple[Path, list[dict[str, object]]]:
    """Run each file in a fresh session and preserve one status record per file."""
    input_root, output_root = _validate_directories(
        input_directory,
        output_directory,
    )

    files = sorted(path for path in input_root.rglob("*") if path.is_file())
    if not files:
        raise BatchInputError(f"No files found under input directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = _unique_run_directory(output_root)
    statuses: list[dict[str, object]] = []

    for audio_path in files:
        relative_path = audio_path.relative_to(input_root)
        file_directory = run_directory / _output_name(relative_path)
        file_directory.mkdir()
        status: dict[str, object] = {
            "source_file": relative_path.as_posix(),
            "session_id": f"voice-batch:{relative_path.as_posix()}",
            "status": "error",
            "stage": "validation",
            "error": None,
        }
        try:
            if audio_path.suffix.casefold() != ".wav":
                raise BatchInputError(
                    f"Unsupported audio type {audio_path.suffix or '[no extension]'}. "
                    "Supported type: PCM WAV (.wav)."
                )
            frames = wav_file_to_stt_frames(audio_path)
            status["stage"] = "stt_agent"
            evidence: list[dict[str, object]] = []
            session = session_factory(
                str(status["session_id"]),
                language,
                evidence,
            )
            result = session.process_audio(frames, language=language)
            if result.conversation_result is None:
                raise RuntimeError("STT returned an empty transcript; no agent turn ran.")

            conversation_result: ConversationResult = result.conversation_result
            (file_directory / "transcript.txt").write_text(
                result.transcript + "\n",
                encoding="utf-8",
            )
            (file_directory / "response.txt").write_text(
                conversation_result.response + "\n",
                encoding="utf-8",
            )
            (file_directory / "retrieved_chunks.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )

            status["stage"] = "tts"
            audio_bytes = b"".join(result.audio_chunks)
            if not audio_bytes:
                raise RuntimeError("TTS returned no audio.")
            (file_directory / "reply.wav").write_bytes(audio_bytes)
            status["status"] = "success"
            status["stage"] = "complete"
            status["decision"] = conversation_result.decision
            status["evidence_ids"] = list(conversation_result.evidence_ids)
            status["ticket_created"] = conversation_result.ticket_created
            status["handoff_performed"] = conversation_result.handoff_performed
        except Exception as error:
            status["error"] = {
                "type": type(error).__name__,
                "message": _safe_error(error),
            }

        statuses.append(status)
        try:
            (file_directory / "status.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            print(
                f"Could not save status for {relative_path.as_posix()}: "
                f"{type(error).__name__}",
                file=sys.stderr,
            )
        print(f"{status['status']}: {relative_path.as_posix()}")

    return run_directory, statuses


def _production_session_factory(index_directory: Path) -> SessionFactory:
    from sentence_transformers import SentenceTransformer

    index = load_vector_index(index_directory)
    encoder = SentenceTransformer(index.manifest["model_name"])
    retriever = EvidenceRetriever(index, encoder)

    def create_session(
        conversation_id: str,
        language: str,
        evidence_capture: list[dict[str, object]],
    ) -> VoiceConversationSession:
        responder = RAGConversationResponder(
            retriever,
            rag_answerer=_capture_retrieved_evidence(
                answer_with_evidence,
                evidence_capture,
            ),
        )
        from tools.tools import create_ticket, lookup_order

        return VoiceConversationSession(
            conversation_id,
            language,
            responder,
            GnaniSTT(),
            GnaniTTS(),
            lookup_order_fn=lookup_order,
            create_ticket_fn=create_ticket,
        )

    return create_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "hi", "hi-en"), default="en")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()

    try:
        _validate_directories(args.input_dir, args.output_dir)
        from llm_client import validate_configuration

        validate_configuration()
        session_factory = _production_session_factory(args.index_dir)
        run_directory, statuses = run_batch(
            args.input_dir,
            args.output_dir,
            session_factory=session_factory,
            language=args.language,
        )
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as error:
        print(f"Voice batch setup failed: {_safe_error(error)}", file=sys.stderr)
        return 2

    failures = sum(status["status"] != "success" for status in statuses)
    print(f"Saved per-file outputs under {run_directory}")
    print(f"Processed {len(statuses)} file(s); {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
