"""One-shot, confirmation-gated local microphone test for Gnani STT."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from voice_adapter import (
    GnaniAPIError,
    GnaniSTT,
    LocalAudioError,
    LocalAudioInterface,
    STTTranscription,
    _api_key_from_environment,
)

_MAX_RECORDING_SECONDS = 5.0
_RESPONSE_TIMEOUT_SECONDS = 8.0
_REQUEST_DESCRIPTION = """This test will:
- record one local microphone utterance (up to 5 seconds; not saved to disk)
- avoid speaking account credentials or order-verification values
- make one Gnani STT WebSocket request to wss://api.vachana.ai/stt/v3/stream
- send raw mono signed 16-bit little-endian PCM at 16 kHz in 1,024-byte frames
- authenticate with X-API-Key-ID loaded from .env (the key will not be displayed)
- request English transcription using lang_code=en-IN
- wait at most 8 seconds for the response; no retries
"""


class AudioRecorder(Protocol):
    def record_one_utterance(self, duration_seconds: float) -> tuple[bytes, ...]: ...


class STTTranscriber(Protocol):
    def transcribe_with_metadata(
        self,
        pcm_frames: Iterable[bytes],
        language: str,
        *,
        sample_rate: int,
    ) -> STTTranscription: ...


def run_smoke_test(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], object] = print,
    audio_interface: AudioRecorder | None = None,
    stt: STTTranscriber | None = None,
) -> int:
    output_fn(_REQUEST_DESCRIPTION)
    confirmation = input_fn("Type YES to record and send one request, or anything else to cancel: ")
    if confirmation.strip() != "YES":
        output_fn("Cancelled. No recording or network request was made.")
        return 0

    audio = audio_interface or LocalAudioInterface(sample_rate=16_000)
    try:
        transcriber = stt or GnaniSTT(
            api_key=_api_key_from_environment(),
            timeout_seconds=_RESPONSE_TIMEOUT_SECONDS,
        )
        pcm_frames = audio.record_one_utterance(_MAX_RECORDING_SECONDS)
        if not pcm_frames:
            output_fn("No non-silent audio was captured; no API request was made.")
            return 1

        output_fn(
            f"Captured {len(pcm_frames)} PCM frame(s). Sending one STT request..."
        )
        result = transcriber.transcribe_with_metadata(
            pcm_frames,
            "en",
            sample_rate=16_000,
        )
    except (GnaniAPIError, LocalAudioError, RuntimeError, ValueError) as error:
        output_fn(f"STT smoke test failed: {error}")
        return 1
    except OSError as error:
        output_fn(
            "STT smoke test failed with a network or audio I/O error "
            f"({type(error).__name__}); no retry was attempted."
        )
        return 1

    output_fn(f"Transcript: {result.text or '[empty transcript]'}")
    if result.segments:
        output_fn("Transcript segment metadata:")
        for index, segment in enumerate(result.segments, start=1):
            metadata = {
                "timestamp": segment.timestamp,
                "audio_duration_ms": segment.audio_duration_ms,
                "segment_id": segment.segment_id,
                "segment_index": segment.segment_index,
                "latency": segment.latency,
            }
            output_fn(f"  Segment {index}: {metadata}")
    else:
        output_fn("No transcript segment metadata was returned.")
    return 0


def main() -> int:
    return run_smoke_test()


if __name__ == "__main__":
    raise SystemExit(main())
