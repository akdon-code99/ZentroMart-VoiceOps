import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock

from conversation_runner import ConversationResult
from rag import RAGAnswer
from retriever import RetrievalHit
from scripts.run_voice_batch import (
    BatchInputError,
    _capture_retrieved_evidence,
    run_batch,
    wav_file_to_stt_frames,
)
from voice_adapter import VoiceTurnResult


def write_wav(path, *, sample_rate=16_000, channels=1, sample_width=2, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames * channels * sample_width))


class FakeVoiceSession:
    def __init__(self, session_id, result=None, error=None, evidence=None):
        self.session_id = session_id
        self.result = result
        self.error = error
        self.evidence = evidence
        self.process_calls = []

    def process_audio(self, frames, *, language):
        self.process_calls.append((tuple(frames), language))
        if self.error:
            raise self.error
        self.evidence.append(
            {
                "evidence_id": "RET-2",
                "text": "Current return policy excerpt.",
                "metadata": {
                    "source_path": "kb/returns.md",
                    "section_id": "RET-2",
                    "status": "current",
                },
            }
        )
        return VoiceTurnResult(
            "I need help with a return.",
            ConversationResult(
                conversation_id=self.session_id,
                turn=1,
                customer_message="I need help with a return.",
                response="You can review the return policy. [RET-2]",
                decision="answer",
                evidence_ids=("RET-2",),
            ),
            (b"wav-audio-bytes",),
        )


class VoiceBatchConversionTests(unittest.TestCase):
    def test_pcm_wav_is_converted_to_required_stt_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            write_wav(path, sample_rate=8_000, channels=2, frames=800)

            frames = wav_file_to_stt_frames(path)

        self.assertEqual(len(frames), 4)
        self.assertTrue(all(len(frame) == 1024 for frame in frames))

    def test_rejects_empty_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.wav"
            write_wav(path, frames=0)
            with self.assertRaisesRegex(BatchInputError, "empty or incomplete"):
                wav_file_to_stt_frames(path)

    def test_rejects_corrupt_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.wav"
            path.write_bytes(b"not a wave")
            with self.assertRaises(BatchInputError):
                wav_file_to_stt_frames(path)


class VoiceBatchRunnerTests(unittest.TestCase):
    def test_evidence_capture_preserves_actual_retrieval_text_and_provenance(self):
        hit = RetrievalHit(
            evidence_id="RET-2",
            text="Returned section text.",
            source_text="Complete source text.",
            metadata={
                "source_path": "kb/returns.md",
                "section_id": "RET-2",
                "version": "4",
                "status": "current",
            },
            score=0.91,
        )
        expected = RAGAnswer("Use the return policy.", [hit])
        answerer = Mock(return_value=expected)
        captured = []
        wrapped = _capture_retrieved_evidence(answerer, captured)

        answer = wrapped("return policy?", "en", object())

        self.assertIs(answer, expected)
        self.assertEqual(captured, [{
            "evidence_id": "RET-2",
            "text": "Returned section text.",
            "source_text": "Complete source text.",
            "score": 0.91,
            "metadata": {
                "source_path": "kb/returns.md",
                "section_id": "RET-2",
                "version": "4",
                "status": "current",
            },
        }])

    def test_input_paths_are_validated_and_output_must_be_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(BatchInputError):
                run_batch(
                    root / "missing",
                    root / "output",
                    session_factory=Mock(),
                )

            input_dir = root / "input"
            input_dir.mkdir()
            write_wav(input_dir / "one.wav")
            with self.assertRaisesRegex(BatchInputError, "inside the input"):
                run_batch(
                    input_dir,
                    input_dir / "out",
                    session_factory=Mock(),
                )

    def test_each_file_gets_a_fresh_session_and_outputs_preserve_actual_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "audio"
            write_wav(input_dir / "first.wav")
            write_wav(input_dir / "nested" / "second.wav")
            created = []

            def factory(session_id, language, evidence):
                session = FakeVoiceSession(session_id, evidence=evidence)
                created.append((session, language))
                return session

            output_dir, statuses = run_batch(
                input_dir,
                root / "outputs",
                session_factory=factory,
            )

            self.assertEqual(len(created), 2)
            self.assertEqual(len({session.session_id for session, _ in created}), 2)
            self.assertTrue(all(language == "en" for _, language in created))
            self.assertTrue(all(status["status"] == "success" for status in statuses))
            per_file_dirs = list(output_dir.iterdir())
            self.assertEqual(len(per_file_dirs), 2)
            for file_dir in per_file_dirs:
                self.assertEqual(
                    (file_dir / "transcript.txt").read_text(encoding="utf-8").strip(),
                    "I need help with a return.",
                )
                self.assertIn(
                    "[RET-2]",
                    (file_dir / "response.txt").read_text(encoding="utf-8"),
                )
                self.assertEqual(
                    (file_dir / "reply.wav").read_bytes(),
                    b"wav-audio-bytes",
                )
                evidence = json.loads(
                    (file_dir / "retrieved_chunks.json").read_text(encoding="utf-8")
                )
                self.assertEqual(evidence[0]["evidence_id"], "RET-2")
                self.assertEqual(evidence[0]["text"], "Current return policy excerpt.")
                self.assertEqual(
                    evidence[0]["metadata"]["source_path"],
                    "kb/returns.md",
                )
                record = json.loads(
                    (file_dir / "status.json").read_text(encoding="utf-8")
                )
                self.assertEqual(record["status"], "success")
                self.assertEqual(record["evidence_ids"], ["RET-2"])

    def test_unsupported_file_and_failed_file_are_recorded_and_later_files_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "audio"
            write_wav(input_dir / "a-fails.wav")
            write_wav(input_dir / "b-succeeds.wav")
            (input_dir / "c-unsupported.mp3").write_bytes(b"not processed")
            calls = []

            def factory(session_id, language, evidence):
                self.assertEqual(language, "en")
                calls.append(session_id)
                if "a-fails" in session_id:
                    return FakeVoiceSession(
                        session_id,
                        error=RuntimeError("mock STT failure"),
                        evidence=evidence,
                    )
                return FakeVoiceSession(session_id, evidence=evidence)

            output_dir, statuses = run_batch(
                input_dir,
                root / "outputs",
                session_factory=factory,
            )

            self.assertEqual(len(statuses), 3)
            self.assertEqual([status["status"] for status in statuses], [
                "error",
                "success",
                "error",
            ])
            self.assertEqual(len(calls), 2)
            self.assertIn("mock STT failure", statuses[0]["error"]["message"])
            self.assertIn("Unsupported audio type", statuses[2]["error"]["message"])
            status_files = list(output_dir.glob("*/status.json"))
            self.assertEqual(len(status_files), 3)
            self.assertEqual(len(list(output_dir.glob("*/reply.wav"))), 1)

    def test_batch_outputs_use_a_new_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "audio"
            write_wav(input_dir / "one.wav")
            def factory(session_id, language, evidence):
                self.assertEqual(language, "en")
                return FakeVoiceSession(session_id, evidence=evidence)

            first, _ = run_batch(input_dir, root / "out", session_factory=factory)
            second, _ = run_batch(input_dir, root / "out", session_factory=factory)

            self.assertNotEqual(first, second)
            self.assertTrue(next(first.glob("*/status.json")).exists())
            self.assertTrue(next(second.glob("*/status.json")).exists())


if __name__ == "__main__":
    unittest.main()
