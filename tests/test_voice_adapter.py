import base64
import json
import io
import unittest
import wave
from unittest.mock import Mock
from urllib.error import HTTPError

import numpy as np

from conversation_runner import AgentReply
from scripts.test_gnani_stt import run_smoke_test
from voice_adapter import (
    GnaniAPIError,
    GnaniSTT,
    GnaniTTS,
    LocalAudioError,
    LocalAudioInterface,
    STTTranscriptSegment,
    STTTranscription,
    VoiceConversationSession,
    format_for_speech,
    pcm_bytes_to_stt_frames,
)


class FakeWebSocket:
    def __init__(self, events):
        self.events = list(events)
        self.sent_frames = []
        self.timeout = None
        self.closed = False

    def recv(self):
        if not self.events:
            raise TimeoutError("mock receive timeout")
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return json.dumps(event)

    def send_binary(self, payload):
        self.sent_frames.append(payload)

    def settimeout(self, timeout):
        self.timeout = timeout

    def close(self):
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, events):
        lines = []
        for event_name, data in events:
            lines.extend(
                (
                    f"event: {event_name}\n".encode(),
                    f"data: {json.dumps(data)}\n".encode(),
                    b"\n",
                )
            )
        self.lines = iter(lines)

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def __iter__(self):
        return self.lines


class FakeSTT:
    def __init__(self, transcripts):
        self.transcripts = iter(transcripts)
        self.calls = []

    def transcribe(self, pcm_frames, language, *, sample_rate=16_000):
        self.calls.append((language, sample_rate, tuple(pcm_frames)))
        return next(self.transcripts)


class FakeTTS:
    def __init__(self, fail=False, audio_chunks=(b"audio-chunk",)):
        self.calls = []
        self.fail = fail
        self.audio_chunks = audio_chunks

    def stream_speech(self, text, language):
        self.calls.append((text, language))
        if self.fail:
            raise GnaniAPIError("mock TTS failure")
        return iter(self.audio_chunks)


class FakeAudioBackend:
    def __init__(self, recording=None, error=None, play_error=None):
        self.recording = recording
        self.error = error
        self.play_error = play_error
        self.record_calls = []
        self.play_calls = []

    def rec(self, frames, **kwargs):
        self.record_calls.append((frames, kwargs))
        if self.error is not None:
            raise self.error
        return self.recording

    def play(self, data, **kwargs):
        self.play_calls.append((data.copy(), kwargs))
        if self.play_error is not None:
            raise self.play_error


def make_wav(data, *, sample_rate=48_000, channels=1, sample_width=2):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)
    return stream.getvalue()


class GnaniSTTTests(unittest.TestCase):
    def test_streams_pcm_and_joins_transcript_events(self):
        connection = FakeWebSocket(
            [
                {"type": "connected"},
                {"type": "processing"},
                {"type": "transcript", "text": "Please check"},
                {"type": "transcript", "text": "my order"},
                TimeoutError(),
            ]
        )
        connector = Mock(return_value=connection)
        pacing = Mock()
        stt = GnaniSTT(
            api_key="test-secret",
            websocket_connector=connector,
            frame_sleep=pacing,
        )

        transcript = stt.transcribe(
            (b"a" * 1024 for _ in range(2)),
            "en",
        )

        self.assertEqual(transcript, "Please check my order")
        connector.assert_called_once()
        args, kwargs = connector.call_args
        self.assertEqual(args[0], "wss://api.vachana.ai/stt/v3/stream")
        self.assertEqual(kwargs["header"]["lang_code"], "en-IN")
        self.assertEqual(kwargs["header"]["X-API-Key-ID"], "test-secret")
        self.assertEqual(kwargs["header"]["x-sample-rate"], "16000")
        self.assertEqual(connection.sent_frames, [b"a" * 1024, b"a" * 1024])
        self.assertEqual(pacing.call_args_list[0].args, (0.032,))
        self.assertTrue(connection.closed)

    def test_empty_transcript_is_returned_without_invoking_agent(self):
        connection = FakeWebSocket(
            [
                {"type": "connected"},
                {"type": "transcript", "text": "  "},
                TimeoutError(),
            ]
        )
        stt = GnaniSTT(
            api_key="test-secret",
            websocket_connector=Mock(return_value=connection),
            frame_sleep=lambda _: None,
        )

        self.assertEqual(stt.transcribe([b"a" * 1024], "hi"), "")

    def test_stt_api_error_is_safe_and_closes_connection(self):
        connection = FakeWebSocket(
            [
                {"type": "connected"},
                {"type": "error", "message": "sensitive server detail"},
            ]
        )
        stt = GnaniSTT(
            api_key="test-secret",
            websocket_connector=Mock(return_value=connection),
            frame_sleep=lambda _: None,
        )

        with self.assertRaises(GnaniAPIError) as raised:
            stt.transcribe([b"a" * 1024], "hi-en")

        self.assertNotIn("test-secret", str(raised.exception))
        self.assertNotIn("sensitive server detail", str(raised.exception))
        self.assertTrue(connection.closed)

    def test_invalid_frames_are_rejected(self):
        stt = GnaniSTT(
            api_key="test-secret",
            websocket_connector=Mock(),
            frame_sleep=lambda _: None,
        )

        with self.assertRaisesRegex(ValueError, "exactly 1024 bytes"):
            stt.transcribe([b"partial"], "en")

    def test_empty_audio_does_not_open_a_websocket(self):
        connector = Mock()
        stt = GnaniSTT(
            api_key="test-secret",
            websocket_connector=connector,
        )

        self.assertEqual(stt.transcribe([], "en"), "")
        connector.assert_not_called()

    def test_transcribe_with_metadata_returns_documented_segment_fields(self):
        connection = FakeWebSocket(
            [
                {"type": "connected"},
                {
                    "type": "transcript",
                    "timestamp": "2026-10-21T12:00:00Z",
                    "text": "hello there",
                    "audio_duration_ms": 1200,
                    "segment_id": "seg-1",
                    "segment_index": 0,
                    "latency": 320,
                },
                TimeoutError(),
            ]
        )
        stt = GnaniSTT(
            api_key="test-secret",
            websocket_connector=Mock(return_value=connection),
            frame_sleep=lambda _: None,
        )

        result = stt.transcribe_with_metadata([b"a" * 1024], "en")

        self.assertEqual(result.text, "hello there")
        self.assertEqual(
            result.segments,
            (
                STTTranscriptSegment(
                    text="hello there",
                    timestamp="2026-10-21T12:00:00Z",
                    audio_duration_ms=1200,
                    segment_id="seg-1",
                    segment_index=0,
                    latency=320,
                ),
            ),
        )


class GnaniTTSTests(unittest.TestCase):
    def test_decodes_streamed_audio_and_uses_documented_request(self):
        captured = {}
        response = FakeHTTPResponse(
            [
                ("start", {"status": "streaming_started"}),
                ("chunk", {"audio": base64.b64encode(b"first").decode()}),
                ("chunk", {"audio": base64.b64encode(b"second").decode()}),
                ("complete", {"is_final": True, "audio": ""}),
            ]
        )

        def open_request(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return response

        tts = GnaniTTS(api_key="test-secret", url_opener=open_request)
        chunks = list(tts.stream_speech("Order number 12345678 costs ₹49.", "hi-en"))

        self.assertEqual(chunks, [b"first", b"second"])
        self.assertEqual(captured["url"], "https://api.vachana.ai/api/v1/tts/sse")
        self.assertEqual(captured["headers"]["X-api-key-id"], "test-secret")
        self.assertEqual(captured["payload"]["model"], "timbre-v2.5")
        self.assertEqual(captured["payload"]["voice"], "Poorvi")
        self.assertEqual(captured["payload"]["language"], "hi-en")
        self.assertEqual(
            captured["payload"]["text"],
            "Order number 1234 5678 costs ₹49.",
        )
        self.assertEqual(captured["payload"]["audio_config"]["container"], "wav")

    def test_tts_server_error_is_not_exposed(self):
        response = FakeHTTPResponse(
            [("error", {"message": "internal error contains sensitive detail"})]
        )

        def return_error_response(_request, **_kwargs):
            return response

        tts = GnaniTTS(
            api_key="test-secret",
            url_opener=return_error_response,
        )

        with self.assertRaises(GnaniAPIError) as raised:
            list(tts.stream_speech("Hello", "en"))

        self.assertNotIn("test-secret", str(raised.exception))
        self.assertNotIn("sensitive detail", str(raised.exception))

    def test_http_error_reports_status_without_response_body(self):
        error = HTTPError(
            "https://api.vachana.ai/api/v1/tts/sse",
            401,
            "unauthorized",
            {},
            None,
        )
        def raise_http_error(*_args, **_kwargs):
            raise error

        tts = GnaniTTS(api_key="test-secret", url_opener=raise_http_error)

        with self.assertRaisesRegex(GnaniAPIError, r"HTTP 401\. Details:"):
            list(tts.stream_speech("Hello", "en"))


class VoiceSessionTests(unittest.TestCase):
    def test_turns_preserve_verification_state_without_replaying_lookups(self):
        stt = FakeSTT(
            [
                "My order number 12345678 arrived damaged.",
                "last four digits 9876",
                "Can you update me?",
            ]
        )
        tts = FakeTTS()
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {
                    "found": True,
                    "verified": True,
                    "status": "delivered",
                    "phone": "55559876",
                    "email": "private@example.com",
                    "delivery_pincode": "654321",
                },
            ]
        )
        ticket = Mock(return_value={"created": False})
        responder_contexts = []

        def responder(customer, context):
            responder_contexts.append((customer, context))
            if context.verification_status == "verified":
                return AgentReply(
                    "A support ticket is required. No direct handoff has occurred.",
                    decision="escalate",
                    required_action="high_value_damage",
                )
            return AgentReply("Please verify your order.", decision="clarify")

        session = VoiceConversationSession(
            "VOICE-1",
            "en",
            responder,
            stt,
            tts,
            lookup_order_fn=lookup,
            create_ticket_fn=ticket,
        )

        first = session.process_audio([b"a" * 1024])
        second = session.process_audio([b"b" * 1024])
        third = session.process_audio([b"c" * 1024])

        self.assertEqual(first.conversation_result.turn, 1)
        self.assertEqual(second.conversation_result.turn, 2)
        self.assertEqual(third.conversation_result.turn, 3)
        self.assertNotIn("9876", second.transcript)
        self.assertEqual(lookup.call_count, 2)
        self.assertEqual(ticket.call_count, 1)
        self.assertFalse(second.conversation_result.ticket_created)
        self.assertFalse(second.conversation_result.handoff_performed)
        self.assertIn("no human handoff has occurred", second.conversation_result.response)
        proof_context = responder_contexts[-1][1]
        self.assertNotIn("9876", repr(proof_context))
        self.assertNotIn("55559876", repr(proof_context))
        self.assertNotIn("private@example.com", repr(proof_context))
        self.assertNotIn("654321", repr(proof_context))
        self.assertNotIn("9876", repr(ticket.call_args.kwargs["summary"]))
        self.assertNotIn("9876", repr(tts.calls))
        self.assertEqual(len(session.runner.snapshot().results), 3)

    def test_empty_transcript_skips_agent_and_tts(self):
        stt = FakeSTT([""])
        tts = FakeTTS()
        responder = Mock(return_value="unexpected")
        session = VoiceConversationSession("VOICE-2", "en", responder, stt, tts)

        result = session.process_audio([b"a" * 1024])

        self.assertIsNone(result.conversation_result)
        responder.assert_not_called()
        self.assertEqual(tts.calls, [])

    def test_refusal_and_policy_evidence_pass_through_unchanged(self):
        stt = FakeSTT(["Can I bypass the return policy?"])
        tts = FakeTTS()
        ticket = Mock()
        response = AgentReply(
            "I can’t help bypass the return policy. [RET-1]",
            decision="refuse",
            evidence_ids=("RET-1",),
        )
        session = VoiceConversationSession(
            "VOICE-REFUSAL",
            "en",
            lambda _customer, _context: response,
            stt,
            tts,
            create_ticket_fn=ticket,
        )

        result = session.process_audio([b"a" * 1024])

        self.assertEqual(result.conversation_result.decision, "refuse")
        self.assertEqual(result.conversation_result.evidence_ids, ("RET-1",))
        self.assertEqual(result.conversation_result.response, response.response)
        self.assertEqual(tts.calls[0][0], response.response)
        ticket.assert_not_called()

    def test_tts_error_does_not_claim_ticket_or_handoff(self):
        stt = FakeSTT(["Please help me."])
        tts = FakeTTS(fail=True)
        session = VoiceConversationSession(
            "VOICE-3",
            "hi-en",
            lambda customer, context: AgentReply(
                "A human review is required.",
                decision="escalate",
                required_action="human_requested",
            ),
            stt,
            tts,
            create_ticket_fn=lambda **_kwargs: {"created": False},
        )

        with self.assertRaisesRegex(GnaniAPIError, "mock TTS failure"):
            session.process_audio([b"a" * 1024])
        self.assertEqual(tts.calls[0][1], "hi-en")
        result = session.runner.snapshot().results[0]
        self.assertFalse(result.ticket_created)
        self.assertFalse(result.handoff_performed)

    def test_language_can_change_between_turns(self):
        stt = FakeSTT(["Hello", "नमस्ते"])
        tts = FakeTTS()
        session = VoiceConversationSession(
            "VOICE-4",
            "en",
            lambda customer, _context: f"Reply in {customer.language}.",
            stt,
            tts,
        )

        session.process_audio([b"a" * 1024])
        session.process_audio([b"b" * 1024], language="hi")

        self.assertEqual([call[0] for call in stt.calls], ["en", "hi"])
        self.assertEqual([call[1] for call in tts.calls], ["en", "hi"])
        self.assertEqual(session.runner.snapshot().results[1].turn, 2)


class SpeechFormattingTests(unittest.TestCase):
    def test_formats_order_id_date_and_rupee_amount_without_changing_claims(self):
        self.assertEqual(
            format_for_speech(
                "Order 12345678 is due on 2026-10-24; the fee is ₹1,249.",
                "en",
            ),
            "Order 1234 5678 is due on 24 October 2026; the fee is ₹1,249.",
        )

    def test_rejects_unknown_speech_language(self):
        with self.assertRaises(ValueError):
            format_for_speech("Hello", "xx")


class GnaniSTTSmokeScriptTests(unittest.TestCase):
    def test_declining_confirmation_makes_no_recording_or_api_call(self):
        audio = Mock()
        stt = Mock()
        output = []

        status = run_smoke_test(
            input_fn=lambda _prompt: "no",
            output_fn=output.append,
            audio_interface=audio,
            stt=stt,
        )

        self.assertEqual(status, 0)
        audio.record_one_utterance.assert_not_called()
        stt.transcribe_with_metadata.assert_not_called()
        self.assertTrue(any("No recording or network request" in line for line in output))

    def test_yes_captures_up_to_five_seconds_and_makes_one_mock_request(self):
        audio = Mock()
        audio.record_one_utterance.return_value = (b"a" * 1024,)
        stt = Mock()
        stt.transcribe_with_metadata.return_value = STTTranscription(
            text="hello from microphone",
            segments=(
                STTTranscriptSegment(
                    text="hello from microphone",
                    audio_duration_ms=900,
                    segment_id="mock-segment",
                    segment_index=0,
                    latency=200,
                ),
            ),
        )
        output = []

        status = run_smoke_test(
            input_fn=lambda _prompt: "YES",
            output_fn=output.append,
            audio_interface=audio,
            stt=stt,
        )

        self.assertEqual(status, 0)
        audio.record_one_utterance.assert_called_once_with(5.0)
        stt.transcribe_with_metadata.assert_called_once_with(
            (b"a" * 1024,),
            "en",
            sample_rate=16_000,
        )
        rendered = "\n".join(output)
        self.assertIn("wss://api.vachana.ai/stt/v3/stream", rendered)
        self.assertIn("hello from microphone", rendered)
        self.assertIn("mock-segment", rendered)
        self.assertNotIn("test-secret", rendered)

    def test_empty_capture_does_not_call_api(self):
        audio = Mock()
        audio.record_one_utterance.return_value = ()
        stt = Mock()
        output = []

        status = run_smoke_test(
            input_fn=lambda _prompt: "YES",
            output_fn=output.append,
            audio_interface=audio,
            stt=stt,
        )

        self.assertEqual(status, 1)
        stt.transcribe_with_metadata.assert_not_called()
        self.assertTrue(any("no API request was made" in line for line in output))

    def test_api_error_is_reported_without_retry_or_secret_output(self):
        audio = Mock()
        audio.record_one_utterance.return_value = (b"a" * 1024,)
        stt = Mock()
        stt.transcribe_with_metadata.side_effect = GnaniAPIError(
            "Gnani STT timed out before returning a transcript."
        )
        output = []

        status = run_smoke_test(
            input_fn=lambda _prompt: "YES",
            output_fn=output.append,
            audio_interface=audio,
            stt=stt,
        )

        self.assertEqual(status, 1)
        stt.transcribe_with_metadata.assert_called_once()
        self.assertIn("timed out", "\n".join(output))
        self.assertNotIn("test-secret", "\n".join(output))


class PCMPreparationTests(unittest.TestCase):
    def test_pcm_is_split_into_stt_frames_and_final_frame_is_padded(self):
        frames = pcm_bytes_to_stt_frames(b"\x01\x02" * 513)

        self.assertEqual(len(frames), 2)
        self.assertEqual([len(frame) for frame in frames], [1024, 1024])
        self.assertEqual(frames[0], b"\x01\x02" * 512)
        self.assertEqual(frames[1][:2], b"\x01\x02")
        self.assertEqual(frames[1][2:], bytes(1022))

    def test_pcm_rejects_wrong_rate_channels_sample_width_and_incomplete_sample(self):
        invalid_formats = (
            ({"sample_rate": 8_000}, "16000 Hz"),
            ({"channels": 2}, "mono"),
            ({"sample_width": 1}, "signed 16-bit"),
        )
        for options, message in invalid_formats:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    pcm_bytes_to_stt_frames(b"\x00\x00", **options)
        with self.assertRaisesRegex(ValueError, "incomplete sample"):
            pcm_bytes_to_stt_frames(b"\x00")

    def test_empty_pcm_has_no_frames(self):
        self.assertEqual(pcm_bytes_to_stt_frames(b""), ())


class LocalAudioInterfaceTests(unittest.TestCase):
    def test_records_one_mono_16khz_int16_utterance(self):
        backend = FakeAudioBackend(np.array([1, 2, 3], dtype=np.int16))
        audio = LocalAudioInterface(backend=backend)

        frames = audio.record_one_utterance(2.0)

        self.assertEqual(backend.record_calls[0][0], 32_000)
        self.assertEqual(
            backend.record_calls[0][1],
            {
                "samplerate": 16_000,
                "channels": 1,
                "dtype": "int16",
                "blocking": True,
                "device": None,
            },
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][:6], b"\x01\x00\x02\x00\x03\x00")
        self.assertEqual(len(frames[0]), 1024)

    def test_empty_or_silent_recording_returns_no_frames(self):
        for recording in (np.array([], dtype=np.int16), np.zeros(16, dtype=np.int16)):
            with self.subTest(size=recording.size):
                audio = LocalAudioInterface(backend=FakeAudioBackend(recording))
                self.assertEqual(audio.record_one_utterance(), ())

    def test_permission_and_device_errors_are_actionable(self):
        cases = (
            ("permission denied", "permission was denied"),
            ("no default input device", "No usable microphone"),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                audio = LocalAudioInterface(
                    backend=FakeAudioBackend(error=RuntimeError(message))
                )
                with self.assertRaisesRegex(LocalAudioError, expected):
                    audio.record_one_utterance()

    def test_unsupported_recording_format_is_rejected(self):
        audio = LocalAudioInterface(
            backend=FakeAudioBackend(np.array([0.25], dtype=np.float32))
        )

        with self.assertRaisesRegex(LocalAudioError, "signed 16-bit PCM"):
            audio.record_one_utterance()

    def test_duration_and_capture_rate_are_validated_before_recording(self):
        backend = FakeAudioBackend(np.ones(2, dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "at most 15 seconds"):
            LocalAudioInterface(backend=backend).record_one_utterance(16)
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            LocalAudioInterface(backend=backend, sample_rate=48_000).record_one_utterance()
        self.assertEqual(backend.record_calls, [])

    def test_plays_concatenated_valid_wav_chunks(self):
        backend = FakeAudioBackend()
        audio = LocalAudioInterface(backend=backend)
        wav_data = make_wav(b"\x01\x00\x02\x00")

        audio.play_wav((wav_data[:12], wav_data[12:]))

        played, options = backend.play_calls[0]
        self.assertEqual(played.tolist(), [[1], [2]])
        self.assertEqual(options["samplerate"], 48_000)
        self.assertTrue(options["blocking"])

    def test_invalid_or_unsupported_wav_is_rejected_without_playback(self):
        backend = FakeAudioBackend()
        audio = LocalAudioInterface(backend=backend)
        with self.assertRaisesRegex(LocalAudioError, "valid PCM WAV"):
            audio.play_wav([b"not wav"])
        with self.assertRaisesRegex(LocalAudioError, "16-bit PCM"):
            audio.play_wav([make_wav(b"\x00", sample_width=1)])
        with self.assertRaisesRegex(LocalAudioError, "mono or stereo"):
            audio.play_wav([make_wav(b"\x00\x00\x00\x00", channels=3)])
        self.assertEqual(backend.play_calls, [])

    def test_truncated_wav_payload_is_rejected_without_playback(self):
        backend = FakeAudioBackend()
        audio = LocalAudioInterface(backend=backend)
        wav_data = make_wav(b"\x01\x00\x02\x00\x03\x00\x04\x00")

        with self.assertRaisesRegex(LocalAudioError, "empty or incomplete"):
            audio.play_wav([wav_data[:-4]])

        self.assertEqual(backend.play_calls, [])

    def test_empty_stream_and_playback_errors_are_reported(self):
        with self.assertRaisesRegex(LocalAudioError, "no playable audio"):
            LocalAudioInterface(backend=FakeAudioBackend()).play_wav([])

        audio = LocalAudioInterface(
            backend=FakeAudioBackend(play_error=RuntimeError("device error"))
        )
        with self.assertRaisesRegex(LocalAudioError, "output device"):
            audio.play_wav([make_wav(b"\x00\x00")])

    def test_run_voice_turn_reuses_session_and_plays_agent_audio(self):
        backend = FakeAudioBackend(np.array([1, 2], dtype=np.int16))
        audio = LocalAudioInterface(backend=backend)
        stt = FakeSTT(["Can I return this?"])
        wav_data = make_wav(b"\x01\x00\x02\x00")
        tts = FakeTTS(audio_chunks=(wav_data[:10], wav_data[10:]))
        session = VoiceConversationSession(
            "VOICE-LOCAL",
            "en",
            lambda _customer, _context: "Please check the current policy.",
            stt,
            tts,
        )

        result = audio.run_voice_turn(session, duration_seconds=1.0)

        self.assertEqual(result.transcript, "Can I return this?")
        self.assertEqual(result.conversation_result.turn, 1)
        self.assertEqual(list(result.audio_chunks), [wav_data[:10], wav_data[10:]])
        self.assertEqual(stt.calls[0][2][0][:4], b"\x01\x00\x02\x00")
        self.assertEqual(len(stt.calls[0][2][0]), 1024)
        self.assertEqual(len(backend.play_calls), 1)


if __name__ == "__main__":
    unittest.main()
