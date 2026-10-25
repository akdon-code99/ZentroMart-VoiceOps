"""Gnani voice adapters and a stateful bridge to the conversation agent."""

import base64
import importlib
import io
import json
import os
import re
import socket
import time
import wave
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from conversation_runner import (
    AgentReply,
    ConversationMessage,
    ConversationResult,
    ConversationRunner,
    CustomerInput,
    ConversationContext,
    OrderLookup,
    TicketCreator,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
_STT_ENDPOINT = "wss://api.vachana.ai/stt/v3/stream"
_TTS_ENDPOINT = "https://api.vachana.ai/api/v1/tts/sse"
_TTS_MODEL = "timbre-v2.5"
_TTS_OPTIONS = {
    "en": ("en-IN", "Kaveri"),
    "hi": ("hi-IN", "Nalini"),
    "hi-en": ("hi-en", "Poorvi"),
}
_STT_LANGUAGES = {
    "en": "en-IN",
    "en-IN": "en-IN",
    "hi": "hi-IN",
    "hi-IN": "hi-IN",
    "hi-en": "en-IN",
}
_MAX_SSE_LINE_BYTES = 1_000_000
_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


class GnaniAPIError(RuntimeError):
    """A safe Gnani error that never includes request credentials or raw payloads."""


@dataclass(frozen=True)
class STTTranscriptSegment:
    """Transcript text and documented metadata for one Gnani speech segment."""

    text: str
    timestamp: str | None = None
    audio_duration_ms: int | None = None
    segment_id: str | None = None
    segment_index: int | None = None
    latency: int | None = None


@dataclass(frozen=True)
class STTTranscription:
    """Combined transcript and per-segment metadata from one WebSocket session."""

    text: str
    segments: tuple[STTTranscriptSegment, ...]


class WebSocketConnection(Protocol):
    def recv(self) -> str | bytes: ...

    def send_binary(self, payload: bytes) -> object: ...

    def settimeout(self, timeout: float) -> object: ...

    def close(self) -> object: ...


WebSocketConnector = Callable[..., WebSocketConnection]
UrlOpener = Callable[..., object]
Responder = Callable[[CustomerInput, ConversationContext], str | AgentReply]
_STT_SAMPLE_RATE = 16_000
_STT_CHANNELS = 1
_STT_SAMPLE_WIDTH = 2
_STT_FRAME_BYTES = 1024
_MAX_RECORDING_SECONDS = 15.0


def _api_key_from_environment() -> str:
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GNANI_API_KEY in the project .env file.")
    return api_key


def _validate_pcm_frame(frame: object) -> None:
    if not isinstance(frame, bytes) or len(frame) != _STT_FRAME_BYTES:
        raise ValueError("Each Gnani STT PCM frame must be exactly 1024 bytes.")


def pcm_bytes_to_stt_frames(
    pcm_data: bytes,
    *,
    sample_rate: int = _STT_SAMPLE_RATE,
    channels: int = _STT_CHANNELS,
    sample_width: int = _STT_SAMPLE_WIDTH,
) -> tuple[bytes, ...]:
    """Validate PCM and pad its final chunk to Gnani's 1,024-byte frame size."""
    if sample_rate != _STT_SAMPLE_RATE:
        raise ValueError("STT audio must be sampled at 16000 Hz.")
    if channels != _STT_CHANNELS:
        raise ValueError("STT audio must be mono.")
    if sample_width != _STT_SAMPLE_WIDTH:
        raise ValueError("STT audio must use signed 16-bit PCM samples.")
    if not isinstance(pcm_data, bytes):
        raise TypeError("PCM audio must be provided as bytes.")
    if len(pcm_data) % sample_width:
        raise ValueError("PCM audio ends with an incomplete sample.")
    if not pcm_data:
        return ()

    frames = []
    for start in range(0, len(pcm_data), _STT_FRAME_BYTES):
        frame = pcm_data[start : start + _STT_FRAME_BYTES]
        if len(frame) < _STT_FRAME_BYTES:
            frame += bytes(_STT_FRAME_BYTES - len(frame))
        _validate_pcm_frame(frame)
        frames.append(frame)
    return tuple(frames)


def _language_key(language: str) -> str:
    normalized = language.strip().casefold()
    if normalized not in _TTS_OPTIONS:
        raise ValueError(f"Unsupported voice language: {language}.")
    return normalized


def _format_date(match: re.Match[str]) -> str:
    year, month, day = match.group("year", "month", "day")
    month_name = _MONTHS[int(month) - 1]
    return f"{int(day)} {month_name} {year}"


def format_for_speech(text: str, language: str) -> str:
    """Apply the capstone's number, date, and currency guidance conservatively."""
    _language_key(language)
    spoken = re.sub(
        r"(?i)\b(order(?:\s+(?:number|no\.?))?\s*[:#]?\s*)(\d{8})\b",
        lambda match: (
            match.group(1)
            + match.group(2)[:4]
            + " "
            + match.group(2)[4:]
        ),
        text,
    )
    spoken = re.sub(
        r"\b(?P<year>\d{4})-(?P<month>0?[1-9]|1[0-2])-(?P<day>0?[1-9]|[12]\d|3[01])\b",
        _format_date,
        spoken,
    )
    month_pattern = "|".join(_MONTHS)
    spoken = re.sub(
        rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{month_pattern})"
        rf"(?:\s+(?P<year>\d{{4}}))?\b",
        lambda match: (
            f"{int(match.group('day'))} {match.group('month')}"
            + (f" {match.group('year')}" if match.group("year") else "")
        ),
        spoken,
        flags=re.IGNORECASE,
    )
    return spoken


@dataclass(frozen=True)
class GnaniSTT:
    """Transcribe raw PCM frames using Gnani's documented STT WebSocket."""

    api_key: str = field(default="", repr=False)
    websocket_connector: WebSocketConnector | None = field(default=None, repr=False)
    frame_sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    timeout_seconds: float = 12.0

    def transcribe(
        self,
        pcm_frames: Iterable[bytes],
        language: str,
        *,
        sample_rate: int = 16_000,
    ) -> str:
        return self.transcribe_with_metadata(
            pcm_frames,
            language,
            sample_rate=sample_rate,
        ).text

    def transcribe_with_metadata(
        self,
        pcm_frames: Iterable[bytes],
        language: str,
        *,
        sample_rate: int = 16_000,
    ) -> STTTranscription:
        language_code = _STT_LANGUAGES.get(language.strip())
        if language_code is None:
            raise ValueError(f"Unsupported STT language: {language}.")
        if sample_rate not in {8_000, 16_000, 44_100, 48_000}:
            raise ValueError("STT sample rate must be 8000, 16000, 44100, or 48000 Hz.")

        frames = iter(pcm_frames)
        first_frame = next(frames, None)
        if first_frame is None:
            return STTTranscription(text="", segments=())
        _validate_pcm_frame(first_frame)

        api_key = self.api_key.strip() or _api_key_from_environment()
        websocket_module = None
        connector = self.websocket_connector
        if connector is None:
            try:
                websocket_module = importlib.import_module("websocket")
            except ImportError:
                raise RuntimeError(
                    "Gnani STT needs the websocket-client dependency; "
                    "install the project requirements before live use."
                ) from None
            connector = websocket_module.create_connection

        connection = connector(
            _STT_ENDPOINT,
            header={
                "X-API-Key-ID": api_key,
                "lang_code": language_code,
                "x-sample-rate": str(sample_rate),
            },
            timeout=self.timeout_seconds,
        )
        timeout_errors: tuple[type[BaseException], ...] = (
            TimeoutError,
            socket.timeout,
        )
        closed_errors: tuple[type[BaseException], ...] = ()
        if websocket_module is not None:
            timeout_errors += (websocket_module.WebSocketTimeoutException,)
            closed_errors = (websocket_module.WebSocketConnectionClosedException,)

        transcript_parts: list[str] = []
        transcript_segments: list[STTTranscriptSegment] = []
        transcript_received = False
        frame_duration_seconds = 1024 / (sample_rate * 2)
        try:
            connection.settimeout(self.timeout_seconds)
            connected = self._receive_event(connection)
            if connected.get("type") != "connected":
                raise GnaniAPIError("Gnani STT did not confirm the WebSocket connection.")

            for frame in chain((first_frame,), frames):
                _validate_pcm_frame(frame)
                connection.send_binary(frame)
                self.frame_sleep(frame_duration_seconds)

            while True:
                try:
                    event = self._receive_event(connection)
                except timeout_errors:
                    if transcript_received:
                        break
                    raise GnaniAPIError("Gnani STT timed out before returning a transcript.") from None
                except closed_errors:
                    if transcript_received:
                        break
                    raise GnaniAPIError("Gnani STT closed before returning a transcript.") from None

                event_type = event.get("type")
                if event_type == "error":
                    raise GnaniAPIError("Gnani STT returned an API error.")
                if event_type == "transcript":
                    transcript_received = True
                    text = event.get("text")
                    segment_text = text.strip() if isinstance(text, str) else ""
                    if segment_text:
                        transcript_parts.append(segment_text)
                    transcript_segments.append(
                        STTTranscriptSegment(
                            text=segment_text,
                            timestamp=(
                                event.get("timestamp")
                                if isinstance(event.get("timestamp"), str)
                                else None
                            ),
                            audio_duration_ms=(
                                event.get("audio_duration_ms")
                                if isinstance(event.get("audio_duration_ms"), int)
                                else None
                            ),
                            segment_id=(
                                event.get("segment_id")
                                if isinstance(event.get("segment_id"), str)
                                else None
                            ),
                            segment_index=(
                                event.get("segment_index")
                                if isinstance(event.get("segment_index"), int)
                                else None
                            ),
                            latency=(
                                event.get("latency")
                                if isinstance(event.get("latency"), int)
                                else None
                            ),
                        )
                    )
                    connection.settimeout(min(0.25, self.timeout_seconds))
        except GnaniAPIError:
            raise
        except (OSError, ValueError, json.JSONDecodeError):
            raise
        except Exception:
            raise GnaniAPIError("Gnani STT WebSocket request failed.") from None
        finally:
            connection.close()

        return STTTranscription(
            text=" ".join(transcript_parts).strip(),
            segments=tuple(transcript_segments),
        )

    @staticmethod
    def _receive_event(connection: WebSocketConnection) -> Mapping[str, object]:
        raw_event = connection.recv()
        if isinstance(raw_event, bytes):
            raw_event = raw_event.decode("utf-8")
        try:
            event = json.loads(raw_event)
        except (TypeError, json.JSONDecodeError):
            raise GnaniAPIError("Gnani STT returned an invalid event.") from None
        if not isinstance(event, dict):
            raise GnaniAPIError("Gnani STT returned an invalid event.")
        return event


@dataclass(frozen=True)
class GnaniTTS:
    """Stream Gnani TTS audio from the documented Server-Sent Events endpoint."""

    api_key: str = field(default="", repr=False)
    url_opener: UrlOpener = field(default=urlopen, repr=False)
    timeout_seconds: float = 30.0

    def stream_speech(self, text: str, language: str) -> Iterator[bytes]:
        if not text.strip():
            return
        language_code, voice = _TTS_OPTIONS[_language_key(language)]
        api_key = self.api_key.strip() or _api_key_from_environment()
        payload = {
            "text": format_for_speech(text.strip(), language),
            "voice": voice,
            "model": _TTS_MODEL,
            "language": language_code,
            "speed": 1.0,
            "audio_config": {
                "sample_rate": 48_000,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "linear_pcm",
                "container": "wav",
            },
        }
        request = Request(
            _TTS_ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key-ID": api_key,
            },
            method="POST",
        )
        try:
            with self.url_opener(request, timeout=self.timeout_seconds) as response:
                yield from self._decode_sse(response)

        except HTTPError as error:
            try:
                details = error.read(1000).decode("utf-8", errors="replace")
            except Exception:
                details = ""

            raise GnaniAPIError(
                f"Gnani TTS returned HTTP {error.code}. "
                f"Details: {details[:1000]}"
            ) from None
        except (URLError, TimeoutError, OSError):
            raise GnaniAPIError("Could not reach Gnani TTS.") from None


    @staticmethod
    def _decode_sse(response: object) -> Iterator[bytes]:
        event_name = ""
        data_lines: list[str] = []
        completed = False

        def consume_event() -> tuple[str, Mapping[str, object] | None]:
            nonlocal event_name, data_lines
            current_event = event_name
            current_data = "\n".join(data_lines)
            event_name = ""
            data_lines = []
            if not current_data:
                return current_event, None
            try:
                value = json.loads(current_data)
            except json.JSONDecodeError:
                raise GnaniAPIError("Gnani TTS returned an invalid stream event.") from None
            if not isinstance(value, dict):
                raise GnaniAPIError("Gnani TTS returned an invalid stream event.")
            return current_event, value

        for raw_line in response:  # type: ignore[union-attr]
            if len(raw_line) > _MAX_SSE_LINE_BYTES:
                raise GnaniAPIError("Gnani TTS returned an oversized stream event.")
            try:
                line = raw_line.decode("utf-8").rstrip("\r\n")
            except (AttributeError, UnicodeDecodeError):
                raise GnaniAPIError("Gnani TTS returned an invalid stream event.") from None
            if not line:
                event, data = consume_event()
                if event == "error":
                    raise GnaniAPIError("Gnani TTS returned an API error.")
                if data is not None and event == "chunk":
                    encoded_audio = data.get("audio")
                    if not isinstance(encoded_audio, str):
                        raise GnaniAPIError("Gnani TTS returned an invalid audio chunk.")
                    try:
                        chunk = base64.b64decode(encoded_audio, validate=True)
                    except (ValueError, base64.binascii.Error):
                        raise GnaniAPIError("Gnani TTS returned an invalid audio chunk.") from None
                    if chunk:
                        yield chunk
                if event == "complete":
                    completed = True
                    break
                continue
            if line.startswith(":"):
                continue
            field_name, separator, field_value = line.partition(":")
            if not separator:
                field_value = ""
            elif field_value.startswith(" "):
                field_value = field_value[1:]
            if field_name == "event":
                event_name = field_value
            elif field_name == "data":
                data_lines.append(field_value)

        if not completed:
            event, data = consume_event()
            if event == "error":
                raise GnaniAPIError("Gnani TTS returned an API error.")
            if data is not None and event == "chunk":
                encoded_audio = data.get("audio")
                if not isinstance(encoded_audio, str):
                    raise GnaniAPIError("Gnani TTS returned an invalid audio chunk.")
                try:
                    chunk = base64.b64decode(encoded_audio, validate=True)
                except (ValueError, base64.binascii.Error):
                    raise GnaniAPIError("Gnani TTS returned an invalid audio chunk.") from None
                if chunk:
                    yield chunk
            completed = event == "complete"
        if not completed:
            raise GnaniAPIError("Gnani TTS stream ended before completion.")


class SpeechToText(Protocol):
    def transcribe(
        self,
        pcm_frames: Iterable[bytes],
        language: str,
        *,
        sample_rate: int = 16_000,
    ) -> str: ...


class TextToSpeech(Protocol):
    def stream_speech(self, text: str, language: str) -> Iterable[bytes]: ...


class AudioBackend(Protocol):
    def rec(self, frames: int, **kwargs: object) -> object: ...

    def play(self, data: object, **kwargs: object) -> object: ...


class LocalAudioError(RuntimeError):
    """A safe, actionable local microphone or playback error."""


@dataclass
class LocalAudioInterface:
    """Capture short PCM utterances and play streamed 16-bit WAV responses."""

    backend: AudioBackend | None = field(default=None, repr=False)
    sample_rate: int = _STT_SAMPLE_RATE
    device: int | str | None = None

    def _audio_backend(self) -> AudioBackend:
        if self.backend is not None:
            return self.backend
        try:
            return importlib.import_module("sounddevice")
        except ImportError:
            raise LocalAudioError(
                "Local audio needs the sounddevice dependency; "
                "install the project requirements before recording or playback."
            ) from None

    def record_one_utterance(
        self,
        duration_seconds: float = 5.0,
    ) -> tuple[bytes, ...]:
        """Record at most 15 seconds from the default or selected microphone."""
        if not 0 < duration_seconds <= _MAX_RECORDING_SECONDS:
            raise ValueError("Recording duration must be greater than 0 and at most 15 seconds.")
        if self.sample_rate != _STT_SAMPLE_RATE:
            raise ValueError("Local capture currently requires a 16000 Hz device stream.")

        backend = self._audio_backend()
        try:
            samples = backend.rec(
                round(duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=_STT_CHANNELS,
                dtype="int16",
                blocking=True,
                device=self.device,
            )
        except Exception as error:
            detail = str(error).casefold()
            error_name = type(error).__name__.casefold()
            if any(word in detail for word in ("permission", "access denied", "not authorized")):
                raise LocalAudioError(
                    "Microphone permission was denied. Allow microphone access and try again."
                ) from None
            if any(word in detail for word in ("no default input", "no input device", "device unavailable")):
                raise LocalAudioError(
                    "No usable microphone is available. Connect or select an input device."
                ) from None
            if "portaudio" in error_name:
                raise LocalAudioError(
                    "Microphone recording failed. Check the selected device and its format."
                ) from None
            raise LocalAudioError("Could not record microphone audio.") from None

        try:
            import numpy as np

            audio = np.asarray(samples)
        except (ImportError, TypeError, ValueError):
            raise LocalAudioError("The audio device returned an unsupported recording format.") from None
        if audio.size == 0:
            return ()
        if audio.dtype.kind != "i" or audio.dtype.itemsize != _STT_SAMPLE_WIDTH:
            raise LocalAudioError("The microphone must provide signed 16-bit PCM audio.")
        if audio.ndim == 2:
            if audio.shape[1] != _STT_CHANNELS:
                raise LocalAudioError("The microphone recording must be mono.")
            audio = audio.reshape(-1)
        elif audio.ndim != 1:
            raise LocalAudioError("The microphone returned an unsupported audio channel layout.")
        pcm_data = audio.astype("<i2", copy=False).tobytes()
        if not pcm_data or not any(pcm_data):
            return ()
        return pcm_bytes_to_stt_frames(pcm_data, sample_rate=self.sample_rate)

    def play_wav(self, audio_chunks: Iterable[bytes]) -> None:
        """Buffer and play TTS WAV chunks, rejecting malformed/unsupported audio."""
        try:
            audio_data = b"".join(audio_chunks)
        except Exception:
            raise LocalAudioError("Could not receive the streamed TTS audio.") from None
        if not audio_data:
            raise LocalAudioError("The TTS stream contained no playable audio.")

        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                compression = wav_file.getcomptype()
                declared_frames = wav_file.getnframes()
                frames = wav_file.readframes(declared_frames)
        except (wave.Error, EOFError, OSError):
            raise LocalAudioError("TTS audio must be a valid PCM WAV file.") from None
        if compression != "NONE" or sample_width != 2:
            raise LocalAudioError("Playback supports only uncompressed 16-bit PCM WAV audio.")
        if channels not in {1, 2}:
            raise LocalAudioError("Playback supports only mono or stereo WAV audio.")
        if sample_rate not in {8_000, 16_000, 22_050, 24_000, 44_100, 48_000}:
            raise LocalAudioError("The TTS WAV sample rate is unsupported.")
        expected_bytes = declared_frames * sample_width * channels
        if not frames or len(frames) != expected_bytes:
            raise LocalAudioError("The TTS WAV audio is empty or incomplete.")

        try:
            import numpy as np

            audio = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
            self._audio_backend().play(
                audio,
                samplerate=sample_rate,
                blocking=True,
                device=self.device,
            )
        except ImportError:
            raise LocalAudioError(
                "Local playback needs NumPy and sounddevice from the project requirements."
            ) from None
        except Exception as error:
            if isinstance(error, LocalAudioError):
                raise
            raise LocalAudioError(
                "Could not play TTS audio. Check the selected output device."
            ) from None

    def run_voice_turn(
        self,
        session: "VoiceConversationSession",
        *,
        duration_seconds: float = 5.0,
        language: str | None = None,
    ) -> "VoiceTurnResult":
        """Capture one utterance, pass it through the existing session, and play audio."""
        frames = self.record_one_utterance(duration_seconds)
        result = session.process_audio(frames, language=language)
        if result.conversation_result is not None:
            audio_chunks = tuple(result.audio_chunks)
            self.play_wav(audio_chunks)
            return VoiceTurnResult(
                transcript=result.transcript,
                conversation_result=result.conversation_result,
                audio_chunks=audio_chunks,
            )
        return result


@dataclass(frozen=True)
class VoiceTurnResult:
    transcript: str
    conversation_result: ConversationResult | None
    audio_chunks: Iterable[bytes]


class VoiceConversationSession:
    """Bridge audio turns to one persistent instance of the conversation runner."""

    def __init__(
        self,
        conversation_id: str,
        language: str,
        responder: Responder,
        stt: SpeechToText,
        tts: TextToSpeech,
        lookup_order_fn: OrderLookup | None = None,
        create_ticket_fn: TicketCreator | None = None,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        if not conversation_id.strip():
            raise ValueError("Conversation ID must not be empty.")
        _language_key(language)
        self.conversation_id = conversation_id
        self.language = language
        self.sample_rate = sample_rate
        self._stt = stt
        self._tts = tts
        self._turn = 0
        self.runner = ConversationRunner(
            responder,
            lookup_order_fn=lookup_order_fn,
            create_ticket_fn=create_ticket_fn,
        )

    def process_audio(
        self,
        pcm_frames: Iterable[bytes],
        *,
        language: str | None = None,
    ) -> VoiceTurnResult:
        turn_language = language or self.language
        _language_key(turn_language)
        transcript = self._stt.transcribe(
            pcm_frames,
            turn_language,
            sample_rate=self.sample_rate,
        ).strip()
        if not transcript:
            return VoiceTurnResult("", None, ())

        next_turn = self._turn + 1
        result = self.runner.run_turn(
            ConversationMessage(
                conversation_id=self.conversation_id,
                turn=next_turn,
                language=turn_language,
                customer_message=transcript,
            )
        )
        self._turn = next_turn
        audio_chunks = self._tts.stream_speech(
            format_for_speech(result.response, turn_language),
            turn_language,
        )
        return VoiceTurnResult(result.customer_message, result, audio_chunks)
