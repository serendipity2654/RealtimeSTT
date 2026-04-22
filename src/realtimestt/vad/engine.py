from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import audioop
import collections
import webrtcvad


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    aggressiveness: int = 2
    min_speech_frames: int = 3
    min_silence_frames: int = 10
    energy_threshold: int = 300


@dataclass
class VADState:
    is_speaking: bool = False
    speech_run: int = 0
    silence_run: int = 0


class VADEngine(Protocol):
    def process_frame(self, frame: bytes) -> bool:
        ...


class WebRTCVadEngine:
    """Low-latency VAD using WebRTC; includes a light RMS fallback gate."""

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._vad = webrtcvad.Vad()
        self._vad.set_mode(self.config.aggressiveness)
        self.state = VADState()

    @property
    def frame_bytes(self) -> int:
        return int(self.config.sample_rate * (self.config.frame_ms / 1000.0) * 2)

    def _is_frame_speech(self, frame: bytes) -> bool:
        try:
            speech = self._vad.is_speech(frame, self.config.sample_rate)
        except Exception:
            speech = False
        energy = audioop.rms(frame, 2)
        return speech or energy >= self.config.energy_threshold

    def process_frame(self, frame: bytes) -> bool:
        is_speech = self._is_frame_speech(frame)

        if is_speech:
            self.state.speech_run += 1
            self.state.silence_run = 0
            if not self.state.is_speaking and self.state.speech_run >= self.config.min_speech_frames:
                self.state.is_speaking = True
        else:
            self.state.silence_run += 1
            self.state.speech_run = 0
            if self.state.is_speaking and self.state.silence_run >= self.config.min_silence_frames:
                self.state.is_speaking = False

        return self.state.is_speaking


class FrameChunker:
    """Converts arbitrary PCM chunks into fixed-size VAD frames."""

    def __init__(self, frame_bytes: int):
        self.frame_bytes = frame_bytes
        self._buffer = bytearray()

    def push(self, pcm_chunk: bytes):
        self._buffer.extend(pcm_chunk)
        while len(self._buffer) >= self.frame_bytes:
            frame = bytes(self._buffer[: self.frame_bytes])
            del self._buffer[: self.frame_bytes]
            yield frame


class SpeechWindow:
    """Rolling audio window used to preserve pre-speech context."""

    def __init__(self, max_frames: int):
        self._frames = collections.deque(maxlen=max_frames)

    def append(self, frame: bytes):
        self._frames.append(frame)

    def dump(self) -> bytes:
        return b"".join(self._frames)

    def clear(self):
        self._frames.clear()
