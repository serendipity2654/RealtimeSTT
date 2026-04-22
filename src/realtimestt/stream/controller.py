from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import threading
import time

import numpy as np

from ..vad.engine import FrameChunker, SpeechWindow, VADConfig, WebRTCVadEngine


TranscribeFn = Callable[[np.ndarray, int], str]
EventFn = Callable[[], None]
TextFn = Callable[[str], None]


@dataclass
class ControllerConfig:
    sample_rate: int = 16000
    pre_speech_ms: int = 300
    min_utterance_ms: int = 250
    max_utterance_s: float = 20.0
    vad: VADConfig = field(default_factory=VADConfig)


class StreamController:
    """Unified streaming controller independent from legacy monolithic recorder."""

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        transcribe_fn: Optional[TranscribeFn] = None,
        on_speech_start: Optional[EventFn] = None,
        on_speech_end: Optional[EventFn] = None,
        on_partial_text: Optional[TextFn] = None,
        on_final_text: Optional[TextFn] = None,
    ):
        self.config = config or ControllerConfig()
        self.vad_engine = WebRTCVadEngine(self.config.vad)
        self.chunker = FrameChunker(self.vad_engine.frame_bytes)
        pre_frames = max(1, int(self.config.pre_speech_ms / self.config.vad.frame_ms))
        self.pre_window = SpeechWindow(pre_frames)

        self.transcribe_fn = transcribe_fn
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_partial_text = on_partial_text
        self.on_final_text = on_final_text

        self._lock = threading.Lock()
        self._speech_frames = []
        self._utterance_started_at = 0.0

    def feed_audio(self, pcm_chunk: bytes):
        with self._lock:
            for frame in self.chunker.push(pcm_chunk):
                self.pre_window.append(frame)
                speaking = self.vad_engine.process_frame(frame)

                if speaking and not self._speech_frames:
                    self._speech_frames.append(self.pre_window.dump())
                    self._utterance_started_at = time.time()
                    if self.on_speech_start:
                        self.on_speech_start()

                if speaking:
                    self._speech_frames.append(frame)
                    self._emit_partial()
                    if time.time() - self._utterance_started_at > self.config.max_utterance_s:
                        self._flush_utterance()
                    continue

                if self._speech_frames:
                    self._speech_frames.append(frame)
                    if not self.vad_engine.state.is_speaking:
                        self._flush_utterance()

    def force_flush(self):
        with self._lock:
            if self._speech_frames:
                self._flush_utterance()

    def _pcm_to_float32(self, pcm: bytes) -> np.ndarray:
        data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        return data / 32768.0

    def _emit_partial(self):
        if not self.on_partial_text or not self.transcribe_fn:
            return
        pcm = b"".join(self._speech_frames[-6:])
        if not pcm:
            return
        text = self.transcribe_fn(self._pcm_to_float32(pcm), self.config.sample_rate).strip()
        if text:
            self.on_partial_text(text)

    def _flush_utterance(self):
        utterance = b"".join(self._speech_frames)
        self._speech_frames.clear()

        min_bytes = int(self.config.sample_rate * (self.config.min_utterance_ms / 1000.0) * 2)
        if len(utterance) < min_bytes:
            return

        if self.on_speech_end:
            self.on_speech_end()

        if self.transcribe_fn and self.on_final_text:
            text = self.transcribe_fn(self._pcm_to_float32(utterance), self.config.sample_rate).strip()
            if text:
                self.on_final_text(text)
