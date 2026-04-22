from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None


@dataclass
class TranscriberConfig:
    model: str = "tiny"
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 3
    language: Optional[str] = None


class FasterWhisperTranscriber:
    def __init__(self, config: Optional[TranscriberConfig] = None):
        self.config = config or TranscriberConfig()
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if WhisperModel is None:
            raise RuntimeError("faster_whisper is not installed")
        self._model = WhisperModel(
            model_size_or_path=self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(self, audio_f32, sample_rate: int) -> str:
        self._ensure_model()
        if sample_rate != 16000:
            logging.warning("Expected 16kHz input, got %sHz", sample_rate)
        segments, _ = self._model.transcribe(
            audio_f32,
            beam_size=self.config.beam_size,
            language=self.config.language,
        )
        return " ".join(seg.text for seg in segments)


class DummyTranscriber:
    def transcribe(self, audio_f32, sample_rate: int) -> str:
        duration = len(audio_f32) / max(1, sample_rate)
        return f"[audio {duration:.2f}s]"
