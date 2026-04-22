"""Compatibility wrapper for the refactored src package."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from realtimestt.core.audio_input import AUDIO_FORMAT, CHANNELS, AudioInput, AudioInputConfig

DESIRED_RATE = 16000
CHUNK_SIZE = 1024

__all__ = [
    "AUDIO_FORMAT",
    "CHANNELS",
    "AudioInput",
    "AudioInputConfig",
    "DESIRED_RATE",
    "CHUNK_SIZE",
]
