"""Compatibility wrapper to keep legacy server entrypoint stable."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from realtimestt.server.stt_server import main


if __name__ == "__main__":
    main()
