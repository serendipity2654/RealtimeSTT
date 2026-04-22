# `src` standalone runtime

This `src` tree is independent from legacy `RealtimeSTT/*` runtime modules.

## Run CLI (microphone -> VAD -> transcription)

```bash
PYTHONPATH=src python3 -m realtimestt.cli.stt_cli --model tiny
```

Use dummy mode (no faster-whisper needed):

```bash
PYTHONPATH=src python3 -m realtimestt.cli.stt_cli --dummy-transcriber
```

## Run WebSocket server

```bash
PYTHONPATH=src python3 -m realtimestt.server.stt_server --model tiny --host 127.0.0.1 --control 8011 --data 8012
```

Dummy mode:

```bash
PYTHONPATH=src python3 -m realtimestt.server.stt_server --dummy-transcriber
```

## Key modules

- `realtimestt/vad/engine.py`: VAD abstraction and engine
- `realtimestt/stream/controller.py`: stream/VAD orchestration interface
- `realtimestt/cli/stt_cli.py`: CLI built on controller interface
- `realtimestt/server/stt_server.py`: server built on controller interface
