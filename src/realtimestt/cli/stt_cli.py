from __future__ import annotations

import argparse
import signal
import sys
import time

from colorama import Fore, Style, init

from ..core.audio_input import AudioInput
from ..stream import ControllerConfig, DummyTranscriber, FasterWhisperTranscriber, StreamController, TranscriberConfig
from ..vad.engine import VADConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RealtimeSTT CLI (src-only)")
    p.add_argument("-L", "--list", action="store_true", help="List audio input devices")
    p.add_argument("-i", "--input-device", type=int, help="Audio input device index")
    p.add_argument("-m", "--model", default="tiny", help="faster-whisper model")
    p.add_argument("-l", "--language", default=None, help="Force language")
    p.add_argument("--device", default="auto", help="Transcriber device")
    p.add_argument("--compute-type", default="default")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--frame-ms", type=int, default=30, choices=[10, 20, 30])
    p.add_argument("--vad-aggr", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--min-speech-frames", type=int, default=3)
    p.add_argument("--min-silence-frames", type=int, default=10)
    p.add_argument("--energy-threshold", type=int, default=300)
    p.add_argument("--dummy-transcriber", action="store_true", help="Run without faster-whisper")
    return p


def run_cli(args=None) -> int:
    init()
    opts = build_arg_parser().parse_args(args=args)

    audio = AudioInput(
        input_device_index=opts.input_device,
        target_samplerate=opts.sample_rate,
        chunk_size=1024,
    )

    if opts.list:
        audio.list_devices()
        return 0

    if not audio.setup():
        print("Failed to initialize microphone", file=sys.stderr)
        return 1

    if opts.dummy_transcriber:
        transcriber = DummyTranscriber()
    else:
        transcriber = FasterWhisperTranscriber(
            TranscriberConfig(
                model=opts.model,
                device=opts.device,
                compute_type=opts.compute_type,
                beam_size=opts.beam_size,
                language=opts.language,
            )
        )

    def on_start():
        print(f"{Fore.GREEN}[speech start]{Style.RESET_ALL}")

    def on_end():
        print(f"{Fore.BLUE}[speech end]{Style.RESET_ALL}")

    def on_partial(text: str):
        sys.stdout.write(f"\r{Fore.YELLOW}{text}{Style.RESET_ALL}")
        sys.stdout.flush()

    def on_final(text: str):
        sys.stdout.write("\r\033[K")
        print(text)

    controller = StreamController(
        config=ControllerConfig(
            sample_rate=opts.sample_rate,
            vad=VADConfig(
                sample_rate=opts.sample_rate,
                frame_ms=opts.frame_ms,
                aggressiveness=opts.vad_aggr,
                min_speech_frames=opts.min_speech_frames,
                min_silence_frames=opts.min_silence_frames,
                energy_threshold=opts.energy_threshold,
            ),
        ),
        transcribe_fn=transcriber.transcribe,
        on_speech_start=on_start,
        on_speech_end=on_end,
        on_partial_text=on_partial,
        on_final_text=on_final,
    )

    running = True

    def _stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while running:
            chunk = audio.read_chunk()
            controller.feed_audio(chunk)
            time.sleep(0.001)
    finally:
        controller.force_flush()
        audio.cleanup()

    return 0


def main():
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
