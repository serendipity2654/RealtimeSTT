from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from colorama import Fore, Style, init
from scipy.signal import butter, filtfilt, resample_poly
import logging
import pyaudio

from .constants import DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLE_RATE

AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1


@dataclass
class AudioInputConfig:
    input_device_index: Optional[int] = None
    debug_mode: bool = False
    target_samplerate: int = DEFAULT_SAMPLE_RATE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    audio_format: int = AUDIO_FORMAT
    channels: int = CHANNELS
    resample_to_target: bool = True


class AudioInput:
    def __init__(
        self,
        input_device_index: Optional[int] = None,
        debug_mode: bool = False,
        target_samplerate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        audio_format: int = AUDIO_FORMAT,
        channels: int = CHANNELS,
        resample_to_target: bool = True,
    ):
        self.config = AudioInputConfig(
            input_device_index=input_device_index,
            debug_mode=debug_mode,
            target_samplerate=target_samplerate,
            chunk_size=chunk_size,
            audio_format=audio_format,
            channels=channels,
            resample_to_target=resample_to_target,
        )
        self.audio_interface = None
        self.stream = None
        self.device_sample_rate = None

    def _ensure_audio_interface(self):
        if self.audio_interface is None:
            self.audio_interface = pyaudio.PyAudio()

    def get_supported_sample_rates(self, device_index: int) -> List[int]:
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        self._ensure_audio_interface()
        device_info = self.audio_interface.get_device_info_by_index(device_index)
        max_channels = device_info.get("maxInputChannels", 1)
        supported_rates = []
        for rate in standard_rates:
            try:
                if self.audio_interface.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=max_channels,
                    input_format=self.config.audio_format,
                ):
                    supported_rates.append(rate)
            except Exception:
                continue
        return supported_rates

    def _get_best_sample_rate(self, device_index: int, desired_rate: int) -> int:
        try:
            supported = self.get_supported_sample_rates(device_index)
            if desired_rate in supported:
                return desired_rate
            if supported:
                return max(supported)
            device_info = self.audio_interface.get_device_info_by_index(device_index)
            return int(device_info.get("defaultSampleRate", 44100))
        except Exception as exc:
            logging.warning("Error determining sample rate: %s", exc)
            return 44100

    def list_devices(self):
        try:
            init()
            self._ensure_audio_interface()
            device_count = self.audio_interface.get_device_count()
            print("Available audio input devices:")
            for idx in range(device_count):
                device_info = self.audio_interface.get_device_info_by_index(idx)
                max_input_channels = device_info.get("maxInputChannels", 0)
                if max_input_channels <= 0:
                    continue
                name = device_info.get("name", f"Device {idx}")
                rates = self.get_supported_sample_rates(idx)
                print(f"{Fore.LIGHTGREEN_EX}Device {Style.RESET_ALL}{idx}{Fore.LIGHTGREEN_EX}: {name}{Style.RESET_ALL}")
                if rates:
                    formatted = ", ".join(f"{Fore.CYAN}{rate}{Style.RESET_ALL}" for rate in rates)
                    print(f"  {Fore.YELLOW}Supported sample rates: {formatted}{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.YELLOW}Supported sample rates: None{Style.RESET_ALL}")
        finally:
            self.cleanup()

    def setup(self) -> bool:
        try:
            self._ensure_audio_interface()
            cfg = self.config
            if cfg.input_device_index is None:
                cfg.input_device_index = self.audio_interface.get_default_input_device_info()["index"]

            self.device_sample_rate = self._get_best_sample_rate(cfg.input_device_index, cfg.target_samplerate)
            self.stream = self.audio_interface.open(
                format=cfg.audio_format,
                channels=cfg.channels,
                rate=self.device_sample_rate,
                input=True,
                frames_per_buffer=cfg.chunk_size,
                input_device_index=cfg.input_device_index,
            )
            return True
        except Exception as exc:
            logging.error("Error initializing audio recording: %s", exc)
            self.cleanup()
            return False

    @staticmethod
    def lowpass_filter(signal, cutoff_freq, sample_rate):
        nyquist = sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(5, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, signal)

    def resample_audio(self, pcm_data, target_sample_rate, original_sample_rate):
        if target_sample_rate < original_sample_rate:
            pcm_data = self.lowpass_filter(pcm_data, target_sample_rate / 2, original_sample_rate)
        return resample_poly(pcm_data, target_sample_rate, original_sample_rate)

    def read_chunk(self):
        return self.stream.read(self.config.chunk_size, exception_on_overflow=False)

    def cleanup(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None
        except Exception as exc:
            logging.warning("Error cleaning up audio resources: %s", exc)
