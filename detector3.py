#!/usr/bin/env python3
# WORKERS OF THE WORLD UNITE ✊
from collections import deque
from scipy.io.wavfile import write
from typing import Callable, Deque, List
import asyncio
import numpy as np
import os
import pdb
import pvporcupine
import resampy
import sounddevice as sd

use_defaults = input("use defaults? (y/n): ")
if use_defaults.lower() == "y":
    DEFAULT_DEVICE = 4
    DEFAULT_SAMPLE_RATE = 48000
    DEFAULT_CHANNELS = 1
    DEFAULT_KEYWORDS = ["computer"]
    DEFAULT_SENSITIVITIES = [0.9]
    DEFAULT_RECORDING_DURATION = 5
    DEFAULT_OUTPUT_FILENAME = "recorded_audio.wav"
else:
    DEFAULT_DEVICE = int(input("Default device: "))
    DEFAULT_SAMPLE_RATE = int(input("Default sample rate: "))
    DEFAULT_CHANNELS = int(input("number of channels: "))
    DEFAULT_KEYWORDS = []
    while True:
        print("Enter keywords separated by a comma, period to end the list")
        default_keywords_input = input("Keywords: ")
        if "." in default_keywords_input:
            break
        DEFAULT_KEYWORDS.extend(default_keywords_input.split(","))
    DEFAULT_SENSITIVITIES = [0.9]
    DEFAULT_RECORDING_DURATION = int(input("Recording duration: "))
    DEFAULT_OUTPUT_FILENAME = input("Recorded file name: ")


class AudioConfiguration:
    def __init__(
        self,
        device: int = DEFAULT_DEVICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels


class WakeWordDetector:
    def __init__(
        self, access_key: str, keywords: List[str], sensitivities: List[float]
    ):
        self.porcupine = pvporcupine.create(
            access_key=access_key, keywords=keywords, sensitivities=sensitivities
        )

    def process(self, audio_buffer) -> int:
        return self.porcupine.process(audio_buffer)

    def delete(self) -> None:
        if self.porcupine is not None:
            self.porcupine.delete()


class WakeWordAudioProcessor:
    def __init__(
        self, config: AudioConfiguration, porcupine_wrapper: WakeWordDetector
    ) -> None:
        self.config = config
        self.porcupine_wrapper = porcupine_wrapper
        self.ratio = (
            self.config.sample_rate / self.porcupine_wrapper.porcupine.sample_rate
        )
        self.min_samples_required = int(
            np.ceil(self.ratio * self.porcupine_wrapper.porcupine.frame_length)
        )
        self.buffer: Deque = deque(maxlen=self.porcupine_wrapper.porcupine.frame_length)

    def detect_wake_word(
        self, audio_buffer: np.ndarray, wake_word_detected: bool
    ) -> bool:
        if audio_buffer.size == 0:
            return False

        pdb.set_trace()
        audio_buffer_resampled = resampy.resample(
            audio_buffer[:, 0],
            self.config.sample_rate,
            self.porcupine_wrapper.porcupine.sample_rate,
        )
        self.buffer.extend(audio_buffer_resampled.astype(np.int16))

        # Add a breakpoint here
        pdb.set_trace()
        while len(self.buffer) >= self.porcupine_wrapper.porcupine.frame_length:
            pcm = np.array(self.buffer)[: self.porcupine_wrapper.porcupine.frame_length]
            keyword_index = self.porcupine_wrapper.process(pcm)
            pdb.set_trace()
            if keyword_index >= 0 and not wake_word_detected:
                print("Wake word detected!")
                wake_word_detected = True
                self.buffer.clear()
            else:
                wake_word_detected = False
                for _ in range(self.porcupine_wrapper.porcupine.frame_length // 2):
                    self.buffer.popleft()
        return wake_word_detected


class PostWakeWordAudioRecorder:
    def __init__(
        self,
        config: AudioConfiguration,
        audio_callback: Callable[[np.ndarray, int, float, sd.CallbackFlags], None],
        audio_buffer: np.ndarray,
        porcupine_wrapper: WakeWordDetector,
    ) -> None:
        self.config = config
        self.audio_callback = audio_callback
        self.audio_buffer = audio_buffer
        self.porcupine_wrapper = porcupine_wrapper

    async def record_post_wake_word_audio(self) -> np.ndarray:
        print("Recording audio...")
        frame_length = self.porcupine_wrapper.porcupine.frame_length
        recorded_data: List[np.ndarray] = []
        stop_recording = False
        loop = asyncio.get_event_loop()
        recording_start_time = loop.time()
        with sd.InputStream(
            callback=self.audio_callback, blocksize=frame_length, dtype="int16"
        ):
            while not stop_recording:
                if len(self.audio_buffer) >= frame_length:
                    recorded_data.append(self.audio_buffer[:frame_length])
                    self.audio_buffer = self.audio_buffer[frame_length:]

                await asyncio.sleep(0.02)

                if loop.time() - recording_start_time >= 5:
                    stop_recording = True

        print("stopped recording")
        recorded_audio = np.concatenate(recorded_data, axis=0)
        return recorded_audio


class WakeWordListener:
    def __init__(
        self, config: AudioConfiguration, porcupine_wrapper: WakeWordDetector
    ) -> None:
        self.config = config
        self.porcupine_wrapper = porcupine_wrapper
        self.audio_processor = WakeWordAudioProcessor(config, porcupine_wrapper)
        self.audio_buffer = np.zeros((0, 1), dtype="int16")

    def audio_callback(
        self, indata: np.ndarray, frames: int, time: float, status: sd.CallbackFlags
    ) -> None:
        indata_mono = indata[:, 0:1].copy()
        self.audio_buffer = np.concatenate((self.audio_buffer, indata_mono), axis=0)

    async def listen_and_detect_wake_word(self) -> np.ndarray:
        print("Listening for wake word...")
        audio_buffer = np.zeros((0, 1), dtype="int16")
        stop_listening = False
        recording_duration = 0
        wake_word_detected = False

        with sd.InputStream(
            callback=self.audio_callback,
            blocksize=self.porcupine_wrapper.porcupine.frame_length,
            dtype="int16",
        ):
            pdb.set_trace()
            while not stop_listening:
                wake_word_detected = self.audio_processor.detect_wake_word(
                    audio_buffer, wake_word_detected
                )
                print(f"wake_word_detected: {wake_word_detected}")

                if wake_word_detected:
                    audio_recorder = PostWakeWordAudioRecorder(
                        self.config,
                        self.audio_callback,
                        audio_buffer,
                        porcupine_wrapper,
                    )
                    recorded_audio = await audio_recorder.record_post_wake_word_audio()
                    return recorded_audio

                await asyncio.sleep(0.02)
                recording_duration += 0.02

                # audio_buffer = np.zeros((0, 1), dtype="int16")

            return np.array([])

    async def start_listening_and_recording_async(self) -> None:
        recorded_audio = await self.listen_and_detect_wake_word()
        if recorded_audio.size > 0:
            print("Saving recorded audio...")
            file_name = "recorded_audio.wav"
            write(file_name, sd.default.samplerate, recorded_audio)
            print(f"recorded audio to {file_name}")
        else:
            print("No audio was recorded.")


if __name__ == "__main__":
    audio_config = AudioConfiguration()
    porcupine_wrapper = WakeWordDetector(
        access_key=os.environ["PICOVOICE_API_KEY"],
        keywords=DEFAULT_KEYWORDS,
        sensitivities=DEFAULT_SENSITIVITIES,
    )
    wake_word_listener = WakeWordListener(audio_config, porcupine_wrapper)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(wake_word_listener.start_listening_and_recording_async())

    porcupine_wrapper.delete()
