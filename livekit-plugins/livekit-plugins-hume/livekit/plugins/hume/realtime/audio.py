# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import utils


@dataclass
class AudioConfig:
    sample_rate: int
    channels: int


class StreamingWavDecoder:
    """Decode streamed WAV bytes where the header may only be present once.

    Hume EVI streams `AudioOutput` as WAV. In practice, the first chunk carries
    the WAV header and subsequent chunks are raw PCM payload bytes.
    """

    def __init__(self) -> None:
        self._header_buffer = bytearray()
        self._audio_cfg = AudioConfig(sample_rate=48000, channels=1)
        self._pcm_stream = utils.audio.AudioByteStream(
            sample_rate=self._audio_cfg.sample_rate,
            num_channels=self._audio_cfg.channels,
            samples_per_channel=self._audio_cfg.sample_rate // 50,  # 20ms
        )
        self._header_parsed = False

    @property
    def audio_config(self) -> AudioConfig:
        return self._audio_cfg

    def _reset_pcm_stream(self) -> None:
        self._pcm_stream = utils.audio.AudioByteStream(
            sample_rate=self._audio_cfg.sample_rate,
            num_channels=self._audio_cfg.channels,
            samples_per_channel=self._audio_cfg.sample_rate // 50,
        )

    def _try_parse_wav_header(self) -> bytes:
        """Parse WAV header from buffered data and return decoded PCM bytes.

        Returns empty bytes when not enough data is available yet.
        """
        if len(self._header_buffer) < 44:
            return b""

        if not (self._header_buffer[:4] == b"RIFF" and self._header_buffer[8:12] == b"WAVE"):
            # Not a WAV header. Treat currently buffered bytes as raw PCM.
            self._header_parsed = True
            pcm = bytes(self._header_buffer)
            self._header_buffer.clear()
            return pcm

        try:
            with wave.open(io.BytesIO(self._header_buffer), "rb") as wavf:
                sampwidth = wavf.getsampwidth()
                if sampwidth != 2:
                    raise ValueError(f"expected 16-bit PCM WAV, got {sampwidth * 8}-bit")

                self._audio_cfg = AudioConfig(
                    sample_rate=wavf.getframerate(),
                    channels=wavf.getnchannels(),
                )
                self._reset_pcm_stream()
                pcm = wavf.readframes(wavf.getnframes())
        except wave.Error:
            # Header/payload not complete enough yet, wait for more bytes.
            return b""

        self._header_parsed = True
        self._header_buffer.clear()
        return pcm

    def push(self, chunk: bytes) -> list[rtc.AudioFrame]:
        if not chunk:
            return []

        # If a new WAV blob appears after parsing, restart decoder state.
        if self._header_parsed and chunk[:4] == b"RIFF" and chunk[8:12] == b"WAVE":
            self._header_parsed = False
            self._header_buffer.clear()
            self._pcm_stream.clear()

        if not self._header_parsed:
            self._header_buffer.extend(chunk)
            pcm = self._try_parse_wav_header()
            if not self._header_parsed:
                return []
            if not pcm:
                return []
            return self._pcm_stream.push(pcm)

        return self._pcm_stream.push(chunk)

    def flush(self) -> list[rtc.AudioFrame]:
        return self._pcm_stream.flush()
