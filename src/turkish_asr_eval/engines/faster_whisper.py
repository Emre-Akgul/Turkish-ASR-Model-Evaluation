from __future__ import annotations

from io import BytesIO
from typing import Any

from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

from turkish_asr_eval.engines.base import ASREngine


class FasterWhisperEngine(ASREngine):
    def load(self) -> None:
        device = str(self.options.get("device") or "cpu")
        compute_type = str(self.options.get("compute_type") or "int8")

        self._model = WhisperModel(
            self.model,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: Any) -> str:
        audio = self._normalize_audio(audio)
        segments, _info = self._model.transcribe(audio)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _normalize_audio(self, audio: Any) -> Any:
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return self._prepare_array(audio["array"], int(audio["sampling_rate"]))
        if isinstance(audio, bytes):
            array, sampling_rate = self._read_audio(BytesIO(audio))
            return self._prepare_array(array, sampling_rate)
        if isinstance(audio, str):
            array, sampling_rate = self._read_audio(audio)
            return self._prepare_array(array, sampling_rate)
        return audio

    def _read_audio(self, source: Any) -> tuple[Any, int]:
        array, sampling_rate = sf.read(source)
        return array, int(sampling_rate)

    def _prepare_array(self, array: Any, sampling_rate: int) -> np.ndarray:
        audio = np.asarray(array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sampling_rate != 16000:
            audio = self._resample(audio, sampling_rate, 16000)
        return audio

    def _resample(
        self, audio: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        if audio.size == 0:
            return audio
        duration = audio.shape[0] / source_rate
        target_length = max(1, int(round(duration * target_rate)))
        source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
