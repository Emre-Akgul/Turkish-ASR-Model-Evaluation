from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf

from turkish_asr_eval.engines.base import ASREngine

TURKISH_LANGUAGE_CODE = "tr-TR"


class NemoEngine(ASREngine):
    def load(self) -> None:
        model_path = Path(self.model)
        if model_path.exists():
            self._model = nemo_asr.models.ASRModel.restore_from(str(model_path))
        else:
            self._model = nemo_asr.models.ASRModel.from_pretrained(self.model)

        if hasattr(self._model, "set_inference_prompt"):
            self._model.set_inference_prompt(TURKISH_LANGUAGE_CODE)

    def transcribe(self, audio: Any) -> str:
        if isinstance(audio, dict) and "path" in audio:
            audio = audio["path"]
        if isinstance(audio, dict):
            raise RuntimeError(
                "The nemo engine currently expects an audio file path. Use a "
                "dataset audio column that provides a local path."
            )

        audio = self._normalize_audio(audio)
        result = self._model.transcribe([audio], target_lang=TURKISH_LANGUAGE_CODE)
        if isinstance(result, (list, tuple)) and result:
            first = result[0]
            if isinstance(first, str):
                return first.strip()
            if hasattr(first, "text"):
                return str(first.text).strip()
        return str(result).strip()

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
        source_positions = np.linspace(
            0.0,
            duration,
            num=audio.shape[0],
            endpoint=False,
        )
        target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
