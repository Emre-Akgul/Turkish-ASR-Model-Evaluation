from __future__ import annotations

from typing import Type

from turkish_asr_eval.engines.base import ASREngine
from turkish_asr_eval.engines.faster_whisper import FasterWhisperEngine
from turkish_asr_eval.engines.nemo import NemoEngine
from turkish_asr_eval.engines.omnilingual import OmnilingualEngine


ENGINE_REGISTRY: dict[str, Type[ASREngine]] = {
    "faster_whisper": FasterWhisperEngine,
    "omnilingual": OmnilingualEngine,
    "nemo": NemoEngine,
}


class UnknownEngineError(ValueError):
    pass


def available_engines() -> tuple[str, ...]:
    return tuple(sorted(ENGINE_REGISTRY))


def get_engine_class(engine: str) -> Type[ASREngine]:
    try:
        return ENGINE_REGISTRY[engine]
    except KeyError as exc:
        valid = ", ".join(available_engines())
        raise UnknownEngineError(
            f"Unknown engine '{engine}'. Valid engines: {valid}"
        ) from exc


def create_engine(engine: str, model: str, **options: object) -> ASREngine:
    return get_engine_class(engine)(model, **options)
