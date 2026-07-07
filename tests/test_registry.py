import pytest

from turkish_asr_eval.engines.registry import (
    UnknownEngineError,
    available_engines,
    get_engine_class,
)


def test_all_engine_names_exist():
    assert available_engines() == (
        "faster_whisper",
        "nemo",
        "omnilingual",
    )


def test_unknown_engine_has_helpful_error():
    with pytest.raises(UnknownEngineError, match="Valid engines"):
        get_engine_class("missing")
