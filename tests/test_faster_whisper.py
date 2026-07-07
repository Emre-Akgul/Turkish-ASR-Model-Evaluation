from turkish_asr_eval.engines import faster_whisper
from turkish_asr_eval.engines.faster_whisper import FasterWhisperEngine


def test_faster_whisper_defaults_to_cpu_int8(monkeypatch):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)

    engine = FasterWhisperEngine("small")
    engine.load()

    assert captured == {
        "model": "small",
        "device": "cpu",
        "compute_type": "int8",
    }


def test_faster_whisper_passes_cuda_options(monkeypatch):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)

    engine = FasterWhisperEngine("small", device="cuda", compute_type="float16")
    engine.load()

    assert captured == {
        "model": "small",
        "device": "cuda",
        "compute_type": "float16",
    }
