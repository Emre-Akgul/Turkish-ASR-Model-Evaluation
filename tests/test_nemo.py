from turkish_asr_eval.engines import nemo
from turkish_asr_eval.engines.nemo import NemoEngine


class FakeNemoModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_files, **kwargs):
        self.calls.append((audio_files, kwargs))
        return ["merhaba"]


def test_nemo_passes_audio_array_and_turkish_target_lang():
    engine = NemoEngine("model")
    engine._model = FakeNemoModel()

    prediction = engine.transcribe([0.0, 0.1])

    assert prediction == "merhaba"
    audio_files, kwargs = engine._model.calls[0]
    assert audio_files == [[0.0, 0.1]]
    assert kwargs == {"target_lang": "tr-TR"}


def test_nemo_load_sets_inference_prompt(monkeypatch):
    prompts = []

    class FakeASRModel:
        @staticmethod
        def from_pretrained(model):
            return FakePromptModel()

    class FakePromptModel:
        def set_inference_prompt(self, target_lang):
            prompts.append(target_lang)

    class FakeNemoAsr:
        class models:
            ASRModel = FakeASRModel

    engine = NemoEngine("remote-model")
    monkeypatch.setattr(nemo, "nemo_asr", FakeNemoAsr())
    engine.load()

    assert prompts == ["tr-TR"]
