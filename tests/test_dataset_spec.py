import pytest

from turkish_asr_eval.datasets import parse_dataset_spec


def test_parse_dataset_spec_valid():
    spec = parse_dataset_spec("fleurs:validation")

    assert spec.name == "fleurs"
    assert spec.hf_name == "google/fleurs"
    assert spec.config == "tr_tr"
    assert spec.split == "validation"
    assert spec.audio_column == "audio"
    assert spec.reference_column == "transcription"


def test_parse_common_voice_spec():
    spec = parse_dataset_spec("common_voice:test")

    assert spec.name == "common_voice"
    assert spec.hf_name == "fixie-ai/common_voice_17_0"
    assert spec.config == "tr"
    assert spec.split == "test"
    assert spec.audio_column == "audio"
    assert spec.reference_column == "sentence"


@pytest.mark.parametrize(
    "value",
    [
        "dataset",
        "dataset:split:extra",
    ],
)
def test_parse_dataset_spec_rejects_missing_parts(value):
    with pytest.raises(ValueError, match="dataset must be in the form"):
        parse_dataset_spec(value)


@pytest.mark.parametrize(
    "value",
    [
        ":split:audio",
        "dataset::audio",
        ":split",
        "dataset:",
    ],
)
def test_parse_dataset_spec_rejects_empty_parts(value):
    with pytest.raises(ValueError, match="dataset must be in the form"):
        parse_dataset_spec(value)


def test_parse_dataset_spec_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unsupported dataset"):
        parse_dataset_spec("unknown:test")
