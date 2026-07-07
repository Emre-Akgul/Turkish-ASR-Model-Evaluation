from turkish_asr_eval.datasets import compute_error_rates


def test_compute_error_rates_returns_percentages():
    wer, cer = compute_error_rates("merhaba dunya", "merhaba")

    assert wer == 50.0
    assert cer > 0
