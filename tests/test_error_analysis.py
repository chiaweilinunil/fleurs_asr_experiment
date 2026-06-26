import pytest
from fleurs_asr.error_analysis import load_predictions, worst_by, summary_stats


def test_worst_by_orders_descending():
    records = [
    {"id": "a", "wer": 0.2, "cer": 0.1},
    {"id": "b", "wer": 2.0, "cer": 1.5},   # catastrophic (wer > 1)
    {"id": "c", "wer": 1.0, "cer": 0.5},
    ]
    result = worst_by(records, "wer", 3)
    assert result[0]["id"] == "b"   
    assert result[-1]["id"] == "a"  

def test_worst_by_limits_n():
    records = [
    {"id": "a", "wer": 0.2, "cer": 0.1},
    {"id": "b", "wer": 2.0, "cer": 1.5},   # catastrophic (wer > 1)
    {"id": "c", "wer": 1.0, "cer": 0.5},
    ]
    assert len(worst_by(records, "wer", 2)) == 2

def test_worst_by_metric():
    # data where WER and CER rank differently, so the metric arg must matter
    records = [
        {"id": "x", "wer": 2.0, "cer": 0.1},   # worst WER, best CER
        {"id": "y", "wer": 0.1, "cer": 2.0},   # best WER, worst CER
    ]
    assert worst_by(records, "wer", 1)[0]["id"] == "x"
    assert worst_by(records, "cer", 1)[0]["id"] == "y"

def test_summary_stats():
    records = [
    {"id": "a", "wer": 0.2, "cer": 0.1},
    {"id": "b", "wer": 2.0, "cer": 1.5},   # catastrophic (wer > 1)
    {"id": "c", "wer": 1.0, "cer": 0.5},
    ]
    s = summary_stats(records)
    assert s["n"] == 3
    assert s["mean_wer"] == pytest.approx(3.2 / 3)
    assert s["median_wer"] == 1.0
    assert s["mean_cer"] == pytest.approx(0.7)
    assert s["median_cer"] == 0.5
    assert s["n_catastrophic"] == 1

def test_load_predictions_roundtrip(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text('{"id": "a", "wer": 0.5}\n{"id": "b", "wer": 1.5}\n', encoding="utf-8")
    recs = load_predictions(p)
    assert len(recs) == 2
    assert recs[1]["wer"] == 1.5
