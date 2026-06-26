import json
from fleurs_asr.reporting import write_predictions_jsonl, write_metrics

def test_write_predictions_creates_file(tmp_path):
    records = [
    {
        "id": "utt-1",
        "language": "yo_ng",
        "model": "openai/whisper-tiny",
        "split": "test",
        "reference_raw": "àwọn èyàn",
        "prediction_raw": "awon eyan",
        "reference_normalized": "àwọn èyàn",
        "prediction_normalized": "awon eyan",
        "wer": 1.0,
        "cer": 0.5,
        "duration_seconds": 3.2,
    },
    {
        "id": "utt-2",
        "language": "yo_ng",
        "model": "openai/whisper-tiny",
        "split": "test",
        "reference_raw": "ọmọ",
        "prediction_raw": "omo",
        "reference_normalized": "ọmọ",
        "prediction_normalized": "omo",
        "wer": 1.0,
        "cer": 0.33,
        "duration_seconds": 1.1,
    },
    ]
    path = write_predictions_jsonl(records, tmp_path, "openai/whisper-tiny", "yo_ng", "test")
    assert path.exists()
    assert path.suffix == ".jsonl"
    
def test_predictions_roundtrip(tmp_path):
    records = [
    {
        "id": "utt-1",
        "language": "yo_ng",
        "model": "openai/whisper-tiny",
        "split": "test",
        "reference_raw": "àwọn èyàn",
        "prediction_raw": "awon eyan",
        "reference_normalized": "àwọn èyàn",
        "prediction_normalized": "awon eyan",
        "wer": 1.0,
        "cer": 0.5,
        "duration_seconds": 3.2,
    },
    {
        "id": "utt-2",
        "language": "yo_ng",
        "model": "openai/whisper-tiny",
        "split": "test",
        "reference_raw": "ọmọ",
        "prediction_raw": "omo",
        "reference_normalized": "ọmọ",
        "prediction_normalized": "omo",
        "wer": 1.0,
        "cer": 0.33,
        "duration_seconds": 1.1,
    },
    ]
    path = write_predictions_jsonl(records, tmp_path, "openai/whisper-tiny", "yo_ng", "test")
    read = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert read == records
    
def test_predictions_preserve_unicode(tmp_path):
    records = [
    {
        "id": "utt-1",
        "language": "yo_ng",
        "model": "openai/whisper-tiny",
        "split": "test",
        "reference_raw": "àwọn èyàn",
        "prediction_raw": "awon eyan",
        "reference_normalized": "àwọn èyàn",
        "prediction_normalized": "awon eyan",
        "wer": 1.0,
        "cer": 0.5,
        "duration_seconds": 3.2,
    },
    {
        "id": "utt-2",
        "language": "yo_ng",
        "model": "openai/whisper-tiny",
        "split": "test",
        "reference_raw": "ọmọ",
        "prediction_raw": "omo",
        "reference_normalized": "ọmọ",
        "prediction_normalized": "omo",
        "wer": 1.0,
        "cer": 0.33,
        "duration_seconds": 1.1,
    },
    ]
    path = write_predictions_jsonl(records, tmp_path, "openai/whisper-tiny", "yo_ng", "test")
    read_back = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert read_back[0]["reference_raw"] == "àwọn èyàn"
    
def test_write_metrics_roundtrip(tmp_path):
    metrics = {
        "experiment_name": "zero_shot_yo_ng_openai/whisper-tiny",
        "language": "yo_ng",
        "model": "openai/whisper-tiny",
        "split": "test",
        "normalization_profile": "wer_standard",
        "num_examples": 5,
        "wer": 1.64,
        "cer": 1.67,
        "sample": "àwọn",          # include a diacritic to also test encoding
    }
    path = write_metrics(metrics, tmp_path, "openai/whisper-tiny", "yo_ng", "test")
    read_back = json.loads(path.read_text(encoding="utf-8"))
    assert read_back == metrics