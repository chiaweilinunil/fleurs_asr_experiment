"""CLI: run a zero-shot Whisper baseline on a FLEURS language.

Thin wrapper around fleurs_asr (data -> models -> decoding -> evaluation ->
reporting). This is the script behind ``make zero-shot``.
Intended usage:
    python scripts/run_zero_shot.py --language yo_ng --model openai/whisper-tiny
"""
from fleurs_asr.data import load_fleurs_split, decode_audio
from fleurs_asr.models import load_asr_model
from fleurs_asr.decoding import transcribe
from fleurs_asr.text_normalization import get_profile, normalize_text
from fleurs_asr.evaluation import compute_wer, compute_cer, compute_corpus_metrics
from fleurs_asr.reporting import write_predictions_jsonl, write_metrics

from datetime import datetime
import argparse

def run_zero_shot(language, model_name, split="test", max_samples=None, profile_name="wer_standard"):
    profile = get_profile(profile_name)
    dataset = load_fleurs_split(language, split, max_samples=max_samples)
    model, processor = load_asr_model(model_name)
    records = []
    refs_norm = []
    hyps_norm = []
    for example in dataset: 
        array, samplerate = decode_audio(example) # audio -> waveform
        prediction_raw = transcribe(model, processor, array, sampling_rate=samplerate) # waveform -> prediction(transcription)
        reference_raw = example["raw_transcription"]
        reference_normalized = normalize_text(reference_raw, profile)
        prediction_normalized = normalize_text(prediction_raw, profile)
        wer = compute_wer(reference_normalized, prediction_normalized)
        cer = compute_cer(reference_normalized, prediction_normalized)
        
        record = {
            "id": str(example["id"]),
            "language": language,
            "model": model_name,
            "split": split,
            "reference_raw": reference_raw,
            "prediction_raw": prediction_raw,
            "reference_normalized": reference_normalized,
            "prediction_normalized": prediction_normalized,
            "wer": float(wer),
            "cer": float(cer),
            "duration_seconds": len(array) / samplerate,
        }
        records.append(record)
        refs_norm.append(reference_normalized)
        hyps_norm.append(prediction_normalized)
        
    corpus_metrics = compute_corpus_metrics(refs_norm, hyps_norm)
    metrics = {
        "experiment_name": f"zero_shot_{language}_{model_name}",
        "language": language,
        "model": model_name,
        "split": split,
        "normalization_profile": profile_name,
        "num_examples": len(records),
        "wer": corpus_metrics["wer"],
        "cer": corpus_metrics["cer"],
        "date": datetime.now().isoformat(),
    }
    pred_path = write_predictions_jsonl(records, "results/predictions", model_name, language, split)
    metrics_path = write_metrics(metrics, "results/metrics", model_name, language, split)

    print(
        f"{language} | {model_name} | split={split} | n={metrics['num_examples']} | "
        f"WER={metrics['wer']:.3f} | CER={metrics['cer']:.3f}"
    )
    print(f"Saved predictions -> {pred_path}")
    print(f"Saved metrics     -> {metrics_path}")

    return records, metrics, pred_path, metrics_path
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--model", default="openai/whisper-tiny")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--normalization", default="wer_standard")
    args = parser.parse_args()
    run_zero_shot(args.language, args.model, args.split, args.max_samples, args.normalization)

if __name__ == "__main__":
    main()