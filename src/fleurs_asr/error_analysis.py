"""Qualitative error analysis over saved predictions.

Responsibilities (to be implemented):
    * load prediction JSONL
    * rank worst utterances by WER / CER
    * surface deletion/insertion/substitution-heavy cases
    * character confusions, frequent word substitutions, diacritic errors
    * performance vs. utterance duration

For low-resource work, this analysis matters as much as the headline score.
"""
import json
from pathlib import Path
import statistics

def load_predictions(path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]

def worst_by(records, metric="wer", n=10) -> list[dict]:
    return sorted(records, key=lambda r: r[metric], reverse=True)[:n]

def summary_stats(records) -> dict:
    wers = [record["wer"] for record in records]
    cers = [record["cer"] for record in records]
    
    mean_wer = statistics.mean(wers)
    mean_cer = statistics.mean(cers)
    median_wer = statistics.median(wers)
    median_cer = statistics.median(cers)
    
    n_catastrophic = len([wer for wer in wers if wer > 1.0]) # how many with wer > 1.0
    
    return {
        "n": len(records),
        "mean_wer":  mean_wer,
        "median_wer": median_wer,
        "mean_cer": mean_cer,
        "median_cer": median_cer,
        "n_catastrophic": n_catastrophic,   
    }