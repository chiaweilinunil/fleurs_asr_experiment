"""WER and CER metrics.

Inputs are assumed already normalized; normalization is a separate step.
Corpus metrics pool raw edits across utterances, never average per-utterance rates.
"""

import jiwer

def compute_wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis)

def compute_cer(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference, hypothesis)

def compute_corpus_metrics(references: list[str], hypotheses: list[str]) -> dict:
    if not references:
        raise ValueError("references should not be empty")
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    
    metrics = {"wer": wer, "cer": cer, "num_examples": len(references)}
    return metrics