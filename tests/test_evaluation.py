"""Tests for WER/CER computation on small mocked examples.

Will cover: perfect match -> 0.0, known edit distances, CER vs WER behavior.
"""
from fleurs_asr.evaluation import compute_wer, compute_cer, compute_corpus_metrics
import pytest

def test_wer_identical():
    wer = compute_wer("a b c", "a b c")
    assert wer == 0.0

def test_cer_identical():
    cer = compute_wer("a b c", "a b c")
    assert cer == 0.0
    
    
def test_wer_one_deletion():
    wer = compute_wer("hello world", "hello")
    assert wer == 0.5
    
def test_cer_one_substitution():
    cer = compute_cer("cat", "car")
    assert cer == pytest.approx(1/3)
    
def test_corpus_pools_not_averages():
    metrics = compute_corpus_metrics(["hello", "the cat sat on the mat"], ["world", "the cat sat on the mat"])
    assert metrics["wer"] == pytest.approx(1/7)

def test_corpus_empty_raises():
    with pytest.raises(ValueError):
        compute_corpus_metrics([], [])