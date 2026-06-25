"""Dataset loading and inspection utilities for Google FLEURS.

Responsibilities (to be implemented):
    * load a FLEURS language subset/split via Hugging Face ``datasets``
    * expose audio arrays + reference transcripts in a clean form
    * report basic statistics (num utterances, total audio hours)

Prefer Hugging Face caching over manual downloads. Keep functions small and
typed so ``scripts/inspect_dataset.py`` and ``scripts/run_zero_shot.py`` can
reuse them.
"""
import datasets

def load_fleurs_split(language, split, streaming=False, max_samples=None):
    if max_samples:
        dataset = datasets.load(dataset("google/fleurs", language, split=split, streaming=streaming))