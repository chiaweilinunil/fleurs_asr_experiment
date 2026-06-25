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
import io 
import numpy as np
import soundfile as sf

def load_fleurs_split(language, split, streaming=False, max_samples=None):
    
    dataset = datasets.load_dataset("google/fleurs", language, split=split, streaming=streaming)
    dataset = dataset.cast_column("audio", datasets.Audio(decode=False)) # For windows that cannot install torchcodec
    if max_samples:
        if streaming:
            dataset = dataset.take(max_samples)
        else: 
            dataset = dataset.select(range(max_samples))
    
    return dataset

def dataset_stats(dataset) -> dict:
    num_utterances = len(dataset)
    total_samples = sum(dataset["num_samples"])
    sampling_rate = 16000
    total_audio_hours = total_samples / sampling_rate / 3600
    stats = {"num_utterances": num_utterances, "total_audio_hours": total_audio_hours} 
    return stats

def decode_audio(example) -> tuple[np.ndarray, int]:
    audio = example["audio"]
    array, samplerate = sf.read(io.BytesIO(audio["bytes"]))
    array = array.astype(np.float32)
    return array, samplerate
