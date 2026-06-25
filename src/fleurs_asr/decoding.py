"""Inference / decoding for ASR models.

Responsibilities (to be implemented):
    * run a model over audio batches and return raw transcriptions
    * expose decoding settings (num_beams, temperature, language/task)

Decoding settings must be recorded alongside predictions for reproducibility.
"""
import torch

def transcribe(model, processor, audio_array, sampling_rate=16000, language=None, task="transcribe", num_beams=1) -> str:
    with torch.no_grad():
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        token_ids = model.generate(inputs.input_features, num_beams=num_beams)
        text = processor.batch_decode(token_ids, skip_special_tokens=True)[0]
    return text 