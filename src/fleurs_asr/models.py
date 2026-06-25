"""Model loading helpers (Whisper first; XLS-R / MMS later).

Responsibilities (to be implemented):
    * load a pretrained ASR model + processor by name
    * be CPU-friendly (no GPU-only code paths)

Keep this thin; decoding logic lives in ``decoding.py``.
"""
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_asr_model(model_name: str):
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor

