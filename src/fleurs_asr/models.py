"""Model loading helpers (Whisper first; XLS-R / MMS later).

Responsibilities (to be implemented):
    * load a pretrained ASR model + processor by name
    * be CPU-friendly (no GPU-only code paths)

Keep this thin; decoding logic lives in ``decoding.py``.
"""

# TODO: implement load_asr_model(name) returning (model, processor).
