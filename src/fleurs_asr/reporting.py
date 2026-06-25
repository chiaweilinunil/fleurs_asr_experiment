"""Persisting predictions/metrics and rendering tables & figures.

Responsibilities (to be implemented):
    * write prediction JSONL (one record per utterance, schema in CLAUDE.md)
    * write metrics CSV/JSON with full provenance (model, lang, profile, date)
    * never overwrite old outputs silently; encode model/lang/split/timestamp
      in filenames
    * render Markdown result tables and matplotlib figures for reports/

Keep formatting concerns here so scripts stay thin.
"""

import json
from pathlib import Path
from datetime import datetime


def write_predictions_jsonl(records, output_dir, model, language, split) -> Path:
    filepath = _make_output_path(output_dir, model, language, split, "jsonl")
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return filepath


def write_metrics(metrics, output_dir, model, language, split) -> Path:
    filepath = _make_output_path(output_dir, model, language, split, "json")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False))

    return filepath


def _make_output_path(output_dir, model, language, split, extension) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_safe = model.replace("/", "--")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_{split}_{model_safe}_{timestamp}.{extension}"
    filepath = Path(output_dir) / filename

    return filepath
