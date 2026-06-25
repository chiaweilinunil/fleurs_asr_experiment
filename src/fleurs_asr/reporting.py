"""Persisting predictions/metrics and rendering tables & figures.

Responsibilities (to be implemented):
    * write prediction JSONL (one record per utterance, schema in CLAUDE.md)
    * write metrics CSV/JSON with full provenance (model, lang, profile, date)
    * never overwrite old outputs silently; encode model/lang/split/timestamp
      in filenames
    * render Markdown result tables and matplotlib figures for reports/

Keep formatting concerns here so scripts stay thin.
"""

# TODO: implement write_predictions_jsonl / write_metrics / render_table.
