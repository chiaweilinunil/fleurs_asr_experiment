# Low-Resource ASR with FLEURS: Multilingual Baselines and Linguistic Error Analysis

> **Status: scaffolding.** Project skeleton is in place; modules are stubs to be
> implemented step by step. This README will grow as the pipeline comes online.

A research-oriented practice project for low-resource and typologically
interesting languages using the Google **FLEURS** speech dataset. The aim is a
clean, reproducible, linguistically informed ASR project — not a notebook dump.

## Planned scope

1. Inspect FLEURS language subsets.
2. Run zero-shot Whisper baselines (tiny → base → small).
3. Fine-tune one model on selected low-resource languages.
4. Evaluate with **WER and CER** (never WER alone) under explicit normalization profiles.
5. Compare languages and perform linguistic error analysis.

Starter languages: Yoruba (`yo_ng`), Hausa (`ha_ng`), Wolof (`wo_sn`), Pashto (`ps_af`).

## Repository layout

```
configs/    YAML experiment + language/model configs (experiments as data)
src/        fleurs_asr — reusable, tested library logic
scripts/    thin command-line wrappers around the library
tests/      unit tests for normalization + metrics (no large downloads)
data/       metadata only — raw audio is never committed
results/    predictions (JSONL) and metrics (CSV/JSON)
reports/    figures and tables for the writeup
notebooks/  exploration + error analysis
```

## Installation & usage

_TODO: filled in once `pyproject.toml` and the pipeline scripts are implemented._

First milestone target command:

```bash
make zero-shot LANGUAGE=yo_ng MODEL=openai/whisper-tiny
```
