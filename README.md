# Low-Resource ASR with FLEURS: Multilingual Baselines and Linguistic Error Analysis

A research-oriented practice project for low-resource, minority, and
typologically interesting languages using Google's **FLEURS** speech dataset.

## Project description

An end-to-end ASR experimentation framework for FLEURS that:

1. Inspect FLEURS language subsets (counts, audio hours, sample transcripts).
2. Run **zero-shot** ASR baselines with pretrained Whisper models.
3. Evaluate with **WER and CER** under explicit, named normalization profiles.
4. Save predictions (JSONL) and metrics (JSON) with full provenance.
5. (Planned) Fine-tune a model and perform linguistic error analysis.

## Languages studied

Starter set (verify codes with `python scripts/list_fleurs_languages.py`):

| Code    | Language | Family / notes                                   |
|---------|----------|--------------------------------------------------|
| `yo_ng` | Yoruba   | Atlantic-Congo; |
| `ha_ng` | Hausa    | Afro-Asiatic (Chadic)  |
| `wo_sn` | Wolof    | Atlantic-Congo |
| `ps_af` | Pashto   | Indo-Iranian; non-Latin (Perso-Arabic) script    |

The pipeline currently has full results for **Yoruba**; the others are the next
runs.

## Models

| Stage     | Models                                              |
|-----------|-----------------------------------------------------|
| Zero-shot | `openai/whisper-tiny` |
| Fine-tune | `openai/whisper-small`        |

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## Usage

List available FLEURS languages:

```bash
python scripts/list_fleurs_languages.py --filter _ng
```

Inspect one language (stats + sample transcripts):

```bash
python scripts/inspect_dataset.py --language yo_ng --num-samples 3
```

Run a zero-shot baseline:

```bash
python scripts/run_zero_shot.py --language yo_ng --model openai/whisper-tiny
```

Useful flags: `--split` (default `test`), `--max-samples N` (quick smoke test),
`--normalization {conservative,wer_standard,aggressive}` (default `wer_standard`).

Outputs are written to:

- `results/predictions/` — per-utterance JSONL (raw + normalized text, WER, CER).
- `results/metrics/` — one JSON summary per run (model, language, profile, WER, CER, date).

Run the tests:

```bash
pytest
```

## Normalization profiles

Diacritics are **preserved by default**.

| Profile        | lowercase | remove punct | strip diacritics | Unicode |
|----------------|:---------:|:------------:|:----------------:|:-------:|
| `conservative` |    no     |      no      |        no        |   NFC   |
| `wer_standard` |    yes    |     yes      |        no        |   NFC   |
| `aggressive`   |    yes    |     yes      |       yes        |  NFKD   |

## Results

Zero-shot, `test` split, `wer_standard` normalization. Lower is better.

| Language | Model        | Setup     | N   | WER ↓ | CER ↓ |
|----------|--------------|-----------|----:|------:|------:|
| Yoruba   | Whisper tiny | zero-shot | 831 | 1.757 | 1.403 |
| Hausa    | Whisper tiny | zero-shot |  —  |  TBD  |  TBD  |
| Wolof    | Whisper tiny | zero-shot |  —  |  TBD  |  TBD  |
| Pashto   | Whisper tiny | zero-shot |  —  |  TBD  |  TBD  |

> WER above 1.0 is expected for unsupported languages: with many insertions and
> substitutions, total edits can exceed the reference word count.

## Error analysis (Yoruba zero-shot)

Run with:

```bash
python scripts/analyze_errors.py --predictions results/predictions/<file>.jsonl --num 5
```

Distribution over the 831-utterance Yoruba test set (`wer_standard`):

| Metric | Mean | Median |
|--------|-----:|-------:|
| WER    | 1.77 |  1.00  |
| CER    | 1.42 |  0.66  |

Catastrophic utterances (WER > 1.0): **248 / 831 ≈ 30%**.


## Repository layout

```text
configs/    YAML experiment + language/model configs
src/        fleurs_asr — reusable, tested library logic
scripts/    thin command-line wrappers around the library
tests/      unit tests (normalization, metrics, reporting)
data/       metadata only — raw audio is never committed
results/    predictions (JSONL) and metrics (JSON)
reports/    figures and tables for the writeup
notebooks/  exploration + error analysis
```
