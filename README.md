# Low-Resource ASR with FLEURS: Multilingual Baselines and Linguistic Error Analysis

A research-oriented practice project for low-resource, minority, and
typologically interesting languages using Google's **FLEURS** speech dataset.
The goal is a clean, reproducible, linguistically informed ASR project — not a
notebook dump.

## What this project does

An end-to-end ASR experimentation framework for FLEURS that can:

1. Inspect FLEURS language subsets (counts, audio hours, sample transcripts).
2. Run **zero-shot** ASR baselines with pretrained Whisper models.
3. Evaluate with **WER and CER** under explicit, named normalization profiles.
4. Save predictions (JSONL) and metrics (JSON) with full provenance.
5. (Planned) Fine-tune a model and perform linguistic error analysis.

## Why FLEURS?

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a
102-language parallel speech dataset built on the FLoRes machine-translation
benchmark. For low-resource ASR practice it is ideal because it provides
read speech with reference transcripts across many languages — including tonal
Niger-Congo languages, non-Latin scripts, and minority languages — under a single
consistent format. That lets us compare a model's behaviour across very different
linguistic settings without juggling many incompatible datasets.

## Languages studied

Starter set (verify codes with `python scripts/list_fleurs_languages.py`):

| Code    | Language | Family / notes                                   |
|---------|----------|--------------------------------------------------|
| `yo_ng` | Yoruba   | Niger-Congo; lexical tone; Latin script + diacritics |
| `ha_ng` | Hausa    | Afro-Asiatic (Chadic); relatively practical orthography |
| `wo_sn` | Wolof    | Niger-Congo (Atlantic); low-resource             |
| `ps_af` | Pashto   | Indo-Iranian; non-Latin (Perso-Arabic) script    |

The pipeline currently has full results for **Yoruba**; the others are the next
runs.

## Models

| Stage     | Models                                              |
|-----------|-----------------------------------------------------|
| Zero-shot | `openai/whisper-tiny` (→ `base`, `small` as needed) |
| Fine-tune | `openai/whisper-small` (planned, Stage 2)           |

Note: several FLEURS languages here (e.g. Yoruba, Wolof) are **not** among
Whisper's supported languages, so zero-shot results are expected to be weak — that
gap is itself a finding, and the motivation for fine-tuning later.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

**Audio decoding note (important on Windows).** `datasets` 4.x decodes audio with
`torchcodec`, which needs FFmpeg and is awkward to install on Windows. This project
sidesteps that: it loads FLEURS with audio decoding **off** and decodes waveforms
itself with `soundfile` (which bundles `libsndfile`, no system FFmpeg). So a plain
`pip install -e ".[dev]"` is enough on Windows, macOS, and Linux.

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

Metrics are meaningless without stating how text was normalized, so every run
records its profile. Diacritics are **preserved by default** — for tonal/diacritic
languages they carry meaning, so stripping them is offered only as an explicit
ablation.

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

## Linguistic notes

### Yoruba (`yo_ng`)

Yoruba is a Niger-Congo language with **lexical tone**. Its standard orthography
uses tone marks (e.g. acute/grave accents) and sub-dot letters (`ẹ`, `ọ`, `ṣ`),
though digital text varies in how consistently these are written. For ASR this
makes **diacritic handling decisive**: stripping diacritics can collapse distinct
words, so this project preserves them under `conservative`/`wer_standard` and only
removes them under the `aggressive` ablation. Whisper-tiny has no dedicated Yoruba
support, so zero-shot output tends to be a phonetic Latin approximation without
tone marks — useful material for error analysis.

> _More per-language notes (Hausa, Wolof, Pashto) to be added as those runs land._

## Limitations

- Zero-shot only so far; no fine-tuning yet.
- Single model size (`whisper-tiny`) for the first baseline.
- Read speech only (FLEURS), which is cleaner than spontaneous speech.
- Whisper does not officially support several of these languages, so baselines are
  weak by construction.

## Future work

- Fine-tune `whisper-small` on selected languages (train/dev/test discipline).
- Add `whisper-base`/`small` zero-shot comparisons.
- Error analysis: worst utterances, diacritic errors, CER vs WER divergence.
- Plots and a short written report under `reports/`.

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
