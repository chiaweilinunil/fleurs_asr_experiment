"""Qualitative error analysis over saved predictions.

Responsibilities (to be implemented):
    * load prediction JSONL
    * rank worst utterances by WER / CER
    * surface deletion/insertion/substitution-heavy cases
    * character confusions, frequent word substitutions, diacritic errors
    * performance vs. utterance duration

For low-resource work, this analysis matters as much as the headline score.
"""

# TODO: implement loaders + ranking/diff helpers over prediction JSONL.
