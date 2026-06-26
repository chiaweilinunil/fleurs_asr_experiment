"""CLI: inspect one FLEURS language (counts, audio hours, sample transcripts).

Thin wrapper around ``fleurs_asr.data``.
Intended usage: ``python scripts/inspect_dataset.py --language yo_ng``
"""

import argparse
from fleurs_asr.data import load_fleurs_split, dataset_stats


def main():
    parser = argparse.ArgumentParser(description="Inspect one FLEURS language split.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    dataset = load_fleurs_split(args.language, args.split)
    stats = dataset_stats(dataset)
    
    print(f"{args.language} / {args.split}: {stats['num_utterances']} utterances, {stats['total_audio_hours']:.2f} h")
    for text in dataset["raw_transcription"][:args.num_samples]:
        print(text)

if __name__ == "__main__":
    main()
