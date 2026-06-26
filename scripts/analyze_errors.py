"""CLI: run qualitative error analysis over saved predictions.

Thin wrapper around ``fleurs_asr.error_analysis``.
Intended usage: ``python scripts/analyze_errors.py --language yo_ng``
"""

import argparse
from fleurs_asr.error_analysis import load_predictions, summary_stats, worst_by

def main():
    parser = argparse.ArgumentParser(description="Error analysis over a predictions JSONL.")
    parser.add_argument("--predictions", required=True, help="path to a predictions .jsonl")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--metric", default="wer")
    args = parser.parse_args()

    records = load_predictions(args.predictions)
    stats = summary_stats(records)
    print("=== summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\n=== {args.num} worst by {args.metric} ===")
    for r in worst_by(records, args.metric, args.num):
        print(f"wer={r['wer']:.2f} cer={r['cer']:.2f}")
        print(f"  REF: {r['reference_normalized']}")
        print(f"  HYP: {r['prediction_normalized']}")

if __name__ == "__main__":
    main()