"""CLI: list available FLEURS config (language) names.

Thin wrapper. Will call into ``fleurs_asr`` once implemented.
Intended usage: ``python scripts/list_fleurs_languages.py``
"""
import argparse
import datasets

def main():
    parser = argparse.ArgumentParser(description="List FLEURS language config names.")
    parser.add_argument("--filter", default=None, help="only show configs containing this substring")
    args = parser.parse_args()
    configs = datasets.get_dataset_config_names("google/fleurs")
    if args.filter:
        configs = [c for c in configs if args.filter in c]
    print(f"{len(configs)} configs")
    for c in configs:
        print(c)
    
if __name__ == "__main__":
    main()