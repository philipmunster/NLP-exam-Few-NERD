#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Few-NERD splits from Hugging Face and export CoNLL-style files.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["inter", "intra", "supervised"],
        help="Dataset mode to download.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required for Hugging Face fallback. "
            "Install it with: uv pip install datasets"
        ) from exc

    subset = "super" if args.mode == "supervised" else args.mode
    dataset = load_dataset("DFKI-SLT/few-nerd", subset)

    fine_label_feature = dataset["train"].features["fine_ner_tags"].feature
    label_names = fine_label_feature.names

    out_dir = Path(__file__).resolve().parent / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}

    for hf_split, filename in split_map.items():
        split = dataset[hf_split]
        out_path = out_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            for row in split:
                tokens = row["tokens"]
                tags = [label_names[idx] for idx in row["fine_ner_tags"]]
                for token, tag in zip(tokens, tags):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")

    print(f"Saved {args.mode} data to {out_dir}")


if __name__ == "__main__":
    main()
