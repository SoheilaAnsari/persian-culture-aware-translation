"""
Step 4 of the CSI pipeline — Prepare CSI-tagged parallel corpus for word
alignment with awesome-align.

Reads `*_with_csi.csv` files (from annotate_corpus.py) and writes one of:
  - trainval.parallel.txt       (forward: `fa ||| en`)        [--direction forward]
  - trainval.parallel.rev.txt   (reverse: `en ||| fa`)        [--direction reverse]

Plus a row-id index file in either case:
  - trainval.row_ids.tsv  or  trainval.row_ids.rev.tsv

These outputs are then consumed by awesome-align externally (see
docs/reproducibility.md Step 4b for the awesome-align commands).

Reference: SilkRoadNLP 2026 paper §5.1 (lexicon construction phase);
thesis §2.7.3 Phase 1 (corpus preparation for alignment).

Usage:
    # Forward: fa ||| en
    python src/csi_evaluation/prepare_alignment_input.py \\
        --train-csv path/to/train_with_csi.csv \\
        --val-csv path/to/val_with_csi.csv \\
        --output-dir data/aa/ \\
        --direction forward

    # Reverse: en ||| fa
    python src/csi_evaluation/prepare_alignment_input.py \\
        --train-csv path/to/train_with_csi.csv \\
        --val-csv path/to/val_with_csi.csv \\
        --output-dir data/aa/ \\
        --direction reverse
"""

import argparse
import csv
import re
from pathlib import Path


CSV_DELIM = ","


def tokenize_en_for_align(text: str) -> str:
    """
    Lightweight English tokenization for word alignment.

    Separates punctuation from words and normalizes whitespace.
    """
    text = text.strip()
    # Separate common punctuation (ASCII + curly quotes) from words
    text = re.sub(r"([.,!?;:()\"\u201C\u201D\u2018\u2019])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_fa_tokens(tokens_str: str) -> str:
    """
    Normalize Persian tokens column: collapse whitespace.

    The input is expected to be the `persian_tokens` column from
    *_with_csi.csv, which is already space-separated word-level tokens.
    """
    return " ".join(tokens_str.strip().split())


def write_split(
    csv_path: Path,
    split_name: str,
    out_parallel,
    out_ids,
    direction: str,
) -> int:
    """
    Read one CSV and append its rows to the parallel + row-id files.

    direction='forward': writes `fa ||| en`
    direction='reverse': writes `en ||| fa`
    """
    kept = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIM)
        for row_id, row in enumerate(reader):
            fa = normalize_fa_tokens(row.get("persian_tokens", ""))
            en = tokenize_en_for_align(row.get("english_translation", ""))

            if not fa or not en:
                continue

            if direction == "forward":
                out_parallel.write(f"{fa} ||| {en}\n")
            elif direction == "reverse":
                out_parallel.write(f"{en} ||| {fa}\n")
            else:
                raise ValueError(f"Unknown direction: {direction}")

            out_ids.write(f"{split_name}\t{row_id}\n")
            kept += 1
    return kept


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare CSI-tagged parallel corpus for awesome-align."
    )
    p.add_argument("--train-csv", required=True,
                   help="Path to train_with_csi.csv (from annotate_corpus.py).")
    p.add_argument("--val-csv", required=True,
                   help="Path to val_with_csi.csv.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write parallel + row_ids files into.")
    p.add_argument("--direction", choices=["forward", "reverse"], required=True,
                   help="forward = `fa ||| en`; reverse = `en ||| fa`.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.direction == "forward":
        out_parallel_path = output_dir / "trainval.parallel.txt"
        out_ids_path = output_dir / "trainval.row_ids.tsv"
    else:
        out_parallel_path = output_dir / "trainval.parallel.rev.txt"
        out_ids_path = output_dir / "trainval.row_ids.rev.tsv"

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    with out_parallel_path.open("w", encoding="utf-8") as out_par, \
         out_ids_path.open("w", encoding="utf-8") as out_ids:

        kept_train = write_split(train_csv, "train", out_par, out_ids, args.direction)
        kept_val = write_split(val_csv, "val", out_par, out_ids, args.direction)

    total = kept_train + kept_val
    print(f"[OK] direction: {args.direction}")
    print(f"[OK] train kept: {kept_train}")
    print(f"[OK] val   kept: {kept_val}")
    print(f"[OK] total kept: {total}")
    print(f"[OK] parallel file: {out_parallel_path}")
    print(f"[OK] row_ids file:  {out_ids_path}")


if __name__ == "__main__":
    main()
