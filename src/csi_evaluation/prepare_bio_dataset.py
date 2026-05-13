"""
Step 1 of the CSI pipeline — Build the BIO NER dataset from gold annotations.

Reads the human-reviewed gold annotations (span-based) and produces:
  - csi_ner_train.jsonl  (subword-level BIO)
  - csi_ner_dev.jsonl    (subword-level BIO)
  - csi_label_mapping.json  (BIO label↔id mapping)

This script implements three linguistically-motivated rules to prevent CSI
labels from leaking onto Persian grammatical morphology:

  1. Majority vote per subword token: a token gets a non-O label only if
     >50% of its characters are non-O in the character-level label array.

  2. Grammatical-suffix blacklist: Persian plurals (ها, های, ان, ات, گان, ین),
     possessive clitics (م, ت, ش, مان, تان, شان), verb endings (یم, ید, ند),
     comparative/superlative (تر, ترین), aspect markers (می, نمی, همی),
     copula/auxiliary forms (است, اند, ام), and ezafe-like ی/یی — forced to O.

  3. Word-start restriction: CSI labels only apply to tokens that start a
     word (mBART '▁' prefix) or continue an existing CSI span.

Reference: SilkRoadNLP 2026 paper §5.1 Step 3; thesis §2.7.2 Phase 3.

Usage:
    python src/csi_evaluation/prepare_bio_dataset.py \\
        --annotated-jsonl data/annotated_csi_filtered.jsonl \\
        --output-dir data/ \\
        --ppft-model-dir runs/ppft/checkpoints/best_hf \\
        --dev-size 0.2 \\
        --seed 42
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

from sklearn.model_selection import train_test_split
from transformers import MBart50TokenizerFast


# ---------------------------------------------------------------------------
# Grammatical-suffix blacklist — these are forced to O regardless of overlap
# ---------------------------------------------------------------------------

GRAMMATICAL_SUFFIXES = {
    # Plurals / collectives
    "ها", "های", "ان", "ات", "گان", "ین",
    # Possessive clitics
    "م", "ت", "ش", "مان", "تان", "شان",
    # Verb personal endings
    "یم", "ید", "ند",
    # Comparative/superlative
    "تر", "ترین",
    # Aspect markers / negation / auxiliaries
    "می", "نمی", "همی",
    # Common copula/auxiliary forms
    "است", "اند", "ام",
    # Ezafe-like / nominal -ی
    "ی", "یی",
}


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_annotated_jsonl(path: Path) -> List[Dict]:
    """Robust JSONL loader with explicit error reporting."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw_line = line
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"\n[JSON ERROR] in {path} line {lineno}: {e}")
                print("  Snippet:", raw_line[:300].rstrip("\n"))
                raise
            examples.append(ex)
    return examples


def write_jsonl(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Character-level labeling
# ---------------------------------------------------------------------------


def build_char_labels(text: str, annotations: List[Dict[str, str]]) -> List[str]:
    """
    Character-level labels: initially all 'O', then mark each span's
    character range with its label. Overlapping spans → later one wins.
    """
    char_labels = ["O"] * len(text)

    for ann in annotations:
        span = ann.get("span", "")
        label = ann.get("label", "").strip()
        if not span or not label:
            continue

        start_search = 0
        while True:
            idx = text.find(span, start_search)
            if idx == -1:
                break
            end = idx + len(span)
            for i in range(idx, min(end, len(text))):
                char_labels[i] = label
            start_search = end

    return char_labels


# ---------------------------------------------------------------------------
# Character-level → subword-level BIO conversion
# ---------------------------------------------------------------------------


def char_to_bio_token_labels(
    text: str,
    char_labels: List[str],
    tokenizer: MBart50TokenizerFast,
    max_length: int,
) -> Tuple[List[str], List[str]]:
    """
    Convert character-level labels to subword-level BIO using mBART offsets.

    Returns:
        tokens: list of mBART subword tokens
        bio_labels: list of BIO labels, same length
    """
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_length,
        truncation=True,
    )
    offsets = encoded["offset_mapping"]
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    bio_labels: List[str] = []

    for idx, ((start, end), tok) in enumerate(zip(offsets, tokens)):
        # Special tokens (<s>, </s>, lang tag) typically have (0, 0) offsets
        if start == 0 and end == 0:
            bio_labels.append("O")
            continue

        if start >= len(char_labels):
            bio_labels.append("O")
            continue

        end = min(end, len(char_labels))
        span_labels = char_labels[start:end]
        if not span_labels:
            bio_labels.append("O")
            continue

        tok_clean = tok.replace("▁", "")

        # Rule 2 — grammatical-suffix blacklist
        if tok_clean in GRAMMATICAL_SUFFIXES:
            bio_labels.append("O")
            continue

        # Rule 1 — majority vote over span_labels
        non_o_labels = [lab for lab in span_labels if lab != "O"]
        if not non_o_labels:
            entity_label = None
        else:
            label_counts = Counter(non_o_labels)
            candidate_label, count = label_counts.most_common(1)[0]
            csi_ratio = count / len(span_labels)
            entity_label = candidate_label if csi_ratio > 0.5 else None

        # Rule 3 — word-start restriction
        if entity_label is not None:
            is_word_start = tok.startswith("▁")
            prev_is_csi = (len(bio_labels) > 0 and bio_labels[-1] != "O")
            if not is_word_start and not prev_is_csi:
                entity_label = None

        # Emit BIO tag
        if entity_label is None:
            bio_labels.append("O")
        else:
            if not bio_labels or bio_labels[-1] == "O":
                bio_labels.append("B-" + entity_label)
            else:
                prev = bio_labels[-1]
                prev_type = prev.split("-", 1)[1] if "-" in prev else None
                if prev_type == entity_label:
                    bio_labels.append("I-" + entity_label)
                else:
                    bio_labels.append("B-" + entity_label)

    assert len(tokens) == len(bio_labels), "tokens and labels must have same length"
    return tokens, bio_labels


# ---------------------------------------------------------------------------
# Label vocab
# ---------------------------------------------------------------------------


def build_label_vocab(bio_sequences: List[List[str]]):
    """label2id / id2label over BIO tags actually used in the dataset."""
    label_set = set()
    for seq in bio_sequences:
        for lab in seq:
            label_set.add(lab)
    label_list = sorted(label_set)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    return {"label2id": label2id, "id2label": id2label}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Step 1 — BIO conversion for CSI tagger training.")
    p.add_argument("--annotated-jsonl", required=True,
                   help="Path to annotated_csi_filtered.jsonl (gold span-based annotations).")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write csi_ner_train.jsonl, csi_ner_dev.jsonl, csi_label_mapping.json.")
    p.add_argument("--ppft-model-dir", default=None,
                   help="Path to PPFT HuggingFace dir OR set the PPFT_MODEL_DIR env var. "
                        "If neither is set, falls back to public mBART-50.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--dev-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve tokenizer source
    ppft_dir = (
        args.ppft_model_dir
        or os.environ.get("PPFT_MODEL_DIR")
        or "facebook/mbart-large-50"
    )

    annotated_path = Path(args.annotated_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "csi_ner_train.jsonl"
    dev_out = output_dir / "csi_ner_dev.jsonl"
    label_mapping_out = output_dir / "csi_label_mapping.json"

    print(f"Loading annotated examples from {annotated_path} ...")
    raw_examples = load_annotated_jsonl(annotated_path)
    print(f"Loaded {len(raw_examples)} examples.")

    print(f"Loading tokenizer from: {ppft_dir}")
    tokenizer = MBart50TokenizerFast.from_pretrained(ppft_dir)
    tokenizer.src_lang = "fa_IR"

    prepared = []
    all_bio = []

    for ex in raw_examples:
        text = ex["persian_text"]
        annotations = ex.get("annotations", [])

        char_labels = build_char_labels(text, annotations)
        tokens, bio_labels = char_to_bio_token_labels(
            text, char_labels, tokenizer, args.max_length
        )

        prepared.append({
            "book_number": ex.get("book_number", None),
            "text": text,
            "tokens": tokens,
            "labels": bio_labels,
        })
        all_bio.append(bio_labels)

    print("Building label vocabulary...")
    mapping = build_label_vocab(all_bio)
    with label_mapping_out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Saved label mapping ({len(mapping['label2id'])} labels) → {label_mapping_out}")

    train_data, dev_data = train_test_split(
        prepared, test_size=args.dev_size, random_state=args.seed
    )

    write_jsonl(train_out, train_data)
    write_jsonl(dev_out, dev_data)

    print(f"Saved {len(train_data)} train examples → {train_out}")
    print(f"Saved {len(dev_data)} dev examples → {dev_out}")


if __name__ == "__main__":
    main()
