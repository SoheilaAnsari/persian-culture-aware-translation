"""
Step 3 of the CSI pipeline — Apply the trained CSI tagger to the parallel
Masnavi corpus.

This produces CSI-tagged CSVs at the WORD LEVEL (not subword level): the
mBART subword outputs from the tagger are merged back into Persian words,
and BIO labels are re-emitted at the word level for downstream alignment.

Inputs (under --parallel-root):
    train/train.csv, val/val.csv, test/test.csv
    Each CSV with columns: book_number, persian_text, english_translation,
                           audio_filename, language

Outputs (under --output-root):
    train_with_csi.csv, val_with_csi.csv, test_with_csi.csv
    Same columns + two new ones:
        - persian_tokens: space-separated word-level tokens
        - csi_tags: space-separated word-level BIO labels (aligned 1:1)

Reference: SilkRoadNLP 2026 paper §5.1 Step 5.

Usage:
    python src/csi_evaluation/annotate_corpus.py \\
        --parallel-root path/to/parallel/csv/folder \\
        --output-root path/to/output/folder \\
        --csi-model-dir runs/csi_tagger
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch import nn
from transformers import (
    MBart50TokenizerFast,
    MBartConfig,
    MBartModel,
    MBartPreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput


MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_DELIMITER = ","


# ---------------------------------------------------------------------------
# Model (same definition as in train_tagger.py)
# ---------------------------------------------------------------------------


class MBartForTokenClassification(MBartPreTrainedModel):
    config_class = MBartConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MBartModel(config)
        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(config, "dropout", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)


def load_csi_model_and_tokenizer(model_dir: Path):
    config = MBartConfig.from_pretrained(model_dir)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
    tokenizer.src_lang = "fa_IR"

    model = MBartForTokenClassification.from_pretrained(model_dir, config=config)
    model.to(DEVICE)
    model.eval()

    id2label: Dict[int, str] = {int(k): v for k, v in config.id2label.items()}
    return model, tokenizer, id2label


# ---------------------------------------------------------------------------
# Subword → word merging
# ---------------------------------------------------------------------------


def base_from_label(tag: str) -> str:
    if tag is None:
        return "O"
    tag = tag.strip()
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag


def collapse_subword_labels_to_base(labels: List[str]) -> str:
    """For labels of one word's subwords, decide a single base label."""
    non_o_bases = [base_from_label(l) for l in labels if base_from_label(l) != "O"]
    if not non_o_bases:
        return "O"
    return non_o_bases[0]


def subwords_to_word_level(
    tokens: List[str],
    labels: List[str],
    tokenizer: MBart50TokenizerFast,
) -> Tuple[List[str], List[str]]:
    """
    Collapse mBART subword tokens & labels into word-level tokens & BIO labels.

    Strategy:
      1. Skip special tokens (lang token, </s>, <pad>, etc.).
      2. Use '▁' as word-start marker.
      3. For each word:
         - join subwords (strip leading '▁') → word token in Persian
         - collapse subword labels → base label (CSI_PERSON, etc., or 'O')
      4. Convert base labels to word-level BIO:
         - Within a contiguous run of same base label:
             first word → 'B-<base>', subsequent → 'I-<base>'
         - 'O' stays 'O'
    """
    assert len(tokens) == len(labels), "tokens and labels must have same length"
    special_tokens = set(tokenizer.all_special_tokens)

    word_tokens: List[str] = []
    word_base_labels: List[str] = []
    current_subwords: List[str] = []
    current_sub_labels: List[str] = []

    def flush_current():
        nonlocal current_subwords, current_sub_labels, word_tokens, word_base_labels
        if not current_subwords:
            return
        pieces = [sw.lstrip("▁") for sw in current_subwords]
        word = "".join(pieces).strip()
        if not word:
            word = "".join(current_subwords)
        base_label = collapse_subword_labels_to_base(current_sub_labels)
        word_tokens.append(word)
        word_base_labels.append(base_label)
        current_subwords = []
        current_sub_labels = []

    for tok, lab in zip(tokens, labels):
        if tok in special_tokens:
            continue
        if tok.startswith("▁"):
            flush_current()
            current_subwords = [tok]
            current_sub_labels = [lab]
        else:
            current_subwords.append(tok)
            current_sub_labels.append(lab)

    flush_current()

    word_labels: List[str] = []
    prev_base = "O"
    for base in word_base_labels:
        if base == "O":
            word_labels.append("O")
        else:
            prefix = "I-" if prev_base == base else "B-"
            word_labels.append(f"{prefix}{base}")
        prev_base = base

    return word_tokens, word_labels


@torch.no_grad()
def annotate_text(
    text: str,
    tokenizer: MBart50TokenizerFast,
    model: MBartForTokenClassification,
    id2label: Dict[int, str],
    max_length: int,
) -> Tuple[List[str], List[str]]:
    """Annotate a single verse. Returns word-level tokens and BIO labels."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=False,
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    outputs = model(**encoded)
    preds = outputs.logits.argmax(dim=-1)[0].cpu().tolist()

    input_ids = encoded["input_ids"][0].cpu().tolist()
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    subword_labels = [id2label[p] for p in preds]

    word_tokens, word_labels = subwords_to_word_level(subword_tokens, subword_labels, tokenizer)
    assert len(word_tokens) == len(word_labels)
    return word_tokens, word_labels


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------


def process_csv(input_path: Path, output_path: Path, tokenizer, model, id2label):
    """Add persian_tokens + csi_tags columns to a parallel CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter=CSV_DELIMITER)
        orig_fieldnames = reader.fieldnames
        if orig_fieldnames is None:
            raise ValueError(f"No header found in: {input_path}")

        # Remove any pre-existing csi_tags column
        fieldnames = [f for f in orig_fieldnames if f != "csi_tags"]
        if "persian_tokens" not in fieldnames:
            fieldnames.append("persian_tokens")
        fieldnames.append("csi_tags")

        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=CSV_DELIMITER)
        writer.writeheader()

        for row in reader:
            text = row["persian_text"]
            word_tokens, word_labels = annotate_text(
                text=text,
                tokenizer=tokenizer,
                model=model,
                id2label=id2label,
                max_length=MAX_LENGTH,
            )
            row["persian_tokens"] = " ".join(word_tokens)
            row["csi_tags"] = " ".join(word_labels)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Step 3 — Annotate parallel corpus with CSI tags.")
    p.add_argument("--parallel-root", required=True,
                   help="Path to parallel CSV folder with train/, val/, test/ subdirectories.")
    p.add_argument("--output-root", required=True,
                   help="Where to write {train,val,test}_with_csi.csv.")
    p.add_argument("--csi-model-dir", required=True,
                   help="HuggingFace dir produced by train_tagger.py.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Which splits to process.")
    return p.parse_args()


def main():
    args = parse_args()
    parallel_root = Path(args.parallel_root)
    output_root = Path(args.output_root)
    csi_model_dir = Path(args.csi_model_dir)

    input_files = {
        split: parallel_root / split / f"{split}.csv"
        for split in args.splits
    }
    output_files = {
        split: output_root / f"{split}_with_csi.csv"
        for split in args.splits
    }

    print(f"[INFO] Loading CSI tagger from: {csi_model_dir}")
    model, tokenizer, id2label = load_csi_model_and_tokenizer(csi_model_dir)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] CSI tagger loaded with {len(id2label)} labels.")

    for split in args.splits:
        in_path = input_files[split]
        out_path = output_files[split]
        print(f"\n[INFO] Annotating {split}: {in_path} -> {out_path}")
        process_csv(in_path, out_path, tokenizer, model, id2label)

    print(f"\n[DONE] Files written under: {output_root}")


if __name__ == "__main__":
    main()
