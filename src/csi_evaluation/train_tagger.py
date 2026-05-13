"""
Step 2 of the CSI pipeline — Train the CSI tagger.

Architecture:
  - Encoder: mBART-50 (recommended: PPFT checkpoint from src/text_baseline/finetune.py)
  - Head: linear token-classification layer over 35 BIO labels

Loss: class-weighted CrossEntropyLoss with inverse-frequency weights
      computed at the base-label level (α=0.5, normalized to mean ≈ 1).
      See thesis §2.7.2 Phase 4.

Inputs (under --data-dir):
  - csi_ner_train.jsonl       — produced by prepare_bio_dataset.py
  - csi_ner_dev.jsonl
  - csi_label_mapping.json

Output:
  - HuggingFace-format model directory at --output-dir
    (config, tokenizer, model weights)

Reference: SilkRoadNLP 2026 paper §5.1 Step 4; thesis §2.7.2 Phase 4.

Usage:
    python src/csi_evaluation/train_tagger.py \\
        --data-dir data/ \\
        --ppft-model-dir runs/ppft/checkpoints/best_hf \\
        --output-dir runs/csi_tagger

This script uses HuggingFace Trainer (single-process); no torchrun needed.
A single A40 GPU is sufficient.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    MBart50TokenizerFast,
    MBartConfig,
    MBartModel,
    MBartPreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from transformers.modeling_outputs import TokenClassifierOutput


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def base_from_label(tag: str) -> str:
    """'B-CSI_PERSON' or 'I-CSI_PERSON' → 'CSI_PERSON'; 'O' → 'O'."""
    if tag is None:
        return "O"
    tag = tag.strip()
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CSINERDataset(Dataset):
    """
    Dataset for CSI NER, reading from JSONL with pre-tokenized data.

    Each line:
        {
          "book_number": int,
          "text": "...",
          "tokens": ["fa_IR", "▁...", ..., "</s>"],
          "labels": ["O", "B-CSI_PERSON", "I-CSI_PERSON", ...]
        }

    Tokens are assumed to be exactly the output of
    tokenizer.convert_ids_to_tokens(input_ids) for the same tokenizer used
    here, so we round-trip them back via convert_tokens_to_ids().
    """

    def __init__(self, jsonl_path: Path, tokenizer: MBart50TokenizerFast,
                 label2id: Dict[str, int], max_length: int):
        self.examples = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                tokens = ex["tokens"]
                labels = ex["labels"]

                if len(tokens) != len(labels):
                    raise ValueError(
                        f"[ERROR] Token/label length mismatch in {jsonl_path.name}: "
                        f"{len(tokens)} tokens vs {len(labels)} labels."
                    )
                self.examples.append(ex)

        print(f"[DATA] Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["tokens"]
        labels = ex["labels"]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        label_ids = [self.label2id[lab] for lab in labels]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MBartForTokenClassification(MBartPreTrainedModel):
    """mBART encoder with a token classification head + class-weighted CE loss."""

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

        class_weights = getattr(config, "class_weights", None)
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if hasattr(self, "class_weights") and self.class_weights is not None:
                weights = self.class_weights.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def compute_class_weights(train_jsonl: Path,
                          label2id: Dict[str, int],
                          alpha: float = 0.5) -> torch.Tensor:
    """
    Inverse-frequency class weights at the BASE label level.

      w_base = (1 / count_base) ** alpha
      normalize so mean(w) ≈ 1
      for each BIO label, weight = w[base_of_label]

    alpha=0.5 (default) is more stable than alpha=1.0 (full inverse) for
    rare classes.
    """
    from collections import Counter

    base_counts = Counter()
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for lab in ex["labels"]:
                base_counts[base_from_label(lab)] += 1

    if not base_counts:
        raise RuntimeError("No labels found in training file; cannot compute class weights.")

    w_base = {b: (1.0 / float(c)) ** alpha for b, c in base_counts.items()}
    mean_w = sum(w_base.values()) / len(w_base)
    for b in w_base:
        w_base[b] /= (mean_w + 1e-12)

    num_labels = len(label2id)
    class_weights = [1.0] * num_labels
    for lab, idx in label2id.items():
        class_weights[idx] = float(w_base.get(base_from_label(lab), 1.0))

    print("\n[WEIGHTS] Base-label counts (most common first):")
    for base, c in base_counts.most_common():
        print(f"  {base:30s}: {c:6d}   raw w_base={w_base[base]:.4f}")

    return torch.tensor(class_weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Step 2 — Train the CSI tagger.")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing csi_ner_train.jsonl, csi_ner_dev.jsonl, csi_label_mapping.json.")
    p.add_argument("--ppft-model-dir", default=None,
                   help="Path to PPFT HuggingFace directory; falls back to PPFT_MODEL_DIR env var or facebook/mbart-large-50.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the trained tagger.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--class-weight-alpha", type=float, default=0.5,
                   help="α for inverse-frequency class weighting (0=uniform, 1=full inverse).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = data_dir / "csi_ner_train.jsonl"
    dev_jsonl = data_dir / "csi_ner_dev.jsonl"
    label_mapping_path = data_dir / "csi_label_mapping.json"

    ppft_dir = (
        args.ppft_model_dir
        or os.environ.get("PPFT_MODEL_DIR")
        or "facebook/mbart-large-50"
    )
    print(f"[INFO] Base model: {ppft_dir}")

    # Label mapping
    with label_mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {k: int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    num_labels = len(label2id)

    # Class weights (computed from training data)
    class_weights = compute_class_weights(
        train_jsonl, label2id, alpha=args.class_weight_alpha,
    )

    print(f"[LABELS] num_labels = {num_labels}")

    # Tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(ppft_dir)
    tokenizer.src_lang = "fa_IR"

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length,
    )

    # Datasets
    train_dataset = CSINERDataset(train_jsonl, tokenizer, label2id, args.max_length)
    dev_dataset = CSINERDataset(dev_jsonl, tokenizer, label2id, args.max_length)

    # Model config + init
    config = MBartConfig.from_pretrained(ppft_dir)
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id
    config.class_weights = class_weights.tolist()

    model = MBartForTokenClassification.from_pretrained(ppft_dir, config=config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        true_labels, true_preds = [], []
        for p, l in zip(preds, labels):
            for pi, li in zip(p, l):
                if li == -100:
                    continue
                true_labels.append(li)
                true_preds.append(pi)
        return {"accuracy": accuracy_score(true_labels, true_preds) if true_labels else 0.0}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[TRAIN] Starting CSI tagger training...")
    trainer.train()

    print("[TRAIN] Saving best model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nCSI tagger trained and saved to: {output_dir}")


if __name__ == "__main__":
    main()
