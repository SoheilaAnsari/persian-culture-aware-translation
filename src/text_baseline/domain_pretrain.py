"""
Stage 1 of PPFT — Domain-Adaptive Pre-training (DAP) of mBART-50 on Persian poetry.

This script performs mBART-style span-masked denoising on a monolingual Persian
poetry corpus, producing a domain-adapted checkpoint that is then fine-tuned on
the parallel Persian-English Masnavi corpus in Stage 2 (`finetune.py`).

Reference: SilkRoadNLP 2026 paper §5.2; thesis §2.4.1.

Implementation notes (matching the paper):
  - Span masking: geometric span length, mean=3.5; mask density=0.3
  - Dynamic max length: sampled in [64, 256] per example
  - Optimizer: AdamW (lr=1e-5, weight_decay=0.01)
  - Warmup: 500 steps
  - 3 epochs, DDP, mixed precision (AMP), gradient accumulation 4×
  - Custom shift_tokens_right to handle -100 labels and ensure stable
    decoder initialization (see thesis §2.4.1.1)

Usage:
    torchrun --nproc_per_node=4 src/text_baseline/domain_pretrain.py \\
        --poetry-corpus path/to/persian_poetry.txt \\
        --output-dir runs/dap

For single-GPU use: --nproc_per_node=1. CPU-only is not feasible.

Hardware used in the paper: 4× NVIDIA A40 (48 GB each).

Public Persian poetry corpus used in the paper:
    https://github.com/amnghd/Persian_poems_corpus/tree/master
"""

import argparse
import os
import math
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    AdamW,
    get_scheduler,
)
from transformers.models.mbart import modeling_mbart
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Custom shift_tokens_right
#
# Thesis §2.4.1.1 documents that the default HuggingFace shift operation
# was unstable on this low-resource poetic corpus. The replacement below
# handles -100 sentinel values from the denoising collator and explicitly
# aligns decoder_start_token_id with the model's internal language settings.
# ---------------------------------------------------------------------------

GLOBAL_DECODER_START_TOKEN_ID = None


def custom_shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id=None):
    """Replace -100 with pad_token_id, then shift tokens right."""
    if decoder_start_token_id is None:
        if GLOBAL_DECODER_START_TOKEN_ID is None:
            raise ValueError("decoder_start_token_id must be provided")
        decoder_start_token_id = GLOBAL_DECODER_START_TOKEN_ID

    input_ids = torch.where(
        input_ids == -100,
        torch.tensor(pad_token_id, device=input_ids.device),
        input_ids,
    )
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


modeling_mbart.shift_tokens_right = custom_shift_tokens_right


# ---------------------------------------------------------------------------
# Denoising collator and dataset
# ---------------------------------------------------------------------------


class PersianDenoisingCollator(DataCollatorForSeq2Seq):
    """mBART-style contiguous span masking with geometric span sampling."""

    def __init__(
        self,
        tokenizer,
        noise_density: float = 0.3,
        mean_noise_span_length: float = 3.5,
        **kwargs,
    ):
        super().__init__(tokenizer, **kwargs)
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

    def span_mask_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a bool mask (same shape as input_ids) of tokens to mask."""
        seq_len = input_ids.size(0)
        masked = torch.zeros(seq_len, dtype=torch.bool)

        # Reserve the leading language tag if present.
        reserved_indices = set()
        lang_id = self.tokenizer.lang_code_to_id[self.tokenizer.src_lang]
        if input_ids[0].item() == lang_id:
            reserved_indices.add(0)

        target_mask_count = int(seq_len * self.noise_density)
        masked_count = 0

        while masked_count < target_mask_count:
            candidates = [
                i for i in range(seq_len)
                if i not in reserved_indices and not masked[i]
            ]
            if not candidates:
                break
            start = random.choice(candidates)
            # Geometric span length with mean = self.mean_noise_span_length
            p = 1.0 / self.mean_noise_span_length
            span_length = int(math.ceil(math.log(1 - random.random()) / math.log(1 - p)))
            span_length = max(1, span_length)
            end = min(start + span_length, seq_len)
            for i in range(start, end):
                if i in reserved_indices:
                    continue
                if not masked[i]:
                    masked[i] = True
                    masked_count += 1
                    if masked_count >= target_mask_count:
                        break
        return masked

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for i in range(batch["input_ids"].size(0)):
            input_ids_i = batch["input_ids"][i]
            mask = self.span_mask_tokens(input_ids_i)
            input_ids_i[mask] = self.tokenizer.mask_token_id
            labels[i][~mask] = -100
            batch["input_ids"][i] = input_ids_i
        batch["labels"] = labels
        return batch


class DenoisingPoetryDataset(Dataset):
    """Tokenized Persian poetry lines with per-example dynamic max length."""

    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        dynamic_max_length = random.randint(64, 256)
        inputs = self.tokenizer(text, truncation=True, max_length=dynamic_max_length)
        return {"input_ids": inputs["input_ids"]}

    def __len__(self):
        return len(self.texts)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp() -> int:
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def load_and_shuffle_lines(corpus_path: str, seed: int = 42):
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="PPFT Stage 1 — Domain-Adaptive Pre-training")
    p.add_argument("--poetry-corpus", required=True,
                   help="Path to monolingual Persian poetry corpus (one verse per line).")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write checkpoints to.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=20)
    p.add_argument("--noise-density", type=float, default=0.3)
    p.add_argument("--mean-noise-span-length", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def domain_adaptive_pretraining(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if local_rank == 0:
        print(f"Running on GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")

    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50",
        normalize_arabic=True,
        normalize_arabic_numbers=True,
    )
    tokenizer.src_lang = "fa_IR"

    # Denoising collator
    collator = PersianDenoisingCollator(
        tokenizer,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # Data
    lines = load_and_shuffle_lines(args.poetry_corpus, seed=args.seed)
    train_lines, val_lines = train_test_split(lines, test_size=0.01, random_state=args.seed)
    train_dataset = DenoisingPoetryDataset(train_lines, tokenizer)
    val_dataset = DenoisingPoetryDataset(val_lines, tokenizer)

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collator, num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collator,
        num_workers=args.num_workers,
    )

    # Model
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
    model.config.dropout = args.dropout
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[tokenizer.src_lang]
    global GLOBAL_DECODER_START_TOKEN_ID
    GLOBAL_DECODER_START_TOKEN_ID = model.config.decoder_start_token_id
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_training_steps = args.epochs * len(train_dataloader)
    scheduler = get_scheduler(
        "linear", optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )
    scaler = GradScaler()

    if local_rank == 0:
        print(
            f"Denoising Pretraining (mBART-50) | Epochs: {args.epochs} | "
            f"Batch Size: {args.batch_size} | LR: {args.learning_rate}"
        )

    model.train()
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(
            train_dataloader,
            disable=(local_rank != 0),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss.mean() / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        model.train()

        val_loss = total_val_loss / len(val_dataloader)
        perplexity = torch.exp(torch.tensor(val_loss * collator.noise_density))

        if local_rank == 0:
            print(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "loss": val_loss,
                }, checkpoint_path)
                print(f"Best model saved -> {checkpoint_path}")

    cleanup_ddp()
    if local_rank == 0:
        print("Denoising Pretraining complete.")


if __name__ == "__main__":
    args = parse_args()
    domain_adaptive_pretraining(args)
