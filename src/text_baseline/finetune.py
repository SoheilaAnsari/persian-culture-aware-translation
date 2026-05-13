"""
Stage 2 of PPFT — Fine-tuning the domain-adapted mBART-50 on the parallel
Persian-English Masnavi corpus.

This script loads the checkpoint produced by `domain_pretrain.py` and
fine-tunes it on the Persian↔English parallel data using supervised seq2seq
training.

Reference: SilkRoadNLP 2026 paper §5.2; thesis §2.4.2.

Implementation notes (matching the paper):
  - Initialization: from Stage 1 (DAP) checkpoint
  - Optimizer: AdamW (lr=3e-5)
  - Schedule: cosine with 3% warmup
  - 5 epochs, batch size 16 per GPU, gradient accumulation 4×, AMP
  - Target-Language Anchoring: forced_bos_token_id = en_XX (thesis §2.4.2 Challenge 2)
  - Early stopping: patience=2 on validation loss

Usage:
    torchrun --nproc_per_node=4 src/text_baseline/finetune.py \\
        --train-csv path/to/train.csv \\
        --val-csv path/to/val.csv \\
        --test-csv path/to/test.csv \\
        --dap-checkpoint runs/dap/checkpoints/best_model.pt \\
        --output-dir runs/ppft

Hardware used in the paper: 4× NVIDIA A40 (48 GB each).
"""

import argparse
import os
import re
import csv
import math
import datetime as dt
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    AdamW,
    get_scheduler,
)
from transformers.utils import logging as hf_logging
from tqdm import tqdm
import pandas as pd


warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
hf_logging.set_verbosity_error()
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Persian normalization (inlined here so the script is self-contained)
# ---------------------------------------------------------------------------

_ARABIC_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_fa(s: str) -> str:
    """Light Persian normalization. See `src/utils/normalizers.py` for the canonical version."""
    if not isinstance(s, str):
        return s
    s = s.replace("\u064A", "\u06CC")   # Arabic Yeh → Persian Yeh
    s = s.replace("\u0643", "\u06A9")   # Arabic Kaf → Persian Kaf
    s = _ARABIC_DIAC.sub("", s).replace("\u0640", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ParallelDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def preprocess_parallel(
    tokenizer,
    csv_path: str,
    max_length: int = 512,
    local_rank: int = 0,
):
    df = pd.read_csv(csv_path)
    src = [normalize_fa(x) for x in df["persian_text"].astype(str).tolist()]
    tgt = df["english_translation"].astype(str).tolist()
    if local_rank == 0:
        print(f"[DATA] {csv_path}: {len(src)} pairs")
    enc = tokenizer(
        src,
        text_target=tgt,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )
    return ParallelDataset(enc)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=dt.timedelta(seconds=3600),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


@torch.no_grad()
def evaluate_ddp(model, data_loader, device):
    """All ranks compute; return the global mean loss (same value on every rank)."""
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss_sum += out.loss.detach()
        count += 1.0

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    return (loss_sum / torch.clamp(count, min=1.0)).item()


@torch.no_grad()
def generate_batch_texts(model, tokenizer, texts, device, max_length: int = 200, num_beams: int = 4):
    model.eval()
    outs = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(device)
        gen = model.generate(
            **enc,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    model.train()
    return outs


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="PPFT Stage 2 — Fine-tuning")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--test-csv", required=False, default=None,
                   help="Optional: also report final test loss after training.")
    p.add_argument("--dap-checkpoint", required=True,
                   help="Path to the Stage 1 (DAP) checkpoint .pt file.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=20)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--warmup-pct", type=float, default=0.03)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fine_tune(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    samples_dir = os.path.join(args.output_dir, "samples")
    if local_rank == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        print(f"[DDP] Fine-tuning on GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        log_csv = os.path.join(args.output_dir, "training_log.csv")

    # Tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    tokenizer.src_lang = "fa_IR"
    tokenizer.tgt_lang = "en_XX"
    if local_rank == 0:
        print("[TOK IDS]", {k: tokenizer.lang_code_to_id[k] for k in ["fa_IR", "en_XX"]})

    # Data
    train_dataset = preprocess_parallel(tokenizer, args.train_csv, local_rank=local_rank)
    dev_dataset = preprocess_parallel(tokenizer, args.val_csv, local_rank=local_rank)
    test_dataset = (
        preprocess_parallel(tokenizer, args.test_csv, local_rank=local_rank)
        if args.test_csv else None
    )

    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset, shuffle=False)

    # Model
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

    # Load Stage 1 DAP weights (strip any DDP 'module.' prefix)
    ckpt = torch.load(args.dap_checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if local_rank == 0:
        print(f"[DAP LOAD] from: {args.dap_checkpoint}")
        print(f"[DAP LOAD] loaded={len(state)} | missing={len(missing)} | unexpected={len(unexpected)}")
        if len(missing) > 20:
            print("[WARN] Many missing keys — double-check that this is a domain_pretrain.py checkpoint.")

    # Lock English generation (thesis §2.4.2 Challenge 2 — Target-Language Anchoring)
    en_id = tokenizer.lang_code_to_id["en_XX"]
    model.config.forced_bos_token_id = en_id
    model.config.decoder_start_token_id = en_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if local_rank == 0:
        print(f"[CFG] forced_bos={model.config.forced_bos_token_id} "
              f"decoder_start={model.config.decoder_start_token_id}")

    model.to(device)

    # Collator & loaders
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", pad_to_multiple_of=8)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=args.prefetch_factor, drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, sampler=dev_sampler,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler,
            collate_fn=collator, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )

    # DDP wrap
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer & schedule
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(100, int(args.warmup_pct * total_steps))
    scheduler = get_scheduler("cosine", optimizer,
                              num_warmup_steps=warmup_steps,
                              num_training_steps=total_steps)
    scaler = GradScaler()

    if local_rank == 0:
        print(f"[TRAIN] Total steps: {total_steps} | Warmup: {warmup_steps} | LR: {args.learning_rate}")

    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        if no_improve_epochs >= args.patience:
            if local_rank == 0:
                print("[EARLY STOP] Patience reached")
            break

        model.train()
        train_loader.sampler.set_epoch(epoch)

        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, disable=(local_rank != 0),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                out = model(**batch)
                loss = out.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if local_rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{running / (step + 1):.4f}",
                        "lr": f"{current_lr:.2e}",
                    })

        # Validation (all ranks compute, then reduce)
        val_loss = evaluate_ddp(model, dev_loader, device)
        if local_rank == 0:
            print(f"\n[VAL] Epoch {epoch + 1}: loss={val_loss:.4f}")

        # Rank-0 only: quick samples + saving
        if local_rank == 0:
            # Save 5 sample translations on the validation set
            K = 5
            df_dev = pd.read_csv(args.val_csv).head(K)
            dev_src = df_dev["persian_text"].astype(str).tolist()
            dev_ref = df_dev["english_translation"].astype(str).tolist()
            dev_hyp = generate_batch_texts(model.module, tokenizer, dev_src, device, num_beams=4)

            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_path = os.path.join(
                samples_dir, f"epoch_{epoch+1:02d}_loss_{val_loss:.4f}_{ts}.txt"
            )
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(f"# EPOCH {epoch+1} | val_loss={val_loss:.4f}\n")
                for i, (s, h, r) in enumerate(zip(dev_src, dev_hyp, dev_ref), 1):
                    f.write(f"\n[{i:02d}] SRC: {s}\n[{i:02d}] HYP: {h}\n[{i:02d}] REF: {r}\n")
            print(f"[SAMPLES] saved {K} lines -> {sample_path}")

            # CSV log
            row = {
                "epoch": epoch + 1,
                "val_loss": f"{val_loss:.6f}",
                "samples_file": sample_path,
            }
            write_header = (not os.path.exists(log_csv))
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    w.writeheader()
                w.writerow(row)

            # Save best
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve_epochs = 0

                ck = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                }
                best_pt = os.path.join(checkpoints_dir, "best_finetuned.pt")
                torch.save(ck, best_pt)
                print(f"[SAVE] PyTorch checkpoint: {best_pt}")

                hf_dir = os.path.join(checkpoints_dir, "best_hf")
                model.module.save_pretrained(hf_dir)
                tokenizer.save_pretrained(hf_dir)
                print(f"[SAVE] HF directory: {hf_dir}")

                # Smoke test on a famous Masnavi verse
                model.eval()
                probe = "گفت پیغمبر که چون کوبی دری عاقبت زان در برون آید سری"
                with torch.no_grad():
                    enc = tokenizer(probe, return_tensors="pt").to(device)
                    out = model.module.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                        num_beams=4,
                        max_new_tokens=40,
                        early_stopping=True,
                    )
                smoke = tokenizer.decode(out[0], skip_special_tokens=True)
                print(f"[SMOKE] Persian: {probe}")
                print(f"[SMOKE] English: {smoke}")
                model.train()
            else:
                no_improve_epochs += 1
                print(f"[INFO] No improvement for {no_improve_epochs} epoch(s)")

        # Barrier AFTER rank-0's short work
        dist.barrier()

    # Final test loss (optional)
    if test_loader is not None:
        test_loss = evaluate_ddp(model, test_loader, device)
        if local_rank == 0:
            print("\n" + "=" * 80)
            print(f"[EVAL] Final Test Loss: {test_loss:.4f}")
            print("=" * 80 + "\n")

    dist.barrier()
    cleanup_ddp()

    if local_rank == 0:
        print("Training completed.")
        print(f"Best checkpoint dir: {checkpoints_dir}")


if __name__ == "__main__":
    args = parse_args()
    fine_tune(args)
