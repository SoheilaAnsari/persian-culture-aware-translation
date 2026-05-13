# train_translation.py - Stage-2 ONLY: Direct W2V→Decoder Translation
# V8 - CORRECTED: Fixed optimizer timing, added gate warm-start, improved diagnostics
# Single Cross-Attention (text←audio), NO prosody

import os

# ---- NCCL / DDP safety ----
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("NCCL_TIMEOUT", "1800")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import math
import csv
import json
import sys
import argparse
from datetime import datetime, timedelta
from typing import List
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.utils import logging as hf_logging
from transformers import get_cosine_schedule_with_warmup
import sacrebleu
from config import get_config
from data_loader import get_dataloader
from model_translation import MultimodalTranslationModel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()

# === Global Configuration ===
ENABLE_ANOMALY_DETECTION = False
ENABLE_TF32 = True
ENABLE_AMP = True
AMP_DTYPE = torch.bfloat16

if ENABLE_ANOMALY_DETECTION:
    torch.autograd.set_detect_anomaly(True)

if ENABLE_TF32:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# === Multimodal-specific hyperparameters ===
ENCODER_LR_MULT = 0.2       # Text encoder learns slower
AUDIO_LR_MULT = 3.0         # Audio projection & cross-attn learn faster

NO_MODDROP_EPOCHS = 4       # Epochs 1–4 with p=0.0, then enable mod-drop
FREEZE_DECODER_UNTIL_EPOCH = 2  # Epochs 1–2 frozen decoder (optional but recommended)


def parse_args():
    """Parse command-line arguments for Stage-2 training."""
    ap = argparse.ArgumentParser(description="Stage-2 Multimodal Translation (W2V Cross-Attention)")

    # Training params
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ap.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    ap.add_argument("--clip", type=float, default=1.0, help="Gradient clipping threshold")

    # Ablations
    ap.add_argument("--ablations_epochs", type=str, default="4,8,10",
                    help="Epochs to run ablations (comma-separated).")
    ap.add_argument("--ablations_max_eval_batches", type=int, default=256,
                    help="Max batches for ablation eval.")

    # Eval control
    ap.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    ap.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval")
    ap.add_argument("--eval_only_ablation", type=str, default=None,
                    choices=["audio_off", "audio_shuffle"])

    # Encoder unfreezing
    ap.add_argument("--encoder_unfreeze_layers", type=int, default=0,
                    help="Unfreeze top-N encoder layers (0 = frozen).")
    ap.add_argument("--encoder_lr", type=float, default=1e-5)

    # Override run directory
    ap.add_argument("--run_dir", type=str, default=None)

    # Required: PPFT text-baseline checkpoint (also overridable by env vars
    # TEXT_BASELINE_PT and TEXT_BASELINE_HF). At least one must be provided.
    ap.add_argument("--text-baseline-pt", "--text_baseline_pt", dest="text_baseline_pt",
                    type=str, default=None,
                    help="Path to PPFT .pt checkpoint (Stage-1+2 from src/text_baseline/). "
                         "Falls back to TEXT_BASELINE_PT env var.")
    ap.add_argument("--text-baseline-hf", "--text_baseline_hf", dest="text_baseline_hf",
                    type=str, default=None,
                    help="Path to PPFT HuggingFace directory. "
                         "Falls back to TEXT_BASELINE_HF env var.")

    return ap.parse_args()


def setup_ddp():
    """Initialize DDP with extended timeout."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=40)
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


def apply_audio_off(batch: dict) -> dict:
    """Zero out audio while preserving exact shapes and masks."""
    a = batch.get("audio", None)
    m = batch.get("audio_mask", None)
    if isinstance(a, torch.Tensor):
        batch["audio"] = torch.zeros_like(a)
    if isinstance(m, torch.Tensor):
        batch["audio_mask"] = torch.zeros_like(m)
    return batch


def apply_audio_shuffle(batch: dict) -> dict:
    """Shuffle audio across the batch and permute mask consistently."""
    a = batch.get("audio", None)
    m = batch.get("audio_mask", None)
    if isinstance(a, torch.Tensor) and a.size(0) > 1:
        idx = torch.randperm(a.size(0))
        batch["audio"] = a.index_select(0, idx)
        if isinstance(m, torch.Tensor) and m.size(0) == a.size(0):
            batch["audio_mask"] = m.index_select(0, idx)
    return batch


def is_dist():
    """Check if distributed training is active."""
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return int(os.environ.get("RANK", "0")) == 0


def barrier():
    """Synchronization barrier for all processes."""
    if is_dist():
        dist.barrier()


def cleanup_ddp():
    """Clean up distributed process group."""
    if is_dist():
        dist.destroy_process_group()


def current_lr(optimizer):
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def set_decoder_frozen(model, frozen: bool):
    """
    Freeze/unfreeze the mBART decoder parameters based on name matching.
    Assumes attributes contain 'mbart' and 'decoder' in parameter names.
    """
    for name, p in model.named_parameters():
        lname = name.lower()
        if "mbart" in lname and "decoder" in lname:
            p.requires_grad = not frozen


def set_text_encoder_frozen(model, frozen: bool):
    """Freeze or unfreeze all text encoder parameters."""
    for n, p in model.named_parameters():
        if n.startswith("mbart.model.encoder"):
            p.requires_grad = (not frozen)


def unfreeze_top_text_layers(model, num_layers: int = 3):
    """Unfreeze only the top K layers of the text encoder."""
    total_layers = getattr(getattr(model, "mbart", None).config, "encoder_layers", 12)
    for n, p in model.named_parameters():
        if n.startswith("mbart.model.encoder.layers."):
            try:
                layer_idx = int(n.split(".")[4])
            except Exception:
                continue
            if layer_idx >= (total_layers - num_layers):
                p.requires_grad = True


def build_optimizer(model, args):
    """
    Build optimizer with separate LR for:
      - text encoder (low LR via ENCODER_LR_MULT)
      - audio-related modules (higher LR via AUDIO_LR_MULT)
      - fusion layers (single cross-attention, gating, etc.)
      - decoder
      - everything else
    """
    base_lr = args.lr

    text_enc_params = []
    audio_params = []
    fusion_params = []
    decoder_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()

        # --- text encoder (mBART encoder) ---
        if "mbart" in lname and "encoder" in lname:
            text_enc_params.append(p)

        # --- decoder (mBART decoder) ---
        elif "mbart" in lname and "decoder" in lname:
            decoder_params.append(p)

        # --- audio branch: wav2vec2 + audio projections + audio cross-attn/resampler ---
        elif ("wav2vec" in lname or "audio_" in lname or "w2v" in lname or "perceiver" in lname) and \
             ("encoder" in lname or "proj" in lname or "projection" in lname or "cross" in lname or "resampler" in lname):
            audio_params.append(p)

        # --- fusion layers (single cross-attention, gating, fusion MLP, etc.) ---
        # NOTE: "dual_xattn" is just a name - your architecture is single cross-attention
        elif "fusion" in lname or "cross_attn" in lname or "crossattention" in lname or "gate" in lname or "dual_xattn" in lname:
            fusion_params.append(p)

        else:
            other_params.append(p)

    # Sanity check for rank 0
    if is_main_process():
        print(f"[OPT] text_enc={len(text_enc_params)}, audio={len(audio_params)}, "
              f"fusion={len(fusion_params)}, decoder={len(decoder_params)}, other={len(other_params)}")

    enc_lr = base_lr * ENCODER_LR_MULT
    audio_lr = base_lr * AUDIO_LR_MULT

    param_groups = []

    if text_enc_params:
        param_groups.append({"params": text_enc_params, "lr": enc_lr, "name": "text_encoder"})

    if audio_params:
        param_groups.append({"params": audio_params, "lr": audio_lr, "name": "audio"})

    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": base_lr, "name": "fusion"})

    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": base_lr, "name": "decoder"})

    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})

    optimizer = AdamW(
        param_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    return optimizer


@torch.no_grad()
def validate(model, loader, cfg, epoch: int = 0, log_file=None, max_eval_batches=None, ablation=None):
    """
    Unified validation function - SINGLE PROCESS ONLY.
    ablation in {None, "audio_off", "audio_shuffle"}.
    """
    with torch.inference_mode():
        model.eval()
        preds: List[str] = []
        refs: List[str] = []
        srcs: List[str] = []

        if max_eval_batches is None:
            max_eval_batches = getattr(cfg, "max_eval_batches", 200)

        empty_count = 0
        printed_debug_once = False
        printed_ablation_checksum = False
        printed_audio_stats_once = False

        for step, batch in enumerate(loader):
            if max_eval_batches is not None and step >= max_eval_batches:
                break

            # --- Apply ablations on CPU tensors
            if ablation == "audio_off":
                batch = apply_audio_off(batch)
            elif ablation == "audio_shuffle":
                batch = apply_audio_shuffle(batch)

            # Always print a single audio checksum
            if (not printed_audio_stats_once) and ("audio" in batch) and (batch["audio"] is not None):
                a = batch["audio"]
                am = batch.get("audio_mask", None)
                cksum = float(a.abs().mean().item()) if isinstance(a, torch.Tensor) else float("nan")
                msg_chk = f"[VAL] audio checksum (mean |abs|): {cksum:.6f}"
                if isinstance(am, torch.Tensor):
                    frac_on = float((am != 0).float().mean().item())
                    msg_chk += f" | mask_on_frac={frac_on:.3f}"
                print(msg_chk)
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(msg_chk + "\n")
                printed_audio_stats_once = True

            # Extra checksum if explicit ablation
            if not printed_ablation_checksum and ablation in {"audio_off", "audio_shuffle"}:
                if "audio" in batch and isinstance(batch["audio"], torch.Tensor):
                    checksum = float(batch["audio"].abs().mean())
                    print(f"[ABLATION:{ablation}] audio checksum: {checksum:.6f}")
                    printed_ablation_checksum = True

            # --- Move to device
            input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(cfg.device, non_blocking=True)
            audio = batch["audio"].to(cfg.device, non_blocking=True) if batch.get("audio", None) is not None else None
            audio_mask = batch["audio_mask"].to(cfg.device, non_blocking=True) if batch.get("audio_mask", None) is not None else None

            # Collect source/refs
            if "src_texts" in batch and isinstance(batch["src_texts"], (list, tuple)):
                srcs.extend(batch["src_texts"])
            else:
                srcs.extend([""] * input_ids.size(0))
            refs.extend(batch["tgt_texts"])

            # Fix shapes: [B,1,T] -> [B,T]
            if isinstance(audio, torch.Tensor) and audio.dim() == 3 and audio.size(1) == 1:
                audio = audio.squeeze(1)
            if isinstance(audio_mask, torch.Tensor) and audio_mask.dtype != torch.bool:
                audio_mask = (audio_mask != 0)

            gen_model = model.module if hasattr(model, "module") else model

            # --- Decode kwargs
            decode_kwargs = dict(getattr(cfg, "decode_kwargs", {}))
            decode_kwargs.setdefault("num_beams", 4)
            decode_kwargs.setdefault("max_new_tokens", 120)
            decode_kwargs.setdefault("min_new_tokens", 3)
            decode_kwargs.setdefault("no_repeat_ngram_size", 3)
            decode_kwargs.setdefault("repetition_penalty", 1.02)
            decode_kwargs.setdefault("length_penalty", 0.9)
            decode_kwargs.setdefault("early_stopping", True)
            decode_kwargs.setdefault("use_cache", True)

            if getattr(cfg, "forced_bos_token_id", None) is not None:
                decode_kwargs["forced_bos_token_id"] = int(cfg.forced_bos_token_id)

            # --- Generate with AMP
            amp_ctx = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if ENABLE_AMP and torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                outs = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio=audio,
                    audio_mask=audio_mask,
                    **decode_kwargs,
                )

            # Debug first sequence once
            if not printed_debug_once and isinstance(outs, torch.Tensor) and outs.numel() > 0:
                first_ids = outs[0][:20].tolist() if outs.dim() > 1 else outs[:20].tolist()
                first_token = outs[0, 0].item() if outs.dim() > 1 else outs[0].item()
                cfg_forced = decode_kwargs.get("forced_bos_token_id", None)
                msg = "[DEBUG] First generated token IDs: " + repr(first_ids)
                msg += f"\n[DEBUG] first_token={first_token} | forced_bos(decode_kw)={cfg_forced}"
                print(msg)
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(msg + "\n")
                printed_debug_once = True

            # Decode to strings
            for seq in outs:
                if isinstance(seq, torch.Tensor):
                    seq = seq.tolist()
                pred = cfg.tokenizer.decode(seq, skip_special_tokens=True)
                if not pred.strip():
                    empty_count += 1
                    preds.append("")
                else:
                    preds.append(pred)

        # --- Aggregate & report
        if not preds:
            return {
                "bleu": 0.0,
                "bert_f1_corrected": 0.0,
                "count": 0,
                "ablation": ablation or "",
                "preds": [],
                "refs": [],
                "srcs": []
            }

        # Sample dumps
        sample_msg = ["\n--- Validation Samples ---" if not ablation else f"\n--- Validation Samples (ablation={ablation}) ---"]
        n = len(preds)
        idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        srcs_available = any(bool(s) for s in srcs)

        for i in idxs:
            if 0 <= i < n:
                if srcs_available:
                    sample_msg.append(f"[SRC ] {srcs[i]}")
                sample_msg.append(f"[REF ] {refs[i]}")
                sample_msg.append(f"[PRED] {preds[i]}")
                sample_msg.append("")
        sample_msg.append(f"[DEBUG] Empty predictions: {empty_count}/{len(preds)}")
        full_msg = "\n".join(sample_msg)
        print(full_msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(full_msg + "\n")

        # Metrics
        metrics_out = {}
        pairs = [(p.strip() or "_", r.strip() or "_") for p, r in zip(preds, refs)]
        _eval_preds, _eval_refs = zip(*pairs) if pairs else ([], [])

        try:
            bleu = sacrebleu.corpus_bleu(list(_eval_preds), [list(_eval_refs)], tokenize="13a", lowercase=False).score \
                   if _eval_preds else 0.0
        except Exception as e:
            print(f"[WARN] BLEU failed: {e}")
            bleu = float("nan")
        metrics_out["bleu"] = float(bleu)

        try:
            if _eval_preds:
                from bert_score import score as bertscore_score
                _, _, F = bertscore_score(
                    list(_eval_preds),
                    list(_eval_refs),
                    lang="en",
                    rescale_with_baseline=False,
                    batch_size=32
                )
                metrics_out["bert_f1_corrected"] = float(F.mean().item())
            else:
                metrics_out["bert_f1_corrected"] = 0.0
        except Exception as e:
            print(f"[WARN] BERTScore failed: {e}")
            metrics_out["bert_f1_corrected"] = float("nan")

        prfx = f"[ABLATION:{ablation}]" if ablation else "[VALID]"
        print(f"{prfx} BLEU: {metrics_out['bleu']:.2f} | BERT-F1: {metrics_out['bert_f1_corrected']:.4f}")
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"{prfx} BLEU: {metrics_out['bleu']:.2f} | BERT-F1: {metrics_out['bert_f1_corrected']:.4f}\n")

        metrics_out["preds"] = preds
        metrics_out["refs"]  = refs
        metrics_out["srcs"]  = srcs
        metrics_out["count"] = len(preds)
        metrics_out["ablation"] = ablation or ""

        return metrics_out


def log_metrics(scores, run_root, epoch, split="dev", optimizer=None):
    """Append metrics to CSV."""
    csv_path = os.path.join(run_root, "metrics.csv")
    header = [
        "epoch", "split", "bleu", "bert_f1", "num_examples",
        "ablated", "delta_bleu_main_off", "delta_bleu_main_shuffle",
        "learning_rate", "skipped_batches"
    ]
    write_header = (not os.path.exists(csv_path))

    lr_val = current_lr(optimizer) if optimizer else ""

    row = {
        "epoch": epoch,
        "split": split,
        "bleu": round(scores.get("bleu", float("nan")), 4),
        "bert_f1": round(scores.get("bert_f1_corrected", float("nan")), 4),
        "num_examples": scores.get("count", ""),
        "ablated": scores.get("ablation", ""),
        "delta_bleu_main_off": scores.get("delta_bleu_main_off", ""),
        "delta_bleu_main_shuffle": scores.get("delta_bleu_main_shuffle", ""),
        "learning_rate": f"{lr_val:.2e}" if lr_val else "",
        "skipped_batches": scores.get("skipped_batches", ""),
    }

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def label_smoothed_ce(logits, target, eps=0.1, ignore_index=-100):
    """Label-smoothed cross-entropy loss."""
    log_probs = F.log_softmax(logits, dim=-1)

    nll = F.nll_loss(
        log_probs.transpose(1, 2),
        target,
        reduction="none",
        ignore_index=ignore_index
    )

    denom = target.ne(ignore_index).sum().clamp(min=1)
    nll = nll.sum() / denom

    smooth = -log_probs.mean(dim=-1)
    smooth = smooth.sum() / denom

    return (1 - eps) * nll + eps * smooth


def main():
    args = parse_args()

    using_ddp, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg_train = get_config("train")
    cfg_dev = get_config("dev")
    cfg_train.device = device
    cfg_dev.device = device

    # Override config with args
    cfg_train.n_epoch = args.epochs
    cfg_train.batch_size = args.batch_size
    cfg_train.learning_rate = args.lr
    cfg_train.weight_decay = args.weight_decay
    cfg_train.warmup_steps = args.warmup_steps
    cfg_train.clip = args.clip

    # === Run directory ===
    # Use --run_dir if explicitly set, otherwise let config.py choose (it
    # respects the TRANS_RUN_DIR env var; defaults to ./runs/multimodal).
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.run_dir:
        run_root = args.run_dir
    else:
        run_root = cfg_train.run_dir  # already set by config.py
    cfg_train.run_dir = run_root

    os.makedirs(run_root, exist_ok=True)

    if is_main_process():
        print(f"[RUN] Outputs: {run_root}")
        print(f"📚 Tokenizer vocab size: {cfg_train.tokenizer.vocab_size}")

    timestamp_readable = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(run_root, f"training_{timestamp_readable}.txt")
    config_json = os.path.join(run_root, "config_snapshot.json")

    # Parse ablation epochs
    ablation_epochs = set(int(e.strip()) for e in args.ablations_epochs.split(",") if e.strip())

    if is_main_process():
        with open(log_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("🎯 STAGE-2 MULTIMODAL TRANSLATION (W2V CROSS-ATTENTION)\n")
            f.write("🔥 CORRECTED: Fixed optimizer timing + gate warm-start + improved diagnostics\n")
            f.write(f"Started: {timestamp_readable}\n")
            f.write(f"Variant: text+w2v_xattn (single cross-attention)\n")
            f.write(f"Run dir: {run_root}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"ENCODER_LR_MULT = {ENCODER_LR_MULT}\n")
            f.write(f"AUDIO_LR_MULT = {AUDIO_LR_MULT}\n")
            f.write(f"NO_MODDROP_EPOCHS = {NO_MODDROP_EPOCHS}\n")
            f.write(f"FREEZE_DECODER_UNTIL_EPOCH = {FREEZE_DECODER_UNTIL_EPOCH}\n")
            f.write("=" * 80 + "\n\n")

        config_snapshot = {
            "timestamp": timestamp_readable,
            "variant": "text+w2v_xattn_CORRECTED",
            "stage": "Stage-2 (Direct W2V Cross-Attention - Single, No Prosody)",
            "run_dir": run_root,
            "encoder_checkpoint": cfg_train.encoder_checkpoint,
            "hyperparameters": {
                "ENCODER_LR_MULT": ENCODER_LR_MULT,
                "AUDIO_LR_MULT": AUDIO_LR_MULT,
                "NO_MODDROP_EPOCHS": NO_MODDROP_EPOCHS,
                "FREEZE_DECODER_UNTIL_EPOCH": FREEZE_DECODER_UNTIL_EPOCH,
            },
            "model": {
                "mbart_model": cfg_train.mbart_model,
                "wav2vec_model": cfg_train.wav2vec_model,
                "audio_latents": cfg_train.audio_latents,
                "attn_heads": cfg_train.attn_heads,
                "use_prosody": cfg_train.use_prosody,
                "prosody_scale": cfg_train.prosody_scale,
                "modality_dropout": cfg_train.modality_dropout,
            },
            "training": {
                "n_epoch": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "clip": cfg_train.clip,
                "max_seq_len": cfg_train.max_seq_len,
                "max_audio_length": cfg_train.max_audio_length,
            },
            "data": {
                "train_csv": cfg_train.csv_file,
                "dev_csv": cfg_dev.csv_file,
                "normalize_medieval": cfg_train.normalize_medieval,
            },
            "generation": cfg_train.decode_kwargs,
            "evaluation": {
                "ablation_epochs": sorted(ablation_epochs),
                "ablation_max_batches": args.ablations_max_eval_batches,
            }
        }
        with open(config_json, "w") as f:
            json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    # Data loaders
    train_loader = get_dataloader(
        csv_file=cfg_train.csv_file,
        audio_dir=cfg_train.audio_dir,
        tokenizer=cfg_train.tokenizer,
        batch_size=cfg_train.batch_size,
        num_workers=cfg_train.num_workers,
        shuffle=True,
        max_src_length=cfg_train.max_seq_len,
        max_tgt_length=cfg_train.max_seq_len,
        max_audio_length=cfg_train.max_audio_length,
        normalize_medieval=cfg_train.normalize_medieval,
        normalize_arabic=cfg_train.normalize_arabic,
        normalize_arabic_numbers=cfg_train.normalize_arabic_numbers,
        normalizer_name=cfg_train.normalizer_name,
        sample_rate=cfg_train.sample_rate,
    )

    dev_loader = get_dataloader(
        csv_file=cfg_dev.csv_file,
        audio_dir=cfg_dev.audio_dir,
        tokenizer=cfg_dev.tokenizer,
        batch_size=cfg_dev.batch_size,
        num_workers=cfg_dev.num_workers,
        shuffle=False,
        max_src_length=cfg_dev.max_seq_len,
        max_tgt_length=cfg_dev.max_seq_len,
        max_audio_length=cfg_dev.max_audio_length,
        normalize_medieval=cfg_dev.normalize_medieval,
        normalize_arabic=cfg_dev.normalize_arabic,
        normalize_arabic_numbers=cfg_dev.normalize_arabic_numbers,
        normalizer_name=cfg_dev.normalizer_name,
        sample_rate=cfg_dev.sample_rate,
    )

    # === Model instantiation: Stage-2 (Direct W2V, no prosody) ===
    # Resolve PPFT (text baseline) checkpoint paths:
    #   1. --text-baseline-pt / --text-baseline-hf CLI flags (highest priority)
    #   2. TEXT_BASELINE_PT / TEXT_BASELINE_HF env vars
    #   3. cfg_train.text_baseline_pt / cfg_train.text_baseline_hf (also from env)
    #   4. None — error
    TEXT_BASELINE_PT = (
        args.text_baseline_pt
        or os.getenv("TEXT_BASELINE_PT")
        or cfg_train.text_baseline_pt
    )
    TEXT_BASELINE_HF = (
        args.text_baseline_hf
        or os.getenv("TEXT_BASELINE_HF")
        or cfg_train.text_baseline_hf
    )

    if not TEXT_BASELINE_PT and not TEXT_BASELINE_HF:
        raise RuntimeError(
            "No PPFT (text baseline) checkpoint specified. Provide one of:\n"
            "  --text-baseline-pt /path/to/best_finetuned.pt\n"
            "  --text-baseline-hf /path/to/best_hf/\n"
            "  TEXT_BASELINE_PT env var\n"
            "  TEXT_BASELINE_HF env var\n"
            "Train the text baseline first via src/text_baseline/finetune.py."
        )

    if is_main_process():
        print(f"\n{'='*80}")
        print(f"MODEL INSTANTIATION - STAGE-2 (W2V SINGLE CROSS-ATTENTION)")
        print(f"{'='*80}")
        print(f"Encoder checkpoint: {TEXT_BASELINE_PT}")
        print(f"Phase-1 alignment: DISABLED (direct W2V→decoder)")
        print(f"Architecture: Single cross-attention (text←audio)")
        print(f"Prosody: DISABLED")
        print(f"{'='*80}\n")

    model = MultimodalTranslationModel(
        tokenizer=cfg_train.tokenizer,
        decoder_lang_id=cfg_train.decoder_start_token_id,
        normalize_medieval=cfg_train.normalize_medieval,
        normalize_arabic=getattr(cfg_train, "normalize_arabic", True),
        normalize_arabic_numbers=getattr(cfg_train, "normalize_arabic_numbers", True),
        normalizer_name=getattr(cfg_train, "normalizer_name", "hazm"),
        encoder_ckpt_path=TEXT_BASELINE_PT,
        hf_baseline_dir=TEXT_BASELINE_HF,
        mbart_model_name=cfg_train.mbart_model,
        num_audio_latents=cfg_train.audio_latents,
        num_heads=cfg_train.attn_heads,
        modality_dropout_p=cfg_train.modality_dropout,
        use_prosody=cfg_train.use_prosody,
        prosody_scale=cfg_train.prosody_scale,
        audio_scale_init=getattr(cfg_train, "audio_scale_init", 1.5),
        modality_dropout_mode="zero",
    ).to(device)

    if using_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False
        )

    # === Trainability diagnostic (rank-0 only) ===
    if is_main_process():
        mod = model.module if hasattr(model, "module") else model

        if hasattr(mod, "start_new_epoch"):
            mod.start_new_epoch()

        print("\n" + "="*80)
        print("V8 PARAMETER TRAINABILITY CHECK (CORRECTED)")
        print("PARAMETER NAME SCAN (Audio/Fusion modules)")
        print("="*80)

        AF_KEYS = [
            "audio_proj", "audio_proj_ln", "audio_resampler",
            "perceiver", "resampler", "dual_xattn", "cross_attn",
            "fusion_out_proj", "gate", "fuse_gate", "gate_mlp",
            "audio_scale", "ln_gate",
        ]
        DECODER_PREFIXES = ["mbart.model.decoder", "lm_head"]

        audio_count = 0
        fusion_count = 0
        decoder_count = 0

        audio_fusion_names = []
        decoder_names = []

        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue

            if any(k in n for k in AF_KEYS):
                if any(k in n for k in ["audio_proj", "audio_proj_ln", "audio_resampler", "perceiver", "resampler", "audio_scale"]):
                    audio_count += p.numel()
                else:
                    fusion_count += p.numel()

                audio_fusion_names.append(n)

            elif any(n.startswith(pref) or f".{pref}." in n for pref in DECODER_PREFIXES):
                decoder_count += p.numel()
                decoder_names.append(n)

        print("\n[AUDIO/FUSION PARAMS] Found", len(audio_fusion_names), "trainable tensors (showing first 20):")
        for name in audio_fusion_names[:20]:
            print(f"  ✓ {name}")
        if len(audio_fusion_names) > 20:
            print(f"  ... and {len(audio_fusion_names) - 20} more")

        print("\n[DECODER PARAMS] Found", len(decoder_names), "trainable tensors (showing first 10):")
        for name in decoder_names[:10]:
            print(f"  ✓ {name}")
        if len(decoder_names) > 10:
            print(f"  ... and {len(decoder_names) - 10} more")

        print("\n[SUMMARY]")
        print(f"  Audio pathway:  {audio_count:>12,} trainable params")
        print(f"  Fusion blocks:  {fusion_count:>12,} trainable params")
        print(f"  Decoder + LM:   {decoder_count:>12,} trainable params")
        print(f"  TOTAL:          {audio_count + fusion_count + decoder_count:>12,} trainable params")

        if hasattr(mod, "dual_xattn") and hasattr(mod.dual_xattn, "audio_scale"):
            try:
                scale_val = float(mod.dual_xattn.audio_scale.item())
                scale_grad = mod.dual_xattn.audio_scale.requires_grad
                print(f"\n[AUDIO SCALE] Initialized to: {scale_val:.2f} | Trainable: {scale_grad}")
            except Exception:
                print("\n[AUDIO SCALE] Found but could not read value (OK).")
        else:
            print("\n[WARNING] audio_scale parameter NOT FOUND in model!")

        if (audio_count + fusion_count) == 0:
            print("\n[ERROR] ✗ Audio/Fusion stack has 0 trainable params!")
            sys.exit(1)
        else:
            print("\n[OK] ✓ Audio/Fusion stack is trainable")

        print("="*80 + "\n")

    # === Eval-only mode ===
    if args.eval_only:
        if args.checkpoint:
            if is_main_process():
                print(f"[EVAL] Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device)

            state = ckpt.get("state_dict", ckpt)

            if using_ddp:
                model.module.load_state_dict(state, strict=True)
            else:
                model.load_state_dict(state, strict=True)

        barrier()

        if is_main_process():
            ablation = args.eval_only_ablation
            scores = validate(
                model, dev_loader, cfg_dev,
                epoch=0,
                log_file=log_path,
                max_eval_batches=args.ablations_max_eval_batches if ablation else None,
                ablation=ablation
            )
            log_metrics(scores, run_root, epoch=0, split=f"eval{'_'+ablation if ablation else ''}")
            print(f"\n[EVAL] Results saved to {run_root}")

        cleanup_ddp()
        sys.exit(0)

    # ============================================================================
    # === CRITICAL FIX: Unwrap model BEFORE building optimizer ===
    # ============================================================================
    unwrapped = model.module if hasattr(model, "module") else model
    device = next(unwrapped.parameters()).device

    # Build optimizer on unwrapped model
    optimizer = build_optimizer(unwrapped, args)

    # === NEW: Sanity check param membership ===
    if is_main_process():
        names_by_id = {id(p): n for n, p in unwrapped.named_parameters()}
        present = set()
        for g in optimizer.param_groups:
            for p in g["params"]:
                present.add(names_by_id.get(id(p), f"<unnamed:{id(p)}>"))

        to_check = [
            "dual_xattn._audio_scale_raw",
            "_audio_gain_raw",
            "audio_proj.weight",
            "audio_resampler",
            "mbart.model.decoder.layers.0.self_attn.k_proj.weight",
        ]
        print("\n[OPT-CHECK] Parameter membership verification:")
        for key in to_check:
            hits = [n for n in present if key in n]
            status = "✓ OK" if hits else "✗ MISSING"
            print(f"  {status:8s} {key}")
            if hits and len(hits) <= 3:
                for h in hits:
                    print(f"           └─ {h}")

        # Verify distinct LR groups exist
        print("\n[OPT-CHECK] LR group verification:")
        for g in optimizer.param_groups:
            name = g.get("name", "unnamed")
            lr = g.get("lr", args.lr)
            n_params = sum(p.numel() for p in g["params"])
            print(f"  ✓ {name:15s} | LR={lr:.2e} | {n_params:,} params")
    # ============================================================================

    # === Training setup ===
    best_bleu = -1.0
    best_bert = 0.0

    total_steps = args.epochs * len(train_loader) // max(1, args.grad_accum)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.03 * total_steps)

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if is_main_process():
        print(f"\n{'='*80}")
        print(f"TRAINING SETUP")
        print(f"{'='*80}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Base learning rate: {args.lr}")
        print(f"Text encoder LR multiplier: {ENCODER_LR_MULT}x")
        print(f"Audio LR multiplier: {AUDIO_LR_MULT}x")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Gradient accumulation: {args.grad_accum}")
        print(f"Gradient clip: {args.clip}")
        print(f"No mod-drop epochs: {NO_MODDROP_EPOCHS}")
        print(f"Freeze decoder until epoch: {FREEZE_DECODER_UNTIL_EPOCH}")
        print(f"Ablation epochs: {sorted(ablation_epochs)}")
        print(f"Ablation batch limit: {args.ablations_max_eval_batches}")
        print(f"AMP enabled: {ENABLE_AMP} (dtype={AMP_DTYPE})")
        print(f"{'='*80}\n")

    # === Training loop ===
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        if is_main_process():
            if hasattr(unwrapped, 'start_new_epoch'):
                unwrapped.start_new_epoch()
                print(f"[TRAIN] ✓ Epoch {epoch} initialization complete")

        # ========================================================================
        # === CRITICAL FIX: Decoder freezing INSIDE training loop (all ranks) ===
        # ========================================================================
        if FREEZE_DECODER_UNTIL_EPOCH > 0:
            freeze_now = (epoch <= FREEZE_DECODER_UNTIL_EPOCH)
            set_decoder_frozen(unwrapped, frozen=freeze_now)

            if is_main_process():
                freeze_msg = f"[FREEZE] epoch={epoch} → decoder_frozen={freeze_now}"
                print(freeze_msg)
                with open(log_path, "a") as f:
                    f.write(freeze_msg + "\n")
        # ========================================================================

        # === Keep frozen encoders in eval() mode (deterministic) ===
        if hasattr(unwrapped, "text_encoder"):
            unwrapped.text_encoder.eval()
        if hasattr(unwrapped, "audio_encoder"):
            unwrapped.audio_encoder.eval()

        # ========================================================================
        # === NEW: Gate warm-start for first 2 epochs (all ranks) ===
        # ========================================================================
        if hasattr(unwrapped, "dual_xattn"):
            unwrapped.dual_xattn.warm_force_alpha = 0.5 if epoch <= 2 else None
            if is_main_process():
                print(f"[GATE] epoch {epoch}: warm_force_alpha = "
                      f"{unwrapped.dual_xattn.warm_force_alpha}")
        # ========================================================================

        # === Modality dropout schedule ===
        if hasattr(unwrapped, "set_mod_drop_p"):
            if epoch <= NO_MODDROP_EPOCHS:
                new_p = 0.0
            else:
                new_p = cfg_train.modality_dropout

            unwrapped.set_mod_drop_p(new_p)

            if is_main_process():
                moddrop_msg = f"[MOD-DROP] epoch={epoch} → p set to {new_p:.2f}"
                print(moddrop_msg)
                with open(log_path, "a") as f:
                    f.write(moddrop_msg + "\n")

        if is_main_process():
            # Pretty-print param groups with live LR
            msg = [f"\n{'='*80}", f"EPOCH {epoch}/{args.epochs}", f"{'='*80}",
                   f"[OPTIM] {len(optimizer.param_groups)} param groups"]
            last_lrs = scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else []
            for i, g in enumerate(optimizer.param_groups):
                n = sum(p.numel() for p in g["params"])
                lr = last_lrs[i] if i < len(last_lrs) else g.get('lr', args.lr)
                name = g.get('name', f'group{i}')
                wd = g.get('weight_decay', 0.0)
                msg.append(f"  - {name}: {n:,} params | lr={lr:.2e} | wd={wd}")
            msg = "\n".join(msg) + "\n"
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

        running_loss = 0.0
        skipped_batches = 0
        grad_accum_steps = 0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch}]") if is_main_process() else train_loader

        for step, batch in enumerate(loop, start=1):
            # Load/prepare batch
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            audio = batch.get("audio", None)
            audio_mask = batch.get("audio_mask", None)
            if audio is not None:
                audio = audio.to(device, non_blocking=True)
                if audio.dim() == 3 and audio.size(1) == 1:
                    audio = audio.squeeze(1)
            if isinstance(audio_mask, torch.Tensor):
                audio_mask = audio_mask.to(device, non_blocking=True)
                if audio_mask.dtype != torch.bool:
                    audio_mask = (audio_mask != 0)

            # One-time audio stats
            if is_main_process() and epoch == 1 and step == 1 and audio is not None:
                aud_mean = float(audio.abs().mean().item())
                aud_max  = float(audio.abs().max().item())
                aud_std  = float(audio.std().item())
                mask_frac = float((audio_mask != 0).float().mean().item()) if audio_mask is not None else 0.0
                aud_msg = (f"[TRAIN] First batch audio: mean_abs={aud_mean:.6f}, "
                           f"max_abs={aud_max:.6f}, std={aud_std:.6f}, mask_frac={mask_frac:.3f}")
                print(aud_msg)
                with open(log_path, "a") as f:
                    f.write(aud_msg + "\n")
                if aud_mean < 1e-3:
                    warn = "[WARNING] Training audio signal is very weak!"
                    print(warn)
                    open(log_path, "a").write(warn + "\n")

            # === Forward with AMP ===
            amp_ctx = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if ENABLE_AMP and torch.cuda.is_available() else nullcontext()

            with amp_ctx:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio=audio,
                    audio_mask=audio_mask,
                    labels=labels
                )

                if out.get("logits", None) is None:
                    loss = (labels.ne(-100).sum() * 0.0).mean()
                    skipped_batches += 1
                else:
                    logits = out["logits"]
                    loss = label_smoothed_ce(
                        logits=logits,
                        target=labels,
                        eps=0.1,
                        ignore_index=-100
                    )

                    if not torch.isfinite(loss):
                        if is_main_process():
                            warn_msg = f"[WARN] Non-finite loss at step {step}; skipping."
                            print(warn_msg)
                            open(log_path, "a").write(warn_msg + "\n")
                        optimizer.zero_grad(set_to_none=True)
                        skipped_batches += 1
                        continue

            # === Gate regularizer ===
            try:
                alpha_reg = torch.tensor(0.0, device=loss.device)

                mean_alpha = None
                if hasattr(unwrapped, "dual_xattn"):
                    mean_alpha = getattr(unwrapped.dual_xattn, "last_mean_alpha", None)

                if mean_alpha is not None:
                    if torch.is_tensor(mean_alpha):
                        mean_alpha_t = mean_alpha
                        if mean_alpha_t.numel() > 1:
                            mean_alpha_t = mean_alpha_t.mean()
                        if mean_alpha_t.device != loss.device:
                            mean_alpha_t = mean_alpha_t.to(loss.device)
                    else:
                        mean_alpha_t = torch.tensor(float(mean_alpha), device=loss.device, requires_grad=True)

                    mean_alpha_t = mean_alpha_t.clamp(0.0, 1.0)
                    alpha_reg = 0.10 * (mean_alpha_t - 0.50) ** 2
                    loss = loss + alpha_reg

                    if is_main_process() and (global_step % 100 == 0):
                        print(f"[GATE-α] step={global_step} mean_alpha={float(mean_alpha_t):.4f} "
                              f"alpha_reg={float(alpha_reg.detach()):.6f}")
            except Exception:
                pass

            # Backward / Accum
            loss = loss / args.grad_accum
            loss.backward()
            grad_accum_steps += 1

            if grad_accum_steps >= args.grad_accum:
                # ================================================================
                # === NEW: Improved gradient diagnostics ===
                # ================================================================
                if is_main_process() and (global_step % 200 == 0):
                    audio_grad = fusion_grad = decoder_grad = 0.0
                    audio_count = fusion_count = decoder_count = 0

                    AF_AUDIO_KEYS = ["audio_proj", "audio_proj_ln", "audio_resampler", "resampler", "perceiver", "_audio_gain"]
                    AF_FUSION_KEYS = ["dual_xattn", "fusion", "fusion_out_proj", "gate", "ln_gate", "_audio_scale"]
                    DECODER_KEYS   = ["mbart.model.decoder", "lm_head"]

                    probe = []

                    for name, p in unwrapped.named_parameters():
                        g = p.grad
                        if (g is None) or (not torch.isfinite(g).all()):
                            continue
                        gn = g.norm().item()

                        if any(k in name for k in AF_AUDIO_KEYS):
                            audio_grad += gn * gn
                            audio_count += 1
                        elif any(k in name for k in AF_FUSION_KEYS):
                            fusion_grad += gn * gn
                            fusion_count += 1
                        elif any(name.startswith(k) or f".{k}." in name for k in DECODER_KEYS):
                            decoder_grad += gn * gn
                            decoder_count += 1

                        # Collect audio/fusion params for probe
                        if any(k in name for k in ["audio_proj", "audio_resampler", "dual_xattn", "audio_scale", "audio_gain"]):
                            probe.append((name, gn))

                    audio_grad = math.sqrt(audio_grad)
                    fusion_grad = math.sqrt(fusion_grad)
                    decoder_grad = math.sqrt(decoder_grad)

                    msg = (f"[GRAD SPLIT] audio={audio_grad:.4f} ({audio_count} params) | "
                           f"fusion={fusion_grad:.4f} ({fusion_count} params) | "
                           f"decoder={decoder_grad:.4f} ({decoder_count} params)")
                    print(msg)
                    open(log_path, "a").write(msg + "\n")

                    # Calculate ratios
                    if decoder_grad > 0:
                        audio_ratio = (audio_grad / decoder_grad) * 100
                        fusion_ratio = (fusion_grad / decoder_grad) * 100
                        ratio_msg = (f"[GRAD RATIO] audio/decoder: {audio_ratio:.1f}% | "
                                   f"fusion/decoder: {fusion_ratio:.1f}%")
                        print(ratio_msg)
                        open(log_path, "a").write(ratio_msg + "\n")

                    # Probe top audio/fusion gradients
                    probe.sort(key=lambda x: -x[1])
                    if probe:
                        probe_msg = "[PROBE] " + "; ".join(
                            f"{k.split('.')[-1]}={v:.3e}" for k, v in probe[:6]
                        )
                        print(probe_msg)
                        open(log_path, "a").write(probe_msg + "\n")
                    else:
                        no_probe_msg = "[PROBE] No audio/fusion parameters with finite gradients found!"
                        print(no_probe_msg)
                        open(log_path, "a").write(no_probe_msg + "\n")

                    # Warnings with specific thresholds
                    if global_step > 1000:
                        if audio_grad < 1e-3:
                            warn = f"[GRAD-WARN] ⚠️ Audio gradients critically low ({audio_grad:.4f} < 1e-3) after 1000 steps!"
                            print(warn)
                            open(log_path, "a").write(warn + "\n")
                        if audio_ratio < 5.0:
                            warn = f"[GRAD-WARN] ⚠️ Audio gradients are only {audio_ratio:.1f}% of decoder (target: ≥5-15%)!"
                            print(warn)
                            open(log_path, "a").write(warn + "\n")
                # ================================================================

                # Clip / Step / Schedule
                torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), cfg_train.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                grad_accum_steps = 0
                global_step += 1

                # Early LR trace
                if is_main_process() and epoch == 1 and global_step <= 5:
                    lr_now = optimizer.param_groups[0]['lr']
                    msg = f"[STEP {global_step}] LR after step: {lr_now:.2e}"
                    print(msg)
                    open(log_path, "a").write(msg + "\n")

            # tqdm meter
            running_loss += float(loss.detach().cpu()) * args.grad_accum
            if is_main_process():
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]['lr']
                loop.set_postfix({"loss": f"{running_loss/step:.4f}", "lr": f"{lr:.2e}"})

        barrier()

        # === VALIDATION ===
        if is_main_process():
            print(f"\n{'='*40}")
            print(f"EPOCH {epoch} VALIDATION")
            print(f"{'='*40}\n")

            main_results = validate(
                model, dev_loader, cfg_dev,
                epoch=epoch,
                log_file=log_path,
                max_eval_batches=getattr(cfg_dev, "max_eval_batches", 200),
                ablation=None
            )
            dev_bleu = main_results["bleu"]
            dev_bert = main_results["bert_f1_corrected"]
            dev_preds = main_results["preds"]
            dev_refs = main_results["refs"]
            dev_srcs = main_results["srcs"]

            main_results["skipped_batches"] = skipped_batches
            log_metrics(main_results, run_root, epoch, split="dev", optimizer=optimizer)

            # ABLATIONS
            audio_off_bleu = None
            audio_off_bert = None
            audio_shuffle_bleu = None
            audio_shuffle_bert = None

            if epoch in ablation_epochs:
                print(f"\n[ABLATION] Running diagnostic evaluations (subset of {args.ablations_max_eval_batches} batches)...")

                audio_off_results = validate(
                    model, dev_loader, cfg_dev,
                    epoch=epoch,
                    log_file=log_path,
                    max_eval_batches=args.ablations_max_eval_batches,
                    ablation="audio_off"
                )
                audio_off_bleu = audio_off_results["bleu"]
                audio_off_bert = audio_off_results["bert_f1_corrected"]
                audio_off_results["delta_bleu_main_off"] = dev_bleu - audio_off_bleu
                log_metrics(audio_off_results, run_root, epoch, split="dev_off", optimizer=optimizer)

                audio_shuffle_results = validate(
                    model, dev_loader, cfg_dev,
                    epoch=epoch,
                    log_file=log_path,
                    max_eval_batches=args.ablations_max_eval_batches,
                    ablation="audio_shuffle"
                )
                audio_shuffle_bleu = audio_shuffle_results["bleu"]
                audio_shuffle_bert = audio_shuffle_results["bert_f1_corrected"]
                audio_shuffle_results["delta_bleu_main_shuffle"] = dev_bleu - audio_shuffle_bleu
                log_metrics(audio_shuffle_results, run_root, epoch, split="dev_shuffle", optimizer=optimizer)

        else:
            dev_bleu = None
            dev_bert = None
            audio_off_bleu = None
            audio_off_bert = None
            audio_shuffle_bleu = None
            audio_shuffle_bert = None
            dev_preds = []
            dev_refs = []
            dev_srcs = []

        barrier()

        # === EPOCH SUMMARY ===
        if is_main_process():
            avg_loss = running_loss / max(1, len(train_loader))
            audio_benefit = (dev_bleu - audio_off_bleu) if audio_off_bleu is not None else None
            audio_shuffle_delta = (dev_bleu - audio_shuffle_bleu) if audio_shuffle_bleu is not None else None
            lr = scheduler.get_last_lr()[0]

            summary = [
                "",
                f"{'='*80}",
                f"EPOCH {epoch} SUMMARY",
                f"{'='*80}",
                f"Train Loss:        {avg_loss:.4f}",
                f"Dev BLEU:          {dev_bleu:.2f}",
                f"Dev BERT-F1:       {dev_bert:.4f}",
            ]

            if audio_off_bleu is not None:
                summary.append(f"[AUDIO-OFF] BLEU:  {audio_off_bleu:.2f} | BERT-F1: {audio_off_bert:.4f}")
                summary.append(f"[Δ MAIN-OFF]:      {audio_benefit:+.2f} BLEU")

            if audio_shuffle_bleu is not None:
                summary.append(f"[AUDIO-SHUFFLE]:   {audio_shuffle_bleu:.2f} BLEU | BERT-F1: {audio_shuffle_bert:.4f}")
                summary.append(f"[Δ MAIN-SHUFFLE]:  {audio_shuffle_delta:+.2f} BLEU")

            summary.extend([
                f"Learning Rate:     {lr:.2e}",
                f"Skipped Batches:   {skipped_batches}",
                f"{'='*80}\n"
            ])

            summary_text = "\n".join(summary)
            print(summary_text)

            with open(log_path, "a") as f:
                f.write(summary_text)

            # Save samples
            samples_file = os.path.join(run_root, f"samples_epoch_{epoch:02d}.txt")
            with open(samples_file, "w", encoding="utf-8") as f:
                f.write(f"EPOCH {epoch} VALIDATION SAMPLES\n")
                f.write("=" * 80 + "\n\n")
                n = len(dev_preds)
                idxs = [0, n // 2, n - 1] if n >= 3 else list(range(min(n, 3)))
                for i in idxs:
                    if i < len(dev_srcs) and i < len(dev_refs) and i < len(dev_preds):
                        f.write(f"Sample {i+1}:\n")
                        f.write(f"[SRC ] {dev_srcs[i]}\n")
                        f.write(f"[REF ] {dev_refs[i]}\n")
                        f.write(f"[PRED] {dev_preds[i]}\n")
                        f.write("\n")

        # === CHECKPOINT SAVING ===
        if is_main_process() and dev_bleu > best_bleu:
            best_bleu = dev_bleu
            best_bert = dev_bert
            best_path = os.path.join(run_root, "best_dev_checkpoint.pt")
            torch.save(
                {
                    "state_dict": unwrapped.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "hyperparameters": {
                        "ENCODER_LR_MULT": ENCODER_LR_MULT,
                        "AUDIO_LR_MULT": AUDIO_LR_MULT,
                        "NO_MODDROP_EPOCHS": NO_MODDROP_EPOCHS,
                        "FREEZE_DECODER_UNTIL_EPOCH": FREEZE_DECODER_UNTIL_EPOCH,
                    }
                },
                best_path
            )

            save_msg = f"✅ New best BLEU {best_bleu:.2f}, BERT-F1 {best_bert:.4f} | Saved: {best_path}\n"
            print(save_msg)
            with open(log_path, "a") as f:
                f.write(save_msg)

        # === EARLY STOPPING CHECK ===
        if is_main_process():
            if epoch >= 5 and dev_bleu < best_bleu - 2.0:
                stop_msg = f"[EARLY STOP] BLEU dropped from {best_bleu:.2f} to {dev_bleu:.2f}. Stopping.\n"
                print(stop_msg)
                with open(log_path, "a") as f:
                    f.write(stop_msg)
                break

        barrier()

    # === TRAINING COMPLETE ===
    if is_main_process():
        final_msg = [
            "",
            "=" * 80,
            "TRAINING COMPLETED",
            "=" * 80,
            f"Final Best BLEU: {best_bleu:.2f}",
            f"Final Best BERT-F1: {best_bert:.4f}",
            f"Metrics saved to: {os.path.join(run_root, 'metrics.csv')}",
            f"Config saved to: {config_json}",
            "",
            "APPLIED PATCHES:",
            f"  ✓ ENCODER_LR_MULT = {ENCODER_LR_MULT}",
            f"  ✓ AUDIO_LR_MULT = {AUDIO_LR_MULT}",
            f"  ✓ NO_MODDROP_EPOCHS = {NO_MODDROP_EPOCHS}",
            f"  ✓ FREEZE_DECODER_UNTIL_EPOCH = {FREEZE_DECODER_UNTIL_EPOCH}",
            f"  ✓ Optimizer unwrapping fixed",
            f"  ✓ Gate warm-start mechanism added",
            f"  ✓ Improved gradient diagnostics",
            "=" * 80
        ]
        final_text = "\n".join(final_msg) + "\n"
        print(final_text)
        with open(log_path, "a") as f:
            f.write(final_text)

    cleanup_ddp()


if __name__ == "__main__":
    main()