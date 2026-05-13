"""
Evaluation script for the PPFT text-only baseline.

Computes BLEU (sacrebleu), chrF++ (sacrebleu), BERTScore F1, and (optionally)
COMET on the Persian↔English Masnavi test set.

Reference: SilkRoadNLP 2026 paper, Table 2.

Usage:
    python src/text_baseline/evaluate.py \\
        --test-csv path/to/test.csv \\
        --checkpoint runs/ppft/checkpoints/best_hf \\
        --output runs/eval_report.json

Single-GPU is enough; runs on CPU too (slower, but works).
"""

import argparse
import json
import os
import re

import torch
import pandas as pd
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


# ---------------------------------------------------------------------------
# Persian normalization (inlined so script is self-contained)
# ---------------------------------------------------------------------------

_ARABIC_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_fa(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("\u064A", "\u06CC")
    s = s.replace("\u0643", "\u06A9")
    s = _ARABIC_DIAC.sub("", s).replace("\u0640", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_from_hf(hf_dir: str, device: torch.device):
    """Load a HuggingFace-saved checkpoint directory (best_hf/)."""
    tok = MBart50TokenizerFast.from_pretrained(hf_dir)
    mdl = MBartForConditionalGeneration.from_pretrained(hf_dir)
    tok.src_lang = "fa_IR"
    tok.tgt_lang = "en_XX"
    en_id = tok.lang_code_to_id["en_XX"]
    mdl.config.forced_bos_token_id = en_id
    mdl.config.decoder_start_token_id = en_id
    mdl.to(device).eval()
    return mdl, tok


def load_from_pt(pt_path: str, device: torch.device):
    """Load a raw .pt state dict checkpoint."""
    tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    tok.src_lang = "fa_IR"
    tok.tgt_lang = "en_XX"
    mdl = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

    ckpt = torch.load(pt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = mdl.load_state_dict(state, strict=False)
    if len(missing) or len(unexpected):
        print(f"[WARN] PT load: missing={len(missing)} unexpected={len(unexpected)}")

    en_id = tok.lang_code_to_id["en_XX"]
    mdl.config.forced_bos_token_id = en_id
    mdl.config.decoder_start_token_id = en_id
    mdl.to(device).eval()
    return mdl, tok


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def translate(
    model,
    tokenizer,
    texts,
    device: torch.device,
    batch_size: int = 32,
    max_len: int = 256,
    num_beams: int = 8,
):
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Decoding"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
        ).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_len,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            length_penalty=1.05,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(hyps, refs, srcs, device: torch.device):
    """Compute BLEU, chrF++, BERTScore, and optionally COMET."""
    metrics = {}

    # BLEU + chrF++
    from sacrebleu import corpus_bleu, corpus_chrf
    metrics["BLEU"] = round(corpus_bleu(hyps, [refs]).score, 4)
    metrics["chrF++"] = round(corpus_chrf(hyps, [refs], word_order=2).score, 4)

    # BERTScore F1
    try:
        from bert_score import score as bert_score
        _, _, F1 = bert_score(hyps, refs, lang="en", verbose=False)
        metrics["BERTScore_F1"] = round(F1.mean().item(), 4)
    except Exception as e:
        print(f"[BERTScore] Skipped ({e})")
        metrics["BERTScore_F1"] = None

    # COMET (optional, downloads model on first use)
    try:
        from comet import download_model, load_from_checkpoint
        ckpt_path = download_model("Unbabel/wmt22-comet-da")
        comet = load_from_checkpoint(ckpt_path).to(device)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        _, sys_score = comet.predict(
            data, batch_size=32,
            gpus=1 if device.type == "cuda" else 0,
        )
        metrics["COMET"] = round(float(sys_score), 4)
    except Exception as e:
        print(f"[COMET] Skipped ({e})")
        metrics["COMET"] = None

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the PPFT text-only baseline.")
    p.add_argument("--test-csv", required=True,
                   help="Test set CSV with columns `persian_text`, `english_translation`.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to HuggingFace dir OR .pt file containing the fine-tuned model.")
    p.add_argument("--output", default=None,
                   help="Optional path to write the metrics JSON.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=8)
    p.add_argument("--num-samples-printed", type=int, default=5,
                   help="Print this many SRC/HYP/REF triples for inspection.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Load data
    df = pd.read_csv(args.test_csv)
    srcs = [normalize_fa(x) for x in df["persian_text"].astype(str).tolist()]
    refs = df["english_translation"].astype(str).tolist()
    print(f"[DATA] {len(srcs)} test pairs from {args.test_csv}")

    # Load model
    if os.path.isdir(args.checkpoint):
        model, tok = load_from_hf(args.checkpoint, device)
        print(f"[MODEL] Loaded HF directory: {args.checkpoint}")
    else:
        model, tok = load_from_pt(args.checkpoint, device)
        print(f"[MODEL] Loaded PT checkpoint: {args.checkpoint}")

    # Translate
    hyps = translate(
        model, tok, srcs, device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_beams=args.num_beams,
    )

    # Print samples
    print("\n--- Sample translations ---")
    for i in range(min(args.num_samples_printed, len(srcs))):
        print(f"\n[{i+1}] SRC: {srcs[i]}")
        print(f"    HYP: {hyps[i]}")
        print(f"    REF: {refs[i]}")

    # Compute metrics
    print("\n--- Computing metrics ---")
    metrics = compute_metrics(hyps, refs, srcs, device)

    print("\n--- Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "test_csv": args.test_csv,
                "n_samples": len(srcs),
                "metrics": metrics,
            }, f, indent=2)
        print(f"\n[SAVE] {args.output}")


if __name__ == "__main__":
    main()
