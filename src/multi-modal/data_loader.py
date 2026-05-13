# data_loader.py
# Handles loading and processing of aligned English, Persian text, and audio files.
# Written by Soheila — V7 tweaks for Stage-2 (text + W2V cross-attention)
# Winter 2025 / Fall 2025

import os
import re
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import get_worker_info

try:
    # Optional: a project-provided poetry normalizer with archaic-form
    # replacements. If not available, we fall back to the inline
    # _basic_persian_normalize() below, which is sufficient for the
    # paper's pipeline.
    from normalizers import persian_poetry_normalizer as external_poetry_norm
except Exception:
    external_poetry_norm = None

# --- Lightweight Arabic/Persian normalizer
_ARABIC_DIACRITICS = re.compile(
    "["  # combining marks (Arabic)
    "\u0610-\u061A"
    "\u064B-\u065F"
    "\u0670"
    "\u06D6-\u06ED"
    "]"
)

# Arabic-Indic and Eastern-Arabic digits to ASCII
_DIGIT_MAP = {
    ord("٠"): "0", ord("١"): "1", ord("٢"): "2", ord("٣"): "3", ord("٤"): "4",
    ord("٥"): "5", ord("٦"): "6", ord("٧"): "7", ord("٨"): "8", ord("٩"): "9",
    ord("۰"): "0", ord("۱"): "1", ord("۲"): "2", ord("۳"): "3", ord("۴"): "4",
    ord("۵"): "5", ord("۶"): "6", ord("۷"): "7", ord("۸"): "8", ord("۹"): "9",
}

def _basic_persian_normalize(
    text: str,
    normalize_arabic: bool = True,
    normalize_arabic_numbers: bool = True,
    use_medieval_poetry_rules: bool = False,
    normalizer_name: str = "arabic_persian_basic",
) -> str:
    """Minimal, conservative normalization for Persian poetry lines."""
    if not isinstance(text, str):
        return text

    s = text

    # Optional project-provided poetry normalizer comes first (if present)
    if use_medieval_poetry_rules and external_poetry_norm is not None:
        try:
            s = external_poetry_norm(s)
        except Exception:
            pass

    if normalize_arabic:
        # Unify Yeh/Kaf variants
        s = s.replace("ي", "ی").replace("ى", "Ý")  # handle rare Yeh variant
        s = s.replace("Ý", "ی")                    # collapse to Persian Yeh
        s = s.replace("ك", "ک")                    # Arabic Kaf -> Persian Kaf
        # Remove tatweel
        s = s.replace("ـ", "")
        # Remove combining diacritics (keep poetry punctuation)
        s = _ARABIC_DIACRITICS.sub("", s)
        s = re.sub(r"\s+", " ", s).strip()

    if normalize_arabic_numbers:
        s = s.translate(_DIGIT_MAP)

    # Replace ZWNJ with space, then collapse spaces
    s = s.replace("\u200c", " ").strip()
    s = re.sub(r"\s+", " ", s)

    return s


EPS = 1e-8

class MultimodalTranslationDataset(Dataset):
    """
    Loads Persian source text, English target text, and aligned audio.
    Applies optional Persian normalization ONLY to the source (never to English).
    """

    REQUIRED_COLS = {"persian_text", "english_translation", "audio_filename"}

    def __init__(
        self,
        csv_file: str,
        audio_dir: str,
        tokenizer,
        max_src_length: int,
        max_tgt_length: int,
        max_audio_length: int,
        normalize_medieval: bool = False,
        normalize_arabic: bool = True,
        normalize_arabic_numbers: bool = True,
        normalizer_name: str = "arabic_persian_basic",
        sample_rate: int = 16000,
        return_paths: bool = False,   # debug helper
    ):
        self.df = pd.read_csv(csv_file)
        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise KeyError(f"CSV is missing required columns: {sorted(missing)} "
                           f"(present: {list(self.df.columns)[:10]}...)")

        self.audio_dir = audio_dir
        self.tokenizer = tokenizer

        self.max_src_length = int(max_src_length)
        self.max_tgt_length = int(max_tgt_length)
        self.max_audio_length = int(max_audio_length)
        self.sample_rate = int(sample_rate)

        # Normalization policy (mirrors config.py flags)
        self.normalize_medieval = bool(normalize_medieval)
        self.normalize_arabic = bool(normalize_arabic)
        self.normalize_arabic_numbers = bool(normalize_arabic_numbers)
        self.normalizer_name = str(normalizer_name)

        self.return_paths = bool(return_paths)

        # Rank/worker one-time prints
        self._did_sanity = False

    def __len__(self):
        return len(self.df)

    def _normalize_src(self, s: str) -> str:
        return _basic_persian_normalize(
            s,
            normalize_arabic=self.normalize_arabic,
            normalize_arabic_numbers=self.normalize_arabic_numbers,
            use_medieval_poetry_rules=self.normalize_medieval,
            normalizer_name=self.normalizer_name,
        )

    def _rms_normalize(self, wav: torch.Tensor, target_rms: float = 0.03) -> torch.Tensor:
        """
        Gentle RMS normalization (post peak-normalization) to keep energy
        in a healthy band for W2V; avoids exploding quiet clips.
        target_rms ~0.02–0.05 works well.
        """
        wav = wav.to(torch.float32)
        rms = wav.pow(2).mean().sqrt()
        if (not torch.isfinite(rms)) or (rms < 1e-4):
            return wav
    
        gain = (target_rms / (rms + 1e-8)).clamp(0.5, 3.0)
        return wav * gain
#        if torch.isfinite(rms) and rms > 0:
#            gain = (target_rms / (rms + 1e-8)).clamp(0.5, 3.0)  # avoid extreme scaling
#            wav = wav * gain
#        return wav

    def _is_rank0_worker0(self):
        wi = get_worker_info()
        is_rank0 = os.environ.get("RANK", "0") == "0"
        is_worker0 = (wi is None) or (wi.id == 0)
        return is_rank0 and is_worker0

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        src_raw = row["persian_text"]
        tgt     = row["english_translation"]

        # Apply Persian normalization (English untouched)
        src = self._normalize_src(src_raw)

        # --- Text tokenization ---
        self.tokenizer.src_lang = "fa_IR"
        src_enc = self.tokenizer(
            src,
            truncation=True,
            padding="max_length",
            max_length=self.max_src_length,
            return_tensors="pt",
        )
        input_ids = src_enc.input_ids.squeeze(0).contiguous()
        attention_mask = src_enc.attention_mask.squeeze(0).contiguous()

        if attention_mask.sum().item() == 0:
            attention_mask[0] = 1  # ensure at least one attended token

        # Target (English) — do NOT normalize
        self.tokenizer.tgt_lang = "en_XX"
        with self.tokenizer.as_target_tokenizer():
            tgt_enc = self.tokenizer(
                tgt,
                truncation=True,
                padding="max_length",
                max_length=self.max_tgt_length,
                return_tensors="pt",
            )

        # Safety checks
        assert src_enc.input_ids.max() < self.tokenizer.vocab_size, \
            f"Source token ID too large: {src_enc.input_ids.max()} >= {self.tokenizer.vocab_size}"
        assert tgt_enc.input_ids.max() < self.tokenizer.vocab_size, \
            f"Target token ID too large: {tgt_enc.input_ids.max()} >= {self.tokenizer.vocab_size}"

        labels = tgt_enc.input_ids.squeeze(0).clone().contiguous()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = torch.clamp(labels, min=-100, max=self.tokenizer.vocab_size - 1)

        # --- One-time SANITY prints (rank0/worker0) ---
        if not self._did_sanity and idx == 0 and self._is_rank0_worker0():
            print(f"[SANITY] label range -> min: {labels.min().item()}, max: {labels.max().item()}")
            print(f"[SANITY] tokenizer.src_lang={self.tokenizer.src_lang} tokenizer.tgt_lang={self.tokenizer.tgt_lang}")
            try:
                print("[SANITY] first 12 source token IDs:", input_ids[:12].tolist())
                toks = self.tokenizer.convert_ids_to_tokens(input_ids[:12].tolist())
                print("[SANITY] first 12 source tokens  :", toks)
            except Exception:
                pass
            print(f"[SANITY] normalization -> medieval={self.normalize_medieval} "
                  f"arabic={self.normalize_arabic} arabic_numbers={self.normalize_arabic_numbers} "
                  f"name={self.normalizer_name}")
            self._did_sanity = True

        # ------------- Audio preprocessing -------------
        wav_path = os.path.join(self.audio_dir, row["audio_filename"])

        # Robust to missing/corrupt audio: return zeros but keep mask honest
        if not os.path.isfile(wav_path):
            wav = torch.zeros(self.max_audio_length, dtype=torch.float32)
            valid_T = 0
        else:
            try:
                wav, orig_sr = torchaudio.load(wav_path)
            except Exception:
                wav = torch.zeros(self.max_audio_length, dtype=torch.float32)
                orig_sr = self.sample_rate

            # --- Resample if needed (functional, no object creation) ---
            if wav.numel() > 0 and orig_sr != self.sample_rate:
                try:
                    wav = torchaudio.functional.resample(wav, orig_sr, self.sample_rate)
                except Exception:
                    # fall back silently; W2V can still accept wrong SR, but we try to avoid it
                    pass

            # --- Sanitize NaNs/Infs early ---
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Mixdown to mono and ensure float32 ---
            if wav.dim() == 2:
                if wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                else:
                    # already (1, T)
                    pass
            wav = wav.squeeze(0).to(torch.float32)

            # --- Peak-normalize, then gentle RMS-normalize ---
            max_abs = wav.abs().max()
            if torch.isfinite(max_abs) and max_abs > 1e-6:
                wav = wav / (max_abs + 1e-8)
            wav = self._rms_normalize(wav, target_rms=0.03)

            # --- Mean-center, clamp ---
            wav = wav - wav.mean()
            wav = wav.clamp(min=-1.0, max=1.0)

            # --- Trim / pad to fixed length ---
            T = wav.size(0)
            max_T = self.max_audio_length
            valid_T = min(T, max_T)

            if T > max_T:
                wav = wav[:max_T]
            elif T < max_T:
                wav = F.pad(wav, (0, max_T - T))  # zero-pad (no tiny noise)

            # --- Optional one-time audio stats (rank0/worker0) ---
            if idx == 0 and self._is_rank0_worker0():
                rms = float(wav.pow(2).mean().sqrt().item())
                print(f"[SANITY] audio stats -> mean={float(wav.mean().item()):.6f}, "
                      f"rms={rms:.6f}, max={float(wav.max().item()):.6f}, "
                      f"min={float(wav.min().item()):.6f}, valid_frames={valid_T}/{max_T}, "
                      f"sr={self.sample_rate}")

        # --- Build mask for valid audio region ---
        audio_mask = torch.zeros(self.max_audio_length, dtype=torch.long)
        audio_mask[:valid_T] = 1

        item = {
            "input_ids":      input_ids.contiguous(),        # (L_src,)
            "attention_mask": attention_mask.contiguous(),   # (L_src,)
            "labels":         labels.contiguous(),           # (L_tgt,)
            "audio":          wav.contiguous().float(),      # (L_audio,)
            "audio_mask":     audio_mask.contiguous(),       # (L_audio,)
            "tgt_text":       tgt,                           # raw English
            "src_text":       src,                           # normalized Persian
        }
        if self.return_paths:
            item["audio_path"] = wav_path
        return item


def collate_fn(batch):
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels         = torch.stack([b["labels"]         for b in batch])
    audio_batch    = torch.stack([b["audio"]          for b in batch])
    audio_mask     = torch.stack([b["audio_mask"]     for b in batch])
    tgt_texts      = [b["tgt_text"] for b in batch]
    src_texts      = [b["src_text"] for b in batch]

    if (os.environ.get("RANK", "0") == "0") and not hasattr(collate_fn, "_printed_audio_len"):
        try:
            checksum = float(audio_batch.abs().mean().item())
            print(f"[SANITY] collate_fn audio_len={audio_batch.shape[-1]} samples | checksum={checksum:.5f}")
        except Exception:
            pass
        collate_fn._printed_audio_len = True

    return {
        "input_ids":      input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "labels":         labels.contiguous(),
        "audio":          audio_batch.contiguous(),     # (B, L)
        "audio_mask":     audio_mask.contiguous(),      # (B, L)
        "tgt_texts":      tgt_texts,
        "src_texts":      src_texts,
    }


def get_dataloader(
    csv_file: str,
    audio_dir: str,
    tokenizer,
    batch_size: int,
    num_workers: int = 16,
    shuffle: bool = True,
    subset_size: int = None,
    max_src_length: int = 512,   # was 256
    max_tgt_length: int = 256,   # was 128
    max_audio_length: int = 160_000,
    normalize_medieval: bool = False,
    normalize_arabic: bool = True,
    normalize_arabic_numbers: bool = True,
    normalizer_name: str = "arabic_persian_basic",
    sample_rate: int = 16000,
    return_paths: bool = False,
    drop_last: bool = None,             # NEW: auto if None (train=True, eval=False)
    pin_memory: bool = None,            # NEW: auto if None
    persistent_workers: bool = None,    # NEW: auto if None
):
    """
    NOTE: Pass the normalization flags from your config here to keep train/dev/test consistent:
      ds = get_dataloader(...,
            normalize_medieval=cfg.normalize_medieval,
            normalize_arabic=cfg.normalize_arabic,
            normalize_arabic_numbers=cfg.normalize_arabic_numbers,
            normalizer_name=cfg.normalizer_name,
            sample_rate=cfg.sample_rate)
    """
    ds = MultimodalTranslationDataset(
        csv_file=csv_file,
        audio_dir=audio_dir,
        tokenizer=tokenizer,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
        max_audio_length=max_audio_length,
        normalize_medieval=normalize_medieval,
        normalize_arabic=normalize_arabic,
        normalize_arabic_numbers=normalize_arabic_numbers,
        normalizer_name=normalizer_name,
        sample_rate=sample_rate,
        return_paths=return_paths,
    )

    if subset_size is not None:
        ds = Subset(ds, list(range(min(subset_size, len(ds)))))

    # Sensible defaults:
    # - drop_last: True for training (shuffle=True), False for eval (shuffle=False)
    if drop_last is None:
        drop_last = bool(shuffle)
    # - pin_memory: True when CUDA available
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    # - persistent_workers: True when using multiple workers
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )