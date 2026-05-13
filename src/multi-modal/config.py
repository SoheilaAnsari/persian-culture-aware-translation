"""
Configuration for the multimodal translation training pipeline.

Reference: SilkRoadNLP 2026 paper §5.2.

Hardware-environment notes:
  - All paths are configurable via environment variables OR keyword arguments
    to get_config(). NO hardcoded paths.
  - Required env vars (or pass via kwargs):
      DATA_DIR        — root containing {split}/{split}.csv and {split}/audio/
      TEXT_BASELINE_HF — HuggingFace dir of the PPFT (text baseline) checkpoint
      TEXT_BASELINE_PT — PyTorch .pt of the PPFT checkpoint (legacy compat)
      TRANS_RUN_DIR   — where to write run outputs (default: ./runs/multimodal)

Hardware used in the paper: 4× NVIDIA A40 (48 GB), DDP, mixed precision.
"""

import os
import torch
from datetime import datetime
from transformers import MBart50TokenizerFast


# Map friendly names to folder splits
_SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "dev": "val", "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test", "eval": "test", "evaluation": "test",
}


def _resolve_split(name_or_split: str) -> str:
    n = str(name_or_split).strip().lower()
    return _SPLIT_ALIASES.get(n, n)


class MultimodalConfig:
    """Configuration for multimodal translation training (text + W2V cross-attention)."""

    def __init__(
        self,
        split="train",
        # Data root (env: DATA_DIR). Must contain {split}/{split}.csv and {split}/audio/
        data_dir=None,
        # Output root (env: TRANS_RUN_DIR)
        run_dir_root=None,
        # PPFT (text baseline) checkpoint paths (env: TEXT_BASELINE_HF / TEXT_BASELINE_PT)
        text_baseline_hf=None,
        text_baseline_pt=None,
        # Normalization
        normalize_arabic=True,
        normalize_arabic_numbers=True,
        normalize_medieval=False,
        normalizer_name="arabic_persian_basic",
        # Decoding
        num_beams=6,
        max_new_tokens=200,
        min_new_tokens=3,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
        length_penalty=1.1,
        # Ablations
        ablations_epochs="2,4,6,8,10",
        ablations_max_eval_batches=256,
        # Training hyperparams
        lr=2e-4,
        weight_decay=0.01,
        warmup_steps=500,
        grad_accum=2,
        epochs=10,
        batch_size=None,
        clip=1.0,
        num_workers=8,
        # Model
        mbart_model="facebook/mbart-large-50",
        wav2vec_model="jonatasgrosman/wav2vec2-large-xlsr-53-persian",
        audio_latents=32,
        audio_scale_init=1.5,
        attn_heads=8,
        modality_dropout=0.10,
        use_prosody=False,
        prosody_scale=0.25,
        # Sequence lengths
        max_src_length=512,
        max_tgt_length=256,
        max_seq_len=512,
        max_audio_length=160000,
        sample_rate=16000,
        # Eval settings
        max_eval_batches=None,
        # Tokenizer
        tokenizer_dir="",
        src_lang="fa_IR",
        tgt_lang="en_XX",
        **kwargs
    ):
        # Split aliasing
        self.orig_split = split
        self.split = _resolve_split(split)

        # Stage-2 (direct W2V cross-attention)
        self.variant = "text+w2v_xattn"

        # Core flags
        self.normalize_arabic = bool(normalize_arabic)
        self.normalize_arabic_numbers = bool(normalize_arabic_numbers)
        self.normalize_medieval = bool(normalize_medieval)
        self.normalizer_name = str(normalizer_name)

        # Decode profile
        self.decode_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(min_new_tokens),
            num_beams=int(num_beams),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            repetition_penalty=float(repetition_penalty),
            length_penalty=float(length_penalty),
            early_stopping=True,
            use_cache=True,
        )
        self.gen_defaults = dict(self.decode_kwargs)

        # Ablations
        self.ablations_epochs = str(ablations_epochs)
        self.ablations_max_eval_batches = int(ablations_max_eval_batches)

        # ---------------------------------------------------------------
        # Paths — all configurable, no hardcoded user paths
        # ---------------------------------------------------------------

        # Data root: kwarg > env > error
        base_data = data_dir or os.getenv("DATA_DIR")
        if not base_data:
            raise RuntimeError(
                "DATA_DIR is required. Pass data_dir=... or set the DATA_DIR "
                "environment variable. Expected layout: "
                "<DATA_DIR>/{train,val,test}/{train,val,test}.csv "
                "and <DATA_DIR>/{train,val,test}/audio/*.wav"
            )

        self.csv_file = os.path.join(base_data, self.split, f"{self.split}.csv")
        self.audio_dir = os.path.join(base_data, self.split, "audio")

        if not os.path.isfile(self.csv_file):
            raise FileNotFoundError(f"CSV not found: {self.csv_file}")
        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        # Output run dir
        trans_root = run_dir_root or os.getenv("TRANS_RUN_DIR", "./runs/multimodal")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(os.path.abspath(trans_root), self.variant, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # PPFT baselines (kwarg > env > None)
        self.text_baseline_hf = text_baseline_hf or os.getenv("TEXT_BASELINE_HF")
        self.text_baseline_pt = text_baseline_pt or os.getenv("TEXT_BASELINE_PT")

        # Legacy compat alias
        self.encoder_checkpoint = self.text_baseline_pt

        # ---------------------------------------------------------------
        # Model hyperparams
        # ---------------------------------------------------------------
        self.mbart_model = str(mbart_model)
        self.wav2vec_model = str(wav2vec_model)
        self.audio_latents = int(audio_latents)
        self.attn_heads = int(attn_heads)
        self.modality_dropout = float(modality_dropout)
        self.use_prosody = bool(use_prosody)
        self.prosody_scale = float(prosody_scale)

        # Training hparams
        self.learning_rate = float(lr)
        self.lr = self.learning_rate
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.grad_accum = int(grad_accum)
        self.n_epoch = int(epochs)
        self.epochs = self.n_epoch
        self.clip = float(clip)
        self.num_workers = int(num_workers)

        # Batch size with split-aware defaults
        if batch_size is None:
            self.batch_size = 8 if self.split == "train" else 16
        else:
            self.batch_size = int(batch_size)

        # Sequence lengths
        self.max_src_length = int(max_src_length)
        self.max_tgt_length = int(max_tgt_length)
        self.max_seq_len = int(max_seq_len)
        self.max_audio_length = int(max_audio_length)
        self.sample_rate = int(sample_rate)

        # Eval settings
        if max_eval_batches is None:
            self.max_eval_batches = 100 if self.split == "train" else None
        else:
            self.max_eval_batches = int(max_eval_batches)

        # ---------------------------------------------------------------
        # Tokenizer
        # ---------------------------------------------------------------
        tokenizer_dir = str(tokenizer_dir).strip() or os.getenv("TEXT_BASELINE_TOKENIZER", "").strip()

        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_dir)
            tokenizer_source = tokenizer_dir
        else:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(self.mbart_model)
            tokenizer_source = self.mbart_model

        # Language codes
        self.src_lang = str(src_lang)
        self.tgt_lang = str(tgt_lang)
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        # Expose IDs
        self.lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", {})
        self.src_lang_id = self.lang_code_to_id.get(self.src_lang)
        self.tgt_lang_id = self.lang_code_to_id.get(self.tgt_lang)

        # Decoder start token — CRITICAL: force English BOS
        if self.tgt_lang_id is not None:
            self.forced_bos_token_id = self.tgt_lang_id
            self.decoder_start_token_id = self.tgt_lang_id
        else:
            print("[ERROR] Target language ID not found in tokenizer!")
            self.forced_bos_token_id = getattr(self.tokenizer, "pad_token_id", 1)
            self.decoder_start_token_id = self.forced_bos_token_id

        self.gen_forced_bos_id = self.forced_bos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        # Arabic bad words filter (light scan)
        def _arabic_badwords(tokenizer, max_scan=30000):
            bad = []
            vocab_scan = min(tokenizer.vocab_size, max_scan)
            for tid in range(vocab_scan):
                try:
                    tok = tokenizer.convert_ids_to_tokens(tid)
                    if any('\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' for ch in tok):
                        bad.append([tid])
                except Exception:
                    pass
            return bad

        self.bad_words_ids_ar = kwargs.get("bad_words_ids_ar", _arabic_badwords(self.tokenizer))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sanity banner
        print(f"\n{'='*80}")
        print(f"MULTIMODAL CONFIG — {self.split.upper()}")
        print(f"{'='*80}")
        print(f"[CFG] split={self.split} | variant={self.variant}")
        print(f"[CFG] src={self.src_lang}({self.src_lang_id}) → tgt={self.tgt_lang}({self.tgt_lang_id})"
              f" | decoder_start={self.decoder_start_token_id}")
        if self.decoder_start_token_id == self.tgt_lang_id:
            print(f"[CFG] ✓ decoder_start_token_id correctly set to target language")
        print(f"[CFG] normalize: arabic={self.normalize_arabic} digits={self.normalize_arabic_numbers}")
        print(f"[CFG] decode: {self.decode_profile_str()}")
        print(f"[CFG] tokenizer from: {tokenizer_source}")
        print(f"[PATH] run_dir={self.run_dir}")
        print(f"[PATH] baseline_hf={self.text_baseline_hf}")
        print(f"[PATH] baseline_pt={self.text_baseline_pt}")
        print(f"[TRAIN] epochs={self.n_epoch} | batch={self.batch_size} | lr={self.learning_rate}"
              f" | warmup={self.warmup_steps} | clip={self.clip}")
        print(f"[GEN] forced_bos={self.forced_bos_token_id} | {self.decode_profile_str()}")
        print(f"{'='*80}\n")

    def is_stage2(self) -> bool:
        return True

    def variant_tag(self) -> str:
        return self.variant

    def decode_profile_str(self) -> str:
        dk = self.decode_kwargs
        return f"beams={dk['num_beams']}, lenp={dk['length_penalty']}, max_new={dk['max_new_tokens']}"

    def __repr__(self):
        return f"MultimodalConfig(split={self.split}, variant={self.variant})"


def get_config(split="train", **kwargs):
    """Main entry point — returns MultimodalConfig with split aliasing."""
    split = str(split).strip().lower()
    actual_split = _SPLIT_ALIASES.get(split, split)
    return MultimodalConfig(split=actual_split, **kwargs)
