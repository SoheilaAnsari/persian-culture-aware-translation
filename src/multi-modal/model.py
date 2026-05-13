"""
Multimodal Translation Model — PPFT + Wav2Vec2 with cross-attention fusion.

Reference: SilkRoadNLP 2026 paper §5.2 (multi-modal extension).

Architecture (V8, Stage-2 direct W2V cross-attention):
  - Text encoder: mBART-50 (frozen, initialized from PPFT checkpoint)
  - Audio encoder: Wav2Vec2-XLS-R 53 Persian (frozen)
  - Fusion: single cross-attention block (text queries audio) + learnable
            sigmoid gate α ∈ [0, 1] producing fused = (1-α)·text + α·text_ctx
  - Decoder: mBART decoder + lm_head (trainable)
  - Generation lock: forced_bos_token_id = en_XX (always English output)

Reproduces paper Table 2 row 2 (BLEU 17.95 / chrF++ 42.95 / BERTScore 0.894 /
COMET 0.635) and Table 3 multi-modal rows (CSI-Recall 82.04% / 89.04% / 75.73%).

Hardware used in the paper: 4× NVIDIA A40 (48 GB), DDP, mixed precision.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

SAMPLE_RATE = 16_000
W2V_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
MBART_NAME = "facebook/mbart-large-50"


def _init_linear(m: nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_ln(m: nn.LayerNorm):
    nn.init.ones_(m.weight)
    nn.init.zeros_(m.bias)


class ModalityDropout(nn.Module):
    """Drops an entire modality with prob p during training (per batch)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, text_tokens, audio_tokens, text_mask=None, audio_mask=None):
        if not self.training or self.p <= 0.0:
            return text_tokens, audio_tokens, text_mask, audio_mask
        device = text_tokens.device
        drop_text  = torch.rand(1, device=device).item() < self.p
        drop_audio = torch.rand(1, device=device).item() < self.p
        if drop_text and not drop_audio:
            text_tokens = torch.zeros_like(text_tokens)
        elif drop_audio and not drop_text:
            audio_tokens = torch.zeros_like(audio_tokens)
            if audio_mask is not None:
                audio_mask = torch.zeros_like(audio_mask, dtype=torch.bool)
        return text_tokens, audio_tokens, text_mask, audio_mask


class TextToAudioCrossAttention(nn.Module):
    """
    Minimal single cross-attention block:
      - Text queries audio: fused = (1-α)·text + α·text_ctx
      - Sigmoid gate α per token (exposes `last_mean_alpha` for diagnostics)
      - Learnable `_audio_scale_raw` for trainer compatibility
    """
    def __init__(self, d_model=1024, num_heads=8, attn_drop=0.1, resid_drop=0.1,
                 gate_bias_init: float = 0.0, audio_scale_init: float = 1.0):
        super().__init__()
        self.t2a = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=attn_drop)
        self.ln_q_text = nn.LayerNorm(d_model)
        self.ln_kv_audio = nn.LayerNorm(d_model)
        self.ln_fuse1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(resid_drop),
            nn.Linear(4 * d_model, d_model), nn.Dropout(resid_drop)
        )
        # simple gate: α = σ(MLP([text, text_ctx]))
        self.ln_gate_t = nn.LayerNorm(d_model)
        self.ln_gate_c = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.constant_(self.gate[-1].bias, float(gate_bias_init))

        # trainer expects an audio scale param (softplus)
        self.audio_scale_floor = 1.0
        target = float(max(1.0 + 1e-6, audio_scale_init))
        y = target - self.audio_scale_floor
        raw = math.log(math.expm1(y)) if y > 1e-6 else -20.0
        self._audio_scale_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

        self.warm_force_alpha = None

    def forward(self, text_tokens, text_mask, audio_tokens, audio_mask):
        if text_mask is not None and text_mask.dtype != torch.bool:
            text_mask = text_mask != 0
        if audio_mask is not None and audio_mask.dtype != torch.bool:
            audio_mask = audio_mask != 0

        # scale audio features softly
        a_scale = F.softplus(self._audio_scale_raw) + self.audio_scale_floor
        audio_tokens = audio_tokens * a_scale

        t_q = self.ln_q_text(text_tokens)
        a_kv = self.ln_kv_audio(audio_tokens)
        kpm_audio = (~audio_mask) if audio_mask is not None else None
        if kpm_audio is not None and kpm_audio.all(dim=1).any():
            bad = kpm_audio.all(dim=1).nonzero(as_tuple=False).view(-1)
            kpm_audio[bad, 0] = False

        # text ← audio
        t2a_out, attn = self.t2a(query=t_q, key=a_kv, value=a_kv,
                                 key_padding_mask=kpm_audio, need_weights=False)
        text_ctx = self.ln_fuse1(text_tokens + t2a_out)
        text_ctx = self.ffn(text_ctx)

        # gate
        t_n = self.ln_gate_t(text_tokens)
        c_n = self.ln_gate_c(text_ctx)
        alpha = torch.sigmoid(self.gate(torch.cat([t_n, c_n], dim=-1)))
        if self.warm_force_alpha is not None and self.training:
            alpha = torch.full_like(alpha, self.warm_force_alpha)

        fused = (1.0 - alpha) * text_tokens + alpha * text_ctx

        self.last_mean_alpha = alpha.mean().detach() if self.training else alpha.mean().detach()
        return fused


class MultimodalTranslationModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        decoder_lang_id=None,
        *,
        normalize_medieval=False,
        normalize_arabic=True,
        normalize_arabic_numbers=True,
        normalizer_name="arabic_persian_basic",
        encoder_ckpt_path="",
        hf_baseline_dir="",
        mbart_model_name=MBART_NAME,
        num_audio_latents=0,
        num_heads=8,
        modality_dropout_mode="zero",
        audio_scale_init=1.5,
        gate_bias_init=0.0,
        modality_dropout_p=0.0,
        use_prosody=False,
        prosody_scale=0.25,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        self.normalize_medieval = bool(normalize_medieval)
        self.normalize_arabic = bool(normalize_arabic)
        self.normalize_arabic_numbers = bool(normalize_arabic_numbers)
        self.normalizer_name = str(normalizer_name)
        self.use_prosody = bool(use_prosody)
        self.prosody_scale = float(prosody_scale)

        # MBART (text) + W2V (audio)
        self.mbart = MBartForConditionalGeneration.from_pretrained(mbart_model_name)
        self.text_encoder = self.mbart.model.encoder
        self.text_decoder = self.mbart.model.decoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained(W2V_NAME)

        # language setup
        self.tokenizer.src_lang = "fa_IR"
        self.tokenizer.tgt_lang = "en_XX"
        lang2id = getattr(self.tokenizer, "lang_code_to_id", {})
        en_id = lang2id.get("en_XX", None)
        self.decoder_lang_id = en_id if decoder_lang_id is None else decoder_lang_id

        if self.decoder_lang_id is not None:
            self.mbart.config.forced_bos_token_id = int(self.decoder_lang_id)
            self.mbart.config.decoder_start_token_id = int(self.decoder_lang_id)

        if self.mbart.config.pad_token_id is None:
            self.mbart.config.pad_token_id = self.pad_token_id

        # Freeze encoders (Stage-2), train decoder + fusion
        self.text_encoder.requires_grad_(False); self.text_encoder.eval()
        self.audio_encoder.requires_grad_(False); self.audio_encoder.eval()
        self.audio_encoder.config.output_hidden_states = False

        # dims
        d_model = self.text_encoder.config.d_model
        a_hidden = self.audio_encoder.config.hidden_size

        # projections
        self.audio_proj = nn.Linear(a_hidden, d_model)
        self.audio_proj_ln = nn.LayerNorm(d_model)
        _init_linear(self.audio_proj); _init_ln(self.audio_proj_ln)

        # simple cross-attention fusion (text ← audio)
        self.dual_xattn = TextToAudioCrossAttention(d_model=d_model, num_heads=num_heads,
                                                    gate_bias_init=gate_bias_init,
                                                    audio_scale_init=audio_scale_init)

        # light output proj
        self.fusion_out_proj = nn.Linear(d_model, d_model)
        _init_linear(self.fusion_out_proj)

        # modality dropout (disabled by default)
        self.mod_dropout = ModalityDropout(p=modality_dropout_p)

        # learnable global audio gain
        self._audio_gain_raw = nn.Parameter(torch.tensor(0.0))
        self.audio_gain_floor = 1.0

        # train decoder + lm head
        for p in self.text_decoder.parameters():
            p.requires_grad = True
        for p in self.mbart.lm_head.parameters():
            p.requires_grad = True

        # Increase decoder dropout to encourage fusion usage
        try:
            if hasattr(self.mbart, "config"):
                self.mbart.config.dropout = 0.15
                self.mbart.config.activation_dropout = 0.15
            if hasattr(self.text_decoder, "layers"):
                for layer in self.text_decoder.layers:
                    if hasattr(layer, "dropout"):
                        layer.dropout = 0.15
                    if hasattr(layer, "activation_dropout"):
                        layer.activation_dropout = 0.15
        except Exception as e:
            print(f"[WARN] Decoder dropout adjustment failed: {e}")

    def _build_text_mask(self, attention_mask):
        if attention_mask is None:
            return None
        m = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > 0)
        if not m.any(dim=1).all():
            bad = (~m.any(dim=1)).nonzero(as_tuple=False).view(-1)
            m[bad, 0] = True
        return m

    def _build_audio_mask(self, audio_waveform, audio_mask, feat_len):
        """Return a [B, feat_len] boolean mask where True = valid frame."""
        B = audio_waveform.size(0)

        if audio_mask is not None:
            if audio_mask.dtype != torch.bool:
                audio_mask = (audio_mask != 0)
            if audio_mask.dim() == 2 and audio_mask.size(1) == feat_len:
                m = audio_mask
            elif audio_mask.dim() == 2:
                L = audio_mask.size(1)
                ratio = max(1, (L + feat_len - 1) // feat_len)
                pad = ratio * feat_len - L
                if pad > 0:
                    audio_mask = F.pad(audio_mask, (0, pad), value=False)
                m = audio_mask.view(B, feat_len, ratio).max(dim=-1).values
            else:
                m = (audio_waveform.abs() > 0)
        else:
            m = (audio_waveform.abs() > 0)

        if m.dim() == 2 and m.size(1) != feat_len:
            L = m.size(1)
            ratio = max(1, (L + feat_len - 1) // feat_len)
            pad = ratio * feat_len - L
            if pad > 0:
                m = F.pad(m, (0, pad), value=False)
            m = m.view(B, feat_len, ratio).max(dim=-1).values

        all_off = (~m.any(dim=1))
        if all_off.any():
            m[all_off, 0] = True
        return m

    def _encode_and_fuse(self, input_ids, attention_mask, audio, audio_mask):
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)

        # text
        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_mask = self._build_text_mask(attention_mask)

        # audio
        audio = audio.to(dtype=torch.float32)
        a_feats = self.audio_encoder(audio).last_hidden_state  # [B, La, Ha]
        B, La, _ = a_feats.shape
        a_feats = self.audio_proj_ln(self.audio_proj(a_feats))

        # soft global gain
        a_gain = F.softplus(self._audio_gain_raw) + self.audio_gain_floor
        a_feats = a_feats * a_gain

        # audio mask at frame-level
        a_mask = self._build_audio_mask(audio, audio_mask, feat_len=La)

        # optional modality dropout
        txt, a_feats, text_mask, a_mask = self.mod_dropout(txt, a_feats, text_mask, a_mask)

        # single cross-att fusion (text ← audio)
        fused = self.dual_xattn(txt, text_mask, a_feats, a_mask)
        fused = self.fusion_out_proj(fused)
        return fused, text_mask

    def forward(self, input_ids, attention_mask, audio, audio_mask, labels=None):
        """Training forward pass."""
        fused_hidden, enc_attn_mask = self._encode_and_fuse(input_ids, attention_mask, audio, audio_mask)
        if labels is not None:
            decoder_input_ids = self._prep_decoder_inputs(labels)
        else:
            decoder_input_ids = None

        out = self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
            attention_mask=enc_attn_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        return {"logits": out.logits, "loss": out.loss}

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, audio=None, audio_mask=None, **gen_kwargs):
        """Inference generation with beam search defaults."""
        fused_hidden, enc_attn_mask = self._encode_and_fuse(input_ids, attention_mask, audio, audio_mask)

        en_bos = int(self.decoder_lang_id) if self.decoder_lang_id is not None else None
        if en_bos is not None:
            gen_kwargs.setdefault("forced_bos_token_id", int(en_bos))
        gen_kwargs.setdefault("decoder_start_token_id", en_bos)
        gen_kwargs.setdefault("max_new_tokens", 200)
        gen_kwargs.setdefault("num_beams", 6)
        gen_kwargs.setdefault("length_penalty", 1.1)
        gen_kwargs.setdefault("repetition_penalty", 1.05)
        gen_kwargs.setdefault("no_repeat_ngram_size", 3)
        gen_kwargs.setdefault("early_stopping", True)
        gen_kwargs.setdefault("use_cache", True)

        return self.mbart.generate(
            inputs=None,
            encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
            attention_mask=enc_attn_mask,
            **gen_kwargs,
        )

    def _prep_decoder_inputs(self, labels):
        if labels is None:
            return None
        B, T = labels.shape
        device = labels.device
        pad = self.pad_token_id
        bos = getattr(self.mbart.config, "forced_bos_token_id", pad)
        dec_in = torch.full((B, T), pad, dtype=torch.long, device=device)
        dec_in[:, 0] = bos
        if T > 1:
            dec_in[:, 1:] = labels[:, :-1].clone()
        dec_in[dec_in == -100] = pad
        return dec_in
