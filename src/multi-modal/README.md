<h1 align="center">Multi-modal Neural Machine Translation for Low-Resource Classical Persian Poetry: A Culture-Aware Evaluation</h1>

<p align="center">
  <a href="https://aclanthology.org/2026.silkroadnlp-1.14/"><img src="https://img.shields.io/badge/ACL-Anthology-red.svg" alt="ACL Anthology"></a>
  <a href="https://doi.org/10.18653/v1/2026.silkroadnlp-1.14"><img src="https://img.shields.io/badge/DOI-10.18653%2Fv1%2F2026.silkroadnlp--1.14-blue.svg" alt="DOI"></a>
  <a href="https://github.com/SoheilaAnsari/persian-culture-aware-translation/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10"></a>
</p>

<p align="center">
  <strong>Soheila Ansari</strong>, <strong>Mounir Boukadoum</strong>, <strong>Fatiha Sadat</strong><br>
  Université du Québec à Montréal<br>
  <em>Proceedings of the First Workshop on NLP and LLMs for the Iranian Language Family (SilkRoadNLP @ ACL 2026)</em><br>
  Rabat, Morocco · March 2026 · pp. 131–139
</p>

---

## Overview

Code companion for the SilkRoadNLP 2026 paper on translating Rumi's *Masnavi-ye-Ma'navi* with a culture-aware evaluation methodology.

This repository releases three of the four components described in the paper:

1. **PPFT** — a text-only Persian→English translation baseline (domain-adaptive pre-training + fine-tuning of mBART-50).
2. **PPFT + Wav2Vec2** — the multi-modal extension with cross-attention fusion of text and audio recitations.
3. **CSI evaluation** — a culture-aware evaluation framework with a Persian–English CSI tagger, lexicon, and the CSI-Recall metric.

Refer to the paper for full methodology, results, related work, and discussion.

---

## Repository layout

```
.
├── README.md
├── LICENSE                MIT (code) + CC-BY 4.0 (data artifacts)
├── CITATION.cff           Citation metadata
├── environment.yml        conda environment
├── .gitignore
│
├── src/
│   ├── text_baseline/                    Pillar 1: PPFT
│   │   ├── domain_pretrain.py            Stage 1: span-masked DAP on Persian poetry
│   │   ├── finetune.py                   Stage 2: parallel fine-tuning (Masnavi)
│   │   └── evaluate.py                   BLEU + chrF++ + BERTScore + COMET
│   │
│   ├── multimodal/                       Pillar 2: PPFT + Wav2Vec2
│   │   ├── config.py                     Multimodal training configuration
│   │   ├── data_loader.py                Loads aligned text + audio
│   │   ├── model.py                      Cross-attention fusion architecture
│   │   └── train.py                      DDP training loop
│   │
│   └── csi_evaluation/                   Pillar 3: Culture-Specific Item evaluation
│       ├── prepare_bio_dataset.py        Step 1: gold spans → BIO subword data
│       ├── train_tagger.py               Step 2: train CSI tagger
│       ├── annotate_corpus.py            Step 3: tag full parallel corpus
│       ├── prepare_alignment_input.py    Step 4: prepare fa↔en for awesome-align
│       ├── extract_lexicon.py            Step 5: build the 3-tier CSI lexicon
│       └── csi_recall_metric.py          Step 6: compute CSI-Recall (user-facing)
│
└── data/
    └── README.md                         Data schema and provenance
```

---

## Installation

```bash
git clone https://github.com/SoheilaAnsari/persian-culture-aware-translation
cd persian-culture-aware-translation
conda env create -f environment.yml
conda activate persian-mt
```

For the lexicon construction step, install [awesome-align](https://github.com/neulab/awesome-align) separately:

```bash
git clone https://github.com/neulab/awesome-align.git
cd awesome-align && pip install -r requirements.txt && python setup.py install
```

---

## Quick start — compute CSI-Recall on your own translations

The CSI-Recall metric is the main user-facing artifact. If you have a Persian→English translation system and you want to measure how well it preserves Culture-Specific Items on the *Masnavi* test set:

```bash
python -m src.csi_evaluation.csi_recall_metric \
    --hypotheses path/to/your_predictions.txt \
    --test-csv data/test_with_csi.csv \
    --lexicon-top1 data/lexicon/top1.canon.tsv \
    --lexicon-top3 data/lexicon/top3.canon.tsv \
    --lexicon-broad data/lexicon/broad.canon.tsv \
    --output runs/ \
    --tag my_system
```

The hypothesis file is one English translation per line, in the same order as the test CSV. The script reports CSI-Recall across three lexicon variants (strict_core, soft_core, broad). See paper §6 for the methodology and Table 3 for expected values on PPFT and PPFT + Wav2Vec2. CPU is sufficient.

---

## Reproducing the PPFT text-only baseline

**Hardware used in the paper:** 4× NVIDIA A40 (48 GB VRAM each), driver 575.57.08, PyTorch DDP via `torchrun`, mixed precision (AMP).

```bash
# Stage 1 — Domain-Adaptive Pre-training (span-masked denoising on Persian poetry)
torchrun --nproc_per_node=4 src/text_baseline/domain_pretrain.py \
    --poetry-corpus path/to/persian_poetry.txt \
    --output-dir runs/dap

# Stage 2 — Fine-tuning on the parallel Masnavi corpus
torchrun --nproc_per_node=4 src/text_baseline/finetune.py \
    --train-csv data/masnavi_corpus/train.csv \
    --val-csv data/masnavi_corpus/val.csv \
    --test-csv data/masnavi_corpus/test.csv \
    --dap-checkpoint runs/dap/checkpoints/best_model.pt \
    --output-dir runs/ppft

# Evaluation (BLEU + chrF++ + BERTScore + COMET)
python src/text_baseline/evaluate.py \
    --test-csv data/masnavi_corpus/test.csv \
    --checkpoint runs/ppft/checkpoints/best_hf \
    --output runs/ppft_eval_report.json
```

The Persian poetry corpus used for Stage 1 (1M lines, Rumi removed) is publicly available: [amnghd/Persian_poems_corpus](https://github.com/amnghd/Persian_poems_corpus/tree/master).

Single-GPU runs work with `--nproc_per_node=1`. CPU is not feasible for training (it is fine for evaluation).

---

## Reproducing the multi-modal extension (PPFT + Wav2Vec2)

The multi-modal model adds a cross-attention fusion layer between the PPFT text encoder and a frozen Wav2Vec2-XLS-R Persian audio encoder. It reproduces the paper's headline numbers: BLEU 17.95 / chrF++ 42.95 / BERTScore 0.894 / COMET 0.635 (Table 2) and CSI-Recall 82.04% / 89.04% / 75.73% (Table 3).

**Prerequisites:**

1. **A trained PPFT checkpoint** (from `src/text_baseline/finetune.py`). The multi-modal training initializes its text encoder from this.
2. **Audio recordings.** Obtain from [masnavi.net](https://masnavi.net/); resample to 16 kHz mono WAV; align filenames with the `audio_filename` column in the CSVs. Audio is NOT redistributed in this repository.
3. **Data layout** under `<DATA_DIR>`:
   ```
   <DATA_DIR>/train/train.csv
   <DATA_DIR>/train/audio/*.wav
   <DATA_DIR>/val/val.csv
   <DATA_DIR>/val/audio/*.wav
   <DATA_DIR>/test/test.csv
   <DATA_DIR>/test/audio/*.wav
   ```

**Run:**

```bash
# Required environment variables (or pass via --text-baseline-* flags)
export DATA_DIR=path/to/masnavi_data
export TEXT_BASELINE_HF=runs/ppft/checkpoints/best_hf
export TEXT_BASELINE_PT=runs/ppft/checkpoints/best_finetuned.pt
export TRANS_RUN_DIR=runs/multimodal

# Train
torchrun --nproc_per_node=4 src/multimodal/train.py \
    --epochs 10 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 2e-4 \
    --warmup_steps 500
```

**Architecture summary** (paper §5.2):
- Text encoder: mBART-50 (frozen, initialized from PPFT)
- Audio encoder: Wav2Vec2-XLS-R 53 Persian (frozen)
- Fusion: single cross-attention block (text queries audio) with a learnable sigmoid gate α ∈ [0,1]
- Decoder: mBART decoder + lm_head (trainable)

---

## Reproducing the CSI evaluation pipeline

Run these in order. They produce the artifacts used by `csi_recall_metric.py`.

```bash
# Step 1 — Convert gold span-based annotations to BIO format
python src/csi_evaluation/prepare_bio_dataset.py \
    --annotated-jsonl data/annotated_csi_filtered.jsonl \
    --output-dir data/ \
    --ppft-model-dir runs/ppft/checkpoints/best_hf

# Step 2 — Train the CSI tagger
python src/csi_evaluation/train_tagger.py \
    --data-dir data/ \
    --ppft-model-dir runs/ppft/checkpoints/best_hf \
    --output-dir runs/csi_tagger

# Step 3 — Apply the tagger to the full parallel corpus
python src/csi_evaluation/annotate_corpus.py \
    --parallel-root data/masnavi_corpus \
    --output-root data/csi_tagged \
    --csi-model-dir runs/csi_tagger

# Step 4 — Prepare for awesome-align (forward + reverse)
python src/csi_evaluation/prepare_alignment_input.py \
    --train-csv data/csi_tagged/train_with_csi.csv \
    --val-csv data/csi_tagged/val_with_csi.csv \
    --output-dir data/aa/ --direction forward
python src/csi_evaluation/prepare_alignment_input.py \
    --train-csv data/csi_tagged/train_with_csi.csv \
    --val-csv data/csi_tagged/val_with_csi.csv \
    --output-dir data/aa/ --direction reverse

# Run awesome-align externally on data/aa/trainval.parallel.txt
# and data/aa/trainval.parallel.rev.txt

# Step 5 — Build the CSI lexicon
python src/csi_evaluation/extract_lexicon.py \
    --aa-data-dir data/aa/ \
    --train-csv data/csi_tagged/train_with_csi.csv \
    --val-csv data/csi_tagged/val_with_csi.csv \
    --output-dir data/lexicon/

# Step 6 — Compute CSI-Recall (see "Quick start" above)
```

---

## Citation

If you use this code or the CSI methodology, please cite the paper:

```bibtex
@inproceedings{ansari-etal-2026-multi,
    title = "Multi-modal Neural Machine Translation for Low-Resource Classical {P}ersian Poetry: A Culture-Aware Evaluation",
    author = "Ansari, Soheila  and
      Boukadoum, Mounir  and
      Sadat, Fatiha",
    editor = "Merchant, Rayyan  and
      Megerdoomian, Karine",
    booktitle = "The Proceedings of the First Workshop on {NLP} and {LLM}s for the {I}ranian Language Family",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.silkroadnlp-1.14/",
    doi = "10.18653/v1/2026.silkroadnlp-1.14",
    pages = "131--139",
    ISBN = "979-8-89176-371-5"
}
```

The `CITATION.cff` file is provided so GitHub renders a "Cite this repository" widget in the sidebar.

---

## Scope and limitations

**This repository releases** the PPFT text-only baseline, the multi-modal extension, the CSI evaluation framework, and the parallel Masnavi corpus (train/val/test splits with English translations).

**This repository does NOT release** audio recordings (obtained from [masnavi.net](https://masnavi.net/) for non-commercial academic research; please obtain directly from that source) or pre-extracted Wav2Vec2 features.

The full thesis (Ansari, 2026, UQAM) describes additional analyses including SOTA comparisons, human evaluation, and an alternative modality alignment phase, which are outside the scope of this code release.

---

## License

- **Source code:** MIT License (see [`LICENSE`](LICENSE)).
- **Data artifacts** (CSI taxonomy, gold annotations, BIO datasets, lexicons, test set with CSI tags, parallel corpus): CC-BY 4.0.

---

## Acknowledgements

- Persian text and audio recordings: [masnavi.net](https://masnavi.net/); reciters Hosayn Āhī (books 1, 6) and Amīr Nūrī (books 2–5); sponsored by Noorsoft.
- English translation: Reynold A. Nicholson (1925–1940), public domain since 2015.
- Persian poetry corpus for DAP: [amnghd/Persian_poems_corpus](https://github.com/amnghd/Persian_poems_corpus).
- Word alignment: [awesome-align](https://github.com/neulab/awesome-align) (Dou & Neubig, 2021).
- Base models: [mBART-50](https://huggingface.co/facebook/mbart-large-50) (Tang et al., 2020); [Wav2Vec2-XLS-R 53 Persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian).
- Computing resources: Université du Québec à Montréal computing cluster (4× NVIDIA A40 GPUs).
