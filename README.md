# persian-culture-aware-translation

Code companion for the SilkRoadNLP 2026 paper:

**[Multi-modal Neural Machine Translation for Low-Resource Classical Persian Poetry: A Culture-Aware Evaluation](https://aclanthology.org/2026.silkroadnlp-1.14/)**
Soheila Ansari, Mounir Boukadoum, Fatiha Sadat
*Proceedings of the First Workshop on NLP and LLMs for the Iranian Language Family (SilkRoadNLP @ ACL 2026), Rabat, Morocco, March 2026, pp. 131–139.*

---

## What this is

A reproducibility package for two contributions from the paper:

1. **PPFT** — a text-only Persian→English translation baseline built by domain-adaptive pre-training and fine-tuning of mBART-50 on classical Persian poetry.
2. **CSI evaluation** — a culture-aware evaluation framework with a Persian–English Culture-Specific Item (CSI) tagger, lexicon, and the CSI-Recall metric.

Refer to the paper for the full methodology, results, related work, and discussion. This repository provides the code so others can re-run the pipelines and apply the CSI metric to their own translation systems.

---

## Repository layout

```
.
├── README.md
├── LICENSE                MIT (code) + CC-BY 4.0 (data artifacts when added)
├── CITATION.cff           Citation metadata; renders as a sidebar widget on GitHub
├── environment.yml        conda environment
├── .gitignore
│
├── src/
│   ├── text_baseline/
│   │   ├── domain_pretrain.py    Stage 1: span-masked DAP on monolingual Persian poetry
│   │   ├── finetune.py           Stage 2: parallel fine-tuning (Masnavi)
│   │   └── evaluate.py           BLEU + chrF++ + BERTScore + COMET
│   │
│   └── csi_evaluation/
│       ├── prepare_bio_dataset.py        Step 1: gold spans → BIO subword data
│       ├── train_tagger.py               Step 2: train CSI tagger
│       ├── annotate_corpus.py            Step 3: tag full parallel corpus
│       ├── prepare_alignment_input.py    Step 4: prepare fa↔en for awesome-align
│       ├── extract_lexicon.py            Step 5: build the CSI lexicon
│       └── csi_recall_metric.py          Step 6: compute CSI-Recall (user-facing)
│
└── data/
    └── README.md          Schema and provenance of data artifacts
```

---

## Installation

```bash
git clone https://github.com/Soheila1992/persian-culture-aware-translation
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

The hypothesis file is one English translation per line, in the same order as the test CSV. The script reports CSI-Recall across three lexicon variants (strict_core, soft_core, broad). See paper §6 for the methodology and Table 3 for expected values on PPFT and PPFT + Wav2Vec2.

CPU is sufficient for this step.

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
    --train-csv path/to/train.csv \
    --val-csv path/to/val.csv \
    --test-csv path/to/test.csv \
    --dap-checkpoint runs/dap/checkpoints/best_model.pt \
    --output-dir runs/ppft

# Evaluation (BLEU + chrF++ + BERTScore + COMET)
python src/text_baseline/evaluate.py \
    --test-csv path/to/test.csv \
    --checkpoint runs/ppft/checkpoints/best_hf \
    --output runs/ppft_eval_report.json
```

The Persian poetry corpus used for Stage 1 (1M lines, Rumi removed) is publicly available: [amnghd/Persian_poems_corpus](https://github.com/amnghd/Persian_poems_corpus/tree/master).

Single-GPU runs work with `--nproc_per_node=1`. CPU is not feasible for training (it is fine for evaluation).

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
    --parallel-root path/to/parallel/csv/folder \
    --output-root path/to/csi_tagged/ \
    --csi-model-dir runs/csi_tagger

# Step 4 — Prepare for awesome-align (forward + reverse)
python src/csi_evaluation/prepare_alignment_input.py \
    --train-csv path/to/csi_tagged/train_with_csi.csv \
    --val-csv path/to/csi_tagged/val_with_csi.csv \
    --output-dir data/aa/ --direction forward
python src/csi_evaluation/prepare_alignment_input.py \
    --train-csv path/to/csi_tagged/train_with_csi.csv \
    --val-csv path/to/csi_tagged/val_with_csi.csv \
    --output-dir data/aa/ --direction reverse

# Run awesome-align externally on data/aa/trainval.parallel.txt
# and data/aa/trainval.parallel.rev.txt; see https://github.com/neulab/awesome-align

# Step 5 — Build the CSI lexicon
python src/csi_evaluation/extract_lexicon.py \
    --aa-data-dir data/aa/ \
    --train-csv path/to/csi_tagged/train_with_csi.csv \
    --val-csv path/to/csi_tagged/val_with_csi.csv \
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

## Scope of this release

This repository releases code for the two components described above. The multi-modal extension (PPFT + Wav2Vec2 with cross-attention fusion) is described in the paper but not included here, as it depends on audio data that is not redistributed.

Audio recitations were obtained from [masnavi.net](https://masnavi.net/) for non-commercial academic research; researchers wishing to reproduce the multi-modal experiments should obtain audio directly from that source. The Persian text and Nicholson's English translation are publicly available; the full parallel corpus is not committed to this repository in its initial release.

---

## License

- **Source code:** MIT License (see `LICENSE`).
- **Data artifacts** (CSI taxonomy, gold annotations, BIO datasets, lexicons, test set with CSI tags, when added to `data/`): CC-BY 4.0.

---

## Acknowledgements

- Persian text and audio recordings: [masnavi.net](https://masnavi.net/); reciters Hosayn Āhī (books 1, 6) and Amīr Nūrī (books 2–5); sponsored by Noorsoft.
- English translation: Reynold A. Nicholson (1925–1940), public domain since 2015.
- Persian poetry corpus for DAP: [amnghd/Persian_poems_corpus](https://github.com/amnghd/Persian_poems_corpus).
- Word alignment: [awesome-align](https://github.com/neulab/awesome-align) (Dou & Neubig, 2021).
- Base model: [mBART-50](https://huggingface.co/facebook/mbart-large-50) (Tang et al., 2020).
- Computing resources: Université du Québec à Montréal computing cluster (4× NVIDIA A40 GPUs).
