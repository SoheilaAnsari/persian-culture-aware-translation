# Data

Place CSI evaluation data here. The CSI-Recall metric in `src/csi_evaluation/csi_recall_metric.py` expects:

```
data/
├── test_with_csi.csv               Test set with CSI annotations
├── annotated_csi_filtered.jsonl    Gold span-based annotations
├── csi_ner_train.jsonl             BIO training data
├── csi_ner_dev.jsonl               BIO development data
├── csi_label_mapping.json          BIO label↔id mapping
└── lexicon/
    ├── top1.canon.tsv              strict_core lexicon (rank=1)
    ├── top3.canon.tsv              soft_core lexicon (rank≤3)
    └── broad.canon.tsv             broad lexicon (rank≤3)
```

These files are released as part of the paper supplementary materials under CC-BY 4.0. The Persian text and English translations they contain derive from public-domain materials.

## Format reference

### `test_with_csi.csv` (2,658 verses)

| Column | Description |
|---|---|
| `book_number` | Masnavi book (1–6) |
| `persian_text` | Persian verse, normalized |
| `english_translation` | Nicholson's English translation (public domain) |
| `audio_filename` | Filename of the audio recitation (audio not redistributed) |
| `language` | Always `fa` |
| `persian_tokens` | Space-separated word-level tokens, aligned 1:1 with `csi_tags` |
| `csi_tags` | Space-separated word-level BIO labels (e.g., `O B-CSI_PERSON I-CSI_PERSON O`) |

### `annotated_csi_filtered.jsonl` (999 gold examples)

```json
{
  "book_number": 6,
  "persian_text": "...",
  "annotations": [
    {"span": "...", "label": "CSI_DIVINE_ATTRIBUTE"},
    {"span": "...", "label": "CSI_QURANIC_REF"}
  ]
}
```

### `csi_ner_{train,dev}.jsonl`

```json
{
  "book_number": 5,
  "text": "...",
  "tokens": ["fa_IR", "▁...", ..., "</s>"],
  "labels": ["O", "B-CSI_...", "I-CSI_...", ..., "O"]
}
```

Tokens are mBART-50 subwords (with `fa_IR` language prefix and `</s>` suffix); labels are BIO tags, same length as tokens.

### `csi_label_mapping.json`

```json
{"label2id": {"B-CSI_ANIMAL_SYMBOL": 0, ..., "O": 34},
 "id2label": {"0": "B-CSI_ANIMAL_SYMBOL", ..., "34": "O"}}
```

35 BIO labels total: 17 CSI categories × {B-, I-} + `O`. See the paper §5.1 for the full taxonomy.

### `lexicon/*.canon.tsv`

Tab-separated, with columns including: `label`, `fa`, `en_raw`, `en_canon`, `count`, `avg_prob`, `score`, `rank`. The CSI-Recall metric reads `label`, `fa`, `en_canon`, and `rank`.

## Obtaining the parallel corpus

The full parallel train/val/test CSVs are not redistributed in this initial release. The Persian text comes from [masnavi.net](https://masnavi.net/); the English translation is Nicholson's (public domain). For the specific verse-aligned CSV used in the paper, contact the corresponding author.

The Persian poetry corpus used for domain pre-training is publicly available: [amnghd/Persian_poems_corpus](https://github.com/amnghd/Persian_poems_corpus/tree/master).
