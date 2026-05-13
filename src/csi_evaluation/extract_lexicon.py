"""
Step 5 of the CSI pipeline — Build the Persian→English CSI lexicon.

Takes:
  - The CSI-tagged train + val CSVs (from annotate_corpus.py), and
  - The awesome-align outputs (forward, reverse, and symmetric intersection),
and produces a CSI lexicon mapping Persian CSI surface forms to their English
realizations, with frequency counts and (optional) alignment-confidence scores.

This is the raw lexicon. The three tiered files used by csi_recall_metric.py
(top1.canon.tsv, top3.canon.tsv, broad.canon.tsv) are derived from this raw
lexicon through canonicalization, frequency/rank filtering, and consistency
checks (see paper §5.1 lexicon section; thesis §2.7.3 Phase 4).

Inputs (under --aa-data-dir):
    trainval.parallel.txt              `fa ||| en` per line
    trainval.sym.intersect.align.txt   symmetric intersection alignments
    trainval.align.t05.txt             forward alignments        (optional)
    trainval.align.t05.prob.txt        forward alignment probs    (optional)
    trainval.rev.align.txt             reverse alignments         (optional)
    trainval.rev.align.prob.txt        reverse alignment probs    (optional)

Inputs (CSI-tagged data, via --train-csv and --val-csv):
    train_with_csi.csv, val_with_csi.csv
    (output of annotate_corpus.py — must include `persian_tokens` and `csi_tags`)

Outputs (under --output-dir):
    csi_lexicon.trainval.json
    csi_lexicon.trainval.tsv

Word alignment toolkit used: awesome-align (Dou & Neubig, 2021)
    https://github.com/neulab/awesome-align

Usage:
    python src/csi_evaluation/extract_lexicon.py \\
        --aa-data-dir data/aa/ \\
        --train-csv path/to/train_with_csi.csv \\
        --val-csv path/to/val_with_csi.csv \\
        --output-dir data/lexicon/
"""

import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def base_label(tag: str) -> str:
    """'B-CSI_PERSON' → 'CSI_PERSON'; 'O' → 'O'."""
    tag = (tag or "O").strip()
    if tag == "O":
        return "O"
    return tag.split("-", 1)[1] if "-" in tag else tag


def parse_edges(line: str) -> List[Tuple[int, int]]:
    """awesome-align edge list 'i-j i-j ...' → [(i,j), ...]"""
    if not line.strip():
        return []
    out = []
    for item in line.split():
        a, b = item.split("-")
        out.append((int(a), int(b)))
    return out


def parse_probs(line: str) -> List[float]:
    if not line.strip():
        return []
    return [float(x) for x in line.split()]


def load_parallel_line(line: str) -> Tuple[List[str], List[str]]:
    fa, en = [x.strip() for x in line.split("|||", 1)]
    return fa.split(), en.split()


def iter_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=","):
            yield row


def extract_csi_spans(tokens: List[str], tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Return CSI spans as (start_idx, end_idx_exclusive, base_label),
    based on word-level BIO tags.
    """
    spans = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tags[i]
        if t == "O":
            i += 1
            continue
        b = base_label(t)
        start = i
        i += 1
        while i < n and base_label(tags[i]) == b and tags[i] != "O":
            i += 1
        spans.append((start, i, b))
    return spans


def build_prob_maps(
    fwd_align_lines: Optional[List[str]],
    fwd_prob_lines: Optional[List[str]],
    rev_align_lines: Optional[List[str]],
    rev_prob_lines: Optional[List[str]],
):
    """
    Build per-line maps:
      fwd_map[line][(fa,en)] = prob
      rev_map_flipped[line][(fa,en)] = prob   (reverse edges flipped to fa,en)
    """
    if not (fwd_align_lines and fwd_prob_lines and rev_align_lines and rev_prob_lines):
        return None, None

    n = min(len(fwd_align_lines), len(fwd_prob_lines),
            len(rev_align_lines), len(rev_prob_lines))
    fwd_maps = []
    rev_maps = []
    for i in range(n):
        fwd_edges = parse_edges(fwd_align_lines[i])
        fwd_probs = parse_probs(fwd_prob_lines[i])
        if len(fwd_edges) != len(fwd_probs):
            fwd_maps.append({})
        else:
            fwd_maps.append({e: p for e, p in zip(fwd_edges, fwd_probs)})

        rev_edges = parse_edges(rev_align_lines[i])   # (en, fa)
        rev_probs = parse_probs(rev_prob_lines[i])
        if len(rev_edges) != len(rev_probs):
            rev_maps.append({})
        else:
            # flip to (fa, en)
            rev_maps.append({(fa, en): p for (en, fa), p in zip(rev_edges, rev_probs)})
    return fwd_maps, rev_maps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 5 — Build the Persian→English CSI lexicon from "
                    "CSI-tagged parallel corpus + awesome-align outputs."
    )
    p.add_argument("--aa-data-dir", required=True,
                   help="Directory containing awesome-align outputs "
                        "(trainval.parallel.txt, trainval.sym.intersect.align.txt, "
                        "and optionally the forward/reverse align + prob files).")
    p.add_argument("--train-csv", required=True,
                   help="Path to train_with_csi.csv.")
    p.add_argument("--val-csv", required=True,
                   help="Path to val_with_csi.csv.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write csi_lexicon.trainval.{json,tsv}.")
    return p.parse_args()


def main():
    args = parse_args()

    aa_dir = Path(args.aa_data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parallel_path = aa_dir / "trainval.parallel.txt"
    align_sym_path = aa_dir / "trainval.sym.intersect.align.txt"

    fwd_align_path = aa_dir / "trainval.align.t05.txt"
    fwd_prob_path = aa_dir / "trainval.align.t05.prob.txt"
    rev_align_path = aa_dir / "trainval.rev.align.txt"
    rev_prob_path = aa_dir / "trainval.rev.align.prob.txt"

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    out_json = out_dir / "csi_lexicon.trainval.json"
    out_tsv = out_dir / "csi_lexicon.trainval.tsv"

    # ----- Load core awesome-align artifacts -----
    parallel_lines = parallel_path.read_text(encoding="utf-8").splitlines()
    sym_lines = align_sym_path.read_text(encoding="utf-8").splitlines()
    n = min(len(parallel_lines), len(sym_lines))

    # ----- Optional probability artifacts (if present) -----
    fwd_maps = rev_maps = None
    if fwd_align_path.exists() and fwd_prob_path.exists() \
            and rev_align_path.exists() and rev_prob_path.exists():
        fwd_align_lines = fwd_align_path.read_text(encoding="utf-8").splitlines()
        fwd_prob_lines = fwd_prob_path.read_text(encoding="utf-8").splitlines()
        rev_align_lines = rev_align_path.read_text(encoding="utf-8").splitlines()
        rev_prob_lines = rev_prob_path.read_text(encoding="utf-8").splitlines()
        fwd_maps, rev_maps = build_prob_maps(
            fwd_align_lines, fwd_prob_lines, rev_align_lines, rev_prob_lines,
        )
        print("[INFO] Using forward+reverse prob files for confidence scoring.")
    else:
        print("[INFO] Prob files not found (or not all present). "
              "Proceeding without confidence scoring.")

    # ----- Load train+val CSI rows -----
    rows = []
    for row in iter_csv_rows(train_csv):
        rows.append(("train", row))
    for row in iter_csv_rows(val_csv):
        rows.append(("val", row))

    if len(rows) != n:
        print(f"[WARN] train+val CSV rows = {len(rows)} but AA lines = {n}.")
        print("       Not necessarily fatal (empty lines may have been filtered).")
        print("       Lexicon quality depends on strict row correspondence.")
    m = min(len(rows), n)

    # ----- Accumulator -----
    lex = defaultdict(lambda: {
        "count": 0,
        "en_candidates": Counter(),
        "score_sum": defaultdict(float),
        "score_count": defaultdict(int),
    })

    dropped_empty = 0
    used_lines = 0

    for i in range(m):
        split, row = rows[i]
        fa_tokens_csv = row["persian_tokens"].split()
        tags = row["csi_tags"].split()
        if len(fa_tokens_csv) != len(tags):
            continue

        fa_tok_aa, en_tok_aa = load_parallel_line(parallel_lines[i])

        # Strictest correctness check: AA source tokens must match CSV persian_tokens
        if fa_tok_aa != fa_tokens_csv:
            continue

        spans = extract_csi_spans(fa_tok_aa, tags)
        if not spans:
            continue

        sym_edges = parse_edges(sym_lines[i])
        # map fa index -> list of en indices
        fa2en = defaultdict(list)
        for fa_idx, en_idx in sym_edges:
            fa2en[fa_idx].append(en_idx)

        for s, e, base in spans:
            fa_phrase = " ".join(fa_tok_aa[s:e]).strip()
            if not fa_phrase:
                continue

            # collect aligned EN indices for all tokens in the span
            en_ids = []
            for fa_idx in range(s, e):
                en_ids.extend(fa2en.get(fa_idx, []))
            en_ids = sorted(set(en_ids))

            if not en_ids:
                dropped_empty += 1
                continue

            en_phrase = " ".join(
                en_tok_aa[j] for j in en_ids if 0 <= j < len(en_tok_aa)
            ).strip()
            if not en_phrase:
                dropped_empty += 1
                continue

            key = (base, fa_phrase)
            lex[key]["count"] += 1
            lex[key]["en_candidates"][en_phrase] += 1

            # Optional confidence: min(prob_fwd, prob_rev) averaged across edges
            if fwd_maps is not None and rev_maps is not None \
                    and i < len(fwd_maps) and i < len(rev_maps):
                conf_vals = []
                for fa_idx in range(s, e):
                    for en_idx in fa2en.get(fa_idx, []):
                        pf = fwd_maps[i].get((fa_idx, en_idx), None)
                        pr = rev_maps[i].get((fa_idx, en_idx), None)
                        if pf is not None and pr is not None:
                            conf_vals.append(min(pf, pr))
                if conf_vals:
                    conf = sum(conf_vals) / len(conf_vals)
                    lex[key]["score_sum"][en_phrase] += conf
                    lex[key]["score_count"][en_phrase] += 1

        used_lines += 1

    # ----- Serialize -----
    out = {
        "meta": {
            "lines_used": used_lines,
            "lines_compared": m,
            "dropped_empty_alignments": dropped_empty,
            "has_confidence": bool(fwd_maps and rev_maps),
        },
        "entries": [],
    }

    for (base, fa_phrase), obj in lex.items():
        cands = obj["en_candidates"].most_common(20)
        cand_list = []
        for en_phrase, cnt in cands:
            avg_conf = None
            if obj["score_count"].get(en_phrase, 0) > 0:
                avg_conf = obj["score_sum"][en_phrase] / obj["score_count"][en_phrase]
            cand_list.append({
                "en": en_phrase,
                "count": cnt,
                "avg_conf": avg_conf,
            })

        out["entries"].append({
            "label": base,
            "fa": fa_phrase,
            "count": obj["count"],
            "top_en": cand_list,
        })

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        f.write("label\tfa\tcount\ten\tencount\tavg_conf\n")
        for entry in sorted(out["entries"], key=lambda x: (-x["count"], x["label"], x["fa"])):
            label = entry["label"]
            fa = entry["fa"]
            total = entry["count"]
            for cand in entry["top_en"]:
                f.write(f"{label}\t{fa}\t{total}\t{cand['en']}\t{cand['count']}\t{cand['avg_conf']}\n")

    print(f"[OK] Lexicon saved:\n  {out_json}\n  {out_tsv}")
    print(f"[STATS] lines_used={used_lines} "
          f"dropped_empty_alignments={dropped_empty} entries={len(out['entries'])}")


if __name__ == "__main__":
    main()
