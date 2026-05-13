"""
Step 6 of the CSI pipeline — Compute CSI-Recall on a system's translations.

This is the user-facing metric from paper Table 3 and thesis §2.7.4. Given a
system's English hypotheses for the Masnavi test set, it reports how well the
system preserved Culture-Specific Items.

For each CSI span in the test set that the lexicon can judge:
  - A span is **covered** if its (label, persian_surface) is in the lexicon.
  - A span is **matched** if any expected English realization (canonicalized)
    appears in the hypothesis (canonicalized).

Three lexicon variants reported (paper §5.1; thesis §2.7.3 Phase 5):

  - strict_core_top1  → strict core lexicon (rank == 1, gold-verified)
  - soft_core_top3    → core lexicon (rank ≤ 3, strong confidence)
  - broad_rank_le3    → broad lexicon (rank ≤ 3, minimal threshold)

Per variant, the script outputs:
  - Total CSI spans in the test set
  - Lexicon-covered spans (= coverage)
  - Matched spans (= CSI-Recall numerator)
  - CSI-Recall = matched / covered

Reference: SilkRoadNLP 2026 paper §6.1; thesis §2.7.4.

Usage:
    python -m src.csi_evaluation.csi_recall_metric \\
        --hypotheses path/to/predictions.txt \\
        --test-csv data/test_with_csi.csv \\
        --lexicon-top1 data/lexicon/top1.canon.tsv \\
        --lexicon-top3 data/lexicon/top3.canon.tsv \\
        --lexicon-broad data/lexicon/broad.canon.tsv \\
        --output runs/ \\
        --tag my_system

The hypothesis file is one English translation per line, in the same order
as the test CSV.

Outputs (under --output):
  - csi_mt_metric_report_<tag>_<variant>.csv  (per-span detail rows, one file
                                                per lexicon variant)
  - A summary on stdout

CPU is fine for this step.
"""

import argparse
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, Set, List, Optional


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------

WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s'-]+", re.UNICODE)
LEADING_ARTICLES_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
LEADING_OF_RE = re.compile(r"^of\s+", re.IGNORECASE)


def clean_en(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    # Normalize curly quotes
    s = s.replace("\u2019", "'").replace("\u201C", '"').replace("\u201D", '"')
    return s


def ascii_fold(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def canon_en(s: str) -> str:
    """Canonicalize an English string for matching.

    Lowercase, strip diacritics, drop leading articles ('the', 'a', 'an')
    and leading 'of', remove punctuation, collapse whitespace.
    """
    s = clean_en(s)
    if not s:
        return ""
    s = ascii_fold(s)
    s = s.lower().strip()

    prev = None
    while prev != s:
        prev = s
        s = LEADING_ARTICLES_RE.sub("", s).strip()
        s = LEADING_OF_RE.sub("", s).strip()

    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# CSI tag utilities
# ---------------------------------------------------------------------------


def base_label(tag: str) -> str:
    tag = (tag or "").strip()
    if tag == "O" or tag == "":
        return "O"
    if tag.startswith("B-") or tag.startswith("I-"):
        return tag.split("-", 1)[1]
    return tag


def extract_spans(persian_tokens: List[str], csi_tags: List[str]) -> List[Dict]:
    """
    Span extraction from word-level tokens + BIO-ish tags.

    Treats contiguous tokens with the same base label as a single span.
    """
    spans = []
    n = min(len(persian_tokens), len(csi_tags))
    i = 0
    while i < n:
        tag = csi_tags[i]
        if tag == "O":
            i += 1
            continue
        lab = base_label(tag)
        start = i
        i += 1
        while i < n and csi_tags[i] != "O" and base_label(csi_tags[i]) == lab:
            i += 1
        end = i
        fa_surface = " ".join(persian_tokens[start:end]).strip()
        if fa_surface:
            spans.append({"label": lab, "fa": fa_surface, "start": start, "end": end})
    return spans


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------


def load_lexicon(path: Path, *, max_rank: Optional[int]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Load a canonicalized lexicon TSV.

    Expects columns:
      label, fa, en_raw, en_canon, count, avg_prob, score, rank, ambiguous_flag

    Returns: dict mapping (label, fa) → set of expected English canonical forms.
    """
    lex = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            label = (r.get("label") or "").strip()
            fa = (r.get("fa") or "").strip()
            enc = (r.get("en_canon") or "").strip()
            try:
                rank = int(r.get("rank") or 999999)
            except (TypeError, ValueError):
                rank = 999999

            if not label or not fa or not enc:
                continue
            if max_rank is not None and rank > max_rank:
                continue

            lex[(label, fa)].add(enc)
    return dict(lex)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_test_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def read_hyps(path: Path) -> List[str]:
    hyps = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            hyps.append(line.strip())
    return hyps


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------


def evaluate_variant(
    *,
    variant_name: str,
    lex: Dict[Tuple[str, str], Set[str]],
    test_rows: List[Dict],
    hyps: List[str],
    out_csv_path: Path,
) -> Dict:
    total_spans = 0
    covered_spans = 0
    matched_spans = 0

    per_label_cov = Counter()
    per_label_match = Counter()

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "variant", "label", "fa", "covered", "matched",
                    "expected_en_canon", "hyp_canon"])

        for row_id, (r, hyp) in enumerate(zip(test_rows, hyps)):
            pers_tok = (r.get("persian_tokens") or "").strip().split()
            tags = (r.get("csi_tags") or "").strip().split()

            spans = extract_spans(pers_tok, tags)
            hyp_c = canon_en(hyp)

            for sp in spans:
                label = sp["label"]
                fa = sp["fa"]

                total_spans += 1

                expected = lex.get((label, fa))
                is_covered = 1 if expected else 0
                is_matched = 0

                if expected:
                    covered_spans += 1
                    per_label_cov[label] += 1

                    # Match if any expected realization (canonicalized) appears
                    # as a substring in the canonicalized hypothesis. Substring
                    # matching is fast and deterministic; alternatives include
                    # whole-token or fuzzy matching (future work).
                    for e in expected:
                        if e and e in hyp_c:
                            is_matched = 1
                            break

                    if is_matched:
                        matched_spans += 1
                        per_label_match[label] += 1

                exp_str = " | ".join(sorted(expected)) if expected else ""
                w.writerow([row_id, variant_name, label, fa, is_covered,
                            is_matched, exp_str, hyp_c])

    recall_given_cov = (matched_spans / covered_spans) if covered_spans > 0 else 0.0

    per_label_stats = []
    for lab, cov in per_label_cov.items():
        m = per_label_match.get(lab, 0)
        per_label_stats.append((lab, cov, m, (m / cov if cov else 0.0)))
    per_label_stats.sort(key=lambda x: x[1], reverse=True)

    return {
        "variant": variant_name,
        "total_spans": total_spans,
        "covered_spans": covered_spans,
        "coverage": (covered_spans / total_spans) if total_spans else 0.0,
        "matched_spans": matched_spans,
        "recall_given_coverage": recall_given_cov,
        "per_label_stats": per_label_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute CSI-Recall on a system's Persian→English translations.",
    )
    p.add_argument("--hypotheses", required=True,
                   help="Path to a .txt file: one hypothesis per line, "
                        "in the same order as the test CSV.")
    p.add_argument("--test-csv", required=True,
                   help="Path to test_with_csi.csv (must contain persian_tokens "
                        "and csi_tags columns).")
    p.add_argument("--lexicon-top1", required=True,
                   help="strict_core lexicon (rank=1, gold-verified).")
    p.add_argument("--lexicon-top3", required=True,
                   help="soft_core lexicon (rank ≤ 3, strong confidence).")
    p.add_argument("--lexicon-broad", required=True,
                   help="broad lexicon (rank ≤ 3, minimal threshold).")
    p.add_argument("--output", default="runs/",
                   help="Output directory for the per-variant detail CSVs.")
    p.add_argument("--tag", default="hypothesis",
                   help="Tag used in report filenames: "
                        "csi_mt_metric_report_<tag>_<variant>.csv")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n================ CSI-aware MT Metric ================\n")

    # Load test rows and hypotheses
    test_rows = read_test_rows(Path(args.test_csv))
    hyps = read_hyps(Path(args.hypotheses))

    if len(test_rows) != len(hyps):
        raise RuntimeError(
            f"Row mismatch: test_rows={len(test_rows)} vs hyps={len(hyps)}.\n"
            f"  test_csv:    {args.test_csv}\n"
            f"  hypotheses:  {args.hypotheses}\n"
            f"The hypothesis file must have the same number of lines as "
            f"the test CSV (data rows; excluding the header row)."
        )

    # Load the three lexicon variants
    core_top1 = load_lexicon(Path(args.lexicon_top1), max_rank=1)
    core_top3 = load_lexicon(Path(args.lexicon_top3), max_rank=3)
    broad_le3 = load_lexicon(Path(args.lexicon_broad), max_rank=3)

    print(f"[INFO] Loaded test rows: {len(test_rows)}")
    print(f"[INFO] Loaded hypotheses: {len(hyps)}")
    print(f"[INFO] Lexicon strict_core_top1: {len(core_top1)} keys")
    print(f"[INFO] Lexicon soft_core_top3:   {len(core_top3)} keys")
    print(f"[INFO] Lexicon broad_rank_le3:   {len(broad_le3)} keys")
    print()

    results = []
    for name, lex in [
        ("strict_core_top1", core_top1),
        ("soft_core_top3", core_top3),
        ("broad_rank_le3", broad_le3),
    ]:
        out_csv = out_dir / f"csi_mt_metric_report_{args.tag}_{name}.csv"
        res = evaluate_variant(
            variant_name=name,
            lex=lex,
            test_rows=test_rows,
            hyps=hyps,
            out_csv_path=out_csv,
        )
        results.append((res, out_csv))

    # Summary
    for res, out_csv in results:
        print(f"[{res['variant']}]")
        print(f"  total CSI spans (from tags):    {res['total_spans']}")
        print(f"  lexicon-covered spans:          {res['covered_spans']}  "
              f"(coverage={res['coverage']*100:.2f}%)")
        print(f"  matched spans:                  {res['matched_spans']}")
        print(f"  CSI recall (matched/covered):   {res['recall_given_coverage']*100:.2f}%")
        print("  per-label recall (top-10 by support):")
        for lab, cov, m, rec in res["per_label_stats"][:10]:
            print(f"    {lab:30s} covered={cov:4d}  matched={m:4d}  recall={rec*100:6.2f}%")
        print(f"\n  [OK] Detailed report: {out_csv}\n")

    print("=====================================================\n")


if __name__ == "__main__":
    main()
