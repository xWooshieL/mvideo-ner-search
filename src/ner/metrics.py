"""Token Accuracy and entity-level Precision / Recall / F1."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def token_accuracy(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
) -> float:
    correct = 0
    total = 0
    for true_seq, pred_seq in zip(y_true, y_pred):
        for t, p in zip(true_seq, pred_seq):
            correct += int(t == p)
            total += 1
    return correct / total if total else 0.0


def _get_spans(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Return list of (label, start, end_exclusive) from BIO tags."""
    spans: List[Tuple[str, int, int]] = []
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{label}":
                j += 1
            spans.append((label, i, j))
            i = j
        else:
            i += 1
    return spans


def entity_f1_report(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    labels: Iterable[str] | None = None,
) -> Dict:
    """Entity-level micro/macro P/R/F1 and per-label scores."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true_seq, pred_seq in zip(y_true, y_pred):
        true_spans = set(_get_spans(true_seq))
        pred_spans = set(_get_spans(pred_seq))
        for span in true_spans & pred_spans:
            tp[span[0]] += 1
        for span in pred_spans - true_spans:
            fp[span[0]] += 1
        for span in true_spans - pred_spans:
            fn[span[0]] += 1

    all_labels = sorted(set(tp) | set(fp) | set(fn) | set(labels or []))
    per_label = {}
    sum_tp = sum_fp = sum_fn = 0
    f1s = []
    for lab in all_labels:
        p = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        r = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[lab] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": tp[lab] + fn[lab],
        }
        sum_tp += tp[lab]
        sum_fp += fp[lab]
        sum_fn += fn[lab]
        f1s.append(f1)

    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro_f1": macro_f1,
        "per_label": per_label,
        "token_accuracy": None,
    }


def summarize_metrics(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
) -> Dict:
    report = entity_f1_report(y_true, y_pred)
    report["token_accuracy"] = token_accuracy(y_true, y_pred)
    return report


def classification_brand_category_accuracy(
    true_brands: Sequence[str | None],
    pred_brands: Sequence[str | None],
    true_cats: Sequence[str | None],
    pred_cats: Sequence[str | None],
) -> Dict[str, float]:
    def _acc(a, b):
        pairs = [(x, y) for x, y in zip(a, b) if x]
        if not pairs:
            return 0.0
        return sum(1 for x, y in pairs if (x or "").lower() == (y or "").lower()) / len(pairs)

    return {
        "brand_accuracy": _acc(true_brands, pred_brands),
        "category_accuracy": _acc(true_cats, pred_cats),
    }
