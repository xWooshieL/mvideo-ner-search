"""
Общие утилиты для GLiNER (zero-shot + fine-tune) на поверх нашей BIO-схемы.

Не трогает CRF (`model_crf.py`) и `WeakLabeler` — отдельный, параллельный слой
каскада (rules → CRF → GLiNER на хвосте).

Наши сущности: BRAND / CATEGORY / MODEL / ATTR (см. `src/ner/labeling.py`).
GLiNER работает по произвольным текстовым лейблам ("brand", "product model", ...),
поэтому здесь — маппинг наш_лейбл <-> промпт-лейбл GLiNER в обе стороны,
плюс конвертация BIO (наш gold/silver) <-> формат обучения GLiNER
(`{"tokenized_text": [...], "ner": [[start, end, label]]}`, end включительно).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.ner.labeling import tokenize

OUR_LABELS = ["BRAND", "CATEGORY", "MODEL", "ATTR"]

# Кейсы, которых в gold почти нет / нет вовсе — стресс-тест хвоста каскада.
# Общий список для zero-shot (01) и fine-tune (02) — сравниваем на одних и тех же примерах.
EDGE_CASES = [
    "асус тюф гейминг а15",  # опечатка + кириллица вместо латиницы (rules/CRF слабы)
    "iphone 16 pro max 256",  # свежая линейка, может не быть в словарях/gold
    "чехол для айфон 15 синий силиконовый",  # длинный, цвет+материал+назначение
    "наушники",  # 1 токен, только категория
    "xiaomi",  # 1 токен, только бренд
    "телевизор 65 дюймов 4к",  # атрибуты без бренда
    "холодильник lg no frost 300 л",  # бренд + техно-фича + объём
    "плойка д/волос",  # сокращение "д/" — токенизация не по словарю
    "смартфон samsung galaxy s24 ultra 512gb черный",  # много сущностей подряд
    "非小米平板",  # мусор/чужой алфавит — не должно взорваться
]

# Несколько наборов промптов — сравниваем на gold, что заходит multi-v2.1 лучше.
LABEL_PROMPTS: Dict[str, Dict[str, str]] = {
    "en": {
        "BRAND": "brand",
        "CATEGORY": "product category",
        "MODEL": "product model or model code",
        "ATTR": "product attribute or specification",
    },
    "ru": {
        "BRAND": "бренд",
        "CATEGORY": "категория товара",
        "MODEL": "модель или код модели товара",
        "ATTR": "характеристика или атрибут товара",
    },
}


def prompt_to_label_map(scheme: str = "en") -> Dict[str, str]:
    """our_label -> gliner_prompt для схемы промптов."""
    return dict(LABEL_PROMPTS[scheme])


def label_to_prompt_map(scheme: str = "en") -> Dict[str, str]:
    """gliner_prompt -> our_label (обратный маппинг, для разбора предсказаний)."""
    return {v: k for k, v in LABEL_PROMPTS[scheme].items()}


def gliner_labels(scheme: str = "en") -> List[str]:
    return [LABEL_PROMPTS[scheme][l] for l in OUR_LABELS]


# ---------------------------------------------------------------------------
# BIO (наш gold/silver) -> формат обучения GLiNER
# ---------------------------------------------------------------------------


def bio_tags_to_token_spans(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    """[(start_tok, end_tok_inclusive, label), ...] из списка BIO-тегов."""
    spans: List[Tuple[int, int, str]] = []
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{label}":
                j += 1
            spans.append((i, j - 1, label))
            i = j
        else:
            i += 1
    return spans


def bio_row_to_gliner_example(
    tokens: Sequence[str],
    tags: Sequence[str],
    *,
    scheme: str = "en",
    only_labels: Optional[Sequence[str]] = None,
) -> Dict:
    """Один пример в формате GLiNER: {"tokenized_text": [...], "ner": [[s,e,prompt], ...]}."""
    prompts = prompt_to_label_map(scheme)
    keep = set(only_labels) if only_labels else set(OUR_LABELS)
    spans = bio_tags_to_token_spans(tags)
    ner = [
        [s, e, prompts[label]]
        for s, e, label in spans
        if label in keep and label in prompts
    ]
    return {"tokenized_text": list(tokens), "ner": ner}


def load_gold_bio_rows(gold_path: Path | str) -> List[Dict]:
    """Читает наш gold BIO jsonl (`data/gold/bio_liza.jsonl`) и ретокенизирует
    тем же `tokenize()`, что и CRF/фичи — чтобы теги и токены были согласованы.

    Возвращает список {"query", "tokens", "tags", "char_spans"} (char_spans —
    [(start,end)] на нормализованном тексте, как в `bio_to_entities`).
    """
    rows: List[Dict] = []
    p = Path(gold_path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = r["query"]
        tags = r["tags"]
        tok_spans = tokenize(q)
        toks = [t for t, _, _ in tok_spans]
        if len(toks) != len(tags):
            continue  # рассинхрон токенизации — пропускаем (как в _run_02)
        rows.append(
            {
                "query": q,
                "tokens": toks,
                "tags": tags,
                "char_spans": [(s, e) for _, s, e in tok_spans],
            }
        )
    return rows


def gold_rows_to_gliner_dataset(
    rows: Sequence[Dict], *, scheme: str = "en"
) -> List[Dict]:
    return [
        bio_row_to_gliner_example(r["tokens"], r["tags"], scheme=scheme)
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Gold в char-спанах (для eval предсказаний GLiNER, которые тоже в char-спанах)
# ---------------------------------------------------------------------------


def gold_row_char_entities(row: Dict) -> List[Dict]:
    """[{"label": "BRAND", "start": int, "end": int, "text": str}, ...]."""
    spans = bio_tags_to_token_spans(row["tags"])
    char_spans = row["char_spans"]
    out = []
    for s, e, label in spans:
        if e >= len(char_spans):
            continue
        a, b = char_spans[s][0], char_spans[e][1]
        out.append({"label": label, "start": a, "end": b, "text": row["query"][a:b]})
    return out


def gliner_preds_to_our_labels(
    preds: Sequence[Dict], *, scheme: str = "en"
) -> List[Dict]:
    """Предсказания GLiNER (label=промпт) -> наши канонические лейблы."""
    rev = label_to_prompt_map(scheme)
    out = []
    for p in preds:
        label = rev.get(p.get("label", ""))
        if label is None:
            continue
        out.append(
            {
                "label": label,
                "start": p["start"],
                "end": p["end"],
                "text": p.get("text", ""),
                "score": p.get("score"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Span-level P/R/F1 (точное совпадение [start,end,label], как в NER-бенчмарках)
# ---------------------------------------------------------------------------


def span_prf1(
    gold: Sequence[Sequence[Dict]], pred: Sequence[Sequence[Dict]]
) -> Dict:
    """gold/pred — списки по запросам, каждый — список {"label","start","end"}.
    Возвращает micro P/R/F1 общий и per-label.
    """
    from collections import Counter

    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for g_ents, p_ents in zip(gold, pred):
        g_set = {(e["label"], e["start"], e["end"]) for e in g_ents}
        p_set = {(e["label"], e["start"], e["end"]) for e in p_ents}
        for key in p_set & g_set:
            tp[key[0]] += 1
        for key in p_set - g_set:
            fp[key[0]] += 1
        for key in g_set - p_set:
            fn[key[0]] += 1

    labels = sorted(set(tp) | set(fp) | set(fn))
    per_label = {}
    for l in labels:
        p_ = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        r_ = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        per_label[l] = {
            "precision": round(p_, 4),
            "recall": round(r_, 4),
            "f1": round(f1_, 4),
            "tp": tp[l],
            "fp": fp[l],
            "fn": fn[l],
        }

    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {
        "micro": {"precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4), "tp": TP, "fp": FP, "fn": FN},
        "per_label": per_label,
    }


def evaluate_rows_char(
    model: Any,
    rows: Sequence[Dict],
    *,
    scheme: str = "en",
    threshold: float = 0.5,
) -> Dict:
    """Гоняет `model.predict_entities` по gold-строкам (из `load_gold_bio_rows`)
    и считает span-level P/R/F1 в наших лейблах + среднюю latency/запрос.

    Общая функция для zero-shot (01) и fine-tune (02) — те же rows/scheme
    дают сравнимые метрики "до/после".
    """
    labels = gliner_labels(scheme)
    gold_ents, pred_ents = [], []
    t0 = time.perf_counter()
    for r in rows:
        raw_preds = model.predict_entities(r["query"], labels, threshold=threshold)
        pred_ents.append(gliner_preds_to_our_labels(raw_preds, scheme=scheme))
        gold_ents.append(gold_row_char_entities(r))
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(rows))
    metrics = span_prf1(gold_ents, pred_ents)
    metrics["avg_latency_ms"] = round(latency_ms, 2)
    return metrics


def predict_edge_cases(
    model: Any, queries: Sequence[str], *, scheme: str = "en", threshold: float = 0.5
) -> List[Dict]:
    """[{"query", "entities": [...]}] для качественного разбора хвостовых кейсов."""
    labels = gliner_labels(scheme)
    out = []
    for q in queries:
        preds = gliner_preds_to_our_labels(
            model.predict_entities(q, labels, threshold=threshold), scheme=scheme
        )
        out.append({"query": q, "entities": preds})
    return out


__all__ = [
    "OUR_LABELS",
    "EDGE_CASES",
    "LABEL_PROMPTS",
    "prompt_to_label_map",
    "label_to_prompt_map",
    "gliner_labels",
    "bio_tags_to_token_spans",
    "bio_row_to_gliner_example",
    "load_gold_bio_rows",
    "gold_rows_to_gliner_dataset",
    "gold_row_char_entities",
    "gliner_preds_to_our_labels",
    "span_prf1",
    "evaluate_rows_char",
    "predict_edge_cases",
]
