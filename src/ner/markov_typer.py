# -*- coding: utf-8 -*-
"""Марковский типизатор атрибутов: span B/I-ATTR -> тип (memory, size, ...).

Готовый алгоритм для обучения на silver-датасете (weak-разметке).

Идея
----
CRF/словарь находят span атрибута («16 гб»), но не его тип. Марковская цепь
1-го порядка выучивает из silver-данных эмпирические частоты:

    P(type | t_i, t_{i+1})   — биграмма токенов внутри span
    P(type | unit)           — запасной словарь единиц измерения

Инференс: смотрим биграмму (16, гб) -> argmax P(type). Если биграмма не
встречалась — fallback на единицу измерения, иначе "unknown". Это табличный
lookup: latency << 1 мс, никаких нейросетей.

Как обучить на silver
---------------------
    from src.ner.markov_typer import MarkovAttrTyper
    typer = MarkovAttrTyper.train_on_silver(queries)   # список строк
    typer.save("models/markov_typer.json")

Дальше в проде:
    typer = MarkovAttrTyper.load("models/markov_typer.json")
    typer.predict(["16", "гб"])   # -> ("memory_storage", 0.97)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_NUM_RE = re.compile(r"^\d+(?:[.,/]\d+)?$")


def _norm_token(tok: str) -> str:
    tok = tok.lower().strip()
    # все числа схлопываем в один плейсхолдер, чтобы цепь обобщала «16 гб» и «256 гб»
    if _NUM_RE.match(tok):
        return "<num>"
    return tok


@dataclass
class MarkovAttrTyper:
    """Табличная марковская модель типизации ATTR-спанов."""

    bigram_to_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    unit_to_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n_spans: int = 0

    # ------------------------------------------------------------------ train
    @classmethod
    def train_on_silver(
        cls,
        queries: Iterable[str],
        max_queries: Optional[int] = None,
    ) -> "MarkovAttrTyper":
        """Обучение на silver: прогоняем regex-разметку и копим частоты.

        Учитель — ATTR_PATTERNS из labeling.py: каждый матч даёт (span, тип).
        Из спанов собираем биграммы и словарь единиц.
        """
        from src.ner.labeling import ATTR_PATTERNS, _split_glued

        typer = cls()
        big = defaultdict(Counter)
        unit = defaultdict(Counter)
        trans = defaultdict(Counter)

        for qi, query in enumerate(queries):
            if max_queries is not None and qi >= max_queries:
                break
            text = _split_glued(str(query).lower())
            for pattern, attr_type in ATTR_PATTERNS:
                for m in pattern.finditer(text):
                    toks = [_norm_token(t) for t in m.group(0).split()]
                    if not toks:
                        continue
                    typer.n_spans += 1
                    # биграммы внутри спана
                    for a, b in zip(toks, toks[1:]):
                        big[f"{a}|{b}"][attr_type] += 1
                        trans[a][b] += 1
                    # единицы (не числовые токены)
                    for t in toks:
                        if t != "<num>":
                            unit[t][attr_type] += 1

        typer.bigram_to_type = {k: dict(v) for k, v in big.items()}
        typer.unit_to_type = {k: dict(v) for k, v in unit.items()}
        typer.transitions = {k: dict(v) for k, v in trans.items()}
        return typer

    # -------------------------------------------------------------- inference
    def predict(self, span_tokens: Sequence[str]) -> Tuple[str, float]:
        """Тип спана + уверенность (эмпирическая вероятность)."""
        toks = [_norm_token(t) for t in span_tokens]

        # 1) биграммы — самый сильный сигнал
        votes: Counter = Counter()
        for a, b in zip(toks, toks[1:]):
            dist = self.bigram_to_type.get(f"{a}|{b}")
            if dist:
                votes.update(dist)
        if votes:
            best, cnt = votes.most_common(1)[0]
            return best, cnt / sum(votes.values())

        # 2) fallback: единица измерения
        for t in toks:
            dist = self.unit_to_type.get(t)
            if dist:
                best = max(dist, key=dist.get)
                return best, dist[best] / sum(dist.values())

        return "unknown", 0.0

    def next_token_dist(self, token: str) -> Dict[str, float]:
        """P(next | token) — полезно для отладки и автодополнения."""
        dist = self.transitions.get(_norm_token(token), {})
        total = sum(dist.values()) or 1
        return {k: v / total for k, v in sorted(dist.items(), key=lambda x: -x[1])}

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        payload = {
            "bigram_to_type": self.bigram_to_type,
            "unit_to_type": self.unit_to_type,
            "transitions": self.transitions,
            "n_spans": self.n_spans,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MarkovAttrTyper":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            bigram_to_type=payload["bigram_to_type"],
            unit_to_type=payload["unit_to_type"],
            transitions=payload.get("transitions", {}),
            n_spans=payload.get("n_spans", 0),
        )


def train_and_save_default(
    parquet_path: str | Path = "data/query_clicks.parquet",
    out_path: str | Path = "models/markov_typer.json",
    n_queries: int = 60_000,
) -> "MarkovAttrTyper":
    """Скрипт-обёртка: обучить на silver из кликов и сохранить."""
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["toValidUTF8(query_text)"]).head(1_500_000)
    queries = df["toValidUTF8(query_text)"].astype(str).drop_duplicates().head(n_queries)
    typer = MarkovAttrTyper.train_on_silver(queries)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    typer.save(out_path)
    print(f"markov typer: {typer.n_spans} spans, "
          f"{len(typer.bigram_to_type)} bigrams, {len(typer.unit_to_type)} units -> {out_path}")
    return typer


if __name__ == "__main__":
    t = train_and_save_default()
    for span in (["16", "гб"], ["55", "дюймов"], ["2", "кг"], ["5"], ["wi-fi"]):
        print(span, "->", t.predict(span))
