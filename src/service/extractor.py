"""Hybrid QueryEntityExtractor: dictionary (rapidfuzz) + CRF NER → JSON."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ner.labeling import (
    WeakLabeler,
    bio_to_entities,
    entities_to_structured,
    tokenize,
    _normalize,
)
from src.ner.model_crf import CRFNerModel


class QueryEntityExtractor:
    """Fast hybrid extractor targeting <100ms per query."""

    def __init__(
        self,
        labeler: WeakLabeler,
        ner_model: Optional[CRFNerModel] = None,
        brand_classifier=None,
        category_classifier=None,
        fuzzy_threshold: int = 92,
        use_fuzzy: bool = True,
    ):
        self.labeler = labeler
        self.ner_model = ner_model
        self.brand_classifier = brand_classifier
        self.category_classifier = category_classifier
        self.fuzzy_threshold = fuzzy_threshold
        self.use_fuzzy = use_fuzzy
        # Prefer longer brands; drop ultra-short noisy keys except whitelist
        short_ok = {"lg", "hp", "jbl", "bq", "tcl", "msi", "bbk", "aoc"}
        self._brand_list = sorted(
            [
                k
                for k in labeler.brand_canonical.keys()
                if len(k) >= 3 or k in short_ok
            ],
            key=len,
            reverse=True,
        )
        self._category_list = sorted(labeler.category_canonical.keys(), key=len, reverse=True)

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path | str = "artifacts",
        models_dir: Path | str = "models",
    ) -> "QueryEntityExtractor":
        artifacts_dir = Path(artifacts_dir)
        models_dir = Path(models_dir)
        labeler = WeakLabeler.from_files(
            artifacts_dir / "brands.txt",
            artifacts_dir / "categories.txt",
        )
        ner_path = models_dir / "ner_crf.pkl"
        ner = CRFNerModel.load(ner_path) if ner_path.exists() else None

        brand_clf = None
        cat_clf = None
        brand_path = models_dir / "brand_clf.joblib"
        cat_path = models_dir / "category_clf.joblib"
        if brand_path.exists():
            import joblib

            brand_clf = joblib.load(brand_path)
        if cat_path.exists():
            import joblib

            cat_clf = joblib.load(cat_path)

        return cls(labeler=labeler, ner_model=ner, brand_classifier=brand_clf, category_classifier=cat_clf)

    def extract(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return {
                "query": query,
                "entities": [],
                "brand": None,
                "category": None,
                "attributes": {},
                "latency_ms": 0.0,
                "source": "empty",
            }

        # Pass 1: dictionary weak labels (exact match — very fast)
        dict_tags = self.labeler.label_query(query)
        entities = bio_to_entities(dict_tags, query=query)

        # Pass 2: CRF fills gaps / refines
        if self.ner_model is not None:
            ml_tags = self.ner_model.predict_query(query)
            ml_entities = bio_to_entities(ml_tags, query=query)
            entities = self._merge_entities(entities, ml_entities)

        # Pass 3: fuzzy brand/category if still missing
        structured = entities_to_structured(entities, self.labeler)
        if self.use_fuzzy and (structured["brand"] is None or structured["category"] is None):
            fuzzy = self._fuzzy_match(query)
            if structured["brand"] is None and fuzzy.get("brand"):
                structured["brand"] = fuzzy["brand"]
                entities.append(
                    {
                        "text": fuzzy["brand_match"],
                        "label": "BRAND",
                        "span": fuzzy.get("brand_span") or [0, 0],
                    }
                )
            if structured["category"] is None and fuzzy.get("category"):
                structured["category"] = fuzzy["category"]
                entities.append(
                    {
                        "text": fuzzy["category_match"],
                        "label": "CATEGORY",
                        "span": fuzzy.get("category_span") or [0, 0],
                    }
                )

        # Pass 4: ML classifiers as soft fallback
        if structured["brand"] is None and self.brand_classifier is not None:
            try:
                pred = self.brand_classifier.predict([query])[0]
                if pred and pred != "__UNK__":
                    structured["brand"] = pred
            except Exception:
                pass
        if structured["category"] is None and self.category_classifier is not None:
            try:
                pred = self.category_classifier.predict([query])[0]
                if pred and pred != "__UNK__":
                    structured["category"] = pred
            except Exception:
                pass

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query": query,
            "entities": entities,
            "brand": structured["brand"],
            "category": structured["category"],
            "attributes": structured["attributes"],
            "latency_ms": round(latency_ms, 3),
        }

    def _merge_entities(self, primary: List[Dict], secondary: List[Dict]) -> List[Dict]:
        """Prefer dictionary spans; add non-overlapping ML entities."""
        result = list(primary)
        occupied = [tuple(e["span"]) for e in primary if "span" in e]

        def overlaps(a, b):
            return not (a[1] <= b[0] or b[1] <= a[0])

        for ent in secondary:
            span = ent.get("span")
            if span is None:
                result.append(ent)
                continue
            if any(overlaps(span, o) for o in occupied):
                continue
            result.append(ent)
            occupied.append(tuple(span))
        # stable order by span start
        result.sort(key=lambda e: e.get("span", [0])[0])
        return result

    def _fuzzy_match(self, query: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        q = _normalize(query)
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return self._substring_fallback(q, query)

        # Brands: extract candidate n-grams from query tokens
        tokens = [t for t, _, _ in tokenize(query)]
        candidates = set()
        lower = [_normalize(t) for t in tokens]
        for n in (1, 2, 3):
            for i in range(len(lower) - n + 1):
                candidates.add(" ".join(lower[i : i + n]))

        if candidates and self._brand_list:
            # Cap pool + candidates for <100ms latency
            brand_pool = self._brand_list[:4000]
            best = None
            best_score = 0
            best_cand = None
            for cand in sorted(candidates, key=len, reverse=True)[:40]:
                if len(cand) < 3:
                    continue
                match = process.extractOne(
                    cand,
                    brand_pool,
                    scorer=fuzz.ratio,
                    score_cutoff=self.fuzzy_threshold,
                )
                if match and match[1] > best_score:
                    best_score = match[1]
                    best = match[0]
                    best_cand = cand
            if best and best_score >= self.fuzzy_threshold:
                out["brand"] = self.labeler.brand_canonical.get(best, best)
                out["brand_match"] = best_cand
                m = re.search(re.escape(best_cand), q)
                if m:
                    out["brand_span"] = [m.start(), m.end()]

        if self._category_list:
            cat_pool = self._category_list[:3000]
            match = process.extractOne(
                q,
                cat_pool,
                scorer=fuzz.partial_ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            # Prefer token-level category match
            for tok in lower:
                if tok in self.labeler.categories:
                    out["category"] = self.labeler.category_canonical[tok]
                    out["category_match"] = tok
                    break
            if "category" not in out and match:
                out["category"] = self.labeler.category_canonical.get(match[0], match[0])
                out["category_match"] = match[0]
        return out

    def _substring_fallback(self, q_norm: str, query: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for brand in self._brand_list[:5000]:
            if brand in q_norm:
                out["brand"] = self.labeler.brand_canonical[brand]
                out["brand_match"] = brand
                break
        for cat in self._category_list[:2000]:
            if cat in q_norm:
                out["category"] = self.labeler.category_canonical[cat]
                out["category_match"] = cat
                break
        return out
