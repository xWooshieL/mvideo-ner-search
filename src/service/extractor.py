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


_BRAND_REJECT = {"NO_BRAND", "UNKNOWN", "__UNK__", "null", "none", ""}


class QueryEntityExtractor:
    """Fast hybrid extractor targeting <100ms per query.

    Cascade: rules (labeling.py) → CRF NER → fuzzy → brand clf fallback → ATTR typer.
    """

    def __init__(
        self,
        labeler: WeakLabeler,
        ner_model: Optional[CRFNerModel] = None,
        brand_classifier=None,
        category_classifier=None,
        attr_type_model=None,
        attr_type_policy: Optional[Dict[str, Any]] = None,
        model_phrases: Optional[set] = None,
        fuzzy_threshold: int = 92,
        use_fuzzy: bool = True,
        use_attr_clf: bool = True,
        spell_fixer=None,
    ):
        self.labeler = labeler
        self.ner_model = ner_model
        self.brand_classifier = brand_classifier
        self.category_classifier = category_classifier
        self.attr_type_model = attr_type_model
        self.attr_type_policy = attr_type_policy or {}
        self.model_phrases = model_phrases or set()
        self.fuzzy_threshold = fuzzy_threshold
        self.use_fuzzy = use_fuzzy
        self.use_attr_clf = use_attr_clf and attr_type_model is not None
        self.spell_fixer = spell_fixer
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
        from src.data_utils import ARTIFACTS_DIR, brands_path, categories_path, model_phrases_path

        artifacts_dir = Path(artifacts_dir)
        models_dir = Path(models_dir)

        def _dict(name: str) -> Path:
            nested = artifacts_dir / "dicts" / name
            flat = artifacts_dir / name
            if nested.exists():
                return nested
            if flat.exists():
                return flat
            # default project layout helpers
            if artifacts_dir.resolve() == ARTIFACTS_DIR.resolve():
                return {"brands.txt": brands_path, "categories.txt": categories_path, "model_phrases.txt": model_phrases_path}[name]()
            return flat

        bp, cp, mp = _dict("brands.txt"), _dict("categories.txt"), _dict("model_phrases.txt")
        labeler = WeakLabeler.from_files(
            bp,
            cp,
            models_path=mp if mp.exists() else None,
        )
        ner_path = models_dir / "ner_crf.pkl"
        ner = CRFNerModel.load(ner_path) if ner_path.exists() else None

        brand_clf = None
        cat_clf = None
        attr_pipe = None
        attr_policy: Dict[str, Any] = {}
        model_phrases: set = set()
        import joblib

        brand_path = models_dir / "brand_clf.joblib"
        cat_path = models_dir / "category_clf.joblib"
        attr_path = models_dir / "attr_type_clf.joblib"
        if brand_path.exists():
            brand_clf = joblib.load(brand_path)
        if cat_path.exists():
            cat_clf = joblib.load(cat_path)
        if attr_path.exists():
            attr_pipe = joblib.load(attr_path)
            from src.ner.attr_type_clf import load_policy

            attr_policy = load_policy()
        if mp.exists():
            model_phrases = {
                ln.strip().lower()
                for ln in mp.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            }

        spell_fixer = None
        try:
            from src.preprocessing.spellfix import SpellFixer

            spell_fixer = SpellFixer.from_artifacts(artifacts_dir)
        except Exception:
            spell_fixer = None

        return cls(
            labeler=labeler,
            ner_model=ner,
            brand_classifier=brand_clf,
            category_classifier=cat_clf,
            attr_type_model=attr_pipe,
            attr_type_policy=attr_policy,
            model_phrases=model_phrases,
            spell_fixer=spell_fixer,
        )

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

        query_raw = query
        spell_changes: List[Dict[str, Any]] = []
        if self.spell_fixer is not None:
            query, spell_changes = self.spell_fixer.fix_query(query)

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
                if fuzzy.get("brand_span"):
                    entities.append(
                        {
                            "text": fuzzy["brand_match"],
                            "label": "BRAND",
                            "span": fuzzy["brand_span"],
                        }
                    )
            if structured["category"] is None and fuzzy.get("category"):
                structured["category"] = fuzzy["category"]
                if fuzzy.get("category_span"):
                    entities.append(
                        {
                            "text": fuzzy["category_match"],
                            "label": "CATEGORY",
                            "span": fuzzy["category_span"],
                        }
                    )

        # Pass 4: ML classifiers as soft fallback (never invent brand-like categories)
        if structured["brand"] is None and self.brand_classifier is not None:
            try:
                pred = str(self.brand_classifier.predict([query])[0])
                if pred and pred not in _BRAND_REJECT:
                    structured["brand"] = pred
            except Exception:
                pass
        if structured["category"] is None and self.category_classifier is not None:
            try:
                pred = self.category_classifier.predict([query])[0]
                if pred and pred != "__UNK__":
                    pn = _normalize(str(pred))
                    if pn not in self.labeler.brands and pn not in self.labeler.brand_canonical:
                        structured["category"] = pred
            except Exception:
                pass

        # Pass 5: ATTR typer (ML clf with reject; falls back to regex teacher)
        attributes = structured["attributes"]
        if self.use_attr_clf:
            attributes = self._type_attributes(
                entities,
                brand=structured.get("brand") or "",
                category=structured.get("category") or "",
                query=query,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        out = {
            "query": query,
            "entities": entities,
            "brand": structured["brand"],
            "category": structured["category"],
            "model": structured.get("model"),
            "attributes": attributes,
            "latency_ms": round(latency_ms, 3),
        }
        if spell_changes:
            out["query_raw"] = query_raw
            out["spell_fixes"] = spell_changes
        return out

    def extract_debug(self, query: str) -> Dict[str, Any]:
        """Same as extract, plus BIO pipelines for UI debugging."""
        t0 = time.perf_counter()
        query = (query or "").strip()
        base = self.extract(query)
        q = base.get("query") or query
        dict_bio = self.labeler.label_query(q) if q else []
        crf_bio = self.ner_model.predict_query(q) if (q and self.ner_model) else []
        base["debug"] = {
            "dict_bio": [{"token": t, "tag": tag} for t, tag in dict_bio],
            "crf_bio": [{"token": t, "tag": tag} for t, tag in crf_bio],
            "has_crf": self.ner_model is not None,
            "has_brand_clf": self.brand_classifier is not None,
            "has_category_clf": self.category_classifier is not None,
            "has_attr_clf": self.use_attr_clf,
            "has_spellfix": self.spell_fixer is not None,
            "n_brands_dict": len(self.labeler.brands),
            "n_categories_dict": len(self.labeler.categories),
            "n_models_dict": len(self.labeler.models),
            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
        return base

    def _type_attributes(
        self,
        entities: List[Dict],
        *,
        brand: str,
        category: str,
        query: str,
    ) -> Dict[str, Any]:
        """Типизация ATTR-спанов через attr_type_clf (+ reject). Teacher — fallback."""
        from src.ner.attr_type_clf import LABEL_UNKNOWN, looks_like_model
        from src.ner.labeling import _guess_attr_type

        attr_ents = [e for e in entities if e.get("label") == "ATTR" and (e.get("text") or "").strip()]
        if not attr_ents:
            return {}

        # mask all ATTR spans for query_masked feature
        chars = list(query)
        for e in sorted(attr_ents, key=lambda x: -(x.get("span") or [0, 0])[0]):
            span = e.get("span")
            if not span:
                continue
            a, b = span
            chars[a:b] = list("<ATTR>")
        query_masked = re.sub(r"\s+", " ", "".join(chars)).strip() or query

        tau = float(self.attr_type_policy.get("min_confidence", 0.55))
        reject = self.attr_type_policy.get("reject_label", LABEL_UNKNOWN)
        pipe = self.attr_type_model
        out: Dict[str, List[str]] = {}

        for e in attr_ents:
            span_text = (e.get("text") or "").strip()
            if looks_like_model(span_text, self.model_phrases):
                label = reject
            else:
                try:
                    from src.ner.attr_type_clf import _row

                    row = _row(span_text, brand or "", category or "", query_masked)
                    raw = str(pipe.predict(row)[0])
                    conf = 1.0
                    if hasattr(pipe, "predict_proba"):
                        proba = pipe.predict_proba(row)[0]
                        conf = float(proba.max())
                        raw = str(pipe.classes_[int(proba.argmax())])
                    label = reject if conf < tau else raw
                except Exception:
                    label = _guess_attr_type(span_text)
            if label in {LABEL_UNKNOWN, "unknown", "other"}:
                # keep teacher type for demos when clf rejects / other
                teacher = _guess_attr_type(span_text)
                label = teacher if teacher != "other" else label
            out.setdefault(label, []).append(span_text)
            e["attr_type"] = label

        return {k: v[0] if len(v) == 1 else v for k, v in out.items()}

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
            # Prefer exact token / phrase match only (avoid fuzzy brand bleed)
            for n in (3, 2, 1):
                for i in range(len(lower) - n + 1):
                    phrase = " ".join(lower[i : i + n])
                    if phrase in self.labeler.categories:
                        out["category"] = self.labeler.category_canonical[phrase]
                        out["category_match"] = phrase
                        # approximate span via normalized search
                        m = re.search(re.escape(phrase), q)
                        if m:
                            out["category_span"] = [m.start(), m.end()]
                        return out
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
