"""ATTR-span type classifier helpers (features + predict + prod policy)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.pipeline import _norm_key

LABEL_UNKNOWN = "UNKNOWN"
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = _ROOT / "models" / "attr_type_clf.joblib"
DEFAULT_POLICY = _ROOT / "artifacts" / "attr_type" / "inference_policy.json"

_MODELISH_RE = re.compile(r"^[a-z]{1,3}(?:\s+[a-z0-9]{1,8}){1,3}$")
_UNIT_HINT = re.compile(
    r"(gb|гб|mb|мб|tb|тб|kb|кб|вт|w|кг|kg|грамм|мм|mm|см|cm|дюйм|inch|л(?!\w)|мл|hz|гц|"
    r"(?:8|16|32|64|128|256|512|1024)\s*[гg]\b)",
    re.I,
)


class Col(BaseEstimator, TransformerMixin):
    """Выбрать колонку DataFrame / dict-rows как список строк для TfidfVectorizer."""

    def __init__(self, col: str):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.col].astype(str).tolist()
        return [x[self.col] for x in X]


def looks_like_model(span_text: str, model_phrases: Optional[set[str]] = None) -> bool:
    s = _norm_key(span_text)
    if re.search(r"\d", s) and _UNIT_HINT.search(s):
        return False
    if model_phrases and s in model_phrases:
        return True
    if _MODELISH_RE.fullmatch(s) and not re.search(r"\d", s):
        if any(x in s for x in ("pro", "air", "max", "tuf", "vivo", "plus")):
            return True
    return False


def _row(span_text: str, brand: str, category: str, query_masked: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "span_text": span_text,
                "context_text": f"{brand} {category}".strip(),
                "query_masked": query_masked or span_text,
                "brand": brand,
                "category": category,
            }
        ]
    )


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_POLICY
    if not p.exists():
        return {"min_confidence": 0.55, "reject_label": LABEL_UNKNOWN}
    return json.loads(p.read_text(encoding="utf-8"))


def predict_attr_type(
    span_text: str,
    *,
    brand: str = "",
    category: str = "",
    query_masked: str = "",
    model_path: str | Path | None = None,
    model_phrases: Optional[set[str]] = None,
    min_confidence: float | None = None,
    return_details: bool = False,
) -> str | dict[str, Any]:
    """Инференс типа ATTR-span для прод.

    modelish → UNKNOWN до clf.
    Если max proba < min_confidence → UNKNOWN (reject).
    """
    policy = load_policy()
    tau = policy.get("min_confidence", 0.55) if min_confidence is None else min_confidence
    reject = policy.get("reject_label", LABEL_UNKNOWN)

    if looks_like_model(span_text, model_phrases):
        out = {
            "label": reject,
            "raw_pred": reject,
            "confidence": 1.0,
            "rejected": True,
            "reason": "modelish_rule",
            "top": [(reject, 1.0)],
        }
        return out if return_details else out["label"]

    path = Path(model_path) if model_path else DEFAULT_MODEL
    pipe = joblib.load(path)
    row = _row(span_text, brand, category, query_masked)
    raw = str(pipe.predict(row)[0])
    conf = 1.0
    top: list[tuple[str, float]] = [(raw, 1.0)]
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(row)[0]
        classes = list(pipe.classes_)
        top = sorted(zip(classes, map(float, proba)), key=lambda x: -x[1])
        conf = float(top[0][1])
        raw = str(top[0][0])

    rejected = conf < float(tau)
    label = reject if rejected else raw
    out = {
        "label": label,
        "raw_pred": raw,
        "confidence": conf,
        "rejected": rejected,
        "reason": "low_confidence" if rejected else "clf",
        "top": top[:5],
        "min_confidence": float(tau),
    }
    return out if return_details else out["label"]
