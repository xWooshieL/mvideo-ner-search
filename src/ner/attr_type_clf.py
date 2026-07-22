"""ATTR-span type classifier helpers (features + predict)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.pipeline import basic_clean, _norm_key

LABEL_UNKNOWN = "UNKNOWN"


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


_MODELISH_RE = re.compile(r"^[a-z]{1,3}(?:\s+[a-z0-9]{1,8}){1,3}$")
_UNIT_HINT = re.compile(
    r"(gb|гб|mb|мб|tb|тб|kb|кб|вт|w|кг|kg|г(?!\w)|мм|mm|см|cm|дюйм|inch|л(?!\w)|мл|hz|гц)",
    re.I,
)


def looks_like_model(span_text: str, model_phrases: Optional[set[str]] = None) -> bool:
    s = _norm_key(span_text)
    # число + единица измерения — это ATTR, не MODEL (даже если попало в model_phrases)
    if re.search(r"\d", s) and _UNIT_HINT.search(s):
        return False
    if model_phrases and s in model_phrases:
        return True
    if _MODELISH_RE.fullmatch(s) and not re.search(r"\d", s):
        if any(x in s for x in ("pro", "air", "max", "tuf", "vivo", "plus")):
            return True
    return False


def predict_attr_type(
    span_text: str,
    *,
    brand: str = "",
    category: str = "",
    query_masked: str = "",
    model_path: str | Path | None = None,
    model_phrases: Optional[set[str]] = None,
) -> str:
    """Инференс типа ATTR-span; modelish → UNKNOWN до clf."""
    if looks_like_model(span_text, model_phrases):
        return LABEL_UNKNOWN
    path = Path(model_path) if model_path else Path("models/attr_type_clf.joblib")
    pipe = joblib.load(path)
    row = pd.DataFrame(
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
    return str(pipe.predict(row)[0])
