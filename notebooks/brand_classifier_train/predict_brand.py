"""Быстрый тест brand-clf на своих запросах.

Пример:
    python notebooks/brand_classifier_train/predict_brand.py
    python notebooks/brand_classifier_train/predict_brand.py "айфон 15" "холодильник"
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import basic_clean, _norm_key  # noqa: E402

# предпочитаем word+char; fallback на общий brand_clf.joblib
_MODEL_CANDIDATES = [
    ROOT / "models" / "brand_clf__logreg_wordchar.joblib",
    ROOT / "artifacts" / "brand_clf" / "train_runs" / "logreg_wordchar.joblib",
    ROOT / "models" / "brand_clf.joblib",
]

_pipe = None


def _load_model():
    global _pipe
    if _pipe is not None:
        return _pipe
    for path in _MODEL_CANDIDATES:
        if path.exists():
            _pipe = joblib.load(path)
            _pipe._model_path = str(path)  # type: ignore[attr-defined]
            return _pipe
    raise FileNotFoundError(
        "Не найден brand_clf.joblib. Сначала прогони "
        "notebooks/brand_classifier_train/_run_01.py или 01_classifier_train.ipynb"
    )


def normalize_query(text: str) -> str:
    """Тот же preprocess, что при сборке silver / обучении."""
    return _norm_key(basic_clean(text, lowercase=False))


def predict_brand(text: str, *, return_proba: bool = False):
    """
    Предсказать класс бренда для произвольного запроса.

    Returns
    -------
    str | tuple[str, dict[str, float]]
        Метка класса (Apple / NO_BRAND / UNKNOWN / …).
        Если return_proba=True — ещё top вероятности по классам.
    """
    pipe = _load_model()
    q = normalize_query(text)
    pred = str(pipe.predict([q])[0])
    if not return_proba:
        return pred
    if not hasattr(pipe, "predict_proba"):
        return pred, {}
    proba = pipe.predict_proba([q])[0]
    classes = list(pipe.named_steps["clf"].classes_)
    ranking = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    return pred, {c: float(p) for c, p in ranking[:8]}


def main(argv: list[str] | None = None) -> None:
    pipe = _load_model()
    print(f"model: {getattr(pipe, '_model_path', '?')}")

    queries = (argv or sys.argv[1:]) or [
        "айфон 15",
        "холодильник",
        "samsung galaxy s24",
        "наушники",
        "стинол холодильник",
        "tuf gaming a15",
    ]
    for q in queries:
        label, top = predict_brand(q, return_proba=True)
        top_s = ", ".join(f"{c}={p:.2f}" for c, p in list(top.items())[:3])
        print(f"{q!r:40} -> {label:12}  [{top_s}]")


if __name__ == "__main__":
    main()
