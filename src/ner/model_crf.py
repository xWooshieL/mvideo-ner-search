"""sklearn-crfsuite NER model wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .features import sent2features, sent2labels, texts_to_feature_sents
from .labeling import tokenize


class CRFNerModel:
    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 80,
        all_possible_transitions: bool = True,
    ):
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.all_possible_transitions = all_possible_transitions
        self.crf = None

    def fit(self, train_sents: Sequence[Sequence[Tuple[str, str]]]):
        try:
            import sklearn_crfsuite
        except ImportError as e:
            raise ImportError(
                "Install sklearn-crfsuite: pip install sklearn-crfsuite"
            ) from e

        X = [sent2features(s) for s in train_sents]
        y = [sent2labels(s) for s in train_sents]
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=self.all_possible_transitions,
        )
        self.crf.fit(X, y)
        return self

    def predict(self, sents: Sequence[Sequence[Tuple[str, str]]]) -> List[List[str]]:
        if self.crf is None:
            raise RuntimeError("Model is not fitted")
        X = [sent2features(s) for s in sents]
        return self.crf.predict(X)

    def predict_tokens(self, token_lists: Sequence[Sequence[str]]) -> List[List[str]]:
        if self.crf is None:
            raise RuntimeError("Model is not fitted")
        X = texts_to_feature_sents(token_lists)
        return self.crf.predict(X)

    def predict_query(self, query: str) -> List[Tuple[str, str]]:
        toks = [t for t, _, _ in tokenize(query)]
        if not toks:
            return []
        tags = self.predict_tokens([toks])[0]
        return list(zip(toks, tags))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "crf": self.crf,
                    "c1": self.c1,
                    "c2": self.c2,
                    "max_iterations": self.max_iterations,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path | str) -> "CRFNerModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            c1=data.get("c1", 0.1),
            c2=data.get("c2", 0.1),
            max_iterations=data.get("max_iterations", 80),
        )
        obj.crf = data["crf"]
        return obj
