"""Feature extraction for sklearn-crfsuite NER."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


def _word_shape(word: str) -> str:
    shape = []
    for ch in word:
        if ch.isdigit():
            shape.append("d")
        elif ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        else:
            shape.append(ch)
    # compress runs
    out = []
    prev = None
    for s in shape:
        if s != prev:
            out.append(s)
            prev = s
    return "".join(out)[:12]


def _token_features(token: str, i: int, tokens: Sequence[str]) -> Dict[str, object]:
    word = token
    lower = word.lower()
    feats: Dict[str, object] = {
        "bias": 1.0,
        "word.lower": lower,
        "word[-3:]": lower[-3:],
        "word[-2:]": lower[-2:],
        "word[:3]": lower[:3],
        "word[:2]": lower[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.has_digit": any(c.isdigit() for c in word),
        "word.isalpha": word.isalpha(),
        "word.len": min(len(word), 20),
        "word.shape": _word_shape(word),
        "word.has_latin": bool(re.search(r"[A-Za-z]", word)),
        "word.has_cyrillic": bool(re.search(r"[А-Яа-яЁё]", word)),
    }
    if i > 0:
        prev = tokens[i - 1]
        feats.update(
            {
                "-1:word.lower": prev.lower(),
                "-1:word.istitle": prev.istitle(),
                "-1:word.isupper": prev.isupper(),
                "-1:word.shape": _word_shape(prev),
            }
        )
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        feats.update(
            {
                "+1:word.lower": nxt.lower(),
                "+1:word.istitle": nxt.istitle(),
                "+1:word.isupper": nxt.isupper(),
                "+1:word.shape": _word_shape(nxt),
            }
        )
    else:
        feats["EOS"] = True
    if i > 1:
        feats["-2:word.lower"] = tokens[i - 2].lower()
    if i < len(tokens) - 2:
        feats["+2:word.lower"] = tokens[i + 2].lower()
    return feats


def sent2features(sent: Sequence[Tuple[str, str]]) -> List[Dict[str, object]]:
    tokens = [t for t, _ in sent]
    return [_token_features(tok, i, tokens) for i, tok in enumerate(tokens)]


def sent2labels(sent: Sequence[Tuple[str, str]]) -> List[str]:
    return [lab for _, lab in sent]


def sent2tokens(sent: Sequence[Tuple[str, str]]) -> List[str]:
    return [t for t, _ in sent]


def texts_to_feature_sents(
    tokenized: Sequence[Sequence[str]],
) -> List[List[Dict[str, object]]]:
    """Build features from plain token lists (no labels)."""
    result = []
    for tokens in tokenized:
        fake = [(t, "O") for t in tokens]
        result.append(sent2features(fake))
    return result
