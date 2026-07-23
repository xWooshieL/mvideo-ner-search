"""
Полный пайплайн предобработки query для NER / классификаторов / weak labels.

Что делает предобработка (и что — нет) — см. notebooks/preprocessing/README.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Сиды: то, что правила/словари уже знают из проекта
# ---------------------------------------------------------------------------

# Линейки / модели, которые часто остаются O (хвост после бренда)
MODEL_SEEDS: Set[str] = {
    # Logitech / gaming headsets & mice
    "g pro", "g pro x", "g pro x se", "g pro x 2", "g pro wireless",
    "g305", "g502", "g915", "g733", "g435",
    # Dyson
    "v8", "v10", "v11", "v12", "v15", "airwrap", "supersonic",
    # Apple lines (многословные / код поколения — без голого "pro"/"air")
    "pro max", "air pods", "macbook air", "macbook pro",
    "m1", "m2", "m3", "m4",
    # Samsung
    "galaxy s", "galaxy a", "galaxy z", "galaxy watch", "galaxy buds",
    # Xiaomi / Redmi / Poco
    "poco x", "poco f", "redmi note",
    # Sony / PS
    "playstation 5", "ps5", "ps4", "dualsense",
    # TV tech (устойчивые маркировки)
    "neo qled", "qled", "oled",
}

# Мультисловные бренды, где первое слово похоже на цвет/прилагательное
# («Красный Октябрь», «Белый кот» …) — не трогать как ATTR-color
PROTECTED_BRAND_SEEDS: Set[str] = {
    "красный октябрь",
    "белая роза",
    "черный жемчуг",  # косметика / встретится реже, но принцип тот же
    "золотой стандарт",
}

# Единицы, которые стоит отклеивать от числа: 128гб → 128 гб
UNIT_SUFFIXES = (
    "gb", "гб", "tb", "тб", "mb", "мб",
    "mm", "мм", "cm", "см",
    "w", "вт", "kw", "квт",
    "mah", "мач", "маh",
    "hz", "гц", "mhz", "мгц",
    "kg", "кг", "g", "г",
    "l", "л", "ml", "мл",
    "v", "в",
    "inch", "дюйм",
)

_UNIT_ALT = "|".join(sorted(UNIT_SUFFIXES, key=len, reverse=True))
_GLUED_UNIT_RE = re.compile(
    rf"(?P<num>\d+(?:[.,]\d+)?)(?P<unit>(?:{_UNIT_ALT}))\b",
    re.IGNORECASE,
)
_GLUED_GENERIC_RE = re.compile(r"(?P<num>\d+)(?P<alpha>[A-Za-zА-Яа-яЁё]{1,4})\b")

# g-pro / g_pro → g pro; схлопывание пробелов
_SEP_RE = re.compile(r"[_\u00A0]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_DASH_BETWEEN_LATIN_RE = re.compile(
    r"(?<=[A-Za-z])\s*[-–—]\s*(?=[A-Za-z])"
)


def _norm_key(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = _SEP_RE.sub(" ", s)
    s = _DASH_BETWEEN_LATIN_RE.sub(" ", s)
    s = re.sub(r"[^\w\s\.\-]", " ", s, flags=re.UNICODE)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def split_glued_alnum(text: str) -> str:
    """128гб → 128 гб; 220в → 220 в (сначала известные единицы, потом generic)."""
    if not text:
        return text

    def _unit_sub(m: re.Match) -> str:
        return f"{m.group('num')} {m.group('unit')}"

    text = _GLUED_UNIT_RE.sub(_unit_sub, text)
    text = _GLUED_GENERIC_RE.sub(r"\g<num> \g<alpha>", text)
    return text


def basic_clean(text: str, *, lowercase: bool = False) -> str:
    """Базовая чистка без агрессивного lower (регистр нужен для контекста)."""
    if text is None:
        return ""
    text = str(text).replace("\u00A0", " ").strip()
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = _SEP_RE.sub(" ", text)
    # унификация кавычек/крестиков размеров, но не трогаем букву x в "g pro x"
    text = text.replace("×", "x").replace("Х", "x")
    text = split_glued_alnum(text)
    text = _DASH_BETWEEN_LATIN_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Токены с char-spans на уже очищенном тексте."""
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(
        r"[A-Za-zА-Яа-яЁё0-9]+(?:[.\-/][A-Za-zА-Яа-яЁё0-9]+)*|[^\s]",
        text,
    ):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def load_phrase_list(path: Path | str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def save_phrase_list(phrases: Iterable[str], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted({_norm_key(x) for x in phrases if _norm_key(x)}, key=lambda s: (-len(s), s))
    p.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")
    return p


def build_model_lexicon_from_titles(
    titles: Sequence[str],
    brands: Sequence[str] | None = None,
    *,
    min_count: int = 3,
    max_phrase_tokens: int = 4,
) -> Dict[str, int]:
    """
    Майнинг кандидатов MODEL из названий SKU.

    Эвристика: после токена-бренда берём 1..max_phrase_tokens следующих
    коротких латинских/цифровых токенов (g pro x se, v15, air …).
    """
    brand_keys = {_norm_key(b) for b in (brands or []) if len(_norm_key(b)) >= 2}
    # частые однотокенные бренды для якоря
    counts: Dict[str, int] = {}

    lat_num = re.compile(r"^[a-z0-9]{1,12}$")

    for title in titles:
        t = _norm_key(basic_clean(str(title), lowercase=True))
        if not t:
            continue
        toks = t.split()
        for i, tok in enumerate(toks):
            if brand_keys and tok not in brand_keys:
                # якорь: logitech/sony/... или сам сид "g" после brand-like
                continue
            # фразы после бренда
            buf: List[str] = []
            for j in range(i + 1, min(len(toks), i + 1 + max_phrase_tokens)):
                w = toks[j]
                if not lat_num.match(w):
                    break
                # отсекаем слишком общие одиночные слова без цифр, кроме известных
                buf.append(w)
                phrase = " ".join(buf)
                if len(buf) == 1 and w in {"for", "with", "and", "the"}:
                    break
                counts[phrase] = counts.get(phrase, 0) + 1

    # также: фразы, начинающиеся на g/v/ps + короткий хвост (без обязательного бренда)
    lead_re = re.compile(r"\b((?:g|v|ps|air|galaxy|redmi|poco|iphone|macbook)(?:\s+[a-z0-9]{1,10}){0,3})\b")
    for title in titles:
        t = _norm_key(basic_clean(str(title), lowercase=True))
        for m in lead_re.finditer(t):
            phrase = m.group(1).strip()
            if phrase in {"g", "v", "air", "galaxy"}:  # слишком коротко одно
                continue
            counts[phrase] = counts.get(phrase, 0) + 1

    def _ok_model_phrase(p: str) -> bool:
        toks = p.split()
        if not toks:
            return False
        # чистые числа / одна цифра — не MODEL (это ATTR)
        if all(t.isdigit() or t.replace(".", "").isdigit() for t in toks):
            return False
        if len(toks) == 1 and len(toks[0]) <= 2:
            return False
        # отсев явного мусора артикулов (слишком много цифр в «слове»)
        if any(sum(ch.isdigit() for ch in t) >= 3 and sum(ch.isalpha() for ch in t) >= 2 for t in toks):
            return False
        return True

    return {p: c for p, c in counts.items() if c >= min_count and _ok_model_phrase(p)}


@dataclass
class PreprocessResult:
    """Результат предобработки одного запроса — контракт для других ноутбуков."""

    original: str
    text: str  # очищенный текст (регистр частично сохранён)
    text_norm: str  # lower для матчинга
    tokens: List[str] = field(default_factory=list)
    token_spans: List[Tuple[int, int]] = field(default_factory=list)
    # подсказки сущностей (ещё не BIO-модель, а словарь/эвристики)
    model_spans: List[Dict] = field(default_factory=list)  # {text, start_tok, end_tok}
    protected_spans: List[Dict] = field(default_factory=list)  # бренды с «ложным» цветом
    titlecase_hints: List[str] = field(default_factory=list)  # «Красный Октябрь»
    steps_applied: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "original": self.original,
            "text": self.text,
            "text_norm": self.text_norm,
            "tokens": self.tokens,
            "model_spans": self.model_spans,
            "protected_spans": self.protected_spans,
            "titlecase_hints": self.titlecase_hints,
            "steps_applied": self.steps_applied,
        }


class QueryPreprocessor:
    """
    Единая точка входа предобработки.

    Пример::

        pp = QueryPreprocessor.from_artifacts("artifacts")
        r = pp(query)
        # r.text_norm, r.model_spans, ...
    """

    def __init__(
        self,
        model_phrases: Optional[Iterable[str]] = None,
        protected_brands: Optional[Iterable[str]] = None,
    ):
        models = set(MODEL_SEEDS)
        if model_phrases:
            models.update(_norm_key(x) for x in model_phrases if _norm_key(x))
        prots = set(PROTECTED_BRAND_SEEDS)
        if protected_brands:
            prots.update(_norm_key(x) for x in protected_brands if _norm_key(x))

        self.model_phrases: List[List[str]] = self._compile_phrases(models)
        self.protected_brands: List[List[str]] = self._compile_phrases(prots)
        self._model_set = models
        self._protected_set = prots

    @staticmethod
    def _compile_phrases(phrases: Set[str]) -> List[List[str]]:
        seq = [p.split() for p in phrases if p]
        seq.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
        return seq

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path | str = "artifacts") -> "QueryPreprocessor":
        from src.data_utils import ARTIFACTS_DIR, model_phrases_path, protected_brands_path

        d = Path(artifacts_dir)

        def _p(name: str, fallback):
            nested = d / "dicts" / name
            flat = d / name
            if nested.exists():
                return nested
            if flat.exists():
                return flat
            if d.resolve() == ARTIFACTS_DIR.resolve():
                return fallback()
            return flat

        return cls(
            model_phrases=load_phrase_list(_p("model_phrases.txt", model_phrases_path)),
            protected_brands=load_phrase_list(_p("protected_brands.txt", protected_brands_path)),
        )

    def __call__(self, query: str) -> PreprocessResult:
        return self.preprocess(query)

    def preprocess(self, query: str) -> PreprocessResult:
        original = query if query is not None else ""
        steps: List[str] = []

        text = basic_clean(original, lowercase=False)
        steps.append("basic_clean+split_glued_units")

        # title-case multiword hints (контекст «Красный Октябрь»)
        titlecase_hints = self._find_titlecase_phrases(original)
        if titlecase_hints:
            steps.append("titlecase_hints")

        text_norm = _norm_key(text)
        steps.append("normalize_key")

        spans = tokenize_with_spans(text_norm)
        tokens = [t for t, _, _ in spans]
        lower_toks = [_norm_key(t) for t in tokens]

        protected = self._match_phrases(lower_toks, self.protected_brands, "PROTECTED_BRAND")
        models = self._match_phrases(lower_toks, self.model_phrases, "MODEL")

        # если titlecase hint совпал с protected — усиливаем
        for hint in titlecase_hints:
            hn = _norm_key(hint)
            if hn in self._protected_set and not any(p["text"] == hn for p in protected):
                # попробуем найти в токенах
                ht = hn.split()
                for i in range(len(lower_toks) - len(ht) + 1):
                    if lower_toks[i : i + len(ht)] == ht:
                        protected.append(
                            {"text": hn, "start_tok": i, "end_tok": i + len(ht), "label": "PROTECTED_BRAND"}
                        )

        if models:
            steps.append("model_lexicon_match")
        if protected:
            steps.append("protected_brand_match")

        return PreprocessResult(
            original=original,
            text=text,
            text_norm=text_norm,
            tokens=tokens,
            token_spans=[(s, e) for _, s, e in spans],
            model_spans=models,
            protected_spans=protected,
            titlecase_hints=titlecase_hints,
            steps_applied=steps,
        )

    @staticmethod
    def _find_titlecase_phrases(text: str) -> List[str]:
        """Ищем последовательности Capitalized слов (кириллица/латиница)."""
        if not text:
            return []
        # Красный Октябрь / White Knight
        pat = re.compile(
            r"\b([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)+)\b"
        )
        return [m.group(1) for m in pat.finditer(text)]

    @staticmethod
    def _match_phrases(
        lower_toks: List[str],
        phrases: List[List[str]],
        label: str,
    ) -> List[Dict]:
        n = len(lower_toks)
        used = [False] * n
        out: List[Dict] = []
        for phrase in phrases:
            plen = len(phrase)
            if plen == 0 or plen > n:
                continue
            for i in range(n - plen + 1):
                if any(used[i + k] for k in range(plen)):
                    continue
                if lower_toks[i : i + plen] == phrase:
                    for k in range(plen):
                        used[i + k] = True
                    out.append(
                        {
                            "text": " ".join(phrase),
                            "start_tok": i,
                            "end_tok": i + plen,
                            "label": label,
                        }
                    )
        return out

    def merge_bio_hints(
        self,
        token_tags: Sequence[Tuple[str, str]],
        result: PreprocessResult,
    ) -> List[Tuple[str, str]]:
        """
        Накладывает MODEL / снимает ложный цвет на protected brand
        поверх уже полученной BIO-разметки (token, tag).
        """
        tags = [t for _, t in token_tags]
        toks = [t for t, _ in token_tags]
        if len(toks) != len(result.tokens) and result.tokens:
            # пересоберём на result.tokens если длины разъехались
            toks = list(result.tokens)
            tags = ["O"] * len(toks)

        # 1) protected: затираем ATTR на этих токенах → BRAND
        for sp in result.protected_spans:
            a, b = sp["start_tok"], sp["end_tok"]
            for i in range(a, min(b, len(tags))):
                tags[i] = "B-BRAND" if i == a else "I-BRAND"

        # 2) model spans — только на O
        for sp in result.model_spans:
            a, b = sp["start_tok"], sp["end_tok"]
            if any(tags[i] != "O" for i in range(a, min(b, len(tags)))):
                continue
            for i in range(a, min(b, len(tags))):
                tags[i] = "B-MODEL" if i == a else "I-MODEL"

        return list(zip(toks, tags))
