"""Weak supervision: BIO labeling for search queries from brand/category dictionaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Attribute regex patterns (value capture groups → ATTR)
ATTR_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\d+)\s*(?:gb|гб|Gb|GB)\b", re.I), "memory"),
    (re.compile(r"\b(\d+)\s*(?:tb|тб|Tb|TB)\b", re.I), "storage"),
    (re.compile(r"\b(\d+)\s*(?:мм|mm|см|cm|дюйм(?:а|ов)?|\"|'')\b", re.I), "size"),
    (re.compile(r"\b(\d{3,4})\s*[xх×]\s*(\d{3,4})(?:\s*[xх×]\s*(\d{3,4}))?\b", re.I), "dimensions"),
    (re.compile(r"\b(\d+)\s*(?:вт|w|Вт)\b", re.I), "power"),
    (re.compile(r"\b(4k|8k|uhd|full\s*hd|fhd|hd)\b", re.I), "resolution"),
    (re.compile(r"\b(wi[- ]?fi|bluetooth|nfc|5g|4g|lte)\b", re.I), "connectivity"),
]

COLORS = {
    "белый", "белая", "белое", "белые",
    "черный", "чёрный", "черная", "чёрная", "черное", "чёрное",
    "серый", "серая", "серое",
    "красный", "красная", "красное",
    "синий", "синяя", "синее",
    "зеленый", "зелёный", "зеленая", "зелёная",
    "золотой", "золотая", "золото",
    "серебристый", "серебряный", "серебро",
    "розовый", "розовая",
    "голубой", "голубая",
    "фиолетовый", "фиолетовая",
    "оранжевый", "оранжевая",
    "коричневый", "коричневая",
    "бежевый", "бежевая",
    "титановый", "графитовый", "space gray", "midnight", "starlight",
}

# Product-line aliases → canonical brand (help weak labels for "iphone", "galaxy", …)
BRAND_ALIASES = {
    "iphone": "Apple",
    "айфон": "Apple",
    "айфоны": "Apple",
    "macbook": "Apple",
    "макбук": "Apple",
    "ipad": "Apple",
    "айпад": "Apple",
    "airpods": "Apple",
    "galaxy": "Samsung",
    "редми": "Xiaomi",
    "redmi": "Xiaomi",
    "poco": "Xiaomi",
    "honor": "HONOR",
    "хуавей": "HUAWEI",
    "huawei": "HUAWEI",
}

# Common Russian e-commerce category keywords (seed + will merge with data-driven)
CATEGORY_SEEDS = {
    "смартфон", "смартфоны", "телефон", "телефоны",
    "ноутбук", "ноутбуки", "планшет", "планшеты",
    "телевизор", "телевизоры", "наушники", "колонка", "колонки",
    "пылесос", "робот", "холодильник", "стиральная", "машинка",
    "микроволновка", "микроволновая", "духовка", "духовой", "шкаф",
    "плита", "варочная", "панель", "посудомойка", "посудомоечная",
    "кондиционер", "фен", "утюг", "блендер", "миксер", "чайник",
    "электрочайник", "мультиварка", "кофемашина", "кофемолка",
    "монитор", "клавиатура", "мышь", "принтер", "роутер",
    "фотоаппарат", "камера", "видеокарта", "процессор", "ssd",
    "hdd", "память", "материнская", "блок", "питания",
    "игровой", "консоль", "приставка", "smart", "часы", "браслет",
    "сушильная", "вытяжка", "водонагреватель", "обогреватель",
}


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Whitespace + punctuation-aware tokenization with char spans."""
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+(?:[.\-/][A-Za-zА-Яа-яЁё0-9]+)*|[^\s]", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace("ё", "е"))


@dataclass
class WeakLabeler:
    brands: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    colors: Set[str] = field(default_factory=lambda: set(COLORS))
    # lower -> canonical display form
    brand_canonical: Dict[str, str] = field(default_factory=dict)
    category_canonical: Dict[str, str] = field(default_factory=dict)
    # sorted longest-first phrase lists (token sequences)
    _brand_phrases: List[List[str]] = field(default_factory=list, repr=False)
    _category_phrases: List[List[str]] = field(default_factory=list, repr=False)

    @classmethod
    def from_files(
        cls,
        brands_path: Path | str,
        categories_path: Path | str,
    ) -> "WeakLabeler":
        brands = _load_lines(brands_path)
        categories = _load_lines(categories_path)
        return cls.from_iterables(brands, categories)

    @classmethod
    def from_iterables(
        cls,
        brands: Iterable[str],
        categories: Iterable[str] | None = None,
    ) -> "WeakLabeler":
        brand_canonical: Dict[str, str] = {}
        for b in brands:
            b = (b or "").strip()
            if len(b) < 2:
                continue
            brand_canonical[_normalize(b)] = b
        # aliases expand brand dictionary
        for alias, canon in BRAND_ALIASES.items():
            brand_canonical[_normalize(alias)] = canon
        cat_canonical: Dict[str, str] = {}
        brand_keys = set(brand_canonical.keys())
        for c in list(categories or []) + list(CATEGORY_SEEDS):
            c = (c or "").strip()
            if len(c) < 2:
                continue
            cn = _normalize(c)
            # do not treat known brands / aliases as categories
            if cn in brand_keys or cn in BRAND_ALIASES:
                continue
            cat_canonical[cn] = c
        obj = cls(
            brands=set(brand_canonical.keys()),
            categories=set(cat_canonical.keys()),
            brand_canonical=brand_canonical,
            category_canonical=cat_canonical,
        )
        obj._compile_phrases()
        return obj

    def _compile_phrases(self) -> None:
        brand_phrases = [p.split() for p in self.brands if p]
        cat_phrases = [p.split() for p in self.categories if p]
        brand_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
        cat_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
        self._brand_phrases = brand_phrases
        self._category_phrases = cat_phrases

    def label_query(self, query: str) -> List[Tuple[str, str]]:
        """Return list of (token, BIO-tag)."""
        tokens = tokenize(query)
        if not tokens:
            return []
        tags = ["O"] * len(tokens)
        lower_toks = [_normalize(t[0]) for t in tokens]

        # 1) Brands (longest match)
        self._apply_phrases(lower_toks, tags, self._brand_phrases, "BRAND")
        # 2) Categories
        self._apply_phrases(lower_toks, tags, self._category_phrases, "CATEGORY")
        # 3) Colors as ATTR
        for i, lt in enumerate(lower_toks):
            if tags[i] != "O":
                continue
            if lt in self.colors or lt.replace("ё", "е") in self.colors:
                tags[i] = "B-ATTR"
        # 4) Regex attributes on original query → map to tokens
        self._apply_attr_patterns(query, tokens, tags)

        return [(tokens[i][0], tags[i]) for i in range(len(tokens))]

    def _apply_phrases(
        self,
        lower_toks: List[str],
        tags: List[str],
        phrases: List[List[str]],
        label: str,
    ) -> None:
        n = len(lower_toks)
        for phrase in phrases:
            plen = len(phrase)
            if plen == 0 or plen > n:
                continue
            for i in range(n - plen + 1):
                if any(tags[i + k] != "O" for k in range(plen)):
                    continue
                if lower_toks[i : i + plen] == phrase:
                    tags[i] = f"B-{label}"
                    for k in range(1, plen):
                        tags[i + k] = f"I-{label}"

    def _apply_attr_patterns(
        self,
        query: str,
        tokens: List[Tuple[str, int, int]],
        tags: List[str],
    ) -> None:
        for pattern, _name in ATTR_PATTERNS:
            for m in pattern.finditer(query):
                start, end = m.start(), m.end()
                first = True
                for i, (_tok, s, e) in enumerate(tokens):
                    if e <= start or s >= end:
                        continue
                    if tags[i] != "O":
                        continue
                    tags[i] = "B-ATTR" if first else "I-ATTR"
                    first = False

    def label_dataset(
        self,
        queries: Sequence[str],
        min_entities: int = 0,
    ) -> List[List[Tuple[str, str]]]:
        labeled = []
        for q in queries:
            sent = self.label_query(q)
            if not sent:
                continue
            n_ent = sum(1 for _, t in sent if t.startswith("B-"))
            if n_ent >= min_entities:
                labeled.append(sent)
        return labeled


def bio_to_entities(
    tokens_tags: Sequence[Tuple[str, str]],
    query: Optional[str] = None,
) -> List[Dict]:
    """Convert BIO sequence to entity dicts with optional char spans."""
    entities: List[Dict] = []
    i = 0
    # rebuild spans if query provided
    if query is not None:
        tok_spans = tokenize(query)
    else:
        tok_spans = [(t, 0, 0) for t, _ in tokens_tags]

    while i < len(tokens_tags):
        tok, tag = tokens_tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < len(tokens_tags) and tokens_tags[j][1] == f"I-{label}":
                j += 1
            text = " ".join(tokens_tags[k][0] for k in range(i, j))
            span_start = tok_spans[i][1] if query is not None else None
            span_end = tok_spans[j - 1][2] if query is not None else None
            ent: Dict = {"text": text, "label": label}
            if span_start is not None:
                # Prefer exact substring from query
                ent["span"] = [span_start, span_end]
                ent["text"] = query[span_start:span_end]
            entities.append(ent)
            i = j
        else:
            i += 1
    return entities


def entities_to_structured(entities: List[Dict], labeler: Optional[WeakLabeler] = None) -> Dict:
    """Collapse entities into brand / category / attributes fields."""
    brand = None
    category = None
    attributes: Dict[str, List[str]] = {}
    for ent in entities:
        label = ent["label"]
        text = ent["text"]
        if label == "BRAND" and brand is None:
            key = _normalize(text)
            brand = labeler.brand_canonical.get(key, text) if labeler else text
        elif label == "CATEGORY" and category is None:
            key = _normalize(text)
            category = labeler.category_canonical.get(key, text) if labeler else text
        elif label == "ATTR":
            attr_type = _guess_attr_type(text)
            attributes.setdefault(attr_type, []).append(text)
    return {
        "brand": brand,
        "category": category,
        "attributes": {k: v[0] if len(v) == 1 else v for k, v in attributes.items()},
    }


def _guess_attr_type(text: str) -> str:
    t = text.lower().replace("ё", "е")
    if re.search(r"\d+\s*(gb|гб)", t):
        return "memory"
    if re.search(r"\d+\s*(tb|тб)", t):
        return "storage"
    if re.search(r"(мм|mm|см|cm|дюйм|\")", t):
        return "size"
    if re.search(r"[xх×]", t):
        return "dimensions"
    if re.search(r"(вт|w)\b", t):
        return "power"
    if t in COLORS or t.replace("ё", "е") in COLORS:
        return "color"
    if re.search(r"(4k|8k|uhd|fhd|hd|wi-?fi|bluetooth|nfc|5g|4g)", t):
        return "tech"
    return "other"


def _load_lines(path: Path | str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
