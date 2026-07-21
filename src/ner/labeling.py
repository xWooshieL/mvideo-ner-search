"""Weak supervision: BIO labeling for search queries with Natasha Lemmatization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ИМПОРТ NATASHA ДЛЯ ЛЕММАТИЗАЦИИ
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Segmenter,
)

segmenter = Segmenter()
morph_vocab = MorphVocab()
# морфологический теггер нужен, чтобы token.lemmatize() имел pos/feats
_emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(_emb)

NUM = r"(?:(?:от\s*)?\d+(?:[.,/]\d+)?\s*(?:-|до|—)\s*)?\d+(?:[.,/]\d+)?"

ATTR_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            rf"\b({NUM})\s*(?:gb|гб|гиг(?:ов)?|tb|тб|терабайт|mb|мб)\b", re.I
        ),
        "memory_storage",
    ),
    (
        re.compile(
            r"\b(\d+(?:[.,]\d+)?)\s*[xх×*]\s*(\d+(?:[.,]\d+)?)(?:\s*[xх×*]\s*(\d+(?:[.,]\d+)?))?\b",
            re.I,
        ),
        "dimensions",
    ),
    (
        re.compile(
            rf"\b({NUM})\s*(?:мм|mm|см|cm|м|m|дюйм(?:а|ов)?|\"|''|inch)\b", re.I
        ),
        "size",
    ),
    (re.compile(rf"\b({NUM})\s*(?:вт|w|ватт|квт|kw)\b", re.I), "power"),
    (re.compile(rf"\b({NUM})\s*(?:а|a|ампер)\b", re.I), "current"),
    (
        re.compile(rf"\b({NUM})\s*(?:м2|m2|кв\.?\s*м|квадрат(?:ов)?)\b", re.I),
        "area",
    ),
    (re.compile(rf"\b({NUM})\s*(?:кг|kg|г|g|грамм)\b", re.I), "weight"),
    (
        re.compile(rf"\b({NUM})\s*(?:л|l|литр(?:а|ов)?|мл|ml)\b", re.I),
        "volume",
    ),
    (
        re.compile(rf"\b({NUM})\s*(?:гц|hz|кгц|khz|мгц|mhz)\b", re.I),
        "frequency",
    ),
    (re.compile(rf"\b({NUM})\s*(?:дб|db)\b", re.I), "noise_level"),
    (
        re.compile(
            r"\b(4k|4к|8k|8к|2k|2к|1080p|720p|1440p|uhd|full\s*hd|fhd|hd)\b",
            re.I,
        ),
        "resolution_standard",
    ),
    (
        re.compile(
            r"\b(wi[- ]?fi|bluetooth|bt|nfc|5g|4g|lte|3g|gps|usb[- ]?c|type[- ]?c|hdmi|vga)\b",
            re.I,
        ),
        "connectivity",
    ),
    (
        re.compile(
            rf"\b({NUM})\s*(?:ч|h|час(?:а|ов)?|мин|min|минут(?:ы|у)?)\b", re.I
        ),
        "time",
    ),
    (
        re.compile(rf"\b({NUM})\s*(?:°C|\*C|°С|\*С|градус(?:ов)?)\b", re.I),
        "temperature",
    ),
    (re.compile(rf"\b({NUM})\s*(?:в|v|вольт)\b", re.I), "voltage"),
    (re.compile(rf"\b({NUM})\s*(?:шт\.?|штук[иа]?)\b", re.I), "quantity"),
    (
        re.compile(
            rf"\b({NUM})\s*(?:год[а]?|лет|мес(?:яц(?:ев|а)?)?|недел[иь])\b", re.I
        ),
        "warranty_period",
    ),
]

COLORS = {
    "белый",
    "черный",
    "серый",
    "красный",
    "синий",
    "зеленый",
    "золотой",
    "серебристый",
    "розовый",
    "голубой",
    "фиолетовый",
    "оранжевый",
    "коричневый",
    "бежевый",
    "желтый",
    "титановый",
    "графитовый",
    "шампань",
    "мятный",
    "бирюзовый",
    "бордовый",
    "space gray",
    "midnight",
    "starlight",
    "silver",
    "gold",
    "black",
    "white",
}

BRAND_ALIASES = {
    "iphone": "Apple",
    "айфон": "Apple",
    "айфоны": "Apple",
    "macbook": "Apple",
    "макбук": "Apple",
    "ipad": "Apple",
    "айпад": "Apple",
    "galaxy": "Samsung",
    "самсунг": "Samsung",
    "редми": "Xiaomi",
    "redmi": "Xiaomi",
    "poco": "Xiaomi",
    "поко": "Xiaomi",
    "сяоми": "Xiaomi",
    "xiaomi": "Xiaomi",
}


def lemmatize_text(text: str) -> List[str]:
    """Приводит русские слова к начальной форме с помощью Natasha."""
    if not text:
        return []

    # Предварительно разделяем склеенные числа и единицы
    text = re.sub(r"(\d+)([А-Яа-яA-Za-z]{1,4})\b", r"\1 \2", text)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemmas = []
    for token in doc.tokens:
        # Для кириллицы применяем лемматизацию (pos/feats уже проставлены tag_morph)
        if re.search(r"[а-яё]", token.text, re.I):
            token.lemmatize(morph_vocab)
            lemma = (token.lemma or token.text).lower().replace("ё", "е")
        else:
            lemma = token.text.lower()
        lemmas.append(lemma)

    return lemmas


def _split_glued(text: str) -> str:
    """Расклеивает число и короткую единицу: «16гб» -> «16 гб»."""
    return re.sub(r"(\d+)([А-Яа-яA-Za-z]{1,4})\b", r"\1 \2", text)


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Токенизация со сдвигами символов.

    ВАЖНО: спаны считаются относительно НОРМАЛИЗОВАННОГО текста
    (после _split_glued). Регэкспы атрибутов должны гоняться по тому же
    нормализованному тексту, иначе спаны разъедутся.
    """
    if not text:
        return []
    normalized_text = _split_glued(text)
    tokens = []
    for m in re.finditer(
        r"[A-Za-zА-Яа-яЁё0-9]+(?:[.\-/][A-Za-zА-Яа-яЁё0-9]+)*|[^\s]",
        normalized_text,
    ):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower().replace("ё", "е")
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class WeakLabeler:
    brands: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    colors: Set[str] = field(default_factory=lambda: set(COLORS))

    brand_canonical: Dict[str, str] = field(default_factory=dict)
    category_canonical: Dict[str, str] = field(default_factory=dict)

    _brand_phrases: List[List[str]] = field(default_factory=list, repr=False)
    _category_phrases: List[List[str]] = field(default_factory=list, repr=False)

    @classmethod
    def from_files(
        cls,
        brands_path: Path | str,
        categories_path: Path | str,
    ) -> "WeakLabeler":
        """Строит лейблер из текстовых словарей brands.txt / categories.txt."""
        brands = _load_lines(brands_path)
        categories = _load_lines(categories_path)
        return cls.from_iterables(brands, categories)

    @classmethod
    def from_iterables(
        cls, brands: Iterable[str], categories: Iterable[str] | None = None
    ) -> "WeakLabeler":
        brand_canonical: Dict[str, str] = {}
        for b in brands:
            b = (b or "").strip()
            if len(b) < 2:
                continue
            key = _normalize(b)
            brand_canonical[key] = b

        for alias, canon in BRAND_ALIASES.items():
            brand_canonical[_normalize(alias)] = canon

        cat_canonical: Dict[str, str] = {}
        for c in categories or []:
            c = (c or "").strip()
            if len(c) < 2:
                continue
            # Лемматизируем категории из словарей
            lemmatized_cat = " ".join(lemmatize_text(c))
            cat_canonical[lemmatized_cat] = c

        obj = cls(
            brands=set(brand_canonical.keys()),
            categories=set(cat_canonical.keys()),
            brand_canonical=brand_canonical,
            category_canonical=cat_canonical,
        )
        obj._compile_phrases()
        return obj

    def _compile_phrases(self) -> None:
        self._brand_phrases = [p.split() for p in self.brands if p]
        self._category_phrases = [p.split() for p in self.categories if p]
        self._brand_phrases.sort(
            key=lambda x: (-len(x), -sum(len(t) for t in x))
        )
        self._category_phrases.sort(
            key=lambda x: (-len(x), -sum(len(t) for t in x))
        )

    def label_query(self, query: str) -> List[Tuple[str, str]]:
        """Главный метод: берет query, лемматизирует через Natasha, но размечает ИСХОДНЫЕ токены."""
        tokens = tokenize(query)
        if not tokens:
            return []

        tags = ["O"] * len(tokens)

        # 1. ЛЕММАТИЗАЦИЯ ВСЕГО ПРЕДЛОЖЕНИЯ ЧЕРЕЗ NATASHA
        lemmatized_tokens = lemmatize_text(query)

        # Синхронизация длин токенов при необходимости
        if len(lemmatized_tokens) != len(tokens):
            lower_toks = [_normalize(t[0]) for t in tokens]
        else:
            lower_toks = lemmatized_tokens

        # 2. ПОИСК БРЕНДОВ В ЛЕММАТИЗИРОВАННОМ ТЕКСТЕ
        self._apply_phrases(lower_toks, tags, self._brand_phrases, "BRAND")

        # 3. ПОИСК КАТЕГОРИЙ В ЛЕММАТИЗИРОВАННОМ ТЕКСТЕ (теперь "холодильниками" найдет "холодильник")
        self._apply_phrases(
            lower_toks, tags, self._category_phrases, "CATEGORY"
        )

        # 4. ПОИСК ЦВЕТОВ (теперь "красную" превратится в "красный" и найдет совпадение)
        for i, lt in enumerate(lower_toks):
            if tags[i] != "O":
                continue
            if lt in self.colors:
                tags[i] = "B-ATTR"

        # 5. РЕГУЛЯРКИ ДЛЯ ЧИСЛОВЫХ АТРИБУТОВ (память, размер, мощность)
        # спаны токенов считаются по нормализованному тексту — гоним регэксп по нему же
        self._apply_attr_patterns(_split_glued(query), tokens, tags)

        # Возвращаем ИСХОДНЫЕ слова и полученные BIO-теги
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
        """Размечает список запросов, оставляя только с >= min_entities сущностей."""
        labeled: List[List[Tuple[str, str]]] = []
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
    """Сворачивает BIO-последовательность в список сущностей с char-спанами."""
    entities: List[Dict] = []
    i = 0
    # спаны и вырезка текста — в координатах нормализованного текста
    norm_query = _split_glued(query) if query is not None else None
    tok_spans = tokenize(query) if query is not None else [
        (t, 0, 0) for t, _ in tokens_tags
    ]

    while i < len(tokens_tags):
        _tok, tag = tokens_tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < len(tokens_tags) and tokens_tags[j][1] == f"I-{label}":
                j += 1
            text = " ".join(tokens_tags[k][0] for k in range(i, j))
            ent: Dict = {"text": text, "label": label}
            if norm_query is not None and j - 1 < len(tok_spans):
                span_start = tok_spans[i][1]
                span_end = tok_spans[j - 1][2]
                ent["span"] = [span_start, span_end]
                ent["text"] = norm_query[span_start:span_end].strip()
            entities.append(ent)
            i = j
        else:
            i += 1
    return entities


def entities_to_structured(
    entities: List[Dict], labeler: Optional["WeakLabeler"] = None
) -> Dict:
    """Сворачивает сущности в поля brand / category / attributes."""
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
            if labeler:
                # категории в словаре канона хранятся в лемматизированной форме
                lemma_key = " ".join(lemmatize_text(text))
                category = labeler.category_canonical.get(
                    lemma_key,
                    labeler.category_canonical.get(key, text),
                )
            else:
                category = text
        elif label == "ATTR":
            attr_type = _guess_attr_type(text)
            attributes.setdefault(attr_type, []).append(text)
    return {
        "brand": brand,
        "category": category,
        "attributes": {
            k: v[0] if len(v) == 1 else v for k, v in attributes.items()
        },
    }


def _guess_attr_type(text: str) -> str:
    t = text.lower().replace("ё", "е")
    # цвет может прийти словоформой («белую») — сверяем и по лемме
    if t in COLORS or " ".join(lemmatize_text(t)) in COLORS:
        return "color"
    if re.search(r"\d+\s*(gb|гб|гиг|mb|мб)", t):
        return "memory"
    if re.search(r"\d+\s*(tb|тб|терабайт)", t):
        return "storage"
    if re.search(r"[xх×*]\s*\d", t):
        return "dimensions"
    if re.search(r"(мм|mm|см|cm|дюйм|\")", t):
        return "size"
    if re.search(r"(вт|w|ватт|квт|kw)\b", t):
        return "power"
    if re.search(r"(гц|hz|кгц|khz|мгц|mhz)", t):
        return "frequency"
    if re.search(r"(л|l|литр|мл|ml)\b", t):
        return "volume"
    if re.search(r"(кг|kg|грамм)\b", t):
        return "weight"
    if re.search(r"(4k|8k|2k|uhd|fhd|hd|1080p|720p|1440p)", t):
        return "resolution"
    if re.search(r"(wi-?fi|bluetooth|bt|nfc|5g|4g|lte|3g|gps|usb|hdmi|vga)", t):
        return "connectivity"
    return "other"


def _load_lines(path: Path | str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        ln.strip()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]