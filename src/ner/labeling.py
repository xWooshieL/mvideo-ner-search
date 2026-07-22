from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# pymorphy2 не работает на Python 3.11+ (удалён inspect.getargspec)
try:
    import pymorphy3 as _pymorphy
except Exception:
    try:
        import pymorphy2 as _pymorphy

        _pymorphy.MorphAnalyzer()  # проверка, что реально работает
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "Нужен pymorphy3 (или рабочий pymorphy2 на Python <3.11). "
            "pip install pymorphy3"
        ) from e

# Базовый паттерн для чисел (добавлена поддержка отрицательных чисел)
NUM = r"(?:(?:от\s*)?-?\d+(?:[.,/]\d+)?\s*(?:-|до|—)\s*)?-?\d+(?:[.,/]\d+)?"

# Умная граница: проверяет, что сразу после единицы измерения не идет другая буква/цифра.
END = r"(?!\w)"

# Регулярные выражения для численных атрибутов (на основе статистики M.Video/YML)
ATTR_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(rf"\b({NUM})\s*(?:gb|гб|гиг(?:ов)?|tb|тб|терабайт|mb|мб|kb|кб|bit|бит){END}", re.I), "memory_storage"),
    (re.compile(rf"\b({NUM})\s*(?:мпикс|пикс|мп|mp|ppi|dpi|lpi|твл|т/д){END}", re.I), "resolution"),
    (re.compile(rf"\b({NUM})\s*(?:мач|mah|ач|ah|мвтч|втч|wh|квтч){END}", re.I), "battery_energy"),
    (re.compile(rf"\b({NUM})\s*(?:об/мин|rpm|уд/мин|имп\.?\s*в\s*мин|кол/мин|стеж\.?/мин){END}", re.I), "rotation_speed"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*[xх×*]\s*(\d+(?:[.,]\d+)?)(?:\s*[xх×*]\s*(\d+(?:[.,]\d+)?))?(?!\w)", re.I), "dimensions"),
    (re.compile(rf"\b({NUM})\s*(?:мм|mm|см|cm|м|m|мкм|нм|nm|дюйм(?:а|ов)?|\"|'|inch){END}", re.I), "size"),
    (re.compile(rf"\b({NUM})\s*(?:вт|w|ватт|квт|kw|мвт|mw|ва|va|qвт|вт/канал|btu|лм/вт){END}", re.I), "power"),
    (re.compile(rf"\b({NUM})\s*(?:а|a|ампер|ма|ma){END}", re.I), "current"),
    (re.compile(rf"\b({NUM})\s*(?:в|v|вольт){END}", re.I), "voltage"),
    (re.compile(rf"\b({NUM})\s*(?:ом|ком|мом){END}", re.I), "resistance"),
    (re.compile(rf"\b({NUM})\s*(?:м2|m2|м²|кв\.?\s*м|квадрат(?:ов)?|см2|см²|мм2|мм²){END}", re.I), "area"),
    (re.compile(rf"\b({NUM})\s*(?:кг|kg|г|g|грамм|гр|mg|мг|т){END}", re.I), "weight"),
    (re.compile(rf"\b({NUM})\s*(?:л|l|литр(?:а|ов)?|мл|ml|м3|м³|см3|см³){END}", re.I), "volume"),
    (re.compile(rf"\b({NUM})\s*(?:гц|hz|кгц|khz|мгц|mhz|ггц|ghz){END}", re.I), "frequency"),
    (re.compile(rf"\b({NUM})\s*(?:мбит/с(?:ек)?|гбит/с(?:ек)?|мб/с(?:ек)?|mbps|кадр/с(?:ек)?|fps|гвыб/сек){END}", re.I), "data_rate"),
    (re.compile(rf"\b({NUM})\s*(?:м3/ч(?:ас)?|м³/ч(?:ас)?|л/мин|л/ч(?:ас)?|л/сек|г/мин|г/ч(?:ас)?|кг/мин|кг/ч(?:ас)?|кг/сутки|мл/сутки|cfm|стр/мин|стр/мес){END}", re.I), "flow_rate"),
    (re.compile(rf"\b({NUM})\s*(?:бар|bar|атм|atm|па|pa|кпа|kpa|мпа|mpa|мм рт\.?\s*ст\.?|мм в\.?\s*ст\.?|mbar){END}", re.I), "pressure"),
    (re.compile(rf"\b({NUM})\s*(?:г/м2|г/м²|кг/м3|кг/м³){END}", re.I), "density"),
    (re.compile(rf"\b({NUM})\s*(?:нит|nit|кд/м2|кд/м²|cd/m2|lm|лм|люкс|лк|cd|ansi){END}", re.I), "brightness"),
    (re.compile(rf"\b({NUM})\s*(?:дб|db|дба|dba|дб/окт){END}", re.I), "noise_level"),
    (re.compile(rf"\b({NUM})\s*(?:°C|\*C|°С|\*С|градус(?:ов|а|ах)?|k|к){END}", re.I), "temperature"),
    (re.compile(r"\b(4k|4к|8k|8к|2k|2к|1080p|720p|1440p|uhd|full\s*hd|fhd|hd)\b", re.I), "resolution_standard"),
    (re.compile(rf"\b({NUM})\s*(?:ч|h|час(?:а|ов)?|мин|min|минут(?:ы|у)?|сек|sec|секунд(?:ы|а)?|мс|мсек){END}", re.I), "time"),
    (re.compile(rf"\b({NUM})\s*(?:год[а]?|лет|мес(?:яц(?:ев|а|\.)?)?|недел[иь]?|дней|дня|суток|день){END}", re.I), "period"),
    (re.compile(rf"\b({NUM})\s*(?:шт\.?|штук[иа]?|лист(?:ов|а|\.)?|стр(?:аниц|\.)?|комплект(?:ов|а)?|порц\.?|человек|чел|персон|зубц(?:а|ов)|зуб(?:а|ьев)|чашк(?:и|ек|у)|секци(?:я|й|и)|розет(?:ок|ки)|симк(?:и|а)|камер|ящик(?:ов|а)|полок|деталей|предмет(?:ов|а)|баноч(?:ек|ки)){END}", re.I), "quantity"),
    (re.compile(rf"\b({NUM})\s*(?:%|проц|процентов){END}", re.I), "percentage"),
    (re.compile(rf"\b({NUM})\s*(?:крат|x|х){END}", re.I), "multiplier"),
    (re.compile(rf"\b({NUM})\s*(?:н\.?м|кн){END}", re.I), "torque_force"),
    (re.compile(rf"\b({NUM})\s*(?:дж|ккал){END}", re.I), "energy"),
    (re.compile(rf"\b({NUM})\s*(?:км/ч|м/мин|м/сек|см/сек){END}", re.I), "speed"),
    (re.compile(rf"\b({NUM})\s*(?:ядер|ядерный|ядерные){END}", re.I), "cores")
]

# Служебные слова, которые не могут быть сущностью сами по себе
# (в MODELS.txt попадает мусор из названий карточек: «для», «и», «с»…)
STOPWORDS: Set[str] = {
    "для", "и", "с", "на", "в", "по", "от", "до", "из", "под", "без",
    "к", "у", "о", "об", "за", "при", "или", "не", "но", "же",
    "купить", "цена", "недорого", "дешево", "дешевый", "новый", "б/у",
}

# Базовые русские цвета: в colors.txt — только оттенки с карточек (alpine green и т.п.),
# обычных «красный»/«белый» там нет, а в запросах они постоянны
COLOR_BASICS: Set[str] = {
    "красный", "белый", "черный", "синий", "зеленый", "желтый", "серый",
    "серебристый", "серебряный", "золотой", "золотистый", "розовый", "голубой",
    "фиолетовый", "оранжевый", "бежевый", "коричневый", "бордовый", "бирюзовый",
    "графитовый", "мятный", "лавандовый", "сиреневый", "хаки", "титановый",
}

# Синонимы брендов
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

# Инициализируем морфологический анализатор один раз
morph = _pymorphy.MorphAnalyzer()

def lemmatize_text(text: str) -> List[str]:
    """Приводит русские слова к начальной форме с помощью Pymorphy2."""
    if not text:
        return []

    text = re.sub(r"(\d+)([А-Яа-яA-Za-z]{1,4})\b", r"\1 \2", text)
    tokens = re.findall(r'[A-Za-zА-Яа-яЁё0-9]+(?:[.\-/][A-Za-zА-Яа-яЁё0-9]+)*|[^\s]', text)

    lemmas = []
    for token in tokens:
        if re.search(r"[а-яё]", token, re.I):
            lemma = morph.parse(token)[0].normal_form.replace("ё", "е")
        else:
            lemma = token.lower()
        lemmas.append(lemma)

    return lemmas


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Токенизация со сдвигами символов для сохранения оригинального текста."""
    if not text:
        return []
    normalized_text = re.sub(r"(\d+)([А-Яа-яA-Za-z]{1,4})\b", r"\1 \2", text)
    tokens = []
    for m in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+(?:[.\-/][A-Za-zА-Яа-яЁё0-9]+)*|[^\s]", normalized_text):
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
    # 7 Супер-словарей под новые классы файлов
    brands: Set[str] = field(default_factory=set)
    models: Set[str] = field(default_factory=set)
    materials: Set[str] = field(default_factory=set)
    colors: Set[str] = field(default_factory=set)
    purposes: Set[str] = field(default_factory=set)
    product_types: Set[str] = field(default_factory=set)
    features: Set[str] = field(default_factory=set)

    _phrases: Dict[str, List[List[str]]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dir(cls, dict_dir: str | Path) -> "WeakLabeler":
        """Загружает все 7 словарей из указанной папки."""
        dict_dir = Path(dict_dir)

        def load_set(filename: str) -> Set[str]:
            path = dict_dir / filename
            if not path.exists():
                # регистронезависимый поиск: на диске brands.txt, в коде BRANDS.txt (важно для macOS/Linux)
                matches = [p for p in dict_dir.glob("*.txt") if p.name.lower() == filename.lower()]
                if matches:
                    path = matches[0]
                else:
                    print(f"⚠️ Файл {filename} не найден в {dict_dir}. Будет использован пустой словарь.")
                    return set()
            loaded = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                if len(line.strip()) >= 2:
                    loaded.add(" ".join(lemmatize_text(line.strip())))
            return loaded

        # Загружаем файлы для всех 7 классов
        brands_set = load_set("BRANDS.txt")
        for alias in BRAND_ALIASES.keys():
            brands_set.add(" ".join(lemmatize_text(alias)))

        colors_set = load_set("COLORS.txt")
        for color in COLOR_BASICS:
            colors_set.add(" ".join(lemmatize_text(color)))

        obj = cls(
            brands=brands_set,
            models=load_set("MODELS.txt"),
            materials=load_set("MATERIALS.txt"),
            colors=colors_set,
            purposes=load_set("PURPOSE.txt"),
            product_types=load_set("PRODUCT_TYPE.txt"),
            features=load_set("FEATURES.txt"),
        )
        obj._compile_phrases()
        return obj

    def _compile_phrases(self) -> None:
        """Подготавливает фразы к поиску и сортирует по убыванию длины."""

        def prepare_phrases(words_set):
            phrases = []
            for p in words_set:
                if not p:
                    continue
                toks = p.split()
                # фраза целиком из служебных слов («для», «и с») — мусор из названий карточек
                if all(t in STOPWORDS for t in toks):
                    continue
                # мусор вида «+ k», «-к»: одиночный токен без 2+ букв/цифр подряд
                if len(toks) == 1 and not re.search(r"[a-zа-яё0-9]{2,}", toks[0], re.I):
                    continue
                phrases.append(toks)
            phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
            return phrases

        # Маппинг всех 7 супер-словарей на BIO теги.
        # Порядок = приоритет: конкретные словари раньше, шумный MODEL — последним,
        # чтобы «наушники» оставались PRODUCT_TYPE, а «красный» — COLOR.
        self._phrases = {
            "BRAND": prepare_phrases(self.brands),
            "PRODUCT_TYPE": prepare_phrases(self.product_types),
            "COLOR": prepare_phrases(self.colors),
            "MATERIAL": prepare_phrases(self.materials),
            "PURPOSE": prepare_phrases(self.purposes),
            "FEATURE": prepare_phrases(self.features),
            "MODEL": prepare_phrases(self.models),
        }

    def label_query(self, query: str) -> List[Tuple[str, str]]:
        """Размечает исходный запрос BIO тегами."""
        tokens = tokenize(query)
        if not tokens:
            return []

        tags = ["O"] * len(tokens)
        lemmatized_tokens = lemmatize_text(query)

        if len(lemmatized_tokens) != len(tokens):
            lower_toks = [_normalize(t[0]) for t in tokens]
        else:
            lower_toks = lemmatized_tokens

        # Проходимся по всем 7 супер-словарям
        for label_name, phrases_list in self._phrases.items():
            self._apply_phrases(lower_toks, tags, phrases_list, label_name)

        # Регулярки для чисел -> тег ATTR
        self._apply_attr_patterns(query, tokens, tags)

        return [(tokens[i][0], tags[i]) for i in range(len(tokens))]

    def _apply_phrases(self, lower_toks: List[str], tags: List[str], phrases: List[List[str]], label: str) -> None:
        n = len(lower_toks)
        for phrase in phrases:
            plen = len(phrase)
            if plen == 0 or plen > n:
                continue
            for i in range(n - plen + 1):
                if any(tags[i + k] != "O" for k in range(plen)):
                    continue
                if lower_toks[i: i + plen] == phrase:
                    tags[i] = f"B-{label}"
                    for k in range(1, plen):
                        tags[i + k] = f"I-{label}"

    def _apply_attr_patterns(self, query: str, tokens: List[Tuple[str, int, int]], tags: List[str]) -> None:
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


def bio_to_entities(
    tokens_tags: Sequence[Tuple[str, str]],
    query: str | None = "",
) -> List[Dict]:
    """Конвертирует BIO в список сущностей; при query добавляет char-span."""
    query = query or ""
    entities: List[Dict] = []
    i = 0
    tok_spans = tokenize(query) if query else [(t, 0, 0) for t, _ in tokens_tags]
    norm_query = (
        re.sub(r"(\d+)([А-Яа-яA-Za-z]{1,4})\b", r"\1 \2", query) if query else None
    )

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
                a, b = tok_spans[i][1], tok_spans[j - 1][2]
                ent["span"] = [a, b]
                ent["text"] = norm_query[a:b].strip()
            entities.append(ent)
            i = j
        else:
            i += 1
    return entities


def entities_to_structured(
    entities: List[Dict], labeler: Optional["WeakLabeler"] = None
) -> Dict:
    """Сворачивает сущности в поля под 7 супер-словарей + typed ATTR."""
    del labeler  # canonical maps убраны; оставляем сырой текст span
    brand = None
    product_type = None
    model = None
    color = None
    material = None
    purpose = None
    feature = None
    attributes: Dict[str, List[str]] = {}

    for ent in entities:
        label = ent["label"]
        text = ent["text"]
        if label == "BRAND" and brand is None:
            brand = text
        elif label in {"PRODUCT_TYPE", "CATEGORY"} and product_type is None:
            product_type = text
        elif label == "MODEL" and model is None:
            model = text
        elif label == "COLOR" and color is None:
            color = text
        elif label == "MATERIAL" and material is None:
            material = text
        elif label == "PURPOSE" and purpose is None:
            purpose = text
        elif label == "FEATURE" and feature is None:
            feature = text
        elif label == "ATTR":
            attr_type = _guess_attr_type(text)
            attributes.setdefault(attr_type, []).append(text)

    return {
        "brand": brand,
        "product_type": product_type,
        "category": product_type,  # alias для старого кода/silver
        "model": model,
        "color": color,
        "material": material,
        "purpose": purpose,
        "feature": feature,
        "attributes": {
            k: v[0] if len(v) == 1 else v for k, v in attributes.items()
        },
    }


def _guess_attr_type(text: str) -> str:
    """Тип ATTR-span: COLOR_BASICS → color, иначе первый матч ATTR_PATTERNS, иначе other.

    Цвета в BIO обычно уже B-COLOR; проверка basics — для хвостов/совместимости.
    Имена типов = группы в ATTR_PATTERNS.
    """
    t = text.lower().replace("ё", "е").strip()
    if not t:
        return "other"

    lemma = " ".join(lemmatize_text(t))
    if t in COLOR_BASICS or lemma in COLOR_BASICS:
        return "color"

    for pattern, name in ATTR_PATTERNS:
        if pattern.search(t):
            return name
    return "other"


def _load_lines(path: Path | str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
