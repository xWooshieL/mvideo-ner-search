"""Weak supervision: BIO labeling for search queries (Natasha lemmatization + dictionaries/MODEL)."""

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
    # Память (оперативная / накопители) — полные единицы
    (re.compile(rf"\b({NUM})\s*(?:gb|гб|гиг(?:ов)?|tb|тб|терабайт|mb|мб|kb|кб)\b", re.I), "memory_storage"),
    # Усечённое «16 г» / «256 g» в поисковых запросах ≈ гб (типичные объёмы RAM/SSD)
    (re.compile(r"\b(8|16|32|64|128|256|512|1024|2048)\s*[гg]\b", re.I), "memory_storage"),
    # Точное разрешение — ДО габаритов, иначе 1920x1080 уходит в dimensions
    (re.compile(r"\b(\d{3,5})\s*[xх×*]\s*(\d{3,5})\s*(?:пикс|pix|p)?\b", re.I), "resolution_exact"),
    # Габариты (ВхШхГ или ДxШ)
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*[xх×*]\s*(\d+(?:[.,]\d+)?)(?:\s*[xх×*]\s*(\d+(?:[.,]\d+)?))?\b", re.I),
     "dimensions"),
    # Размеры и диагонали экрана
    (re.compile(rf"\b({NUM})\s*(?:мм|mm|см|cm|м|m|дюйм(?:а|ов)?|\"|''|inch)\b", re.I), "size"),
    # Мощность
    (re.compile(rf"\b({NUM})\s*(?:вт|w|ватт|квт|kw)\b", re.I), "power"),
    # Сила тока
    (re.compile(rf"\b({NUM})\s*(?:а|a|ампер)\b", re.I), "current"),
    # Площадь обогрева
    (re.compile(rf"\b({NUM})\s*(?:м2|m2|кв\.?\s*м|квадрат(?:ов)?)\b", re.I), "area"),
    # Вес — без голого «г»/«g» (иначе 16 г / 5 g / 256 g шумят)
    (re.compile(rf"\b({NUM})\s*(?:кг|kg|грамм(?:а|ов)?)\b", re.I), "weight"),
    # Объем
    (re.compile(rf"\b({NUM})\s*(?:л|l|литр(?:а|ов)?|мл|ml)\b", re.I), "volume"),
    # Частота
    (re.compile(rf"\b({NUM})\s*(?:гц|hz|кгц|khz|мгц|mhz)\b", re.I), "frequency"),
    # Шум и чувствительность
    (re.compile(rf"\b({NUM})\s*(?:дб|db)\b", re.I), "noise_level"),
    # Кратность увеличения
    (re.compile(rf"\b({NUM})\s*(?:xх|крат)\b", re.I), "magnification"),
    # Расход / Скорость
    (re.compile(rf"\b({NUM})\s*(?:г/мин|g/min|л/мин|l/min)\b", re.I), "flow_rate"),
    # Разрешение экрана (стандарты)
    (re.compile(r"\b(4k|4к|8k|8к|2k|2к|1080p|720p|1440p|uhd|full\s*hd|fhd|hd)\b", re.I), "resolution_standard"),
    # Степень защиты
    (re.compile(r"\b(IP\s*[0-6X][0-9X])\b", re.I), "ip_rating"),
    # Технологии связи и порты
    (re.compile(r"\b(wi[- ]?fi|bluetooth|bt|nfc|5g|4g|lte|3g|gps|usb[- ]?c|type[- ]?c|hdmi|vga)\b", re.I),
     "connectivity"),
    # Время и длительность
    (re.compile(rf"\b({NUM})\s*(?:ч|h|час(?:а|ов)?|мин|min|минут(?:ы|у)?)\b", re.I), "time"),
    # Температура
    (re.compile(rf"\b({NUM})\s*(?:°C|\*C|°С|\*С|градус(?:ов)?)\b", re.I), "temperature"),
    # Напряжение
    (re.compile(rf"\b({NUM})\s*(?:в|v|вольт)\b", re.I), "voltage"),
    # Количество (штуки, упаковки)
    (re.compile(rf"\b({NUM})\s*(?:шт\.?|штук[иа]?)\b", re.I), "quantity"),
    # Возрастные ограничения
    (re.compile(r"\b(0|6|12|16|18)\s*\+\b", re.I), "age_restriction"),
    # Ресурс расходников
    (re.compile(rf"\b({NUM})\s*(?:стр\.?|страниц(?:ы)?)\b", re.I), "pages_yield"),
    # Гарантия и срок службы
    (re.compile(rf"\b({NUM})\s*(?:год[а]?|лет|мес(?:яц(?:ев|а)?)?|недел[иь])\b", re.I), "warranty_period"),
    # Сопротивление
    (re.compile(rf"\b({NUM})\s*(?:ом|ohm)\b", re.I), "impedance"),
    # Скорость вращения
    (re.compile(rf"\b({NUM})\s*(?:об/мин|rpm)\b", re.I), "rpm"),
    # Разрешение матриц
    (re.compile(rf"\b({NUM})\s*(?:мпикс|mpix|мп|mp)\b", re.I), "megapixels"),
    # Производительность / Воздухообмен
    (re.compile(rf"\b({NUM})\s*(?:м3/ч|m3/h|м³/ч)\b", re.I), "airflow_capacity"),
    # Плотность
    (re.compile(rf"\b({NUM})\s*(?:г/м2|g/m2|г/м²)\b", re.I), "density"),
]

COLORS = {
    # Базовые русские цвета и все формы
    "белый", "белая", "белое", "белые",
    "черный", "чёрный", "черная", "чёрная", "черное", "чёрное",
    "серый", "серая", "серое", "серые",
    "красный", "красная", "красное", "красные",
    "синий", "синяя", "синее", "синие",
    "зеленый", "зелёный", "зеленая", "зелёная", "зеленое", "зелёное", "зеленые", "зелёные",
    "золотой", "золотая", "золотое", "золото",
    "серебристый", "серебряный", "серебристая", "серебряная", "серебро",
    "розовый", "розовая", "розовое",
    "голубой", "голубая", "голубое",
    "фиолетовый", "фиолетовая", "фиолетовое",
    "оранжевый", "оранжевая", "оранжевое",
    "коричневый", "коричневая", "коричневое",
    "бежевый", "бежевая", "бежевое",
    "желтый", "жёлтый", "желтая", "жёлтая", "желтое", "жёлтое",
    "золотистый", "золотистая", "золотистое",

    # Популярные технологические / дизайнерские цвета из каталогов (добавлены медный и шоколадный)
    "титановый", "титан", "графитовый", "графит", "мокрый асфальт",
    "шампань", "персиковый", "бронзовый", "бронза", "латунь",
    "мятный", "оливковый", "бирюзовый", "бордовый", "сливовый",
    "медный", "шоколадный",

    # Англоязычные цвета и фирменные наименования
    "space gray", "midnight", "starlight", "silver", "gold",
    "black", "white", "pink", "blue", "green", "graphite", "titanium"
}

# Расширенная карта алиасов
BRAND_ALIASES = {
    # Apple
    "iphone": "Apple",
    "айфон": "Apple",
    "айфоны": "Apple",
    "macbook": "Apple",
    "макбук": "Apple",
    "ipad": "Apple",
    "айпад": "Apple",
    "airpods": "Apple",
    "аирподс": "Apple",

    # Samsung
    "galaxy": "Samsung",
    "самсунг": "Samsung",
    "samsung": "Samsung",

    # Xiaomi / Poco / Redmi
    "редми": "Xiaomi",
    "redmi": "Xiaomi",
    "poco": "Xiaomi",
    "поко": "Xiaomi",
    "сяоми": "Xiaomi",
    "ксяоми": "Xiaomi",
    "xiaomi": "Xiaomi",

    # Huawei / Honor
    "honor": "HONOR",
    "хонор": "HONOR",
    "хуавей": "HUAWEI",
    "huawei": "HUAWEI",

    # Другие популярные бренды
    "ленова": "Lenovo",
    "леново": "Lenovo",
    "lenovo": "Lenovo",
    "асус": "ASUS",
    "asus": "ASUS",
    "асер": "Acer",
    "acer": "Acer",
    "тошиба": "Toshiba",
    "toshiba": "Toshiba",
    "хайер": "Haier",
    "haier": "Haier",
    "китфорт": "Kitfort",
    "kitfort": "Kitfort",
}

CATEGORY_SEEDS = {
    # Смартфоны, гаджеты и связь
    "смартфон", "смартфоны", "телефон", "телефоны", "смарт-часы", "фитнес-браслет",
    "планшет", "планшеты", "умные часы", "часы", "браслет",

    # Компьютеры, ноутбуки и комплектующие
    "ноутбук", "ноутбуки", "льтрабук", "нетбук", "монитор", "мониторы",
    "клавиатура", "клавиатуры", "мышь", "мышка", "мышки", "принтер", "принтеры",
    "роутер", "роутеры", "маршрутизатор", "видеокарта", "видеокарты",
    "процессор", "процессоры", "ssd", "hdd", "память", "материнская", "плата",
    "блок", "питания", "корпус", "кулер", "оперативная память",

    # Крупная бытовая техника (КБТ)
    "холодильник", "холодильники", "стиральная", "машинка", "стиральная машина",
    "сушильная", "сушильная машина", "посудомойка", "посудомоечная", "посудомоечная машина",
    "плита", "плиты", "варочная", "панель", "варочная панель",
    "духовка", "духовой", "шкаф", "духовой шкаф", "микроволновка", "микроволновая",
    "микроволновая печь", "вытяжка", "вытяжки", "водонагреватель", "обогреватель",
    "кондиционер", "сплит-система",

    # Мелкая бытовая техника (МБТ) и кухня
    "пылесос", "робот", "робот-пылесос", "вертикальный пылесос",
    "чайник", "электрочайник", "кофемашина", "кофемолка", "кофеварка",
    "блендер", "миксер", "мультиварка", "фен", "утюг", "отпариватель",
    "мясорубка", "тостер", "гриль", "микроволновка", "пароварка", "фритюрница",
    "электровафельница",

    # ТВ, аудио и развлечения
    "телевизор", "телевизоры", "наушники", "колонка", "колонки",
    "саундбар", "музыкальный центр", "проектор", "игровой", "консоль",
    "приставка", "геймпад", "smart", "джжойстик",

    # Фото, видео и оптика
    "фотоаппарат", "камера", "видеокамера", "объектив", "штатив",
    "микроскоп", "телескоп", "бинокль", "лупа",

    # Сантехника и ремонт
    "раковина", "унитаз", "ванна", "душевой", "уголок", "душевая кабина",
    "смеситель", "полотенцесушитель", "инсталляция", "душевой поддон",

    # Мебель, свет и интерьер (добавлены прожектор и торшер)
    "кресло", "компьютерное кресло", "игровое кресло", "офисное кресло",
    "шкаф", "этажерка", "вешалка", "напольная вешалка", "стол", "стул",
    "камин", "электрокамин", "прожектор", "торшер",

    # Посуда и товары для дома
    "кастрюля", "сковорода", "ковш", "набор посуды", "кружка", "чашка",
    "термос", "чайник заварочный", "фильтр для воды", "швабра", "насадка для швабры",
    "шейкер", "поднос",

    # Аксессуары, чехлы, расходники и уход (добавлены расческа, мотыга, укрывной материал)
    "чехол", "чехлы", "клип-кейс", "защитное стекло", "пленка",
    "картридж", "тонер", "аккумулятор", "батарейка", "зарядное устройство",
    "кабель", "переходник", "рюкзак", "сумка для ноутбука",
    "расческа", "массажная щетка", "мотыга", "укрывной материал", "держатель"
}

# --- ATTR type / purpose (lexical; app gold уже размечает эти сабтипы) ---
# Леммы Natasha + сырые формы; матч по токену/фразе после lemmatize.
TYPE_SEEDS: Set[str] = {
    # connectivity form-factor (Natasha: беспроводные → беспроводный)
    "беспроводный", "беспроводной", "беспроводные", "беспроводная", "беспроводное",
    "проводный", "проводной", "проводные", "проводная",
    "накладный", "накладной", "накладные", "вкладыш", "вкладыши",
    # smart / product kind
    "смарт", "smart", "android", "сотовый", "сотовая",
    # energy / install
    "газовый", "газовая", "газовые", "газовое",
    "электрический", "электрическая", "электрические", "электрическое",
    "индукционный", "индукционная", "индукционные",
    "встраиваемый", "встраиваемая", "встраиваемые", "встраемый", "встраемая",
    "напольный", "напольная", "напольные",
    "настенный", "настенная",
    # size/form adjectives used as type in app
    "узкий", "узкая", "узкие", "широкий", "широкая",
    "масляный", "масляная", "лазерный", "лазерная", "цветной", "цветная",
    "строительный", "строительная", "автомобильный", "автомобильная",
    "планетарный", "планетарная", "вертикальный", "вертикальная",
    "ретро", "ларь", "разветвитель",
    "кухонный", "кухонная",
    "тестомесильный", "тестомесильная",
    "электроподжиг", "электроподжогом",
    "smart tv", "smarttv",
}

PURPOSE_SEEDS: Set[str] = {
    "спорт", "спорта", "кухня", "кухни",
    "смузи", "принтер", "принтера",
    "окно", "окон", "мойки", "мойки окон",
    "телефон", "телефона",
    "сушильный", "сушильных", "машин", "сушильных машин",
    "плита", "плиты", "индукционной", "индукционной плиты",
    "часы", "смарт часы",
    "компьютер", "компьютера",
}

# «для …» — весь хвост после «для» как purpose-span при детекции
PURPOSE_FOR_RE = re.compile(
    r"\bдля\s+([А-Яа-яA-Za-z0-9][А-Яа-яA-Za-z0-9\s\-]{0,40})",
    re.I,
)

# App gold subtype → teacher canon (eval; jsonl не переписываем)
GOLD_TO_CANON: Dict[str, str] = {
    "type": "type",
    "gas": "type",
    "floor": "type",
    "style": "type",
    "function": "type",
    "purpose": "purpose",
    "food": "purpose",
    "depth": "size",
    "width": "size",
    "heigh": "size",
    "length": "size",
    "counts": "quantity",
    "material": "other",
    "chip": "other",
    "sim": "other",
    "game": "other",
    "used": "other",
    "new": "other",
    "country": "other",
    "release date": "other",
    "delivery": "other",
    "speed": "other",
}


def gold_subtype_to_canon(subtype: str) -> str:
    """Сводит app gold subtype к канону teacher. Неизвестное → as-is."""
    if not subtype:
        return "other"
    s = subtype.strip().lower()
    return GOLD_TO_CANON.get(s, s)


def _split_glued(text: str) -> str:
    """Расклеивает число и короткую единицу: «16гб» -> «16 гб»."""
    return re.sub(r"(\d+)([А-Яа-яA-Za-z]{1,4})\b", r"\1 \2", text)


def lemmatize_text(text: str) -> List[str]:
    """Приводит русские слова к начальной форме с помощью Natasha."""
    if not text:
        return []

    text = _split_glued(text)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemmas = []
    for token in doc.tokens:
        if re.search(r"[а-яё]", token.text, re.I):
            token.lemmatize(morph_vocab)
            lemma = (token.lemma or token.text).lower().replace("ё", "е")
        else:
            lemma = token.text.lower()
        lemmas.append(lemma)

    return lemmas


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Токенизация со сдвигами символов.

    ВАЖНО: спаны считаются относительно НОРМАЛИЗОВАННОГО текста
    (после _split_glued). Регэкспы атрибутов должны гоняться по тому же
    нормализованному тексту, иначе спаны разъедутся.
    """
    if not text:
        return []
    normalized_text = _split_glued(text)
    tokens: List[Tuple[str, int, int]] = []
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
    genres: Set[str] = field(default_factory=set)  # НОВОЕ: для жанров фильмов/игр
    persons: Set[str] = field(default_factory=set)  # НОВОЕ: для актеров и режиссеров
    models: Set[str] = field(default_factory=set)  # линейки / product lines (g pro x se, v15, …)
    colors: Set[str] = field(default_factory=lambda: set(COLORS))

    # lower -> canonical display form
    brand_canonical: Dict[str, str] = field(default_factory=dict)
    category_canonical: Dict[str, str] = field(default_factory=dict)
    genre_canonical: Dict[str, str] = field(default_factory=dict)  # НОВОЕ
    person_canonical: Dict[str, str] = field(default_factory=dict)  # НОВОЕ
    model_canonical: Dict[str, str] = field(default_factory=dict)

    # sorted longest-first phrase lists (token sequences)
    _brand_phrases: List[List[str]] = field(default_factory=list, repr=False)
    _category_phrases: List[List[str]] = field(default_factory=list, repr=False)
    _genre_phrases: List[List[str]] = field(default_factory=list, repr=False)  # НОВОЕ
    _person_phrases: List[List[str]] = field(default_factory=list, repr=False)
    _model_phrases: List[List[str]] = field(default_factory=list, repr=False)
    _type_phrases: List[List[str]] = field(default_factory=list, repr=False)
    _purpose_phrases: List[List[str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._type_phrases or not self._purpose_phrases:
            type_phrases = [p.split() for p in TYPE_SEEDS if p]
            purpose_phrases = [p.split() for p in PURPOSE_SEEDS if p]
            type_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
            purpose_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
            if not self._type_phrases:
                self._type_phrases = type_phrases
            if not self._purpose_phrases:
                self._purpose_phrases = purpose_phrases

    @classmethod
    def from_files(
            cls,
            brands_path: Path | str,
            categories_path: Path | str,
            genres_path: Path | str | None = None,  # НОВОЕ: опциональный путь к жанрам
            persons_path: Path | str | None = None,  # НОВОЕ: опциональный путь к персонам
            models_path: Path | str | None = None,
    ) -> "WeakLabeler":
        brands = _load_lines(brands_path)
        categories = _load_lines(categories_path)
        genres = _load_lines(genres_path) if genres_path else []
        persons = _load_lines(persons_path) if persons_path else []
        models = _load_lines(models_path) if models_path else []
        return cls.from_iterables(brands, categories, genres, persons, models)

    @classmethod
    def from_iterables(
            cls,
            brands: Iterable[str],
            categories: Iterable[str] | None = None,
            genres: Iterable[str] | None = None,  # НОВОЕ
            persons: Iterable[str] | None = None,  # НОВОЕ
            models: Iterable[str] | None = None,
    ) -> "WeakLabeler":
        # 1. Обработка брендов / вендоров
        brand_canonical: Dict[str, str] = {}
        for b in brands:
            b = (b or "").strip()
            if len(b) < 2:
                continue
            key = _normalize(b)
            if key not in brand_canonical or (
                    brand_canonical[key].isupper() and not b.isupper()
            ):
                brand_canonical[key] = b

        for alias, canon in BRAND_ALIASES.items():
            brand_canonical[_normalize(alias)] = canon
            brand_canonical[_normalize(canon)] = canon

        # 2. Обработка категорий (ключи — леммы Natasha + fallback _normalize)
        cat_canonical: Dict[str, str] = {}
        brand_keys = set(brand_canonical.keys())
        for c in list(categories or []) + list(CATEGORY_SEEDS):
            c = (c or "").strip()
            if len(c) < 2:
                continue
            cn = _normalize(c)
            if cn in brand_keys or cn in BRAND_ALIASES:
                continue
            lemmatized_cat = " ".join(lemmatize_text(c)) or cn
            cat_canonical[lemmatized_cat] = c
            cat_canonical.setdefault(cn, c)

        # 3. НОВОЕ: Обработка жанров (например, боевик, драма, ужасы из JSON)
        genre_canonical: Dict[str, str] = {}
        default_genres = {"боевик", "драма", "триллер", "комедия", "ужасы", "фантастика", "фэнтези", "детектив",
                          "мелодрама", "криминал", "приключения"}
        for g in list(genres or []) + list(default_genres):
            g = (g or "").strip()
            if len(g) < 2:
                continue
            gn = _normalize(g)
            genre_canonical[gn] = g

        # 4. НОВОЕ: Обработка персон (актеры, режиссеры)
        person_canonical: Dict[str, str] = {}
        for p in persons or []:
            p = (p or "").strip()
            if len(p) < 3:  # Имена обычно длиннее
                continue
            pn = _normalize(p)
            person_canonical[pn] = p

        # 5. Product lines / MODEL (g pro x se, v15, …) — не путать с ATTR-числами
        model_canonical: Dict[str, str] = {}
        for m in models or []:
            m = (m or "").strip()
            if len(m) < 2:
                continue
            mn = _normalize(m)
            toks = mn.split()
            if not toks:
                continue
            if all(t.replace(".", "").isdigit() for t in toks):
                continue
            model_canonical[mn] = m

        obj = cls(
            brands=set(brand_canonical.keys()),
            categories=set(cat_canonical.keys()),
            genres=set(genre_canonical.keys()),  # НОВОЕ
            persons=set(person_canonical.keys()),  # НОВОЕ
            models=set(model_canonical.keys()),
            brand_canonical=brand_canonical,
            category_canonical=cat_canonical,
            genre_canonical=genre_canonical,  # НОВОЕ
            person_canonical=person_canonical,  # НОВОЕ
            model_canonical=model_canonical,
        )
        obj._compile_phrases()
        return obj

    def _compile_phrases(self) -> None:
        brand_phrases = [p.split() for p in self.brands if p]
        cat_phrases = [p.split() for p in self.categories if p]
        genre_phrases = [p.split() for p in self.genres if p]  # НОВОЕ
        person_phrases = [p.split() for p in self.persons if p]  # НОВОЕ
        model_phrases = [p.split() for p in self.models if p]
        type_phrases = [p.split() for p in TYPE_SEEDS if p]
        purpose_phrases = [p.split() for p in PURPOSE_SEEDS if p]

        brand_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
        cat_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
        genre_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))  # НОВОЕ
        person_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))  # НОВОЕ
        model_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
        type_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))
        purpose_phrases.sort(key=lambda x: (-len(x), -sum(len(t) for t in x)))

        self._brand_phrases = brand_phrases
        self._category_phrases = cat_phrases
        self._genre_phrases = genre_phrases  # НОВОЕ
        self._person_phrases = person_phrases  # НОВОЕ
        self._model_phrases = model_phrases
        self._type_phrases = type_phrases
        self._purpose_phrases = purpose_phrases

    def label_query(self, query: str) -> List[Tuple[str, str]]:
        """Лемматизирует через Natasha для матчинга словарей, возвращает исходные токены + BIO."""
        tokens = tokenize(query)
        if not tokens:
            return []
        tags = ["O"] * len(tokens)

        # 1. Лемматизация всего предложения через Natasha
        lemmatized_tokens = lemmatize_text(query)
        # Синхронизация длин токенов при необходимости
        if len(lemmatized_tokens) != len(tokens):
            lower_toks = [_normalize(t[0]) for t in tokens]
        else:
            lower_toks = lemmatized_tokens

        # 2) Brands
        self._apply_phrases(lower_toks, tags, self._brand_phrases, "BRAND")
        # 3) MODEL / product line
        self._apply_phrases(lower_toks, tags, self._model_phrases, "MODEL")
        # 4) Categories (леммы: «холодильниками» → «холодильник»)
        self._apply_phrases(lower_toks, tags, self._category_phrases, "CATEGORY")
        # 5) Genres
        self._apply_phrases(lower_toks, tags, self._genre_phrases, "GENRE")
        # 6) Persons
        self._apply_phrases(lower_toks, tags, self._person_phrases, "PERSON")

        # 7) Colors as ATTR (лемма «красную» → «красный»)
        for i, lt in enumerate(lower_toks):
            if tags[i] != "O":
                continue
            if lt in self.colors or lt.replace("ё", "е") in self.colors:
                tags[i] = "B-ATTR"
        # 8) Regex attributes — по тому же тексту, что и tokenize
        norm_q = _split_glued(query)
        self._apply_attr_patterns(norm_q, tokens, tags)
        # 9) purpose («для …») затем type/purpose seeds — только на оставшихся O
        self._apply_purpose_for(norm_q, tokens, tags)
        self._apply_phrases(lower_toks, tags, self._purpose_phrases, "ATTR")
        self._apply_phrases(lower_toks, tags, self._type_phrases, "ATTR")

        return [(tokens[i][0], tags[i]) for i in range(len(tokens))]

    def _apply_purpose_for(
            self,
            query: str,
            tokens: List[Tuple[str, int, int]],
            tags: List[str],
    ) -> None:
        """Помечает хвост после «для» как ATTR (purpose), не трогая уже занятые токены."""
        for m in PURPOSE_FOR_RE.finditer(query):
            start, end = m.start(1), m.end(1)
            first = True
            for i, (_tok, s, e) in enumerate(tokens):
                if e <= start or s >= end:
                    continue
                if tags[i] != "O":
                    continue
                tags[i] = "B-ATTR" if first else "I-ATTR"
                first = False

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
                if lower_toks[i: i + plen] == phrase:
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
    """Сворачивает сущности в поля brand / category / model / attributes."""
    brand = None
    category = None
    model = None
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
                lemma_key = " ".join(lemmatize_text(text))
                category = labeler.category_canonical.get(
                    lemma_key,
                    labeler.category_canonical.get(key, text),
                )
            else:
                category = text
        elif label == "MODEL" and model is None:
            key = _normalize(text)
            model = labeler.model_canonical.get(key, text) if labeler else text
        elif label == "ATTR":
            attr_type = _guess_attr_type(text)
            attributes.setdefault(attr_type, []).append(text)
    return {
        "brand": brand,
        "category": category,
        "model": model,
        "attributes": {
            k: v[0] if len(v) == 1 else v for k, v in attributes.items()
        },
    }


def _token_norms(text: str) -> List[str]:
    """Леммы/нормы токенов span для lexical ATTR."""
    t = text.lower().replace("ё", "е").strip()
    if not t:
        return []
    lemmas = lemmatize_text(t)
    if lemmas:
        return lemmas
    return [_normalize(x) for x in t.split() if x]


def _looks_like_purpose(text: str) -> bool:
    t = text.lower().replace("ё", "е").strip()
    if not t:
        return False
    if PURPOSE_FOR_RE.search(t) or t.startswith("для "):
        return True
    norms = _token_norms(t)
    joined = " ".join(norms)
    if joined in PURPOSE_SEEDS or t in PURPOSE_SEEDS:
        return True
    # multiword seed as raw (мойки окон / сушильных машин)
    if any(s in t for s in PURPOSE_SEEDS if " " in s):
        return True
    if all(n in PURPOSE_SEEDS for n in norms) and norms:
        return True
    if len(norms) == 1 and norms[0] in PURPOSE_SEEDS:
        return True
    return False


def _looks_like_type(text: str) -> bool:
    t = text.lower().replace("ё", "е").strip()
    if not t:
        return False
    if t in TYPE_SEEDS or t.replace(" ", "") in {"smarttv"}:
        return True
    norms = _token_norms(t)
    joined = " ".join(norms)
    if joined in TYPE_SEEDS:
        return True
    # каждый токен — type seed (напр. «лазерный цветной»)
    if norms and all(n in TYPE_SEEDS for n in norms):
        return True
    # первый токен type-модификатор (узкая / беспроводные / смарт)
    if norms and norms[0] in TYPE_SEEDS:
        return True
    return False


def _guess_attr_type(text: str) -> str:
    """Тип ATTR-span: color → units → purpose → type → other.

    Имена unit-типов совпадают с ATTR_PATTERNS; type/purpose — lexical (app gold).
    """
    t = text.lower().replace("ё", "е").strip()
    if not t:
        return "other"

    if t in COLORS or " ".join(lemmatize_text(t)) in COLORS:
        return "color"

    for pattern, name in ATTR_PATTERNS:
        if pattern.search(t):
            return name

    if _looks_like_purpose(t):
        return "purpose"
    if _looks_like_type(t):
        return "type"
    return "other"


def _load_lines(path: Path | str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
