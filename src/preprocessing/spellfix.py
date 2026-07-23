"""
Исправление опечаток в поисковых запросах (spell-fix слой).

Идея (не трогает CRF и не требует отдельной разметки):
    токен не из словаря
      -> кандидаты из categories / brands (похожая длина)
      -> score = похожесть строк (Левенштейн) + близость букв на клавиатуре
      -> score высокий -> заменить на канон
      -> иначе оставить как есть

Где применять (ОДНА И ТА ЖЕ функция в двух местах):
    1) при подготовке данных для обучения (чистим тексты -> потом учим модель);
    2) в проде на живом запросе пользователя (чистим -> потом CRF/правила).
Тогда модель и на обучении, и в бою видит уже исправленный текст,
и переобучать CRF отдельно из-за этого слоя НЕ нужно.

Модуль самостоятельный: не импортирует и не меняет labeling.py / model_crf.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Близость букв на клавиатуре (ЙЦУКЕН + qwerty) — «б» рядом с «ь», «о» с «р»
# ---------------------------------------------------------------------------

_KEYBOARD_ROWS = [
    "йцукенгшщзхъ",
    "фывапролджэ",
    "ячсмитьбю",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "1234567890",
]

# сосед -> множество соседних клавиш (в т.ч. по диагонали в пределах строки)
def _build_neighbors() -> Dict[str, set]:
    nb: Dict[str, set] = {}
    for row in _KEYBOARD_ROWS:
        for i, ch in enumerate(row):
            nb.setdefault(ch, set())
            if i > 0:
                nb[ch].add(row[i - 1])
            if i < len(row) - 1:
                nb[ch].add(row[i + 1])
    return nb


_NEIGHBORS = _build_neighbors()


def keyboard_adjacent(a: str, b: str) -> bool:
    """Стоят ли буквы рядом на клавиатуре (для «дешёвой» замены)."""
    a, b = a.lower(), b.lower()
    return b in _NEIGHBORS.get(a, set())


# ---------------------------------------------------------------------------
# 2. Взвешенное расстояние Левенштейна: замена на соседнюю клавишу дешевле
# ---------------------------------------------------------------------------

def weighted_edit_distance(s1: str, s2: str, *, sub_adjacent: float = 0.5) -> float:
    """
    Расстояние редактирования, где замена на соседнюю по клавиатуре букву
    стоит `sub_adjacent` (дешевле обычной замены = 1.0). Вставка/удаление = 1.0.
    """
    s1 = s1.lower().replace("ё", "е")
    s2 = s2.lower().replace("ё", "е")
    n, m = len(s1), len(s2)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    prev = list(range(m + 1))
    cur = [0.0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = float(i)
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                sub_cost = 0.0
            elif keyboard_adjacent(s1[i - 1], s2[j - 1]):
                sub_cost = sub_adjacent
            else:
                sub_cost = 1.0
            cur[j] = min(
                prev[j] + 1.0,        # удаление
                cur[j - 1] + 1.0,     # вставка
                prev[j - 1] + sub_cost,  # замена
            )
        prev, cur = cur, prev
    return prev[m]


def similarity_score(token: str, candidate: str) -> float:
    """
    Нормированный score похожести 0..1: 1 — идентичны.
    Учитывает взвешенное расстояние (клавиатурные замены дешевле).
    """
    if not token or not candidate:
        return 0.0
    dist = weighted_edit_distance(token, candidate)
    max_len = max(len(token), len(candidate))
    return 1.0 - dist / max_len


# ---------------------------------------------------------------------------
# 3. Словарь единиц измерения (частые опечатки типа «гь» -> «гб»)
# ---------------------------------------------------------------------------

UNIT_VOCAB = ["гб", "тб", "мб", "кб", "вт", "квт", "мач", "гц", "мгц",
              "кг", "мм", "см", "мл", "дюйм", "gb", "tb", "mb", "w"]


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("ё", "е")


@dataclass
class SpellFixer:
    """
    Исправляет опечатки в словах, опираясь на словари правильных слов.

    vocab: список канонов (категории + бренды + единицы). Каждый канон —
    строка (может быть многословной, но для пословной коррекции берём
    однословные ключи).
    """

    vocab: List[str] = field(default_factory=list)
    # порог принятия замены (0..1). Ниже — оставляем токен как есть.
    accept_score: float = 0.8
    # не трогать токены короче этого (слишком легко «перекорректировать»)
    min_len: int = 4
    # максимум правок относительно длины (чтобы не менять слово целиком)
    max_edit_ratio: float = 0.34

    _by_len: Dict[int, List[str]] = field(default_factory=dict, repr=False)
    _vocab_set: set = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        words = {_norm(w) for w in self.vocab if _norm(w)}
        # только однословные ключи для пословной коррекции
        single = {w for w in words if " " not in w}
        single.update(UNIT_VOCAB)
        self._vocab_set = single
        self._by_len = {}
        for w in single:
            self._by_len.setdefault(len(w), []).append(w)

    # ---- фабрики -----------------------------------------------------------
    @classmethod
    def from_files(cls, *paths: Path | str, **kwargs) -> "SpellFixer":
        vocab: List[str] = []
        for p in paths:
            p = Path(p)
            if p.exists():
                vocab += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return cls(vocab=vocab, **kwargs)

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path | str = "artifacts", **kwargs) -> "SpellFixer":
        d = Path(artifacts_dir)
        cats = d / "dicts" / "categories.txt"
        brands = d / "dicts" / "brands.txt"
        if not cats.exists():
            cats = d / "categories.txt"
        if not brands.exists():
            brands = d / "brands.txt"
        return cls.from_files(cats, brands, **kwargs)

    # ---- ядро --------------------------------------------------------------
    def best_candidate(self, token: str) -> Tuple[Optional[str], float]:
        """Лучший кандидат-канон для токена + его score (0..1)."""
        t = _norm(token)
        if not t or len(t) < self.min_len:
            return None, 0.0
        if t in self._vocab_set:
            return t, 1.0  # уже правильное слово

        # кандидаты: длина ±2 (опечатки редко меняют длину сильно)
        cand: List[str] = []
        for L in range(len(t) - 2, len(t) + 3):
            cand.extend(self._by_len.get(L, ()))
        if not cand:
            return None, 0.0

        best, best_score = None, 0.0
        for c in cand:
            sc = similarity_score(t, c)
            if sc > best_score:
                best, best_score = c, sc
        return best, best_score

    def fix_token(self, token: str) -> Tuple[str, Optional[Dict]]:
        """
        Возвращает (исправленный_токен, инфо_или_None).
        Если замена не прошла порог — токен без изменений, инфо None.
        """
        t = _norm(token)
        cand, score = self.best_candidate(token)
        if cand is None or cand == t:
            return token, None
        if score < self.accept_score:
            return token, None
        # не переписываем слово целиком: ограничение по числу правок
        dist = weighted_edit_distance(t, cand)
        if dist > max(1.0, len(t) * self.max_edit_ratio):
            return token, None
        info = {"from": token, "to": cand, "score": round(score, 3)}
        return cand, info

    # опечатка единицы, приклеенной к числу: 16гь -> 16 гб
    _GLUED_RE = re.compile(r"^(?P<num>\d+)(?P<unit>[а-яёa-z]{1,5})$", re.IGNORECASE)
    # короткие единицы чиним мягче: одна клавиатурная замена на длине 2 = score 0.75
    _UNIT_ACCEPT = 0.7

    def _correct_unit(self, unit: str) -> Tuple[Optional[str], float]:
        """Лучшая единица-канон для (возможно, опечатанного) хвоста единицы."""
        u0 = _norm(unit)
        if u0 in UNIT_VOCAB:
            return u0, 1.0
        best, best_score = None, 0.0
        for u in UNIT_VOCAB:
            sc = similarity_score(u0, u)
            if sc > best_score:
                best, best_score = u, sc
        return best, best_score

    def _fix_glued_unit(self, token: str) -> Tuple[str, Optional[Dict]]:
        """16гь -> 16 гб: хвост после числа похож на единицу — чиним и отклеиваем."""
        m = self._GLUED_RE.match(_norm(token))
        if not m:
            return token, None
        num, unit = m.group("num"), m.group("unit")
        best, score = self._correct_unit(unit)
        if best and score >= self._UNIT_ACCEPT:
            to = f"{num} {best}"
            if _norm(unit) == best:
                to = f"{num} {best}"  # только отклейка, без «исправления»
            info = {"from": token, "to": to, "score": round(score, 3)}
            return to, info
        return token, None

    def _fix_standalone_unit(self, token: str) -> Tuple[str, Optional[Dict]]:
        """Отдельный токен-единица с опечаткой: 'гь' -> 'гб' (только если это похоже на единицу)."""
        t = _norm(token)
        if len(t) > 4 or t in self._vocab_set and t not in UNIT_VOCAB:
            return token, None
        best, score = self._correct_unit(t)
        if best and best != t and score >= self._UNIT_ACCEPT:
            info = {"from": token, "to": best, "score": round(score, 3)}
            return best, info
        return token, None

    def fix_query(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Исправляет опечатки во всём запросе.
        Возвращает (исправленная_строка, список_замен).
        Числа не корректируются; единицы у числа (16гь) чинятся и отклеиваются.
        """
        if not query:
            return query, []
        parts = re.findall(r"\w+|\W+", query, flags=re.UNICODE)
        fixed_parts: List[str] = []
        changes: List[Dict] = []
        prev_word_is_number = False
        for p in parts:
            if not p.strip() or not re.match(r"\w+", p, flags=re.UNICODE):
                fixed_parts.append(p)
                continue
            if p.isdigit():  # чистые числа не трогаем
                fixed_parts.append(p)
                prev_word_is_number = True
                continue
            # число + (опечатка единицы) одним токеном: 16гь -> 16 гб
            if re.match(r"^\d+[а-яёa-z]", _norm(p)):
                new, info = self._fix_glued_unit(p)
                fixed_parts.append(new if info else p)
                if info:
                    changes.append(info)
                prev_word_is_number = False
                continue
            # отдельная единица сразу после числа: "16 гь" -> "16 гб"
            if prev_word_is_number:
                new, info = self._fix_standalone_unit(p)
                if info:
                    fixed_parts.append(new)
                    changes.append(info)
                    prev_word_is_number = False
                    continue
            new, info = self.fix_token(p)
            fixed_parts.append(new if info else p)
            if info:
                changes.append(info)
            prev_word_is_number = False
        return "".join(fixed_parts), changes


__all__ = [
    "SpellFixer",
    "weighted_edit_distance",
    "similarity_score",
    "keyboard_adjacent",
]
