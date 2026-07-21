#!/usr/bin/env python
"""Быстрая проверка целостности skus.pkl (YML-каталог М.Видео).

Отвечает на вопрос «а данные-то нормально легли в pickle?»:
структура, число офферов, наличие ключевых полей (vendor/model/param/picture),
покрытие атрибутов и явные признаки битой кодировки.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

# Юникод в консоли Windows (cp1251) не должен ронять скрипт.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import SKUS_PKL_PATH  # noqa: E402


def find_offers(raw: object) -> list:
    """Достаёт список офферов из произвольной вложенности YML-дампа."""
    stack = [raw]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in ("offers", "offer"):
                if key in node:
                    val = node[key]
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        return val
                    if isinstance(val, dict) and isinstance(val.get("offer"), list):
                        return val["offer"]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return []


def main() -> None:
    print(f"Файл: {SKUS_PKL_PATH}")
    print(f"Размер: {SKUS_PKL_PATH.stat().st_size / 1e6:.1f} МБ")
    print("Загружаю pickle (может занять минуту)...")

    with open(SKUS_PKL_PATH, "rb") as f:
        raw = pickle.load(f)

    print(f"Тип верхнего уровня: {type(raw).__name__}")
    if isinstance(raw, dict):
        print(f"Ключи верхнего уровня: {list(raw.keys())}")

    offers = find_offers(raw)
    print(f"\nНайдено офферов: {len(offers)}")
    if not offers:
        print("ВНИМАНИЕ: офферы не найдены — структура не та, что ждали.")
        return

    field_counter: Counter[str] = Counter()
    has_picture = has_param = has_vendor = has_model = 0
    param_names: Counter[str] = Counter()
    empty_names = 0
    sample = None

    for o in offers:
        if not isinstance(o, dict):
            continue
        for k in o:
            field_counter[k] += 1
        if o.get("picture"):
            has_picture += 1
        if o.get("vendor"):
            has_vendor += 1
        if o.get("model"):
            has_model += 1
        params = o.get("param")
        if params:
            has_param += 1
            if isinstance(params, list):
                for p in params:
                    if isinstance(p, dict):
                        param_names[p.get("@name", "?")] += 1
        if not o.get("name"):
            empty_names += 1
        if sample is None and o.get("param"):
            sample = o

    n = len(offers)
    print("\nПокрытие ключевых полей (доля офферов):")
    for name, cnt in [
        ("name", n - empty_names),
        ("vendor", has_vendor),
        ("model", has_model),
        ("param", has_param),
        ("picture", has_picture),
    ]:
        print(f"  {name:10s}: {cnt/n:6.1%}  ({cnt})")

    print("\nТоп-15 атрибутов в param:")
    for name, cnt in param_names.most_common(15):
        print(f"  {cnt:7d}  {name}")

    if sample is not None:
        print("\nПример оффера (обрезано):")
        for k in ("@id", "vendor", "model", "name", "picture"):
            v = sample.get(k)
            if isinstance(v, str) and len(v) > 90:
                v = v[:90] + "…"
            print(f"  {k}: {v}")

    # Грубый детектор битой кодировки (mojibake).
    text = " ".join(str(o.get("name", "")) for o in offers[:2000] if isinstance(o, dict))
    suspicious = sum(text.count(ch) for ch in "ÐÑ�Ã")
    print(f"\nПодозрительных символов (mojibake) в 2000 именах: {suspicious}")
    print("Кириллица читается ок" if suspicious < 20 else "ВОЗМОЖНА битая кодировка!")


if __name__ == "__main__":
    main()
