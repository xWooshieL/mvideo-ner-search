"""Ручной аудит качества датасета кейса M.Video.

Ищем следы человеческих ошибок при сборке:
  - перепутанные местами колонки (brand в name, цена как id и т.п.)
  - транслит / смесь языков (латиница в кириллических полях и наоборот)
  - дубли брендов из-за регистра/раскладки
  - несоответствие sku_brand_name и того, что реально в sku_name
  - аномалии цен, позиций, id
Печатает связный отчёт с примерами.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from src.data_utils import load_query_clicks  # noqa: E402

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 160)

CYR = re.compile(r"[а-яё]", re.I)
LAT = re.compile(r"[a-z]", re.I)
DIGIT = re.compile(r"\d")


def has_cyr(s: str) -> bool:
    return bool(CYR.search(s))


def has_lat(s: str) -> bool:
    return bool(LAT.search(s))


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    print("Загружаю семпл query_clicks (600k строк)...")
    df = load_query_clicks(n=600_000, seed=7)
    df["query_text"] = df["query_text"].astype("string")
    df["sku_name"] = df["sku_name"].astype("string")
    df["sku_brand_name"] = df["sku_brand_name"].astype("string")

    sep("0. СХЕМА И БАЗОВАЯ ЦЕЛОСТНОСТЬ")
    print("Колонки:", list(df.columns))
    print("\nТипы:")
    print(df.dtypes)
    print("\nПропуски (кол-во):")
    print(df.isna().sum())
    print("\nПустые строки '' по текстовым полям:")
    for c in ["query_text", "sku_name", "sku_brand_name"]:
        empty = (df[c].fillna("").str.strip() == "").sum()
        print(f"  {c}: {empty}")

    # ------------------------------------------------------------------
    sep("1. ЦЕНЫ: аномалии (0, отрицательные, гигантские)")
    p = df["sku_price"]
    print(p.describe())
    print("\nЦена <= 0:", int((p <= 0).sum()))
    print("Цена > 3 млн ₽:", int((p > 3_000_000).sum()))
    if (p <= 0).any():
        print("\nПримеры с ценой <= 0:")
        print(df.loc[p <= 0, ["query_text", "sku_name", "sku_brand_name", "sku_price"]].head(8).to_string(index=False))
    if (p > 3_000_000).any():
        print("\nПримеры с ценой > 3 млн:")
        print(df.loc[p > 3_000_000, ["sku_name", "sku_brand_name", "sku_price"]].head(8).to_string(index=False))

    # ------------------------------------------------------------------
    sep("2. ПОЗИЦИЯ КЛИКА: аномалии")
    if "sku_position" in df.columns:
        pos = df["sku_position"]
        print(pos.describe())
        print("Позиция <= 0:", int((pos <= 0).sum()))
        print("Позиция > 1000:", int((pos > 1000).sum()))

    # ------------------------------------------------------------------
    sep("3. ПЕРЕПУТАННЫЕ КОЛОНКИ: бренд похож на цену/число или наоборот")
    brand = df["sku_brand_name"].fillna("")
    # бренд, состоящий только из цифр -> подозрение на смещение колонок
    brand_numeric = df[brand.str.match(r"^\s*\d[\d\s.,]*$") & (brand.str.strip() != "")]
    print("Бренд состоит только из цифр:", len(brand_numeric))
    if len(brand_numeric):
        print(brand_numeric[["query_text", "sku_name", "sku_brand_name", "sku_price"]].head(8).to_string(index=False))

    # sku_name, состоящий только из цифр
    name = df["sku_name"].fillna("")
    name_numeric = df[name.str.match(r"^\s*\d[\d\s.,]*$") & (name.str.strip() != "")]
    print("\nНазвание состоит только из цифр:", len(name_numeric))
    if len(name_numeric):
        print(name_numeric[["query_text", "sku_name", "sku_brand_name"]].head(8).to_string(index=False))

    # query, подозрительно совпадающий с брендом целиком (мог быть вставлен не туда)
    # и очень длинный query (похоже на название товара, попавшее в поле запроса)
    qlen = df["query_text"].fillna("").str.len()
    long_q = df[qlen > 60]
    print("\nОчень длинные запросы (>60 симв., возможно туда попало название товара):", len(long_q))
    if len(long_q):
        print(long_q[["query_text", "sku_name"]].head(6).to_string(index=False))

    # ------------------------------------------------------------------
    sep("4. БРЕНД НЕ СОВПАДАЕТ С НАЗВАНИЕМ (первое слово названия != бренд)")
    # берём непустые бренды с латиницей/кириллицей, где бренд явно не встречается в названии
    sub = df.dropna(subset=["sku_brand_name", "sku_name"]).copy()
    sub = sub[sub["sku_brand_name"].str.strip() != ""]
    sub = sub.drop_duplicates(subset=["sku_name", "sku_brand_name"])

    def brand_in_name(row) -> bool:
        b = str(row["sku_brand_name"]).lower().strip()
        n = str(row["sku_name"]).lower()
        if not b:
            return True
        # точное вхождение или по первому токену бренда
        return b in n or b.split()[0] in n

    sub["brand_hit"] = sub.apply(brand_in_name, axis=1)
    mism = sub[~sub["brand_hit"]]
    print(f"Уникальных пар name/brand: {len(sub)}, бренд НЕ найден в названии: {len(mism)} ({len(mism)/max(len(sub),1):.1%})")
    print("\nПримеры (бренд отсутствует в тексте названия):")
    print(mism[["sku_name", "sku_brand_name"]].head(12).to_string(index=False))

    # ------------------------------------------------------------------
    sep("5. ТРАНСЛИТ / СМЕСЬ ЯЗЫКОВ")
    brands = df["sku_brand_name"].dropna()
    brands = brands[brands.str.strip() != ""]
    ub = brands.drop_duplicates()
    mixed = ub[ub.apply(lambda s: has_cyr(s) and has_lat(s))]
    print(f"Бренды со СМЕСЬЮ кириллицы+латиницы (подозрение на транслит/опечатку): {len(mixed)}")
    print(list(mixed.head(30)))

    # запросы: латиница-транслит русских слов и наоборот
    q = df["query_text"].dropna()
    q = q[q.str.strip() != ""]
    uq = q.drop_duplicates()
    mixed_q = uq[uq.apply(lambda s: has_cyr(s) and has_lat(s))]
    print(f"\nЗапросы со смесью языков: {len(mixed_q)} из {len(uq)} уникальных")
    print(list(mixed_q.head(25)))

    # ------------------------------------------------------------------
    sep("6. ДУБЛИ БРЕНДОВ ИЗ-ЗА РЕГИСТРА / ПРОБЕЛОВ / РАСКЛАДКИ")
    norm_map = defaultdict(set)
    for b in ub:
        key = re.sub(r"\s+", "", str(b).lower())
        norm_map[key].add(str(b))
    dup_case = {k: v for k, v in norm_map.items() if len(v) > 1}
    print(f"Группы брендов, различающихся только регистром/пробелами: {len(dup_case)}")
    for k, v in list(dup_case.items())[:20]:
        print(f"  {sorted(v)}")

    # похожие бренды на разных раскладках (кириллица <-> латиница-омоглифы)
    homo = str.maketrans("асекорхувтмн", "acekopxybtmh")  # кир -> лат по виду
    latin_view = defaultdict(set)
    for b in ub:
        s = str(b).lower()
        lv = s.translate(homo)
        latin_view[re.sub(r"\s+", "", lv)].add(str(b))
    homo_dups = {k: v for k, v in latin_view.items() if len(v) > 1}
    print(f"\nВозможные омоглифы (кир/лат буквы визуально одинаковы): {len(homo_dups)}")
    for k, v in list(homo_dups.items())[:20]:
        # показываем только те, где реально разные наборы символов
        if len({re.sub(r'\s+', '', x.lower()) for x in v}) > 1:
            print(f"  {sorted(v)}")

    # ------------------------------------------------------------------
    sep("7. ДУБЛИ СТРОК И query->несколько брендов")
    dup_rows = df.duplicated().sum()
    print("Полных дублей строк:", int(dup_rows))
    # один и тот же query -> сколько разных брендов кликают (норм), но ищем аномалию: query==brand
    q_eq_brand = df[(df["query_text"].str.lower().str.strip() ==
                     df["sku_brand_name"].str.lower().str.strip()) &
                    (df["query_text"].str.strip() != "")]
    print("query дословно == бренду (ок, но полезно знать):", len(q_eq_brand))


if __name__ == "__main__":
    main()
