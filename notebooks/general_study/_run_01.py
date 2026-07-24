"""General study — шаг 1: единый silver-датасет + синтетические "битые" запросы.

Общий источник данных для _run_02.py (CRF + GLiNER обучаются на ОДНОМ и том же silver,
а не каждый на своём) плюс отдельный held-out набор с опечатками (не участвует
в обучении) — специально под "появились broken queries" из artifacts/history/broken_queries.md.

Пайплайн: query -> SpellFixer v2 (typo + units + homoglyphs + алиасы транслитерации)
-> WeakLabeler (BRAND/CATEGORY/MODEL/ATTR) -> silver BIO.

Не трогает notebooks/crf_ner_classifier и notebooks/gliner — отдельная, самостоятельная
копия данных под этот joint-experiment.
"""
from __future__ import annotations

import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import pandas as pd

from src.data_utils import (
    ARTIFACTS_DIR,
    DARK_SLATE,
    FIGURES_DIR,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    brands_path,
    categories_path,
    model_phrases_path,
    load_query_clicks,
)
from src.ner.labeling import WeakLabeler, bio_to_entities, tokenize
from src.preprocessing.pipeline import basic_clean, _norm_key
from src.preprocessing.spellfix import SpellFixer

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
SAMPLE_N = 60_000
MAX_QUERIES = 6_000  # чуть больше, чем у crf_ner_classifier (5000) — общий silver для двух моделей
BROKEN_N = 400  # сколько битых запросов сгенерировать для стресс-теста

OUT = ARTIFACTS_DIR / "silver" / "general_study"
FIG = FIGURES_DIR / "general_study"
REPORT = Path(__file__).resolve().parent / "01_general_silver_report.md"


def log(msg: str) -> None:
    print(msg, flush=True)


def entity_counts(tags: list[str]) -> Counter:
    c: Counter = Counter()
    for t in tags:
        if t.startswith("B-"):
            c[t[2:]] += 1
    return c


def sent_to_row(query: str, tags_pairs: list[tuple[str, str]], source: str) -> dict:
    tokens = [t for t, _ in tags_pairs]
    tags = [g for _, g in tags_pairs]
    ents = bio_to_entities(tags_pairs, query=query)
    ec = entity_counts(tags)
    return {
        "query": query,
        "n_tokens": len(tokens),
        "tokens": tokens,
        "tags": tags,
        "n_entities": sum(1 for t in tags if t.startswith("B-")),
        "n_BRAND": ec["BRAND"],
        "n_CATEGORY": ec["CATEGORY"],
        "n_MODEL": ec["MODEL"],
        "n_ATTR": ec["ATTR"],
        "has_entity": any(t != "O" for t in tags),
        "bio_str": " ".join(f"{a}/{b}" for a, b in tags_pairs),
        "entities_json": json.dumps(
            [{"text": e["text"], "label": e["label"]} for e in ents], ensure_ascii=False
        ),
        "source": source,
    }


# ---------------------------------------------------------------------------
# Синтетические "битые" запросы: корёжим токены, но СОХРАНЯЕМ их количество,
# чтобы BIO-теги (по токенам) остались валидны без пересчёта спанов.
# ---------------------------------------------------------------------------

_KEYBOARD_NEIGHBORS = {
    "а": "оыв", "о": "аыр", "е": "ирп", "и": "еыу", "у": "ицк", "ы": "аосв",
    "с": "ачм", "к": "угп", "н": "гоь", "т": "ер", "р": "оеп", "л": "од",
    "в": "аыц", "м": "сит", "п": "рек", "д": "лж", "я": "чс", "з": "хй",
    "х": "зъ", "ж": "дэ", "б": "юь", "ю": "б", "г": "нк", "ш": "щи", "щ": "шз",
    "a": "sq", "s": "adw", "d": "sfe", "f": "dgr", "g": "fht", "h": "gjy",
    "j": "hku", "k": "jli", "l": "ko", "q": "wa", "w": "qes", "e": "wrd",
    "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl",
    "z": "x", "x": "zc", "c": "xv", "v": "cb", "b": "vn", "n": "bm", "m": "n",
}
_HOMOGLYPH_INJECT = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у"}
_CORRUPT_OPS = ["keyboard", "homoglyph", "duplicate", "drop"]


def _corrupt_token(tok: str, rng: random.Random) -> tuple[str, bool]:
    """Одна случайная порча токена (typo/раскладка/гомоглиф/фэт-фингер). len(токена) может
    меняться, но КОЛИЧЕСТВО токенов в запросе — нет (порча внутри одной строки-токена)."""
    if len(tok) < 3 or not tok.isalpha():
        return tok, False
    op = rng.choice(_CORRUPT_OPS)
    pos = rng.randrange(1, len(tok) - 1) if len(tok) > 2 else 0
    ch = tok[pos]
    if op == "keyboard":
        neigh = _KEYBOARD_NEIGHBORS.get(ch.lower())
        if not neigh:
            return tok, False
        rep = rng.choice(neigh)
        rep = rep.upper() if ch.isupper() else rep
        return tok[:pos] + rep + tok[pos + 1 :], True
    if op == "homoglyph":
        rep = _HOMOGLYPH_INJECT.get(ch.lower())
        if not rep:
            return tok, False
        rep = rep.upper() if ch.isupper() else rep
        return tok[:pos] + rep + tok[pos + 1 :], True
    if op == "duplicate":
        return tok[: pos + 1] + tok[pos] + tok[pos + 1 :], True
    if op == "drop":
        return tok[:pos] + tok[pos + 1 :], True
    return tok, False


def corrupt_row(tokens: list[str], rng: random.Random, p: float = 0.5) -> tuple[list[str], int]:
    """Портит ~p долю подходящих токенов строки. Возвращает (новые_токены, сколько_испорчено)."""
    new_tokens = list(tokens)
    n_changed = 0
    for i, tok in enumerate(tokens):
        if rng.random() < p:
            new_tok, changed = _corrupt_token(tok, rng)
            if changed:
                new_tokens[i] = new_tok
                n_changed += 1
    return new_tokens, n_changed


def build_broken_queries(silver_ent: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pool = silver_ent[silver_ent["n_tokens"] >= 2].copy()
    if len(pool) > n:
        pool = pool.sample(n=n, random_state=seed)
    rows = []
    for _, r in pool.iterrows():
        tokens = list(r["tokens"])
        tags = list(r["tags"])
        new_tokens, n_changed = corrupt_row(tokens, rng, p=0.5)
        if n_changed == 0:
            continue
        query_broken = " ".join(new_tokens)
        # пересобираем токенизацию заново — sanity: длины должны совпасть 1:1
        retok = [t for t, _, _ in tokenize(query_broken)]
        if len(retok) != len(tags):
            continue
        rows.append(
            {
                "query": query_broken,
                "query_orig": r["query"],
                "tokens": retok,
                "tags": tags,
                "n_corrupted_tokens": n_changed,
                "n_tokens": len(retok),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    labeler = WeakLabeler.from_files(
        brands_path(), categories_path(), models_path=model_phrases_path()
    )
    log(f"dicts brands={len(labeler.brands)} cats={len(labeler.categories)} models={len(labeler.models)}")

    spell = SpellFixer.from_artifacts(ARTIFACTS_DIR)
    log(f"spell vocab={len(spell._vocab_set)} aliases={len(spell._alias_to_canon)}")

    clicks = load_query_clicks()
    if len(clicks) > SAMPLE_N:
        clicks = clicks.sample(n=SAMPLE_N, random_state=SEED)
    qcol = "query_text" if "query_text" in clicks.columns else "query"
    queries = (
        clicks[qcol]
        .astype(str)
        .map(lambda x: basic_clean(x, lowercase=False))
        .map(_norm_key)
        .drop_duplicates()
    )
    queries = [q for q in queries.tolist() if len(q) >= 2][:MAX_QUERIES]
    log(f"unique queries: {len(queries)}")

    n_spell_fixed = 0
    fixed_queries: list[str] = []
    spell_examples: list[dict] = []
    for q in queries:
        q2, changes = spell.fix_query(q)
        if changes:
            n_spell_fixed += 1
            if len(spell_examples) < 20:
                spell_examples.append({"before": q, "after": q2, "changes": changes})
        fixed_queries.append(q2)
    log(f"spellfix v2 touched {n_spell_fixed}/{len(fixed_queries)} queries")

    rows = []
    for i, q in enumerate(fixed_queries):
        if i and i % 1000 == 0:
            log(f"  labeled {i}/{len(fixed_queries)}")
        sent = labeler.label_query(q)
        if sent:
            rows.append(sent_to_row(q, sent, "general_silver"))
    silver = pd.DataFrame(rows)
    silver_ent = silver[silver["has_entity"]].copy()
    log(f"silver rows={len(silver)} with_entity={len(silver_ent)}")

    broken = build_broken_queries(silver_ent, BROKEN_N, SEED)
    log(f"broken_queries_eval: {len(broken)} rows (corrupted, held out, NOT for training)")

    # --- save ---
    silver.to_parquet(OUT / "general_silver_bio.parquet", index=False)
    silver_ent.to_parquet(OUT / "general_silver_bio_ent.parquet", index=False)
    broken.to_parquet(OUT / "broken_queries_eval.parquet", index=False)
    log(f"saved -> {OUT}")

    type_counts = Counter()
    for _, r in silver_ent.iterrows():
        for lab in ("BRAND", "CATEGORY", "MODEL", "ATTR"):
            type_counts[lab] += int(r[f"n_{lab}"])

    meta = {
        "seed": SEED,
        "max_queries": MAX_QUERIES,
        "n_silver": len(silver),
        "n_with_entity": len(silver_ent),
        "n_broken_queries": len(broken),
        "n_spell_fixed": n_spell_fixed,
        "entity_counts": dict(type_counts),
        "note": "joint silver для _run_02 (CRF + GLiNER); spellfix v2 = typo+units+homoglyphs+aliases",
    }
    (OUT / "eda_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- plot ---
    labs = ["BRAND", "CATEGORY", "MODEL", "ATTR"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(labs, [type_counts[l] for l in labs], color=[MVIDEO_RED, DARK_SLATE, "#5B8C5A", "#C4A35A"])
    ax.set_title("General silver — entity counts (spellfix v2)")
    fig.tight_layout()
    fig.savefig(FIG / "01_entity_counts.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- report ---
    lines = [
        "# 01. General study — joint silver + broken queries",
        "",
        f"Queries сэмплированы: **{len(queries)}**, silver с сущностями: **{len(silver_ent)}**.",
        f"SpellFix v2 (typo + units + homoglyphs + алиасы) тронул **{n_spell_fixed}/{len(fixed_queries)}** запросов.",
        f"Синтетический broken_queries_eval: **{len(broken)}** строк "
        "(порча токенов: раскладка/гомоглиф/дубль/пропуск буквы; количество токенов сохранено — теги валидны).",
        "",
        "## Entity counts (with_entity slice)",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    for l in labs:
        lines.append(f"| {l} | {type_counts[l]} |")
    lines += [
        "",
        "![entities](../../figures/general_study/01_entity_counts.png)",
        "",
        "## SpellFix v2 — примеры (before -> after)",
        "",
        "| before | after | changes |",
        "|---|---|---|",
    ]
    for e in spell_examples[:12]:
        ch = ", ".join(f"{c.get('from')}→{c.get('to')}" for c in e["changes"])
        lines.append(f"| `{e['before']}` | `{e['after']}` | {ch} |")
    lines += [
        "",
        "## Broken queries — примеры (query_orig -> query битый)",
        "",
        "| orig | broken | n_corrupted |",
        "|---|---|---:|",
    ]
    for _, r in broken.head(12).iterrows():
        lines.append(f"| `{r['query_orig']}` | `{r['query']}` | {r['n_corrupted_tokens']} |")
    lines += [
        "",
        "## Дальше",
        "",
        "`python notebooks/general_study/_run_02.py` — обучение CRF + GLiNER на этом silver "
        "и метрика **всего каскада** (rules → CRF → GLiNER) на gold и на broken_queries_eval.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE report={REPORT}")


if __name__ == "__main__":
    main()
