"""CLI для тестирования полного каскада извлечения фактов из поисковых запросов.

Примеры:
  python -m src.cli.cli "ноутбук asus zenbook 16 гб"
  python -m src.cli.cli -i
  python -m src.cli.cli -f queries.txt --debug
  python -m src.cli.cli --demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.extractor import QueryEntityExtractor  # noqa: E402

DEMO_QUERIES = [
    "ноутбук asus zenbook 16 гб серый",
    "пылесос dyson v15",
    "айфон 15 pro max",
    "беспроводные наушники sony",
    "телевизор samsung 55 дюйм",
    "холодильник",
    "кофемашина delonghi для капсул",
]


def load_extractor(
    artifacts: Path | str | None = None,
    models: Path | str | None = None,
) -> QueryEntityExtractor:
    art = Path(artifacts) if artifacts else ROOT / "artifacts"
    mod = Path(models) if models else ROOT / "models"
    return QueryEntityExtractor.from_artifacts(art, mod)


def format_pretty(result: dict[str, Any], *, debug: bool = False) -> str:
    lines = [
        f"query:      {result.get('query')!r}",
        f"brand:      {result.get('brand')}",
        f"category:   {result.get('category')}",
        f"model:      {result.get('model')}",
        f"attributes: {json.dumps(result.get('attributes') or {}, ensure_ascii=False)}",
        f"latency:    {result.get('latency_ms')} ms",
    ]
    ents = result.get("entities") or []
    if ents:
        lines.append("entities:")
        for e in ents:
            span = e.get("span")
            extra = f" type={e.get('attr_type')}" if e.get("attr_type") else ""
            lines.append(
                f"  - {e.get('label'):8} {e.get('text')!r}"
                f"{f'  [{span[0]}:{span[1]}]' if span else ''}{extra}"
            )
    if debug and result.get("debug"):
        d = result["debug"]
        lines.append("debug:")
        lines.append(
            f"  layers: crf={d.get('has_crf')} brand_clf={d.get('has_brand_clf')} "
            f"attr_clf={d.get('has_attr_clf')} "
            f"dicts B/C/M={d.get('n_brands_dict')}/{d.get('n_categories_dict')}/{d.get('n_models_dict')}"
        )
        if d.get("dict_bio"):
            bio = " ".join(f"{x['token']}/{x['tag']}" for x in d["dict_bio"])
            lines.append(f"  rules BIO: {bio}")
        if d.get("crf_bio"):
            bio = " ".join(f"{x['token']}/{x['tag']}" for x in d["crf_bio"])
            lines.append(f"  CRF   BIO: {bio}")
    return "\n".join(lines)


def run_one(
    ex: QueryEntityExtractor,
    query: str,
    *,
    debug: bool = False,
    as_json: bool = False,
) -> dict[str, Any]:
    result = ex.extract_debug(query) if debug else ex.extract(query)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_pretty(result, debug=debug))
        print("-" * 60)
    return result


def run_interactive(ex: QueryEntityExtractor, *, debug: bool = False, as_json: bool = False) -> None:
    print("M.Video NER CLI — полный каскад (rules → CRF → brand clf → ATTR typer)")
    print("Команды: пустая строка / exit / quit — выход; :debug — переключить debug")
    print("-" * 60)
    dbg = debug
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            break
        if q == ":debug":
            dbg = not dbg
            print(f"debug={'ON' if dbg else 'OFF'}")
            continue
        run_one(ex, q, debug=dbg, as_json=as_json)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cli.cli",
        description="Извлечение фактов из поисковых запросов (полный MVP-каскад)",
    )
    p.add_argument("query", nargs="?", help="Один запрос (если не -i / -f / --demo)")
    p.add_argument("-i", "--interactive", action="store_true", help="REPL")
    p.add_argument("-f", "--file", type=Path, help="Файл: один запрос на строку")
    p.add_argument("--demo", action="store_true", help="Прогнать демо-набор запросов")
    p.add_argument("--debug", action="store_true", help="BIO rules/CRF + флаги слоёв")
    p.add_argument("--json", action="store_true", help="Сырой JSON на stdout")
    p.add_argument("--artifacts", type=Path, default=None, help="Папка artifacts/")
    p.add_argument("--models", type=Path, default=None, help="Папка models/")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print("Loading extractor (dicts + 3 models)…", file=sys.stderr)
    ex = load_extractor(args.artifacts, args.models)
    status = (
        f"ready: crf={ex.ner_model is not None} "
        f"brand_clf={ex.brand_classifier is not None} "
        f"attr_clf={ex.use_attr_clf} "
        f"dicts B/C/M={len(ex.labeler.brands)}/{len(ex.labeler.categories)}/{len(ex.labeler.models)}"
    )
    print(status, file=sys.stderr)

    if args.interactive or (
        not args.query and not args.file and not args.demo
    ):
        # no args → interactive by default
        if not args.query and not args.file and not args.demo and not args.interactive:
            # if literally nothing: show help-ish and enter REPL
            pass
        run_interactive(ex, debug=args.debug, as_json=args.json)
        return 0

    queries: list[str] = []
    if args.demo:
        queries.extend(DEMO_QUERIES)
    if args.file:
        text = args.file.read_text(encoding="utf-8")
        queries.extend(ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#"))
    if args.query:
        queries.append(args.query)

    for q in queries:
        run_one(ex, q, debug=args.debug, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
