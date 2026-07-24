"""CLI runner for latency SLA bench (same logic as tests.ipynb).

Usage (server must be up):
  python src/server/_run_latency.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:8000"
GOLD_PATH = ROOT / "artifacts" / "gold" / "bio_liza.jsonl"
OUT_JSON = ROOT / "artifacts" / "metrics" / "latency_sla.json"
OUT_CSV = ROOT / "artifacts" / "metrics" / "latency_sla_rows.csv"
OUT_PNG = ROOT / "artifacts" / "metrics" / "latency_sla_cdf.png"

EASY = [
    "samsung",
    "пылесос",
    "геймпад xbox",
    "nintendo switch 2",
    "телевизоры haier 50",
]
MEDIUM = [
    "телевизор samsung 55 дюймов",
    "ноутбук asus 16 гб",
    "смартфон xiaomi 128gb",
    "пылесос dyson v15",
    "холодильник indesit no frost",
    "наушники jbl tune 510",
    "стиральная машина bosch 8 кг",
    "монитор lg 27",
    "планшет samsung tab s9",
    "чайник redmond",
]
HARD = [
    "телфон 16 гь",
    "сони плейстейшен 5",
    "ноутбок asus 16гь",
    "планше тxiaomi",
    "айфон 16 про макс",
    "лэптоп ксяоми редми",
    "playstation5 sony slim",
    "тв самсунг 65 4k",
    "морозилка индесит 200л",
    "х0лодильник bosch",
]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=float), p))


def summarize(lat_server: list[float], lat_client: list[float], *, n_ok: int, n_err: int) -> dict:
    def block(xs: list[float]) -> dict:
        return {
            "n": len(xs),
            "mean_ms": round(float(np.mean(xs)), 3) if xs else None,
            "std_ms": round(float(np.std(xs)), 3) if xs else None,
            "min_ms": round(float(np.min(xs)), 3) if xs else None,
            "p50_ms": round(percentile(xs, 50), 3) if xs else None,
            "p90_ms": round(percentile(xs, 90), 3) if xs else None,
            "p95_ms": round(percentile(xs, 95), 3) if xs else None,
            "p99_ms": round(percentile(xs, 99), 3) if xs else None,
            "max_ms": round(float(np.max(xs)), 3) if xs else None,
        }

    total = n_ok + n_err
    return {
        "n_total": total,
        "n_ok": n_ok,
        "n_err": n_err,
        "success_rate": round(n_ok / total, 4) if total else 0.0,
        "server_latency_ms": block(lat_server),
        "client_rtt_ms": block(lat_client),
        "sla": {
            "target_p95_server_ms": 100.0,
            "p95_server_ok": bool(lat_server) and percentile(lat_server, 95) <= 100.0,
            "target_p95_client_ms": 150.0,
            "p95_client_ok": bool(lat_client) and percentile(lat_client, 95) <= 150.0,
        },
    }


def run_bench(name: str, queries: list[str], *, repeats: int, client: httpx.Client):
    rows = []
    lat_s, lat_c = [], []
    n_ok = n_err = 0
    for rep in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            try:
                r = client.get("/extract", params={"query": q})
                rtt = (time.perf_counter() - t0) * 1000.0
                r.raise_for_status()
                body = r.json()
                srv = float(body.get("latency_ms") or 0.0)
                n_ok += 1
                lat_s.append(srv)
                lat_c.append(rtt)
                rows.append(
                    {
                        "suite": name,
                        "repeat": rep,
                        "query": q,
                        "ok": True,
                        "server_ms": srv,
                        "client_ms": round(rtt, 3),
                        "brand": body.get("brand"),
                        "category": body.get("category"),
                        "model": body.get("model"),
                        "n_entities": len(body.get("entities") or []),
                        "n_spell_fixes": len(body.get("spell_fixes") or []),
                        "error": None,
                    }
                )
            except Exception as e:
                rtt = (time.perf_counter() - t0) * 1000.0
                n_err += 1
                rows.append(
                    {
                        "suite": name,
                        "repeat": rep,
                        "query": q,
                        "ok": False,
                        "server_ms": None,
                        "client_ms": round(rtt, 3),
                        "brand": None,
                        "category": None,
                        "model": None,
                        "n_entities": None,
                        "n_spell_fixes": None,
                        "error": str(e)[:200],
                    }
                )
    summary = summarize(lat_s, lat_c, n_ok=n_ok, n_err=n_err)
    summary["suite"] = name
    return summary, pd.DataFrame(rows)


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    gold_queries = [
        json.loads(line)["query"]
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        h = client.get("/health")
        h.raise_for_status()
        print("health:", h.json())
        for q in ["samsung", "пылесос dyson", "телефон"]:
            client.get("/extract", params={"query": q})
        print("warmup done")

        summaries = []
        all_rows = []
        for name, qs, reps in [
            ("demo_easy", EASY, 3),
            ("demo_medium", MEDIUM, 3),
            ("demo_hard", HARD, 3),
            ("gold", gold_queries, 1),
        ]:
            print(f"\n=== {name} (n={len(qs)} x{reps}) ===")
            s, df = run_bench(name, qs, repeats=reps, client=client)
            summaries.append(s)
            all_rows.append(df)
            srv = s["server_latency_ms"]
            print(
                f"ok={s['n_ok']}/{s['n_total']}  "
                f"server p50={srv['p50_ms']} p90={srv['p90_ms']} p95={srv['p95_ms']} "
                f"p99={srv['p99_ms']} max={srv['max_ms']}  "
                f"SLA p95<=100: {s['sla']['p95_server_ok']}"
            )

    rows_df = pd.concat(all_rows, ignore_index=True)
    payload = {
        "base_url": BASE_URL,
        "gold_path": str(GOLD_PATH.as_posix()),
        "suites": summaries,
        "demo_queries": {"easy": EASY, "medium": MEDIUM, "hard": HARD},
        "notes": (
            "server_latency_ms = extractor cascade; client_rtt_ms = HTTP RTT. "
            "Warmup excluded. Demo suites repeated 3x."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_df.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_JSON)
    print("wrote", OUT_CSV)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for suite, g in rows_df[rows_df["ok"]].groupby("suite"):
            xs = sorted(g["server_ms"].tolist())
            ys = np.linspace(0, 1, len(xs), endpoint=True)
            ax.plot(xs, ys, label=suite)
        ax.axvline(100, color="red", ls="--", lw=1, label="SLA p95=100ms")
        ax.set_xlabel("server latency_ms")
        ax.set_ylabel("CDF")
        ax.set_title("Latency CDF by suite")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        print("wrote", OUT_PNG)
    except Exception as e:
        print("plot skipped:", e)

    print("\n=== SLA table ===")
    for s in summaries:
        print(
            f"{s['suite']:12} n={s['n_ok']:3}  "
            f"p50={s['server_latency_ms']['p50_ms']:7}  "
            f"p90={s['server_latency_ms']['p90_ms']:7}  "
            f"p95={s['server_latency_ms']['p95_ms']:7}  "
            f"p99={s['server_latency_ms']['p99_ms']:7}  "
            f"max={s['server_latency_ms']['max_ms']:7}  "
            f"pass={s['sla']['p95_server_ok']}"
        )


if __name__ == "__main__":
    main()
