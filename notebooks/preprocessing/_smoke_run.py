import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
import numpy as np
from src.data_utils import load_query_clicks, ARTIFACTS_DIR, ensure_dirs, save_stats, apply_plot_style, FIGURES_DIR, MVIDEO_RED
from src.preprocessing import QueryPreprocessor, build_model_lexicon_from_titles, save_phrase_list
from src.preprocessing.pipeline import MODEL_SEEDS, PROTECTED_BRAND_SEEDS
from src.ner.labeling import WeakLabeler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logp = ROOT / "notebooks" / "preprocessing" / "_smoke_log.txt"
def log(m):
    print(m, flush=True)
    with logp.open("a", encoding="utf-8") as f: f.write(m+"\n")
if logp.exists(): logp.unlink()
ensure_dirs(); apply_plot_style()
FIG = FIGURES_DIR / "preprocessing"; FIG.mkdir(parents=True, exist_ok=True)
ART = ARTIFACTS_DIR
clicks = load_query_clicks(n=80_000, seed=42, random=True)
log(f"clicks={len(clicks)}")
brands = clicks["sku_brand_name"].astype(str).str.strip().replace("", np.nan).dropna().value_counts().head(250).index.tolist()
mined = build_model_lexicon_from_titles(clicks["sku_name"].astype(str).dropna().tolist(), brands, min_count=5, max_phrase_tokens=4)
log(f"mined={len(mined)}")

def keep(p):
    toks = p.split()
    if not toks: return False
    if all(t.replace(".","").isdigit() for t in toks): return False
    if p in MODEL_SEEDS: return True
    if any(c.isdigit() for c in p) and any(c.isalpha() for c in p): return True
    if toks[0] in {"g","v","ps","galaxy","redmi","poco","iphone","macbook"}: return len(toks)>=2
    return len(toks)>=2 and mined.get(p,0)>=12

filtered = {p for p in (set(MODEL_SEEDS)|set(mined)) if keep(p)}
save_phrase_list(filtered, ART/"model_phrases.txt")
save_phrase_list(PROTECTED_BRAND_SEEDS, ART/"protected_brands.txt")
log(f"saved_models={len(filtered)} has_gproxse={'g pro x se' in filtered}")

pp = QueryPreprocessor.from_artifacts(ART)
labeler = WeakLabeler.from_files(ART/"brands.txt", ART/"categories.txt")
for q in ["наушники logitech g-pro x se", "Красный Октябрь", "красный телефон", "Ноутбук 16гб", "dyson v15"]:
    r = pp(q)
    merged = pp.merge_bio_hints(labeler.label_query(r.text_norm), r)
    log(f"Q={q!r} norm={r.text_norm!r} model={r.model_spans} prot={r.protected_spans} bio={merged}")

uq = clicks["query_text"].astype(str).str.strip()
sample_q = uq[uq.str.len()>=2].drop_duplicates().sample(n=min(5000, uq.nunique()), random_state=42).tolist()
n_model = sum(1 for q in sample_q if pp(q).model_spans)
log(f"share_model={n_model/len(sample_q):.4f}")
top = pd.DataFrame(sorted(mined.items(), key=lambda x: -x[1])[:15], columns=["phrase","count"])
fig,ax=plt.subplots(figsize=(9,4)); ax.barh(top["phrase"][::-1], top["count"][::-1], color=MVIDEO_RED)
fig.tight_layout(); fig.savefig(FIG/"01_mined_model_phrases.png", dpi=160, bbox_inches="tight"); plt.close()
save_stats({"n_model_phrases": len(filtered), "share_model_span_sample": n_model/len(sample_q), "n_mined_raw": len(mined)}, "preprocessing_stats.json")
log("DONE")
