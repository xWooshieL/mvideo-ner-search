import sys
from pathlib import Path
from collections import Counter
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_utils import load_query_clicks, ARTIFACTS_DIR, FIGURES_DIR, MVIDEO_RED, MUTED, apply_plot_style, ensure_dirs
from src.ner.labeling import WeakLabeler
from src.preprocessing import QueryPreprocessor
ensure_dirs(); apply_plot_style()
FIG = FIGURES_DIR / "complex_eda" / "model_tag"
clicks = load_query_clicks(n=100000, seed=42, random=True)
lab = WeakLabeler.from_files(ARTIFACTS_DIR/"brands.txt", ARTIFACTS_DIR/"categories.txt")

def brand_latin_o_tail(tags):
    brand_idx = [i for i, (_, t) in enumerate(tags) if t.endswith("BRAND")]
    if not brand_idx: return None
    last = max(brand_idx); tail = tags[last+1:]
    if not tail or not all(t == "O" for _, t in tail): return None
    toks = [tok for tok, _ in tail]
    if np.mean([tok.isascii() and tok.isalnum() for tok in toks]) < 0.6: return None
    return toks

uq = clicks["query_text"].astype(str).str.strip()
sample = uq[uq.str.len()>=2].drop_duplicates().sample(n=min(8000, uq.nunique()), random_state=42)
pp = QueryPreprocessor(); tail_phrases = Counter()
for q in sample:
    t = brand_latin_o_tail(lab.label_query(pp(q).text_norm))
    if t: tail_phrases[" ".join(t[:5])] += 1
top_all = pd.DataFrame(tail_phrases.most_common(25), columns=["o_tail_phrase","count"])
top_modelish = top_all[top_all["o_tail_phrase"].str.contains(r"[A-Za-zА-Яа-я]", regex=True)].head(15)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
axes[0].barh(top_all.head(12)["o_tail_phrase"][::-1], top_all.head(12)["count"][::-1], color=MUTED)
axes[0].set_title("All O-tails (often bare numbers)")
axes[1].barh(top_modelish["o_tail_phrase"][::-1], top_modelish["count"][::-1], color=MVIDEO_RED)
axes[1].set_title("Letter O-tails — MODEL candidates")
fig.suptitle("No MODEL tag: tail after BRAND stays O", y=1.02)
fig.tight_layout()
fig.savefig(FIG/"02_top_lost_tails.png", dpi=170, bbox_inches="tight")
print("DONE", len(top_modelish), flush=True)
