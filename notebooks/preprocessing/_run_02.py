import sys
from pathlib import Path
from collections import Counter
ROOT = Path(".").resolve(); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_utils import load_query_clicks, ARTIFACTS_DIR, FIGURES_DIR, MVIDEO_RED, DARK_SLATE, MUTED, apply_plot_style, ensure_dirs, save_stats
from src.ner.labeling import WeakLabeler, tokenize, lemmatize_text
from src.preprocessing import QueryPreprocessor, load_phrase_list

logp = ROOT/"notebooks/preprocessing/_run_02_log.txt"
def log(m):
    print(m, flush=True)
    with logp.open("a", encoding="utf-8") as f: f.write(m+"\n")
if logp.exists(): logp.unlink()
ensure_dirs(); apply_plot_style()
FIG = FIGURES_DIR/"preprocessing"; FIG.mkdir(exist_ok=True)
ART = ARTIFACTS_DIR
phrases = load_phrase_list(ART/"model_phrases.txt")
log(f"phrases={len(phrases)}")

def flags(p):
    toks=p.split()
    long_sku=any(sum(ch.isdigit() for ch in t)>=5 for t in toks)
    has_alpha=any(any(c.isalpha() for c in t) for t in toks)
    pure_num=all(t.replace(".","").isdigit() for t in toks)
    very_long=len(toks)>=5
    return has_alpha and not long_sku and not pure_num and not very_long
ok=sum(1 for p in phrases if flags(p))
log(f"ok_candidate={ok}/{len(phrases)}={ok/len(phrases):.3f}")

lab0=WeakLabeler.from_files(ART/"brands.txt", ART/"categories.txt")
lab1=WeakLabeler.from_files(ART/"brands.txt", ART/"categories.txt", models_path=ART/"model_phrases.txt")
pp=QueryPreprocessor.from_artifacts(ART)
log(f"asus no {lab0.label_query('asus tuf gaming a15')}")
log(f"asus yes {lab1.label_query('asus tuf gaming a15')}")

clicks=load_query_clicks(n=80000, seed=42, random=True)
uq=clicks["query_text"].astype(str).str.strip()
sample=uq[uq.str.len()>=2].drop_duplicates().sample(n=min(5000,uq.nunique()), random_state=42).tolist()

def eval_lab(lab):
    n_model=n_empty=n_mis=n_otail=0
    for q in sample:
        qn=pp(q).text_norm
        if len(tokenize(qn))!=len(lemmatize_text(qn)): n_mis+=1
        tags=lab.label_query(qn)
        bs=[t[2:] for _,t in tags if t.startswith("B-")]
        if not bs: n_empty+=1
        if "MODEL" in bs: n_model+=1
        bidx=[i for i,(_,t) in enumerate(tags) if t.endswith("BRAND")]
        if bidx:
            last=max(bidx); tail=tags[last+1:]
            if tail and all(t=="O" for _,t in tail):
                tt=[tok for tok,_ in tail]
                if np.mean([x.isascii() and x.isalnum() for x in tt])>=0.6: n_otail+=1
    n=len(sample)
    return dict(empty=n_empty/n, model=n_model/n, otail=n_otail/n, mis=n_mis/n)

s0,s1=eval_lab(lab0),eval_lab(lab1)
log(f"off={s0}")
log(f"on={s1}")
fig,ax=plt.subplots(figsize=(8.5,4))
x=np.arange(2); w=0.25
for i,(m,c) in enumerate([("empty",MUTED),("model",DARK_SLATE),("otail",MVIDEO_RED)]):
    ax.bar(x+(i-1)*w, [s0[m], s1[m]], w, label=m, color=c)
ax.set_xticks(x); ax.set_xticklabels(["no models_path","with models_path"]); ax.set_ylim(0,1)
ax.legend(fontsize=8); ax.set_title("Silver BIO coverage")
fig.tight_layout(); fig.savefig(FIG/"11_silver_coverage_with_without_model.png", dpi=160, bbox_inches="tight")
save_stats({"n_phrases":len(phrases),"ok_share":ok/len(phrases),"silver_off":s0,"silver_on":s1,"verdict":"start_gold_now_parallel_to_dict_cleanup"}, "preprocessed_data_overview.json")
log("DONE")
