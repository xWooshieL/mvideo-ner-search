# -*- coding: utf-8 -*-
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_dir = ROOT / "notebooks"
nb_dir.mkdir(exist_ok=True)


def nb(cells):
    out = []
    for kind, src in cells:
        if isinstance(src, str):
            lines = src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
        else:
            lines = src
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": out,
    }


setup = """import sys
from pathlib import Path
ROOT = Path('..').resolve()
sys.path.insert(0, str(ROOT))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 4)
"""

nb06 = nb(
    [
        (
            "md",
            """# 06. Weak Supervision NER

Построение BIO-разметки поисковых запросов M.Video из словарей брендов/категорий и регулярных атрибутов (цвет, память, размер).

**Цель:** получить слаборазмеченный датасет для обучения CRF NER без ручной разметки.
""",
        ),
        ("code", setup),
        (
            "code",
            """from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd

from src.ner.labeling import WeakLabeler, bio_to_entities

DATA = ROOT / 'файлы' / 'query_clicks.parquet'
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
""",
        ),
        ("md", "## Словари брендов и категорий"),
        (
            "code",
            """brands_p, cats_p = ART / 'brands.txt', ART / 'categories.txt'
if not brands_p.exists() or not cats_p.exists():
    import subprocess
    subprocess.check_call([
        sys.executable, str(ROOT / 'scripts' / 'build_dictionaries.py'),
        '--max-rows', '300000',
    ])
print('brands:', len(brands_p.read_text(encoding='utf-8').splitlines()))
print('categories:', len(cats_p.read_text(encoding='utf-8').splitlines()))
print('top brands:', brands_p.read_text(encoding='utf-8').splitlines()[:10])
""",
        ),
        ("md", "## Выборка запросов"),
        (
            "code",
            """pf = pq.ParquetFile(DATA)
parts = []
for i in range(min(3, pf.num_row_groups)):
    t = pf.read_row_group(i, columns=['toValidUTF8(query_text)', 'toValidUTF8(sku_brand_name)'])
    df = t.to_pandas()
    df.columns = ['query', 'brand']
    parts.append(df)
df = pd.concat(parts).dropna(subset=['query'])
df['query'] = df['query'].astype(str).str.strip()
df = df[df['query'].str.len() >= 2].drop_duplicates('query').head(20000)
print(len(df), 'уникальных запросов')
df.head()
""",
        ),
        ("md", "## BIO-разметка"),
        (
            "code",
            """labeler = WeakLabeler.from_files(ART / 'brands.txt', ART / 'categories.txt')
examples = []
for q in df['query'].head(5000):
    sent = labeler.label_query(q)
    ents = bio_to_entities(sent, query=q)
    if ents:
        examples.append((q, sent, ents))

print(f'Запросов с ≥1 сущностью (из 5000): {len(examples)}')
for q, sent, ents in examples[:8]:
    print('Q:', q)
    print('BIO:', ' '.join(f'{t}/{tag}' for t, tag in sent))
    print('ENT:', ents)
    print('-' * 60)
""",
        ),
        ("md", "## Распределение сущностей"),
        (
            "code",
            """labels = Counter()
for q in df['query']:
    for _, tag in labeler.label_query(q):
        if tag.startswith('B-'):
            labels[tag[2:]] += 1
print(labels)
fig, ax = plt.subplots()
ax.bar(list(labels.keys()), list(labels.values()), color='#E31E24')
ax.set_title('Распределение сущностей (weak labels)')
fig.savefig(FIG / '20_entity_distribution.png', dpi=140, bbox_inches='tight')
plt.show()
""",
        ),
    ]
)

nb07 = nb(
    [
        (
            "md",
            """# 07. Baseline: TF-IDF + LogisticRegression

Мультиклассовая классификация бренда по тексту запроса.

Метрики: Accuracy, F1-macro/micro, confusion matrix.
""",
        ),
        ("code", setup),
        (
            "code",
            """import joblib
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

DATA = ROOT / 'файлы' / 'query_clicks.parquet'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
""",
        ),
        (
            "code",
            """pf = pq.ParquetFile(DATA)
parts = []
seen = 0
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i, columns=['toValidUTF8(query_text)', 'toValidUTF8(sku_brand_name)'])
    df = t.to_pandas()
    df.columns = ['query', 'brand']
    parts.append(df)
    seen += len(df)
    if seen >= 150000:
        break
df = pd.concat(parts, ignore_index=True)
df['query'] = df['query'].fillna('').astype(str).str.strip()
df['brand'] = df['brand'].fillna('').astype(str).str.strip()
df = df[(df['query'].str.len() >= 2) & (df['brand'].str.len() >= 2)].drop_duplicates('query')
top = {b for b, _ in Counter(df['brand']).most_common(80)}
df = df[df['brand'].isin(top)]
print(df.shape, 'classes', df['brand'].nunique())
""",
        ),
        (
            "code",
            """Xtr, Xte, ytr, yte = train_test_split(
    df['query'], df['brand'], test_size=0.2, random_state=42, stratify=df['brand']
)
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_features=50000)),
    ('clf', LogisticRegression(max_iter=200, solver='saga', n_jobs=-1)),
])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
print('accuracy', (pred == yte).mean())
print('f1_macro', f1_score(yte, pred, average='macro'))
print('f1_micro', f1_score(yte, pred, average='micro'))
print(classification_report(yte, pred, zero_division=0)[:2000])
joblib.dump(pipe, MODELS / 'brand_clf.joblib')
""",
        ),
        (
            "code",
            """top15 = [b for b, _ in Counter(yte).most_common(15)]
mask = yte.isin(top15)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    yte[mask], pred[mask], labels=top15, ax=ax, xticks_rotation=45, colorbar=False
)
ax.set_title('Confusion matrix — бренды (top-15)')
fig.tight_layout()
fig.savefig(FIG / '13_confusion_matrix.png', dpi=140)
plt.show()
""",
        ),
    ]
)

nb08 = nb(
    [
        (
            "md",
            """# 08. Обучение NER (sklearn-crfsuite)

Лёгкий **CRF** на CPU с лингвистическими фичами токенов.

Метрики: token Accuracy, entity Precision/Recall/F1; learning curve.
""",
        ),
        ("code", setup),
        (
            "code",
            """import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from src.ner.labeling import WeakLabeler
from src.ner.features import sent2labels
from src.ner.model_crf import CRFNerModel
from src.ner.metrics import summarize_metrics

DATA = ROOT / 'файлы' / 'query_clicks.parquet'
ART = ROOT / 'artifacts'
MODELS = ROOT / 'models'
FIG = ROOT / 'figures'
for p in (ART, MODELS, FIG):
    p.mkdir(exist_ok=True)
""",
        ),
        (
            "code",
            """pf = pq.ParquetFile(DATA)
parts = []
seen = 0
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i, columns=['toValidUTF8(query_text)'])
    df = t.to_pandas()
    df.columns = ['query']
    parts.append(df)
    seen += len(df)
    if seen >= 100000:
        break
queries = pd.concat(parts)['query'].dropna().astype(str).str.strip()
queries = queries[queries.str.len() >= 2].drop_duplicates().tolist()
print(len(queries))
labeler = WeakLabeler.from_files(ART / 'brands.txt', ART / 'categories.txt')
sents = labeler.label_dataset(queries, min_entities=1)
print('labeled', len(sents))
if len(sents) > 30000:
    rng = np.random.default_rng(42)
    sents = [sents[i] for i in rng.choice(len(sents), 30000, replace=False)]
""",
        ),
        (
            "code",
            """train, test = train_test_split(sents, test_size=0.2, random_state=42)
sizes, f1s, accs = [], [], []
for frac in [0.25, 0.5, 0.75, 1.0]:
    n = max(100, int(len(train) * frac))
    subset = train[:n]
    m = CRFNerModel(max_iterations=60)
    t0 = time.time()
    m.fit(subset)
    print(f'n={n} train_sec={time.time() - t0:.1f}')
    yt = [sent2labels(s) for s in test]
    yp = m.predict(test)
    r = summarize_metrics(yt, yp)
    sizes.append(n)
    f1s.append(r['micro']['f1'])
    accs.append(r['token_accuracy'])
final = CRFNerModel(max_iterations=80)
final.fit(train)
final.save(MODELS / 'ner_crf.pkl')
yt = [sent2labels(s) for s in test]
yp = final.predict(test)
report = summarize_metrics(yt, yp)
print(json.dumps(report, ensure_ascii=False, indent=2)[:1500])
""",
        ),
        (
            "code",
            """fig, ax = plt.subplots()
ax.plot(sizes, f1s, 'o-', color='#E31E24', label='entity micro-F1')
ax.plot(sizes, accs, 's--', color='#333', label='token Accuracy')
ax.set_title('Learning curve — CRF NER')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(FIG / '15_learning_curve.png', dpi=140, bbox_inches='tight')
plt.show()

per = report['per_label']
labs = list(per)
vals = [per[l]['f1'] for l in labs]
fig, ax = plt.subplots()
ax.bar(labs, vals, color='#E31E24')
ax.set_ylim(0, 1.05)
ax.set_title('NER F1 по типам сущностей')
fig.savefig(FIG / '14_ner_f1_by_entity.png', dpi=140, bbox_inches='tight')
plt.show()
""",
        ),
    ]
)

nb09 = nb(
    [
        (
            "md",
            """# 09. Embeddings и сходство запросов

TF-IDF (char_wb) + опционально Word2Vec / TruncatedSVD. Визуализация: матрица сходства и t-SNE.
""",
        ),
        (
            "code",
            setup
            + """
import subprocess
from IPython.display import Image, display
subprocess.check_call([
    sys.executable, str(ROOT / 'scripts' / 'build_embeddings_figures.py'), '--n', '2500'
])
display(Image(filename=str(ROOT / 'figures' / '19_similarity_queries.png')))
display(Image(filename=str(ROOT / 'figures' / '16_embedding_tsne.png')))
""",
        ),
        (
            "md",
            "Скрипт `scripts/build_embeddings_figures.py` сохраняет фигуры 16 и 19 в `figures/`.",
        ),
    ]
)

nb10 = nb(
    [
        (
            "md",
            """# 10. Бенчмарк сервиса

Latency p50/p95/p99, доля ответов <100 ms, примеры JSON.
""",
        ),
        ("code", setup),
        (
            "code",
            """import json
import subprocess
from IPython.display import Image, display

subprocess.check_call([
    sys.executable, str(ROOT / 'scripts' / 'benchmark_latency.py'), '--n', '800'
])
bench = json.loads((ROOT / 'artifacts' / 'benchmark.json').read_text(encoding='utf-8'))
print({k: bench[k] for k in bench if k != 'examples'})
display(Image(filename=str(ROOT / 'figures' / '17_latency_histogram.png')))
print('Примеры JSON:')
for ex in bench.get('examples', [])[:5]:
    print(json.dumps(ex, ensure_ascii=False, indent=2))
    print('---')
""",
        ),
        (
            "md",
            """## Запуск FastAPI

```bash
uvicorn src.service.app:app --host 0.0.0.0 --port 8000
# POST http://localhost:8000/extract  {"query":"iphone 15 256gb черный"}
```
""",
        ),
    ]
)

for name, obj in [
    ("06_weak_supervision_ner.ipynb", nb06),
    ("07_train_baseline_classifier.ipynb", nb07),
    ("08_train_ner_model.ipynb", nb08),
    ("09_embeddings_similarity.ipynb", nb09),
    ("10_service_benchmark.ipynb", nb10),
]:
    path = nb_dir / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)
print("done")
