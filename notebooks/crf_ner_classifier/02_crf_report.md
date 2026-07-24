# 02 CRF NER — train report

Model: `models\ner_crf.pkl`  
Silver: `D:\Projects-26-06-2026\mvideo-ner-search\artifacts\silver\ner_bio\silver_bio_slice.parquet`  
Train/val: **3578** / **895** (seed=42)

## Features (not TF-IDF)

Per-token: `word.lower`, prefix/suffix, shape, digit/latin/cyrillic, ±1/±2 neighbors, BOS/EOS.
Typos: weak (no edit-distance).

## Silver-val (weak↔weak, optimistic)

- token accuracy: **0.876**
- entity micro-F1: **0.856** (P=0.861 R=0.852)
- macro-F1: **0.817**

| label | P | R | F1 | support |
|---|---:|---:|---:|---:|
| BRAND | 0.957 | 0.959 | 0.958 | 510 |
| CATEGORY | 0.826 | 0.828 | 0.827 | 774 |
| MODEL | 0.777 | 0.757 | 0.767 | 115 |
| ATTR | 0.766 | 0.676 | 0.718 | 145 |

## Gold (`bio_liza.jsonl`) — primary MVP metric

- used **200/200** (tokenize_align=181, skipped=0)
- token accuracy: **0.579**
- entity micro-F1: **0.595** (P=0.689 R=0.524)
- macro-F1: **0.526**

| label | P | R | F1 | support |
|---|---:|---:|---:|---:|
| BRAND | 0.795 | 0.840 | 0.817 | 106 |
| CATEGORY | 0.627 | 0.627 | 0.627 | 153 |
| MODEL | 0.577 | 0.203 | 0.300 | 74 |
| ATTR | 0.710 | 0.242 | 0.361 | 91 |

![f1](../../figures/ner/02_crf_entity_f1.png)

## Demos

| query | BIO |
|---|---|
| `asus tuf gaming a15 16 гб` | `asus/B-BRAND tuf/B-MODEL gaming/I-MODEL a15/I-MODEL 16/B-ATTR гб/I-ATTR` |
| `ноутбук asus 16 гб` | `ноутбук/B-CATEGORY asus/B-BRAND 16/B-ATTR гб/I-ATTR` |
| `iphone 15 pro max` | `iphone/B-BRAND 15/B-MODEL pro/I-MODEL max/B-BRAND` |
| `беспроводные наушники sony` | `беспроводные/B-CATEGORY наушники/I-CATEGORY sony/B-BRAND` |

## Notes

1. Silver includes **MODEL** (`models_path`); old 06/08 did not.
2. Trust **gold** more than silver-val.
3. Expand `silver_bio_slice` (more queries) before claiming prod-ready F1.

Artifacts: `models/ner_crf.pkl`, `artifacts/silver/ner_bio/crf_train_metrics.json`.