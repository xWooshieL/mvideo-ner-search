# 02 CRF NER — train report

Model: `models\ner_crf.pkl`  
Silver: `D:\Projects-26-06-2026\mvideo-ner-search\artifacts\silver\ner_bio\silver_bio_slice.parquet`  
Train/val: **3570** / **893** (seed=42)

## Features (not TF-IDF)

Per-token: `word.lower`, prefix/suffix, shape, digit/latin/cyrillic, ±1/±2 neighbors, BOS/EOS.
Typos: weak (no edit-distance).

## Silver-val (weak↔weak, optimistic)

- token accuracy: **0.875**
- entity micro-F1: **0.856** (P=0.864 R=0.848)
- macro-F1: **0.824**

| label | P | R | F1 | support |
|---|---:|---:|---:|---:|
| BRAND | 0.958 | 0.947 | 0.953 | 511 |
| CATEGORY | 0.825 | 0.823 | 0.824 | 791 |
| MODEL | 0.748 | 0.702 | 0.724 | 114 |
| ATTR | 0.834 | 0.761 | 0.796 | 159 |

## Gold (`bio_liza.jsonl`) — primary MVP metric

- used **200/200** (tokenize_align=181, skipped=0)
- token accuracy: **0.578**
- entity micro-F1: **0.602** (P=0.703 R=0.526)
- macro-F1: **0.528**

| label | P | R | F1 | support |
|---|---:|---:|---:|---:|
| BRAND | 0.793 | 0.830 | 0.811 | 106 |
| CATEGORY | 0.658 | 0.641 | 0.649 | 153 |
| MODEL | 0.500 | 0.176 | 0.260 | 74 |
| ATTR | 0.774 | 0.264 | 0.393 | 91 |

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