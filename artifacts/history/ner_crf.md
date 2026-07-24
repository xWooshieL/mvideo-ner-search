# ner_crf — история экспериментов

Хронология снизу вверх не нужна: новые записи **добавляются в конец**.
Gold micro-F1 важнее silver-val.

---

## 2026-07-24 00:42 — `baseline-pre-spellfix`

**Note:** Snapshot before rebuilding silver with SpellFixer (colleague commit 0f6e63d). Gold microF1~0.58, silver-val~0.85.

| метрика | значение |
|---|---:|
| silver-val micro-F1 | 0.8475 |
| silver-val tokAcc | 0.8636 |
| **gold micro-F1** | 0.5824 |
| gold P / R | 0.6677 / 0.5165 |
| rules gold micro-F1 | 0.5707 |
| n_train / n_val | 3475 / 869 |
| spellfix touched | None / 5000 |

Gold entity F1 per tag:

| tag | F1 |
|---|---:|
| BRAND | 0.8037 |
| CATEGORY | 0.6429 |
| MODEL | 0.2593 |
| ATTR | 0.3279 |

## 2026-07-24 00:46 — `spellfix-v1`

**Note:** Colleague SpellFixer (0f6e63d): normalize typos/units before WeakLabeler in silver+_run_01 and in extractor.spellfix touched 767/5000 silver queries. Retrain CRF.

| метрика | значение |
|---|---:|
| silver-val micro-F1 | 0.8561 |
| silver-val tokAcc | 0.8754 |
| **gold micro-F1** | 0.5939 |
| gold P / R | 0.6820 / 0.5259 |
| rules gold micro-F1 | 0.5660 |
| n_train / n_val | 3570 / 893 |
| spellfix touched | 767 / 5000 |

Gold entity F1 per tag:

| tag | F1 |
|---|---:|
| BRAND | 0.8000 |
| CATEGORY | 0.6512 |
| MODEL | 0.2308 |
| ATTR | 0.3968 |

## 2026-07-24 16:37 — `spellfix-v2`

**Note:** SpellFix v2 (homoglyphs + spell_aliases.txt). category_clf disabled. silver spellfix 896/5000. Rebuild CRF for demo.

| метрика | значение |
|---|---:|
| silver-val micro-F1 | 0.8564 |
| silver-val tokAcc | 0.8759 |
| **gold micro-F1** | 0.5870 |
| gold P / R | 0.6717 / 0.5212 |
| rules gold micro-F1 | 0.5660 |
| n_train / n_val | 3578 / 895 |
| spellfix touched | 896 / 5000 |

Gold entity F1 per tag:

| tag | F1 |
|---|---:|
| BRAND | 0.8165 |
| CATEGORY | 0.6295 |
| MODEL | 0.2642 |
| ATTR | 0.3548 |

