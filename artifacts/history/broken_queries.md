# Broken queries — качественный журнал

Сюда пишем руками / скриптом примеры опечаток до/после фикса. Не метрика, а sanity.

---

## 2026-07-24 00:46 — after spellfix-v1 (CRF rebuild)

| query | spellfix | CRF BIO (on fixed) | full brand / cat / model |
|---|---|---|---|
| телефон 16 | — | телефон/B-CATEGORY 16/O | None / телефоны / None |
| телфон 16 гь | телфон→телефон, гь→гб | телефон/B-CATEGORY 16/B-ATTR гб/I-ATTR | None / телефоны / None |
| моильник 16 гб | — | моильник/B-CATEGORY 16/B-ATTR гб/I-ATTR | None / моильник / None |
| laptop ксяоми | — | laptop/B-BRAND ксяоми/O | laptop / ноутбук / None |
| планше тxiaomi | планше→планшет, тxiaomi→xiaomi | планшет/B-CATEGORY xiaomi/B-BRAND | Xiaomi / планшет / None |
| айфон 16 | — | айфон/B-BRAND 16/O | Apple / айфон16 / None |
| холодильник Indesit 16 | — | холодильник/B-CATEGORY Indesit/B-BRAND 16/O | Indesit / холодильник / None |
| шкаф | — | шкаф/B-CATEGORY | None / шкаф / None |
| сони плейстейшен 5 | — | сони/O плейстейшен/O 5/O | None / фен / None |
| sony playstation 5 | — | sony/B-BRAND playstation/B-BRAND 5/O | Sony / фен / None |
| sony ps5 | — | sony/B-BRAND ps5/B-MODEL | Sony / смартфон / ps5 |
| сони ps5 | — | сони/O ps5/B-MODEL | None / монитор / ps5 |
| ноутбок asus 16гь | ноутбок→ноутбук, 16гь→16 гб | ноутбук/B-CATEGORY asus/B-BRAND 16/B-ATTR гб/I-ATTR | ASUS / ноутбук / None |
| пылесос dyson v15 | — | пылесос/B-CATEGORY dyson/B-BRAND v15/B-MODEL | Dyson / пылесос / v15 |

Extractor spellfix=True, category_clf loaded=True

## 2026-07-24 — after spellfix-v2 + category_clf off

| query | spellfix | brand / cat / model | attrs |
|---|---|---|---|
| сони плейстейшен 5 | сони→sony, плейстейшен→playstation | Sony / None / None | {} |
| sony playstation 5 | — | Sony / None / None | {} |
| телфон 16 гь | телфон→телефон, гь→гб | None / телефоны / None | {'memory_storage': '16 гб'} |
| планше тxiaomi | планше→планшет, тxiaomi→xiaomi | Xiaomi / планшет / None | {} |
| ноутбок asus 16гь | ноутбок→ноутбук, 16гь→16 гб | ASUS / ноутбук / None | {'memory_storage': '16 гб'} |
| моильник 16 гб | — | None / моильник / None | {'memory_storage': '16 гб'} |
| laptop ксяоми | ксяоми→xiaomi | Xiaomi / None / None | {} |
| айфон 16 | айфон→iphone | Apple / None / None | {} |

category_clf loaded=False

