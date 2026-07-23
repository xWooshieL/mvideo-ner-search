# История экспериментов (вне GitHub-диффов)

Сюда пишем **снимки метрик** после каждого осмысленного прогона: что меняли, что получили,
сравнимо ли с прошлым. Git хранит код; эта папка — человекочитаемый журнал.

## Файлы

| Файл | Что |
|---|---|
| [`experiments.jsonl`](./experiments.jsonl) | машинный лог (одна строка = один эксперимент) |
| [`ner_crf.md`](./ner_crf.md) | CRF NER: silver-val + gold |
| [`brand_clf.md`](./brand_clf.md) | классификатор бренда (silver-val) |
| [`attr_type_clf.md`](./attr_type_clf.md) | типизатор ATTR (silver-val + gold agree) |
| [`broken_queries.md`](./broken_queries.md) | качественные кейсы с опечатками |

## Как логировать новый прогон

```bash
# после train / eval_on_gold
python scripts/log_experiment.py \
  --module ner_crf \
  --tag spellfix-v1 \
  --note "SpellFixer before WeakLabeler; rebuild silver + CRF" \
  --from-metrics
```

Или вручную дописать секцию в соответствующий `.md` и строку в `experiments.jsonl`
(скрипт делает оба).

## Правила

1. **Не переписывать** старые записи — только append.
2. В `note` — зачем эксперимент (1–2 предложения), не «rerun».
3. Gold-метрики важнее silver-val (silver завышен).
4. Если меняли только один модуль — логируем только его (+ broken_queries при необходимости).
