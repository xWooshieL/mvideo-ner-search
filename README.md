<div align="center">

<img src="cpp/mvsearch/assets/logo_mark.png" width="120" alt="М.Видео">

# М.Видео · Интеллектуальный поиск

**Извлечение структурированных фактов из поисковых запросов за миллисекунды**

![Версия](https://img.shields.io/badge/версия-0.1.0-f20601?style=for-the-badge)
![Платформа](https://img.shields.io/badge/Windows-10%2F11-0078d4?style=for-the-badge&logo=windows)
![Qt](https://img.shields.io/badge/Qt-6.8-41cd52?style=for-the-badge&logo=qt)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599c?style=for-the-badge&logo=cplusplus)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python)

![NER](https://img.shields.io/badge/NLP-NER_%2B_CRF_%2B_Markov-1c1c1e)
![Классификатор](https://img.shields.io/badge/бренд--классификатор-macro--F1_0.95-18794e)
![SLA](https://img.shields.io/badge/извлечение_фактов-1--2_мс-success)
![Сборка](https://img.shields.io/badge/сборка-CMake_%2B_MSVC-orange)
![Установщик](https://img.shields.io/badge/установщик-Inno_Setup_6-8b5cf6)
![Офлайн](https://img.shields.io/badge/работает-полностью_офлайн-6e788c)

</div>

---

Пользователь пишет «ноутбук asus zenbook 16 гб серый» — система за миллисекунды понимает, *что* он ищет, и отдаёт структурированные факты, по которым можно искать и ранжировать каталог:

```json
{
  "query": "ноутбук asus zenbook 16 гб серый",
  "brand": "ASUS",
  "category": "ноутбук",
  "entities": [
    {"text": "ноутбук",  "label": "CATEGORY"},
    {"text": "asus",     "label": "BRAND"},
    {"text": "zenbook",  "label": "MODEL"},
    {"text": "16 гб",    "label": "ATTR", "type": "memory_storage"}
  ],
  "attributes": {"memory_storage": "16 гб", "color": "серый"},
  "latency_ms": 1.4
}
```

<div align="center">
<img src="figures/apps/cpp_mvp_search.png" width="800" alt="Умный поиск">
<br><em>Умный поиск: факты чипами и каталог, отранжированный только по фактам</em>
</div>

## 🖥 Два нативных приложения (C++17 / Qt6 QML)

| | |
|:---:|:---:|
| <img src="figures/apps/cpp_mvp_json.png" width="420"><br>**Умный поиск** — раскрывающийся JSON, таймер извлечения в шапке | <img src="figures/apps/cpp_mvp_stats.png" width="420"><br>**Статистика движка** — словари, каталог, метрики, каскад |
| <img src="figures/apps/cpp_label_stage1.png" width="420"><br>**Разметка · этап 1** — B/I/O с клавиатуры, метка над блоком | <img src="figures/apps/cpp_label_stage2.png" width="420"><br>**Разметка · этап 2** — тип на сущность целиком, с описаниями |
| <img src="figures/apps/cpp_label_stage3.png" width="420"><br>**Разметка · этап 3** — подтип атрибута, «другое» вручную | <img src="figures/apps/cpp_label_match.png" width="420"><br>**Соответствие 1/0** — гайд, авто-0 для пустых карточек |

### MVideo_SmartSearch_Setup — «Умный поиск»

Готовый установщик — в [Releases](../../releases). Анимированный вход, поиск с чипами фактов, раскрывающийся JSON, страница статистики и **RecSys по фактам**: карточки каталога ранжируются только по извлечённым фактам (бренд, категория, модель, атрибуты), не по сырому тексту. Извлечение — 1–2 мс. Тёмная тема в настройках.

### MVideo_Labeling_Setup — «Разметка» (один установщик на команду)

Три ярлыка в меню Пуск — **Никита, Некит, Лиза** — по 1500 непересекающихся запросов, плюс переключатель аккаунтов прямо в приложении. Трёхэтапный мастер:

1. **BIO** — клавиши `B`/`I`/`O` (работает и русская раскладка: `и`/`ш`/`щ`), метка над блоком, `Backspace` снимает и шагает назад;
2. **Тип сущности** — `1–5` или стрелками; тип ставится на сущность целиком (B I … I — одна группа);
3. **Подтип атрибута** — с описанием и переводом, «другое» — ручной ввод.

Режим **«Соответствие 1/0»** с гайдом (пустая карточка получает 0 автоматически), история с правкой, две кнопки «назад» (на начало запроса / на последнее состояние), крестик сворачивает мастер в историю. Всё сохраняется в JSONL рядом с приложением.

## 🏗 Архитектура: гибридный каскад

```
запрос ❯ ПРАВИЛА (словари · регулярки · модели) ❯ CRF ❯ КЛАССИФИКАТОР БРЕНДА ❯ МАРКОВСКИЙ ТИПИЗАТОР
                                                                                       │
                                             merge + нормализация → факты JSON, < 100 мс
```

| Слой | Что закрывает | Как учится |
|---|---|---|
| **Правила и словари** | явные бренды, «16 гб», линейки моделей | 7 словарей из каталога + майнинг |
| **CRF** | морфология, порядок слов, границы спанов | слабая (silver) BIO-разметка |
| **Классификатор бренда** | бренд не написан в тексте (73% кликов!) | silver из кликов: 67 классов, включая NO_BRAND и UNKNOWN |
| **Марковский типизатор** | тип атрибута: «16 гб» → память | частоты биграмм от регулярок-учителей |

### Классификатор бренда: 4 модели, честные отказы

Silver собран из кликов (28 664 запроса, 67 классов): бренды top-65 + `NO_BRAND` («холодильник» — бренда нет) + `UNKNOWN` (бренд вне списка — модель учится отказываться, а не угадывать). Обучены четыре линейные модели на TF-IDF; в прод выбрана **LogReg (слова 1–2 + символьные n-граммы 2–4)**: macro-F1 **0,950**, ложный бренд на запросах-категориях всего **6,9%**. Пороги отказа (`TAU_ACCEPT` 0.42, `TAU_MARGIN` 0.08) отсекают неуверенные ответы — лучше пусто, чем ложный Indesit.


## 📦 Структура репозитория

```
cpp/
├── mvsearch/             # «Умный поиск» — C++17 / Qt6 QML
│   ├── src/              #   движок: словари, регулярки, марковский типизатор, RecSys
│   └── qml/              #   интерфейс: сплэш, поиск, статистика, настройки
└── mvlabel/              # «Разметка» — C++17 / Qt6 QML, трёхэтапный мастер

src/
├── ner/                  # ядро NER (Python)
│   ├── labeling.py       #   слабая разметка: 7 словарей + 30 регулярок + лемматизация
│   ├── model_crf.py      #   CRF sequence labeling
│   └── markov_typer.py   #   марковский типизатор атрибутов
├── preprocessing/        # единый препроцессинг (расклейка, сепараторы, MODEL-спаны)
└── service/              # extractor + FastAPI-сервис

artifacts/                # словари (BRANDS, MODELS, COLORS, FEATURES, …) + brand_clf
models/                   # обученные модели (CRF, brand_clf, markov_typer)
notebooks/                # EDA + обучение классификатора бренда
installer/                # Inno Setup: два установщика
docs/                     # презентации Дней 1–3 (XeLaTeX, фирменный стиль)
figures/                  # 40+ графиков + скриншоты приложений
```

## 🛠 Стек

- **C++17 + Qt 6.8** (QML, Qt Quick Controls 2) — оба настольных приложения
- **CMake + MSVC 2022** — сборка
- **Inno Setup 6** — установщики (~17 МБ каждый)
- **Python 3.10+** — NER-пайплайн: pandas, scikit-learn, sklearn-crfsuite, pymorphy3
- **XeLaTeX** — презентации в фирменном стиле

## 🔨 Сборка

### Windows

Требования: Qt 6.5+ (MSVC), Visual Studio Build Tools, CMake, Inno Setup 6.

```powershell
# приложения
cd cpp/mvsearch && cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.8.2/msvc2022_64" && cmake --build build --config Release
cd cpp/mvlabel  && cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.8.2/msvc2022_64" && cmake --build build --config Release

# установщики
cd installer
iscc setup_mvsearch.iss
iscc setup_mvlabel.iss
```

### macOS

CMake-файлы обоих приложений кросс-платформенные (`MACOSX_BUNDLE`, `.icns`, пути к данным и к разметке через `QStandardPaths` на маке), но собрать `.app`/`.dmg` можно только **на самом Mac** — Qt для macOS компилируется под конкретную ОС, кросс-компиляция с Windows не поддерживается.

Требования на маке: Xcode Command Line Tools (`xcode-select --install`), CMake, Qt 6.5+ для macOS (Qt Online Installer или `brew install qt`).

```bash
git clone https://github.com/xWooshieL/mvideo-ner-search.git
cd mvideo-ner-search
chmod +x scripts/build-macos.sh
./scripts/build-macos.sh ~/Qt/6.8.2/macos     # путь к твоей установке Qt для macOS
```

Скрипт сам сгенерирует `.icns` из `.iconset` (см. `scripts/make_iconset.py`, если нужно пересобрать иконки из PNG), соберёт оба `.app` через CMake, прогонит `macdeployqt` и упакует в `.dmg` — результат в `dist-macos/`. Ярлыки Никита/Некит/Лиза в приложении разметки работают так же, как на Windows, плюс переключатель аккаунтов внутри приложения.

Python-пайплайн:

```bash
pip install -r requirements.txt

# извлечение фактов
python -c "
from src.service.extractor import QueryEntityExtractor
ex = QueryEntityExtractor.from_artifacts()
print(ex.extract('пылесос dyson v15'))
"

# слабая разметка на 7 словарях
python -c "
from src.ner.labeling import WeakLabeler
wl = WeakLabeler.from_dir('artifacts')
print(wl.label_query('телевизор samsung 55 дюймов чёрный'))
"
```

## 📊 Метрики

> Числа на silver-валидации — проверка «повторила ли модель правила», поэтому оптимистичны.
> Честную оценку даст ручная золотая разметка (собираем приложением, 3×1500 запросов).

| Компонент | Метрика | Значение |
|---|---|---|
| Классификатор бренда (LogReg слова+симв.) | macro-F1 | **0,950** |
| Классификатор бренда | ложный бренд на категориях | **6,9%** |
| Классификатор бренда | F1 NO_BRAND / UNKNOWN | 0,88 / 0,86 |
| CRF на слабой разметке | точность по токенам | 0,91 |
| CRF | F1 по сущностям | 0,875 |
| Марковский типизатор | точность против правил | 0,62 (36% — честное «не знаю») |
| C++ движок | извлечение фактов | 1–2 мс |

## 🗺 Roadmap

| Этап | Что планируется |
|---|---|
| Золотая разметка | 3×1500 запросов командой через приложение, честный тест и калибровка порогов |
| CRF v2 | переобучение на разметке с тегом MODEL, сверка с золотой |
| RNN-типизатор | лёгкая BiLSTM для спанов, где цепь говорит «не знаю» (36%) |
| Модель 1/0 | бустинг на ручных парах запрос↔карточка, чистка кликового шума |
| macOS-сборка | код и CMake готовы (`scripts/build-macos.sh`) — осталось прогнать на реальном Маке и подписать |

## 👥 Команда

Буткемп-команда М.Видео: **Никита · Некит · Лиза**
