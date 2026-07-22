# -*- coding: utf-8 -*-
"""М.Видео · Умный поиск — MVP-приложение (PyQt6).

Стиль и анимации перенесены из проекта ГК МОС:
- сплэш: эмблема «выпрыгивает» с пружинкой OutBack + разгорается свечение,
- переходы экранов через плавный fade,
- карточный интерфейс, красно-белая палитра М.Видео.

Фичи:
- поиск: запрос -> факты (бренд / категория / модель / атрибуты) с подсветкой,
- JSON и статистика по клику (раскрывающийся блок),
- RecSys: ранжирование карточек каталога ТОЛЬКО по извлечённым фактам
  (совпадение бренда/категории/атрибутов + косинус по словам факта),
- SLA-таймер на каждый запрос.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QParallelAnimationGroup, QSequentialAnimationGroup,
    QEasingCurve, QTimer, pyqtProperty, QPoint,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QBrush, QPixmap, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QStackedWidget, QFrame, QScrollArea,
    QGraphicsOpacityEffect, QGraphicsBlurEffect, QSizePolicy, QTextEdit,
)

MVRED = "#F20601"
MVDARK = "#1C1C1E"
MVGRAY = "#6E6E73"
CARDBG = "#F5F5F6"
PINK = "#FDECEC"
OK = "#1F8A50"
ORANGE = "#C75000"
GREEN_SOFT = "#E0F2E9"

TAG_COLORS = {"BRAND": MVDARK, "CATEGORY": MVRED, "MODEL": OK, "ATTR": ORANGE}
TAG_RU = {"BRAND": "бренд", "CATEGORY": "категория", "MODEL": "модель", "ATTR": "атрибут"}


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def find_root() -> Path:
    """Ищем корень проекта (models/, artifacts/) от exe или из репо."""
    for cand in (app_dir(), app_dir().parent, app_dir().parent.parent,
                 app_dir().parent.parent.parent):
        if (cand / "artifacts" / "brands.txt").exists():
            return cand
    return app_dir()


# ================================================================= бэкенд
class Backend:
    """Обёртка над экстрактором + каталог + RecSys только по фактам."""

    def __init__(self):
        self.extractor = None
        self.catalog: list[dict] = []
        self.error = None
        try:
            root = find_root()
            sys.path.insert(0, str(root))
            from src.service.extractor import QueryEntityExtractor
            self.extractor = QueryEntityExtractor.from_artifacts(
                artifacts_dir=root / "artifacts", models_dir=root / "models")
            self._load_catalog(root)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)

    def _load_catalog(self, root: Path):
        cat_file = app_dir() / "catalog.json"
        if not cat_file.exists():
            cat_file = root / "apps" / "mvp" / "catalog.json"
        if cat_file.exists():
            self.catalog = json.loads(cat_file.read_text(encoding="utf-8"))

    def extract(self, query: str) -> dict:
        if self.extractor is None:
            return {"query": query, "entities": [], "brand": None, "category": None,
                    "attributes": {}, "latency_ms": 0.0, "error": self.error}
        return self.extractor.extract_debug(query)

    # ---------- RecSys ТОЛЬКО ПО ФАКТАМ ----------
    @staticmethod
    def _tokens(s: str) -> list[str]:
        return re.findall(r"[a-zа-я0-9]+", (s or "").lower())

    @staticmethod
    def _cossim(a: Counter, b: Counter) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return num / den if den else 0.0

    def rank_catalog(self, facts: dict, top_n: int = 12) -> list[dict]:
        """Ранжируем карточки, опираясь ТОЛЬКО на извлечённые факты.

        Смесь сигналов:
          +3.0  бренд карточки == бренд-факт
          +2.0  категория-факт входит в название
          +1.5  модель-факт входит в название
          +1.0  каждый атрибут-факт, найденный в названии
          +cos  косинус слов ФАКТОВ (не запроса!) с названием карточки
        """
        brand = (facts.get("brand") or "").lower()
        category = (facts.get("category") or "").lower()
        attrs = facts.get("attributes") or {}
        model = ""
        fact_words: list[str] = []
        for e in facts.get("entities", []):
            if e.get("label") == "MODEL":
                model = (e.get("text") or "").lower()
            fact_words += self._tokens(e.get("text", ""))
        if brand:
            fact_words += self._tokens(brand)
        if category:
            fact_words += self._tokens(category)
        fvec = Counter(fact_words)

        scored = []
        for card in self.catalog:
            name_l = card["name"].lower()
            ntoks = Counter(self._tokens(name_l))
            s = 0.0
            why = []
            if brand and (card.get("brand", "").lower() == brand or brand in name_l):
                s += 3.0
                why.append("бренд")
            if category and any(t in ntoks for t in self._tokens(category)):
                s += 2.0
                why.append("категория")
            if model and model in name_l:
                s += 1.5
                why.append("модель")
            for a_type, a_vals in attrs.items():
                vals = a_vals if isinstance(a_vals, list) else [a_vals]
                for v in vals:
                    v_toks = self._tokens(str(v))
                    if v_toks and all(t in ntoks for t in v_toks):
                        s += 1.0
                        why.append(f"атрибут {v}")
                        break
            cos = self._cossim(fvec, ntoks)
            s += cos
            if s > 0.05:
                scored.append({**card, "score": round(s, 3), "why": why,
                               "cos": round(cos, 3)})
        scored.sort(key=lambda x: -x["score"])
        return scored[:top_n]


# ================================================================= сплэш
class SplashScreen(QWidget):
    """Анимация входа как в ГК МОС: пружинка OutBack + свечение + текст."""

    def __init__(self, on_done, parent=None):
        super().__init__(parent)
        self.on_done = on_done
        self.setStyleSheet("background:white;")

        lay = QVBoxLayout(self)
        lay.addStretch(3)

        # свечение (blur-круг за эмблемой)
        self.glow = QLabel(self)
        self.glow.setFixedSize(260, 260)
        self.glow.setStyleSheet(f"background:{MVRED}; border-radius:130px;")
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(90)
        self.glow.setGraphicsEffect(blur)
        self.glow_op = QGraphicsOpacityEffect(self.glow)
        self.glow.hide()  # рисуем вручную поверх layout

        self.mark = QLabel()
        logo = app_dir() / "assets" / "logo_mark.png"
        if logo.exists():
            pm = QPixmap(str(logo)).scaledToHeight(
                150, Qt.TransformationMode.SmoothTransformation)
            self.mark.setPixmap(pm)
        else:
            self.mark.setText("М")
            self.mark.setStyleSheet(f"color:{MVRED}; font-size:110px; font-weight:900;")
        self.mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.mark, alignment=Qt.AlignmentFlag.AlignCenter)

        lay.addSpacing(26)
        self.title = QLabel("М.Видео · Умный поиск")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(f"color:{MVDARK}; font-size:26px; font-weight:800;")
        lay.addWidget(self.title)

        self.subtitle = QLabel("Факты из запроса за < 100 мс · NER · CRF · Markov")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet(f"color:{MVGRAY}; font-size:13px;")
        lay.addWidget(self.subtitle)
        lay.addStretch(4)

        # эффекты для анимации
        self.mark_op = QGraphicsOpacityEffect(self.mark)
        self.mark.setGraphicsEffect(self.mark_op)
        self.mark_op.setOpacity(0.0)
        self.title_op = QGraphicsOpacityEffect(self.title)
        self.title.setGraphicsEffect(self.title_op)
        self.title_op.setOpacity(0.0)
        self.sub_op = QGraphicsOpacityEffect(self.subtitle)
        self.subtitle.setGraphicsEffect(self.sub_op)
        self.sub_op.setOpacity(0.0)

        QTimer.singleShot(300, self.start_animation)

    # масштаб эмблемы через свойство (пружинка OutBack, как в ГК МОС)
    def get_scale(self):
        return getattr(self, "_scale", 0.55)

    def set_scale(self, v):
        self._scale = v
        pm = getattr(self, "_orig_pm", None)
        if pm is None and self.mark.pixmap() is not None:
            self._orig_pm = self.mark.pixmap()
            pm = self._orig_pm
        if pm is not None:
            h = max(1, int(150 * v))
            self.mark.setPixmap(pm.scaledToHeight(
                h, Qt.TransformationMode.SmoothTransformation))

    scale = pyqtProperty(float, get_scale, set_scale)

    def start_animation(self):
        seq = QSequentialAnimationGroup(self)

        par = QParallelAnimationGroup()
        fade = QPropertyAnimation(self.mark_op, b"opacity")
        fade.setDuration(850)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        par.addAnimation(fade)

        spring = QPropertyAnimation(self, b"scale")
        spring.setDuration(1100)
        spring.setStartValue(0.55)
        spring.setEndValue(1.0)
        curve = QEasingCurve(QEasingCurve.Type.OutBack)
        curve.setOvershoot(1.25)
        spring.setEasingCurve(curve)
        par.addAnimation(spring)
        seq.addAnimation(par)

        t_fade = QPropertyAnimation(self.title_op, b"opacity")
        t_fade.setDuration(450)
        t_fade.setStartValue(0.0)
        t_fade.setEndValue(1.0)
        seq.addAnimation(t_fade)

        s_fade = QPropertyAnimation(self.sub_op, b"opacity")
        s_fade.setDuration(400)
        s_fade.setStartValue(0.0)
        s_fade.setEndValue(1.0)
        seq.addAnimation(s_fade)

        seq.addPause(550)
        seq.finished.connect(self.on_done)
        self._seq = seq
        seq.start()

    def mousePressEvent(self, _):
        self.on_done()


# ================================================================= чипы фактов
class FactChip(QLabel):
    def __init__(self, text: str, label: str):
        super().__init__(f" {TAG_RU.get(label, label)} · {text} ")
        col = TAG_COLORS.get(label, MVGRAY)
        self.setStyleSheet(
            f"background:{col}; color:white; border-radius:9px; padding:7px 12px;"
            f"font-size:13px; font-weight:700;")


# ================================================================= главный экран
class SearchScreen(QWidget):
    def __init__(self, backend: Backend):
        super().__init__()
        self.backend = backend
        self.last_result = None
        self.setStyleSheet("background:white;")
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(f"background:{MVRED};")
        header.setFixedHeight(66)
        h = QHBoxLayout(header)
        h.setContentsMargins(26, 0, 26, 0)
        logo = QLabel()
        lp = app_dir() / "assets" / "logo_mark_white.png"
        if lp.exists():
            logo.setPixmap(QPixmap(str(lp)).scaledToHeight(
                34, Qt.TransformationMode.SmoothTransformation))
        else:
            logo.setText("М")
            logo.setStyleSheet("color:white; font-size:30px; font-weight:900;")
        h.addWidget(logo)
        t = QLabel("Умный поиск · извлечение фактов")
        t.setStyleSheet("color:white; font-size:16px; font-weight:800; margin-left:10px;")
        h.addWidget(t)
        h.addStretch(1)
        self.sla = QLabel("")
        self.sla.setStyleSheet("color:white; font-size:13px; font-weight:700;")
        h.addWidget(self.sla)
        outer.addWidget(header)

        body = QVBoxLayout()
        body.setContentsMargins(30, 22, 30, 22)
        body.setSpacing(14)

        # поиск
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Например: ноутбук asus zenbook 16 гб серый …")
        self.input.setStyleSheet(
            f"QLineEdit{{border:2px solid #E3E3E5; border-radius:12px; padding:13px 18px;"
            f"font-size:15px; color:{MVDARK}; background:white;}}"
            f"QLineEdit:focus{{border:2px solid {MVRED};}}")
        self.input.returnPressed.connect(self.run_search)
        row.addWidget(self.input, 1)
        btn = QPushButton("Найти")
        btn.setStyleSheet(
            f"QPushButton{{background:{MVRED}; color:white; border:none; border-radius:12px;"
            f"padding:13px 34px; font-size:15px; font-weight:800;}}"
            f"QPushButton:hover{{background:#D00500;}}")
        btn.clicked.connect(self.run_search)
        row.addWidget(btn)
        body.addLayout(row)

        # область результатов со скроллом
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        self.results_holder = QWidget()
        self.results_lay = QVBoxLayout(self.results_holder)
        self.results_lay.setContentsMargins(0, 0, 0, 0)
        self.results_lay.setSpacing(12)
        self.results_lay.addStretch(1)
        self.scroll.setWidget(self.results_holder)
        body.addWidget(self.scroll, 1)

        outer.addLayout(body)

    # ---------- поиск
    def run_search(self):
        query = self.input.text().strip()
        if not query:
            return
        res = self.backend.extract(query)
        self.last_result = res
        self.sla.setText(f"{res.get('latency_ms', 0):.1f} мс")
        # очистка
        while self.results_lay.count() > 1:
            it = self.results_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # --- карточка фактов
        facts_card = QFrame()
        facts_card.setStyleSheet(f"background:{CARDBG}; border-radius:14px;")
        fl = QVBoxLayout(facts_card)
        fl.setContentsMargins(20, 16, 20, 16)
        cap = QLabel("Извлечённые факты")
        cap.setStyleSheet(f"color:{MVGRAY}; font-size:11px; font-weight:700;")
        fl.addWidget(cap)

        chips = QHBoxLayout()
        chips.setSpacing(8)
        added = False
        if res.get("brand"):
            chips.addWidget(FactChip(res["brand"], "BRAND"))
            added = True
        if res.get("category"):
            chips.addWidget(FactChip(res["category"], "CATEGORY"))
            added = True
        for e in res.get("entities", []):
            if e.get("label") == "MODEL":
                chips.addWidget(FactChip(e.get("text", ""), "MODEL"))
                added = True
        for a_type, vals in (res.get("attributes") or {}).items():
            vv = vals if isinstance(vals, list) else [vals]
            for v in vv:
                chips.addWidget(FactChip(f"{v}", "ATTR"))
                added = True
        if not added:
            nf = QLabel("Факты не найдены — запрос слишком общий")
            nf.setStyleSheet(f"color:{MVGRAY}; font-size:13px;")
            chips.addWidget(nf)
        chips.addStretch(1)
        fl.addLayout(chips)

        # кнопка JSON
        jbtn = QPushButton("Показать JSON и статистику ▾")
        jbtn.setStyleSheet(
            f"QPushButton{{background:transparent; color:{MVRED}; border:none;"
            f"font-size:12px; font-weight:700; text-align:left; padding:4px 0;}}")
        self.json_view = QTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setVisible(False)
        self.json_view.setFixedHeight(220)
        self.json_view.setStyleSheet(
            f"QTextEdit{{background:{MVDARK}; color:#E8ECF4; border-radius:10px;"
            f"font-family:Consolas; font-size:11px; padding:10px;}}")
        payload = {k: v for k, v in res.items() if k != "debug"}
        dbg = res.get("debug", {})
        stats = {
            "задержка_мс": res.get("latency_ms"),
            "слои": {
                "CRF подключён": dbg.get("has_crf"),
                "классификатор_бренда": dbg.get("has_brand_clf"),
                "классификатор_категории": dbg.get("has_category_clf"),
            },
            "словари": {
                "брендов": dbg.get("n_brands_dict"),
                "категорий": dbg.get("n_categories_dict"),
            },
        }
        self.json_view.setPlainText(
            json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\n// статистика\n"
            + json.dumps(stats, ensure_ascii=False, indent=2))
        jbtn.clicked.connect(lambda: self.json_view.setVisible(not self.json_view.isVisible()))
        fl.addWidget(jbtn)
        fl.addWidget(self.json_view)
        self.results_lay.insertWidget(0, facts_card)

        # --- RecSys по фактам
        ranked = self.backend.rank_catalog(res)
        rec_cap = QLabel(f"Товары по фактам ({len(ranked)}) — ранжирование только по извлечённым фактам")
        rec_cap.setStyleSheet(f"color:{MVDARK}; font-size:13px; font-weight:800; margin-top:6px;")
        self.results_lay.insertWidget(1, rec_cap)

        for i, card in enumerate(ranked):
            w = QFrame()
            w.setStyleSheet(
                f"QFrame{{background:white; border:1px solid #E7E7E9; border-radius:12px;}}"
                f"QFrame:hover{{border:1px solid {MVRED};}}")
            cl = QHBoxLayout(w)
            cl.setContentsMargins(16, 12, 16, 12)
            num = QLabel(f"{i+1}")
            num.setFixedSize(30, 30)
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setStyleSheet(
                f"background:{MVRED}; color:white; border-radius:15px; font-weight:800;")
            cl.addWidget(num)
            info = QVBoxLayout()
            name = QLabel(card["name"])
            name.setStyleSheet(f"color:{MVDARK}; font-size:13px; font-weight:600; border:none;")
            name.setWordWrap(True)
            info.addWidget(name)
            price = f"{card['price']:,.0f} ₽".replace(",", " ") if card.get("price") else ""
            why = " · ".join(card.get("why", [])) or "близость по словам фактов"
            meta = QLabel(f"{card.get('brand','')} · {price} · счёт {card['score']} ({why})")
            meta.setStyleSheet(f"color:{MVGRAY}; font-size:11px; border:none;")
            info.addWidget(meta)
            cl.addLayout(info, 1)
            self.results_lay.insertWidget(2 + i, w)

        # плавное появление результатов (fade, как переходы в ГК МОС)
        eff = QGraphicsOpacityEffect(self.results_holder)
        self.results_holder.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity")
        anim.setDuration(350)
        anim.setStartValue(0.3)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self.results_holder.setGraphicsEffect(None))
        self._anim = anim
        anim.start()


# ================================================================= окно
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("М.Видео · Умный поиск")
        self.resize(1120, 800)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.backend = Backend()
        self.splash = SplashScreen(self.show_main)
        self.search = SearchScreen(self.backend)
        self.stack.addWidget(self.splash)
        self.stack.addWidget(self.search)
        self.stack.setCurrentIndex(0)

    def show_main(self):
        # fade-переход, как replaceEnter в ГК МОС (350 мс OutCubic)
        self.stack.setCurrentIndex(1)
        eff = QGraphicsOpacityEffect(self.search)
        self.search.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity")
        anim.setDuration(350)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self.search.setGraphicsEffect(None))
        self._anim = anim
        anim.start()
        self.search.input.setFocus()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
