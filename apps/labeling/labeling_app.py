# -*- coding: utf-8 -*-
"""М.Видео · Разметка — приложение для gold BIO-разметки и match 1/0.

Запуск:  python labeling_app.py [nikita|nekit|liza]
Сборка:  pyinstaller --onefile --windowed --name Разметка_Никита labeling_app.py -- nikita

Горячие клавиши (режим BIO):
    B / I / O     — поставить тег текущему токену
    1..5          — выбрать категорию (BRAND/MODEL/CATEGORY/ATTR/GENRE)
    стрелки ←/→   — переключение токенов
    Enter         — сохранить и следующий запрос
    Backspace     — предыдущий запрос
Режим Match 1/0:
    1 / 0         — метка соответствия
    Enter         — сохранить и дальше
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QColor, QFont, QKeySequence, QShortcut, QPainter, QBrush, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QStackedWidget, QFrame,
    QScrollArea, QComboBox, QSizePolicy, QMessageBox,
)

# ---------------------------------------------------------------- палитра
MVRED = "#F20601"
MVDARK = "#1C1C1E"
MVGRAY = "#6E6E73"
CARDBG = "#F5F5F6"
PINK = "#FDECEC"
OK = "#1F8A50"
ORANGE = "#C75000"

CATEGORIES = ["BRAND", "MODEL", "CATEGORY", "ATTR", "GENRE"]
CAT_COLORS = {
    "BRAND": MVDARK, "MODEL": OK, "CATEGORY": MVRED, "ATTR": ORANGE, "GENRE": "#6A35B0",
}
ATTR_SUBTYPES = ["memory_storage", "size", "color", "connectivity", "weight",
                 "volume", "power", "resolution", "other"]


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def data_dir() -> Path:
    d = app_dir() / "data"
    if d.exists():
        return d
    # PyInstaller onefile: данные рядом с exe
    return app_dir()


# ---------------------------------------------------------------- прогресс-бар
class NiceProgress(QWidget):
    """Красивый анимированный прогресс-бар в стиле М.Видео."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._target = 0.0
        self.setFixedHeight(26)
        self._anim = QPropertyAnimation(self, b"value")
        self._anim.setDuration(350)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._text = ""

    def get_value(self):
        return self._value

    def set_value(self, v):
        self._value = v
        self.update()

    value = pyqtProperty(float, get_value, set_value)

    def set_progress(self, done: int, total: int):
        self._text = f"{done} / {total}"
        target = done / max(total, 1)
        self._anim.stop()
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(target)
        self._anim.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 4, -1, -4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(CARDBG)))
        p.drawRoundedRect(r, 9, 9)
        w = int(r.width() * self._value)
        if w > 4:
            p.setBrush(QBrush(QColor(MVRED)))
            p.drawRoundedRect(r.x(), r.y(), w, r.height(), 9, 9)
        p.setPen(QPen(QColor(MVDARK)))
        f = p.font(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)


# ---------------------------------------------------------------- токен-чип
class TokenChip(QLabel):
    def __init__(self, token: str, idx: int, owner):
        super().__init__(token)
        self.idx = idx
        self.owner = owner
        self.tag = "O"
        self.cat = None
        self.subtype = None
        self.selected = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(46)
        self.setContentsMargins(14, 6, 14, 6)
        f = QFont("Segoe UI", 13); f.setBold(True)
        self.setFont(f)
        self.refresh()

    def mousePressEvent(self, e):
        self.owner.select_token(self.idx)

    def refresh(self):
        border = f"3px solid {MVRED}" if self.selected else "1px solid #DDDDDD"
        if self.tag == "O" or self.cat is None:
            bg, fg = CARDBG, MVDARK
        else:
            bg, fg = CAT_COLORS.get(self.cat, MVGRAY), "white"
        self.setStyleSheet(
            f"background:{bg}; color:{fg}; border:{border}; border-radius:8px; padding:2px 8px;")
        tag_str = self.tag if self.cat is None or self.tag == "O" else f"{self.tag}-{self.cat}"
        sub = f" ({self.subtype})" if self.subtype else ""
        self.setToolTip(tag_str + sub)


# ---------------------------------------------------------------- BIO-страница
class BioPage(QWidget):
    def __init__(self, annotator: str, queries: list[str], out_path: Path):
        super().__init__()
        self.annotator = annotator
        self.queries = queries
        self.out_path = out_path
        self.records: dict[int, dict] = {}
        self._load_existing()
        self.pos = self._first_unlabeled()
        self.chips: list[TokenChip] = []
        self.sel = 0
        self.current_cat = "BRAND"
        self._build()
        self.show_query()

    # ---------- данные
    def _load_existing(self):
        if self.out_path.exists():
            for line in self.out_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.records[r["index"]] = r
                except Exception:
                    pass

    def _first_unlabeled(self) -> int:
        for i in range(len(self.queries)):
            if i not in self.records:
                return i
        return 0

    # ---------- UI
    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 18, 28, 18)
        lay.setSpacing(10)

        top = QHBoxLayout()
        self.lbl_pos = QLabel()
        self.lbl_pos.setStyleSheet(f"color:{MVGRAY}; font-size:12px;")
        top.addWidget(self.lbl_pos)
        top.addStretch(1)
        btn_folder = QPushButton("Открыть папку с разметкой")
        btn_folder.setStyleSheet(self._ghost_btn())
        btn_folder.clicked.connect(self.open_folder)
        top.addWidget(btn_folder)
        lay.addLayout(top)

        self.progress = NiceProgress()
        lay.addWidget(self.progress)

        self.lbl_query = QLabel()
        self.lbl_query.setStyleSheet(
            f"color:{MVDARK}; font-size:15px; font-weight:600; padding:4px 0;")
        self.lbl_query.setWordWrap(True)
        lay.addWidget(self.lbl_query)

        self.chips_row = QHBoxLayout()
        self.chips_row.setSpacing(8)
        chips_holder = QWidget()
        chips_holder.setLayout(self.chips_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(chips_holder)
        scroll.setFixedHeight(80)
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        lay.addWidget(scroll)

        # панель категорий
        cats = QHBoxLayout()
        cats.setSpacing(6)
        self.cat_buttons = {}
        for i, c in enumerate(CATEGORIES):
            b = QPushButton(f"{i+1} · {c}")
            b.setCheckable(True)
            b.clicked.connect(lambda _, cc=c: self.set_category(cc))
            self.cat_buttons[c] = b
            cats.addWidget(b)
        cats.addStretch(1)
        cats.addWidget(QLabel("подтип ATTR:"))
        self.sub_combo = QComboBox()
        self.sub_combo.addItems(["—"] + ATTR_SUBTYPES)
        self.sub_combo.currentTextChanged.connect(self.set_subtype)
        cats.addWidget(self.sub_combo)
        lay.addLayout(cats)
        self._restyle_cats()

        # кнопки тегов
        tags = QHBoxLayout()
        for t, tip in (("B", "начало сущности"), ("I", "продолжение"), ("O", "не сущность")):
            b = QPushButton(f"{t} — {tip}")
            b.setStyleSheet(self._tag_btn())
            b.clicked.connect(lambda _, tt=t: self.set_tag(tt))
            tags.addWidget(b)
        lay.addLayout(tags)

        # низ: навигация
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("← Назад")
        self.btn_next = QPushButton("Сохранить и дальше →")
        self.btn_prev.setStyleSheet(self._ghost_btn())
        self.btn_next.setStyleSheet(self._accent_btn())
        self.btn_prev.clicked.connect(self.prev_query)
        self.btn_next.clicked.connect(self.save_and_next)
        nav.addWidget(self.btn_prev)
        nav.addStretch(1)
        nav.addWidget(self.btn_next)
        lay.addLayout(nav)

        # история
        lay.addWidget(QLabel("История (двойной клик — редактировать):"))
        self.history = QListWidget()
        self.history.setFixedHeight(140)
        self.history.itemDoubleClicked.connect(self.jump_history)
        self.history.setStyleSheet(
            f"QListWidget{{background:{CARDBG}; border:1px solid #DDDDDD; border-radius:8px;"
            f"font-size:11px; color:{MVDARK};}}")
        lay.addWidget(self.history)
        self._fill_history()

        hint = QLabel("Клавиши: B/I/O — тег · 1-5 — категория · ←/→ — токены · Enter — сохранить · Backspace — назад")
        hint.setStyleSheet(f"color:{MVGRAY}; font-size:10px;")
        lay.addWidget(hint)

    def _accent_btn(self):
        return (f"QPushButton{{background:{MVRED}; color:white; border:none; border-radius:8px;"
                f"padding:10px 22px; font-weight:700; font-size:13px;}}"
                f"QPushButton:hover{{background:#D00500;}}")

    def _ghost_btn(self):
        return (f"QPushButton{{background:transparent; color:{MVDARK}; border:1px solid #CCCCCC;"
                f"border-radius:8px; padding:8px 16px; font-size:12px;}}"
                f"QPushButton:hover{{background:{CARDBG};}}")

    def _tag_btn(self):
        return (f"QPushButton{{background:{MVDARK}; color:white; border:none; border-radius:8px;"
                f"padding:9px 18px; font-weight:600;}} QPushButton:hover{{background:#000;}}")

    def _restyle_cats(self):
        for c, b in self.cat_buttons.items():
            active = (c == self.current_cat)
            col = CAT_COLORS[c]
            if active:
                b.setStyleSheet(f"QPushButton{{background:{col}; color:white; border:none;"
                                f"border-radius:8px; padding:8px 14px; font-weight:700;}}")
            else:
                b.setStyleSheet(f"QPushButton{{background:{CARDBG}; color:{MVDARK};"
                                f"border:1px solid #DDDDDD; border-radius:8px; padding:8px 14px;}}")

    # ---------- логика
    def show_query(self):
        q = self.queries[self.pos]
        self.lbl_query.setText(f"Запрос: «{q}»")
        self.lbl_pos.setText(f"Запрос {self.pos + 1} из {len(self.queries)} · размечено {len(self.records)}")
        self.progress.set_progress(len(self.records), len(self.queries))
        # пересоздать чипы
        while self.chips_row.count():
            it = self.chips_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.chips = []
        tokens = q.split()
        saved = self.records.get(self.pos)
        for i, t in enumerate(tokens):
            chip = TokenChip(t, i, self)
            if saved and i < len(saved["tags"]):
                tg = saved["tags"][i]
                if tg != "O" and "-" in tg:
                    chip.tag, chip.cat = tg.split("-", 1)[0], tg.split("-", 1)[1]
                else:
                    chip.tag = tg
                subs = saved.get("subtypes") or {}
                chip.subtype = subs.get(str(i))
                chip.refresh()
            self.chips.append(chip)
            self.chips_row.addWidget(chip)
        self.chips_row.addStretch(1)
        self.sel = 0
        self._select_refresh()

    def _select_refresh(self):
        for c in self.chips:
            c.selected = (c.idx == self.sel)
            c.refresh()

    def select_token(self, idx: int):
        self.sel = idx
        self._select_refresh()

    def set_category(self, cat: str):
        self.current_cat = cat
        self._restyle_cats()
        if self.chips and self.chips[self.sel].tag != "O":
            self.chips[self.sel].cat = cat
            self.chips[self.sel].refresh()

    def set_subtype(self, s: str):
        if self.chips:
            self.chips[self.sel].subtype = None if s == "—" else s
            self.chips[self.sel].refresh()

    def set_tag(self, tag: str):
        if not self.chips:
            return
        chip = self.chips[self.sel]
        chip.tag = tag
        chip.cat = None if tag == "O" else self.current_cat
        chip.refresh()
        # автопереход к следующему токену
        if self.sel < len(self.chips) - 1:
            self.sel += 1
            self._select_refresh()

    def move_sel(self, d: int):
        if self.chips:
            self.sel = max(0, min(len(self.chips) - 1, self.sel + d))
            self._select_refresh()

    def save_and_next(self):
        tags = []
        subtypes = {}
        for c in self.chips:
            tags.append("O" if c.tag == "O" or c.cat is None else f"{c.tag}-{c.cat}")
            if c.subtype:
                subtypes[str(c.idx)] = c.subtype
        rec = {
            "index": self.pos,
            "query": self.queries[self.pos],
            "tags": tags,
            "subtypes": subtypes,
            "annotator": self.annotator,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self.records[self.pos] = rec
        self._flush()
        self._fill_history()
        if self.pos < len(self.queries) - 1:
            self.pos += 1
        self.show_query()

    def prev_query(self):
        if self.pos > 0:
            self.pos -= 1
            self.show_query()

    def jump_history(self, item: QListWidgetItem):
        self.pos = item.data(Qt.ItemDataRole.UserRole)
        self.show_query()

    def _fill_history(self):
        self.history.clear()
        for i in sorted(self.records, reverse=True)[:200]:
            r = self.records[i]
            tag_str = " ".join(r["tags"])
            it = QListWidgetItem(f"#{i+1} · {r['query']}  →  {tag_str}")
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.history.addItem(it)

    def _flush(self):
        lines = [json.dumps(self.records[k], ensure_ascii=False)
                 for k in sorted(self.records)]
        self.out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def open_folder(self):
        subprocess.Popen(f'explorer /select,"{self.out_path}"')

    # клавиатура
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_B:
            self.set_tag("B")
        elif k == Qt.Key.Key_I:
            self.set_tag("I")
        elif k == Qt.Key.Key_O:
            self.set_tag("O")
        elif Qt.Key.Key_1 <= k <= Qt.Key.Key_5:
            self.set_category(CATEGORIES[k - Qt.Key.Key_1])
        elif k == Qt.Key.Key_Left:
            self.move_sel(-1)
        elif k == Qt.Key.Key_Right:
            self.move_sel(1)
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.save_and_next()
        elif k == Qt.Key.Key_Backspace:
            self.prev_query()
        else:
            super().keyPressEvent(e)


# ---------------------------------------------------------------- Match 1/0
class MatchPage(QWidget):
    def __init__(self, annotator: str, pairs: list[dict], out_path: Path):
        super().__init__()
        self.annotator = annotator
        self.pairs = pairs
        self.out_path = out_path
        self.records: dict[int, dict] = {}
        self._load_existing()
        self.pos = self._first_unlabeled()
        self._build()
        self.show_pair()

    def _load_existing(self):
        if self.out_path.exists():
            for line in self.out_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.records[r["index"]] = r
                except Exception:
                    pass

    def _first_unlabeled(self):
        for i in range(len(self.pairs)):
            if i not in self.records:
                return i
        return 0

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 18, 28, 18)
        lay.setSpacing(12)

        top = QHBoxLayout()
        self.lbl_pos = QLabel()
        self.lbl_pos.setStyleSheet(f"color:{MVGRAY}; font-size:12px;")
        top.addWidget(self.lbl_pos)
        top.addStretch(1)
        btn_folder = QPushButton("Открыть папку с разметкой")
        btn_folder.setStyleSheet(
            f"QPushButton{{background:transparent; color:{MVDARK}; border:1px solid #CCCCCC;"
            f"border-radius:8px; padding:8px 16px;}} QPushButton:hover{{background:{CARDBG};}}")
        btn_folder.clicked.connect(self.open_folder)
        top.addWidget(btn_folder)
        lay.addLayout(top)

        self.progress = NiceProgress()
        lay.addWidget(self.progress)

        card = QFrame()
        card.setStyleSheet(f"background:{CARDBG}; border-radius:12px;")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(22, 18, 22, 18)
        self.lbl_query = QLabel()
        self.lbl_query.setStyleSheet(f"color:{MVRED}; font-size:17px; font-weight:700;")
        self.lbl_query.setWordWrap(True)
        self.lbl_sku = QLabel()
        self.lbl_sku.setStyleSheet(f"color:{MVDARK}; font-size:14px; font-weight:600;")
        self.lbl_sku.setWordWrap(True)
        self.lbl_meta = QLabel()
        self.lbl_meta.setStyleSheet(f"color:{MVGRAY}; font-size:12px;")
        cl.addWidget(QLabel("Запрос пользователя:"))
        cl.addWidget(self.lbl_query)
        cl.addSpacing(8)
        cl.addWidget(QLabel("Карточка товара:"))
        cl.addWidget(self.lbl_sku)
        cl.addWidget(self.lbl_meta)
        lay.addWidget(card)

        btns = QHBoxLayout()
        self.btn1 = QPushButton("1 — соответствует")
        self.btn0 = QPushButton("0 — не соответствует")
        self.btn1.setStyleSheet(
            f"QPushButton{{background:{OK}; color:white; border:none; border-radius:10px;"
            f"padding:16px; font-size:15px; font-weight:700;}} QPushButton:hover{{background:#166B3E;}}")
        self.btn0.setStyleSheet(
            f"QPushButton{{background:{MVRED}; color:white; border:none; border-radius:10px;"
            f"padding:16px; font-size:15px; font-weight:700;}} QPushButton:hover{{background:#C00500;}}")
        self.btn1.clicked.connect(lambda: self.mark(1))
        self.btn0.clicked.connect(lambda: self.mark(0))
        btns.addWidget(self.btn1)
        btns.addWidget(self.btn0)
        lay.addLayout(btns)

        nav = QHBoxLayout()
        b_prev = QPushButton("← Назад")
        b_next = QPushButton("Пропустить →")
        for b in (b_prev, b_next):
            b.setStyleSheet(
                f"QPushButton{{background:transparent; color:{MVDARK}; border:1px solid #CCCCCC;"
                f"border-radius:8px; padding:8px 18px;}} QPushButton:hover{{background:{CARDBG};}}")
        b_prev.clicked.connect(self.prev_pair)
        b_next.clicked.connect(self.next_pair)
        nav.addWidget(b_prev)
        nav.addStretch(1)
        nav.addWidget(b_next)
        lay.addLayout(nav)

        lay.addWidget(QLabel("История (двойной клик — редактировать):"))
        self.history = QListWidget()
        self.history.setFixedHeight(150)
        self.history.itemDoubleClicked.connect(self.jump_history)
        self.history.setStyleSheet(
            f"QListWidget{{background:{CARDBG}; border:1px solid #DDDDDD; border-radius:8px;"
            f"font-size:11px; color:{MVDARK};}}")
        lay.addWidget(self.history)
        self._fill_history()

        hint = QLabel("Клавиши: 1 / 0 — метка · ←/→ — навигация")
        hint.setStyleSheet(f"color:{MVGRAY}; font-size:10px;")
        lay.addWidget(hint)

    def show_pair(self):
        p = self.pairs[self.pos]
        self.lbl_query.setText(f"«{p['query']}»")
        self.lbl_sku.setText(p["sku_name"])
        price = f"{p['price']:,.0f} ₽".replace(",", " ") if p.get("price") else "—"
        self.lbl_meta.setText(f"Бренд: {p.get('brand','—')} · Цена: {price}")
        self.lbl_pos.setText(
            f"Пара {self.pos + 1} из {len(self.pairs)} · размечено {len(self.records)}")
        self.progress.set_progress(len(self.records), len(self.pairs))

    def mark(self, label: int):
        p = self.pairs[self.pos]
        rec = {
            "index": self.pos,
            "query": p["query"],
            "sku_name": p["sku_name"],
            "brand": p.get("brand"),
            "label": label,
            "annotator": self.annotator,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self.records[self.pos] = rec
        self._flush()
        self._fill_history()
        self.next_pair()

    def next_pair(self):
        if self.pos < len(self.pairs) - 1:
            self.pos += 1
        self.show_pair()

    def prev_pair(self):
        if self.pos > 0:
            self.pos -= 1
        self.show_pair()

    def jump_history(self, item):
        self.pos = item.data(Qt.ItemDataRole.UserRole)
        self.show_pair()

    def _fill_history(self):
        self.history.clear()
        for i in sorted(self.records, reverse=True)[:200]:
            r = self.records[i]
            it = QListWidgetItem(f"#{i+1} · [{r['label']}] {r['query']} ↔ {r['sku_name'][:60]}")
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.history.addItem(it)

    def _flush(self):
        lines = [json.dumps(self.records[k], ensure_ascii=False) for k in sorted(self.records)]
        self.out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def open_folder(self):
        subprocess.Popen(f'explorer /select,"{self.out_path}"')

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_1:
            self.mark(1)
        elif k == Qt.Key.Key_0:
            self.mark(0)
        elif k == Qt.Key.Key_Left:
            self.prev_pair()
        elif k == Qt.Key.Key_Right:
            self.next_pair()
        else:
            super().keyPressEvent(e)


# ---------------------------------------------------------------- главное окно
class MainWindow(QMainWindow):
    def __init__(self, annotator_key: str):
        super().__init__()
        dd = data_dir()
        q_file = dd / f"queries_{annotator_key}.json"
        p_file = dd / f"pairs_{annotator_key}.json"
        if not q_file.exists():
            QMessageBox.critical(self, "Ошибка", f"Не найден файл данных: {q_file}")
            sys.exit(1)
        qdata = json.loads(q_file.read_text(encoding="utf-8"))
        pdata = json.loads(p_file.read_text(encoding="utf-8"))
        display = qdata.get("annotator", annotator_key)

        self.setWindowTitle(f"М.Видео · Разметка — {display}")
        self.resize(1060, 780)
        self.setStyleSheet(f"QMainWindow{{background:white;}} QLabel{{color:{MVDARK};}}")

        out_dir = app_dir() / "labels"
        out_dir.mkdir(exist_ok=True)

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # шапка
        header = QFrame()
        header.setStyleSheet(f"background:{MVRED};")
        header.setFixedHeight(64)
        h = QHBoxLayout(header)
        h.setContentsMargins(24, 0, 24, 0)
        t = QLabel(f"М.Видео · Разметка · {display}")
        t.setStyleSheet("color:white; font-size:17px; font-weight:800;")
        h.addWidget(t)
        h.addStretch(1)
        self.btn_bio = QPushButton("BIO-разметка")
        self.btn_match = QPushButton("Match 1/0")
        for b in (self.btn_bio, self.btn_match):
            h.addWidget(b)
        v.addWidget(header)

        self.stack = QStackedWidget()
        self.bio = BioPage(display, qdata["queries"], out_dir / f"bio_{annotator_key}.jsonl")
        self.match = MatchPage(display, pdata["pairs"], out_dir / f"match_{annotator_key}.jsonl")
        self.stack.addWidget(self.bio)
        self.stack.addWidget(self.match)
        v.addWidget(self.stack)
        self.setCentralWidget(central)

        self.btn_bio.clicked.connect(lambda: self.switch(0))
        self.btn_match.clicked.connect(lambda: self.switch(1))
        self.switch(0)

    def switch(self, idx: int):
        self.stack.setCurrentIndex(idx)
        on = ("QPushButton{background:white; color:#F20601; border:none; border-radius:8px;"
              "padding:9px 18px; font-weight:800;}")
        off = ("QPushButton{background:rgba(255,255,255,0.18); color:white; border:none;"
               "border-radius:8px; padding:9px 18px; font-weight:600;}")
        self.btn_bio.setStyleSheet(on if idx == 0 else off)
        self.btn_match.setStyleSheet(off if idx == 0 else on)
        self.stack.currentWidget().setFocus()


def main():
    annotator = "nikita"
    for a in sys.argv[1:]:
        if a.lower() in ("nikita", "nekit", "liza"):
            annotator = a.lower()
    # при сборке exe аргумент зашивается через env
    annotator = os.environ.get("MV_ANNOTATOR", annotator)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(annotator)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
