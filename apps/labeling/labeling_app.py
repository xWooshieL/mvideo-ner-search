# -*- coding: utf-8 -*-
"""М.Видео · Разметка — трёхэтапный мастер разметки запросов.

Сценарий работы (BIO-режим):
  Этап 1 — BIO. Запрос разбит на блоки по центру, фон заблюрен.
           Клавиши B / I / O ставят тег (метка появляется над блоком) и
           двигают вперёд. Стрелки — навигация, Backspace — снять текущий
           тег и шагнуть назад. Enter на последнем блоке -> «Подтверждаете?»
  Этап 2 — Типы. Для каждого блока с тегом B/I выбираем тип (1-5 или
           стрелками): BRAND, MODEL, CATEGORY, ATTR, GENRE. У каждого типа
           описание — что это и когда ставить. Enter на последнем ->
           «Точно всё проставили?»
  Этап 3 — Подтипы ATTR (если есть атрибуты). Для каждого ATTR-спана
           выбираем подтип с описанием и переводом; «другое» — ручной ввод.
  После подтверждения запись сохраняется и открывается следующий запрос.

Режим Match 1/0:
  1 — карточка соответствует запросу, 0 — нет. Если карточки нет
  (пустая пара), ставится 0 автоматически. На экране — краткий гайд,
  когда ставить 1, а когда 0.

Запуск: python labeling_app.py [nikita|nekit|liza]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QColor, QFont, QPainter, QBrush, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QStackedWidget, QFrame,
    QScrollArea, QMessageBox, QGraphicsBlurEffect, QLineEdit, QDialog,
)

MVRED = "#F20601"
MVDARK = "#1C1C1E"
MVGRAY = "#6E6E73"
CARDBG = "#F5F5F6"
PINK = "#FDECEC"
OK = "#1F8A50"
ORANGE = "#C75000"
PURPLE = "#6A35B0"

CATEGORIES = ["BRAND", "MODEL", "CATEGORY", "ATTR", "GENRE"]
CAT_COLORS = {"BRAND": MVDARK, "MODEL": OK, "CATEGORY": MVRED, "ATTR": ORANGE, "GENRE": PURPLE}
CAT_DESC = {
    "BRAND": "Производитель товара: apple, samsung, dyson, xiaomi. Ставим, когда слово — название компании-бренда (в т.ч. русской транслитерацией: «самсунг»).",
    "MODEL": "Линейка или модель после бренда: zenbook, v15, galaxy s24, airpods pro. Ставим на хвост, который уточняет конкретную серию устройства.",
    "CATEGORY": "Тип товара: ноутбук, телефон, пылесос, наушники. Ставим на слово, которое говорит, ЧТО ищет человек.",
    "ATTR": "Характеристика: 16 гб, 55 дюймов, красный, wi-fi. Ставим на числа с единицами, цвета и свойства. Подтип уточним на следующем шаге.",
    "GENRE": "Жанр/тематика для игр, книг, фильмов: хоррор, стратегия, фэнтези. Ставим редко — только для медиа-товаров.",
}
ATTR_SUBTYPES = [
    ("memory_storage", "память / накопитель", "объём памяти: 16 гб, 512 gb, 1 тб"),
    ("size", "размер / диагональ", "габариты и диагонали: 55 дюймов, 60 см"),
    ("color", "цвет", "белый, чёрный, красный…"),
    ("connectivity", "связь", "wi-fi, bluetooth, nfc, 5g, usb-c"),
    ("weight", "вес", "2 кг, 500 г"),
    ("volume", "объём", "1 л, 500 мл"),
    ("power", "мощность", "2000 вт, 1.5 квт"),
    ("resolution", "разрешение", "4k, full hd, 1920x1080"),
    ("other", "другое (ввести вручную)", "если ни один подтип не подходит"),
]


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def data_dir() -> Path:
    d = app_dir() / "data"
    return d if d.exists() else app_dir()


# ---------------------------------------------------------------- прогресс-бар
class NiceProgress(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self.setFixedHeight(24)
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
        self._anim.stop()
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(done / max(total, 1))
        self._anim.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 3, -1, -3)
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


# ---------------------------------------------------------------- блок токена
class TokenBlock(QWidget):
    """Блок токена с меткой разметки над ним."""

    def __init__(self, token: str, idx: int, owner):
        super().__init__()
        self.token = token
        self.idx = idx
        self.owner = owner
        self.bio = None        # "B" | "I" | "O" | None
        self.cat = None        # тип сущности
        self.subtype = None    # подтип ATTR
        self.active = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(4)
        self.tag_lbl = QLabel(" ")
        self.tag_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tag_lbl.setFixedHeight(24)
        lay.addWidget(self.tag_lbl)
        self.tok_lbl = QLabel(token)
        self.tok_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tok_lbl.setMinimumHeight(56)
        self.tok_lbl.setContentsMargins(18, 8, 18, 8)
        f = QFont("Segoe UI", 15); f.setBold(True)
        self.tok_lbl.setFont(f)
        lay.addWidget(self.tok_lbl)
        self.refresh()

    def mousePressEvent(self, _):
        self.owner.jump_to(self.idx)

    def refresh(self):
        # метка над блоком
        if self.bio is None:
            self.tag_lbl.setText(" ")
            self.tag_lbl.setStyleSheet("background:transparent;")
        else:
            parts = [self.bio]
            if self.cat and self.bio != "O":
                parts.append(self.cat)
            txt = "-".join(parts)
            if self.subtype:
                txt += f" · {self.subtype}"
            col = CAT_COLORS.get(self.cat, MVGRAY) if self.bio != "O" else MVGRAY
            self.tag_lbl.setText(txt)
            self.tag_lbl.setStyleSheet(
                f"background:{col}; color:white; border-radius:6px;"
                f"font-size:10px; font-weight:700; padding:2px 6px;")
        # сам блок
        border = f"3px solid {MVRED}" if self.active else "1px solid #D6D6D8"
        if self.bio == "O":
            bg, fg = "#ECECEE", MVGRAY
        elif self.bio and self.cat:
            bg, fg = CAT_COLORS.get(self.cat, CARDBG), "white"
        elif self.bio:
            bg, fg = MVDARK, "white"
        else:
            bg, fg = "white", MVDARK
        self.tok_lbl.setStyleSheet(
            f"background:{bg}; color:{fg}; border:{border}; border-radius:10px;")


# ---------------------------------------------------------------- подтверждение
def ask_confirm(parent, title: str, text: str) -> bool:
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    yes = box.addButton("Да, подтверждаю", QMessageBox.ButtonRole.YesRole)
    box.addButton("Нет, ещё поправлю", QMessageBox.ButtonRole.NoRole)
    box.setStyleSheet(
        f"QMessageBox{{background:white;}} QLabel{{color:{MVDARK}; font-size:13px;}}"
        f"QPushButton{{background:{MVRED}; color:white; border:none; border-radius:8px;"
        f"padding:8px 18px; font-weight:700;}}")
    box.exec()
    return box.clickedButton() is yes


# ---------------------------------------------------------------- BIO-мастер
class BioPage(QWidget):
    """Трёхэтапный мастер: BIO -> типы -> подтипы ATTR."""

    ST_BIO, ST_TYPE, ST_SUBTYPE = 0, 1, 2

    def __init__(self, annotator: str, queries: list[str], out_path: Path):
        super().__init__()
        self.annotator = annotator
        self.queries = queries
        self.out_path = out_path
        self.records: dict[int, dict] = {}
        self._load_existing()
        self.pos = self._first_unlabeled()
        self.blocks: list[TokenBlock] = []
        self.stage = self.ST_BIO
        self.cursor = 0            # индекс активного блока (этап 1/2) или ATTR-спана (этап 3)
        self.attr_ids: list[int] = []
        self.sub_choice = 0
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
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 14, 24, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.lbl_pos = QLabel()
        self.lbl_pos.setStyleSheet(f"color:{MVGRAY}; font-size:12px;")
        top.addWidget(self.lbl_pos)
        top.addStretch(1)
        btn_folder = QPushButton("Открыть папку с разметкой")
        btn_folder.setStyleSheet(self._ghost())
        btn_folder.clicked.connect(self.open_folder)
        top.addWidget(btn_folder)
        root.addLayout(top)

        self.progress = NiceProgress()
        root.addWidget(self.progress)

        # ---- центральная сцена: блюр-фон + блоки поверх
        scene = QFrame()
        scene.setStyleSheet("background:transparent;")
        scene_lay = QVBoxLayout(scene)
        scene_lay.setContentsMargins(0, 4, 0, 4)

        # фоновая "витрина" с историей — её и блюрим
        self.backdrop = QFrame()
        self.backdrop.setStyleSheet(
            f"background:{CARDBG}; border-radius:14px;")
        bd_lay = QVBoxLayout(self.backdrop)
        bd_lay.setContentsMargins(18, 12, 18, 12)
        bd_cap = QLabel("История разметок (двойной клик — редактировать)")
        bd_cap.setStyleSheet(f"color:{MVGRAY}; font-size:11px; font-weight:700;")
        bd_lay.addWidget(bd_cap)
        self.history = QListWidget()
        self.history.itemDoubleClicked.connect(self.jump_history)
        self.history.setStyleSheet(
            f"QListWidget{{background:white; border:1px solid #E3E3E5; border-radius:8px;"
            f"font-size:11px; color:{MVDARK};}}")
        bd_lay.addWidget(self.history)
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(7)
        self.backdrop.setGraphicsEffect(blur)
        self._blur = blur

        # передний план: этап + блоки + панель этапа
        self.front = QFrame()
        self.front.setStyleSheet(
            "background:rgba(255,255,255,0.92); border-radius:16px;")
        fr = QVBoxLayout(self.front)
        fr.setContentsMargins(26, 16, 26, 16)
        fr.setSpacing(10)

        self.stage_lbl = QLabel()
        self.stage_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage_lbl.setStyleSheet(
            f"color:{MVRED}; font-size:12px; font-weight:800; letter-spacing:1px;")
        fr.addWidget(self.stage_lbl)

        self.query_lbl = QLabel()
        self.query_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.query_lbl.setStyleSheet(f"color:{MVGRAY}; font-size:12px;")
        fr.addWidget(self.query_lbl)

        # ряд блоков по центру
        self.blocks_row = QHBoxLayout()
        self.blocks_row.setSpacing(10)
        holder = QWidget()
        holder.setLayout(self.blocks_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        scroll.setFixedHeight(112)
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        fr.addWidget(scroll)

        # панель подсказки этапа (описания типов/подтипов)
        self.panel = QFrame()
        self.panel.setStyleSheet(f"background:{PINK}; border-radius:10px;")
        pl = QVBoxLayout(self.panel)
        pl.setContentsMargins(14, 10, 14, 10)
        pl.setSpacing(4)
        self.panel_title = QLabel()
        self.panel_title.setStyleSheet(f"color:{MVRED}; font-size:12px; font-weight:800;")
        pl.addWidget(self.panel_title)
        self.panel_body = QLabel()
        self.panel_body.setWordWrap(True)
        self.panel_body.setStyleSheet(f"color:{MVDARK}; font-size:12px;")
        self.panel_body.setMinimumHeight(150)
        self.panel_body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        pl.addWidget(self.panel_body)
        fr.addWidget(self.panel)

        # поле ручного ввода подтипа (для «другое»)
        self.manual_edit = QLineEdit()
        self.manual_edit.setPlaceholderText("Введите свой подтип и нажмите Enter…")
        self.manual_edit.setStyleSheet(
            f"QLineEdit{{border:2px solid {ORANGE}; border-radius:8px; padding:8px 12px;"
            f"font-size:13px; color:{MVDARK};}}")
        self.manual_edit.returnPressed.connect(self._manual_done)
        self.manual_edit.hide()
        fr.addWidget(self.manual_edit)

        self.hint_lbl = QLabel()
        self.hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_lbl.setStyleSheet(f"color:{MVGRAY}; font-size:10px;")
        fr.addWidget(self.hint_lbl)

        # стопка: блюр-фон под передним планом
        stack_holder = QFrame()
        sh = QVBoxLayout(stack_holder)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(self.backdrop)
        self.front.setParent(stack_holder)
        scene_lay.addWidget(stack_holder)
        root.addWidget(scene, 1)
        self._stack_holder = stack_holder

        # низ: навигация по запросам
        nav = QHBoxLayout()
        b_prev = QPushButton("← Предыдущий запрос")
        b_next = QPushButton("Следующий запрос →")
        for b in (b_prev, b_next):
            b.setStyleSheet(self._ghost())
        b_prev.clicked.connect(self.prev_query)
        b_next.clicked.connect(self.skip_query)
        nav.addWidget(b_prev)
        nav.addStretch(1)
        nav.addWidget(b_next)
        root.addLayout(nav)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # передний план поверх блюра, по центру
        if hasattr(self, "_stack_holder"):
            w = self._stack_holder.width()
            h = self._stack_holder.height()
            fw = min(w - 30, 940)
            fh = min(h - 16, 470)
            self.front.setGeometry((w - fw) // 2, (h - fh) // 2, fw, fh)
            self.front.raise_()

    def _ghost(self):
        return (f"QPushButton{{background:transparent; color:{MVDARK}; border:1px solid #CCCCCC;"
                f"border-radius:8px; padding:8px 16px; font-size:12px;}}"
                f"QPushButton:hover{{background:{CARDBG};}}")

    # ---------- показ запроса
    def show_query(self):
        q = self.queries[self.pos]
        self.stage = self.ST_BIO
        self.cursor = 0
        self.sub_choice = 0
        self.manual_edit.hide()
        self.query_lbl.setText(f"Запрос {self.pos + 1} из {len(self.queries)}: «{q}»")
        self.lbl_pos.setText(f"Размечено {len(self.records)} из {len(self.queries)}")
        self.progress.set_progress(len(self.records), len(self.queries))

        while self.blocks_row.count():
            it = self.blocks_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.blocks = []
        self.blocks_row.addStretch(1)
        saved = self.records.get(self.pos)
        for i, t in enumerate(q.split()):
            blk = TokenBlock(t, i, self)
            if saved and i < len(saved["tags"]):
                tg = saved["tags"][i]
                if tg == "O":
                    blk.bio = "O"
                elif "-" in tg:
                    blk.bio, blk.cat = tg.split("-", 1)
                blk.subtype = (saved.get("subtypes") or {}).get(str(i))
            self.blocks.append(blk)
            self.blocks_row.insertWidget(self.blocks_row.count() - 1, blk)
        self.blocks_row.addStretch(1)
        self._fill_history()
        self._sync_stage()

    # ---------- этапы
    def _sync_stage(self):
        for b in self.blocks:
            b.active = False
        if self.stage == self.ST_BIO:
            self.stage_lbl.setText("ЭТАП 1 · РАЗМЕТКА B / I / O")
            self.panel_title.setText("Клавиши: B — начало сущности · I — продолжение · O — не сущность")
            self.panel_body.setText(
                "B ставим на первое слово сущности, I — на её продолжение, O — на служебные слова "
                "(«для», «купить», «недорого»). Метка появляется над блоком. "
                "Backspace — снять метку и шагнуть назад, стрелки — навигация, Enter на последнем блоке — подтвердить этап.")
            if self.blocks:
                self.cursor = min(self.cursor, len(self.blocks) - 1)
                self.blocks[self.cursor].active = True
            self.hint_lbl.setText("B / I / O — тег · ←/→ — блоки · Backspace — снять и назад · Enter — подтвердить")
        elif self.stage == self.ST_TYPE:
            self.stage_lbl.setText("ЭТАП 2 · ТИП СУЩНОСТИ (1–5)")
            ent = [i for i, b in enumerate(self.blocks) if b.bio in ("B", "I")]
            if not ent:
                self._finish_types()
                return
            self.cursor = min(self.cursor, len(ent) - 1)
            cur_block = self.blocks[ent[self.cursor]]
            cur_block.active = True
            cur = cur_block.cat or "BRAND"
            short = {
                "BRAND": "производитель: apple, dyson, «самсунг»",
                "MODEL": "линейка/модель: zenbook, v15, galaxy s24",
                "CATEGORY": "тип товара: ноутбук, пылесос",
                "ATTR": "характеристика: 16 гб, красный, wi-fi",
                "GENRE": "жанр для игр/книг/фильмов",
            }
            lines = []
            for i, c in enumerate(CATEGORIES):
                mark = "▶" if c == cur else "   "
                lines.append(f"{mark} {i+1} · {c} — {short[c]}")
            lines.append("")
            lines.append(f"Выбран {cur}: {CAT_DESC[cur]}")
            self.panel_title.setText(f"Блок «{cur_block.token}» — выберите тип:")
            self.panel_body.setText("\n".join(lines))
            self.hint_lbl.setText("1–5 — тип · ↑/↓ — выбор типа · ←/→ — блоки · Enter — подтвердить этап · Backspace — назад")
        else:
            self.stage_lbl.setText("ЭТАП 3 · ПОДТИП АТРИБУТА")
            if not self.attr_ids:
                self._save_record()
                return
            self.cursor = min(self.cursor, len(self.attr_ids) - 1)
            blk = self.blocks[self.attr_ids[self.cursor]]
            blk.active = True
            lines = []
            for i, (code, ru, desc) in enumerate(ATTR_SUBTYPES):
                mark = "▶" if i == self.sub_choice else " "
                lines.append(f"{mark} {i+1} · {ru} ({code}) — {desc}")
            self.panel_title.setText(f"Атрибут «{blk.token}» — выберите подтип:")
            self.panel_body.setText("\n".join(lines))
            self.hint_lbl.setText("1–9 или ↑/↓ — подтип · Enter — применить и дальше · Backspace — назад")
        for b in self.blocks:
            b.refresh()

    # ---------- этап 1: BIO
    def _bio_set(self, tag: str):
        blk = self.blocks[self.cursor]
        blk.bio = tag
        if tag == "O":
            blk.cat = None
            blk.subtype = None
        blk.refresh()
        if self.cursor < len(self.blocks) - 1:
            self.cursor += 1
        self._sync_stage()

    def _bio_backspace(self):
        blk = self.blocks[self.cursor]
        if blk.bio is not None:
            blk.bio = None
            blk.cat = None
            blk.refresh()
        if self.cursor > 0:
            self.cursor -= 1
        self._sync_stage()

    def _bio_enter(self):
        unset = [b.token for b in self.blocks if b.bio is None]
        if unset:
            QMessageBox.information(self, "Не всё размечено",
                                    "Остались блоки без метки: " + ", ".join(unset))
            return
        if self.cursor == len(self.blocks) - 1 or all(b.bio for b in self.blocks):
            if ask_confirm(self, "Подтверждение", "Вы подтверждаете разметку B/I/O?"):
                self.stage = self.ST_TYPE
                self.cursor = 0
                # дефолтный тип для сущностей
                for b in self.blocks:
                    if b.bio in ("B", "I") and b.cat is None:
                        b.cat = "BRAND"
                self._sync_stage()

    # ---------- этап 2: типы
    def _type_set(self, cat: str):
        ent = [i for i, b in enumerate(self.blocks) if b.bio in ("B", "I")]
        if not ent:
            return
        blk = self.blocks[ent[self.cursor]]
        blk.cat = cat
        blk.refresh()
        if self.cursor < len(ent) - 1:
            self.cursor += 1
        self._sync_stage()

    def _type_cycle(self, d: int):
        ent = [i for i, b in enumerate(self.blocks) if b.bio in ("B", "I")]
        if not ent:
            return
        blk = self.blocks[ent[self.cursor]]
        cur = CATEGORIES.index(blk.cat or "BRAND")
        blk.cat = CATEGORIES[(cur + d) % len(CATEGORIES)]
        blk.refresh()
        self._sync_stage()

    def _type_enter(self):
        ent = [i for i, b in enumerate(self.blocks) if b.bio in ("B", "I")]
        if self.cursor >= len(ent) - 1:
            if ask_confirm(self, "Подтверждение", "Точно проставили все типы?"):
                self._finish_types()
        else:
            self.cursor += 1
            self._sync_stage()

    def _finish_types(self):
        self.attr_ids = [i for i, b in enumerate(self.blocks) if b.bio in ("B", "I") and b.cat == "ATTR"]
        if self.attr_ids:
            self.stage = self.ST_SUBTYPE
            self.cursor = 0
            self.sub_choice = 0
            self._sync_stage()
        else:
            self._save_record()

    # ---------- этап 3: подтипы
    def _sub_enter(self):
        code, _ru, _d = ATTR_SUBTYPES[self.sub_choice]
        blk = self.blocks[self.attr_ids[self.cursor]]
        if code == "other":
            self.manual_edit.show()
            self.manual_edit.setFocus()
            return
        blk.subtype = code
        blk.refresh()
        self._sub_next()

    def _manual_done(self):
        txt = self.manual_edit.text().strip()
        if not txt:
            return
        blk = self.blocks[self.attr_ids[self.cursor]]
        blk.subtype = txt
        blk.refresh()
        self.manual_edit.clear()
        self.manual_edit.hide()
        self.setFocus()
        self._sub_next()

    def _sub_next(self):
        if self.cursor < len(self.attr_ids) - 1:
            self.cursor += 1
            self.sub_choice = 0
            self._sync_stage()
        else:
            self._save_record()

    # ---------- сохранение
    def _save_record(self):
        tags, subtypes = [], {}
        for b in self.blocks:
            tags.append("O" if b.bio in (None, "O") else f"{b.bio}-{b.cat}")
            if b.subtype:
                subtypes[str(b.idx)] = b.subtype
        rec = {
            "index": self.pos,
            "query": self.queries[self.pos],
            "tags": tags,
            "subtypes": subtypes,
            "annotator": self.annotator,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self.records[self.pos] = rec
        lines = [json.dumps(self.records[k], ensure_ascii=False) for k in sorted(self.records)]
        self.out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.pos < len(self.queries) - 1:
            self.pos += 1
        self.show_query()

    # ---------- навигация по запросам
    def prev_query(self):
        if self.pos > 0:
            self.pos -= 1
            self.show_query()

    def skip_query(self):
        if self.pos < len(self.queries) - 1:
            self.pos += 1
            self.show_query()

    def jump_to(self, idx: int):
        if self.stage == self.ST_BIO:
            self.cursor = idx
            self._sync_stage()

    def jump_history(self, item: QListWidgetItem):
        self.pos = item.data(Qt.ItemDataRole.UserRole)
        self.show_query()

    def _fill_history(self):
        self.history.clear()
        for i in sorted(self.records, reverse=True)[:300]:
            r = self.records[i]
            it = QListWidgetItem(f"#{i+1} · {r['query']}  →  {' '.join(r['tags'])}")
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.history.addItem(it)

    def open_folder(self):
        subprocess.Popen(f'explorer /select,"{self.out_path}"')

    # ---------- клавиатура
    def keyPressEvent(self, e):
        k = e.key()
        if self.manual_edit.isVisible():
            super().keyPressEvent(e)
            return
        if self.stage == self.ST_BIO:
            if k == Qt.Key.Key_B:
                self._bio_set("B")
            elif k == Qt.Key.Key_I:
                self._bio_set("I")
            elif k == Qt.Key.Key_O:
                self._bio_set("O")
            elif k == Qt.Key.Key_Left:
                self.cursor = max(0, self.cursor - 1); self._sync_stage()
            elif k == Qt.Key.Key_Right:
                self.cursor = min(len(self.blocks) - 1, self.cursor + 1); self._sync_stage()
            elif k == Qt.Key.Key_Backspace:
                self._bio_backspace()
            elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._bio_enter()
            else:
                super().keyPressEvent(e)
        elif self.stage == self.ST_TYPE:
            if Qt.Key.Key_1 <= k <= Qt.Key.Key_5:
                self._type_set(CATEGORIES[k - Qt.Key.Key_1])
            elif k == Qt.Key.Key_Up:
                self._type_cycle(-1)
            elif k == Qt.Key.Key_Down:
                self._type_cycle(1)
            elif k == Qt.Key.Key_Left:
                self.cursor = max(0, self.cursor - 1); self._sync_stage()
            elif k == Qt.Key.Key_Right:
                ent = [i for i, b in enumerate(self.blocks) if b.bio in ("B", "I")]
                self.cursor = min(len(ent) - 1, self.cursor + 1); self._sync_stage()
            elif k == Qt.Key.Key_Backspace:
                self.stage = self.ST_BIO
                self.cursor = len(self.blocks) - 1
                self._sync_stage()
            elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._type_enter()
            else:
                super().keyPressEvent(e)
        else:
            if Qt.Key.Key_1 <= k <= Qt.Key.Key_9:
                idx = k - Qt.Key.Key_1
                if idx < len(ATTR_SUBTYPES):
                    self.sub_choice = idx
                    self._sub_enter()
            elif k == Qt.Key.Key_Up:
                self.sub_choice = (self.sub_choice - 1) % len(ATTR_SUBTYPES); self._sync_stage()
            elif k == Qt.Key.Key_Down:
                self.sub_choice = (self.sub_choice + 1) % len(ATTR_SUBTYPES); self._sync_stage()
            elif k == Qt.Key.Key_Backspace:
                self.stage = self.ST_TYPE
                self.cursor = 0
                self._sync_stage()
            elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._sub_enter()
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
        lay.setContentsMargins(26, 14, 26, 12)
        lay.setSpacing(10)

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

        # гайд: когда 1, когда 0
        guide = QFrame()
        guide.setStyleSheet(f"background:{PINK}; border-radius:10px;")
        gl = QVBoxLayout(guide)
        gl.setContentsMargins(14, 10, 14, 10)
        g1 = QLabel("Когда ставить 1: карточка — именно тот товар (или его вариант по цвету/памяти), "
                    "который искали. «ноутбук asus» + ASUS VivoBook — это 1.")
        g0 = QLabel("Когда ставить 0: другой тип товара, аксессуар вместо товара, другой бренд, "
                    "явно случайный клик. «айфон 15» + чехол для айфона — это 0. "
                    "Если карточки нет вовсе — 0 ставится автоматически.")
        for g in (g1, g0):
            g.setWordWrap(True)
            g.setStyleSheet(f"color:{MVDARK}; font-size:12px;")
        g1.setStyleSheet(f"color:{OK}; font-size:12px; font-weight:600;")
        g0.setStyleSheet(f"color:{MVRED}; font-size:12px; font-weight:600;")
        gl.addWidget(g1)
        gl.addWidget(g0)
        lay.addWidget(guide)

        card = QFrame()
        card.setStyleSheet(f"background:{CARDBG}; border-radius:12px;")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 14, 20, 14)
        self.lbl_query = QLabel()
        self.lbl_query.setStyleSheet(f"color:{MVRED}; font-size:16px; font-weight:700;")
        self.lbl_query.setWordWrap(True)
        self.lbl_sku = QLabel()
        self.lbl_sku.setStyleSheet(f"color:{MVDARK}; font-size:13px; font-weight:600;")
        self.lbl_sku.setWordWrap(True)
        self.lbl_meta = QLabel()
        self.lbl_meta.setStyleSheet(f"color:{MVGRAY}; font-size:11px;")
        cl.addWidget(QLabel("Запрос пользователя:"))
        cl.addWidget(self.lbl_query)
        cl.addSpacing(6)
        cl.addWidget(QLabel("Карточка товара:"))
        cl.addWidget(self.lbl_sku)
        cl.addWidget(self.lbl_meta)
        lay.addWidget(card)

        btns = QHBoxLayout()
        self.btn1 = QPushButton("1 — соответствует")
        self.btn0 = QPushButton("0 — не соответствует")
        self.btn1.setStyleSheet(
            f"QPushButton{{background:{OK}; color:white; border:none; border-radius:10px;"
            f"padding:14px; font-size:14px; font-weight:700;}} QPushButton:hover{{background:#166B3E;}}")
        self.btn0.setStyleSheet(
            f"QPushButton{{background:{MVRED}; color:white; border:none; border-radius:10px;"
            f"padding:14px; font-size:14px; font-weight:700;}} QPushButton:hover{{background:#C00500;}}")
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
        self.history.setFixedHeight(120)
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
        sku = (p.get("sku_name") or "").strip()
        self.lbl_query.setText(f"«{p['query']}»")
        if not sku or sku.lower() in ("nan", "none"):
            # пустая карточка -> автоматический 0
            self.lbl_sku.setText("— карточки нет (пустой клик) —")
            self.lbl_meta.setText("Авто-разметка: 0")
            self.lbl_pos.setText(f"Пара {self.pos + 1} из {len(self.pairs)} · размечено {len(self.records)}")
            self.progress.set_progress(len(self.records), len(self.pairs))
            self.mark(0, auto=True)
            return
        self.lbl_sku.setText(sku)
        price = f"{p['price']:,.0f} ₽".replace(",", " ") if p.get("price") else "—"
        self.lbl_meta.setText(f"Бренд: {p.get('brand','—')} · Цена: {price}")
        self.lbl_pos.setText(f"Пара {self.pos + 1} из {len(self.pairs)} · размечено {len(self.records)}")
        self.progress.set_progress(len(self.records), len(self.pairs))

    def mark(self, label: int, auto: bool = False):
        p = self.pairs[self.pos]
        rec = {
            "index": self.pos,
            "query": p["query"],
            "sku_name": p.get("sku_name"),
            "brand": p.get("brand"),
            "label": label,
            "auto": auto,
            "annotator": self.annotator,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self.records[self.pos] = rec
        lines = [json.dumps(self.records[k], ensure_ascii=False) for k in sorted(self.records)]
        self.out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        for i in sorted(self.records, reverse=True)[:300]:
            r = self.records[i]
            mark = f"[{r['label']}{' авто' if r.get('auto') else ''}]"
            it = QListWidgetItem(f"#{i+1} · {mark} {r['query']} ↔ {(r.get('sku_name') or '—')[:60]}")
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.history.addItem(it)

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


# ---------------------------------------------------------------- окно
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
        self.resize(1120, 800)
        self.setStyleSheet(f"QMainWindow{{background:white;}} QLabel{{color:{MVDARK};}}")

        out_dir = app_dir() / "labels"
        out_dir.mkdir(exist_ok=True)

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(f"background:{MVRED};")
        header.setFixedHeight(60)
        h = QHBoxLayout(header)
        h.setContentsMargins(24, 0, 24, 0)
        t = QLabel(f"М.Видео · Разметка · {display}")
        t.setStyleSheet("color:white; font-size:16px; font-weight:800;")
        h.addWidget(t)
        h.addStretch(1)
        self.btn_bio = QPushButton("BIO-разметка")
        self.btn_match = QPushButton("Соответствие 1/0")
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
    annotator = os.environ.get("MV_ANNOTATOR", annotator)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(annotator)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
