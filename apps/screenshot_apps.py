# -*- coding: utf-8 -*-
"""Скриншоты приложений для презентации (новый трёхэтапный мастер)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "apps"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "apps" / "labeling"))
sys.path.insert(0, str(ROOT / "apps" / "mvp"))
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

app = QApplication(sys.argv)
app.setStyle("Fusion")


def grab(widget, name):
    pm = widget.grab()
    pm.save(str(OUT / name))
    print("saved", name)


def main():
    import os
    os.environ["MV_ANNOTATOR"] = "nikita"

    import labeling_app
    lw = labeling_app.MainWindow("nikita")
    lw.resize(1120, 800)
    lw.show()
    app.processEvents()

    def shots():
        bio = lw.bio
        # этап 1: частично размечено
        demo_bio = ["B", "I", "O", "B", "I"]
        for i, blk in enumerate(bio.blocks):
            if i < len(bio.blocks) - 1:
                blk.bio = demo_bio[i % len(demo_bio)]
                blk.refresh()
        bio.cursor = min(len(bio.blocks) - 1, 2)
        bio._sync_stage()
        app.processEvents()
        grab(lw, "labeling_stage1_bio.png")

        # этап 2: типы (все блоки размечены, показываем выбор типа)
        demo_cat = ["CATEGORY", "ATTR", "BRAND", "MODEL", "ATTR"]
        for i, blk in enumerate(bio.blocks):
            blk.bio = demo_bio[i % len(demo_bio)] if i < len(demo_bio) else "O"
            if blk.bio in ("B", "I"):
                blk.cat = demo_cat[i % len(demo_cat)]
            blk.refresh()
        bio.stage = bio.ST_TYPE
        bio.cursor = 0
        bio._sync_stage()
        app.processEvents()
        grab(lw, "labeling_stage2_types.png")

        # этап 3: подтипы ATTR
        bio.attr_ids = [i for i, b in enumerate(bio.blocks) if b.cat == "ATTR"]
        if bio.attr_ids:
            bio.stage = bio.ST_SUBTYPE
            bio.cursor = 0
            bio.sub_choice = 0
            bio._sync_stage()
            app.processEvents()
            grab(lw, "labeling_stage3_subtypes.png")

        # match-режим
        lw.switch(1)
        app.processEvents()
        grab(lw, "labeling_match.png")
        lw.close()
        app.quit()

    QTimer.singleShot(700, shots)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
