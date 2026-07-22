# -*- coding: utf-8 -*-
"""Скриншоты обоих приложений для презентации (без ручного запуска)."""
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

shots = []


def grab(widget, name):
    pm = widget.grab()
    pm.save(str(OUT / name))
    print("saved", name, pm.width(), "x", pm.height())


def main():
    import os
    os.environ["MV_ANNOTATOR"] = "nikita"

    # ---------- приложение разметки
    import labeling_app
    lw = labeling_app.MainWindow("nikita")
    lw.resize(1060, 780)
    lw.show()
    app.processEvents()

    def shot_labeling():
        # BIO: разметить пример для красоты скрина
        bio = lw.bio
        if bio.chips:
            # покажем пример разметки: первый токен B-CATEGORY и т.п.
            demo = ["B-CATEGORY", "B-BRAND", "B-MODEL", "B-ATTR", "I-ATTR"]
            for i, chip in enumerate(bio.chips[:5]):
                tag = demo[i % len(demo)]
                chip.tag, chip.cat = tag.split("-", 1)
                chip.refresh()
        app.processEvents()
        grab(lw, "labeling_bio.png")
        lw.switch(1)
        app.processEvents()
        grab(lw, "labeling_match.png")
        lw.close()
        start_mvp()

    QTimer.singleShot(600, shot_labeling)

    def start_mvp():
        import mvp_app
        mw = mvp_app.MainWindow()
        mw.resize(1120, 800)
        mw.show()
        app.processEvents()

        def splash_shot():
            grab(mw, "mvp_splash.png")
            mw.show_main()
            app.processEvents()
            mw.search.input.setText("ноутбук asus zenbook 16 гб серый")
            mw.search.run_search()
            app.processEvents()

            def search_shot():
                grab(mw, "mvp_search.png")
                # открыть JSON
                mw.search.json_view.setVisible(True)
                app.processEvents()
                grab(mw, "mvp_json.png")
                mw.close()
                app.quit()

            QTimer.singleShot(900, search_shot)

        QTimer.singleShot(2600, splash_shot)  # после пружинки

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
