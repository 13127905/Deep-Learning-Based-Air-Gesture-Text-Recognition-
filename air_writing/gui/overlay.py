# gui/overlay.py

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

# =========================
# COLORS
# =========================
_ACC = (255, 180, 0)
_CYN = (255, 255, 0)
_GRN = (0, 255, 120)
_RED = (0, 70, 255)
_ORG = (0, 165, 255)
_WHT = (255, 255, 255)
_DARK = (15, 15, 15)
_PANEL = (25, 25, 25)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

SHORTCUTS = [
    ("H", "Toggle Help"),
    ("C", "Clear Canvas"),
    ("S", "Speak Text"),
    ("P", "Pen Up / Down"),
    ("ESC", "Quit"),
]


# =========================
# BASIC HELPERS
# =========================
def _rect(img, x1, y1, x2, y2, color, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _t(img, txt, pos, color=_WHT, scale=0.6, thick=1):
    cv2.putText(
        img,
        txt,
        pos,
        _FONT,
        scale,
        color,
        thick,
        cv2.LINE_AA
    )


# =========================
# HUD CLASS
# =========================
class HUD:

    def __init__(self):
        self.show_help = False

    # =====================
    # MAIN DRAW FUNCTION
    # =====================
    def draw(
        self,
        frame: np.ndarray,
        prediction: str = "---",
        confidence: float = 0.0,
        text_buffer: str = "",
        fps: int = 0,
        voice_on: bool = True,
        hand_detected: bool = False,
        pen_down: bool = False,
        canvas_preview: np.ndarray | None = None
    ) -> np.ndarray:

        h, w = frame.shape[:2]

        # top bar
        self._top_bar(
            frame,
            w,
            voice_on,
            hand_detected,
            pen_down
        )

        # left prediction panel
        self._prediction_panel(
            frame,
            prediction,
            confidence,
            fps
        )

        # canvas preview
        if canvas_preview is not None:
            self._canvas_preview(
                frame,
                canvas_preview
            )

        # bottom text bar
        self._bottom_bar(
            frame,
            text_buffer
        )

        # help screen
        if self.show_help:
            self.draw_help(frame)

        return frame

    # =====================
    # HELP SCREEN
    # =====================
    def draw_help(self, frame: np.ndarray):

        h, w = frame.shape[:2]

        x1, y1 = w // 4, h // 6
        x2, y2 = 3 * w // 4, 5 * h // 6

        _rect(frame, x1, y1, x2, y2, _DARK, 0.95)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            _ACC,
            2
        )

        _t(
            frame,
            "KEYBOARD SHORTCUTS",
            (x1 + 25, y1 + 40),
            _ACC,
            0.8,
            2
        )

        for i, (k, d) in enumerate(SHORTCUTS):

            yy = y1 + 90 + i * 45

            _t(
                frame,
                f"{k:<10}",
                (x1 + 35, yy),
                _CYN,
                0.65,
                2
            )

            _t(
                frame,
                d,
                (x1 + 180, yy),
                _WHT,
                0.65,
                1
            )

    # =====================
    # TOP BAR
    # =====================
    def _top_bar(self, f, w, vo, hand, pu):

        _rect(f, 0, 0, w, 70, _DARK, 0.92)

        cv2.line(
            f,
            (0, 68),
            (w, 68),
            _ACC,
            2
        )

        _t(
            f,
            "AI AIR WRITING RECOGNITION  |  REAL-TIME DEEP LEARNING SYSTEM",
            (16, 42),
            _ACC,
            0.82,
            2
        )

        icons = [
            (
                f"VOICE {'ON' if vo else 'OFF'}",
                _GRN if vo else _RED
            ),
            (
                f"HAND {'OK' if hand else '--'}",
                _GRN if hand else _ORG
            ),
            (
                f"PEN {'DOWN' if pu else 'UP'}",
                _ORG if pu else _CYN
            )
        ]

        x = w - 20

        for lbl, col in reversed(icons):

            (tw, _), _ = cv2.getTextSize(
                lbl,
                _FONT,
                0.58,
                2
            )

            x -= tw + 24

            _t(
                f,
                lbl,
                (x, 42),
                col,
                0.58,
                2
            )

    # =====================
    # PREDICTION PANEL
    # =====================
    def _prediction_panel(
        self,
        f,
        pred,
        conf,
        fps
    ):

        x1, y1 = 15, 95
        x2, y2 = 220, 330

        _rect(f, x1, y1, x2, y2, _PANEL, 0.92)

        cv2.rectangle(
            f,
            (x1, y1),
            (x2, y2),
            _ACC,
            2
        )

        _t(
            f,
            "LIVE PREDICTION",
            (x1 + 15, y1 + 35),
            _ACC,
            0.72,
            2
        )

        _t(
            f,
            pred,
            (x1 + 40, y1 + 120),
            _WHT,
            2.2,
            4
        )

        bar_w = 170
        fill = int(bar_w * conf)

        cv2.rectangle(
            f,
            (x1 + 18, y1 + 175),
            (x1 + 18 + bar_w, y1 + 200),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            f,
            (x1 + 18, y1 + 175),
            (x1 + 18 + fill, y1 + 200),
            _GRN,
            -1
        )

        _t(
            f,
            f"{int(conf * 100)} %",
            (x1 + 18, y1 + 235),
            _WHT,
            0.72,
            2
        )

        _t(
            f,
            f"FPS : {fps}",
            (x1 + 18, y1 + 275),
            _CYN,
            0.65,
            2
        )

    # =====================
    # CANVAS PREVIEW
    # =====================
    def _canvas_preview(
        self,
        frame,
        canvas
    ):

        h, w = frame.shape[:2]

        preview = cv2.resize(
            canvas,
            (240, 240)
        )

        if len(preview.shape) == 2:
            preview = cv2.cvtColor(
                preview,
                cv2.COLOR_GRAY2BGR
            )

        x1 = w - 270
        y1 = 100

        _rect(
            frame,
            x1 - 10,
            y1 - 10,
            x1 + 250,
            y1 + 250,
            _PANEL,
            0.92
        )

        frame[
            y1:y1 + 240,
            x1:x1 + 240
        ] = preview

        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + 240, y1 + 240),
            _ACC,
            2
        )

        _t(
            frame,
            "Canvas Preview",
            (x1 + 25, y1 - 18),
            _CYN,
            0.65,
            2
        )

    # =====================
    # BOTTOM BAR
    # =====================
    def _bottom_bar(
        self,
        f,
        txt
    ):

        h, w = f.shape[:2]

        _rect(
            f,
            0,
            h - 90,
            w,
            h,
            _DARK,
            0.94
        )

        cv2.line(
            f,
            (0, h - 92),
            (w, h - 92),
            _ACC,
            2
        )

        _t(
            f,
            "OUTPUT TEXT",
            (20, h - 52),
            _ACC,
            0.72,
            2
        )

        shown = txt[-70:]

        _t(
            f,
            shown if shown else "[ Start writing in air ]",
            (20, h - 18),
            _WHT,
            0.85,
            2
        )