"""
canvas_manager.py
Virtual writing canvas — draws strokes, exports 28×28 for CNN.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import center_of_mass
import utils.config as C
from utils.logger import get_logger

log = get_logger(__name__, C.LOG_DIR, C.DEBUG_MODE)


class CanvasManager:
    def __init__(self):
        self._cw, self._ch = C.CANVAS_W, C.CANVAS_H
        self._canvas    = self._blank()
        self._strokes:  List[List[Tuple[int, int]]] = []
        self._current:  List[Tuple[int, int]] = []
        self._drawing   = False
        self._prev:     Optional[Tuple[int, int]] = None
        self._idle      = 0
        log.info("Canvas ready (%dx%d).", self._cw, self._ch)

    # ── public ───────────────────────────────────────────────────────────────

    def update(self, frame_pt: Optional[Tuple[int, int]],
               frame_shape: Tuple[int, int],
               pen_up: bool) -> Tuple[bool, int]:
        """
        Call every frame.
        Returns (stroke_done, idle_frames).
        stroke_done=True → trigger prediction.
        """
        if frame_pt is None or pen_up:
            if self._drawing and self._current:
                self._strokes.append(list(self._current))
                self._current = []
                self._drawing = False
                self._prev    = None
                self._idle   += 1
                return True, self._idle
            self._idle += 1
            self._prev  = None
            return False, self._idle

        cx, cy = self._to_canvas(frame_pt, frame_shape)

        # Ignore tiny jitter
        if self._prev:
            if abs(cx - self._prev[0]) < C.MIN_MOVE_PX and \
               abs(cy - self._prev[1]) < C.MIN_MOVE_PX:
                return False, 0

        if self._prev:
            cv2.line(self._canvas, self._prev, (cx, cy),
                     255, C.STROKE_THICKNESS, cv2.LINE_AA)

        self._current.append((cx, cy))
        self._prev    = (cx, cy)
        self._drawing = True
        self._idle    = 0
        return False, 0

    def clear(self):
        self._canvas  = self._blank()
        self._strokes = []
        self._current = []
        self._prev    = None
        self._drawing = False
        self._idle    = 0

    def has_content(self) -> bool:
        return bool(self._strokes) or bool(self._current)

    def export_for_cnn(self) -> Optional[np.ndarray]:
        """
        Returns (28,28,1) float32 in [0,1] ready for CNN.
        None if canvas empty.
        """
        if not self.has_content():
            return None
        img = self._canvas.copy()
        ys, xs = np.where(img > 0)
        if len(xs) == 0:
            return None

        pad = 22
        x1 = max(0, xs.min() - pad);  x2 = min(self._cw - 1, xs.max() + pad)
        y1 = max(0, ys.min() - pad);  y2 = min(self._ch - 1, ys.max() + pad)
        crop = img[y1:y2+1, x1:x2+1]

        # Make square with padding
        h, w  = crop.shape
        side  = max(h, w)
        sq    = np.zeros((side, side), dtype=np.uint8)
        dy, dx = (side - h) // 2, (side - w) // 2
        sq[dy:dy+h, dx:dx+w] = crop

        # Resize to 28×28
        resized = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_AREA)

        # Centre-of-mass alignment (MNIST standard)
        cy_cm, cx_cm = center_of_mass(resized)
        if not (np.isnan(cy_cm) or np.isnan(cx_cm)):
            M = np.float32([[1, 0, 14 - cx_cm], [0, 1, 14 - cy_cm]])
            resized = cv2.warpAffine(resized, M, (28, 28))

        out = resized.astype(np.float32) / 255.0
        return out[..., np.newaxis]   # (28,28,1)

    def get_canvas_bgr(self) -> np.ndarray:
        return cv2.cvtColor(self._canvas, cv2.COLOR_GRAY2BGR)

    def draw_fingertip(self, frame: np.ndarray,
                       pt: Optional[Tuple[int, int]]):
        if pt is None:
            return
        cv2.circle(frame, pt, 10, C.TRAIL_COLOR_BGR, -1)
        cv2.circle(frame, pt, 13, (255, 255, 255), 1, cv2.LINE_AA)

    # ── internal ─────────────────────────────────────────────────────────────

    def _blank(self) -> np.ndarray:
        return np.zeros((self._ch, self._cw), dtype=np.uint8)

    def _to_canvas(self, pt: Tuple[int, int],
                   shape: Tuple[int, int]) -> Tuple[int, int]:
        fh, fw = shape
        cx = int(pt[0] / fw * self._cw)
        cy = int(pt[1] / fh * self._ch)
        return (max(0, min(self._cw - 1, cx)),
                max(0, min(self._ch - 1, cy)))
