"""
hand_tracker.py
Real-time hand tracking using MediaPipe Hands.
Returns fingertip position, pen-up gesture (pinch), and clear gesture (fist).
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
import utils.config as C
from utils.logger import get_logger

log = get_logger(__name__, C.LOG_DIR, C.DEBUG_MODE)


class TrackingResult:
    """One frame's worth of tracking data."""
    __slots__ = ("fingertip", "pen_up", "clear", "landmarks", "frame_wh")

    def __init__(self, fingertip=None, pen_up=False, clear=False,
                 landmarks=None, frame_wh=(0, 0)):
        self.fingertip  = fingertip        # (x, y) px or None
        self.pen_up     = pen_up           # True → don't draw
        self.clear      = clear            # True → erase canvas
        self.landmarks  = landmarks        # mp NormalizedLandmarkList
        self.frame_wh   = frame_wh         # (w, h)

    @property
    def hand_detected(self): return self.fingertip is not None


class HandTracker:
    """Wraps MediaPipe Hands with smoothing and gesture detection."""

    def __init__(self):
        self._mp   = mp.solutions.hands
        self._draw = mp.solutions.drawing_utils
        self._sty  = mp.solutions.drawing_styles
        self._hands = self._mp.Hands(
            static_image_mode=False,
            max_num_hands=C.MAX_HANDS,
            model_complexity=C.MODEL_COMPLEXITY,
            min_detection_confidence=C.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=C.MIN_TRACKING_CONFIDENCE,
        )
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None
        self._fist_cnt = 0
        log.info("HandTracker ready.")

    # ── public ───────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> TrackingResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not res.multi_hand_landmarks:
            self._sx = self._sy = None
            self._fist_cnt = 0
            return TrackingResult(frame_wh=(w, h))

        lm = res.multi_hand_landmarks[0]

        # ── fingertip ────────────────────────────────────────────────────────
        tip   = lm.landmark[C.FINGERTIP_LANDMARK_ID]
        sx, sy = self._smooth(int(tip.x * w), int(tip.y * h))

        # ── pen-up: thumb-index pinch ────────────────────────────────────────
        thumb = lm.landmark[4]
        pinch = math.hypot(tip.x - thumb.x, tip.y - thumb.y)
        pen_up = (pinch < C.PINCH_THRESHOLD)

        # ── clear: closed fist ───────────────────────────────────────────────
        wrist = lm.landmark[0]
        avg_d = float(np.mean([
            math.hypot(lm.landmark[i].x - wrist.x,
                       lm.landmark[i].y - wrist.y)
            for i in [4, 8, 12, 16, 20]
        ]))
        fist = avg_d < C.FIST_THRESHOLD
        self._fist_cnt = self._fist_cnt + 1 if fist else 0
        clear = (self._fist_cnt >= C.CLEAR_HOLD_FRAMES)

        return TrackingResult(
            fingertip=(sx, sy), pen_up=pen_up, clear=clear,
            landmarks=lm, frame_wh=(w, h),
        )

    def draw_landmarks(self, frame: np.ndarray, result: TrackingResult):
        if result.landmarks is None:
            return
        self._draw.draw_landmarks(
            frame, result.landmarks, self._mp.HAND_CONNECTIONS,
            self._sty.get_default_hand_landmarks_style(),
            self._sty.get_default_hand_connections_style(),
        )

    def release(self):
        self._hands.close()
        log.info("HandTracker released.")

    # ── internal ─────────────────────────────────────────────────────────────

    def _smooth(self, x: int, y: int) -> Tuple[int, int]:
        if self._sx is None:
            self._sx, self._sy = float(x), float(y)
        else:
            a = C.SMOOTH_ALPHA
            self._sx = a * x + (1 - a) * self._sx
            self._sy = a * y + (1 - a) * self._sy
        return int(self._sx), int(self._sy)
