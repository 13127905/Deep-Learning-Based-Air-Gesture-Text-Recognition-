"""
main.py — AI Air Writing Recognition System
============================================
FULL LIVE RECOGNITION PIPELINE

How it works:
  1. Webcam feeds 30 FPS
  2. MediaPipe detects hand, finds fingertip (index tip)
  3. Fingertip position draws ink on virtual canvas
  4. Pinch gesture (thumb touches index) = pen UP (stop drawing)
  5. After pen-up + short idle → CNN predicts the character
  6. Character appended to text; canvas cleared; repeat

Keyboard shortcuts:
  SPACE      — insert space
  ENTER      — speak text via TTS
  C          — clear canvas
  S          — save text to file
  V          — toggle voice
  BACKSPACE  — delete last char
  H          — help overlay
  D          — debug mode
  ESC / Q    — quit

Gestures:
  Index finger raised       — draw / write
  Thumb-index pinch         — lift pen (triggers prediction)
  Closed fist (hold ~1 s)   — clear canvas
"""
from __future__ import annotations
import datetime, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import utils.config        as C
from utils.logger          import get_logger
from utils.hand_tracker    import HandTracker
from utils.canvas_manager  import CanvasManager
from utils.voice_output    import VoiceOutput
from models.predictor      import CharacterPredictor, AutoCompleter
from gui.overlay           import HUD

log = get_logger("main", C.LOG_DIR, C.DEBUG_MODE)


# ── helpers ───────────────────────────────────────────────────────────────────

def save_text(text: str) -> str:
    os.makedirs(C.SAVED_TEXT_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(C.SAVED_TEXT_DIR, f"airwrite_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("Saved → %s", path)
    return path


# ── Application ───────────────────────────────────────────────────────────────

class AirWritingApp:
    """Main application class — runs the real-time loop."""

    def __init__(self):
        log.info("=" * 58)
        log.info("  AI AIR WRITING RECOGNITION  —  Starting …")
        log.info("=" * 58)

        # subsystems
        self.tracker   = HandTracker()
        self.canvas    = CanvasManager()
        self.predictor = CharacterPredictor()
        self.voice     = VoiceOutput()
        self.ac        = AutoCompleter()
        self.hud       = HUD()

        # state
        self.text          : str          = ""
        self.label         : str | None   = None
        self.conf          : float        = 0.0
        self.suggestions   : list         = []
        self.show_help     : bool         = False
        self._status_msg   : str          = ""
        self._status_exp   : float        = 0.0
        self._pred_shown   : bool         = False
        self._idle         : int          = 0

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(C.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          C.CAMERA_FPS)
        # Use MJPG codec for better FPS on most cameras
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not cap.isOpened():
            log.error(
                "Cannot open camera (index %d).\n"
                "  → Try changing CAMERA_INDEX in utils/config.py",
                C.CAMERA_INDEX,
            )
            return

        WIN = "AI Air Writing  [ H = help | ESC = quit ]"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, C.CAMERA_WIDTH, C.CAMERA_HEIGHT)

        if not self.predictor.model_loaded:
            self._status(
                "MODEL NOT FOUND — run:  python training/train.py", 99.0
            )
            log.warning(
                "\n\n"
                "  ╔══════════════════════════════════════════════╗\n"
                "  ║  No trained model found!                      ║\n"
                "  ║  Open a NEW terminal and run:                 ║\n"
                "  ║      python training/train.py                 ║\n"
                "  ║  Then restart main.py                         ║\n"
                "  ╚══════════════════════════════════════════════╝\n"
            )

        log.info("Webcam open. Press H for help, ESC/Q to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)      # mirror for natural use
                key   = cv2.waitKey(1) & 0xFF

                # ── Hand tracking ──────────────────────────────────────────
                result = self.tracker.process(frame)
                self.tracker.draw_landmarks(frame, result)

                # ── Canvas ─────────────────────────────────────────────────
                stroke_done, idle = self.canvas.update(
                    frame_pt    = result.fingertip,
                    frame_shape = frame.shape[:2],
                    pen_up      = result.pen_up,
                )
                self._idle = idle
                self.canvas.draw_fingertip(frame, result.fingertip)

                # ── Gesture: clear canvas ──────────────────────────────────
                if result.clear:
                    self._clear("Gesture: canvas cleared")

                # ── Auto-predict ───────────────────────────────────────────
                # Trigger when: stroke just finished OR idle long enough
                need_pred = stroke_done or (
                    idle >= C.IDLE_FRAMES_THRESH
                    and self.canvas.has_content()
                    and not self._pred_shown
                )
                if need_pred:
                    self._predict()
                    self._pred_shown = True

                if not self.canvas.has_content():
                    self._pred_shown = False

                # ── Keyboard ───────────────────────────────────────────────
                if key in (27, ord("q")):           # ESC / Q
                    break
                elif key == ord(" "):
                    self.text += " "
                    self._clear("Space added")
                elif key == 13:                     # ENTER
                    if self.text.strip():
                        self.voice.speak(self.text)
                        self._status("Speaking …")
                elif key == ord("c"):
                    self._clear("Canvas cleared")
                elif key == ord("s"):
                    if self.text.strip():
                        p = save_text(self.text)
                        self._status(f"Saved: {os.path.basename(p)}")
                    else:
                        self._status("Nothing to save")
                elif key == ord("v"):
                    on = self.voice.toggle()
                    self._status(f"Voice {'ON' if on else 'OFF'}")
                    C.VOICE_ENABLED = on
                elif key == ord("h"):
                    self.show_help = not self.show_help
                elif key == ord("d"):
                    C.DEBUG_MODE = not C.DEBUG_MODE
                    self._status(f"Debug {'ON' if C.DEBUG_MODE else 'OFF'}")
                elif key == 8:                      # BACKSPACE
                    self.text = self.text[:-1]

                # ── Auto-complete ──────────────────────────────────────────
                words = self.text.split()
                last  = words[-1] if words else ""
                self.suggestions = (
                    self.ac.suggest(last) if len(last) >= 2 else []
                )

                # ── HUD render ─────────────────────────────────────────────
                status = (self._status_msg
                          if time.time() < self._status_exp else "")
                self.hud.draw(
                    frame=frame,
                    prediction=self.label,
                    confidence=self.conf,
                    text_buffer=self.text,
                    fps=30,
                    voice_on=C.VOICE_ENABLED,
                    hand_detected=result.hand_detected,
                    pen_down=not result.pen_up,
                    canvas_preview=self.canvas.get_canvas_bgr()
    )
                if self.show_help:
                    self.hud.draw_help(frame)

                cv2.imshow(WIN, frame)

        finally:
            log.info("Shutting down …")
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.release()
            self.voice.shutdown()
            log.info("Bye!")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _predict(self):
        img = self.canvas.export_for_cnn()
        if img is None:
            return

        label, conf = self.predictor.predict(img)
        self.label  = label
        self.conf   = conf

        if label and conf >= C.CONF_THRESHOLD:
            self.text += label
            log.info("CHAR '%s' (%.2f) → '%s'", label, conf, self.text)
            self.voice.speak(label)
            self._clear()
        else:
            log.debug("Low-conf: label=%s conf=%.2f", label, conf)

    def _clear(self, msg: str = ""):
        self.canvas.clear()
        self.predictor.reset_smoother()
        self._pred_shown = False
        if msg:
            self._status(msg)

    def _status(self, msg: str, dur: float = 2.5):
        self._status_msg = msg
        self._status_exp = time.time() + dur


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="AI Air Writing Recognition System")
    p.add_argument("--camera",   type=int,  default=C.CAMERA_INDEX,
                   help="Webcam index (default 0)")
    p.add_argument("--no-voice", action="store_true",
                   help="Disable text-to-speech")
    p.add_argument("--debug",    action="store_true",
                   help="Verbose debug logging")
    a = p.parse_args()

    C.CAMERA_INDEX  = a.camera
    C.VOICE_ENABLED = not a.no_voice
    C.DEBUG_MODE    = a.debug

    AirWritingApp().run()


if __name__ == "__main__":
    main()
