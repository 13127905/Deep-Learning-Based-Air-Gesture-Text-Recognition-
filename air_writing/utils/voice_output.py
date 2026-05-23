"""voice_output.py — Non-blocking TTS via background daemon thread."""
from __future__ import annotations
import queue, threading
from typing import Optional
import utils.config as C
from utils.logger import get_logger

log = get_logger(__name__, C.LOG_DIR, C.DEBUG_MODE)


class VoiceOutput:
    def __init__(self):
        self._enabled = C.VOICE_ENABLED
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        self._engine  = None
        self._thread: Optional[threading.Thread] = None
        if self._enabled:
            self._start()

    def speak(self, text: str):
        if self._enabled and text.strip():
            self._q.put(text)

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        log.info("Voice %s.", "ON" if self._enabled else "OFF")
        return self._enabled

    def shutdown(self):
        if self._thread and self._thread.is_alive():
            self._q.put(None)
            self._thread.join(timeout=3)

    def _start(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   C.VOICE_RATE)
            self._engine.setProperty("volume", C.VOICE_VOLUME)
            voices = self._engine.getProperty("voices")
            if voices:
                self._engine.setProperty("voice", voices[0].id)
            self._thread = threading.Thread(
                target=self._worker, daemon=True, name="TTS")
            self._thread.start()
            log.info("VoiceOutput ready.")
        except Exception as e:
            log.warning("TTS unavailable: %s", e)
            self._enabled = False

    def _worker(self):
        while True:
            text = self._q.get()
            if text is None:
                break
            if self._enabled and self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    log.error("TTS error: %s", e)
