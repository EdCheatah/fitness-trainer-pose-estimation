"""
Text-to-Speech manager for spoken exercise feedback.

Speaks form-feedback messages through the PC speakers while an exercise is
running. Designed to be driven from the real-time frame loop without ever
blocking it.

Key design points:
- pyttsx3's runAndWait() is blocking and unreliable when called from a thread
  other than the one that created the engine. So a single dedicated worker
  thread owns the engine and is the only place that ever speaks. Everyone else
  just enqueues messages.
- Debounce: the same message is not repeated more often than COOLDOWN_SECS, and
  only one message is spoken at a time (no pile-up).
- Graceful fallback: if pyttsx3 is not installed, the manager becomes a no-op
  instead of crashing the app.
"""

import os
import time
import queue
import logging
import threading
from typing import Dict, List

logger = logging.getLogger(__name__)

# Per-message cooldown: don't repeat the same message more often than this.
COOLDOWN_SECS = 4.0

# Speak the most important active message first when several fire at once.
SEVERITY_PRIORITY = {"error": 3, "warning": 2, "info": 1}


def _tts_enabled() -> bool:
    """TTS is on by default; set FITNESS_TTS=0 to silence without code changes."""
    return os.environ.get("FITNESS_TTS", "1").strip().lower() not in ("0", "false", "no")


class TTSManager:
    """Speaks exercise feedback through the PC speakers, off the frame loop."""

    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=4)
        self._last_spoken: Dict[str, float] = {}
        self._busy = False
        self._enabled = _tts_enabled()
        self._available = False

        if not self._enabled:
            logger.info("TTS disabled via FITNESS_TTS env var")
            return

        # Probe the dependency up front; if missing, stay a no-op.
        try:
            import pyttsx3  # noqa: F401

            self._available = True
        except ImportError:
            logger.warning("pyttsx3 not installed - spoken feedback disabled")
            return

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        logger.info("TTS manager started")

    def _run(self):
        """Worker thread: owns the engine, speaks queued messages one by one."""
        import pyttsx3

        try:
            engine = pyttsx3.init()
        except Exception as e:  # pragma: no cover - driver/COM init failure
            logger.warning(f"Failed to initialize TTS engine: {e}")
            self._available = False
            return

        while True:
            message = self._queue.get()
            if message is None:  # shutdown sentinel
                break
            self._busy = True
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                logger.warning(f"TTS playback error: {e}")
            finally:
                self._busy = False

    def speak_feedback(self, feedback_list: List[Dict]):
        """
        Enqueue the most relevant active feedback message, if any.

        Non-blocking and safe to call every frame. Picks one message by severity
        priority and respects the per-message cooldown.

        Args:
            feedback_list: list of dicts with "message" and "severity" keys.
        """
        if not self._available or self._busy or not feedback_list:
            return

        # Highest-severity message wins when several are active this frame.
        best = max(
            feedback_list,
            key=lambda fb: SEVERITY_PRIORITY.get(fb.get("severity", "warning"), 2),
        )
        message = best.get("message")
        if not message:
            return

        now = time.time()
        if now - self._last_spoken.get(message, 0.0) < COOLDOWN_SECS:
            return

        try:
            self._queue.put_nowait(message)
            self._last_spoken[message] = now
        except queue.Full:
            pass  # backlog already; drop this one

    def reset(self):
        """Flush pending speech and cooldowns (call when a workout stops/ends)."""
        self._last_spoken.clear()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
