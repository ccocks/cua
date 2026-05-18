"""
screen_recorder.py
───────────────────
Records the entire macOS screen while the agent is running.
Uses `screencapture -v` (built-in macOS screen recording).

Usage:
    recorder = ScreenRecorder(output_path="screenshot.mp4")
    recorder.start()
    # ... agent runs ...
    recorder.stop()
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class ScreenRecorder:
    """Wraps macOS `screencapture -v` to record the screen."""

    def __init__(self, output_path: str | Path = "screenshot.mp4") -> None:
        self.output_path = Path(output_path)
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        """Begin screen recording as a background process."""
        log.info("Starting screen recording → %s", self.output_path)
        # -v = video mode, -f = full screen recording
        self._process = subprocess.Popen(
            ["screencapture", "-v", "-f", str(self.output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("Screen recorder started (pid=%s)", self._process.pid)

    def stop(self) -> None:
        """Stop screen recording and finalize the video file."""
        if self._process is None:
            log.warning("Screen recorder was never started")
            return

        if self._process.poll() is not None:
            log.warning("Screen recorder already exited (rc=%s)", self._process.returncode)
            return

        log.info("Stopping screen recording…")
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("Screen recorder did not stop gracefully, killing…")
            self._process.kill()
            self._process.wait(timeout=5)

        if self.output_path.exists():
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            log.info(
                "Screen recording saved → %s (%.1f MB)",
                self.output_path,
                size_mb,
            )
        else:
            log.error("Screen recording file was not created")
