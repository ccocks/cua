from __future__ import annotations

import logging
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)


class ScreenRecorder:
    """Records the macOS screen via ffmpeg + AVFoundation."""

    def __init__(self, output_path: str | Path = "recording.mp4") -> None:
        self.output_path = Path(output_path)
        self._process: subprocess.Popen | None = None
        self._start_time: float | None = None
        self._ffmpeg: str | None = None

    def start(self) -> bool:
        self._ffmpeg = shutil.which("ffmpeg")
        if not self._ffmpeg:
            log.warning("ffmpeg not found on PATH — screen recording disabled")
            return False

        screen_idx = self._detect_screen_index()
        if screen_idx is None:
            log.warning("No AVFoundation screen input found — recording disabled")
            return False

        cmd = [
            self._ffmpeg,
            "-f", "avfoundation",
            "-capture_cursor", "1",
            "-i", screen_idx,
            "-r", "5",
            "-vcodec", "libx264",
            "-crf", "28",
            "-preset", "ultrafast",
            "-y",
            str(self.output_path),
        ]

        log.info("Starting screen recording: %s", " ".join(cmd))
        self._start_time = time.time()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        if self._process.poll() is not None:
            log.warning("ffmpeg exited immediately (rc=%d) — recording disabled", self._process.returncode)
            self._process = None
            return False

        log.info("Screen recording active (pid=%s)", self._process.pid)
        return True

    def stop(self) -> None:
        if self._process is None:
            return
        elapsed = time.time() - (self._start_time or time.time())
        log.info("Stopping screen recording after %.1fs…", elapsed)

        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg did not stop on SIGTERM — killing")
            self._process.kill()
            self._process.wait(timeout=5)

        self._process = None

        if self.output_path.exists():
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            log.info("Recording saved → %s (%.1f MB, %.1fs)", self.output_path, size_mb, elapsed)
        else:
            log.error("Recording file was not created — ffmpeg may have failed silently")

    @staticmethod
    def _detect_screen_index() -> str | None:
        """Find the AVFoundation screen input index (e.g. '1')."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        text = result.stderr + result.stdout
        for line in text.splitlines():
            m = re.search(r"\[(\d+)\]\s+Capture screen", line)
            if m:
                return m.group(1)

        log.warning("No 'Capture screen' device found in ffmpeg device list")
        log.debug("ffmpeg avfoundation device list:\n%s", text)
        return None
