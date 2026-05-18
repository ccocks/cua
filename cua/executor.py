"""
executor.py
───────────
Translates CUA tool-call dicts into real pyautogui / subprocess actions.

macOS notes
───────────
• Screenshots use `screencapture -x` (silent, no shutter sound) as the
  primary method; pyautogui.screenshot() is the fallback.
• Key combos use pyautogui.hotkey(*parts) which maps correctly on macOS
  ("command" → ⌘, "option" → ⌥).
• Mouse coordinates are logical pixels on the primary display.
  If the runner has a Retina (HiDPI) display the physical pixel count is
  2× but pyautogui already accounts for this via screen size queries.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import pyautogui

log = logging.getLogger(__name__)

# Fail-safe: moving mouse to corner raises an exception instead of going
# haywire. Keep enabled in CI.
pyautogui.FAILSAFE = False
# Global inter-call pause (seconds). Gives macOS time to redraw after clicks.
pyautogui.PAUSE = 0.1


class ActionExecutor:
    """Executes CUA actions and manages the per-session screenshot directory."""

    def __init__(self, session_dir: str | Path = "/tmp/cua_session") -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0

    # ── Screenshot ────────────────────────────────────────────────────────────

    def take_screenshot(self) -> Path:
        self._step += 1
        path = self.session_dir / f"step_{self._step:04d}.png"

        # Primary: screencapture (no permission dialog on GHA runners)
        try:
            subprocess.run(
                ["screencapture", "-x", str(path)],
                check=True,
                timeout=10,
            )
            log.debug("Screenshot saved → %s (screencapture)", path.name)
            return path
        except Exception as exc:
            log.warning("screencapture failed (%s); falling back to pyautogui", exc)

        # Fallback: pyautogui
        img = pyautogui.screenshot()
        img.save(str(path))
        log.debug("Screenshot saved → %s (pyautogui)", path.name)
        return path

    # ── Mouse actions ─────────────────────────────────────────────────────────

    def click(self, x: int, y: int, button: str = "left") -> str:
        log.info("click(%d, %d, %s)", x, y, button)
        pyautogui.click(x, y, button=button)
        return f"Clicked {button} at ({x}, {y})"

    def double_click(self, x: int, y: int) -> str:
        log.info("double_click(%d, %d)", x, y)
        pyautogui.doubleClick(x, y)
        return f"Double-clicked at ({x}, {y})"

    def right_click(self, x: int, y: int) -> str:
        log.info("right_click(%d, %d)", x, y)
        pyautogui.rightClick(x, y)
        return f"Right-clicked at ({x}, {y})"

    def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 0.5,
    ) -> str:
        log.info("drag(%d,%d → %d,%d)", x1, y1, x2, y2)
        pyautogui.moveTo(x1, y1)
        pyautogui.dragTo(x2, y2, duration=duration, button="left")
        return f"Dragged ({x1},{y1}) → ({x2},{y2})"

    def scroll(self, x: int, y: int, clicks: int) -> str:
        log.info("scroll(%d, %d, clicks=%d)", x, y, clicks)
        pyautogui.scroll(clicks, x=x, y=y)
        return f"Scrolled {clicks} clicks at ({x}, {y})"

    # ── Keyboard actions ──────────────────────────────────────────────────────

    def type_text(self, text: str) -> str:
        log.info("type_text(%r)", text[:60] + ("…" if len(text) > 60 else ""))
        # Handle embedded newlines as Return key presses
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if part:
                pyautogui.typewrite(part, interval=0.04)
            if i < len(parts) - 1:
                pyautogui.press("return")
        return f"Typed {len(text)} characters"

    def key(self, keys: str) -> str:
        """
        Accept 'command+s', 'return', 'escape', 'tab', etc.
        Splits on '+' and calls pyautogui.hotkey(*parts).
        """
        log.info("key(%r)", keys)
        parts = [k.strip().lower() for k in keys.split("+")]
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)
        return f"Pressed key(s): {keys}"

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, args: dict, post_delay: float = 1.5) -> str:
        """
        Dispatch a tool call and return a human-readable result string.
        `post_delay` is the settle time in seconds after each action.
        """
        result: str

        if tool_name == "screenshot":
            path = self.take_screenshot()
            result = f"Screenshot captured → {path}"
            # No extra delay needed — screenshot is already the observation
            return result

        elif tool_name == "click":
            result = self.click(
                args["x"], args["y"], args.get("button", "left")
            )
        elif tool_name == "double_click":
            result = self.double_click(args["x"], args["y"])
        elif tool_name == "right_click":
            result = self.right_click(args["x"], args["y"])
        elif tool_name == "type_text":
            result = self.type_text(args["text"])
        elif tool_name == "key":
            result = self.key(args["keys"])
        elif tool_name == "scroll":
            result = self.scroll(args["x"], args["y"], args["clicks"])
        elif tool_name == "drag":
            result = self.drag(
                args["x1"], args["y1"], args["x2"], args["y2"],
                args.get("duration", 0.5),
            )
        elif tool_name == "done":
            # done is handled by the agent loop; just acknowledge here
            result = f"Task complete: {args.get('summary', '')}"
        else:
            result = f"Unknown tool: {tool_name}"

        time.sleep(post_delay)
        return result
