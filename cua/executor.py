"""
executor.py
───────────
Translates CUA tool-call dicts into real pyautogui actions.

Coordinate handling
───────────────────
The model sometimes outputs normalized floats in [0, 1] instead of absolute
pixel coordinates.  _resolve() detects this and converts automatically, so
either format works.  The model is instructed to use absolute pixels in the
system prompt, but this is a safety net.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pyautogui

log = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class ActionExecutor:
    def __init__(self, session_dir: str | Path = "/tmp/cua_session") -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._sw, self._sh = pyautogui.size()
        log.info("Screen size: %dx%d", self._sw, self._sh)

    # ── Coordinate normalisation ──────────────────────────────────────────────

    def _resolve(self, x: float | int, y: float | int) -> tuple[int, int]:
        """
        Accept either absolute pixels (int or float > 1) or normalised [0,1]
        floats and return absolute pixel ints.

        A value is treated as normalised when it is a float strictly between
        0 and 1 (exclusive).  Integers and floats >= 1 are left as-is.
        """
        rx = int(x * self._sw) if isinstance(x, float) and 0.0 < x < 1.0 else int(x)
        ry = int(y * self._sh) if isinstance(y, float) and 0.0 < y < 1.0 else int(y)
        # Clamp to screen bounds
        rx = max(0, min(rx, self._sw - 1))
        ry = max(0, min(ry, self._sh - 1))
        return rx, ry

    # ── Screenshot ────────────────────────────────────────────────────────────

    def take_screenshot(self) -> Path:
        self._step += 1
        path = self.session_dir / f"step_{self._step:04d}.png"

        img = pyautogui.screenshot()
        img.save(str(path))
        log.debug("Screenshot → %s (pyautogui)", path.name)
        return path

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def click(self, x: float, y: float, button: str = "left") -> str:
        ax, ay = self._resolve(x, y)
        log.info("click(%s,%s → %d,%d, %s)", x, y, ax, ay, button)
        pyautogui.click(ax, ay, button=button)
        return f"Clicked {button} at ({ax}, {ay})"

    def double_click(self, x: float, y: float) -> str:
        ax, ay = self._resolve(x, y)
        log.info("double_click(%s,%s → %d,%d)", x, y, ax, ay)
        pyautogui.doubleClick(ax, ay)
        return f"Double-clicked at ({ax}, {ay})"

    def right_click(self, x: float, y: float) -> str:
        ax, ay = self._resolve(x, y)
        log.info("right_click(%s,%s → %d,%d)", x, y, ax, ay)
        pyautogui.rightClick(ax, ay)
        return f"Right-clicked at ({ax}, {ay})"

    def drag(self, x1: float, y1: float, x2: float, y2: float, duration: float = 0.5) -> str:
        ax1, ay1 = self._resolve(x1, y1)
        ax2, ay2 = self._resolve(x2, y2)
        log.info("drag(%s,%s → %s,%s  resolved %d,%d → %d,%d)", x1, y1, x2, y2, ax1, ay1, ax2, ay2)
        pyautogui.moveTo(ax1, ay1)
        pyautogui.dragTo(ax2, ay2, duration=duration, button="left")
        return f"Dragged ({ax1},{ay1}) → ({ax2},{ay2})"

    def scroll(self, x: float, y: float, clicks: int) -> str:
        ax, ay = self._resolve(x, y)
        log.info("scroll(%s,%s → %d,%d, clicks=%d)", x, y, ax, ay, clicks)
        pyautogui.scroll(clicks, x=ax, y=ay)
        return f"Scrolled {clicks} at ({ax}, {ay})"

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def type_text(self, text: str) -> str:
        log.info("type_text(%r)", text[:60] + ("…" if len(text) > 60 else ""))
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if part:
                pyautogui.typewrite(part, interval=0.04)
            if i < len(parts) - 1:
                pyautogui.press("return")
        return f"Typed {len(text)} characters"

    def key(self, keys: str) -> str:
        log.info("key(%r)", keys)
        parts = [k.strip().lower() for k in keys.split("+")]
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)
        return f"Pressed key(s): {keys}"

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, args: dict, post_delay: float = 1.5) -> str:
        if tool_name == "screenshot":
            path = self.take_screenshot()
            return f"Screenshot captured → {path}"
        elif tool_name == "click":
            result = self.click(args["x"], args["y"], args.get("button", "left"))
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
            result = f"Task complete: {args.get('summary', '')}"
        else:
            result = f"Unknown tool: {tool_name}"

        time.sleep(post_delay)
        return result