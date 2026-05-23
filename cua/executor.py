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

    def _resolve(self, x: float, y: float) -> tuple[int, int]:
        rx = int(x * self._sw) if 0.0 < x < 1.0 else int(x)
        ry = int(y * self._sh) if 0.0 < y < 1.0 else int(y)
        rx = max(0, min(rx, self._sw - 1))
        ry = max(0, min(ry, self._sh - 1))
        return rx, ry

    def take_screenshot(self) -> Path:
        self._step += 1
        path = self.session_dir / f"step_{self._step:04d}.png"
        img = pyautogui.screenshot()
        img.save(str(path))
        log.debug("Screenshot -> %s", path.name)
        return path

    def click_image(self, image_path: str | Path, button: str = "left", confidence: float = 0.8) -> str:
        img = str(image_path)
        try:
            pos = pyautogui.locateCenterOnScreen(img, confidence=confidence)
        except pyautogui.ImageNotFoundException:
            pos = None
        if pos is None:
            msg = f"Image not found on screen: {img}"
            log.warning(msg)
            return msg
        pyautogui.click(pos.x, pos.y, button=button)
        msg = f"click {button} at ({pos.x}, {pos.y}) via {Path(img).name}"
        log.info(msg)
        return msg

    def click(self, position: list[float], button: str = "left") -> str:
        x, y = position[0], position[1]
        ax, ay = self._resolve(x, y)
        pyautogui.click(ax, ay, button=button)
        msg = f"click {button} at ({ax}, {ay})"
        log.info(msg)
        return msg

    def double_click(self, position: list[float]) -> str:
        x, y = position[0], position[1]
        ax, ay = self._resolve(x, y)
        pyautogui.doubleClick(ax, ay)
        msg = f"double_click at ({ax}, {ay})"
        log.info(msg)
        return msg

    def type_text(self, text: str) -> str:
        log.info("type_text(%r)", text[:80])
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
        return f"Pressed: {keys}"

    def execute(self, tool_name: str, args: dict, post_delay: float = 1.5) -> str:
        dispatch = {
            "click": lambda: self.click(
                args.get("position", [0, 0]), args.get("button", "left")
            ),
            "double_click": lambda: self.double_click(
                args.get("position", [0, 0])
            ),
            "type_text": lambda: self.type_text(args.get("text", "")),
            "key": lambda: self.key(args.get("keys", "")),
            "done": lambda: f"Task complete: {args.get('summary', '')}",
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            result = f"Unknown tool: {tool_name}"
            log.warning(result)
        else:
            result = fn()
        time.sleep(post_delay)
        return result
