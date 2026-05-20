"""
nim_client.py
─────────────
Thin wrapper around the NVIDIA NIM OpenAI-compatible endpoint for Kimi K2.5.
Handles:
  • Base64 screenshot injection into the vision message
  • Tool-call schema definition (the CUA action set)
  • Retry logic with exponential back-off via tenacity
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

# ── NIM endpoint ──────────────────────────────────────────────────────────────
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Check https://build.nvidia.com/models for the exact slug if this changes.
KIMI_MODEL = "moonshotai/kimi-k2.6"


# ── CUA tool definitions ──────────────────────────────────────────────────────
# NOTE: keep this list SHORT. Every extra tool adds cognitive load on the model
# and increases the chance of malformed JSON arguments.
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": "Show the current screen. Call this first, and after every action that changes the screen.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click at a position on screen. Supply position as [x, y] — pixel coords (e.g. [500, 300]) or normalized 0-1 decimals (e.g. [0.5, 0.33]).",
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[x, y] — pixel coords or normalized 0-1 decimals",
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button (default: left)",
                    },
                },
                "required": ["position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double-click at a position. Same format as click.",
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[x, y] — pixel coords or normalized 0-1 decimals",
                    },
                },
                "required": ["position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type characters at the current cursor location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to type. Use \\n for Enter/Return.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "key",
            "description": "Press a key or keyboard shortcut. Join modifier keys with '+'. Valid key names: command, option, shift, control, return, escape, tab, space, delete, up, down, left, right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Key(s) to press, e.g. 'command+s' to save, 'return' to press Enter, 'escape' to press Esc.",
                    }
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call this when the task is finished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-sentence summary of what was accomplished.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a computer-use agent controlling a real macOS desktop.
You see screenshots and decide which tool to call next.

Available tools:
- screenshot: see the screen (call this first, and after every action)
- click(position, button="left"): click at [x, y]
- double_click(position): double-click at [x, y]
- type_text(text): type characters
- key(keys): press a key or keyboard shortcut
- done(summary): signal task complete

Rules:
1. Start every task with screenshot.
2. After any action that changes the screen, call screenshot again.
3. Position is an array of two numbers: [x, y]. Values can be pixel coords (e.g. [500, 300]) OR decimals 0-1 (e.g. [0.5, 0.33]).
4. macOS: use "command" not "ctrl", "option" not "alt".
5. When finished, call done(summary="what was done").
6. If stuck (same screen >3 attempts), call done(summary="Stuck: reason").
"""


def _encode_image(path: str | Path) -> str:
    """Return base64-encoded PNG."""
    return base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")


def _image_message(b64: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


class NIMClient:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(
            base_url=NIM_BASE_URL,
            api_key=api_key or os.environ["NVIDIA_API_KEY"],
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def chat(self, messages: list[dict]) -> Any:
        """Send messages to Kimi K2.5 and return the raw response."""
        log.debug("→ NIM request | messages=%d", len(messages))

        response = self._client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=8192,  # 1024 caused degeneration on complex reasoning
        )

        finish = response.choices[0].finish_reason
        log.debug("← NIM response | finish=%s", finish)
        if finish == "length":
            log.warning("Response hit max_tokens — consider raising it further.")
        return response

    # ── Layered argument parsing ──────────────────────────────────────────────
    # Kimi K2.5 sometimes generates structurally broken JSON in tool-call
    # arguments.  We try up to three strategies in order:
    #   1.  Standard json.loads after basic string repairs.
    #   2.  Regex extraction for known malformation patterns.
    #   3.  Post-parse validation that catches things like numeric-key swaps.
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _repair_json(raw: str) -> str:
        """Fix common JSON generation issues from LLMs."""
        s = raw.strip()
        if "```" in s:
            for marker in ("```json\n", "```\n", "```"):
                if marker in s:
                    s = s.split(marker, 1)[-1]
            s = s.rsplit("```", 1)[0].strip()
        s = s.replace(",\n}", "\n}").replace(",}", "}").replace(",\n]", "\n]").replace(",]", "]")
        return s

    @staticmethod
    def _try_parse(s: str) -> dict | None:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _rx(pattern: str, text: str) -> str | None:
        m = re.search(pattern, text)
        return m.group(1) if m else None

    @classmethod
    def _regex_extract(cls, tool_name: str, raw: str) -> dict | None:
        """Fallback: regex extraction when JSON parsing fails."""
        args: dict = {}

        if tool_name in ("click", "double_click"):
            # Try position array format: "position": [x, y]
            pos = cls._rx(r'"position"\s*:\s*\[\s*([0-9.-]+)\s*,\s*([0-9.-]+)', raw)
            if pos:
                parts = pos.split(",")
                args["position"] = [float(parts[0]), float(parts[1])]
            else:
                # Fall back to old-style "x"/"y" keys
                x_raw = cls._rx(r'"x"\s*:\s*([0-9.]+)', raw)
                y_raw = cls._rx(r'"y"\s*:\s*([0-9.]+)', raw)
                if not y_raw:
                    dangling = cls._rx(r'"([0-9]+\.[0-9]+)"', raw)
                    if dangling:
                        y_raw = dangling
                if x_raw and y_raw:
                    args["position"] = [float(x_raw), float(y_raw)]
            # Extract button if present
            btn = cls._rx(r'"button"\s*:\s*"([^"]+)"', raw)
            if btn:
                args["button"] = btn

        elif tool_name == "type_text":
            txt = cls._rx(r'"text"\s*:\s*"([^"]*)"', raw)
            if txt:
                args["text"] = txt

        elif tool_name == "key":
            keys = cls._rx(r'"keys"\s*:\s*"([^"]*)"', raw)
            if keys:
                args["keys"] = keys

        elif tool_name == "done":
            summary = cls._rx(r'"summary"\s*:\s*"([^"]*)"', raw)
            if summary:
                args["summary"] = summary

        return args if args else None

    @staticmethod
    def _validate_and_repair(tool_name: str, args: dict) -> dict:
        """
        Post-parse validation that handles the model's various coordinate errors.

        1. Old-style "x"/"y" keys → convert to "position": [x, y]
        2. Numeric-key swap: {"x": 0.274, "0.332": "TextEdit"} → position
        3. Missing/empty position → fill with [0, 0]
        """
        if tool_name in ("click", "double_click"):
            # Legacy format: has "x" and "y" but no "position"
            if "position" not in args and "x" in args and "y" in args:
                args["position"] = [float(args.pop("x")), float(args.pop("y"))]

            # Numeric-key swap: keys that look like floats
            numeric_keys = [k for k in args if re.match(r'^[0-9]+\.[0-9]+$', str(k))]
            for nk in numeric_keys:
                if "position" in args:
                    break
                if "x" in args:
                    # {"x": 0.274, "0.332": "y_val"} — y is in the key name
                    x_val = float(args.pop("x"))
                    y_val = float(nk)
                    args["position"] = [x_val, y_val]
                    args.pop(nk, None)

            # Ensure position exists
            if "position" not in args or not isinstance(args["position"], list) or len(args["position"]) < 2:
                args["position"] = [0, 0]
        return args

    def parse_tool_calls(self, response: Any) -> list[tuple[str, dict]]:
        """Extract [(tool_name, args_dict), ...] from a response."""
        choice = response.choices[0]
        calls = []
        if not choice.message.tool_calls:
            return calls

        for tc in choice.message.tool_calls:
            name = tc.function.name
            raw = tc.function.arguments or "{}"

            # Layer 1 — standard JSON
            args = self._try_parse(self._repair_json(raw))

            # Layer 2 — regex fallback
            if args is None:
                args = self._regex_extract(name, raw)
                if args is not None:
                    log.info("Regex-repaired args for %s: %s", name, args)

            # Layer 3 — empty dict as last resort
            if args is None:
                args = {}
                log.warning("All parsing failed for %s: %r", name, raw)

            # Post-parse validation
            args = self._validate_and_repair(name, args)

            calls.append((name, args))
        return calls

    def assistant_message_dict(self, response: Any) -> dict:
        """Convert response to a dict suitable for appending to message history."""
        msg = response.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ] or None,
        }

    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def trim_context(self, messages: list[dict], max_images: int = 5) -> list[dict]:
        """Trim message history to keep at most *max_images* screenshot observations.

        Preserves the system prompt and initial user task, then keeps only the
        most recent conversation groups that contain screenshots.
        """
        if len(messages) < 3:
            return messages

        screenshot_indices = [
            i
            for i, msg in enumerate(messages)
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and any(
                    isinstance(c, dict) and c.get("type") == "image_url"
                    for c in msg["content"]
                )
            )
        ]

        if len(screenshot_indices) > max_images:
            keep_from = screenshot_indices[-max_images]
            return messages[:2] + messages[keep_from:]

        return messages

    def screenshot_observation_message(self, screenshot_path: str | Path) -> dict:
        """
        Build a user-role message that delivers a screenshot to the model.

        This MUST be a user message (not a tool result) because the OpenAI
        spec requires tool result messages to contain only text.  We append
        this after all tool results for a given step so the model always
        sees the current screen before its next action.
        """
        b64 = _encode_image(screenshot_path)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current screen state after last action:"},
                _image_message(b64),
            ],
        }