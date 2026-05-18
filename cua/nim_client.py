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
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": (
                "Capture the current state of the screen. "
                "Call this whenever you need to see what's on screen "
                "before deciding the next action."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Single mouse click at screen coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate in pixels"},
                    "y": {"type": "integer", "description": "Y coordinate in pixels"},
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to press (default: left)",
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double-click at screen coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Right-click at screen coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type a string of text at the current focus location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to type. Use \\n for Enter.",
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
            "description": (
                "Press a key or key combination. "
                "Use pyautogui hotkey names joined by '+', e.g. 'command+s', "
                "'return', 'escape', 'tab', 'space', 'delete'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Key combo string, e.g. 'command+s'",
                    }
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the mouse wheel at a given position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate to scroll at"},
                    "y": {"type": "integer", "description": "Y coordinate to scroll at"},
                    "clicks": {
                        "type": "integer",
                        "description": "Number of scroll clicks. Positive = up, negative = down.",
                    },
                },
                "required": ["x", "y", "clicks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Click-drag from one screen position to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "integer", "description": "Start X"},
                    "y1": {"type": "integer", "description": "Start Y"},
                    "x2": {"type": "integer", "description": "End X"},
                    "y2": {"type": "integer", "description": "End Y"},
                    "duration": {
                        "type": "number",
                        "description": "Drag duration in seconds (default 0.5)",
                    },
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal that the task is complete. "
                "Call this as soon as the task goal is fully achieved."
            ),
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
You receive screenshots and must decide which UI actions to take to complete the user's task.

Rules:
1. Always call `screenshot` first to see the current state of the screen.
2. Plan actions step by step; take one action at a time.
3. Prefer clicking UI elements over keyboard shortcuts unless keyboard is clearer.
4. After any action that changes the screen (click, type, key), call `screenshot` again.
5. When the task is fully done, call `done` with a concise summary.
6. If you are stuck (same screen after 3 tries), explain why in `done` with status "stuck".
7. macOS key names: use "command" not "ctrl", "option" not "alt", "delete" for Backspace.
8. Do not hallucinate UI elements; only interact with what you can see on screen.
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
    def chat(
        self,
        messages: list[dict],
        screenshot_path: str | Path | None = None,
    ) -> Any:
        """
        Send messages to Kimi K2.5.

        If `screenshot_path` is provided the image is injected as the last
        content item of the final user message.
        """
        # Optionally inject screenshot into the last user message
        if screenshot_path:
            last = messages[-1]
            if last["role"] == "user":
                content = last["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                content = list(content) + [_image_message(_encode_image(screenshot_path))]
                messages = messages[:-1] + [{"role": "user", "content": content}]

        log.debug("→ NIM request | messages=%d", len(messages))

        response = self._client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1024,
        )

        log.debug("← NIM response | finish=%s", response.choices[0].finish_reason)
        return response

    def parse_tool_calls(self, response: Any) -> list[tuple[str, dict]]:
        """Extract [(tool_name, args_dict), ...] from a response."""
        choice = response.choices[0]
        calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append((tc.function.name, args))
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
