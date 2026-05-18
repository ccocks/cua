"""
agent.py
────────
Main entry point for the Kimi K2.5 / NIM Computer Use Agent on macOS.

Environment variables (set by the GitHub Actions workflow):
  NVIDIA_API_KEY            – NIM API key (from Actions secret)
  CUA_TASK                  – Natural-language task description
  CUA_MAX_STEPS             – Maximum action steps (default: 25)
  CUA_SCREENSHOT_INTERVAL   – Seconds to wait after each non-screenshot
                              action before the next model call (default: 1.5)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from executor import ActionExecutor
from nim_client import NIMClient, SYSTEM_PROMPT
from screen_recorder import ScreenRecorder

# ── Logging ───────────────────────────────────────────────────────────────────
SESSION_DIR = Path("/tmp/cua_session")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

log_path = SESSION_DIR / "session.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path)),
    ],
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
TASK = os.environ.get("CUA_TASK", "").strip()
MAX_STEPS = int(os.environ.get("CUA_MAX_STEPS", "25"))
SCREENSHOT_INTERVAL = float(os.environ.get("CUA_SCREENSHOT_INTERVAL", "1.5"))


def run_agent(task: str) -> None:
    if not task:
        log.error("CUA_TASK is empty — nothing to do.")
        sys.exit(1)

    log.info("═" * 60)
    log.info("TASK  : %s", task)
    log.info("STEPS : up to %d", MAX_STEPS)
    log.info("═" * 60)

    # Determine output path: repo root (parent of cua/) / screenshot.mp4
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "screenshot.mp4"

    recorder = ScreenRecorder(output_path=output_path)
    recorder.start()

    try:
        _run_agent_loop(task)
    finally:
        recorder.stop()


def _run_agent_loop(task: str) -> None:

    client = NIMClient()
    executor = ActionExecutor(session_dir=SESSION_DIR)

    # ── Conversation history ──────────────────────────────────────────────────
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Your task:\n{task}\n\n"
                        "Start by calling `screenshot` to see the current screen."
                    ),
                }
            ],
        },
    ]

    step = 0
    done = False
    latest_screenshot: Path | None = None

    while step < MAX_STEPS and not done:
        step += 1
        log.info("── Step %d / %d ──────────────────────────", step, MAX_STEPS)

        # ── Call the model ────────────────────────────────────────────────────
        try:
            response = client.chat(messages, screenshot_path=latest_screenshot)
        except Exception as exc:
            log.error("NIM API error: %s", exc)
            sys.exit(2)

        assistant_msg = client.assistant_message_dict(response)
        messages.append(assistant_msg)

        if assistant_msg.get("content"):
            log.info("Model: %s", assistant_msg["content"])

        tool_calls = client.parse_tool_calls(response)

        if not tool_calls:
            log.warning("Model returned no tool calls — stopping.")
            break

        # ── Execute each tool call ────────────────────────────────────────────
        for tc_name, tc_args in tool_calls:
            log.info("Tool call: %s(%s)", tc_name, json.dumps(tc_args, ensure_ascii=False))

            if tc_name == "done":
                summary = tc_args.get("summary", "(no summary)")
                log.info("✓ DONE — %s", summary)
                done = True
                # Still add a tool-result so the conversation stays valid
                result_str = f"Task complete: {summary}"
            elif tc_name == "screenshot":
                # Actually capture and record the path for next model call
                latest_screenshot = executor.take_screenshot()
                result_str = f"Screenshot captured → {latest_screenshot}"
                log.info(result_str)
            else:
                result_str = executor.execute(
                    tc_name, tc_args, post_delay=SCREENSHOT_INTERVAL
                )
                log.info("Result: %s", result_str)
                # After a non-screenshot action, grab a fresh screenshot
                # so the model sees the updated screen on the next turn
                latest_screenshot = executor.take_screenshot()

            # Inject tool result back into history
            # Find the matching tool_call_id from the assistant message
            tc_id = _find_tool_call_id(assistant_msg, tc_name)
            messages.append(
                client.tool_result_message(tc_id, result_str)
            )

            if done:
                break

    if not done:
        log.warning("Max steps (%d) reached without 'done' signal.", MAX_STEPS)

    log.info("Session screenshots → %s", SESSION_DIR)
    log.info("Total steps executed: %d", step)


def _find_tool_call_id(assistant_msg: dict, tool_name: str) -> str:
    """Return the tool_call_id for a given tool name in the assistant message."""
    for tc in assistant_msg.get("tool_calls") or []:
        if tc["function"]["name"] == tool_name:
            return tc["id"]
    # Fallback (shouldn't happen with well-formed responses)
    return f"call_{tool_name}_{int(time.time())}"


if __name__ == "__main__":
    run_agent(TASK)
