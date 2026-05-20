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
import subprocess
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
MAX_CONSECUTIVE_FAILURES = 3

VALID_TOOLS = frozenset({"screenshot", "click", "double_click", "type_text", "key", "done"})


def _tcc_prime(executor: ActionExecutor) -> None:
    """Dismiss the initial Screen Recording TCC dialog if present."""
    # Trigger the dialog with screencapture so the TCC grant can be saved
    tmp = SESSION_DIR / "_tcc_prime.png"
    try:
        subprocess.run(
            ["screencapture", "-D", "1", "-x", str(tmp)],
            check=True, timeout=5,
        )
    except Exception:
        pass
    tmp.unlink(missing_ok=True)

    time.sleep(1.5)
    log.info("Clicked left at (510, 354)")
    executor.click([510, 354])
    time.sleep(1)
    executor.take_screenshot()


def run_agent(task: str) -> None:
    if not task:
        log.error("CUA_TASK is empty — nothing to do.")
        sys.exit(1)

    log.info("═" * 60)
    log.info("TASK  : %s", task)
    log.info("STEPS : up to %d", MAX_STEPS)
    log.info("═" * 60)

    _run_agent_loop(task)


def _run_agent_loop(task: str) -> None:

    client = NIMClient()
    executor = ActionExecutor(session_dir=SESSION_DIR)

    # ── Screen recording ─────────────────────────────────────────────────────
    video_path = SESSION_DIR / "recording.mp4"
    recorder = ScreenRecorder(output_path=video_path)
    recorder.start()

    try:
        _tcc_prime(executor)

        # ── Conversation history ──────────────────────────────────────────────
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
        consecutive_failures = 0
        latest_screenshot: Path | None = None

        while step < MAX_STEPS and not done:
            step += 1
            log.info("── Step %d / %d ──────────────────────────", step, MAX_STEPS)

            # ── Call the model ────────────────────────────────────────────────
            if latest_screenshot:
                messages.append(client.screenshot_observation_message(latest_screenshot))
                latest_screenshot = None

            messages = client.trim_context(messages)

            try:
                response = client.chat(messages)
            except Exception as exc:
                log.error("NIM API error: %s", exc)
                sys.exit(2)

            assistant_msg = client.assistant_message_dict(response)
            messages.append(assistant_msg)

            if assistant_msg.get("content"):
                log.info("Model: %s", assistant_msg["content"])

            tool_calls = client.parse_tool_calls(response)

            # ── Retry: no tool calls or all-gibberish ─────────────────────────
            if not tool_calls:
                consecutive_failures += 1
                feedback = "You did not call any tool. Call screenshot to see the screen, or done(summary=...) when finished."
            elif all(name not in VALID_TOOLS for name, _ in tool_calls):
                bad = ", ".join(name for name, _ in tool_calls)
                consecutive_failures += 1
                feedback = f"Unknown tool(s): {bad}. Available tools: screenshot, click, double_click, type_text, key, done."
            else:
                consecutive_failures = 0

            if consecutive_failures:
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log.warning("Too many consecutive failures (%d) — stopping.", consecutive_failures)
                    break
                log.info("Retry #%d: %s", consecutive_failures, feedback)
                messages.append({"role": "user", "content": feedback})
                step -= 1
                continue

            # ── Execute each tool call ────────────────────────────────────────
            for tc_name, tc_args in tool_calls:
                log.info("Tool call: %s(%s)", tc_name, json.dumps(tc_args, ensure_ascii=False))

                if not isinstance(tc_args, dict):
                    log.warning("Tool args not a dict (%r) — resetting to {}", tc_args, {})
                    tc_args = {}

                if tc_name == "done":
                    summary = tc_args.get("summary", "(no summary)")
                    log.info("DONE — %s", summary)
                    done = True
                    result_str = f"Task complete: {summary}"
                elif tc_name == "screenshot":
                    latest_screenshot = executor.take_screenshot()
                    result_str = f"Screenshot captured -> {latest_screenshot}"
                    log.info(result_str)
                elif tc_name in ("click", "double_click", "type_text", "key"):
                    result_str = executor.execute(
                        tc_name, tc_args, post_delay=SCREENSHOT_INTERVAL
                    )
                    log.info("Result: %s", result_str)
                    latest_screenshot = executor.take_screenshot()
                else:
                    log.warning("Unknown tool call: %s — ignoring", tc_name)
                    result_str = f"Unknown tool: {tc_name}"

                tc_id = _find_tool_call_id(assistant_msg, tc_name)
                messages.append(client.tool_result_message(tc_id, result_str))

                if done:
                    break

        if not done:
            log.warning("Max steps (%d) reached without 'done' signal.", MAX_STEPS)

    finally:
        recorder.stop()

    log.info("Session screenshots → %s", SESSION_DIR)
    log.info("Total steps executed: %d", step)


def _find_tool_call_id(assistant_msg: dict, tool_name: str) -> str:
    """Return the tool_call_id for a given tool name in the assistant message."""
    for tc in assistant_msg.get("tool_calls") or []:
        if tc["function"]["name"] == tool_name:
            return tc["id"]
    return f"call_{tool_name}_{int(time.time())}"


if __name__ == "__main__":
    run_agent(TASK)
