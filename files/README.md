# Kimi K2.5 CUA — GitHub Actions macOS Runner

A minimal **Computer Use Agent** pipeline that runs on a GitHub-hosted
`macos-latest` runner.  It uses **Kimi K2.5** via the **NVIDIA NIM** API
(OpenAI-compatible endpoint) to plan actions, and **pyautogui** to execute
them against the real macOS display.

```
┌─────────────────────────────────────────────────────┐
│  GitHub Actions  (macos-latest runner)              │
│                                                     │
│  ┌──────────┐   screenshot    ┌──────────────────┐  │
│  │ macOS    │ ◄──────────────  │  executor.py     │  │
│  │ display  │                 │  (pyautogui)     │  │
│  │          │ ──action──────► │                  │  │
│  └──────────┘                 └──────┬───────────┘  │
│                                      │ tool result   │
│                               ┌──────▼───────────┐  │
│                               │  agent.py        │  │
│                               │  (loop + history)│  │
│                               └──────┬───────────┘  │
│                                      │ messages      │
│                               ┌──────▼───────────┐  │
│                               │  nim_client.py   │  │
│                               │  Kimi K2.5 / NIM │  │
│                               └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Quick start

### 1. Add the secret

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|------|-------|
| `NVIDIA_API_KEY` | Your key from [build.nvidia.com](https://build.nvidia.com) |

### 2. Trigger the workflow

**Actions → Computer Use Agent (Kimi K2.5 via NIM) → Run workflow**

Fill in the `task` field with anything you'd ask a human operator to do on
the desktop, e.g.:

```
Open TextEdit, type "Hello from Kimi!", and save as hello.txt on the Desktop.
```

### 3. Inspect the artefacts

After the run, download **cua-session-screenshots** to replay what the agent
saw and did, step by step.

---

## File layout

```
.github/workflows/cua.yml   # Workflow definition
cua/
  agent.py                  # Main agent loop (orchestration)
  nim_client.py             # NIM API client + CUA tool schema
  executor.py               # pyautogui action dispatcher
README.md
```

---

## Tool set

| Tool | Description |
|------|-------------|
| `screenshot` | Capture current screen state |
| `click` | Single click at (x, y) |
| `double_click` | Double-click at (x, y) |
| `right_click` | Right-click at (x, y) |
| `type_text` | Type a string (supports `\n` for Enter) |
| `key` | Press a key combo e.g. `command+s`, `escape` |
| `scroll` | Scroll wheel at (x, y) |
| `drag` | Click-drag from one point to another |
| `done` | Signal task completion |

---

## Tuning knobs (workflow inputs)

| Input | Default | Notes |
|-------|---------|-------|
| `task` | _(required)_ | Natural-language task |
| `max_steps` | `25` | Hard cap on action steps |
| `screenshot_interval` | `1.5` | Seconds to wait after each action |

---

## Limitations & known issues

- **Coordinate accuracy** — The model receives full-resolution screenshots
  but must guess pixel coordinates.  For Retina (2×) displays pyautogui
  already converts logical → physical pixels internally.
- **Accessibility permissions** — GitHub-hosted `macos-latest` runners have
  Accessibility pre-granted.  Self-hosted runners need manual TCC approval:
  *System Settings → Privacy & Security → Accessibility → add your Terminal*.
- **NIM model slug** — The exact model ID for Kimi K2.5 on NIM may differ;
  check [build.nvidia.com/models](https://build.nvidia.com/models) and update
  `KIMI_MODEL` in `nim_client.py` if needed.
- **No parallel tool calls** — The loop processes one tool call per model
  turn to keep state consistent.  If the model returns multiple calls in a
  single turn, all are executed in order before the next screenshot.

---

## Local testing

```bash
export NVIDIA_API_KEY=nvapi-...
export CUA_TASK="Open Calculator and compute 99 * 42"
export CUA_MAX_STEPS=15
cd cua
pip install pyautogui pillow openai tenacity
python agent.py
```
