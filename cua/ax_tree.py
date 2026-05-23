"""
ax_tree.py
──────────
macOS Accessibility (AX) API — focused element info + screen UI tree.

Safe to import unconditionally: all PyObjC imports are lazy.
If PyObjC or AX permission is missing, public functions return empty results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class AXNode:
    role: str = ""
    title: str = ""
    description: str = ""
    value: str = ""
    enabled: bool = True
    focused: bool = False
    position: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = (0.0, 0.0)
    children: list[AXNode] = field(default_factory=list)


# ── Lazy import / permission state ─────────────────────────────────────────────

_ready: bool | None = None


def ensure_ax() -> bool:
    """Return True if the AX API is usable (PyObjC loaded + permission granted)."""
    global _ready, Quartz, AppKit, HIServices
    if _ready is not None:
        return _ready

    try:
        import Quartz as _Q
        import AppKit as _A
        import HIServices as _H

        Quartz = _Q
        AppKit = _A
        HIServices = _H
    except ImportError:
        log.warning("PyObjC not installed — AX features disabled (pip install pyobjc-framework-cocoa pyobjc-framework-quartz)")
        _ready = False
        return False

    try:
        trusted = HIServices.AXIsProcessTrusted()
        if not trusted:
            log.warning(
                "Accessibility permission not granted. "
                "Enable in: System Settings → Privacy & Security → Accessibility"
            )
        _ready = trusted
        return trusted
    except Exception as exc:
        log.debug("AX permission check failed: %s", exc)
        _ready = False
        return False


# ── Low-level attribute readers ────────────────────────────────────────────────

_SUCCESS = 0  # kAXErrorSuccess


def _s(el: Any, attr: str, maxlen: int = 200) -> str:
    err, val = Quartz.AXUIElementCopyAttributeValue(el, attr, None)
    if err != _SUCCESS or val is None:
        return ""
    if isinstance(val, str):
        return val[:maxlen]
    if hasattr(val, "description"):
        return val.description()[:maxlen]
    return str(val)[:maxlen]


def _b(el: Any, attr: str) -> bool:
    err, val = Quartz.AXUIElementCopyAttributeValue(el, attr, None)
    return bool(val) if err == _SUCCESS else False


def _pt(el: Any, attr: str) -> tuple[float, float]:
    err, val = Quartz.AXUIElementCopyAttributeValue(el, attr, None)
    if err == _SUCCESS and hasattr(val, "pointValue"):
        p = val.pointValue()
        if hasattr(p, "x"):
            return (float(p.x), float(p.y))
        try:
            return (float(p[0]), float(p[1]))
        except (TypeError, IndexError):
            pass
    return (0.0, 0.0)


def _sz(el: Any, attr: str) -> tuple[float, float]:
    err, val = Quartz.AXUIElementCopyAttributeValue(el, attr, None)
    if err == _SUCCESS and hasattr(val, "sizeValue"):
        s = val.sizeValue()
        if hasattr(s, "width"):
            return (float(s.width), float(s.height))
        try:
            return (float(s[0]), float(s[1]))
        except (TypeError, IndexError):
            pass
    return (0.0, 0.0)


# ── Tree builder ───────────────────────────────────────────────────────────────

_SKIP_ROLES = frozenset({
    "AXLayoutItem",
    "AXSeparator",
    "AXMenuButton",
    "AXMenuBarItem",
    "AXMenuItem",
})


def _build_tree(
    el: Any,
    depth: int = 0,
    max_depth: int = 4,
    max_children: int = 20,
) -> AXNode | None:
    if depth > max_depth:
        return None

    role = _s(el, "AXRole")
    if not role:
        return None

    node = AXNode(
        role=role,
        title=_s(el, "AXTitle"),
        description=_s(el, "AXDescription"),
        value=_s(el, "AXValue", 80),
        enabled=_b(el, "AXEnabled"),
        focused=_b(el, "AXFocused"),
        position=_pt(el, "AXPosition"),
        size=_sz(el, "AXSize"),
    )

    collapsed = role in _SKIP_ROLES and depth > 1

    if not collapsed:
        err, children = Quartz.AXUIElementCopyAttributeValue(el, "AXChildren", None)
        if err == _SUCCESS and isinstance(children, (list, tuple)):
            count = 0
            for child in children:
                if count >= max_children:
                    remaining = len(children) - count
                    node.children.append(AXNode(role=f"... ({remaining} more)"))
                    break
                child_node = _build_tree(child, depth + 1, max_depth, max_children)
                if child_node is not None:
                    node.children.append(child_node)
                    count += 1

    return node


def _format(node: AXNode, indent: int = 0) -> str:
    """Return compact single-line representation of *node* and its children."""
    tag = "  " * indent
    parts = [node.role]
    if node.title:
        parts.append(f'"{node.title}"')
    if node.description and node.description != node.title:
        parts.append(f"[{node.description}]")
    if node.value and node.value not in (node.title, ""):
        v = node.value[:60].replace("\n", "\\n")
        parts.append(f"‹{v}›")
    if node.focused:
        parts.append("← FOCUSED")
    line = f"{tag}{' '.join(parts)}"
    children_lines = [_format(c, indent + 1) for c in node.children]
    if children_lines:
        return line + "\n" + "\n".join(children_lines)
    return line


# ── Public API ─────────────────────────────────────────────────────────────────

def get_focused_element() -> dict[str, Any]:
    """Return a dict describing the currently focused AX element, or empty."""
    if not ensure_ax():
        return {}

    try:
        ws = AppKit.NSWorkspace.sharedWorkspace()
        front = ws.frontmostApplication()
        if front is None:
            return {}
        pid = front.processIdentifier()
        app_name = str(front.localizedName() or pid)

        app_el = Quartz.AXUIElementCreateApplication(pid)
        err, focused = Quartz.AXUIElementCopyAttributeValue(
            app_el, "AXFocusedUIElement", None
        )
        if err != _SUCCESS:
            return {"app": app_name}

        px, py = _pt(focused, "AXPosition")
        sx, sy = _sz(focused, "AXSize")
        return {
            "app": app_name,
            "role": _s(focused, "AXRole"),
            "role_desc": _s(focused, "AXRoleDescription"),
            "title": _s(focused, "AXTitle"),
            "description": _s(focused, "AXDescription"),
            "value": _s(focused, "AXValue", 200),
            "enabled": _b(focused, "AXEnabled"),
            "center": [px + sx / 2.0, py + sy / 2.0],
            "size": [sx, sy],
        }
    except Exception as exc:
        log.debug("AX get_focused_element failed: %s", exc)
        return {}


def get_ui_tree(max_depth: int = 4) -> str:
    """Return a compact text tree of the frontmost app's accessible UI."""
    if not ensure_ax():
        return ""

    try:
        ws = AppKit.NSWorkspace.sharedWorkspace()
        front = ws.frontmostApplication()
        if front is None:
            return ""
        pid = front.processIdentifier()
        app_name = str(front.localizedName() or pid)

        app_el = Quartz.AXUIElementCreateApplication(pid)
        err, windows = Quartz.AXUIElementCopyAttributeValue(app_el, "AXWindows", None)
        if err != _SUCCESS or not isinstance(windows, (list, tuple)):
            return f"{app_name} — no accessible windows"

        lines = [f"App: {app_name} (PID {pid})"]
        for win in windows:
            node = _build_tree(win, max_depth=max_depth)
            if node:
                lines.append(_format(node, indent=1))

        return "\n".join(lines)
    except Exception as exc:
        log.debug("AX get_ui_tree failed: %s", exc)
        return ""
