"""PII-in-logs lint (docs/05 §4, §6): fail if a log call could carry transcript text or audio.

Flags calls like ``log.info(...)`` / ``logger.warning(...)`` / ``print(...)`` whose keyword
names or referenced variables/attributes look like content (text, transcript, audio, pcm,
utterance). The runtime scrubber in ``galecore.logging`` is the second line of defence; this
lint stops it being the only one.

Usage: python -m tools.lint_pii_logs [paths...]   (default: packages services tools ml)
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "msg", "log"}
LOGGER_NAMES = {"log", "logger", "_log", "_logger", "structlog", "logging"}
CONTENT_PARTS = ("text", "transcript", "audio", "pcm", "utterance")
ALLOWED_NAMES = {"text_len", "audio_seconds", "audio_duration_s"}  # sizes, not content
DEFAULT_PATHS = ["packages", "services", "tools", "ml"]


def _looks_like_content(name: str) -> bool:
    n = name.lower()
    return n not in ALLOWED_NAMES and any(p in n for p in CONTENT_PARTS)


def _is_log_call(node: ast.Call) -> bool:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id == "print"
    if isinstance(f, ast.Attribute) and f.attr in LOG_METHODS:
        target = f.value
        while isinstance(target, ast.Attribute | ast.Call):
            target = target.func if isinstance(target, ast.Call) else target.value
        return isinstance(target, ast.Name) and (
            target.id in LOGGER_NAMES or target.id.endswith(("log", "logger"))
        )
    return False


def _content_refs(node: ast.AST) -> Iterator[str]:
    # method names being called (path.read_text()) are not data; their receivers still are
    called = {id(sub.func) for sub in ast.walk(node) if isinstance(sub, ast.Call)}
    for sub in ast.walk(node):
        if id(sub) in called:
            continue
        if isinstance(sub, ast.Name) and _looks_like_content(sub.id):
            yield sub.id
        elif isinstance(sub, ast.Attribute) and _looks_like_content(sub.attr):
            yield sub.attr


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: syntax error"]
    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_log_call(node)):
            continue
        for kw in node.keywords:
            if kw.arg and _looks_like_content(kw.arg):
                problems.append(f"{path}:{node.lineno}: log keyword '{kw.arg}' may carry content")
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            for ref in _content_refs(arg):
                problems.append(f"{path}:{node.lineno}: log call references '{ref}'")
    return problems


def iter_py(paths: list[str]) -> Iterator[Path]:
    for p in map(Path, paths):
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for f in p.rglob("*.py"):
                if "tests" not in f.parts and ".venv" not in f.parts:
                    yield f


def main(argv: list[str] | None = None) -> int:
    paths = (argv if argv is not None else sys.argv[1:]) or DEFAULT_PATHS
    problems = [msg for f in iter_py(paths) for msg in check_file(f)]
    for msg in problems:
        print(msg)
    if problems:
        print(f"PII-in-logs lint: {len(problems)} problem(s)")
        return 1
    print("PII-in-logs lint: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
