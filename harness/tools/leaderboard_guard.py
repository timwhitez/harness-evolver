"""Terminal-Bench leaderboard integrity guardrails for executable tools."""

from __future__ import annotations

import re


READLIKE_COMMAND = r"\b(?:cat|sed|grep|head|tail|less|more|find|ls|python3?\b[^;\n]*open)\b"

PROHIBITED_COMMAND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"https?://[^\s'\"<>]*tbench\.ai", re.IGNORECASE),
        "Terminal-Bench website access is not allowed during evaluation",
    ),
    (
        re.compile(
            r"https?://[^\s'\"<>]*github\.com[^\s'\"<>]*terminal[-_]?bench",
            re.IGNORECASE,
        ),
        "Terminal-Bench GitHub repository access is not allowed during evaluation",
    ),
    (
        re.compile(r"https?://[^\s'\"<>]*(?:^|[/_.-])solutions?(?:[/_.-]|$)", re.IGNORECASE),
        "External solution URL access is not allowed during evaluation",
    ),
    (
        re.compile(
            r"(?:https?://[^\s'\"<>]+.*\bAGENTS\.md\b|\bAGENTS\.md\b.*https?://)",
            re.IGNORECASE,
        ),
        "Fetching external instructions into AGENTS.md is not allowed during evaluation",
    ),
    (
        re.compile(r"\bimport\s+(harbor|terminal_bench|terminalbench)\b", re.IGNORECASE),
        "Harbor or Terminal-Bench internals may not be inspected during evaluation",
    ),
    (
        re.compile(
            rf"{READLIKE_COMMAND}[^;\n]*(?:^|\s)(?:/tests(?:/|\b)|/solutions?(?:/|\b)|/task\.toml\b)",
            re.IGNORECASE,
        ),
        "Hidden tests, official solutions, or benchmark definitions may not be inspected during evaluation",
    ),
    (
        re.compile(
            rf"{READLIKE_COMMAND}[^;\n]*(?:terminal-bench-tasks|terminal[-_]?bench[^\s'\"<>;]*"
            r"(?:/tests(?:/|\b)|/solutions?(?:/|\b)|/task\.toml\b))",
            re.IGNORECASE,
        ),
        "Hidden tests, official solutions, or benchmark definitions may not be inspected during evaluation",
    ),
    (
        re.compile(
            r"(site-packages|dist-packages)[^\s'\"<>]*(harbor|terminal[_-]?bench)",
            re.IGNORECASE,
        ),
        "Harbor or Terminal-Bench internals may not be inspected during evaluation",
    ),
)

PROHIBITED_WRITE_NAMES = {"AGENTS.md"}
PROHIBITED_READWRITE_PATHS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(^|/)terminal-bench-tasks(/|$)", re.IGNORECASE),
        "local Terminal-Bench task source paths may not be inspected during evaluation",
    ),
    (
        re.compile(r"(^|/)terminal[-_]?bench(/|$).*(^|/)(tests|solutions?|task\.toml)(/|$)", re.IGNORECASE),
        "official Terminal-Bench tests, solutions, or task definitions may not be inspected during evaluation",
    ),
    (
        re.compile(r"^/(tests|solutions?)(/|$)", re.IGNORECASE),
        "hidden verifier tests or official solutions may not be inspected during evaluation",
    ),
)


def prohibited_command_reason(command: str) -> str:
    """Return a leaderboard-integrity reason if a shell command is disallowed."""

    for pattern, reason in PROHIBITED_COMMAND_PATTERNS:
        if pattern.search(command):
            return reason
    return ""


def prohibited_path_reason(path: str, *, operation: str) -> str:
    """Return a leaderboard-integrity reason if a file path is disallowed."""

    normalized = path.replace("\\", "/")
    name = normalized.rstrip("/").rsplit("/", 1)[-1]
    if operation in {"write", "edit"} and name in PROHIBITED_WRITE_NAMES:
        return "Writing AGENTS.md during evaluation is not allowed"
    for pattern, reason in PROHIBITED_READWRITE_PATHS:
        if pattern.search(normalized):
            return reason
    return ""
