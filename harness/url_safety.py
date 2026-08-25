"""Secret-safe endpoint URL parsing for logs and persisted metadata."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit


@dataclass(frozen=True)
class SafeEndpoint:
    """Non-secret endpoint components suitable for logs and reports."""

    hostname: str | None
    port: int | None
    scheme: str | None
    valid: bool


def _invalid_endpoint() -> SafeEndpoint:
    return SafeEndpoint(hostname=None, port=None, scheme=None, valid=False)


def _contains_unsafe_endpoint_text(value: str) -> bool:
    """Reject parser-normalized whitespace/control and backslash ambiguities."""

    return any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    )


def safe_endpoint(value: str | None) -> SafeEndpoint:
    """Parse an endpoint without retaining userinfo, path, query, or fragment.

    The parser is intentionally fail-closed for literal whitespace, control
    characters, backslashes, and unsafe host delimiters. ``urlsplit`` otherwise
    normalizes some of those inputs into plausible hostnames, which would make a
    malformed endpoint look valid in persisted metadata.
    """

    # Inspect the exact caller-provided text before any trimming. Stripping first
    # would silently accept leading/trailing whitespace even though this parser's
    # contract is to fail closed on every literal whitespace/control character.
    text = str(value or "")
    if not text:
        return _invalid_endpoint()
    if _contains_unsafe_endpoint_text(text) or "\\" in text:
        return _invalid_endpoint()

    candidate = text if "://" in text or text.startswith("//") else f"//{text}"
    try:
        parsed: SplitResult = urlsplit(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return _invalid_endpoint()

    if not hostname:
        return _invalid_endpoint()
    if (
        _contains_unsafe_endpoint_text(hostname)
        or any(character in hostname for character in "/\\?#@[]%")
    ):
        return _invalid_endpoint()

    normalized_hostname = hostname.lower().rstrip(".")
    if not normalized_hostname or ".." in normalized_hostname:
        return _invalid_endpoint()

    return SafeEndpoint(
        hostname=normalized_hostname,
        port=port,
        scheme=parsed.scheme.lower() or None,
        valid=True,
    )


def safe_endpoint_hostname(value: str | None) -> str | None:
    """Return only the normalized hostname from an endpoint URL."""

    return safe_endpoint(value).hostname
