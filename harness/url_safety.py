"""Secret-safe endpoint URL parsing for logs and persisted metadata."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import AddressValueError, IPv6Address
import re
from urllib.parse import SplitResult, urlsplit


_DNS_LABEL = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z",
    flags=re.ASCII,
)


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


def _authority_has_empty_port(parsed: SplitResult) -> bool:
    """Detect an explicit colon for which ``urlsplit`` reports no port.

    ``SplitResult.port`` correctly raises for non-numeric and out-of-range ports,
    but an authority ending in ``:`` (including ``[::1]:``) is normalized to
    ``port=None``. Treat that spelling as malformed instead of silently making it
    indistinguishable from an endpoint that never supplied a port.
    """

    authority = parsed.netloc.rsplit("@", 1)[-1]
    return authority.endswith(":")


def _normalise_hostname(hostname: str) -> str | None:
    """Return a log-safe DNS/IDNA or IPv6 hostname, otherwise fail closed.

    RFC 3986 permits many ``reg-name`` characters that are not hostnames. Python
    consequently accepts authorities such as ``example.com;token=abc`` and
    exposes the complete string through ``SplitResult.hostname``. Persisting that
    value as host metadata would retain credential-like material. This helper
    deliberately supports only DNS/IDNA labels and canonical IPv6 literals.
    """

    raw = hostname.lower()
    if ":" in raw:
        try:
            return IPv6Address(raw).compressed.lower()
        except AddressValueError:
            return None

    if raw.startswith(".") or ".." in raw:
        return None
    raw = raw.rstrip(".")
    if not raw:
        return None

    try:
        ascii_hostname = raw.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if len(ascii_hostname) > 253:
        return None

    labels = ascii_hostname.split(".")
    if any(
        not label
        or len(label) > 63
        or _DNS_LABEL.fullmatch(label) is None
        for label in labels
    ):
        return None
    return ascii_hostname


def safe_endpoint(value: str | None) -> SafeEndpoint:
    """Parse an endpoint without retaining userinfo, path, query, or fragment.

    The parser is intentionally fail-closed for literal whitespace, control
    characters, backslashes, empty/malformed ports, and unsafe host delimiters.
    ``urlsplit`` otherwise normalizes some of those inputs into plausible
    hostnames, which would make a malformed endpoint look valid in persisted
    metadata.
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

    if not hostname or _authority_has_empty_port(parsed):
        return _invalid_endpoint()
    if (
        _contains_unsafe_endpoint_text(hostname)
        or any(character in hostname for character in "/\\?#@[]%")
    ):
        return _invalid_endpoint()

    normalized_hostname = _normalise_hostname(hostname)
    if normalized_hostname is None:
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
