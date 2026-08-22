"""Error patterns — known failure → recovery mapping.

This is a living component: the meta-agent adds new patterns
as it discovers recurring failure modes. Over time this becomes
a growing knowledge base of recovery strategies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from harness.recovery.base import ErrorRecovery


@dataclass
class ErrorPatterns(ErrorRecovery):
    name: str = "error_patterns"
    version: str = "0.1.0"

    patterns: dict[str, str] = field(default_factory=lambda: {
        "command_not_found": "Check package installation. Use apt-get or pip to install missing tools.",
        "permission_denied": "Check file permissions. Use chmod or sudo if appropriate.",
        "file_not_found": "Verify the file path. Check the current working directory.",
        "syntax_error": "Re-read the file. Check for typos, missing imports, or incorrect syntax.",
        "test_failure": "Read the test output carefully. Understand what assertion failed and why.",
        "timeout": (
            "The operation took too long. Treat this as a strategy signal, not a "
            "master, sub-agent, or Worker loop stop condition. Prefer a smaller "
            "input, cached/local artifact, incremental check, or more efficient "
            "command before any unchanged retry."
        ),
        "out_of_memory": "The operation used too much memory. Process data in smaller batches.",
        "ssl_certificate_verification": (
            "Repeated TLS/SSL certificate verification failures usually mean the runtime "
            "cannot find a trusted CA bundle. Before retrying package installs or verifier "
            "commands, inspect the container trust store and set or mount an existing CA "
            "bundle through SSL_CERT_FILE, REQUESTS_CA_BUNDLE, and CURL_CA_BUNDLE. Prefer "
            "repairing CA trust or Harbor env/mount configuration; do not disable TLS "
            "verification or edit benchmark tests, solutions, or task definitions."
        ),
        "network_error": "Check network connectivity. Verify URLs and API endpoints.",
    })
    triggers: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "command_not_found": ("command not found",),
        "permission_denied": ("permission denied",),
        "file_not_found": ("file not found", "no such file"),
        "syntax_error": ("syntax error",),
        "test_failure": ("assertion failed", "test failed", "tests failed"),
        "timeout": ("timed out", "timeout"),
        "out_of_memory": ("out of memory", "oom", "killed"),
        "ssl_certificate_verification": (
            "certificate_verify_failed",
            "certificate verify failed",
            "ssl: certificate",
            "sslcertverificationerror",
            "requests.exceptions.sslerror",
            "unable to get local issuer certificate",
        ),
        "network_error": ("network is unreachable", "connection refused", "temporary failure"),
    })

    def classify(self, errors: str | Iterable[str]) -> list[str]:
        """Return recovery pattern keys matching one or more error messages."""
        text = self._normalise_errors(errors)
        matches: list[str] = []
        for key, triggers in self.triggers.items():
            if any(trigger in text for trigger in triggers):
                matches.append(key)
        return matches

    def recovery_for(self, errors: str | Iterable[str]) -> str | None:
        """Return the highest-priority recovery guidance for the observed errors."""
        for key in self.classify(errors):
            guidance = self.patterns.get(key)
            if guidance:
                return guidance
        return None

    def render(self, context: dict[str, object]) -> str:
        raw_errors = context.get("raw_errors") or context.get("errors") or context.get("error")
        if raw_errors:
            matched = self.classify(self._coerce_errors(raw_errors))
            if matched:
                lines = ["## Known Recovery Patterns", ""]
                for key in matched:
                    lines.append(f"- {key}: {self.patterns[key]}")
                return "\n".join(lines)
        return "\n".join(f"- {key}: {value}" for key, value in self.patterns.items())

    def validate(self) -> list[str]:
        errors: list[str] = []
        missing_triggers = sorted(set(self.patterns) - set(self.triggers))
        if missing_triggers:
            errors.append(f"Missing triggers for patterns: {', '.join(missing_triggers)}")
        for key in self.triggers:
            if key not in self.patterns:
                errors.append(f"Trigger configured for unknown pattern: {key}")
        return errors

    def raw_content(self) -> str:
        return self.render({})

    def _normalise_errors(self, errors: str | Iterable[str]) -> str:
        return "\n".join(self._coerce_errors(errors)).lower()

    def _coerce_errors(self, errors: object) -> list[str]:
        if isinstance(errors, str):
            return [errors]
        if isinstance(errors, Iterable):
            return [str(error) for error in errors]
        return [str(errors)]
