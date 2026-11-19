"""Model-scope helpers for comparable regression evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from harness.url_safety import canonical_endpoint_host


MODEL_SCOPE_KEYS = (
    "provider",
    "base_url_host",
    "model",
    "reasoning_effort",
    "reasoning_max_tokens",
    "max_output_tokens",
)


def model_config_from_trial(trial: Any) -> dict[str, str]:
    metadata = getattr(trial, "metadata", {}) or {}
    raw = metadata.get("model_config") if isinstance(metadata, dict) else {}
    return normalize_model_config(raw or {}, fallback_model=getattr(trial, "model_used", ""))


def model_scope_from_trial(trial: Any) -> str:
    return model_scope_from_config(
        (getattr(trial, "metadata", {}) or {}).get("model_config", {}),
        fallback_model=getattr(trial, "model_used", ""),
    )


def model_scope_from_agent_config(agent_config: Mapping[str, Any] | None) -> str:
    return model_scope_from_config(agent_config or {})


def model_scope_from_config(
    config: Mapping[str, Any] | None,
    *,
    fallback_model: str = "",
) -> str:
    normalized = normalize_model_config(config or {}, fallback_model=fallback_model)
    if not normalized.get("model"):
        return ""
    return "|".join(
        f"{key}={normalized.get(key, '')}" for key in MODEL_SCOPE_KEYS
    )


def normalize_model_config(
    config: Mapping[str, Any] | None,
    *,
    fallback_model: str = "",
) -> dict[str, str]:
    config = config or {}
    raw_host = config.get("base_url_host")
    raw_url = config.get("base_url")
    raw_port = config.get("base_url_port")
    raw_scheme = config.get("base_url_scheme")
    base_url_host: str | None = None
    if raw_host is not None and str(raw_host).strip() != "":
        host_text = str(raw_host).strip()
        if host_text == "<invalid>":
            base_url_host = "<invalid>"
        else:
            base_url_host = (
                canonical_endpoint_host(host_text, port=raw_port, scheme=raw_scheme)
                or "<invalid>"
            )
    elif raw_url is not None and str(raw_url).strip() != "":
        base_url_host = _base_url_host(str(raw_url))
    model = config.get("model") or fallback_model
    values = {
        "provider": config.get("provider"),
        "base_url_host": base_url_host,
        "model": model,
        "reasoning_effort": config.get("reasoning_effort"),
        "reasoning_max_tokens": config.get("reasoning_max_tokens"),
        "max_output_tokens": config.get("max_output_tokens"),
    }
    return {
        key: str(value)
        for key, value in values.items()
        if value is not None and str(value) != ""
    }


def canonicalize_model_scope(model_scope: str) -> str:
    """Canonicalize a ``key=value|...`` model_scope string using unified endpoint rules."""
    text = str(model_scope or "").strip()
    if not text or "|" not in text or "=" not in text:
        return text
    parsed: dict[str, str] = {}
    for segment in text.split("|"):
        if "=" not in segment:
            return text
        key, value = segment.split("=", 1)
        key = key.strip()
        if key not in MODEL_SCOPE_KEYS:
            return text
        value = value.strip()
        if value:
            parsed[key] = value
    if not parsed.get("model"):
        return text
    return model_scope_from_config(parsed)


def diff_model_scopes(expected_scope: str, actual_scope: str) -> list[str]:
    """Return per-field differences between two canonical model_scope strings."""
    exp_canon = canonicalize_model_scope(expected_scope)
    act_canon = canonicalize_model_scope(actual_scope)

    def _parse(scope: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for segment in scope.split("|"):
            if "=" in segment:
                key, value = segment.split("=", 1)
                result[key.strip()] = value.strip()
        return result

    exp_map = _parse(exp_canon)
    act_map = _parse(act_canon)
    diffs: list[str] = []
    for key in MODEL_SCOPE_KEYS:
        if exp_map.get(key, "") != act_map.get(key, ""):
            diffs.append(
                f"{key}: expected={exp_map.get(key, '')!r}, actual={act_map.get(key, '')!r}"
            )
    if not diffs and exp_canon != act_canon:
        diffs.append(f"scope: expected={exp_canon!r}, actual={act_canon!r}")
    return diffs


def model_scope_matches(snapshot_scope: str, current_scope: str) -> bool:
    if current_scope:
        return canonicalize_model_scope(snapshot_scope) == canonicalize_model_scope(current_scope)
    return snapshot_scope == ""


def safe_model_scope_name(model_scope: str) -> str:
    digest = hashlib.sha256(model_scope.encode("utf-8")).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_scope).strip("_")
    if len(label) > 80:
        label = label[:80].rstrip("_")
    return f"{label or 'model'}__{digest}"


def _base_url_host(base_url: str) -> str:
    text = str(base_url or "")
    if not text:
        return ""
    resolved = canonical_endpoint_host(text)
    return resolved if resolved is not None else "<invalid>"
