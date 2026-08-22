"""Model-scope helpers for comparable regression evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping
from urllib.parse import urlparse


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
    base_url_host = config.get("base_url_host")
    if not base_url_host and config.get("base_url"):
        base_url_host = _base_url_host(str(config.get("base_url")))
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


def model_scope_matches(snapshot_scope: str, current_scope: str) -> bool:
    if current_scope:
        return snapshot_scope == current_scope
    return snapshot_scope == ""


def safe_model_scope_name(model_scope: str) -> str:
    digest = hashlib.sha256(model_scope.encode("utf-8")).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_scope).strip("_")
    if len(label) > 80:
        label = label[:80].rstrip("_")
    return f"{label or 'model'}__{digest}"


def _base_url_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.netloc:
        return parsed.netloc
    return base_url.split("/")[0]
