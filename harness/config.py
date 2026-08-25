"""Versioned harness configuration with secret-safe endpoint metadata.

The effective-defaults implementation from PR #28 is retained in
:mod:`harness._config_issue9_base`; this module replaces only log-safe
redaction behavior required by issue #9.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from harness import _config_issue9_base as _base
from harness.url_safety import safe_endpoint

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|bearer|credential|password|passwd|secret|token)",
    flags=re.IGNORECASE,
)
_URL_KEY = re.compile(
    r"(?:^|[_-])(?:base[_-]?urls?|api[_-]?bases?|endpoints?|urls?)(?:$|[_-])",
    flags=re.IGNORECASE,
)


def _safe_endpoint_metadata(value: object) -> dict[str, Any]:
    endpoint = safe_endpoint(str(value or ""))
    metadata: dict[str, Any] = {"valid": endpoint.valid}
    if endpoint.hostname is not None:
        metadata["hostname"] = endpoint.hostname
    if endpoint.port is not None:
        metadata["port"] = endpoint.port
    if endpoint.scheme is not None:
        metadata["scheme"] = endpoint.scheme
    return metadata


def _redact_untyped_config(
    value: Any,
    *,
    key: str = "",
    _url_context: bool = False,
) -> Any:
    """Recursively sanitize legacy/untyped values before persistence.

    URL-valued container fields carry their semantic context through both lists
    and nested mappings. This prevents a scheme-less credential URL below an
    ``endpoints``/``base_urls`` mapping from becoming an unclassified ordinary
    string merely because an intermediate dictionary key is named ``primary``.
    """

    if key and _SECRET_KEY.search(key):
        return "<redacted>"

    url_context = _url_context or bool(key and _URL_KEY.search(key))
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(item_key): _redact_untyped_config(
                item_value,
                key=str(item_key),
                _url_context=url_context,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _redact_untyped_config(
                item,
                key=key,
                _url_context=url_context,
            )
            for item in value
        ]
    if isinstance(value, str) and (
        url_context
        or "://" in value
        or value.startswith("//")
    ):
        return _safe_endpoint_metadata(value)
    return value


def _role_model_config_redacted(self: Any) -> dict[str, Any]:
    """Return effective role configuration without endpoint or key secrets."""

    data = self.model_dump(mode="json")
    endpoint = safe_endpoint(self.base_url)

    # Never retain the caller-provided URL: userinfo, path, query, and fragment
    # can all contain credentials or private gateway details.
    data.pop("base_url", None)
    data.pop("extra", None)

    if self.api_key_env:
        data["api_key_env"] = self.api_key_env
        data["api_key"] = "<from env>"
    if self.base_url:
        data["base_url_valid"] = endpoint.valid
        if endpoint.hostname is not None:
            data["base_url_host"] = endpoint.hostname
        if endpoint.port is not None:
            data["base_url_port"] = endpoint.port
        if endpoint.scheme is not None:
            data["base_url_scheme"] = endpoint.scheme

    return data


def _models_config_redacted(self: Any) -> dict[str, Any]:
    """Preserve PR #28's effective defaults while sanitizing serialization."""

    return {
        "roles": {name: self.get_role(name).redacted() for name in self.roles},
        "defaults": _redact_untyped_config(self.defaults),
    }


_base.RoleModelConfig.redacted = _role_model_config_redacted
_base.ModelsConfig.redacted = _models_config_redacted
RoleModelConfig = _base.RoleModelConfig
ModelsConfig = _base.ModelsConfig
ProviderRequestDefaults = _base.ProviderRequestDefaults
