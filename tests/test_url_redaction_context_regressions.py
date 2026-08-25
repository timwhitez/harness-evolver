from __future__ import annotations

import json

import pytest

from harness.config import RoleModelConfig, _redact_untyped_config
from harness.url_safety import safe_endpoint


def test_url_mapping_context_propagates_through_nested_dictionaries() -> None:
    legacy = {
        "endpoints": {
            "primary": "alice:password@Primary.Example:9443/v1?token=abc",
            "nested": {
                "secondary": "//other-user:secret@Secondary.Example/v2#private",
            },
            "api_key": "raw-secret",
        }
    }

    redacted = _redact_untyped_config(legacy)
    serialized = json.dumps(redacted, sort_keys=True)

    assert redacted["endpoints"]["primary"] == {
        "valid": True,
        "hostname": "primary.example",
        "port": 9443,
    }
    assert redacted["endpoints"]["nested"]["secondary"] == {
        "valid": True,
        "hostname": "secondary.example",
    }
    assert redacted["endpoints"]["api_key"] == "<redacted>"
    for secret in [
        "alice",
        "password",
        "/v1",
        "token",
        "abc",
        "other-user",
        "secret",
        "/v2",
        "private",
        "raw-secret",
    ]:
        assert secret not in serialized


@pytest.mark.parametrize(
    "raw_url",
    [
        " https://user:secret@Gateway.Example/v1",
        "https://user:secret@Gateway.Example/v1 ",
        "\thttps://user:secret@Gateway.Example/v1",
    ],
)
def test_literal_edge_whitespace_fails_closed_before_normalization(raw_url: str) -> None:
    endpoint = safe_endpoint(raw_url)
    role = RoleModelConfig(model="example", base_url=raw_url)
    redacted = role.redacted()
    serialized = json.dumps(redacted, sort_keys=True)

    assert endpoint.valid is False
    assert endpoint.hostname is None
    assert redacted["base_url_valid"] is False
    assert "base_url_host" not in redacted
    for raw_component in ["user", "secret", "Gateway.Example", "/v1"]:
        assert raw_component not in serialized


@pytest.mark.parametrize(
    "raw_url",
    [
        "gateway.example:",
        "https://user:secret@gateway.example:/v1",
        "https://user:secret@[2001:db8::1]:/v1",
        ".gateway.example/v1",
        "gateway.example../v1",
    ],
)
def test_empty_ports_and_invalid_hostname_roots_fail_closed(raw_url: str) -> None:
    endpoint = safe_endpoint(raw_url)
    role = RoleModelConfig(model="example", base_url=raw_url)
    redacted = role.redacted()
    serialized = json.dumps(redacted, sort_keys=True)

    assert endpoint.valid is False
    assert endpoint.hostname is None
    assert endpoint.port is None
    assert redacted["base_url_valid"] is False
    assert "base_url_host" not in redacted
    assert "base_url_port" not in redacted
    for raw_component in ["user", "secret", "gateway.example", "2001:db8::1", "/v1"]:
        assert raw_component not in serialized
