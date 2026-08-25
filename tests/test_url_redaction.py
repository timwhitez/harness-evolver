from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bench.harbor import HarborRunner
from harness.config import ModelsConfig, RoleModelConfig
from harness.config import _redact_untyped_config
from harness.url_safety import safe_endpoint


def test_role_redaction_removes_every_secret_url_component() -> None:
    role = RoleModelConfig(
        provider="openai_compatible",
        model="example",
        base_url="https://user:secret@Gateway.Example:8443/v1?token=abc#private",
        api_key_env="EXAMPLE_API_KEY",
    )

    redacted = role.redacted()
    serialized = json.dumps(redacted, sort_keys=True)

    assert "base_url" not in redacted
    assert redacted["base_url_host"] == "gateway.example"
    assert redacted["base_url_port"] == 8443
    assert redacted["base_url_scheme"] == "https"
    assert redacted["base_url_valid"] is True
    for secret in ["user", "secret", "/v1", "token", "abc", "private"]:
        assert secret not in serialized


def test_models_redaction_preserves_effective_request_defaults() -> None:
    models = ModelsConfig(
        roles={
            "worker": RoleModelConfig(
                model="example",
                base_url="https://name:password@example.test/v1?api_key=value",
            )
        },
        defaults={"timeout": 321, "max_retries": 4},
    )

    redacted = models.redacted()
    serialized = json.dumps(redacted, sort_keys=True)

    assert redacted["roles"]["worker"]["timeout_seconds"] == 321
    assert redacted["roles"]["worker"]["max_retries"] == 4
    assert redacted["defaults"] == {"timeout_seconds": 321, "max_retries": 4}
    assert "password" not in serialized
    assert "api_key=value" not in serialized
    assert "example.test" in serialized


def test_strict_defaults_reject_secret_bearing_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ModelsConfig(
            roles={"worker": RoleModelConfig(model="example")},
            defaults={
                "base_url": "https://user:password@example.test/v1?token=abc",
                "api_key": "raw-secret",
            },
        )


def test_legacy_untyped_values_are_recursively_sanitized_defensively() -> None:
    legacy = {
        "base_url": "https://user:password@Gateway.Example:9443/v1?token=abc#private",
        "headers": {
            "Authorization": "Bearer header-secret",
            "x-api-key": "key-secret",
        },
        "fallbacks": [
            "https://name:pass@backup.example/v2?credential=value",
            {"access_token": "nested-secret"},
        ],
        "timeout": 300,
    }

    redacted = _redact_untyped_config(legacy)
    serialized = json.dumps(redacted, sort_keys=True)

    assert redacted["base_url"] == {
        "valid": True,
        "hostname": "gateway.example",
        "port": 9443,
        "scheme": "https",
    }
    assert redacted["timeout"] == 300
    for secret in [
        "user",
        "password",
        "/v1",
        "token=abc",
        "private",
        "header-secret",
        "key-secret",
        "name:pass",
        "/v2",
        "credential=value",
        "nested-secret",
    ]:
        assert secret not in serialized


def test_url_list_items_keep_parent_key_context_for_scheme_less_values() -> None:
    legacy = {
        "base_urls": [
            "alice:password@Backup.Example:9443/v1?token=abc",
            "//other:secret@Secondary.Example/v2#private",
        ]
    }

    redacted = _redact_untyped_config(legacy)
    serialized = json.dumps(redacted, sort_keys=True)

    assert redacted["base_urls"] == [
        {"valid": True, "hostname": "backup.example", "port": 9443},
        {"valid": True, "hostname": "secondary.example"},
    ]
    for secret in [
        "alice",
        "password",
        "/v1",
        "token",
        "abc",
        "other",
        "secret",
        "/v2",
        "private",
    ]:
        assert secret not in serialized


@pytest.mark.parametrize(
    ("url", "host", "port"),
    [
        ("gateway.example/v1", "gateway.example", None),
        ("//gateway.example/v1", "gateway.example", None),
        ("localhost:8000/v1", "localhost", 8000),
        ("https://[2001:db8::1]:9443/v1", "2001:db8::1", 9443),
    ],
)
def test_safe_endpoint_handles_scheme_less_hosts_and_ipv6(
    url: str,
    host: str,
    port: int | None,
) -> None:
    parsed = safe_endpoint(url)

    assert parsed.valid is True
    assert parsed.hostname == host
    assert parsed.port == port


@pytest.mark.parametrize(
    "url",
    [
        "gateway example/v1",
        "https://gateway.example\n.evil/v1",
        "https://gateway.example\\private-segment/v1",
        "https://%65xample.com/v1",
        "https://[bad/v1",
    ],
)
def test_malformed_endpoints_fail_closed_without_retaining_raw_text(url: str) -> None:
    role = RoleModelConfig(model="example", base_url=url)

    redacted = role.redacted()
    serialized = json.dumps(redacted, sort_keys=True)

    assert redacted["base_url_valid"] is False
    assert "base_url_host" not in redacted
    for raw_component in [
        "gateway example",
        "evil",
        "private-segment",
        "%65xample",
        "[bad",
    ]:
        assert raw_component not in serialized


def test_malformed_port_never_falls_back_to_raw_url() -> None:
    role = RoleModelConfig(
        model="example",
        base_url="https://user:secret@example.test:bad/v1",
    )

    redacted = role.redacted()
    serialized = json.dumps(redacted, sort_keys=True)

    assert redacted["base_url_valid"] is False
    assert "base_url_host" not in redacted
    assert "user" not in serialized
    assert "secret" not in serialized
    assert "bad" not in serialized


def test_harbor_metadata_contains_hostname_only() -> None:
    runner = HarborRunner()

    metadata = runner._model_config_metadata(
        {
            "model": "example",
            "base_url": "https://user:secret@Gateway.Example:8443/v1?token=abc",
        }
    )

    assert metadata["base_url_host"] == "gateway.example"
    serialized = json.dumps(metadata, sort_keys=True)
    assert "user" not in serialized
    assert "secret" not in serialized
    assert "8443" not in serialized
    assert "token" not in serialized
