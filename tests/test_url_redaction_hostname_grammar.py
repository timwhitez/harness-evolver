from __future__ import annotations

import json

import pytest

from harness.config import RoleModelConfig
from harness.url_safety import safe_endpoint


@pytest.mark.parametrize(
    ("raw_url", "secret_fragment"),
    [
        ("https://example.com;token=abc/v1", "token=abc"),
        ("https://example.com=secret/v1", "secret"),
        ("https://example.com,credential/v1", "credential"),
        ("https://example.com!password/v1", "password"),
        ("https://example.com$apikey/v1", "apikey"),
        ("https://example.com&bearer/v1", "bearer"),
        ("https://service_name/v1", "service_name"),
        ("https://-leading.example/v1", "leading"),
        ("https://trailing-.example/v1", "trailing"),
    ],
)
def test_non_hostname_reg_name_authorities_fail_closed(
    raw_url: str,
    secret_fragment: str,
) -> None:
    endpoint = safe_endpoint(raw_url)
    redacted = RoleModelConfig(model="example", base_url=raw_url).redacted()
    serialized = json.dumps(redacted, sort_keys=True)

    assert endpoint.valid is False
    assert endpoint.hostname is None
    assert redacted["base_url_valid"] is False
    assert "base_url_host" not in redacted
    assert secret_fragment not in serialized


def test_internationalized_hostname_is_normalized_to_idna_ascii() -> None:
    endpoint = safe_endpoint("https://Bücher.Example/v1")

    assert endpoint.valid is True
    assert endpoint.hostname == "xn--bcher-kva.example"
    assert endpoint.scheme == "https"


def test_ipv6_literal_remains_canonical_and_supported() -> None:
    endpoint = safe_endpoint("https://[2001:0db8:0:0:0:0:0:1]:9443/v1")

    assert endpoint.valid is True
    assert endpoint.hostname == "2001:db8::1"
    assert endpoint.port == 9443
