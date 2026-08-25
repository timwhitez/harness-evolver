from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.config import ModelsConfig


def test_defaults_are_applied_to_effective_role_and_litellm_kwargs() -> None:
    models = ModelsConfig.model_validate(
        {
            "roles": {
                "worker": {
                    "provider": "openai",
                    "model": "gpt-example",
                }
            },
            "defaults": {
                "timeout": 300,
                "max_retries": 3,
            },
        }
    )

    role = models.get_role("worker")

    assert role.timeout_seconds == 300
    assert role.max_retries == 3
    assert role.litellm_kwargs()["timeout"] == 300
    assert role.litellm_kwargs()["num_retries"] == 3


def test_role_values_override_defaults() -> None:
    models = ModelsConfig.model_validate(
        {
            "roles": {
                "worker": {
                    "provider": "openai",
                    "model": "gpt-example",
                    "timeout_seconds": 900,
                    "max_retries": 7,
                }
            },
            "defaults": {
                "timeout": 300,
                "max_retries": 3,
            },
        }
    )

    role = models.get_role("worker")

    assert role.timeout_seconds == 900
    assert role.max_retries == 7


def test_redacted_snapshot_reports_effective_values() -> None:
    models = ModelsConfig.model_validate(
        {
            "roles": {"worker": {"model": "gpt-example"}},
            "defaults": {"timeout": 300, "max_retries": 3},
        }
    )

    snapshot = models.redacted()

    assert snapshot["roles"]["worker"]["timeout_seconds"] == 300
    assert snapshot["roles"]["worker"]["max_retries"] == 3
    assert snapshot["defaults"] == {
        "timeout_seconds": 300,
        "max_retries": 3,
    }


def test_unknown_or_unimplemented_defaults_are_rejected() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        ModelsConfig.model_validate(
            {
                "roles": {"worker": {"model": "gpt-example"}},
                "defaults": {"temperature": 0.0},
            }
        )


@pytest.mark.parametrize(
    "defaults",
    [
        {"timeout": 0},
        {"timeout": -1},
        {"max_retries": -1},
        {"timeout": True},
        {"timeout": False},
        {"max_retries": True},
        {"max_retries": False},
        {"timeout": "300"},
        {"max_retries": "3"},
        {"timeout": 300.0},
        {"max_retries": 3.0},
    ],
)
def test_invalid_default_values_are_rejected(defaults: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(
            {
                "roles": {"worker": {"model": "gpt-example"}},
                "defaults": defaults,
            }
        )
