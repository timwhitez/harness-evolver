from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_trial import resolve_agent_config


def _write_models_config(path: Path) -> None:
    path.write_text(
        """
roles:
  worker:
    provider: openai_compatible
    model: configured-model
    base_url: https://gateway.example/v1
    api_key_env: EXAMPLE_API_KEY
    reasoning:
      effort: high
      max_tokens: 4096
      exclude: false
    max_output_tokens: 8192
    timeout_seconds: 300
    max_retries: 5
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _args(models_config: Path, trials_config: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "models_config": str(models_config),
        "trials_config": str(trials_config),
        "env_file": None,
        "worker_role": "worker",
        "agent": "hl-worker",
        "model": None,
        "provider": None,
        "base_url": None,
        "api_key_env": None,
        "reasoning_effort": None,
        "reasoning_max_tokens": None,
        "max_output_tokens": None,
        "llm_timeout_seconds": None,
        "tool_timeout_seconds": None,
        "max_retries": None,
        "max_turns_audit": None,
        "n_attempts": None,
        "mounts_json": None,
        "verifier_env": None,
        "yes": False,
        "docker_resource_enabled": True,
        "docker_memory": None,
        "docker_memory_swap": None,
        "docker_cpus": None,
        "docker_pids_limit": None,
        "docker_label": None,
        "docker_log_max_size": None,
        "docker_log_max_file": None,
        "force_build": None,
        "network_hardened_environment": False,
        "network_preflight_mode": None,
        "network_preflight_timeout": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_explicit_zero_cli_values_override_positive_role_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    models_config = tmp_path / "models.yaml"
    _write_models_config(models_config)
    monkeypatch.chdir(tmp_path)

    result = resolve_agent_config(
        _args(
            models_config,
            tmp_path / "missing-trials.yaml",
            reasoning_max_tokens=0,
            max_output_tokens="0",
            llm_timeout_seconds=0,
            max_retries=0,
        ),
        argparse.ArgumentParser(),
    )

    assert result["reasoning_max_tokens"] == 0
    assert result["max_output_tokens"] == "0"
    assert result["timeout_seconds"] == 0
    assert result["max_retries"] == 0


def test_none_cli_values_still_fall_back_to_role_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    models_config = tmp_path / "models.yaml"
    _write_models_config(models_config)
    monkeypatch.chdir(tmp_path)

    result = resolve_agent_config(
        _args(models_config, tmp_path / "missing-trials.yaml"),
        argparse.ArgumentParser(),
    )

    assert result["model"] == "configured-model"
    assert result["reasoning_max_tokens"] == 4096
    assert result["max_output_tokens"] == "8192"
    assert result["timeout_seconds"] == 300
    assert result["max_retries"] == 5
