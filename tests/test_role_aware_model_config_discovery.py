from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from scripts.run_trial import resolve_agent_config


def _write_config(path: Path, role: str, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
roles:
  {role}:
    provider: openai
    model: {model}
    api_key_env: TEST_API_KEY
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_roles(path: Path, roles: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["roles:"]
    for role, model in roles.items():
        lines.extend(
            [
                f"  {role}:",
                "    provider: openai",
                f"    model: {model}",
                "    api_key_env: TEST_API_KEY",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _args(
    *,
    worker_role: str | None = None,
    models_config: str | None = None,
    env_file: str | None = None,
    **overrides: object,
) -> argparse.Namespace:
    values: dict[str, object] = {
        "models_config": models_config,
        "trials_config": "missing-trials.yaml",
        "env_file": env_file,
        "worker_role": worker_role,
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


def test_cli_role_selects_local_config_that_has_no_literal_worker_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path / "config/local.yaml", "worker_gpt", "local-gpt")
    _write_config(tmp_path / "config/models.yaml", "worker", "default-worker")
    monkeypatch.chdir(tmp_path)

    result = resolve_agent_config(
        _args(worker_role="worker_gpt"),
        argparse.ArgumentParser(prog="test"),
    )

    assert result["model"] == "local-gpt"
    assert result["worker_role"] == "worker_gpt"
    assert result["worker_role_source"] == "cli"
    assert result["models_config_path"] == "config/local.yaml"


def test_process_environment_role_uses_role_aware_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path / "config/local.yaml", "worker_gpt", "local-gpt")
    _write_config(tmp_path / "config/models.yaml", "worker", "default-worker")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HL_WORKER_ROLE", "worker_gpt")

    result = resolve_agent_config(_args(), argparse.ArgumentParser(prog="test"))

    assert result["model"] == "local-gpt"
    assert result["worker_role"] == "worker_gpt"
    assert result["worker_role_source"] == "process_environment"
    assert result["models_config_path"] == "config/local.yaml"


def test_selected_dotenv_role_is_loaded_before_model_config_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path / "config/local.yaml", "worker_gpt", "local-gpt")
    env_file = tmp_path / ".env.selected"
    env_file.write_text("HL_WORKER_ROLE=worker_gpt\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HL_WORKER_ROLE", "")

    result = resolve_agent_config(
        _args(env_file=str(env_file)),
        argparse.ArgumentParser(prog="test"),
    )

    assert result["model"] == "local-gpt"
    assert result["worker_role"] == "worker_gpt"
    assert result["worker_role_source"] == "selected_dotenv"
    assert result["models_config_path"] == "config/local.yaml"


def test_process_environment_precedes_conflicting_selected_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_roles(
        tmp_path / "config/local.yaml",
        {"process_role": "process-model", "dotenv_role": "dotenv-model"},
    )
    env_file = tmp_path / ".env.selected"
    env_file.write_text(
        "HL_WORKER_ROLE=dotenv_role\nTEST_SECRET=dotenv-value\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HL_WORKER_ROLE", "process_role")
    monkeypatch.setenv("TEST_SECRET", "process-value")

    result = resolve_agent_config(
        _args(env_file=str(env_file)),
        argparse.ArgumentParser(prog="test"),
    )

    assert result["worker_role"] == "process_role"
    assert result["worker_role_source"] == "process_environment"
    assert result["model"] == "process-model"
    assert os.environ["TEST_SECRET"] == "process-value"


def test_cli_role_precedes_process_environment_and_selected_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_roles(
        tmp_path / "config/local.yaml",
        {
            "cli_role": "cli-model",
            "process_role": "process-model",
            "dotenv_role": "dotenv-model",
        },
    )
    env_file = tmp_path / ".env.selected"
    env_file.write_text("HL_WORKER_ROLE=dotenv_role\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HL_WORKER_ROLE", "process_role")

    result = resolve_agent_config(
        _args(worker_role="cli_role", env_file=str(env_file)),
        argparse.ArgumentParser(prog="test"),
    )

    assert result["worker_role"] == "cli_role"
    assert result["worker_role_source"] == "cli"
    assert result["model"] == "cli-model"


def test_empty_cli_role_is_rejected_instead_of_falling_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path / "config/models.yaml", "worker", "default-worker")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        resolve_agent_config(
            _args(worker_role=""),
            argparse.ArgumentParser(prog="test"),
        )

    assert "--worker-role must not be empty" in capsys.readouterr().err


def test_missing_environment_role_fails_with_checked_paths_and_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(tmp_path / "config/local.yaml", "worker_gpt", "local-gpt")
    _write_config(tmp_path / "config/models.yaml", "worker", "default-worker")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HL_WORKER_ROLE", "typo")

    with pytest.raises(SystemExit):
        resolve_agent_config(_args(), argparse.ArgumentParser(prog="test"))

    stderr = capsys.readouterr().err
    assert "Worker role 'typo'" in stderr
    assert "config/local.yaml" in stderr
    assert "worker_gpt" in stderr
    assert "config/models.yaml" in stderr
    assert "worker" in stderr


def test_explicit_config_validates_environment_selected_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    explicit = tmp_path / "models.yaml"
    _write_config(explicit, "worker", "default-worker")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HL_WORKER_ROLE", "worker_gpt")

    with pytest.raises(SystemExit):
        resolve_agent_config(
            _args(models_config=str(explicit)),
            argparse.ArgumentParser(prog="test"),
        )

    stderr = capsys.readouterr().err
    assert "Worker role 'worker_gpt'" in stderr
    assert str(explicit) in stderr
    assert "Available roles: worker" in stderr


def test_default_role_skips_local_config_when_only_checked_in_config_has_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path / "config/local.yaml", "worker_gpt", "local-gpt")
    _write_config(tmp_path / "config/models.yaml", "worker", "default-worker")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HL_WORKER_ROLE", raising=False)

    result = resolve_agent_config(_args(), argparse.ArgumentParser(prog="test"))

    assert result["model"] == "default-worker"
    assert result["worker_role"] == "worker"
    assert result["worker_role_source"] == "default"
    assert result["models_config_path"] == "config/models.yaml"


def test_cli_only_model_request_remains_supported_without_config_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HL_WORKER_ROLE", raising=False)

    result = resolve_agent_config(
        _args(
            model="provider/model",
            provider="openai",
            api_key_env="CLI_API_KEY",
            llm_timeout_seconds=30,
            max_output_tokens="1024",
            max_retries=0,
        ),
        argparse.ArgumentParser(prog="test"),
    )

    assert result["model"] == "provider/model"
    assert result["provider"] == "openai"
    assert result["api_key_env"] == "CLI_API_KEY"
    assert result["timeout_seconds"] == 30
    assert result["max_output_tokens"] == "1024"
    assert result["max_retries"] == 0
    assert result["worker_role"] == "worker"
    assert result["worker_role_source"] == "default"
    assert result["models_config_path"] is None


def test_missing_all_model_configs_and_explicit_model_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HL_WORKER_ROLE", raising=False)

    with pytest.raises(SystemExit):
        resolve_agent_config(_args(), argparse.ArgumentParser(prog="test"))

    stderr = capsys.readouterr().err
    assert "No model configuration file was found" in stderr
    assert "no explicit --model" in stderr
