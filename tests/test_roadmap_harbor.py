import json
import asyncio
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import bench.harbor as harbor_module
from bench.harbor_adapter import (
    HarborGlobTool,
    HLWorkerHarborAgent,
    HarborFileEditTool,
    HarborFileReadTool,
    HarborFileWriteTool,
    HarborShellTool,
    HarborVerifyTool,
)
from bench.harbor import HarborRunner
from bench.network_environment import (
    DockerResourceConfig,
    PREBUILT_WARMUP_FAILURE_RECEIPT,
    PREBUILT_WARMUP_FAILURE_RECEIPT_SCHEMA,
    write_docker_resource_compose_file,
)
from bench.tasks import TaskCatalog
from harness.tools.registry import ToolRegistry
from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus


def _assert_policy_guard_is_non_terminal(metadata):
    assert metadata["policy_guard_stop_condition"] is False
    assert metadata["operation_guard_stop_condition"] is False
    assert metadata["loop_stop_condition"] is False


def _assert_terminal_environment_metadata(metadata):
    assert metadata["terminal_environment_unavailable"] is True
    assert metadata["hard_environment_evidence"] is True
    assert metadata["terminal_environment_hard_evidence"] is True
    assert metadata["terminal_environment_unavailable_stop_condition"] is False
    assert metadata["terminal_environment_recovery_stop_condition"] is False
    assert metadata["loop_stop_condition"] is False
    assert metadata["worker_loop_stop_condition"] is False
    assert metadata["master_loop_stop_condition"] is False
    assert metadata["sub_agent_loop_stop_condition"] is False
    assert metadata["time_round_token_limit_driven"] is False
    assert metadata["terminal_environment_marker"]
    assert metadata["terminal_environment_first_observation"]
    assert metadata["terminal_environment_evidence_source"] == "harbor_environment_exec"


def test_harbor_command_uses_installed_cli_shape(tmp_path):
    runner = HarborRunner(dataset_path="terminal-bench-tasks/terminal-bench", jobs_dir=tmp_path)
    command = runner.build_command(
        "vulnerable-secret",
        {
            "agent": "hl-worker",
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "xhigh",
            "env_file": ".env.local",
            "force_build": True,
            "docker_hub_mirror": "docker.m.daocloud.io",
            "mounts_json": [
                {
                    "type": "bind",
                    "source": "/etc/ssl/certs/ca-certificates.crt",
                    "target": "/etc/ssl/certs/ca-certificates.crt",
                    "read_only": True,
                }
            ],
            "verifier_env": [
                "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
                "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt",
            ],
            "yes": True,
            "timeout_seconds": 900,
            "tool_timeout_seconds": 120,
            "max_retries": 5,
            "custom_llm_provider": "openai",
            "n_attempts": 5,
            "goal_path": str(tmp_path / "goals" / "campaign.json"),
            "memory_path": str(tmp_path / "trials"),
        },
        job_name="job1",
    )
    assert "--agent-config" not in command.argv
    assert "--output" not in command.argv
    assert "--agent-import-path" in command.argv
    assert "--include-task-name" in command.argv
    assert "--env-file" in command.argv
    assert ".env.local" in command.argv
    assert "--force-build" in command.argv
    assert "--agent-timeout-multiplier" not in command.argv
    assert "--verifier-timeout-multiplier" not in command.argv
    assert "--environment-build-timeout-multiplier" not in command.argv
    assert "--mounts-json" in command.argv
    assert "--environment-import-path" in command.argv
    assert "bench.network_environment:AptMirrorDockerEnvironment" in command.argv
    assert "--verifier-env" in command.argv
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in command.argv
    assert "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in command.argv
    assert "--yes" in command.argv
    assert "--n-attempts" in command.argv
    assert command.argv[command.argv.index("--n-attempts") + 1] == "5"
    assert "timeout_seconds=900" in command.argv
    assert "max_retries=5" in command.argv
    assert "custom_llm_provider=openai" in command.argv
    assert f"goal_path={tmp_path / 'goals' / 'campaign.json'}" in command.argv
    assert f"memory_path={tmp_path / 'trials'}" in command.argv
    assert "n_attempts=5" not in command.argv
    assert "tool_timeout_seconds=120" in command.argv
    assert command.config["agent"]["kwargs"]["goal_path"] == str(
        tmp_path / "goals" / "campaign.json"
    )
    assert command.config["agent"]["kwargs"]["memory_path"] == str(tmp_path / "trials")
    assert "--agent-env" in command.argv
    assert "HL_WORKER_API_KEY_ENV=OPENAI_API_KEY" in command.argv
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in command.argv
    assert "REQUESTS_CA_BUNDLE" not in " ".join(command.argv)
    assert command.config["environment"]["force_build"] is True
    assert command.config["n_attempts"] == 5
    assert command.config["timeouts"] == {}
    assert command.config["environment"]["build_timeout_multiplier"] is None
    assert (
        command.config["environment"]["import_path"]
        == "bench.network_environment:AptMirrorDockerEnvironment"
    )
    assert command.config["environment"]["kwargs"]["docker_hub_mirror"] == "docker.m.daocloud.io"
    assert (
        command.config["environment"]["kwargs"]["prebuilt_docker_hub_mirror"]
        == ""
    )
    assert command.config["environment"]["mounts_json"][0]["type"] == "bind"
    assert command.config["verifier_env"] == [
        "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
        "CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt",
    ]
    assert command.config["agent"]["env"] == {
        "HL_WORKER_API_KEY_ENV": "OPENAI_API_KEY",
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    }
    mounts_index = command.argv.index("--mounts-json") + 1
    assert json.loads(command.argv[mounts_index])[0]["read_only"] is True
    assert command.job_dir == tmp_path / "job1"


def test_harbor_n_attempts_non_positive_is_not_a_loop_stop_condition(tmp_path):
    runner = HarborRunner(dataset_path="terminal-bench-tasks/terminal-bench", jobs_dir=tmp_path)

    command = runner.build_command(
        "fix-git",
        {"agent": "hl-worker", "n_attempts": 0},
        job_name="job1",
    )

    assert "--n-attempts" not in command.argv
    assert command.config["n_attempts"] is None


@pytest.mark.parametrize(
    "agent_name",
    [
        "aider",
        "amp",
        "codex",
        "claude-code",
        "cursor-agent",
        "droid",
        "factory",
        "forgecode",
        "factory-droid",
        "gemini",
        "gemini-cli",
        "openai-codex",
        "opencode",
    ],
)
def test_harbor_command_rejects_external_agent_delegate(tmp_path, agent_name):
    runner = HarborRunner(dataset_path="terminal-bench-tasks/terminal-bench", jobs_dir=tmp_path)

    with pytest.raises(ValueError, match="self-owned hl-worker"):
        runner.build_command(
            "vulnerable-secret",
            {"agent": agent_name},
            job_name="job1",
        )


@pytest.mark.parametrize(
    "agent_name",
    [
        "codex exec --json 'fix this'",
        "openai-codex exec --json 'fix this'",
        "python -m codex exec 'fix this'",
        "python -m openai.codex exec 'fix this'",
        "python -c \"import runpy; runpy.run_module('codex.cli', run_name='__main__')\"",
        "npx --yes codex exec 'fix this'",
        "uvx openai-codex exec 'fix this'",
        "pipx run openai-codex exec 'fix this'",
        "uv tool run codex exec 'fix this'",
        "bash -lc 'codex exec --json fix'",
        "env -i claude-code --print 'fix this'",
        "opencode run 'fix this'",
    ],
)
def test_harbor_command_rejects_wrapped_external_agent_delegate(tmp_path, agent_name):
    runner = HarborRunner(dataset_path="terminal-bench-tasks/terminal-bench", jobs_dir=tmp_path)

    with pytest.raises(ValueError, match="Only the master HL orchestrator"):
        runner.build_command(
            "vulnerable-secret",
            {"agent": agent_name},
            job_name="job1",
        )


def test_harbor_command_rejects_timeout_resource_overrides(tmp_path):
    runner = HarborRunner(dataset_path="terminal-bench-tasks/terminal-bench", jobs_dir=tmp_path)

    with pytest.raises(ValueError, match="official task timeouts/resources unchanged"):
        runner.build_command(
            "vulnerable-secret",
            {
                "agent": "hl-worker",
                "agent_timeout_multiplier": 2.0,
            },
            job_name="job1",
        )


def test_harbor_command_rejects_unsafe_resource_override_aliases(tmp_path):
    runner = HarborRunner(dataset_path="terminal-bench-tasks/terminal-bench", jobs_dir=tmp_path)

    with pytest.raises(ValueError, match="official task timeouts/resources unchanged"):
        runner.build_command(
            "vulnerable-secret",
            {
                "agent": "hl-worker",
                "override_storage_mb": 10240,
            },
            job_name="job1",
        )


def test_harbor_command_allows_local_docker_safety_caps(tmp_path):
    runner = HarborRunner(dataset_path="terminal-bench-tasks/terminal-bench", jobs_dir=tmp_path)

    command = runner.build_command(
        "vulnerable-secret",
        {
            "agent": "hl-worker",
            "docker_memory": "2g",
            "docker_memory_swap": "2g",
            "docker_cpus": 1,
            "docker_pids_limit": 512,
            "docker_labels": {"com.example.scope": "test"},
        },
        job_name="job1",
    )

    assert "--override-cpus" in command.argv
    assert command.argv[command.argv.index("--override-cpus") + 1] == "1"
    assert "--override-memory-mb" in command.argv
    assert command.argv[command.argv.index("--override-memory-mb") + 1] == "2048"
    assert "--delete" in command.argv
    resources = command.config["environment"]["docker_resources"]
    assert resources["memory"] == "2g"
    assert resources["memory_swap"] == "2g"
    assert resources["pids_limit"] == 512
    assert resources["delete_volumes"] is False
    assert resources["labels"]["com.harness-evolver.managed"] == "true"
    assert resources["labels"]["com.example.scope"] == "test"
    kwargs = command.config["environment"]["kwargs"]
    assert kwargs["docker_memory"] == "2g"
    assert kwargs["docker_memory_swap"] == "2g"
    assert kwargs["docker_pids_limit"] == 512


def test_docker_resource_compose_uses_consistent_pids_limits(tmp_path):
    path = write_docker_resource_compose_file(
        tmp_path / "hl-docker-resources.json",
        service_names=["main"],
        config=DockerResourceConfig(pids_limit=512),
        session_id="session-1",
        environment_name="fix-git",
    )

    payload = json.loads(path.read_text())
    service = payload["services"]["main"]

    assert service["pids_limit"] == 512
    assert service["deploy"]["resources"]["limits"]["pids"] == 512


def test_harbor_runner_retries_infra_errors_beyond_audit_reference(tmp_path, monkeypatch):
    runner = HarborRunner(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "runs")
    calls = []
    sleep_calls = []

    def fail_on_sleep(seconds):
        sleep_calls.append(seconds)
        raise AssertionError("infra retry delay is audit-only and must not sleep")

    monkeypatch.setattr(harbor_module.time, "sleep", fail_on_sleep)

    def fake_run_command(argv, *, timeout_audit):
        calls.append(argv)
        job_name = argv[argv.index("--job-name") + 1]
        job_dir = tmp_path / "jobs" / job_name
        job_dir.mkdir(parents=True)
        trial_name = f"cobol-modernization__{len(calls)}"
        if len(calls) > 2:
            (job_dir / "result.json").write_text(
                json.dumps(
                    {
                        "trial_results": [
                            {
                                "task_name": "cobol-modernization",
                                "trial_name": trial_name,
                                "verifier_result": {"rewards": {"reward": 0.0}},
                            }
                        ]
                    }
                )
            )
            return subprocess.CompletedProcess(argv, 0, "", "")
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "task_name": "cobol-modernization",
                            "trial_name": trial_name,
                            "exception_info": {
                                "exception_type": "EnvironmentStartTimeoutError",
                                "exception_message": (
                                    "Environment start timed out after a transient "
                                    "Debian package fetch failure"
                                ),
                            },
                        }
                    ]
                }
            )
        )
        return subprocess.CompletedProcess(argv, 1, "", "")

    runner._run_command = fake_run_command

    result = runner.run_task(
        "cobol-modernization",
        {
            "agent": "hl-worker",
            "infra_retries": 0,
            "network_hardened_environment": True,
        },
        job_name="job",
        jobs_dir=tmp_path / "jobs",
    )

    assert len(calls) == 3
    assert sleep_calls == []
    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.metadata["infra_error_detected"] is False
    assert result.metadata["infra_retries_configured"] == 0
    assert result.metadata["infra_retries_audit_only"] == 0
    assert result.metadata["infra_retries_stop_condition"] is False
    assert result.metadata["infra_retry_attempt_count_stop_condition"] is False
    assert result.metadata["infra_retry_reference_stop_condition"] is False
    assert result.metadata["infra_retry_loop_stop_condition"] is False
    assert result.metadata["infra_retry_unbounded_by_attempt_count"] is True
    assert result.metadata["time_round_token_limit_driven"] is False
    assert result.metadata["infra_retry_reference_exceeded"] is True
    assert "score_exclusion_reason" not in result.metadata
    assert [item["job_dir"] for item in result.metadata["infra_retry_attempts"]] == [
        str(tmp_path / "jobs" / "job"),
        str(tmp_path / "jobs" / "job_infra_retry1"),
        str(tmp_path / "jobs" / "job_infra_retry2"),
    ]
    assert [
        item["infra_error_detected"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [True, True, False]
    assert [
        item["infra_retry_reference_exceeded"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, True, True]
    assert [
        item["infra_retry_attempt_index_audit_only"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [0, 1, 2]
    assert [
        item["infra_retry_reference_stop_condition"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, False, False]
    assert [
        item["infra_retry_loop_stop_condition"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, False, False]
    assert [
        item["infra_retry_unbounded_by_attempt_count"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [True, True, True]
    assert [
        item["time_round_token_limit_driven"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, False, False]
    assert [
        item.get("infra_retry_delay_stop_condition")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert [
        item.get("infra_retry_cooldown_stop_condition")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert [
        item.get("infra_retry_delay_runtime_wait_condition")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert [
        item.get("infra_retry_delay_wait_executed")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert "infra_retry_delay_stop_condition" not in result.metadata["infra_retry_attempts"][2]


def test_harbor_runner_retries_verifier_runtime_prepare_timeout_beyond_audit_reference(
    tmp_path,
    monkeypatch,
):
    runner = HarborRunner(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "runs")
    calls = []
    monkeypatch.setattr(harbor_module.time, "sleep", lambda _seconds: None)

    def fake_run_command(argv, *, timeout_audit):
        calls.append(argv)
        job_name = argv[argv.index("--job-name") + 1]
        job_dir = tmp_path / "jobs" / job_name
        trial_name = f"regex-log__{len(calls)}"
        trial_dir = job_dir / trial_name
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True)
        if len(calls) > 2:
            (job_dir / "result.json").write_text(
                json.dumps(
                    {
                        "trial_results": [
                            {
                                "task_name": "regex-log",
                                "trial_name": trial_name,
                                "verifier_result": {"rewards": {"reward": 0.0}},
                            }
                        ]
                    }
                )
            )
            return subprocess.CompletedProcess(argv, 0, "", "")
        (verifier_dir / "test-stdout.txt").write_text(
            "Verifier runtime network preparation timed out after 90 seconds\n"
            "while running /tmp/hl-verifier-network-prepared setup\n"
        )
        (trial_dir / "exception.txt").write_text(
            "Traceback (most recent call last):\n"
            "  File \"bench/network_environment.py\", line 188, "
            "in _prepare_verifier_runtime\n"
            "TimeoutError: Verifier runtime network preparation timed out after 90 seconds\n"
        )
        (job_dir / "result.json").write_text(
            json.dumps({"stats": {"n_running_trials": 1, "n_cancelled_trials": 1}})
        )
        return subprocess.CompletedProcess(argv, 1, "", "")

    runner._run_command = fake_run_command

    result = runner.run_task(
        "regex-log",
        {
            "agent": "hl-worker",
            "infra_retries": 0,
            "network_hardened_environment": True,
        },
        job_name="job",
        jobs_dir=tmp_path / "jobs",
    )

    assert len(calls) == 3
    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.metadata["infra_error_detected"] is False
    assert result.metadata["infra_retries_configured"] == 0
    assert result.metadata["infra_retry_reference_exceeded"] is True
    assert [
        item["status"] for item in result.metadata["infra_retry_attempts"]
    ] == ["timeout", "timeout", "failed"]
    assert [
        item["infra_error_detected"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [True, True, False]
    assert [
        item["infra_retry_reference_stop_condition"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, False, False]
    assert [
        item["infra_retry_loop_stop_condition"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, False, False]
    assert [
        item["infra_retry_unbounded_by_attempt_count"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [True, True, True]
    assert [
        item.get("infra_retry_delay_stop_condition")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert [
        item.get("infra_retry_cooldown_stop_condition")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert "infra_retry_delay_stop_condition" not in result.metadata[
        "infra_retry_attempts"
    ][2]


def test_harbor_runner_does_not_spin_on_prebuilt_warmup_403(tmp_path):
    runner = HarborRunner(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "runs")
    calls = []

    def fake_run_command(argv, *, timeout_audit):
        calls.append(argv)
        job_name = argv[argv.index("--job-name") + 1]
        job_dir = tmp_path / "jobs" / job_name
        trial_name = f"mailman__{len(calls)}"
        trial_dir = job_dir / trial_name
        trial_dir.mkdir(parents=True)
        message = (
            "Prebuilt Docker image cache warmup failed for image "
            "docker.1panel.live/alexgshaw/mailman:20251031 with return code 1. "
            "Stderr: failed to copy: httpReadSeeker: failed open: unexpected "
            "status code https://docker.1panel.live/v2/alexgshaw/mailman/manifests/"
            "20251031: 403 Forbidden"
        )
        (trial_dir / PREBUILT_WARMUP_FAILURE_RECEIPT).write_text(
            json.dumps(
                {
                    "schema": PREBUILT_WARMUP_FAILURE_RECEIPT_SCHEMA,
                    "kind": "prebuilt_image_cache_warmup_failure",
                    "source": "apt_mirror_docker_environment_start",
                    "deterministic_access_failure": True,
                }
            )
        )
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "task_name": "mailman",
                            "trial_name": trial_name,
                            "exception_info": {
                                "exception_type": "RuntimeError",
                                "exception_message": message,
                            },
                        }
                    ]
                }
            )
        )
        return subprocess.CompletedProcess(argv, 1, "", "")

    runner._run_command = fake_run_command

    result = runner.run_task(
        "mailman",
        {
            "agent": "hl-worker",
            "infra_retries": 0,
            "network_hardened_environment": True,
        },
        job_name="job",
        jobs_dir=tmp_path / "jobs",
    )

    assert len(calls) == 1
    assert result.status == TrialStatus.ERROR
    assert result.metadata["infra_error_detected"] is True
    assert result.metadata["prebuilt_warmup_failure_nonretryable"] is True
    assert result.metadata["infra_retry_suppressed_reason"] == (
        "deterministic_prebuilt_image_warmup_failure"
    )
    assert result.metadata["infra_retry_suppressed_stop_condition"] is False
    assert result.metadata.get("infra_retry_scheduled") is not True
    assert result.metadata["infra_retry_attempts"][0]["infra_error_detected"] is True
    assert len(result.metadata["infra_retry_attempts"]) == 1


def test_harbor_runner_marks_docker_build_download_failure_as_infra():
    trial = TrialResult(
        trial_id="qemu-startup__abc",
        task_id="qemu-startup",
        task_domain=TaskDomain.SYSTEM_ADMINISTRATION,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.ERROR,
        error_log=[
            "Docker compose command failed for environment qemu-startup. "
            "failed to solve: process \"/bin/sh -c wget -q "
            "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/"
            "alpine-extended-3.19.0-x86_64.iso -O /app/alpine.iso\" "
            "did not complete successfully: exit code: 5"
        ],
        harbor_stdout='Acquire::http::Timeout "30";',
    )

    assert HarborRunner().is_infra_error(trial) is True


def test_parse_harbor_job_result_requires_verifier_reward(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "vulnerable-secret__abc"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "vulnerable-secret",
                        "trial_name": "vulnerable-secret__abc",
                        "task_id": {"path": "vulnerable-secret"},
                        "agent_info": {
                            "name": "hl-worker",
                            "version": "0.1.0",
                            "model_info": {"name": "gpt-5.4", "provider": "openai"},
                        },
                        "agent_result": {"n_input_tokens": 10, "n_output_tokens": 5},
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    }
                ]
            }
        )
    )
    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="vulnerable-secret",
    )
    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.score == 0.0
    assert result.token_usage == {"input": 10, "output": 5}
    assert result.metadata["model_config"] == {
        "provider": "openai",
        "model": "gpt-5.4",
    }


def test_parse_harbor_job_result_preserves_verifier_environment_logs(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "fix-git__abc"
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "test-stdout.txt").write_text(
        "curl: (60) SSL certificate problem: unable to get local issuer certificate\n"
        "/tests/test.sh: line 10: /root/.local/bin/env: No such file or directory\n"
        "/tests/test.sh: line 19: uvx: command not found\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "fix-git",
                        "trial_name": "fix-git__abc",
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    }
                ]
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="fix-git",
    )

    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.metadata["verifier_infra_error"] is True
    assert "uvx: command not found" in result.metadata["verifier_logs"]
    assert "SSL certificate problem" in "\n".join(result.error_log)
    assert HarborRunner().is_infra_error(result) is True


def test_parse_harbor_job_result_marks_verifier_cache_permission_as_infra(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "mailman__abc"
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "test-stdout.txt").write_text(
        "x Failed to download and build `mailman==3.3.8`\n"
        "|- Failed to write to the distribution cache\n"
        "`- failed to rename file from /tmp/hl-verifier-cache/uv/.tmp2XLEC3 "
        "to /tmp/hl-verifier-cache/uv/archive-v0/oxEyQsybl_cOPz6B3GE5T: "
        "Permission denied (os error 13)\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "mailman",
                        "trial_name": "mailman__abc",
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    }
                ]
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="mailman",
    )

    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.metadata["verifier_infra_error"] is True
    assert "Failed to write to the distribution cache" in result.metadata["verifier_logs"]
    assert HarborRunner().is_infra_error(result) is True


def test_harbor_runner_excludes_verified_verifier_infra_failures_from_score(tmp_path):
    runner = HarborRunner(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "runs")

    def fake_run_command(argv, *, timeout_audit):
        job_name = argv[argv.index("--job-name") + 1]
        job_dir = tmp_path / "jobs" / job_name
        trial_dir = job_dir / "pypi-server__abc"
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True)
        (verifier_dir / "test-stdout.txt").write_text(
            "curl: (6) Could not resolve host: github.com\n"
            "/tests/test.sh: line 10: /root/.local/bin/env: No such file or directory\n"
            "/tests/test.sh: line 19: uvx: command not found\n"
        )
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "task_name": "pypi-server",
                            "trial_name": "pypi-server__abc",
                            "verifier_result": {"rewards": {"reward": 0.0}},
                        }
                    ]
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    runner._run_command = fake_run_command

    result = runner.run_task(
        "pypi-server",
        {"agent": "hl-worker", "network_hardened_environment": True},
        job_name="job",
        jobs_dir=tmp_path / "jobs",
    )

    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.metadata["verifier_infra_error"] is True
    assert result.metadata["infra_error_detected"] is True
    assert result.metadata["score_exclusion_reason"] == "infrastructure_error"


def test_parse_harbor_job_result_does_not_treat_successful_apt_logs_as_infra(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "video-processing__abc"
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "test-stdout.txt").write_text(
        "Hit:1 http://deb.debian.org/debian bookworm InRelease\n"
        "Setting up curl (7.88.1-10+deb12u14) ...\n"
        "FAILED ../tests/test_outputs.py::test_jump_analyzer_example_video\n"
        "E       AssertionError: Takeoff frame 55 not within inclusive range [50, 54]\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "video-processing",
                        "trial_name": "video-processing__abc",
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    }
                ]
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="video-processing",
    )

    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.metadata["verifier_infra_error"] is False
    assert HarborRunner().is_infra_error(result) is False


def test_parse_harbor_job_result_records_redacted_model_config(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "task-a__abc"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "task-a",
                        "trial_name": "task-a__abc",
                        "agent_info": {
                            "name": "hl-worker",
                            "model_info": {"name": "gpt-5.5", "provider": "openai"},
                        },
                        "agent_result": {
                            "n_input_tokens": 12,
                            "n_cache_tokens": 3,
                            "n_output_tokens": 4,
                            "cost_usd": 0.123,
                            "n_turns": 5,
                            "n_api_calls": 7,
                            "api_error_count": 1,
                            "provider_latency_ms": 250.5,
                        },
                        "verifier_result": {"rewards": {"reward": 1.0}},
                    }
                ]
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="task-a",
        agent_config={
            "worker_role": "worker_gpt",
            "provider": "openai",
            "model": "gpt-5.4",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "xhigh",
            "timeout_seconds": 900,
            "max_retries": 5,
        },
    )

    model_config = result.metadata["model_config"]
    assert result.model_used == "openai/gpt-5.5"
    assert result.token_usage == {"input": 12, "cache": 3, "output": 4}
    assert result.metadata["trial_metrics"] == {
        "cost_usd": 0.123,
        "n_turns": 5,
        "n_api_calls": 7,
        "api_error_count": 1,
        "provider_latency_ms": 250.5,
        "cache_hit_ratio": 0.2,
    }
    assert model_config["worker_role"] == "worker_gpt"
    assert model_config["provider"] == "openai"
    assert model_config["model"] == "gpt-5.4"
    assert model_config["base_url_host"] == "api.openai.com"
    assert model_config["api_key_env"] == "OPENAI_API_KEY"
    assert model_config["reasoning_effort"] == "xhigh"
    assert "api_key" not in model_config
    assert "sk-" not in json.dumps(model_config)


def test_parse_harbor_job_result_treats_verified_reward_pass_as_pass_with_warning(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "task-a__abc"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "task-a",
                        "trial_name": "task-a__abc",
                        "verifier_result": {"rewards": {"reward": 1.0}},
                        "exception_info": {
                            "exception_type": "AgentTimeoutError",
                            "exception_message": "Agent execution timed out after 1800s",
                        },
                    }
                ]
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="task-a",
    )

    assert result.status == TrialStatus.PASSED
    assert result.verified is True
    assert result.score == 1.0
    assert result.metadata["verified_pass_with_agent_exception"] is True
    assert result.metadata["completion_hygiene_warning"] is True
    assert result.metadata["agent_exception_type"] == "AgentTimeoutError"
    assert any("completion hygiene" in error for error in result.error_log)


def test_parse_harbor_job_result_flags_exception_after_done_tool(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "task-a__abc"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "trajectory.jsonl").write_text(
        '\n'.join(
            [
                json.dumps({"type": "tool_call", "tool": "bash", "success": False}),
                json.dumps({"type": "tool_call", "tool": "done", "success": True}),
            ]
        )
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
                "trial_results": [
                    {
                        "task_name": "task-a",
                        "trial_name": "task-a__abc",
                        "exception_info": {
                            "exception_type": "RuntimeError",
                            "exception_message": "Command timed out after 90 seconds",
                        },
                    }
                ],
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="task-a",
    )

    assert result.status == TrialStatus.ERROR
    assert result.verified is False
    assert result.metadata["post_completion_agent_exception"] is True
    assert result.metadata["job_result_status_counts"] == {
        "completed": 1,
        "errored": 1,
        "trial_results": 1,
    }
    assert any("post-completion exception" in error for error in result.error_log)


def test_parse_harbor_job_result_uses_exception_traceback_for_verifier_runtime_prepare(
    tmp_path,
):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "task-a__abc"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "trajectory.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "tool_call", "tool": "write", "success": True}),
                json.dumps({"type": "tool_call", "tool": "done", "success": True}),
            ]
        )
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
                "trial_results": [
                    {
                        "task_name": "task-a",
                        "trial_name": "task-a__abc",
                        "exception_info": {
                            "exception_type": "RuntimeError",
                            "exception_message": "Command timed out after 90 seconds",
                            "exception_traceback": (
                                "Traceback (most recent call last):\n"
                                "  File \"bench/network_environment.py\", line 191, "
                                "in _prepare_verifier_runtime\n"
                                "    result = await super().exec(...)\n"
                                "RuntimeError: Command timed out after 90 seconds"
                            ),
                        },
                    }
                ],
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="task-a",
    )

    assert result.status == TrialStatus.ERROR
    assert result.verified is False
    assert result.metadata["timeout_phase"] == "verifier_runtime_prepare"
    assert result.metadata["verifier_runtime_prepare_timeout"] is True
    assert result.metadata["verifier_infra_error"] is True
    assert result.metadata["infra_error_detected"] is True
    assert result.metadata["score_exclusion_reason"] == "infrastructure_error"
    assert HarborRunner().is_infra_error(result) is True
    assert result.metadata["post_completion_agent_exception"] is False
    assert not any("post-completion exception" in error for error in result.error_log)


def test_parse_harbor_job_selected_result_traceback_marks_verifier_runtime_prepare_infra(
    tmp_path,
):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "regex-log__abc"
    trial_dir.mkdir(parents=True)
    (trial_dir / "exception.txt").write_text(
        "Traceback (most recent call last):\n"
        "  File \"/workspace/bench/network_environment.py\", line 191, "
        "in _prepare_verifier_runtime\n"
        "RuntimeError: Command timed out after 90 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "regex-log",
                        "trial_name": "regex-log__abc",
                        "exception_info": {
                            "exception_type": "RuntimeError",
                            "exception_message": "Command timed out after 90 seconds",
                            "exception_traceback": (
                                "Traceback (most recent call last):\n"
                                "  File \"/usr/local/lib/python3.12/dist-packages/harbor/"
                                "verifier/verifier.py\", line 178, in verify\n"
                                "    await self._environment.exec(...)\n"
                                "  File \"/workspace/bench/network_environment.py\", "
                                "line 191, in _prepare_verifier_runtime\n"
                                "RuntimeError: Command timed out after 90 seconds\n"
                            ),
                        },
                    }
                ]
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="regex-log",
    )

    assert result.status == TrialStatus.ERROR
    assert result.verified is False
    assert result.metadata["timeout_phase"] == "verifier_runtime_prepare"
    assert result.metadata["verifier_runtime_prepare_timeout"] is True
    assert result.metadata["verifier_infra_error"] is True
    assert result.metadata["infra_error_detected"] is True
    assert result.metadata["score_exclusion_reason"] == "infrastructure_error"
    assert HarborRunner().is_infra_error(result) is True


def test_harbor_agent_live_trajectory_sink_appends_jsonl(tmp_path):
    agent = HLWorkerHarborAgent(logs_dir=tmp_path / "agent", model_name="test-model")

    path = agent._reset_live_trajectory()
    sink = agent._live_trajectory_sink(path)
    sink({"type": "entrypoint_scan", "success": True})
    sink({"type": "tool_call", "tool": "bash"})

    assert path.read_text().splitlines() == [
        '{"type": "entrypoint_scan", "success": true}',
        '{"type": "tool_call", "tool": "bash"}',
    ]


def test_harbor_agent_passes_custom_llm_provider_to_worker(tmp_path):
    agent = HLWorkerHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="deepseek-ai/DeepSeek-V3.2",
        provider="openai_compatible",
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        custom_llm_provider="openai",
    )

    worker = agent._build_agent(ToolRegistry())

    assert worker.role_config.model == "deepseek-ai/DeepSeek-V3.2"
    assert worker.role_config.extra["custom_llm_provider"] == "openai"


def test_harbor_agent_treats_max_turns_as_audit_only(tmp_path):
    agent = HLWorkerHarborAgent(
        logs_dir=tmp_path / "agent",
        model_name="test-model",
        max_turns=3,
    )

    worker = agent._build_agent(ToolRegistry())

    assert agent.max_turns_audit == 3
    assert worker.max_turns_audit == 0
    assert worker.max_turns == 0
    assert worker._rust_worker_request("task", {"task_id": "task"})["max_turns_audit"] == 0
    assert "max_turns" not in worker._rust_worker_request("task", {"task_id": "task"})


def test_harbor_agent_live_trajectory_reset_removes_stale_events(tmp_path):
    agent = HLWorkerHarborAgent(logs_dir=tmp_path / "agent", model_name="test-model")
    path = agent.logs_dir / "trajectory.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"type":"stale"}\n')

    reset_path = agent._reset_live_trajectory()

    assert reset_path == path
    assert path.read_text() == ""


def test_harbor_agent_cancels_worker_when_coroutine_is_cancelled(monkeypatch, tmp_path):
    class SlowWorker:
        def __init__(self):
            self.cancel_reasons = []
            self.trajectory_event_sink = None
            self._cancelled = threading.Event()

        def run(self, instruction, task_context):
            self._cancelled.wait(5)
            return TrialResult(
                trial_id="cancel-task",
                task_id="cancel-task",
                status=TrialStatus.ERROR,
                score=0.0,
                verified=False,
            )

        def cancel_current_run(self, reason="cancelled"):
            self.cancel_reasons.append(reason)
            self._cancelled.set()

    class FakeEnvironment:
        environment_name = "cancel-task"

    worker = SlowWorker()
    adapter = HLWorkerHarborAgent(logs_dir=tmp_path / "agent", model_name="test-model")
    monkeypatch.setattr(adapter, "_build_agent", lambda registry: worker)
    monkeypatch.setattr(
        adapter,
        "_build_environment_registry",
        lambda environment, loop: ToolRegistry(),
    )

    async def run_and_cancel():
        task = asyncio.create_task(
            adapter.run("instruction", FakeEnvironment(), SimpleNamespace())
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        raise AssertionError("adapter run was not cancelled")

    asyncio.run(run_and_cancel())

    assert worker.cancel_reasons == ["harbor_agent_cancelled"]


def test_parse_incomplete_harbor_job_uses_trial_artifacts_for_timeout_phase(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "task-a__abcdefg"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "trajectory.jsonl").write_text('{"type":"tool_call","tool":"bash"}\n')
    (trial_dir / "exception.txt").write_text(
        "harbor.trial.trial.AgentTimeoutError: "
        "Agent execution timed out after 900.0 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "n_completed_trials": 0,
                    "n_errored_trials": 0,
                    "n_running_trials": 1,
                    "n_cancelled_trials": 1,
                }
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="task-a",
        returncode=-1,
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.harbor_trial_dir == str(trial_dir)
    assert result.trajectory == [{"type": "tool_call", "tool": "bash"}]
    assert result.metadata["timeout_phase"] == "agent_execution"
    assert result.metadata["partial_harbor_artifacts"] is True
    assert result.metadata["job_result_status_counts"]["running"] == 1


def test_outer_harbor_timeout_preserves_partial_artifacts(tmp_path):
    runner = HarborRunner(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "runs")
    job_dir = tmp_path / "jobs" / "job1"
    trial_dir = job_dir / "task-a__abcdefg"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "trajectory.jsonl").write_text('{"type":"tool_call","tool":"bash"}\n')
    (trial_dir / "exception.txt").write_text(
        "harbor.trial.trial.AgentTimeoutError: "
        "Agent execution timed out after 900.0 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps({"stats": {"n_running_trials": 1, "n_cancelled_trials": 1}})
    )

    def fake_run_command(argv, *, timeout_audit):
        raise subprocess.TimeoutExpired(
            argv,
            timeout_audit,
            output="partial stdout",
            stderr="",
        )

    runner._run_command = fake_run_command

    result = runner.run_task(
        "task-a",
        {"agent": "hl-worker", "tool_timeout_seconds": 120},
        timeout=3600,
        job_name="job1",
        jobs_dir=tmp_path / "jobs",
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.metadata["outer_harbor_timeout"] is True
    assert result.metadata["outer_harbor_interrupted"] is True
    assert result.metadata["outer_harbor_timeout_seconds_audit_only"] == 3600
    assert result.metadata["outer_harbor_timeout_stop_condition"] is False
    assert result.metadata["outer_harbor_timeout_loop_stop_condition"] is False
    assert result.metadata["timeout_expired_exception_stop_condition"] is False
    assert result.metadata["time_round_token_limit_driven"] is False
    assert result.metadata["partial_harbor_artifacts"] is True
    assert result.metadata["timeout_phase"] == "agent_execution"
    assert result.trajectory == [{"type": "tool_call", "tool": "bash"}]
    assert "compatibility path" in result.error_log[0]
    assert "audit reference 3600s" in result.error_log[0]


def test_outer_harbor_timeout_without_artifacts_is_audit_compatibility_path(tmp_path):
    runner = HarborRunner(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "runs")
    (tmp_path / "jobs" / "job1").mkdir(parents=True)

    def fake_run_command(argv, *, timeout_audit):
        raise subprocess.TimeoutExpired(
            argv,
            timeout_audit,
            output="partial stdout",
            stderr="",
        )

    runner._run_command = fake_run_command

    result = runner.run_task(
        "task-a",
        {"agent": "hl-worker", "tool_timeout_seconds": 120},
        timeout=9,
        job_name="job1",
        jobs_dir=tmp_path / "jobs",
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.metadata["outer_harbor_timeout_seconds_audit_only"] == 9
    assert result.metadata["outer_harbor_timeout_stop_condition"] is False
    assert result.metadata["outer_harbor_timeout_loop_stop_condition"] is False
    assert result.metadata["timeout_expired_exception_stop_condition"] is False
    assert result.metadata["time_round_token_limit_driven"] is False
    assert result.metadata["partial_harbor_artifacts"] is False
    assert "compatibility path" in result.error_log[0]
    assert "audit reference 9s" in result.error_log[0]


def test_parse_harbor_job_result_classifies_environment_start_timeout(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "reshard-c4-data__abc"
    trial_dir.mkdir(parents=True)
    (trial_dir / "exception.txt").write_text(
        "Traceback (most recent call last):\n"
        "  File \"/usr/local/lib/python3.12/dist-packages/harbor/trial/trial.py\", "
        "line 313, in _setup_environment\n"
        "    await self._start_environment_with_retry()\n"
        "harbor.trial.trial.EnvironmentStartTimeoutError: "
        "Environment start timed out after 600.0 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 1,
                    "n_running_trials": 0,
                    "n_cancelled_trials": 0,
                },
                "trial_results": [
                    {
                        "task_name": "reshard-c4-data",
                        "trial_name": "reshard-c4-data__abc",
                        "exception_info": {
                            "exception_type": "EnvironmentStartTimeoutError",
                            "exception_message": (
                                "Environment start timed out after 600.0 seconds"
                            ),
                        },
                    }
                ],
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="reshard-c4-data",
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.metadata["timeout_phase"] == "environment_start"
    assert HarborRunner().is_infra_error(result) is True


def test_parse_environment_start_timeout_records_image_and_dockerfile_evidence(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "hf-model-inference__abc"
    patched_env = trial_dir / "hl_patched_environment"
    patched_env.mkdir(parents=True)
    (job_dir / "job.log").write_text(
        "Skipping image OS validation for "
        "docker.1panel.live/alexgshaw/hf-model-inference:20251031: "
        "docker inspect returned 1\n"
    )
    (trial_dir / "trial.log").write_text(
        "Skipping image OS validation for "
        "docker.1panel.live/alexgshaw/hf-model-inference:20251031: "
        "docker inspect returned 1\n"
        "Trial hf-model-inference__abc failed: "
        "Environment start timed out after 600.0 seconds\n"
    )
    (patched_env / "Dockerfile").write_text(
        "FROM docker.m.daocloud.io/library/python:3.13-slim-bookworm\n"
        "RUN pip install transformers==4.56.0 torch==2.7.1 flask==3.1.1\n"
    )
    (patched_env / ".hl_apt_mirror.json").write_text(
        json.dumps(
            {
                "patched_files": ["Dockerfile"],
                "docker_hub_mirror": "docker.m.daocloud.io",
                "prebuilt_docker_hub_mirror": "docker.1panel.live",
                "pypi_index_url": "https://mirrors.cloud.tencent.com/pypi/simple/",
                "apt_timeout_seconds": 30,
                "pip_timeout_seconds": 30,
            }
        )
    )
    (trial_dir / "exception.txt").write_text(
        "harbor.trial.trial.EnvironmentStartTimeoutError: "
        "Environment start timed out after 600.0 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
                "trial_results": [
                    {
                        "task_name": "hf-model-inference",
                        "trial_name": "hf-model-inference__abc",
                        "exception_info": {
                            "exception_type": "EnvironmentStartTimeoutError",
                            "exception_message": (
                                "Environment start timed out after 600.0 seconds"
                            ),
                        },
                    }
                ],
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="hf-model-inference",
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.metadata["timeout_phase"] == "environment_start"
    assert result.metadata["docker_image_validation_failed"] is True
    assert result.metadata["docker_image_validation_events"] == [
        {
            "operation": "docker_inspect",
            "image": "docker.1panel.live/alexgshaw/hf-model-inference:20251031",
            "returncode": 1,
        }
    ]
    assert result.metadata["prebuilt_image_cache_miss_detected"] is True
    assert result.metadata["network_preflight_recommended"] is True
    assert result.metadata["prebuilt_image_cache_warmup_commands"] == [
        "docker pull docker.1panel.live/alexgshaw/hf-model-inference:20251031"
    ]
    assert result.metadata["prebuilt_image_cache_warmup_targets"] == [
        {
            "effective_image": "docker.1panel.live/alexgshaw/hf-model-inference:20251031",
            "docker_pull_command": "docker pull docker.1panel.live/alexgshaw/hf-model-inference:20251031",
            "operation": "docker_inspect",
            "returncode": 1,
            "configured_prebuilt_docker_hub_mirror": "docker.1panel.live",
            "original_image": "alexgshaw/hf-model-inference:20251031",
        }
    ]
    assert result.metadata["heavy_dockerfile_install_detected"] is True
    assert result.metadata["heavy_dockerfile_install_steps"][0]["packages"] == [
        "torch",
        "transformers",
    ]
    marker = result.metadata["network_hardened_environment_marker"]
    assert marker["patched_files"] == ["Dockerfile"]
    assert marker["prebuilt_docker_hub_mirror"] == "docker.1panel.live"
    evidence = result.metadata["environment_start_evidence"]
    assert evidence["docker_image_validation"] == result.metadata[
        "docker_image_validation_events"
    ]
    assert evidence["prebuilt_image_cache_warmup"] == result.metadata[
        "prebuilt_image_cache_warmup"
    ]
    assert "prebuilt image inspect failed" in result.metadata[
        "environment_start_attribution_hint"
    ]
    assert "pre-pull/cache-warm prebuilt image" in result.metadata[
        "environment_start_attribution_hint"
    ]
    assert "heavy Dockerfile dependency install" in result.metadata[
        "environment_start_attribution_hint"
    ]


def test_parse_environment_start_timeout_records_environment_config_evidence(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "mteb-retrieve__abc"
    trial_dir.mkdir(parents=True)
    environment_config = {
        "force_build": False,
        "mounts_json": [
            {"type": "bind", "source": "/cert.pem", "target": "/cert.pem"},
            {"type": "bind", "source": "/cache", "target": "/cache"},
        ],
        "import_path": "bench.network_environment:AptMirrorDockerEnvironment",
        "kwargs": {
            "docker_hub_mirror": "docker.m.daocloud.io",
            "prebuilt_docker_hub_mirror": "docker.1panel.live",
            "docker_image_overrides": {"python": "mirror.local/python"},
            "download_url_rewrites": {"https://example.invalid": "https://mirror.local"},
            "pypi_index_url": "https://mirrors.cloud.tencent.com/pypi/simple/",
            "pypi_trusted_host": "mirrors.cloud.tencent.com",
            "apt_retries": 5,
            "apt_timeout_seconds": 30,
            "pip_retries": 5,
            "pip_timeout_seconds": 30,
            "bootstrap_ca_certificates": True,
            "download_retry_wrapper": True,
        },
    }
    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": {"path": "terminal-bench-tasks/terminal-bench"},
                "environment": environment_config,
                "agent": {"env": {"HL_WORKER_API_KEY_ENV": "DEEP****KEY"}},
            }
        )
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": {"task_path": "terminal-bench-tasks/terminal-bench/mteb-retrieve"},
                "environment": environment_config,
            }
        )
    )
    (trial_dir / "trial.log").write_text(
        "Trial mteb-retrieve__abc failed: Environment start timed out after 600.0 seconds\n"
    )
    (trial_dir / "exception.txt").write_text(
        "harbor.trial.trial.EnvironmentStartTimeoutError: "
        "Environment start timed out after 600.0 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
                "trial_results": [
                    {
                        "task_name": "mteb-retrieve",
                        "trial_name": "mteb-retrieve__abc",
                        "exception_info": {
                            "exception_type": "EnvironmentStartTimeoutError",
                            "exception_message": (
                                "Environment start timed out after 600.0 seconds"
                            ),
                        },
                    }
                ],
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="mteb-retrieve",
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.metadata["timeout_phase"] == "environment_start"
    assert HarborRunner().is_infra_error(result) is True
    assert result.metadata["network_hardened_environment_config"] is True
    evidence = result.metadata["environment_start_evidence"]
    config_evidence = evidence["environment_config"]
    assert config_evidence == result.metadata["environment_config_evidence"]
    assert config_evidence["job_config_path"] == str(job_dir / "config.json")
    assert config_evidence["trial_config_path"] == str(trial_dir / "config.json")
    assert (
        config_evidence["environment_import_path"]
        == "bench.network_environment:AptMirrorDockerEnvironment"
    )
    assert config_evidence["force_build"] is False
    assert config_evidence["mounts_json_count"] == 2
    assert config_evidence["task_path"] == "terminal-bench-tasks/terminal-bench"
    kwargs = config_evidence["environment_kwargs"]
    assert kwargs["prebuilt_docker_hub_mirror"] == "docker.1panel.live"
    assert kwargs["docker_hub_mirror"] == "docker.m.daocloud.io"
    assert kwargs["apt_retries"] == 5
    assert kwargs["pip_timeout_seconds"] == 30
    assert "prebuilt_docker_pull_timeout_seconds" not in kwargs
    assert kwargs["download_retry_wrapper"] is True
    assert "agent" not in config_evidence
    assert "DEEP" not in json.dumps(config_evidence)
    assert result.metadata.get("timeout_seconds_stop_condition") is not True
    assert result.metadata.get("round_limit_stop_condition") is not True
    assert result.metadata.get("max_turns_stop_condition") is not True
    hint = result.metadata["environment_start_attribution_hint"]
    assert "network-hardened environment configured" in hint
    assert "prebuilt image mirror configured: docker.1panel.live" in hint
    assert "Docker Hub mirror configured: docker.m.daocloud.io" in hint


def test_parse_environment_start_timeout_records_prebuilt_pull_warmup_timeout(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "hf-model-inference__abc"
    patched_env = trial_dir / "hl_patched_environment"
    patched_env.mkdir(parents=True)
    environment_config = {
        "force_build": False,
        "import_path": "bench.network_environment:AptMirrorDockerEnvironment",
        "kwargs": {
            "docker_hub_mirror": "docker.m.daocloud.io",
            "prebuilt_docker_hub_mirror": "docker.1panel.live",
            "prebuilt_docker_pull_timeout_seconds": 600,
        },
    }
    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": environment_config,
                "dataset": {"path": "terminal-bench-tasks/terminal-bench"},
            }
        )
    )
    (trial_dir / "config.json").write_text(json.dumps({"environment": environment_config}))
    (patched_env / ".hl_apt_mirror.json").write_text(
        json.dumps(
            {
                "patched_files": ["Dockerfile"],
                "docker_hub_mirror": "docker.m.daocloud.io",
                "prebuilt_docker_hub_mirror": "docker.1panel.live",
                "prebuilt_docker_pull_timeout_seconds": 600,
            }
        )
    )
    message = (
        "Prebuilt Docker image cache warmup timed out after 600 seconds for image "
        "docker.1panel.live/alexgshaw/hf-model-inference:20251031. Run `docker pull "
        "docker.1panel.live/alexgshaw/hf-model-inference:20251031` or "
        "`python scripts/network_preflight.py --quick` to diagnose Docker registry access."
    )
    (job_dir / "job.log").write_text(message + "\n")
    (trial_dir / "trial.log").write_text(
        message + "\nTrial hf-model-inference__abc failed: Environment start timed out after 600.0 seconds\n"
    )
    (trial_dir / "exception.txt").write_text(
        "RuntimeError: " + message + "\n"
        "harbor.trial.trial.EnvironmentStartTimeoutError: Environment start timed out after 600.0 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
                "trial_results": [
                    {
                        "task_name": "hf-model-inference",
                        "trial_name": "hf-model-inference__abc",
                        "exception_info": {
                            "exception_type": "EnvironmentStartTimeoutError",
                            "exception_message": (
                                "Environment start timed out after 600.0 seconds"
                            ),
                        },
                    }
                ],
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="hf-model-inference",
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.metadata["timeout_phase"] == "environment_start"
    assert result.metadata["prebuilt_image_cache_miss_detected"] is True
    assert result.metadata["network_preflight_recommended"] is True
    assert result.metadata["docker_image_validation_failed"] is True
    assert result.metadata["docker_image_validation_events"] == [
        {
            "operation": "docker_pull",
            "image": "docker.1panel.live/alexgshaw/hf-model-inference:20251031",
            "timeout_seconds": 600,
        }
    ]
    assert result.metadata["prebuilt_image_cache_warmup_commands"] == [
        "docker pull docker.1panel.live/alexgshaw/hf-model-inference:20251031"
    ]
    assert result.metadata["prebuilt_image_cache_warmup_targets"] == [
        {
            "effective_image": "docker.1panel.live/alexgshaw/hf-model-inference:20251031",
            "docker_pull_command": "docker pull docker.1panel.live/alexgshaw/hf-model-inference:20251031",
            "operation": "docker_pull",
            "timeout_seconds": 600,
            "configured_prebuilt_docker_hub_mirror": "docker.1panel.live",
            "original_image": "alexgshaw/hf-model-inference:20251031",
        }
    ]
    config_evidence = result.metadata["environment_config_evidence"]
    assert config_evidence["environment_kwargs"]["prebuilt_docker_pull_timeout_seconds"] == 600
    marker = result.metadata["network_hardened_environment_marker"]
    assert marker["prebuilt_docker_pull_timeout_seconds"] == 600
    hint = result.metadata["environment_start_attribution_hint"]
    assert "prebuilt image inspect failed" in hint or "prebuilt image" in hint
    assert "pre-pull/cache-warm prebuilt image" in hint


def test_runner_retries_unverified_environment_start_timeout_until_verified(
    tmp_path,
    monkeypatch,
):
    runner = HarborRunner(jobs_dir=tmp_path / "jobs", output_dir=tmp_path / "runs")
    calls = []
    monkeypatch.setattr(harbor_module.time, "sleep", lambda _seconds: None)

    def fake_run_command(argv, *, timeout_audit):
        calls.append(argv)
        job_name = argv[argv.index("--job-name") + 1]
        job_dir = tmp_path / "jobs" / job_name
        trial_name = f"reshard-c4-data__{len(calls)}"
        trial_dir = job_dir / trial_name
        trial_dir.mkdir(parents=True)
        if len(calls) > 2:
            (job_dir / "result.json").write_text(
                json.dumps(
                    {
                        "stats": {"n_completed_trials": 1, "n_errored_trials": 0},
                        "trial_results": [
                            {
                                "task_name": "reshard-c4-data",
                                "trial_name": trial_name,
                                "verifier_result": {"rewards": {"reward": 1.0}},
                            }
                        ],
                    }
                )
            )
            return subprocess.CompletedProcess(argv, 0, "", "")
        (trial_dir / "exception.txt").write_text(
            "harbor.trial.trial.EnvironmentStartTimeoutError: "
            "Environment start timed out after 600.0 seconds\n"
        )
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
                    "trial_results": [
                        {
                            "task_name": "reshard-c4-data",
                            "trial_name": trial_name,
                            "exception_info": {
                                "exception_type": "EnvironmentStartTimeoutError",
                                "exception_message": (
                                    "Environment start timed out after 600.0 seconds"
                                ),
                            },
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    runner._run_command = fake_run_command

    result = runner.run_task(
        "reshard-c4-data",
        {"agent": "hl-worker", "infra_retries": 1},
        job_name="job",
        jobs_dir=tmp_path / "jobs",
    )

    assert len(calls) == 3
    assert result.status == TrialStatus.PASSED
    assert result.verified is True
    assert result.metadata["infra_error_detected"] is False
    assert result.metadata["infra_retries_audit_only"] == 1
    assert result.metadata["infra_retries_stop_condition"] is False
    assert result.metadata["infra_retry_attempt_count_stop_condition"] is False
    assert result.metadata["infra_retry_reference_stop_condition"] is False
    assert result.metadata["infra_retry_loop_stop_condition"] is False
    assert result.metadata["infra_retry_unbounded_by_attempt_count"] is True
    assert result.metadata["time_round_token_limit_driven"] is False
    assert result.metadata["infra_retry_reference_exceeded"] is True
    assert [
        item["status"] for item in result.metadata["infra_retry_attempts"]
    ] == ["timeout", "timeout", "passed"]
    assert [
        item.get("infra_retry_delay_stop_condition")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert [
        item.get("infra_retry_cooldown_stop_condition")
        for item in result.metadata["infra_retry_attempts"][:2]
    ] == [False, False]
    assert [
        item["infra_retry_reference_stop_condition"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, False, False]
    assert [
        item["infra_retry_loop_stop_condition"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [False, False, False]
    assert [
        item["infra_retry_unbounded_by_attempt_count"]
        for item in result.metadata["infra_retry_attempts"]
    ] == [True, True, True]


def test_incomplete_harbor_job_classifies_verifier_cancellation(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "video-processing__abcdefg"
    trial_dir.mkdir(parents=True)
    (trial_dir / "exception.txt").write_text(
        "Traceback (most recent call last):\n"
        "  File \"/usr/local/lib/python3.12/dist-packages/harbor/trial/trial.py\", "
        "line 986, in run\n"
        "    await self._run_verification()\n"
        "  File \"/usr/local/lib/python3.12/dist-packages/harbor/trial/trial.py\", "
        "line 391, in _run_verification\n"
        "    await self._verify_with_retry()\n"
        "asyncio.exceptions.CancelledError\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps({"stats": {"n_running_trials": 1, "n_cancelled_trials": 1}})
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="video-processing",
        returncode=-1,
    )

    assert result.status == TrialStatus.CANCELLED
    assert result.metadata["timeout_phase"] == "verifier"


def test_parse_harbor_job_result_classifies_verifier_runtime_prepare_timeout(tmp_path):
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "video-processing__abcdefg"
    verifier_dir = trial_dir / "verifier"
    patched_env = trial_dir / "hl_patched_environment"
    verifier_dir.mkdir(parents=True)
    patched_env.mkdir(parents=True)
    environment_config = {
        "force_build": False,
        "mounts_json": [
            {
                "type": "bind",
                "source": "/etc/ssl/certs/ca-certificates.crt",
                "target": "/tmp/hl-host-ca/ca-certificates.crt",
                "read_only": True,
            },
            {
                "type": "bind",
                "source": str(tmp_path / "verifier-cache"),
                "target": "/tmp/hl-verifier-cache",
                "read_only": False,
            },
        ],
        "import_path": "bench.network_environment:AptMirrorDockerEnvironment",
        "kwargs": {
            "debian_mirror": "http://mirrors.aliyun.com/debian",
            "docker_hub_mirror": "docker.m.daocloud.io",
            "prebuilt_docker_hub_mirror": "docker.1panel.live",
            "pypi_index_url": "https://mirrors.cloud.tencent.com/pypi/simple/",
            "pypi_trusted_host": "mirrors.cloud.tencent.com",
            "apt_retries": 5,
            "apt_timeout_seconds": 30,
            "pip_retries": 5,
            "pip_timeout_seconds": 30,
            "bootstrap_ca_certificates": True,
            "download_retry_wrapper": True,
        },
    }
    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "environment": environment_config,
                "agent": {"env": {"HL_WORKER_API_KEY_ENV": "DEEP****KEY"}},
            }
        )
    )
    (trial_dir / "config.json").write_text(json.dumps({"environment": environment_config}))
    (patched_env / ".hl_apt_mirror.json").write_text(
        json.dumps(
            {
                "patched_files": ["Dockerfile"],
                "docker_hub_mirror": "docker.m.daocloud.io",
                "prebuilt_docker_hub_mirror": "docker.1panel.live",
                "pypi_index_url": "https://mirrors.cloud.tencent.com/pypi/simple/",
                "apt_timeout_seconds": 30,
                "pip_timeout_seconds": 30,
            }
        )
    )
    (job_dir / "job.log").write_text(
        "Skipping image OS validation for "
        "docker.1panel.live/alexgshaw/video-processing:20251031: "
        "docker inspect returned 1\n"
        "Trial video-processing__abcdefg failed: Command timed out after 90 seconds\n"
    )
    (verifier_dir / "test-stdout.txt").write_text(
        "Verifier runtime network preparation timed out after 90 seconds\n"
        "while running /tmp/hl-verifier-network-prepared setup\n"
    )
    (trial_dir / "exception.txt").write_text(
        "Traceback (most recent call last):\n"
        "  File \"/usr/local/lib/python3.12/dist-packages/harbor/trial/trial.py\", "
        "line 986, in run\n"
        "    await self._run_verification()\n"
        "  File \"/workspace/bench/network_environment.py\", line 188, "
        "in _prepare_verifier_runtime\n"
        "TimeoutError: Verifier runtime network preparation timed out after 90 seconds\n"
    )
    (job_dir / "result.json").write_text(
        json.dumps({"stats": {"n_running_trials": 1, "n_cancelled_trials": 1}})
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="video-processing",
        returncode=-1,
    )

    assert result.status == TrialStatus.TIMEOUT
    assert result.metadata["timeout_phase"] == "verifier_runtime_prepare"
    assert result.metadata["verifier_runtime_prepare_timeout"] is True
    assert result.metadata["verifier_infra_error"] is True
    assert result.metadata["infra_error_detected"] is True
    assert result.metadata["score_exclusion_reason"] == "infrastructure_error"
    assert HarborRunner().is_infra_error(result) is True
    assert result.metadata["verifier_runtime_prepare_timeout_seconds_observed"] == 90
    assert result.metadata["verifier_runtime_prepare_network_hardened_config"] is True
    evidence = result.metadata["verifier_runtime_prepare_evidence"]
    assert evidence["timeout_seconds"] == 90
    assert evidence["prepare_marker_path"] == "/tmp/hl-verifier-network-prepared"
    assert "_prepare_verifier_runtime" in evidence["prepare_function"]
    assert "Command timed out after 90 seconds" in evidence["evidence_text_tail"]
    config_evidence = evidence["environment_config"]
    assert config_evidence == result.metadata["verifier_runtime_prepare_environment_config"]
    assert (
        config_evidence["environment_import_path"]
        == "bench.network_environment:AptMirrorDockerEnvironment"
    )
    assert config_evidence["mounts_json_count"] == 2
    assert config_evidence["environment_kwargs"]["pypi_index_url"] == (
        "https://mirrors.cloud.tencent.com/pypi/simple/"
    )
    assert config_evidence["environment_kwargs"]["apt_timeout_seconds"] == 30
    assert evidence["patched_environment_marker"]["patched_files"] == ["Dockerfile"]
    assert evidence["docker_image_validation"] == [
        {
            "operation": "docker_inspect",
            "image": "docker.1panel.live/alexgshaw/video-processing:20251031",
            "returncode": 1,
        }
    ]
    assert result.metadata["prebuilt_image_cache_miss_detected"] is True
    assert result.metadata["network_preflight_recommended"] is True
    assert result.metadata[
        "verifier_runtime_prepare_prebuilt_image_cache_miss_detected"
    ] is True
    assert result.metadata["prebuilt_image_cache_warmup_commands"] == [
        "docker pull docker.1panel.live/alexgshaw/video-processing:20251031"
    ]
    assert evidence["prebuilt_image_cache_warmup"]["targets"][0][
        "original_image"
    ] == "alexgshaw/video-processing:20251031"
    assert "agent" not in config_evidence
    assert "DEEP" not in json.dumps(evidence)
    hint = result.metadata["verifier_runtime_prepare_attribution_hint"]
    assert "verifier runtime network preparation timed out after 90s" in hint
    assert "runtime network-hardened environment configured" in hint
    assert "runtime PyPI mirror configured" in hint
    assert "pre-pull/cache-warm prebuilt image" in hint
    assert result.metadata.get("validation_timeout_stop_condition") is not True
    assert result.metadata.get("timeout_seconds_stop_condition") is not True
    assert result.metadata.get("round_limit_stop_condition") is not True


def test_parse_harbor_job_result_reads_task_toml_metadata(tmp_path):
    dataset_task = tmp_path / "terminal-bench" / "web-task"
    dataset_task.mkdir(parents=True)
    (dataset_task / "task.toml").write_text(
        """
[metadata]
difficulty = "hard"
category = "web-development"
tags = ["frontend", "debugging"]
expert_time_estimate_min = 10.0
"""
    )
    job_dir = tmp_path / "job1"
    trial_dir = job_dir / "web-task__abc"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "web-task",
                        "trial_name": "web-task__abc",
                        "config": {"task": {"path": str(dataset_task)}},
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    }
                ]
            }
        )
    )

    result = HarborRunner(output_dir=tmp_path / "runs").parse_job_dir(
        job_dir,
        task_id="web-task",
    )

    assert result.task_domain == TaskDomain.WEB_DEVELOPMENT
    assert result.task_difficulty == TaskDifficulty.HARD
    assert result.metadata["task_metadata"]["task_type"] == "frontend"
    assert result.metadata["task_metadata"]["tags"] == ["frontend", "debugging"]


def test_task_catalog_loads_local_task_metadata(tmp_path):
    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
[metadata]
difficulty = "easy"
category = "security"
"""
    )
    catalog = TaskCatalog.from_terminal_bench_path(tmp_path)
    entry = catalog.get("task-a")
    assert entry is not None
    assert entry.domain == "security"
    assert entry.difficulty == "easy"


def test_task_catalog_selects_curriculum_sets(tmp_path):
    _write_task(tmp_path, "sec-easy", category="security", difficulty="easy")
    _write_task(tmp_path, "sec-hard", category="security", difficulty="hard")
    _write_task(tmp_path, "db-easy", category="database", difficulty="easy")
    _write_task(tmp_path, "dev-medium", category="devops", difficulty="medium")

    catalog = TaskCatalog.from_terminal_bench_path(tmp_path)

    assert catalog.select_curriculum("smoke") == [
        "db-easy",
        "sec-easy",
        "dev-medium",
        "sec-hard",
    ]
    assert catalog.select_curriculum("smoke", max_tasks=2) == [
        "db-easy",
        "sec-easy",
        "dev-medium",
        "sec-hard",
    ]
    assert catalog.select_curriculum("hard-focus") == ["sec-hard"]
    assert catalog.select_curriculum("hard-focus", max_tasks=1) == ["sec-hard"]
    assert catalog.select_curriculum("full") == [
        "db-easy",
        "sec-easy",
        "dev-medium",
        "sec-hard",
    ]
    assert catalog.select_curriculum("full", max_tasks=1) == [
        "db-easy",
        "sec-easy",
        "dev-medium",
        "sec-hard",
    ]
    balanced = catalog.select_curriculum("domain-balanced", max_tasks=3)
    assert balanced == ["db-easy", "dev-medium", "sec-easy", "sec-hard"]


def test_task_catalog_selects_random_and_indexed_tasks(tmp_path):
    _write_task(tmp_path, "sec-easy", category="security", difficulty="easy")
    _write_task(tmp_path, "sec-hard", category="security", difficulty="hard")
    _write_task(tmp_path, "db-easy", category="database", difficulty="easy")
    _write_task(tmp_path, "dev-medium", category="devops", difficulty="medium")

    catalog = TaskCatalog.from_terminal_bench_path(tmp_path)

    assert catalog.select_by_indices([1, 3]) == ["db-easy", "dev-medium"]
    first = catalog.select_random(count=2, seed="campaign-a")
    second = catalog.select_random(count=2, seed="campaign-a")
    assert first == second
    assert len(first) == 4
    assert set(first) == set(catalog.select_curriculum("full"))


def test_harbor_verify_tool_executes_inside_environment():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="ok", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            cwd="/app/task",
            timeout_seconds=5,
        )
        result = tool.execute(command="pwd", timeout=3)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert result.output == "ok"
    assert environment.calls == [
        {
            "command": "pwd",
            "cwd": "/app/task",
            "env": None,
            "timeout_sec": 3,
        }
    ]


def test_harbor_shell_caps_requested_timeout_to_policy():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="ok", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="pwd", timeout=30)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert environment.calls[0]["timeout_sec"] == 5
    assert result.metadata["timeout_capped"] is True
    assert "Requested timeout 30s was capped at 5s" in result.output
    assert "Worker operation timeout policy" in result.output
    assert "Worker tool timeout policy" not in result.output
    assert result.metadata["operation_timeout_stop_condition"] is False
    assert result.metadata["timeout_seconds_stop_condition"] is False
    assert result.metadata["loop_stop_condition"] is False


def test_harbor_shell_marks_terminal_environment_unavailable_as_hard_evidence():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=1,
                stdout="",
                stderr='service "main" is not running',
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(environment=environment, loop=loop, timeout_seconds=120)
        result = tool.execute(command="ls /app", timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert "TerminalBench task environment became unavailable" in result.error
    _assert_terminal_environment_metadata(result.metadata)
    assert result.metadata["terminal_environment_marker"] == 'service "main" is not running'
    assert result.metadata["exit_code"] == 1
    assert result.metadata["timeout_seconds"] == 120
    assert len(environment.calls) == 1


def test_harbor_file_append_preserves_terminal_environment_unavailable_evidence():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=1,
                stdout="",
                stderr="container is not running",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(environment=environment, loop=loop, timeout_seconds=5)
        result = tool.execute("/app/out.txt", "more", append=True)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert "TerminalBench task environment became unavailable" in result.error
    assert "staged dependency" not in result.error.lower()
    _assert_terminal_environment_metadata(result.metadata)
    assert result.metadata["terminal_environment_marker"] == "container is not running"
    assert result.metadata["exit_code"] == 1
    assert len(environment.calls) == 1


def test_harbor_file_edit_preserves_terminal_environment_unavailable_evidence():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=1,
                stdout="",
                stderr="no such container: main",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileEditTool(environment=environment, loop=loop, timeout_seconds=5)
        result = tool.execute("/app/out.txt", "old", "new")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert "TerminalBench task environment became unavailable" in result.error
    assert "staged dependency" not in result.error.lower()
    _assert_terminal_environment_metadata(result.metadata)
    assert result.metadata["terminal_environment_marker"] == "no such container"
    assert result.metadata["exit_code"] == 1
    assert len(environment.calls) == 1


def test_harbor_shell_caps_package_manager_timeout_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="ok", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="apt-get install -y gcc", timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert environment.calls[0]["timeout_sec"] == 60
    assert result.metadata["timeout_seconds"] == 60
    assert result.metadata["timeout_capped"] is True
    assert "Package-manager command timeout was capped at 60s" in result.output
    assert "operation-level evidence" in result.output
    assert result.metadata["operation_timeout_stop_condition"] is False
    assert result.metadata["timeout_seconds_stop_condition"] is False
    assert result.metadata["loop_stop_condition"] is False


def test_harbor_shell_timeout_returns_command_telemetry():
    tool = HarborShellTool(
        environment=object(),
        loop=object(),
        timeout_seconds=5,
    )

    def raise_timeout(command, *, timeout=None, env=None):
        exc = TimeoutError("timed out waiting for environment exec")
        exc.stdout = "partial stdout before timeout"
        exc.stderr = "partial stderr before timeout"
        raise exc

    tool._exec = raise_timeout

    result = tool.execute(command="python slow.py", timeout=3)

    assert result.success is False
    assert result.output == "partial stdout before timeout"
    assert result.metadata["timed_out"] is True
    assert result.metadata["timeout_seconds"] == 3
    assert result.metadata["requested_timeout_seconds"] == 3
    assert result.metadata["tool_timeout_telemetry"] is True
    assert result.metadata["tool_timeout_telemetry_source"] == "harbor_shell"
    assert result.metadata["elapsed_ms"] >= 0
    assert result.metadata["stdout_tail"] == "partial stdout before timeout"
    assert result.metadata["stderr_tail"] == "partial stderr before timeout"
    assert result.metadata["operation_timeout_stop_condition"] is False
    assert result.metadata["timeout_seconds_stop_condition"] is False
    assert result.metadata["tool_timeout_telemetry_stop_condition"] is False
    assert result.metadata["loop_stop_condition"] is False


def test_harbor_shell_blocks_background_package_manager_commands():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="started", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="nohup apt-get install -y g++ >/tmp/apt.log 2>&1 &")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "background_package_command_guard"
    _assert_policy_guard_is_non_terminal(result.metadata)


def test_harbor_shell_blocks_host_hl_memory_search_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(command)
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(environment=environment, loop=loop, timeout_seconds=5)
        result = tool.execute("find /trials/runs -name trajectory.jsonl | head -20")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "host_memory_guard"
    assert result.metadata["blocked_reason"] == "host_memory_search"
    assert environment.calls == []


def test_harbor_read_tool_blocks_host_hl_memory_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(command)
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileReadTool(environment=environment, loop=loop, timeout_seconds=5)
        result = tool.execute("/host/trials/runs/old/result.json")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "host_memory_guard"
    assert result.metadata["blocked_reason"] == "host_memory_search"
    assert environment.calls == []


@pytest.mark.parametrize(
    "command",
    [
        (
            "python3 -c \"import urllib.request; "
            "urllib.request.urlopen('https://files.pythonhosted.org/pkg.whl')\""
        ),
        (
            "python3 -c \"import urllib.request; "
            "urllib.request.urlopen('https://mirrors.tuna.tsinghua.edu.cn/pypi/web/"
            "packages/d0/httpstan-4.13.0-cp311-cp311-linux_x86_64.whl')\""
        ),
    ],
)
def test_harbor_shell_blocks_manual_dependency_downloads_before_exec(command):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command=command)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "manual_dependency_download_guard"
    assert "hand-written dependency downloads" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_direct_scripted_package_manager_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            command=(
                "python3 <<'PY'\n"
                "import pip._internal.network.session\n"
                "import pip._internal.cli.main as pip_main\n"
                "pip_main.main()  # --break-system-packages files.pythonhosted.org httpstan\n"
                "PY"
            )
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "scripted_package_manager_guard"
    assert "inline script wraps package-manager" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_package_manager_state_repair_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="dpkg --configure -a 2>&1 | head -20")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "background_package_command_guard"
    assert "broken-state repair" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_heavy_scientific_dependency_installs_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="pip install --trusted-host pypi.org httpstan")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "heavy_scientific_dependency_guard"
    _assert_policy_guard_is_non_terminal(result.metadata)
    assert "heavy scientific/ML dependency installs" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_heavy_graphics_runtime_installs_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="apt-get install -y libgl1 mesa-vulkan-drivers")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "heavy_graphics_runtime_dependency_guard"
    assert "heavy graphics/CV runtime installs" in result.error
    assert "dependency-light CV artifact" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_ubuntu_pool_manual_dependency_downloads_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            command=(
                "curl -s -k 'http://archive.ubuntu.com/ubuntu/pool/universe/r/"
                "r-cran-stanheaders/r-cran-stanheaders_2.32.5-1_amd64.deb' "
                "-o /tmp/r-cran-stanheaders.deb"
            )
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "manual_dependency_download_guard"
    assert environment.calls == []


def test_harbor_shell_blocks_large_toolchain_installs_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="apt-get install -y clang")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "large_toolchain_install_guard"
    assert "large compiler or cross-toolchain installs" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_manual_toolchain_deb_installs_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(
            command=(
                "dpkg -i /var/cache/apt/archives/"
                "binutils-mipsel-linux-gnu_2.40-2cross2_amd64.deb"
            )
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "large_toolchain_install_guard"
    assert "binutils-mipsel-linux-gnu" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_manual_deb_dependency_chasing_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(
            command="dpkg -i /tmp/r-cran-rcppparallel_5.1.10-1_amd64.deb"
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "manual_deb_dependency_chase_guard"
    assert "r-cran-rcppparallel" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_unbounded_root_find_commands():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command='find / -name "*.so" -path "*/numpy/*" 2>/dev/null')
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "broad_root_find_guard"
    assert "system-prefix searches" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_unbounded_system_prefix_find_commands():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command='find /usr -name "*mips*" -type f 2>/dev/null')
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "broad_root_find_guard"
    assert "system-prefix searches" in result.error
    assert environment.calls == []


def test_harbor_shell_blocks_broad_proc_scan_commands():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            command="cat /proc/*/cmdline 2>/dev/null | tr '\\0' ' ' | head -50"
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "broad_proc_scan_guard"
    assert "specific known PID" in result.error
    assert environment.calls == []


@pytest.mark.parametrize(
    "command",
    [
        "codex --help",
        "codex exec --json 'fix this'",
        "command codex --help",
        "sudo -E codex exec --json 'fix this'",
        "sudo --user nobody codex exec --json 'fix this'",
        "env -i codex exec --json 'fix this'",
        "env -u OPENAI_API_KEY codex exec --json 'fix this'",
        "exec codex exec --json 'fix this'",
        "nohup codex exec --json 'fix this'",
        "setsid codex exec --json 'fix this'",
        "bash -lc 'codex exec --json fix'",
        "bash -c codex --help",
        "bash -lc codex --help",
        "bash -o pipefail -c codex --help",
        "gemini --prompt 'fix this'",
        "opencode run 'fix this'",
        "aider --message 'fix this'",
        "amp run 'fix this'",
        "cursor-agent 'fix this'",
        "npx --yes codex exec 'fix this'",
        "pipx run codex exec 'fix this'",
        "uv tool run codex exec 'fix this'",
        "poetry run codex exec 'fix this'",
        "bun x codex exec 'fix this'",
        "python -m codex --help",
        "python -m codex.cli exec 'fix this'",
        "python -m openai.codex exec 'fix this'",
        "python -c \"import subprocess; subprocess.run(['codex', 'exec', 'fix'])\"",
        "python -c \"import subprocess; subprocess.run(['claude-code', '--print', 'fix'])\"",
        "python -c \"import subprocess as sp; sp.run(['codex', 'exec', 'fix'])\"",
        "python -c \"from subprocess import run as rr; rr(['opencode', 'run', 'fix'])\"",
        "python -c \"import subprocess; getattr(subprocess, 'run')(['codex','exec','fix'])\"",
        "python -c \"__import__('subprocess').run(['opencode','run','fix'])\"",
        "python -c \"import subprocess; rr = subprocess.run; rr(['codex','exec','fix'])\"",
        "python -c \"from importlib import import_module; import_module('subprocess').run(['opencode','run','fix'])\"",
        "python -c \"import os; getattr(os, 'system')('codex exec fix')\"",
        "node -e \"require('child_process').spawn('codex',['exec','fix'])\"",
        "node -e \"require('child_process').spawnSync('codex',['exec','fix'])\"",
        "node -e \"const cp = require('child_process'); cp.spawn('codex',['exec','fix'])\"",
        "node -e \"const cp = require('child_process'); cp.execSync('codex exec fix')\"",
        "node -e \"import {spawnSync} from 'child_process'; spawnSync('codex',['exec','fix'])\"",
        "node -e \"import {spawn as s} from 'node:child_process'; s('opencode',['run','fix'])\"",
        "node -e \"import * as cp from 'node:child_process'; cp.execSync('codex exec fix')\"",
        "ruby -e \"spawn 'codex exec fix'\"",
        "ruby -e \"send(:system, 'codex exec fix')\"",
        "f(){ codex exec fix; }; f",
        "alias c='codex exec'; c fix",
        "export c=codex; $c exec fix",
        "env c=codex bash -lc '$c exec fix'",
        "python -c \"import subprocess; cmd = 'cod' + 'ex'; subprocess.run([cmd, 'exec', 'fix'])\"",
        "python -c \"import subprocess; subprocess.run(['co' 'dex', 'exec', 'fix'])\"",
        "python -c \"import subprocess; cmd = ''.join(['co','dex']); subprocess.run([cmd, 'exec', 'fix'])\"",
        "python -c \"import subprocess; cmd = f'codex'; subprocess.run([cmd, 'exec', 'fix'])\"",
        "python -c \"import subprocess; cmd = chr(99)+chr(111)+chr(100)+chr(101)+chr(120); subprocess.run([cmd, 'exec', 'fix'])\"",
        "python -c \"import subprocess; cmd = bytes([99,111,100,101,120]).decode(); subprocess.run([cmd, 'exec', 'fix'])\"",
        "python -c \"import subprocess; cmd = ('codex').replace('x','x'); subprocess.run([cmd, 'exec', 'fix'])\"",
        "python -c \"import subprocess; cmd = 'factory'; subprocess.run([cmd, 'mission', 'run'])\"",
        "python -c \"import subprocess; cmd = 'droid'; subprocess.run([cmd, 'mission', 'run'])\"",
        "node -e \"const c = ['co','dex'].join(''); require('child_process').spawn(c, ['exec','fix'])\"",
        "node -e \"const c = 'factory'; require('child_process').spawn(c, ['mission','run'])\"",
        "node -e \"const c = String.fromCharCode(99,111,100,101,120); require('child_process').spawn(c, ['exec','fix'])\"",
        "node -e \"const c = 'cod' + 'ex'; require('child_process').spawn(c, ['exec','fix'])\"",
        "node -e \"require('child_process').spawn('factory',['mission','run'])\"",
        "node -e \"const cp = require('child_process'); cp.spawn('droid',['mission','run'])\"",
        "ruby -e \"c=['co','dex'].join; spawn c, 'exec', 'fix'\"",
        "ruby -e \"c='cod'+'ex'; spawn c, 'exec', 'fix'\"",
        "ruby -e \"c=%q{codex}; spawn c, 'exec', 'fix'\"",
        "ruby -e \"c='droid'; spawn c, 'mission', 'run'\"",
        "ruby -e \"spawn 'factory mission run'\"",
        "ruby -e \"send(:system, 'droid mission run')\"",
        "lua -e \"os.execute('codex exec fix')\"",
        "php -r \"exec('codex exec fix');\"",
        "printf '#!/bin/sh\\ncodex exec fix\\n' > /tmp/run_agent.sh",
        "python -c \"from pathlib import Path; Path('/tmp/run_agent.sh').write_text('codex exec fix')\"",
    ],
)
def test_harbor_shell_blocks_nested_external_agent_creation_before_exec(command):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(environment=environment, loop=loop, timeout_seconds=120)
        result = tool.execute(command=command, timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "nested_sub_agent_creation_guard"
    assert result.metadata["sub_agent_creation_guard"] is True
    assert result.metadata["nested_sub_agent_creation_allowed"] is False
    assert result.metadata["only_master_loop_may_create_sub_agents"] is True
    _assert_policy_guard_is_non_terminal(result.metadata)
    assert "only the master HL orchestrator may create sub-agents" in result.error
    assert environment.calls == []


def test_harbor_shell_detects_package_failure_hidden_by_pipeline():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "Could not fetch URL https://pypi.org/simple/numpy/: "
                    "certificate verify failed\n"
                    "ERROR: Could not find a version that satisfies the requirement numpy\n"
                    "ERROR: No matching distribution found for numpy\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="pip install numpy 2>&1 | tail -5", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "package_manager_failure"
    assert "package manager output indicates failure" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_network_probe_missing_tool_hidden_by_pipeline():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout="bash: line 1: ping: command not found\n",
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            command="ping -c 1 -W 2 deb.debian.org 2>&1 | head -5",
            timeout=5,
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "network_probe_tool_missing"
    assert "network probe output indicates" in result.error
    assert "package manager output" not in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_dependency_setup_sigkill_as_package_failure():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=137, stdout="building wheel\n", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="pip install numpy", timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 137
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "package_manager_failure"
    assert "package manager output indicates failure" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_blocks_heavy_scientific_sigkill_setup_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=137, stdout="installing to library\n", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="R CMD INSTALL /tmp/rstan_2.32.7.tar.gz", timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "heavy_scientific_dependency_guard"
    _assert_policy_guard_is_non_terminal(result.metadata)
    assert "heavy scientific/ML dependency installs" in result.error
    assert environment.calls == []


def test_harbor_shell_detects_masked_build_failure_hidden_by_success_suffix():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "cc -o solver solver.c\n"
                    "solver.c:3:10: fatal error: missing.h: No such file or directory\n"
                    "compilation terminated.\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="make all 2>&1 || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "masked_build_test_failure"
    assert "build/test output indicates failure" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_threshold_failure_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "Verification of my_warrior.red:\n"
                    "=== All opponent tests:\n"
                    "  stone: 69 wins (Results: 69 29 2)\n"
                    "  paper: 93 wins (Results: 93 0 7)\n"
                    "  vampire: 90 wins (Results: 90 7 3)\n"
                    "  snake: 50 wins (Results: 50 40 10)\n"
                    "  g2-clear: 68 wins (Results: 68 30 2)\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="cd /app && ./local-check.sh", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "corewar_threshold_failure"
    assert "verification output indicates an unmet threshold" in result.error
    assert "stone: 69 wins" in result.output
    assert len(environment.calls) == 1


def test_harbor_verify_detects_explicit_failed_threshold_line():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout="Forward annealing length 0: FAIL (need 15-45)\n",
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python check_lengths.py", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["semantic_failure_kind"] == "verification_threshold_failure"
    assert "unmet threshold" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_regex_backreference_failure_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "fen = re.sub(pattern, repl, fen)\n"
                    "E           re.PatternError: invalid group reference 10 at position 19\n"
                    "Error at pair 6174: invalid group reference 2 at position 5\n"
                    "  Pattern: '(?m) (w|b) [^-]+ [a-h][36] '\n"
                    "  Replacement: ' \\1 \\2 - '\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 /app/check_regex_rules.py || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert (
        result.metadata["semantic_failure_kind"]
        == "regex_replacement_backreference_failure"
    )
    assert "invalid Python re.sub replacement backreference" in result.error
    assert "invalid group reference 10" in result.output
    assert len(environment.calls) == 1


def test_harbor_shell_does_not_apply_verify_threshold_semantics():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout="Forward annealing length 0: FAIL (need 15-45)\n",
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="cat prior-check.log", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert "semantic_failure_detected" not in result.metadata
    assert len(environment.calls) == 1


def test_harbor_shell_detects_large_toolchain_plan_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=137,
                stdout=(
                    "The following NEW packages will be installed:\n"
                    "  clang clang-14 libclang-cpp14 libllvm14 llvm-14 llvm-14-dev\n"
                    "101 newly installed, 0 to remove and 38 not upgraded.\n"
                    "Need to get 161 MB of archives.\n"
                    "After this operation, 884 MB of additional disk space will be used.\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="printf cached apt output", timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 137
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "large_toolchain_install_plan"
    assert "large compiler/toolchain install plan" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_large_graphics_runtime_plan_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "The following NEW packages will be installed:\n"
                    "  libdrm-amdgpu1 libdrm-common libdrm-intel1 libdrm2 libgbm1 libgl1\n"
                    "  libgl1-mesa-dri libglvnd0 libglx-mesa0 libglx0 libllvm19 libpciaccess0\n"
                    "  libsensors-config libsensors5 libvulkan1 libwayland-client0\n"
                    "  libwayland-server0 libx11-xcb1 libxcb-dri3-0 libxcb-glx0\n"
                    "  libxcb-present0 libxcb-randr0 libxcb-sync1 libxcb-xfixes0\n"
                    "  libxshmfence1 libxxf86vm1 libz3-4 mesa-libgallium mesa-vulkan-drivers\n"
                    "0 upgraded, 29 newly installed, 0 to remove and 118 not upgraded.\n"
                    "Need to get 60.1 MB of archives.\n"
                    "After this operation, 289 MB of additional disk space will be used.\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="printf cached apt output", timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "large_graphics_runtime_install_plan"
    assert "large graphics/CV runtime install plan" in result.error
    assert "visible output contract" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_heavy_ml_cv_import_failure_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "/usr/local/lib/python3.11/site-packages/cv2/__init__.py: line 181\n"
                    "ImportError: libGL.so.1: cannot open shared object file: "
                    "No such file or directory\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -c 'import cv2' || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "heavy_ml_cv_import_failure"
    assert "heavy ML/CV import output" in result.error
    assert "dependency-light artifact" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_numpy_eigensolver_failure_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "from eigen import find_dominant_eigenvalue_and_eigenvector\n"
                    "numpy._core._exceptions._UFuncOutputCastingError: Cannot cast "
                    "ufunc 'subtract' output from dtype('complex128') to "
                    "dtype('float64') with casting rule 'same_kind'\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            command="python3 -c 'from eigen import find_dominant_eigenvalue_and_eigenvector' || true",
            timeout=5,
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "numpy_eigensolver_failure"
    assert "NumPy eigensolver output" in result.error
    assert "eigen.py" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_numpy_eigensolver_speed_failure_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "test_outputs.py::test_speedup\n"
                    "from eigen import find_dominant_eigenvalue_and_eigenvector\n"
                    ">       assert dt < ref_dt, f\"{dt:.6f} seconds/call > {ref_dt:.6f} seconds/call\"\n"
                    "E       AssertionError: 0.000026 seconds/call > 0.000025 seconds/call\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            command="python3 -m pytest /tests/test_outputs.py || true",
            timeout=5,
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert (
        result.metadata["semantic_failure_kind"]
        == "numpy_eigensolver_speed_threshold_failure"
    )
    assert "slower than the reference" in result.error
    assert "sizes 2-10" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_numpy_eigensolver_speed_failure_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "test_outputs.py::test_speedup\n"
                    "size = 8\n"
                    "from eigen import find_dominant_eigenvalue_and_eigenvector\n"
                    ">       assert dt < ref_dt, f\"{dt:.6f} seconds/call > {ref_dt:.6f} seconds/call\"\n"
                    "E       AssertionError: 0.000026 seconds/call > 0.000025 seconds/call\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert (
        result.metadata["semantic_failure_kind"]
        == "numpy_eigensolver_speed_threshold_failure"
    )
    assert "slower than the reference" in result.error
    assert "timing harness" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_single_file_deliverable_directory_failure_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_fibonacci_polyglot():\n"
                    "    # Verify that only main.rs exists in /app/polyglot\n"
                    "    polyglot_files = os.listdir(\"/app/polyglot\")\n"
                    ">       assert polyglot_files == [\"main.rs\"], (\n"
                    "E       AssertionError: Expected only main.rs, found: ['test_poly.rs', 'main.rs', 'cmain']\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert (
        result.metadata["semantic_failure_kind"]
        == "single_file_deliverable_directory_contract"
    )
    assert "single-file deliverable directory" in result.error
    assert "os.listdir exact-file-list" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_single_file_deliverable_directory_missing_dir_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_fibonacci_polyglot():\n"
                    ">       polyglot_files = os.listdir(\"/app/polyglot\")\n"
                    "E       FileNotFoundError: [Errno 2] No such file or directory: '/app/polyglot'\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == (
        "single_file_deliverable_directory_contract"
    )
    assert "create /app/polyglot early" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_gpt2_codegolf_size_failure_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_gpt2_implementation():\n"
                    "    gpt2_path = Path(\"/app/gpt2.c\")\n"
                    "E       AssertionError: File /app/gpt2.c is larger than 5000 bytes\n"
                    "E       assert 8394 < 5000\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "gpt2_codegolf_text_contract"
    assert "GPT2 codegolf text contract" in result.error
    assert "under 5000 bytes" in result.error
    assert "WARRANTY OF ANY KIND, EXPRESS OR IMPLIED" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_gpt2_codegolf_timeout_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_gpt2_implementation():\n"
                    "    gpt2_path = Path(\"/app/gpt2.c\")\n"
                    "    run_result = subprocess.run([\"/app/a.out\", \"gpt2-124M.ckpt\", "
                    "\"vocab.bpe\", 'THIS SOFTWARE IS PROVIDED \"AS IS\", WITHOUT'], "
                    "capture_output=True, text=True, timeout=90)\n"
                    "E       subprocess.TimeoutExpired: Command '['/app/a.out', "
                    "'gpt2-124M.ckpt', 'vocab.bpe', 'THIS SOFTWARE IS PROVIDED "
                    "\"AS IS\", WITHOUT']' timed out after 90 seconds\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "gpt2_codegolf_text_contract"
    assert "GPT2 codegolf text contract" in result.error
    assert "90s timeout" in result.error
    assert "valid UTF-8 continuation text" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_structured_csv_table_failure_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_summary_csv_content():\n"
                    "    summary_file = Path('/app/invoices/summary.csv')\n"
                    "    df = pd.read_csv(summary_file)\n"
                    ">       assert len(df) == len(expected_data), 'Expected 11 rows'\n"
                    "E       AssertionError: Expected 11 rows\n"
                    "E       assert 10 == 11\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "structured_csv_table_contract"
    assert "structured CSV/table contract" in result.error
    assert "pd.read_csv" in result.error
    assert "header/column order" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_structured_csv_table_failure_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_cell_metadata_csv():\n"
                    "    df = pd.read_csv(args.csv_path)\n"
                    "    for _, row in df.iterrows():\n"
                    ">       assert row['cell_id'] in expected_data\n"
                    "E       AssertionError: unexpected row key\n"
                    "E       assert 'cell-17' in expected_data\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "structured_csv_table_contract"
    assert "structured CSV/table contract" in result.error
    assert "key or identifier values" in result.error
    assert "expected keyed row content" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_dna_insert_primer_pair_failure_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_primers():\n"
                    "    primers_path = Path('/app/primers.fasta')\n"
                    "    assert len(lines) == 4, 'Invalid number of lines in primers.fasta.'\n"
                    "    fwd_primer = lines[1].lower()\n"
                    "    rev_primer = lines[3].lower()\n"
                    "    primers_concat = rc(rev_primer) + fwd_primer\n"
                    "    insert_start = primers_concat.find(insert)\n"
                    ">   assert insert_start != -1, 'Primer must contain inserted DNA.'\n"
                    "E   AssertionError: Primer must contain inserted DNA.\n"
                    "Forward annealing length 0: FAIL (need 15-45)\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "dna_insert_primer_pair_contract"
    assert "DNA insert primer-pair contract" in result.error
    assert "focused parser" in result.error
    assert "primer3/toolchain/package expansion" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_dna_assembly_primer_failure_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_primers():\n"
                    "    primers_path = Path(\"/app/primers.fasta\")\n"
                    "    assert len(lines) == 16, \"Invalid number of lines in primers.fasta.\"\n"
                    "    assert all(k in primers for k in [\"input_fwd\", \"input_rev\", "
                    "\"egfp_fwd\", \"egfp_rev\", \"flag_fwd\", \"flag_rev\", "
                    "\"snap_fwd\", \"snap_rev\"])\n"
                    "    def parse_bsai_primer(primer):\n"
                    "        \"\"\"Primer (5'->3'): [clamp] ggtctc [oooo] [binding]\"\"\"\n"
                    "        site = \"ggtctc\"\n"
                    "        i = primer.find(site)\n"
                    ">       assert i >= 1, \"Primer must have clamp of at least 1 nucleotide before BsaI site.\"\n"
                    "E       AssertionError: Primer must have clamp of at least 1 nucleotide before BsaI site.\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "dna_assembly_primer_contract"
    assert "DNA assembly primer contract" in result.error
    assert "four-base overhang" in result.error
    assert "parse_bsai_primer/make_fragment" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_missing_output_artifact_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_model_exists():\n"
                    "    model_path = Path('/app/model.bin')\n"
                    ">       assert model_path.exists(), f'File {model_path} does not exist'\n"
                    "E       AssertionError: File /app/model.bin does not exist\n"
                    "E        + where exists = PosixPath('/app/model.bin').exists\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py || true", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "missing_output_artifact_contract"
    assert "missing output artifact contract" in result.error
    assert "verifier-named /app artifact path" in result.error
    assert "format-specific parser" in result.error
    assert len(environment.calls) == 1


def test_harbor_verify_detects_missing_output_artifact_hidden_by_success_exit():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "def test_compressed_file_exists():\n"
                    "    compressed_path = Path(\"/app/data.comp\")\n"
                    ">       assert compressed_path.exists(), f\"File {compressed_path} does not exist\"\n"
                    "E       AssertionError: File /app/data.comp does not exist\n"
                    "E        + where exists = PosixPath('/app/data.comp').exists\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborVerifyTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(command="python3 -m pytest /tests/test_outputs.py", timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "missing_output_artifact_contract"
    assert "missing output artifact contract" in result.error
    assert "test -s" in result.error
    assert "artifact-wide searches" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_large_package_install_plan_hidden_by_shell_status():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "The following NEW packages will be installed:\n"
                    "  r-cran-rstan r-cran-stanheaders r-cran-rcppparallel r-cran-rcppeigen\n"
                    "327 newly installed, 0 to remove and 12 not upgraded.\n"
                    "Need to get 412 MB of archives.\n"
                    "After this operation, 2191 MB of additional disk space will be used.\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=120,
        )
        result = tool.execute(command="printf cached apt output", timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "large_package_install_plan"
    assert "large transitive package install plan" in result.error
    assert "dependency-free implementation" in result.error
    assert len(environment.calls) == 1


def test_harbor_shell_detects_packaging_backend_failure_hidden_by_pipeline():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    "pip._vendor.pyproject_hooks._impl.BackendUnavailable: "
                    "Cannot import 'setuptools.build_meta'\n"
                ),
                stderr="",
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborShellTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            command="cd /tmp/fasttext && python3 -m pip install . 2>&1 | tail -20",
            timeout=5,
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["semantic_failure_detected"] is True
    assert result.metadata["semantic_failure_kind"] == "package_manager_failure"
    assert "package manager output indicates failure" in result.error
    assert len(environment.calls) == 1


def test_harbor_write_tool_does_not_require_target_python(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        path = tmp_path / "nested" / "result.txt"
        result = tool.execute(str(path), "line 1\nquoted '$PATH'\n")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert path.read_text() == "line 1\nquoted '$PATH'\n"
    assert environment.calls
    assert "python3" not in environment.calls[0]["command"]
    assert "base64" in environment.calls[0]["command"]


def test_harbor_write_tool_blocks_staged_dependency_script_before_exec(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            str(tmp_path / "download_httpstan.py"),
            (
                "import urllib.request\n"
                "import ssl\n"
                "ctx = ssl._create_unverified_context()\n"
                "urllib.request.urlopen('https://pypi.org/pypi/httpstan/4.13.0/json', context=ctx)\n"
            ),
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
    _assert_policy_guard_is_non_terminal(result.metadata)
    assert "staged script" in result.error
    assert environment.calls == []


def test_harbor_write_tool_blocks_oversized_gpt2_codegolf_before_exec(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(str(tmp_path / "app" / "gpt2.c"), "x" * 5000)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
    _assert_policy_guard_is_non_terminal(result.metadata)
    assert result.metadata["content_bytes"] == 5000
    assert result.metadata["limit_bytes"] == 5000
    assert "under 5000 bytes" in result.error
    assert environment.calls == []


def test_harbor_write_tool_blocks_staged_dependency_script_after_append(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    path = tmp_path / "download_httpstan.py"
    original = (
        "#!/usr/bin/env python3\n"
        "import ssl\n"
        "import urllib.request\n"
        "ctx = ssl._create_unverified_context()\n"
        "def fetch(url):\n"
        "    return urllib.request.urlopen(url, context=ctx, timeout=30)\n"
        "url = "
    )
    path.write_text(original)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            str(path),
            "'https://pypi.org/pypi/httpstan/4.13.0/json'\nfetch(url)\n",
            append=True,
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
    assert "hand-written dependency downloads" in result.error
    assert path.read_text() == original
    assert len(environment.calls) == 1
    assert environment.calls[0]["command"].startswith("python3 ")


def test_harbor_write_tool_blocks_gpt2_codegolf_append_after_composed_size(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    path = tmp_path / "app" / "gpt2.c"
    path.parent.mkdir()
    original = "x" * 4990
    path.write_text(original)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(str(path), "y" * 20, append=True)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
    assert result.metadata["content_bytes"] == 5010
    assert path.read_text() == original
    assert len(environment.calls) == 1
    assert environment.calls[0]["command"].startswith("python3 ")


def test_harbor_write_tool_blocks_host_hl_memory_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(command)
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute("/host/trials/runs/old/result.json", "{}")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "host_memory_guard"
    assert result.metadata["blocked_reason"] == "host_memory_search"
    assert environment.calls == []


def test_harbor_read_tool_falls_back_when_target_lacks_python(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            if command.startswith("python3 "):
                return SimpleNamespace(
                    return_code=127,
                    stdout="",
                    stderr="sh: 1: python3: not found\n",
                )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    path = tmp_path / "input.txt"
    path.write_text("alpha\nbeta\n")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileReadTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(str(path), offset=1, limit=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert "1\talpha" in result.output
    assert "2\tbeta" in result.output
    assert len(environment.calls) == 2
    assert environment.calls[0]["command"].startswith("python3 ")
    assert "awk" in environment.calls[1]["command"]


def test_harbor_read_tool_falls_back_when_missing_python_is_stdout(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            if command.startswith("python3 "):
                return SimpleNamespace(
                    return_code=127,
                    stdout="bash: line 1: python3: command not found\n",
                    stderr="",
                )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    path = tmp_path / "university_graph.ttl"
    path.write_text("@prefix ex: <http://example.test/> .\nex:a ex:b ex:c .\n")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileReadTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(str(path), offset=1, limit=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert "1\t@prefix ex:" in result.output
    assert "2\tex:a ex:b ex:c ." in result.output
    assert len(environment.calls) == 2
    assert environment.calls[0]["command"].startswith("python3 ")
    assert "awk" in environment.calls[1]["command"]


def test_harbor_glob_tool_falls_back_when_target_lacks_python(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            if command.startswith("python3 "):
                return SimpleNamespace(
                    return_code=127,
                    stdout="",
                    stderr="sh: 1: python3: not found\n",
                )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "main.c"
    target.write_text("int main(void) { return 0; }\n")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborGlobTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute("*.c", path=str(tmp_path))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert str(target) in result.output
    assert len(environment.calls) == 2
    assert environment.calls[0]["command"].startswith("python3 ")
    assert "find" in environment.calls[1]["command"]


def test_harbor_edit_tool_falls_back_when_target_lacks_python(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            if command.startswith("python3 "):
                return SimpleNamespace(
                    return_code=127,
                    stdout="",
                    stderr="sh: 1: python3: not found\n",
                )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    path = tmp_path / "program.c"
    path.write_text("return 1;\n")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileEditTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(str(path), "return 1;", "return 0;")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is True
    assert "replaced 1 occurrence" in result.output
    assert path.read_text() == "return 0;\n"
    assert len(environment.calls) == 4
    assert environment.calls[0]["command"].startswith("python3 ")
    assert "cat" in environment.calls[1]["command"]
    assert environment.calls[2]["command"].startswith("python3 ")
    assert "perl" in environment.calls[3]["command"]


def test_harbor_edit_tool_blocks_staged_dependency_script_before_exec(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileEditTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            str(tmp_path / "download.py"),
            "old",
            (
                "import ssl\n"
                "import pip._internal.network.session\n"
                "import pip._internal.cli.main as pip_main\n"
                "pip_main.main()  # --break-system-packages files.pythonhosted.org\n"
            ),
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
    assert "staged script" in result.error
    assert environment.calls == []


def test_harbor_write_tool_blocks_staged_nested_agent_script_before_exec(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileWriteTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            str(tmp_path / "delegate.js"),
            "import {spawnSync} from 'child_process'; spawnSync('codex', ['exec', 'fix'])\n",
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
    _assert_policy_guard_is_non_terminal(result.metadata)
    assert "only the master HL orchestrator may create sub-agents" in result.error
    assert environment.calls == []


def test_harbor_edit_tool_blocks_staged_dependency_script_after_composed_edit(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    path = tmp_path / "download_httpstan.py"
    original = (
        "#!/usr/bin/env python3\n"
        "import ssl\n"
        "import urllib.request\n"
        "ctx = ssl._create_unverified_context()\n"
        "url = PLACEHOLDER\n"
        "urllib.request.urlopen(url, context=ctx, timeout=30)\n"
    )
    path.write_text(original)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileEditTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(
            str(path),
            "PLACEHOLDER",
            "'https://pypi.org/pypi/httpstan/4.13.0/json'",
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
    assert "hand-written dependency downloads" in result.error
    assert path.read_text() == original
    assert len(environment.calls) == 1
    assert environment.calls[0]["command"].startswith("python3 ")


def test_harbor_edit_tool_blocks_gpt2_codegolf_size_cap_after_composed_edit(tmp_path):
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    path = tmp_path / "app" / "gpt2.c"
    path.parent.mkdir()
    original = "int main(void){return 0;}\n"
    path.write_text(original)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileEditTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute(str(path), "return 0;", "x" * 5000)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
    assert result.metadata["content_bytes"] > 5000
    assert "under 5000 bytes" in result.error
    assert path.read_text() == original
    assert len(environment.calls) == 1
    assert environment.calls[0]["command"].startswith("python3 ")


def test_harbor_edit_tool_blocks_host_hl_memory_before_exec():
    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.calls.append(command)
            return SimpleNamespace(return_code=0, stdout="should not run", stderr="")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        environment = FakeEnvironment()
        tool = HarborFileEditTool(
            environment=environment,
            loop=loop,
            timeout_seconds=5,
        )
        result = tool.execute("/host/trials/runs/old/result.json", "old", "new")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert result.success is False
    assert result.metadata["blocked_by"] == "host_memory_guard"
    assert result.metadata["blocked_reason"] == "host_memory_search"
    assert environment.calls == []


def test_harbor_runner_command_timeout_is_audit_only(monkeypatch):
    runner = HarborRunner()
    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, *args, **kwargs):
            captured["communicate_args"] = args
            captured["communicate_kwargs"] = kwargs
            return "stdout", "stderr"

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(harbor_module.subprocess, "Popen", fake_popen)

    completed = runner._run_command(
        [sys.executable, "-c", "print('ok')"],
        timeout_audit=1,
    )

    assert completed.returncode == 0
    assert completed.stdout == "stdout"
    assert captured["communicate_args"] == ()
    assert captured["communicate_kwargs"] == {}


def _write_task(tmp_path, task_id: str, *, category: str, difficulty: str) -> None:
    task_dir = tmp_path / task_id
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        f"""
[metadata]
difficulty = "{difficulty}"
category = "{category}"
"""
    )
