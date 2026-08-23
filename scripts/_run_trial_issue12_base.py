#!/usr/bin/env python3
"""Run one HL Worker trial through Harbor."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.network_environment import (  # noqa: E402
    DEFAULT_DOCKER_CPUS,
    DEFAULT_DOCKER_LABELS,
    DEFAULT_DOCKER_LOG_MAX_FILE,
    DEFAULT_DOCKER_LOG_MAX_SIZE,
    DEFAULT_DOCKER_MEMORY,
    DEFAULT_DOCKER_MEMORY_SWAP,
    DEFAULT_DOCKER_PIDS_LIMIT,
)
from harness.config import ModelsConfig


DISALLOWED_TIMEOUT_RESOURCE_FIELDS = {
    "timeout_multiplier": "--timeout-multiplier",
    "agent_timeout_multiplier": "--agent-timeout-multiplier",
    "verifier_timeout_multiplier": "--verifier-timeout-multiplier",
    "agent_setup_timeout_multiplier": "--agent-setup-timeout-multiplier",
    "environment_build_timeout_multiplier": "--environment-build-timeout-multiplier",
}
DISALLOWED_RESOURCE_CONFIG_FIELDS = (
    "storage",
    "storage_gb",
    "resources",
    "override_storage_mb",
    "override_gpus",
)
OFFICIAL_LIMITS_ERROR = (
    "Terminal-Bench 2.0 leaderboard runs must keep official task "
    "timeouts/resources unchanged. Remove {source}; timeout failures must be "
    "fixed by Worker strategy within the task's official limits."
)


T = TypeVar("T")


def _prefer_explicit(value: T | None, fallback: T | None) -> T | None:
    """Use a fallback only when the higher-precedence value is absent."""

    return fallback if value is None else value


def add_docker_resource_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--docker-resource-enabled",
        dest="docker_resource_enabled",
        action="store_true",
        default=None,
        help="Enable local Docker memory/cpu/pid caps for Harbor containers.",
    )
    parser.add_argument(
        "--no-docker-resource-limits",
        dest="docker_resource_enabled",
        action="store_false",
        help="Disable local Docker resource caps. Unsafe for concurrent runs.",
    )
    parser.add_argument("--docker-memory", default=None, help="Per-container memory cap, e.g. 4g")
    parser.add_argument(
        "--docker-memory-swap",
        default=None,
        help="Per-container memory+swap cap; set equal to --docker-memory to disable extra swap.",
    )
    parser.add_argument("--docker-cpus", type=int, default=None, help="Per-container CPU cap")
    parser.add_argument("--docker-pids-limit", type=int, default=None, help="Per-container pids limit")
    parser.add_argument(
        "--docker-label",
        action="append",
        default=None,
        help="Extra Docker label as KEY=VALUE; repeatable.",
    )
    parser.add_argument("--docker-log-max-size", default=None, help="json-file max-size, e.g. 20m")
    parser.add_argument("--docker-log-max-file", default=None, help="json-file max-file count")


def docker_resource_forward_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    if hasattr(args, "docker_resource_enabled"):
        if _docker_resource_enabled(args):
            forwarded.append("--docker-resource-enabled")
        else:
            forwarded.append("--no-docker-resource-limits")
    for flag, attr in [
        ("--docker-memory", "docker_memory"),
        ("--docker-memory-swap", "docker_memory_swap"),
        ("--docker-cpus", "docker_cpus"),
        ("--docker-pids-limit", "docker_pids_limit"),
        ("--docker-log-max-size", "docker_log_max_size"),
        ("--docker-log-max-file", "docker_log_max_file"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            forwarded.extend([flag, str(value)])
    for label in getattr(args, "docker_label", None) or []:
        forwarded.extend(["--docker-label", str(label)])
    return forwarded


def docker_run_resource_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
    *,
    include_rm: bool = True,
) -> list[str]:
    flags: list[str] = ["--rm"] if include_rm else []
    if not _docker_resource_enabled(args):
        return flags
    for flag, attr in [
        ("--memory", "docker_memory"),
        ("--memory-swap", "docker_memory_swap"),
        ("--cpus", "docker_cpus"),
        ("--pids-limit", "docker_pids_limit"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            flags.extend([flag, str(value)])
    try:
        labels = _parse_docker_label_items(getattr(args, "docker_label", None))
    except ValueError as exc:
        if parser is not None:
            parser.error(str(exc))
        raise
    for key, value in labels.items():
        flags.extend(["--label", f"{key}={value}"])
    log_max_size = getattr(args, "docker_log_max_size", None)
    log_max_file = getattr(args, "docker_log_max_file", None)
    if log_max_size or log_max_file:
        flags.extend(["--log-driver", "json-file"])
    if log_max_size:
        flags.extend(["--log-opt", f"max-size={log_max_size}"])
    if log_max_file:
        flags.extend(["--log-opt", f"max-file={log_max_file}"])
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an HL Worker trial through Harbor")
    parser.add_argument("--path", default="terminal-bench-tasks/terminal-bench")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task", required=False, help="Task name to run")
    parser.add_argument("--all", action="store_true", help="List/run all tasks is not default")
    parser.add_argument("--agent", default="hl-worker")
    parser.add_argument("--worker-role", default=None, help="Model role to use from config roles")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--reasoning-max-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", default=None)
    parser.add_argument("--llm-timeout-seconds", type=int, default=None)
    parser.add_argument("--tool-timeout-seconds", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument(
        "--max-turns-audit",
        dest="max_turns_audit",
        type=int,
        default=None,
        help=(
            "Audit/progress reference for Worker heuristics. This is not a "
            "turn-count stop condition."
        ),
    )
    parser.add_argument(
        "--max-turns",
        dest="max_turns_audit",
        type=int,
        default=None,
        help=(
            "Deprecated alias for --max-turns-audit; retained for old local "
            "scripts and never used as a Worker loop limit."
        ),
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=None,
        help=(
            "Harbor attempts per task. Leaderboard-candidate Terminal-Bench 2.0 "
            "runs should use 5; local smoke runs may use 1."
        ),
    )
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--trials-config", default="config/trials.yaml")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--output", default="trials/runs")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Audit-only host Harbor wait reference in seconds; defaults to "
            "execution.timeout_per_task and does not stop the Harbor/Worker loop"
        ),
    )
    parser.add_argument(
        "--skip-network-preflight",
        action="store_true",
        help="Skip Docker/apt network checks before launching Harbor",
    )
    parser.add_argument(
        "--network-preflight-mode",
        choices=["quick", "full"],
        default=None,
    )
    parser.add_argument("--network-preflight-timeout", type=int, default=None)
    force_build = parser.add_mutually_exclusive_group()
    force_build.add_argument("--force-build", dest="force_build", action="store_true", default=None)
    force_build.add_argument("--no-force-build", dest="force_build", action="store_false")
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--verifier-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--agent-setup-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--environment-build-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--mounts-json",
        default=None,
        help="JSON array of Docker Compose service volume mounts for Harbor",
    )
    parser.add_argument(
        "--verifier-env",
        action="append",
        default=None,
        help="Verifier environment variable in KEY=VALUE format; repeatable",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Pass Harbor --yes for explicit non-interactive prompt confirmation",
    )
    parser.add_argument(
        "--no-network-hardened-environment",
        dest="network_hardened_environment",
        action="store_false",
        default=None,
        help="Do not inject the apt mirror Harbor environment wrapper",
    )
    add_docker_resource_args(parser)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _apply_execution_defaults(args, parser)

    from bench.harbor import HarborRunner
    from bench.tasks import TaskCatalog

    if args.all and not args.task:
        catalog = TaskCatalog.load_or_empty(args.path)
        print(json.dumps({"tasks": catalog.list_task_ids(), "count": catalog.total_count}, indent=2))
        return 0

    if not args.task:
        parser.error("--task is required unless --all is used")

    runner = HarborRunner(
        dataset_path=args.path if not args.dataset else None,
        dataset_name=args.dataset,
        jobs_dir=Path(args.jobs_dir),
        output_dir=Path(args.output),
    )
    agent_config = resolve_agent_config(args, parser)
    command = runner.build_command(
        args.task,
        agent_config,
        job_name=args.job_name,
        jobs_dir=args.jobs_dir,
    )

    if args.dry_run:
        print("Harbor command:")
        print(command.shell_command())
        print("\nJob config summary:")
        print(json.dumps(command.config, indent=2))
        return 0

    _require_worker_api_key(agent_config, parser)

    if not args.skip_network_preflight:
        _run_network_preflight(args, blocking=False)

    result = runner.run_task(
        task_id=args.task,
        agent_config=agent_config,
        timeout_audit=args.timeout,
        job_name=args.job_name,
        jobs_dir=args.jobs_dir,
    )
    print(f"Task: {result.task_id}")
    print(f"Status: {result.status.value}")
    print(f"Score: {result.score}")
    print(f"Verified: {result.verified}")
    print(f"Trial: {result.trial_id}")
    print(f"Harbor job: {result.harbor_job_dir}")
    if result.error_log:
        print("Errors:")
        for error in result.error_log[:3]:
            print(f"- {error[:500]}")
    return 0 if result.status.value in ("passed", "failed") else 1


def resolve_agent_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, object]:
    """Resolve CLI overrides, local env, and role config into Harbor agent config."""
    _reject_timeout_resource_overrides(args, parser)
    model = getattr(args, "model", None)
    provider = getattr(args, "provider", None)
    base_url = getattr(args, "base_url", None)
    api_key_env = getattr(args, "api_key_env", None)
    reasoning_effort = getattr(args, "reasoning_effort", None)
    reasoning_max_tokens = getattr(args, "reasoning_max_tokens", None)
    max_output_tokens = getattr(args, "max_output_tokens", None)
    llm_timeout_seconds = getattr(args, "llm_timeout_seconds", None)
    tool_timeout_seconds = getattr(args, "tool_timeout_seconds", None)
    max_retries = getattr(args, "max_retries", None)
    max_turns_audit = getattr(args, "max_turns_audit", None)
    custom_llm_provider = None
    worker_role = getattr(args, "worker_role", None) or os.environ.get("HL_WORKER_ROLE") or "worker"

    env_file = _resolve_env_file(getattr(args, "env_file", None), parser)
    if env_file:
        _load_dotenv(env_file)

    models_path = _select_models_config(getattr(args, "models_config", None), parser)
    if models_path is not None:
        models = ModelsConfig.from_yaml(models_path)
        if worker_role in models.roles:
            worker = models.get_role(worker_role)
            model = _prefer_explicit(model, worker.model)
            provider = _prefer_explicit(provider, worker.provider)
            base_url = _prefer_explicit(base_url, worker.base_url)
            api_key_env = _prefer_explicit(api_key_env, worker.api_key_env)
            reasoning_effort = _prefer_explicit(reasoning_effort, worker.reasoning.effort)
            reasoning_max_tokens = _prefer_explicit(
                reasoning_max_tokens,
                worker.reasoning.max_tokens,
            )
            worker_max_output_tokens = (
                str(worker.max_output_tokens)
                if worker.max_output_tokens is not None
                else None
            )
            max_output_tokens = _prefer_explicit(
                max_output_tokens,
                worker_max_output_tokens,
            )
            llm_timeout_seconds = _prefer_explicit(
                llm_timeout_seconds,
                worker.timeout_seconds,
            )
            max_retries = _prefer_explicit(max_retries, worker.max_retries)
            custom_llm_provider = worker.extra.get("custom_llm_provider")
        elif getattr(args, "worker_role", None):
            parser.error(
                f"Worker role {worker_role!r} is not configured in {models_path}. "
                f"Available roles: {', '.join(sorted(models.roles))}"
            )

    agent_config: dict[str, object] = {
        "agent": getattr(args, "agent", "hl-worker"),
        "worker_role": worker_role,
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "reasoning_effort": reasoning_effort,
        "reasoning_max_tokens": reasoning_max_tokens,
        "max_output_tokens": max_output_tokens,
        "max_turns_audit": max_turns_audit,
        "n_attempts": getattr(args, "n_attempts", None),
        "timeout_seconds": llm_timeout_seconds,
        "tool_timeout_seconds": tool_timeout_seconds,
        "max_retries": max_retries,
        "custom_llm_provider": custom_llm_provider,
        "mounts_json": _parse_mounts_json(getattr(args, "mounts_json", None), parser),
        "verifier_env": _validate_env_args(getattr(args, "verifier_env", None), parser),
        "yes": bool(getattr(args, "yes", False)),
        "docker_resource_enabled": _docker_resource_enabled(args),
        "docker_memory": getattr(args, "docker_memory", None) or DEFAULT_DOCKER_MEMORY,
        "docker_memory_swap": (
            getattr(args, "docker_memory_swap", None)
            or getattr(args, "docker_memory", None)
            or DEFAULT_DOCKER_MEMORY_SWAP
        ),
        "docker_cpus": getattr(args, "docker_cpus", None) or DEFAULT_DOCKER_CPUS,
        "docker_pids_limit": getattr(args, "docker_pids_limit", None)
        or DEFAULT_DOCKER_PIDS_LIMIT,
        "docker_labels": _parse_docker_label_args(
            getattr(args, "docker_label", None),
            parser,
        ),
        "docker_log_max_size": getattr(args, "docker_log_max_size", None)
        or DEFAULT_DOCKER_LOG_MAX_SIZE,
        "docker_log_max_file": getattr(args, "docker_log_max_file", None)
        or DEFAULT_DOCKER_LOG_MAX_FILE,
    }
    if env_file:
        agent_config["env_file"] = str(env_file)
    if getattr(args, "force_build", False):
        agent_config["force_build"] = True
    _apply_network_execution_defaults(args, agent_config)
    return agent_config


def _require_worker_api_key(
    agent_config: dict[str, object],
    parser: argparse.ArgumentParser,
) -> None:
    """Fail before Harbor launch if the configured worker key is unavailable."""

    api_key_env = str(agent_config.get("api_key_env") or "").strip()
    if not api_key_env:
        parser.error(
            "worker provider is missing api_key_env; configure the role with an "
            "environment variable name before launching Harbor"
        )
    if not str(os.environ.get(api_key_env) or "").strip():
        env_file = str(agent_config.get("env_file") or ".env.local")
        parser.error(
            f"worker provider key environment variable {api_key_env} is not set "
            f"after loading {env_file}; set it before launching Harbor"
        )


def _apply_execution_defaults(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    trials_path = Path(getattr(args, "trials_config", "config/trials.yaml"))
    if not trials_path.exists():
        _reject_timeout_resource_overrides(args, parser)
        _apply_docker_resource_defaults(args, {}, parser, trials_path)
        return
    try:
        import yaml

        data = yaml.safe_load(trials_path.read_text()) or {}
    except Exception as exc:
        parser.error(f"Could not load trials config {trials_path}: {exc}")
    if not isinstance(data, dict):
        return
    execution = data.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    _reject_timeout_resource_overrides(
        args,
        parser,
        trials_path=trials_path,
        execution=execution,
    )
    if getattr(args, "timeout", None) is None:
        if execution.get("timeout_per_task") is not None:
            try:
                args.timeout = int(execution["timeout_per_task"])
            except (TypeError, ValueError):
                parser.error(f"execution.timeout_per_task must be an integer in {trials_path}")
        else:
            args.timeout = 1800
    if args.timeout <= 0:
        args.timeout = None
    if not hasattr(args, "max_turns_audit"):
        setattr(args, "max_turns_audit", getattr(args, "max_turns", None))
    if getattr(args, "max_turns_audit", None) is None:
        legacy_max_turns = execution.get("max_turns_per_task")
        max_turns_audit = execution.get("max_turns_audit", legacy_max_turns)
        if max_turns_audit is not None:
            try:
                args.max_turns_audit = int(max_turns_audit)
            except (TypeError, ValueError):
                parser.error(
                    f"execution.max_turns_audit must be an integer in {trials_path}"
                )
    if getattr(args, "max_turns_audit", None) is not None and args.max_turns_audit <= 0:
        args.max_turns_audit = None
    if hasattr(args, "max_turns"):
        args.max_turns = args.max_turns_audit
    if (
        hasattr(args, "tool_timeout_seconds")
        and getattr(args, "tool_timeout_seconds", None) is None
        and execution.get("tool_timeout_seconds") is not None
    ):
        try:
            args.tool_timeout_seconds = int(execution["tool_timeout_seconds"])
        except (TypeError, ValueError):
            parser.error(f"execution.tool_timeout_seconds must be an integer in {trials_path}")
    if (
        hasattr(args, "tool_timeout_seconds")
        and getattr(args, "tool_timeout_seconds", None) is not None
        and args.tool_timeout_seconds <= 0
    ):
        parser.error("--tool-timeout-seconds and execution.tool_timeout_seconds must be positive")
    if (
        hasattr(args, "n_attempts")
        and getattr(args, "n_attempts", None) is None
        and execution.get("n_attempts") is not None
    ):
        try:
            args.n_attempts = int(execution["n_attempts"])
        except (TypeError, ValueError):
            parser.error(f"execution.n_attempts must be an integer in {trials_path}")
    if hasattr(args, "n_attempts") and getattr(args, "n_attempts", None) is not None:
        if args.n_attempts <= 0:
            args.n_attempts = None
    if getattr(args, "force_build", None) is None and execution.get("force_build") is not None:
        args.force_build = bool(execution["force_build"])
    network = data.get("network", {})
    if isinstance(network, dict):
        if getattr(args, "network_preflight_mode", None) is None and network.get("preflight_mode"):
            args.network_preflight_mode = str(network["preflight_mode"])
        if (
            getattr(args, "network_preflight_timeout", None) is None
            and network.get("preflight_timeout_seconds") is not None
        ):
            try:
                args.network_preflight_timeout = int(network["preflight_timeout_seconds"])
            except (TypeError, ValueError):
                parser.error(
                    "network.preflight_timeout_seconds must be numeric "
                    f"in {trials_path}"
                )
    docker_resources = data.get("docker_resources", {})
    if docker_resources is None:
        docker_resources = {}
    if not isinstance(docker_resources, dict):
        parser.error(f"docker_resources must be a mapping in {trials_path}")
    _apply_docker_resource_defaults(args, docker_resources, parser, trials_path)


def _reject_timeout_resource_overrides(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    trials_path: Path | None = None,
    execution: dict[str, object] | None = None,
) -> None:
    for field_name, flag in DISALLOWED_TIMEOUT_RESOURCE_FIELDS.items():
        if hasattr(args, field_name) and getattr(args, field_name, None) is not None:
            parser.error(OFFICIAL_LIMITS_ERROR.format(source=flag))

    if execution is None:
        return

    for field_name in DISALLOWED_TIMEOUT_RESOURCE_FIELDS:
        if execution.get(field_name) is not None:
            parser.error(
                OFFICIAL_LIMITS_ERROR.format(
                    source=f"execution.{field_name} from {trials_path}"
                )
            )
    for field_name in DISALLOWED_RESOURCE_CONFIG_FIELDS:
        if execution.get(field_name) is not None:
            parser.error(
                OFFICIAL_LIMITS_ERROR.format(
                    source=f"execution.{field_name} from {trials_path}"
                )
            )


def _select_models_config(explicit: str | None, parser: argparse.ArgumentParser) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            parser.error(f"--models-config path does not exist: {path}")
        return path

    for path in (Path("config/local.yaml"), Path("config/models.yaml")):
        if not path.exists():
            continue
        try:
            models = ModelsConfig.from_yaml(path)
        except Exception as exc:
            parser.error(f"Could not load model config {path}: {exc}")
        if "worker" in models.roles:
            return path
    return None


def _resolve_env_file(explicit: str | None, parser: argparse.ArgumentParser) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            parser.error(f"--env-file path does not exist: {path}")
        return path

    path = Path(".env.local")
    return path if path.exists() else None


def _load_dotenv(path: Path) -> None:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _parse_mounts_json(
    raw: str | None,
    parser: argparse.ArgumentParser,
) -> list[dict[str, object]] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"--mounts-json must be valid JSON: {exc}")
    if not isinstance(parsed, list):
        parser.error("--mounts-json must be a JSON array")
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            parser.error(f"--mounts-json item {index} must be an object")
        missing = {"type", "source", "target"} - set(item)
        if missing:
            parser.error(
                f"--mounts-json item {index} is missing required keys: {', '.join(sorted(missing))}"
            )
    return parsed


def _validate_env_args(
    values: list[str] | None,
    parser: argparse.ArgumentParser,
) -> list[str]:
    env_args = values or []
    for value in env_args:
        key, separator, _ = value.partition("=")
        if not separator or not key.strip():
            parser.error(f"Invalid --verifier-env {value!r}; expected KEY=VALUE")
    return env_args


def _apply_network_execution_defaults(
    args: argparse.Namespace,
    agent_config: dict[str, object],
) -> None:
    trials_path = Path(getattr(args, "trials_config", "config/trials.yaml"))
    network = _network_config_from_trials(trials_path)
    if getattr(args, "network_preflight_mode", None) is None and network.get("preflight_mode"):
        args.network_preflight_mode = str(network["preflight_mode"])
    if (
        getattr(args, "network_preflight_timeout", None) is None
        and network.get("preflight_timeout_seconds") is not None
    ):
        try:
            args.network_preflight_timeout = int(network["preflight_timeout_seconds"])
        except (TypeError, ValueError):
            args.network_preflight_timeout = None
    enabled = getattr(args, "network_hardened_environment", None)
    if enabled is None:
        enabled = bool(network.get("hardened_environment", True))
    agent_config["network_hardened_environment"] = bool(enabled)
    for source_key, target_key in [
        ("infra_retries", "infra_retries"),
        ("debian_mirror", "debian_mirror"),
        ("debian_security_mirror", "debian_security_mirror"),
        ("ubuntu_mirror", "ubuntu_mirror"),
        ("docker_hub_mirror", "docker_hub_mirror"),
        ("prebuilt_docker_hub_mirror", "prebuilt_docker_hub_mirror"),
        ("docker_image_overrides", "docker_image_overrides"),
        ("download_url_rewrites", "download_url_rewrites"),
        ("pypi_index_url", "pypi_index_url"),
        ("pypi_trusted_host", "pypi_trusted_host"),
        ("apt_retries", "apt_retries"),
        ("apt_timeout_seconds", "apt_timeout_seconds"),
        ("pip_retries", "pip_retries"),
        ("pip_timeout_seconds", "pip_timeout_seconds"),
        ("prebuilt_docker_pull_timeout_seconds", "prebuilt_docker_pull_timeout_seconds"),
        ("bootstrap_ca_certificates", "bootstrap_ca_certificates"),
        ("download_retry_wrapper", "download_retry_wrapper"),
    ]:
        if network.get(source_key) is not None:
            agent_config[target_key] = network[source_key]
    _apply_host_ca_defaults(args, agent_config, network)
    _apply_verifier_network_defaults(agent_config, network)


def _apply_verifier_network_defaults(
    agent_config: dict[str, object],
    network: dict[str, object],
) -> None:
    """Propagate network hardening into Harbor verifier commands.

    The Dockerfile wrapper covers build-time apt/pip work, but many
    TerminalBench verifiers install pytest/uv/opencv inside tests/test.sh. Those
    verifier-time downloads happen after Harbor starts the trial container, so
    they need explicit --verifier-env values.
    """
    verifier_env = list(agent_config.get("verifier_env") or [])
    existing_keys = {str(item).split("=", 1)[0] for item in verifier_env if "=" in str(item)}

    def append_default(key: str, value: object | None) -> None:
        if value is None or key in existing_keys:
            return
        text = str(value)
        if not text:
            return
        verifier_env.append(f"{key}={text}")
        existing_keys.add(key)

    pypi_index_url = network.get("pypi_index_url")
    pypi_trusted_host = network.get("pypi_trusted_host")
    pip_timeout_seconds = network.get("pip_timeout_seconds")
    pip_retries = network.get("pip_retries")

    append_default("PIP_INDEX_URL", pypi_index_url)
    append_default("PIP_DEFAULT_TIMEOUT", pip_timeout_seconds)
    append_default("PIP_RETRIES", pip_retries)
    append_default("PIP_TRUSTED_HOST", pypi_trusted_host)
    append_default("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    append_default("PIP_NO_INPUT", "1")
    append_default("UV_INDEX_URL", pypi_index_url)
    append_default("UV_DEFAULT_INDEX", pypi_index_url)
    append_default("UV_INDEX_STRATEGY", "unsafe-first-match")
    append_default("UV_NATIVE_TLS", "true")
    append_default("UV_NO_PROGRESS", "1")
    append_default("DEBIAN_FRONTEND", "noninteractive")
    append_default("APT_LISTCHANGES_FRONTEND", "none")
    append_default("HL_VERIFIER_NETWORK_PREPARE", "1")
    cache_container_dir = str(network.get("verifier_cache_container_dir") or "/tmp/hl-verifier-cache")
    append_default("PIP_CACHE_DIR", f"{cache_container_dir}/pip")
    append_default("UV_CACHE_DIR", f"{cache_container_dir}/uv")

    agent_config["verifier_env"] = verifier_env
    if bool(network.get("verifier_dependency_cache", True)):
        _append_verifier_cache_mount(agent_config, network, cache_container_dir)


def _append_verifier_cache_mount(
    agent_config: dict[str, object],
    network: dict[str, object],
    cache_container_dir: str,
) -> None:
    cache_host_dir = Path(str(network.get("verifier_cache_host_dir") or "trials/cache/verifier"))
    _prepare_verifier_cache_host_dir(cache_host_dir)
    source = str(cache_host_dir.expanduser().resolve())
    mounts_json = list(agent_config.get("mounts_json") or [])
    target_exists = any(
        isinstance(item, dict) and item.get("target") == cache_container_dir
        for item in mounts_json
    )
    if target_exists:
        return
    mounts_json.append(
        {
            "type": "bind",
            "source": source,
            "target": cache_container_dir,
            "read_only": False,
        }
    )
    agent_config["mounts_json"] = mounts_json


def _prepare_verifier_cache_host_dir(cache_host_dir: Path) -> None:
    cache_host_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_host_dir.chmod(0o777)
    except OSError:
        pass
    for relative in [
        "pip",
        "uv",
        "uv/archive-v0",
        "uv/wheels-v5",
        "uv/sdists-v9",
        "uv/simple-v18",
        "uv/builds-v0",
        "uv/interpreter-v4",
        "uv-python",
        "uv-bin",
    ]:
        path = cache_host_dir / relative
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o777)
        except OSError:
            pass


def _apply_docker_resource_defaults(
    args: argparse.Namespace,
    docker_resources: dict[str, object],
    parser: argparse.ArgumentParser,
    trials_path: Path,
) -> None:
    def apply(attr: str, key: str, default: object) -> None:
        if not hasattr(args, attr):
            return
        if getattr(args, attr, None) is not None:
            return
        setattr(args, attr, docker_resources.get(key, default))

    apply("docker_resource_enabled", "enabled", True)
    apply("docker_memory", "memory", DEFAULT_DOCKER_MEMORY)
    apply(
        "docker_memory_swap",
        "memory_swap",
        docker_resources.get("memory", DEFAULT_DOCKER_MEMORY_SWAP),
    )
    apply("docker_cpus", "cpus", DEFAULT_DOCKER_CPUS)
    apply("docker_pids_limit", "pids_limit", DEFAULT_DOCKER_PIDS_LIMIT)
    apply("docker_log_max_size", "log_max_size", DEFAULT_DOCKER_LOG_MAX_SIZE)
    apply("docker_log_max_file", "log_max_file", DEFAULT_DOCKER_LOG_MAX_FILE)

    labels = docker_resources.get("labels")
    if getattr(args, "docker_label", None) is None and labels is not None:
        if not isinstance(labels, dict):
            parser.error(f"docker_resources.labels must be a mapping in {trials_path}")
        args.docker_label = [f"{key}={value}" for key, value in labels.items()]

    for attr, source in [
        ("docker_cpus", "docker_resources.cpus"),
        ("docker_pids_limit", "docker_resources.pids_limit"),
    ]:
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr, None)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parser.error(f"{source} must be a positive integer in {trials_path}")
        if parsed <= 0:
            parser.error(f"{source} must be a positive integer in {trials_path}")
        setattr(args, attr, parsed)

    for attr in ("docker_memory", "docker_memory_swap"):
        if hasattr(args, attr) and not str(getattr(args, attr, "")).strip():
            parser.error(f"{attr} must not be empty in {trials_path}")


def _docker_resource_enabled(args: argparse.Namespace) -> bool:
    value = getattr(args, "docker_resource_enabled", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_docker_label_args(
    values: list[str] | None,
    parser: argparse.ArgumentParser,
) -> dict[str, str]:
    try:
        return _parse_docker_label_items(values)
    except ValueError as exc:
        parser.error(str(exc))


def _parse_docker_label_items(values: list[str] | None) -> dict[str, str]:
    labels = dict(DEFAULT_DOCKER_LABELS)
    for value in values or []:
        key, separator, label_value = str(value).partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid --docker-label {value!r}; expected KEY=VALUE")
        labels[key.strip()] = label_value.strip()
    return labels


def docker_memory_limit_mb(args: argparse.Namespace) -> int | None:
    return _docker_memory_to_mb(getattr(args, "docker_memory", None))


def validate_docker_concurrency_budget(
    *,
    concurrency: int,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    source: str,
) -> dict[str, object]:
    enabled = _docker_resource_enabled(args)
    memory_mb = docker_memory_limit_mb(args)
    available_mb = _memavailable_mb()
    proof = {
        "concurrency": concurrency,
        "docker_memory": getattr(args, "docker_memory", None),
        "memory_limit_mb": memory_mb,
        "memavailable_mb": available_mb,
        "allowed_peak_mb": int(available_mb * 0.6) if available_mb else None,
        "projected_peak_mb": (memory_mb * concurrency) if memory_mb else None,
        "within_60_percent_memavailable": None,
    }
    if concurrency <= 1:
        proof["within_60_percent_memavailable"] = True
        return proof
    if not enabled:
        parser.error(f"{source} > 1 requires docker_resources.enabled=true")
    if memory_mb is None:
        parser.error(f"{source} > 1 requires a parseable --docker-memory cap")
    if available_mb is None:
        parser.error(
            f"{source} > 1 requires /proc/meminfo MemAvailable to prove Docker "
            "peak memory stays under 60% of WSL available memory"
        )
    projected = memory_mb * concurrency
    allowed = int(available_mb * 0.6)
    proof["within_60_percent_memavailable"] = projected <= allowed
    if projected > allowed:
        parser.error(
            f"{source}={concurrency} with docker memory {memory_mb}MiB projects "
            f"{projected}MiB peak, above 60% of current MemAvailable "
            f"({allowed}MiB of {available_mb}MiB). Lower concurrency or "
            "docker_resources.memory."
        )
    return proof


def _docker_memory_to_mb(value: object) -> int | None:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]?i?b?)?", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "m").rstrip("b")
    multipliers = {
        "": 1 / (1024 * 1024),
        "k": 1 / 1024,
        "ki": 1 / 1024,
        "m": 1,
        "mi": 1,
        "g": 1024,
        "gi": 1024,
        "t": 1024 * 1024,
        "ti": 1024 * 1024,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return max(1, int(amount * multiplier))


def _memavailable_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return int(parts[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _apply_host_ca_defaults(
    args: argparse.Namespace,
    agent_config: dict[str, object],
    network: dict[str, object],
) -> None:
    if not bool(network.get("mount_host_ca_certificates", False)):
        return
    bundle = str(network.get("host_ca_cert_bundle") or "/etc/ssl/certs/ca-certificates.crt")
    if not Path(bundle).is_file():
        return
    container_bundle = str(
        network.get("host_ca_cert_container_path")
        or "/tmp/hl-host-ca/ca-certificates.crt"
    )
    if agent_config.get("mounts_json") is None and getattr(args, "mounts_json", None) is None:
        agent_config["mounts_json"] = [
            {
                "type": "bind",
                "source": bundle,
                "target": container_bundle,
                "read_only": True,
            }
        ]
    verifier_env = list(agent_config.get("verifier_env") or [])
    existing_keys = {str(item).split("=", 1)[0] for item in verifier_env if "=" in str(item)}
    for key in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        if key not in existing_keys:
            verifier_env.append(f"{key}={container_bundle}")
    agent_config["verifier_env"] = verifier_env


def _network_config_from_trials(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    network = data.get("network") if isinstance(data, dict) else {}
    return network if isinstance(network, dict) else {}


@dataclass(frozen=True)
class NetworkPreflightResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    command: list[str] | None = None


def _run_network_preflight(
    args: argparse.Namespace,
    *,
    blocking: bool = False,
) -> NetworkPreflightResult:
    """Run host network diagnostics without turning failures into loop stops."""
    _ = blocking
    timeout = args.network_preflight_timeout or 120
    argv = _network_preflight_argv(args)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=max(int(timeout) + 30, 60),
        )
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            argv,
            124,
            stdout=exc.stdout or "",
            stderr=(
                f"network preflight process timed out after {max(int(timeout) + 30, 60)}s\n"
                f"{exc.stderr or ''}"
            ),
        )
    result = NetworkPreflightResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command=argv,
    )
    if completed.returncode == 0:
        return result
    sys.stderr.write(
        "Network preflight failed; recording diagnostics and continuing. "
        "This is not a master, sub-agent, Harbor, or Worker loop stop condition.\n"
    )
    if completed.stdout:
        sys.stderr.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            sys.stderr.write("\n")
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        if not completed.stderr.endswith("\n"):
            sys.stderr.write("\n")
    return result


def _network_preflight_argv(args: argparse.Namespace) -> list[str]:
    mode = args.network_preflight_mode or "quick"
    timeout = args.network_preflight_timeout or 120
    argv = [
        sys.executable,
        "scripts/network_preflight.py",
        "--json",
        f"--{mode}",
        "--timeout",
        str(timeout),
    ]
    argv.extend(docker_resource_forward_args(args))
    return argv


if __name__ == "__main__":
    raise SystemExit(main())
