#!/usr/bin/env python3
"""Guided setup/preflight for HarnessEvolver."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup HarnessEvolver local config")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-path", default="terminal-bench-tasks/terminal-bench")
    parser.add_argument("--worker-provider", default="openai_compatible")
    parser.add_argument("--worker-model", default="deepseek-v4-pro")
    parser.add_argument("--worker-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--worker-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--worker-reasoning-effort", default="max")
    parser.add_argument("--worker-timeout-seconds", type=int, default=900)
    parser.add_argument("--worker-max-retries", type=int, default=5)
    parser.add_argument("--codex-model", default="gpt-5.4")
    parser.add_argument("--codex-sandbox", default="danger-full-access")
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=None,
        help=(
            "Compatibility/audit field only. Codex update sub-agent runs are "
            "not killed by this value."
        ),
    )
    parser.add_argument("--submit-threshold", type=float, default=0.80)
    parser.add_argument("--submit-visibility", choices=["private", "public"], default="private")
    parser.add_argument("--submit-share-org", action="append", default=None)
    parser.add_argument("--submit-share-user", action="append", default=None)
    parser.add_argument("--submit-share-yes", action="store_true")
    parser.add_argument(
        "--overwrite-local-config",
        action="store_true",
        help="Regenerate config/local.yaml instead of preserving existing local roles",
    )
    args = parser.parse_args()

    local_path = Path("config/local.yaml")
    existing_config = None if args.overwrite_local_config else load_existing_config(local_path)
    local_config = build_local_config(args, existing_config=existing_config)
    _load_dotenv(Path(".env.local"))
    checks = run_checks(args.task_path, local_config=local_config)
    env_example = build_env_example(args, local_config)
    output = {
        "checks": checks,
        "local_config": local_config,
        "local_config_source": str(local_path) if existing_config else "generated",
        "env_example": env_example,
    }

    if args.dry_run:
        print(json.dumps(output, indent=2))
        return 0 if all(item["ok"] for item in checks) else 1

    if not args.non_interactive:
        print("This writes .env.example and config/local.yaml with redacted local settings.")
        response = input("Continue? (Y/n): ").strip().lower()
        if response in {"n", "no"}:
            return 0

    Path(".env.example").write_text(env_example)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(yaml.safe_dump(local_config, sort_keys=False))
    print("Wrote .env.example and config/local.yaml.")
    print_repair_guidance(checks)
    return 0 if all(item["ok"] for item in checks) else 1


def run_checks(
    task_path: str,
    *,
    local_config: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    checks = [
        check("python", sys.version_info >= (3, 11), sys.version.split()[0], "Install Python >= 3.11."),
        check("git", command_ok(["git", "rev-parse", "--is-inside-work-tree"]), "", "Run inside a git repository."),
        check("docker", shutil.which("docker") is not None, shutil.which("docker") or "", "Install Docker and start the daemon."),
        check(
            "docker_compose_v2",
            command_ok(["docker", "compose", "version"]),
            "docker compose version",
            "Install Docker Compose v2, for example: sudo apt-get install docker-compose-v2",
        ),
        check("harbor", shutil.which("harbor") is not None, shutil.which("harbor") or "", "Install Harbor CLI."),
        check("codex", shutil.which("codex") is not None, shutil.which("codex") or "", "Install/login Codex CLI."),
        check("task_path", Path(task_path).exists(), task_path, "Clone TerminalBench tasks to terminal-bench-tasks/terminal-bench."),
        check("trials_writable", writable(Path("trials")), "trials", "Ensure trials/ is writable."),
        check("jobs_writable", writable(Path("jobs")), "jobs", "Ensure jobs/ is writable."),
        check("harbor_auth", command_ok(["harbor", "auth", "status"]), "", "Run: harbor auth login"),
    ]
    checks.extend(worker_api_key_checks(local_config or {}))
    if not os.environ.get("CODEX_API_KEY"):
        checks.append(
            check(
                "codex_auth_hint",
                Path.home().joinpath(".codex", "auth.json").exists(),
                "CODEX_API_KEY or ~/.codex/auth.json",
                "Run codex login or set CODEX_API_KEY.",
            )
        )
    return checks


def worker_api_key_checks(local_config: dict[str, Any]) -> list[dict[str, object]]:
    roles = ((local_config.get("models") or {}).get("roles") or {})
    worker = roles.get("worker") if isinstance(roles, dict) else None
    if not isinstance(worker, dict):
        return []
    api_key_env = worker.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        return []
    present = bool(os.environ.get(api_key_env))
    return [
        check(
            f"api_key:{api_key_env}",
            present,
            f"{api_key_env}=<redacted>" if present else f"{api_key_env} missing",
            f"Set {api_key_env} in .env.local or the process environment.",
        )
    ]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def load_existing_config(path: Path) -> dict[str, Any] | None:
    """Load an existing gitignored local config without reading secret values."""
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def build_local_config(
    args: argparse.Namespace,
    existing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_role = {
        "provider": args.worker_provider,
        "base_url": args.worker_base_url,
        "api_key_env": args.worker_api_key_env,
        "model": args.worker_model,
        "reasoning": {
            "effort": args.worker_reasoning_effort,
            "max_tokens": None,
            "exclude": False,
        },
        "max_output_tokens": 8000,
        "timeout_seconds": args.worker_timeout_seconds,
        "max_retries": args.worker_max_retries,
    }
    generated = {
        "models": {
            "roles": {
                "worker": worker_role,
                "worker_deepseek": {
                    "provider": "openai_compatible",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "model": "deepseek-v4-pro",
                    "reasoning": {
                        "effort": "max",
                        "max_tokens": None,
                        "exclude": False,
                    },
                    "max_output_tokens": 8000,
                    "timeout_seconds": 900,
                    "max_retries": 5,
                },
                "worker_gpt": {
                    "provider": "openai",
                    "base_url": None,
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-5.4",
                    "reasoning": {
                        "effort": "xhigh",
                        "max_tokens": None,
                        "exclude": False,
                    },
                    "max_output_tokens": 8000,
                    "timeout_seconds": 1800,
                    "max_retries": 5,
                },
                "orchestrator": {
                    "runner": "codex",
                    "model": args.codex_model,
                    "reasoning": {
                        "effort": "xhigh",
                        "max_tokens": None,
                        "exclude": False,
                    },
                    "sandbox": args.codex_sandbox,
                    "timeout_seconds": args.codex_timeout_seconds,
                },
            }
        },
        "benchmark": {
            "path": args.task_path,
            "max_iterations": None,
            "regression_policy": "smoke",
        },
        "submit": {
            "enabled": False,
            "trigger_score": args.submit_threshold,
            "min_tasks_evaluated": 89,
            "require_full_regression": True,
            "require_clean_git": True,
            "require_no_uncommitted_harness_diff": True,
            "harbor_upload": True,
            "visibility": args.submit_visibility,
            "share_orgs": args.submit_share_org or [],
            "share_users": args.submit_share_user or [],
            "share_yes": args.submit_share_yes,
            "stop_after_submit_attempt": True,
            "once_per_campaign": True,
        },
    }

    if not existing_config:
        return generated

    merged = copy.deepcopy(existing_config)
    merged.setdefault("models", {})
    merged["models"].setdefault("roles", {})
    for role_name, role_config in generated["models"]["roles"].items():
        merged["models"]["roles"].setdefault(role_name, role_config)

    for section_name in ("benchmark", "submit"):
        merged.setdefault(section_name, {})
        for key, value in generated[section_name].items():
            merged[section_name].setdefault(key, value)

    return merged


def build_env_example(
    args: argparse.Namespace,
    local_config: dict[str, Any] | None = None,
) -> str:
    names: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in names:
            names.append(name)

    roles = (((local_config or {}).get("models") or {}).get("roles") or {})
    if isinstance(roles, dict):
        for role in roles.values():
            if isinstance(role, dict):
                add(role.get("api_key_env"))

    add(args.worker_api_key_env)
    add("DEEPSEEK_API_KEY")
    add("CODEX_API_KEY")
    add("OPENAI_BASE_URL")
    add("ANTHROPIC_API_KEY")
    return "\n".join(
        [f"{name}=" for name in names] + [""]
    )


def check(name: str, ok: bool, detail: str, repair: str) -> dict[str, object]:
    return {"name": name, "ok": ok, "detail": detail, "repair": repair}


def writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def command_ok(command: list[str]) -> bool:
    if shutil.which(command[0]) is None:
        return False
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode == 0


def print_repair_guidance(checks: list[dict[str, object]]) -> None:
    failed = [item for item in checks if not item["ok"]]
    if not failed:
        print("All setup checks passed.")
        return
    print("Repair guidance:")
    for item in failed:
        print(f"- {item['name']}: {item['repair']}")


if __name__ == "__main__":
    raise SystemExit(main())
