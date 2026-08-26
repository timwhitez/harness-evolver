"""Trial CLI with role-aware model-config discovery.

This branch is stacked on the issue-12 precedence fix. It resolves the effective
Worker role only after reading the selected dotenv, while preserving the
explicit precedence contract:

    CLI > pre-existing process environment > selected dotenv > default

The exact role is then used for every explicit or automatic model-config
selection and recorded in effective metadata with its source.
"""

from __future__ import annotations

from contextvars import ContextVar
import os
from pathlib import Path
from typing import Any

from scripts import _run_trial_issue11_base as _base
from scripts import _run_trial_issue12_base as _core

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)

_BASE_RESOLVE_AGENT_CONFIG = _base.resolve_agent_config
_EFFECTIVE_WORKER_ROLE: ContextVar[str | None] = ContextVar(
    "harness_evolver_effective_worker_role",
    default=None,
)
_SELECTED_MODELS_PATH: ContextVar[Path | None] = ContextVar(
    "harness_evolver_selected_models_path",
    default=None,
)


def _dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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
        values[key] = value
    return values


def _load_dotenv_without_overriding_process_environment(path: Path) -> None:
    for key, value in _dotenv_values(path).items():
        if key not in os.environ:
            os.environ[key] = value


def _select_models_config_for_effective_role(
    explicit: str | None,
    parser: Any,
) -> Path | None:
    role = _EFFECTIVE_WORKER_ROLE.get()
    if not role:
        parser.error("internal error: effective Worker role was not resolved")

    if explicit:
        path = Path(explicit)
        if not path.exists():
            parser.error(f"--models-config path does not exist: {path}")
        try:
            models = _core.ModelsConfig.from_yaml(path)
        except Exception as exc:
            parser.error(f"Could not load model config {path}: {exc}")
        if role not in models.roles:
            parser.error(
                f"Worker role {role!r} is not configured in {path}. "
                f"Available roles: {', '.join(sorted(models.roles)) or '(none)'}"
            )
        _SELECTED_MODELS_PATH.set(path)
        return path

    observed: list[tuple[Path, list[str]]] = []
    for path in (Path("config/local.yaml"), Path("config/models.yaml")):
        if not path.exists():
            continue
        try:
            models = _core.ModelsConfig.from_yaml(path)
        except Exception as exc:
            parser.error(f"Could not load model config {path}: {exc}")
        available_roles = sorted(models.roles)
        observed.append((path, available_roles))
        if role in models.roles:
            _SELECTED_MODELS_PATH.set(path)
            return path

    if observed:
        checked = "; ".join(
            f"{path}: {', '.join(roles) or '(none)'}"
            for path, roles in observed
        )
        parser.error(
            f"Worker role {role!r} is not configured in any discovered model "
            f"config. Checked roles: {checked}"
        )

    _SELECTED_MODELS_PATH.set(None)
    return None


def resolve_agent_config(args: Any, parser: Any) -> dict[str, object]:
    env_file = _core._resolve_env_file(getattr(args, "env_file", None), parser)
    process_role = str(os.environ.get("HL_WORKER_ROLE") or "").strip()
    dotenv_role = ""
    if env_file:
        dotenv_role = str(_dotenv_values(env_file).get("HL_WORKER_ROLE") or "").strip()
        _load_dotenv_without_overriding_process_environment(env_file)

    raw_cli_role = getattr(args, "worker_role", None)
    if raw_cli_role is not None and not str(raw_cli_role).strip():
        parser.error("--worker-role must not be empty")
    cli_role = str(raw_cli_role).strip() if raw_cli_role is not None else ""

    if cli_role:
        worker_role = cli_role
        role_source = "cli"
    elif process_role:
        worker_role = process_role
        role_source = "process_environment"
    elif dotenv_role:
        worker_role = dotenv_role
        role_source = "selected_dotenv"
    else:
        worker_role = "worker"
        role_source = "default"

    previous_role = os.environ.get("HL_WORKER_ROLE")
    os.environ["HL_WORKER_ROLE"] = worker_role
    role_token = _EFFECTIVE_WORKER_ROLE.set(worker_role)
    path_token = _SELECTED_MODELS_PATH.set(None)
    try:
        config = _BASE_RESOLVE_AGENT_CONFIG(args, parser)
        selected_path = _SELECTED_MODELS_PATH.get()
    finally:
        _SELECTED_MODELS_PATH.reset(path_token)
        _EFFECTIVE_WORKER_ROLE.reset(role_token)
        if previous_role is None:
            os.environ.pop("HL_WORKER_ROLE", None)
        else:
            os.environ["HL_WORKER_ROLE"] = previous_role

    if selected_path is None and not str(config.get("model") or "").strip():
        parser.error(
            "No model configuration file was found and no explicit --model was "
            "provided. Supply --models-config, create config/local.yaml or "
            "config/models.yaml, or provide the model request settings on the CLI."
        )

    config["worker_role"] = worker_role
    config["worker_role_source"] = role_source
    config["models_config_path"] = (
        str(selected_path) if selected_path is not None else None
    )
    return config


_core._load_dotenv = _load_dotenv_without_overriding_process_environment
_core._select_models_config = _select_models_config_for_effective_role
_core.resolve_agent_config = resolve_agent_config
_base._load_dotenv = _load_dotenv_without_overriding_process_environment
_base.resolve_agent_config = resolve_agent_config
main = _base.main


if __name__ == "__main__":
    raise SystemExit(main())
