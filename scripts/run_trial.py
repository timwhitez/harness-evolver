"""Trial CLI with role-aware model-config discovery.

This branch is intentionally stacked on the issue-12 precedence fix. The
precedence/validation implementation is retained in
:mod:`scripts._run_trial_issue11_base`; this facade resolves the effective role
after loading the selected environment file and validates that exact role in
every model configuration that is explicitly selected or automatically found.
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
        globals()[_name] = _value

_BASE_RESOLVE_AGENT_CONFIG = _base.resolve_agent_config
_EFFECTIVE_WORKER_ROLE: ContextVar[str | None] = ContextVar(
    "harness_evolver_effective_worker_role",
    default=None,
)
_SELECTED_MODELS_PATH: ContextVar[Path | None] = ContextVar(
    "harness_evolver_selected_models_path",
    default=None,
)


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

    # Preserve the audit-baseline CLI-only path. The parser exposes explicit
    # model/provider/base-url/key arguments, so absence of every config file is
    # not itself a role typo. The issue concerns selecting the wrong discovered
    # file or silently accepting a role missing from a selected file. Effective
    # request-value validation remains owned by the stacked issue-12 resolver.
    _SELECTED_MODELS_PATH.set(None)
    return None


def resolve_agent_config(args: Any, parser: Any) -> dict[str, object]:
    """Load env, resolve role source, select exact-role config, then merge."""

    # Role selection must happen after the selected env file is loaded. The
    # underlying implementation loads it again; dotenv loading is idempotent.
    env_file = _core._resolve_env_file(getattr(args, "env_file", None), parser)
    if env_file:
        _core._load_dotenv(env_file)

    cli_role = getattr(args, "worker_role", None)
    env_role = os.environ.get("HL_WORKER_ROLE")
    worker_role = cli_role or env_role or "worker"
    role_source = "cli" if cli_role else "environment" if env_role else "default"

    role_token = _EFFECTIVE_WORKER_ROLE.set(worker_role)
    path_token = _SELECTED_MODELS_PATH.set(None)
    try:
        config = _BASE_RESOLVE_AGENT_CONFIG(args, parser)
        selected_path = _SELECTED_MODELS_PATH.get()
    finally:
        _SELECTED_MODELS_PATH.reset(path_token)
        _EFFECTIVE_WORKER_ROLE.reset(role_token)

    if selected_path is None and not str(config.get("model") or "").strip():
        parser.error(
            "No model configuration file was found and no explicit --model was "
            "provided. Supply --models-config, create config/local.yaml or "
            "config/models.yaml, or provide the model request settings on the CLI."
        )

    config["worker_role"] = worker_role
    config["worker_role_source"] = role_source
    config["models_config_path"] = str(selected_path) if selected_path is not None else None
    return config


# The retained issue-12 resolver delegates into the original module, whose
# global config selector and main() lookup must both see these replacements.
_core._select_models_config = _select_models_config_for_effective_role
_core.resolve_agent_config = resolve_agent_config
_base.resolve_agent_config = resolve_agent_config
main = _base.main


if __name__ == "__main__":
    raise SystemExit(main())
