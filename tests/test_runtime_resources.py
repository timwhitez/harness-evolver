from __future__ import annotations

from importlib import resources
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib
import zipfile

import pytest

from bench.agent import HLAgent
from bench.runtime_resources import (
    bundled_config_path,
    materialize_runtime_resources,
    runtime_resource_digest,
    worker_crate_root,
)
from harness_evolver.cli import _trial_argv_with_runtime_defaults


def test_materializes_complete_resources_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("HL_RUNTIME_CACHE_DIR", str(cache))
    monkeypatch.chdir(outside)

    root = materialize_runtime_resources()

    assert root.parent == cache
    assert root.name == runtime_resource_digest()
    assert (root / "hl-worker-core/Cargo.toml").is_file()
    assert (root / "hl-worker-core/Cargo.lock").is_file()
    assert (root / "hl-worker-core/src/main.rs").is_file()
    for name in ("benchmark.yaml", "default.yaml", "models.yaml", "trials.yaml"):
        assert (root / "config" / name).is_file()
    assert materialize_runtime_resources() == root


def test_materializes_from_non_filesystem_zip_resources(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "installed-package.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for relative in (
            Path("bench/__init__.py"),
            Path("bench/runtime_resources.py"),
            Path("bench/resources/__init__.py"),
            Path("bench/resources/config/benchmark.yaml"),
            Path("bench/resources/config/default.yaml"),
            Path("bench/resources/config/models.yaml"),
            Path("bench/resources/config/trials.yaml"),
            Path("bench/resources/hl-worker-core/Cargo.toml"),
            Path("bench/resources/hl-worker-core/Cargo.lock"),
            Path("bench/resources/hl-worker-core/src/main.rs"),
        ):
            bundle.write(project_root / relative, relative.as_posix())

    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "zip-cache"
    program = (
        "import os, pathlib, sys; "
        f"sys.path.insert(0, {str(archive)!r}); "
        f"os.environ['HL_RUNTIME_CACHE_DIR'] = {str(cache)!r}; "
        "from bench.runtime_resources import worker_crate_root; "
        "root = worker_crate_root(); "
        "assert (root / 'Cargo.toml').is_file(); "
        "assert (root / 'src/main.rs').is_file()"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=outside,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_incomplete_cache_is_repaired(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    root = materialize_runtime_resources(cache)
    missing = root / "hl-worker-core" / "src" / "main.rs"
    expected = missing.read_bytes()
    missing.unlink()

    repaired = materialize_runtime_resources(cache)

    assert repaired == root
    assert missing.read_bytes() == expected


def test_materialized_worker_bytes_match_package_resources(tmp_path: Path) -> None:
    crate = worker_crate_root(tmp_path / "cache")
    packaged = resources.files("bench.resources").joinpath("hl-worker-core", "src", "main.rs")

    assert (crate / "src/main.rs").read_bytes() == packaged.read_bytes()


@pytest.mark.parametrize(
    "name",
    ["benchmark.yaml", "default.yaml", "models.yaml", "trials.yaml"],
)
def test_bundled_config_bytes_match_the_effective_source_config(name: str) -> None:
    """Prevent installed defaults from drifting from the reviewed source config."""

    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "config" / name
    packaged = project_root / "bench" / "resources" / "config" / name

    assert packaged.read_bytes() == source.read_bytes()


def test_worker_command_uses_cached_manifest_not_source_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HL_RUNTIME_CACHE_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("HL_WORKER_RUST_BIN", raising=False)

    def fake_which(name: str) -> str | None:
        return "/toolchain/rustup" if name == "rustup" else "/toolchain/cargo"

    monkeypatch.setattr("bench.agent.shutil.which", fake_which)
    agent = object.__new__(HLAgent)
    command = agent._rust_worker_command()

    manifest = Path(command[command.index("--manifest-path") + 1])
    assert command[:4] == ["/toolchain/rustup", "run", "stable", "cargo"]
    assert "--locked" in command
    assert manifest.is_file()
    assert manifest.parent == worker_crate_root()
    assert "crates/hl-worker-core" not in manifest.as_posix()


def test_prebuilt_cached_worker_is_preferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HL_RUNTIME_CACHE_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("HL_WORKER_RUST_BIN", raising=False)
    crate = worker_crate_root()
    binary_name = "hl-worker-core.exe" if os.name == "nt" else "hl-worker-core"
    binary = crate / "target" / "release" / binary_name
    binary.parent.mkdir(parents=True)
    binary.write_text("worker", encoding="utf-8")
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(HLAgent, "_rust_worker_binary_is_current", lambda *_: True)
    monkeypatch.setattr("bench.agent.shutil.which", lambda _: None)

    agent = object.__new__(HLAgent)

    assert agent._rust_worker_command() == [str(binary)]


def test_missing_toolchain_fails_with_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HL_RUNTIME_CACHE_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("HL_WORKER_RUST_BIN", raising=False)
    monkeypatch.setattr("bench.agent.shutil.which", lambda _: None)
    agent = object.__new__(HLAgent)

    with pytest.raises(RuntimeError, match="Cargo is unavailable"):
        agent._rust_worker_command()


def test_trial_cli_injects_packaged_defaults_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HL_RUNTIME_CACHE_DIR", str(tmp_path / "runtime"))
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    argv = _trial_argv_with_runtime_defaults(["harness-evolver-trial", "--task", "demo"])

    trials_index = argv.index("--trials-config") + 1
    models_index = argv.index("--models-config") + 1
    assert Path(argv[trials_index]) == bundled_config_path("trials.yaml")
    assert Path(argv[models_index]) == bundled_config_path("models.yaml")


def test_trial_cli_preserves_explicit_and_local_config_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "trials.yaml").write_text("execution: {}\n", encoding="utf-8")
    (config / "models.yaml").write_text("roles: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    local = _trial_argv_with_runtime_defaults(["harness-evolver-trial", "--task", "demo"])
    explicit = _trial_argv_with_runtime_defaults(
        [
            "harness-evolver-trial",
            "--trials-config=/custom/trials.yaml",
            "--models-config",
            "/custom/models.yaml",
        ]
    )

    assert "--trials-config" not in local
    assert "--models-config" not in local
    assert explicit.count("--models-config") == 1
    assert sum(value.startswith("--trials-config") for value in explicit) == 1


def test_trial_help_does_not_materialize_runtime_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(_: str) -> Path:
        raise AssertionError("help must not materialize package resources")

    monkeypatch.setattr("harness_evolver.cli.bundled_config_path", unexpected)

    assert _trial_argv_with_runtime_defaults(["harness-evolver-trial", "--help"]) == [
        "harness-evolver-trial",
        "--help",
    ]


def test_pyproject_declares_every_runtime_package_data_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = set(config["tool"]["setuptools"]["package-data"]["bench.resources"])

    assert {
        "config/*.yaml",
        "hl-worker-core/Cargo.toml",
        "hl-worker-core/Cargo.lock",
        "hl-worker-core/src/*.rs",
    } <= patterns
