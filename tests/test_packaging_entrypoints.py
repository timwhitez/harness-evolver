from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tomllib
import venv
import zipfile

from setuptools.discovery import PEP420PackageFinder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODULES = {
    "harness_evolver.cli",
    "scripts.run_trial",
    "scripts.regression_check",
    "scripts.compare_trials",
    "scripts.mission_debug",
}
EXPECTED_RUNTIME_RESOURCES = {
    "bench/resources/config/benchmark.yaml",
    "bench/resources/config/default.yaml",
    "bench/resources/config/models.yaml",
    "bench/resources/config/trials.yaml",
    "bench/resources/hl-worker-core/Cargo.toml",
    "bench/resources/hl-worker-core/Cargo.lock",
    "bench/resources/hl-worker-core/src/main.rs",
}


def test_console_entrypoint_modules_are_in_discovered_packages() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    discovered = set(
        PEP420PackageFinder.find(
            where=str(PROJECT_ROOT),
            include=include,
        )
    )

    entry_points = config["project"]["scripts"]
    for target in entry_points.values():
        module_name, separator, callable_name = target.partition(":")
        assert separator == ":"
        assert module_name in EXPECTED_MODULES
        assert callable_name

    for module_name in EXPECTED_MODULES:
        package_name = module_name.rpartition(".")[0]
        module_path = PROJECT_ROOT / f"{module_name.replace('.', '/')}.py"
        assert package_name in discovered
        assert module_path.is_file()


def test_noneditable_wheel_install_runs_every_console_help_outside_checkout(
    tmp_path: Path,
) -> None:
    """Exercise package placement and the installed Worker protocol boundary.

    The fresh virtual environment intentionally inherits the test interpreter's
    already-installed third-party packages as a dependency fixture. The project
    wheel itself is installed non-editably with ``--no-deps``; assertions below
    require project packages and runtime resources to resolve from that wheel,
    never the copied checkout or original repository.
    """

    checkout = tmp_path / "checkout"
    shutil.copytree(
        PROJECT_ROOT,
        checkout,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    build_script = (
        "import os,sys; "
        "from setuptools import build_meta; "
        "os.chdir(sys.argv[1]); "
        "print(build_meta.build_wheel(sys.argv[2]))"
    )
    built = subprocess.run(
        [sys.executable, "-c", build_script, str(checkout), str(wheel_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    for module_name in EXPECTED_MODULES:
        assert f"{module_name.replace('.', '/')}.py" in names
    assert EXPECTED_RUNTIME_RESOURCES <= names

    environment = tmp_path / "venv"
    venv.EnvBuilder(
        with_pip=True,
        # Third-party dependencies are a fixture; project packages must still
        # come from the non-editable wheel installed below.
        system_site_packages=True,
        clear=True,
    ).create(environment)
    scripts_dir = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    pip = [str(python), "-m", "pip"]
    installed = subprocess.run(
        [*pip, "install", "--no-deps", "--no-index", str(wheel)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr

    isolated_environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
    }
    probe = (
        "import pathlib,sysconfig,bench,harness_evolver,scripts; "
        "site=pathlib.Path(sysconfig.get_paths()['purelib']).resolve(); "
        "assert pathlib.Path(bench.__file__).resolve().is_relative_to(site); "
        "assert pathlib.Path(harness_evolver.__file__).resolve().is_relative_to(site); "
        "assert pathlib.Path(scripts.__file__).resolve().is_relative_to(site)"
    )
    probed = subprocess.run(
        [str(python), "-c", probe],
        cwd=tmp_path,
        env=isolated_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probed.returncode == 0, probed.stderr

    for command in (
        "harness-evolver-trial",
        "harness-evolver-regression",
        "harness-evolver-compare",
        "harness-evolver-mission-debug",
    ):
        executable = scripts_dir / (f"{command}.exe" if os.name == "nt" else command)
        completed = subprocess.run(
            [str(executable), "--help"],
            cwd=tmp_path,
            env=isolated_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            command,
            completed.stdout,
            completed.stderr,
        )

    # Windows still validates wheel membership, installed imports, runtime
    # resources, and every console entry point above. A bare .py executable is
    # not a portable CreateProcess target for HL_WORKER_RUST_BIN, so the direct
    # executable-override protocol fixture below is intentionally POSIX-only.
    if os.name == "nt":
        return

    # Complete the installed Python bridge's real JSONL handshake without
    # requiring a networked Cargo build. The explicit override is part of the
    # supported runtime contract; package-resource assertions above independently
    # prove that the install also contains the manifest/source build resource.
    final_event = json.dumps(
        {
            "type": "final",
            "result": {
                "trial_id": "wheel-protocol",
                "task_id": "wheel-protocol",
                "status": "failed",
                "score": 0.0,
                "verified": False,
                "tool_calls": [],
                "trajectory": [],
                "token_usage": {},
                "error_log": [],
                "metadata": {"protocol_handshake": True},
            },
        }
    )
    worker = tmp_path / "protocol-worker.py"
    worker.write_text(
        f"#!{python}\n"
        "import json, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        "assert request.get('type') == 'start'\n"
        f"print({final_event!r}, flush=True)\n",
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)

    runtime_cache = tmp_path / "runtime-cache"
    handshake_probe = (
        "import pathlib,sysconfig,bench; "
        "from bench.agent import HLAgent; "
        "from bench.runtime_resources import bundled_config_path,worker_crate_root; "
        "site=pathlib.Path(sysconfig.get_paths()['purelib']).resolve(); "
        "assert pathlib.Path(bench.__file__).resolve().is_relative_to(site); "
        "crate=worker_crate_root(); "
        "assert (crate/'Cargo.toml').is_file(); "
        "assert (crate/'Cargo.lock').is_file(); "
        "assert (crate/'src/main.rs').is_file(); "
        "assert bundled_config_path('models.yaml').is_file(); "
        "assert bundled_config_path('trials.yaml').is_file(); "
        "result=HLAgent().run('wheel smoke', {'task_id':'wheel-protocol'}); "
        "assert result.task_id == 'wheel-protocol'; "
        "assert result.metadata.get('protocol_handshake') is True; "
        "assert result.metadata.get('rust_worker_core_error') is not True"
    )
    handshaken = subprocess.run(
        [str(python), "-c", handshake_probe],
        cwd=tmp_path,
        env={
            **isolated_environment,
            "HL_RUNTIME_CACHE_DIR": str(runtime_cache),
            "HL_WORKER_RUST_BIN": str(worker),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert handshaken.returncode == 0, (
        handshaken.stdout,
        handshaken.stderr,
    )
