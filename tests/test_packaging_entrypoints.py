from __future__ import annotations

import os
from pathlib import Path
import shutil
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


def test_clean_wheel_install_runs_every_console_help_outside_checkout(
    tmp_path: Path,
) -> None:
    """Exercise the built artifact rather than only setuptools discovery."""

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

    environment = tmp_path / "venv"
    venv.EnvBuilder(
        with_pip=True,
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

    probe = (
        "import pathlib,sysconfig,harness_evolver,scripts; "
        "site=pathlib.Path(sysconfig.get_paths()['purelib']).resolve(); "
        "assert pathlib.Path(harness_evolver.__file__).resolve().is_relative_to(site); "
        "assert pathlib.Path(scripts.__file__).resolve().is_relative_to(site)"
    )
    probed = subprocess.run(
        [str(python), "-c", probe],
        cwd=tmp_path,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""},
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
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            command,
            completed.stdout,
            completed.stderr,
        )
