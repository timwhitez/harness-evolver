from __future__ import annotations

from pathlib import Path
import tomllib

from setuptools.discovery import PEP420PackageFinder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    expected_modules = {
        "harness_evolver.cli",
        "scripts.run_trial",
        "scripts.regression_check",
        "scripts.compare_trials",
        "scripts.mission_debug",
    }

    for target in entry_points.values():
        module_name, separator, callable_name = target.partition(":")
        assert separator == ":"
        assert module_name in expected_modules
        assert callable_name

    for module_name in expected_modules:
        package_name = module_name.rpartition(".")[0]
        module_path = PROJECT_ROOT / f"{module_name.replace('.', '/')}.py"
        assert package_name in discovered
        assert module_path.is_file()
