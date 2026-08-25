from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

from harness_evolver.cli import regression


def test_regression_entrypoint_injects_packaged_defaults_outside_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "runtime"
    monkeypatch.chdir(outside)
    monkeypatch.setenv("HL_RUNTIME_CACHE_DIR", str(cache))

    captured: dict[str, list[str]] = {}
    fake_module = ModuleType("scripts.regression_check")

    def fake_main() -> int:
        captured["argv"] = list(sys.argv)
        return 0

    fake_module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts.regression_check", fake_module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["harness-evolver-regression", "--dry-run"],
    )

    assert regression() == 0

    argv = captured["argv"]
    trials_path = Path(argv[argv.index("--trials-config") + 1])
    models_path = Path(argv[argv.index("--models-config") + 1])
    assert trials_path.is_file()
    assert models_path.is_file()
    assert trials_path.is_relative_to(cache)
    assert models_path.is_relative_to(cache)
