from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.run_trial import _load_dotenv_without_overriding_process_environment


_ENV_KEY = "HL_TEST_DOTENV_PRECEDENCE"


def _selected_env(tmp_path: Path, value: str = "from-dotenv") -> Path:
    path = tmp_path / "selected.env"
    path.write_text(f"{_ENV_KEY}={value}\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("process_value", ["", "from-process"])
def test_existing_process_key_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_value: str,
) -> None:
    monkeypatch.setenv(_ENV_KEY, process_value)

    _load_dotenv_without_overriding_process_environment(_selected_env(tmp_path))

    assert os.environ[_ENV_KEY] == process_value


def test_missing_process_key_is_loaded_from_selected_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_KEY, raising=False)

    _load_dotenv_without_overriding_process_environment(_selected_env(tmp_path))

    assert os.environ[_ENV_KEY] == "from-dotenv"
