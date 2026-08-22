import json

import pytest

from scripts import docker_cleanup


def test_cleanup_commands_never_delete_volumes():
    for mode in ("containers", "conservative", "aggressive"):
        commands = docker_cleanup.cleanup_commands(mode, docker_cleanup.DEFAULT_LABELS)
        docker_cleanup._assert_no_volume_delete(commands)

        rendered = "\n".join(command.shell_command() for command in commands)
        assert "--volumes" not in rendered
        assert "volume prune" not in rendered
        assert "volume rm" not in rendered


def test_explicit_dry_run_prints_commands_without_execution(monkeypatch, capsys):
    monkeypatch.setattr(
        docker_cleanup.sys,
        "argv",
        ["docker_cleanup.py", "--mode", "conservative", "--dry-run", "--json"],
    )

    assert docker_cleanup.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "conservative"
    assert payload["dry_run"] is True
    assert payload["volume_delete_allowed"] is False
    assert payload["results"]
    assert all(result["executed"] is False for result in payload["results"])


def test_execute_and_dry_run_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        docker_cleanup.sys,
        "argv",
        ["docker_cleanup.py", "--mode", "containers", "--execute", "--dry-run"],
    )

    with pytest.raises(SystemExit) as exc_info:
        docker_cleanup.main()

    assert exc_info.value.code == 2
