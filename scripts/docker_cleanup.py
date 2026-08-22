#!/usr/bin/env python3
"""Safe Docker cleanup helpers for local HarnessEvolver runs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_LABELS = {
    "com.harness-evolver.managed": "true",
    "com.harness-evolver.cleanup": "task",
}


@dataclass(frozen=True)
class CleanupCommand:
    name: str
    mode: str
    argv: list[str]
    risk: str

    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clean local HarnessEvolver Docker artifacts without deleting volumes. "
            "Dry-run is the default."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["containers", "conservative", "aggressive"],
        default="containers",
        help=(
            "containers removes only stopped labeled task containers; conservative "
            "also prunes old builder cache and dangling images; aggressive prunes "
            "unused images and builder cache more broadly. No mode deletes volumes."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run cleanup commands. Without this, commands are printed only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cleanup commands explicitly. This is also the default.",
    )
    parser.add_argument(
        "--list-volumes",
        action="store_true",
        help="List Docker volumes and inspect metadata; never deletes them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Container prune label filter as KEY=VALUE; repeatable.",
    )
    args = parser.parse_args()
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")

    labels = _labels(args.label)
    commands = cleanup_commands(args.mode, labels)
    _assert_no_volume_delete(commands)

    results: list[dict[str, Any]] = []
    for command in commands:
        record: dict[str, Any] = {
            **asdict(command),
            "command": command.shell_command(),
            "executed": bool(args.execute),
        }
        if args.execute:
            completed = subprocess.run(
                command.argv,
                capture_output=True,
                text=True,
                check=False,
            )
            record.update(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            if completed.returncode != 0:
                results.append(record)
                return _finish(args, commands, results, exit_code=completed.returncode)
        results.append(record)

    volume_report = list_volumes() if args.list_volumes else None
    return _finish(args, commands, results, volume_report=volume_report)


def cleanup_commands(mode: str, labels: dict[str, str]) -> list[CleanupCommand]:
    label_filters = []
    for key, value in labels.items():
        label_filters.extend(["--filter", f"label={key}={value}"])
    commands = [
        CleanupCommand(
            name="stopped_labeled_task_containers",
            mode=mode,
            argv=[
                "docker",
                "container",
                "prune",
                "--force",
                *label_filters,
            ],
            risk=(
                "Deletes only stopped containers matching HL label filters; "
                "running containers, images, cache, and volumes are untouched."
            ),
        )
    ]
    if mode in {"conservative", "aggressive"}:
        commands.extend(
            [
                CleanupCommand(
                    name="builder_cache",
                    mode=mode,
                    argv=_builder_prune_argv(mode),
                    risk=(
                        "Deletes Docker build cache only. This may make later builds "
                        "slower but does not delete images or volumes."
                    ),
                ),
                CleanupCommand(
                    name="images",
                    mode=mode,
                    argv=_image_prune_argv(mode),
                    risk=(
                        "Conservative mode deletes dangling images only. Aggressive "
                        "mode deletes unused images older than the configured filter. "
                        "Volumes are untouched."
                    ),
                ),
            ]
        )
    return commands


def _builder_prune_argv(mode: str) -> list[str]:
    if mode == "aggressive":
        return [
            "docker",
            "builder",
            "prune",
            "--all",
            "--force",
            "--filter",
            "until=24h",
        ]
    return [
        "docker",
        "builder",
        "prune",
        "--force",
        "--filter",
        "until=168h",
    ]


def _image_prune_argv(mode: str) -> list[str]:
    if mode == "aggressive":
        return [
            "docker",
            "image",
            "prune",
            "--all",
            "--force",
            "--filter",
            "until=168h",
        ]
    return ["docker", "image", "prune", "--force"]


def list_volumes() -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "volume", "ls", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    volumes = []
    for line in (completed.stdout or "").splitlines():
        try:
            volumes.append(json.loads(line))
        except json.JSONDecodeError:
            volumes.append({"raw": line})
    return {
        "command": "docker volume ls --format '{{json .}}'",
        "returncode": completed.returncode,
        "volumes": volumes,
        "stderr": completed.stderr,
        "volume_delete_allowed": False,
        "note": (
            "This command only lists volumes. Do not delete volumes unless names, "
            "purpose, owners, and data-loss risk are reviewed and confirmed."
        ),
    }


def _labels(values: list[str] | None) -> dict[str, str]:
    labels = dict(DEFAULT_LABELS)
    for value in values or []:
        key, separator, label_value = value.partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"Invalid --label {value!r}; expected KEY=VALUE")
        labels[key.strip()] = label_value.strip()
    return labels


def _assert_no_volume_delete(commands: list[CleanupCommand]) -> None:
    for command in commands:
        joined = " ".join(command.argv)
        if "--volumes" in command.argv or "volume prune" in joined or "volume rm" in joined:
            raise SystemExit(f"Refusing unsafe volume-delete command: {command.shell_command()}")


def _finish(
    args: argparse.Namespace,
    commands: list[CleanupCommand],
    results: list[dict[str, Any]],
    *,
    volume_report: dict[str, Any] | None = None,
    exit_code: int = 0,
) -> int:
    payload = {
        "mode": args.mode,
        "dry_run": not args.execute,
        "volume_delete_allowed": False,
        "commands": [asdict(command) | {"command": command.shell_command()} for command in commands],
        "results": results,
        "volumes": volume_report,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Docker cleanup mode: {args.mode}")
        print(f"Dry run: {not args.execute}")
        print("Volume deletion: disabled")
        for command in commands:
            print(f"- {command.name}: {command.shell_command()}")
            print(f"  risk: {command.risk}")
        if volume_report is not None:
            print(f"Volumes listed: {len(volume_report.get('volumes') or [])}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
