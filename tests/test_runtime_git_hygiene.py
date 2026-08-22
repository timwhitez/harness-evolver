import subprocess

import pytest


def test_runtime_artifacts_are_not_tracked_by_git():
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if inside.returncode != 0:
        pytest.skip("git worktree is not available")

    tracked = subprocess.run(
        ["git", "ls-files", "jobs", "trials"],
        capture_output=True,
        text=True,
        check=True,
    )
    offenders = [
        path
        for path in tracked.stdout.splitlines()
        if path and not path.endswith("/.gitkeep")
    ]

    assert offenders == []


def test_known_runtime_paths_are_ignored_by_git():
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if inside.returncode != 0:
        pytest.skip("git worktree is not available")

    paths = [
        "jobs/example/result.json",
        "trials/runs/example/result.json",
        "trials/summaries/example_campaign.json",
        "trials/diffs/example/review.json",
        "trials/goals/example.json",
        "trials/memory/example.md",
        "trials/regressions/example.json",
        "trials/submissions/example.dry_run.json",
        "trials/background_logs/example.log",
        "trials/background_logs/example.pid",
    ]
    ignored = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert ignored.returncode == 0
    assert ignored.stdout.splitlines() == paths
