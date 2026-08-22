"""Tests for Harbor integration (mocked)."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from bench.harbor import HarborRunner
from hl.types import TrialStatus


class TestHarborRunner:
    def test_parse_result_from_reward_file(self, tmp_path):
        runner = HarborRunner(output_dir=tmp_path)
        trial_dir = tmp_path / "test" / "logs" / "verifier"
        trial_dir.mkdir(parents=True)
        (trial_dir / "reward.txt").write_text("1.0")

        trial = runner._parse_result(
            trial_id="test",
            task_id="domain::task::1.0",
            returncode=0,
            stdout="OK",
            stderr="",
            wall_time=10.0,
        )
        assert trial.status == TrialStatus.PASSED
        assert trial.score == 1.0

    def test_parse_failed_result(self, tmp_path):
        runner = HarborRunner(output_dir=tmp_path)
        trial_dir = tmp_path / "test" / "logs" / "verifier"
        trial_dir.mkdir(parents=True)
        (trial_dir / "reward.txt").write_text("0.0")

        trial = runner._parse_result(
            trial_id="test",
            task_id="domain::task::1.0",
            returncode=0,
            stdout="",
            stderr="some error",
            wall_time=5.0,
        )
        assert trial.status == TrialStatus.FAILED
        assert trial.score == 0.0

    def test_parse_missing_reward_file(self, tmp_path):
        runner = HarborRunner(output_dir=tmp_path)

        trial = runner._parse_result(
            trial_id="test",
            task_id="domain::task::1.0",
            returncode=1,
            stdout="",
            stderr="",
            wall_time=5.0,
        )
        assert trial.status == TrialStatus.FAILED

    def test_parse_timeout(self, tmp_path):
        runner = HarborRunner(output_dir=tmp_path)

        trial = runner._parse_result(
            trial_id="test",
            task_id="domain::task::1.0",
            returncode=-1,
            stdout="",
            stderr="",
            wall_time=1800,
        )
        assert trial.status == TrialStatus.FAILED
