"""Tests for file-system MemoryStore."""

from datetime import datetime
import json

from hl.attribution import FailureAttributor
from hl.memory import FileSystemMemory
from hl.model_scope import model_scope_from_config
from hl.types import (
    TrialResult,
    TrialStatus,
    TrialSummary,
    RegressionSnapshot,
    HarnessPatch,
    TaskDomain,
    TaskDifficulty,
)


def test_failure_attribution_keeps_agent_execution_timeout_without_verifier_semantics():
    trial = TrialResult(
        trial_id="agent-timeout",
        task_id="task-a",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.TIMEOUT,
        score=0.0,
        verified=False,
        error_log=["Agent execution timed out after 900.0 seconds"],
        metadata={
            "timeout_phase": "agent_execution",
            "verifier_infra_error": False,
        },
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "agent_execution_timeout"
    assert result.affected_components == [
        "bench/agent",
        "recovery/patterns",
        "context/compaction",
    ]


def test_failure_attribution_separates_agent_timeout_with_verifier_mismatch():
    ctrf = {
        "results": {
            "summary": {"tests": 6, "passed": 1, "failed": 5},
            "tests": [
                {
                    "name": "test_outputs.py::test_caffe_version_and_source",
                    "raw_status": "call_failed",
                    "trace": (
                        "E       FileNotFoundError: [Errno 2] No such file or "
                        "directory: '/app/caffe/.build_release/tools/caffe.bin'"
                    ),
                }
            ],
        }
    }
    trial = TrialResult(
        trial_id="agent-timeout-with-ctrf",
        task_id="caffe-cifar-10",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.TIMEOUT,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "Agent execution timed out after 1200.0 seconds",
            "## ctrf.json\n" + json.dumps(ctrf),
        ],
        metadata={
            "timeout_phase": "agent_execution",
            "verifier_infra_error": False,
        },
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "agent_timeout_with_verifier_mismatch"
    assert result.affected_components == [
        "bench/agent",
        "recovery/patterns",
        "context/compaction",
        "verification/checks",
    ]


def test_failure_attribution_promotes_cython_optional_import_recovery():
    trial = TrialResult(
        trial_id="build-cython-ext__optional-import",
        task_id="build-cython-ext",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.EASY,
        status=TrialStatus.TIMEOUT,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "Agent execution timed out after 900.0 seconds",
            "## ctrf.json\n"
            "test_outputs.py::test_chelpers_cython_extension raw_status=call_failed\n"
            "spec = importlib.util.find_spec(\"pyknotid.spacecurves.chelpers\")\n"
            "from pyknotid.visualise import plot_line\n"
            "E   ModuleNotFoundError: No module named 'vispy'",
        ],
        trajectory=[
            {
                "tool": "bash",
                "command": "pip install vispy cython numpy",
                "success": False,
                "output": "Package-manager command timeout was capped at 60s",
            }
        ],
        metadata={"timeout_phase": "agent_execution", "verifier_infra_error": False},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "cython_extension_optional_import_pivot_mechanism"
    assert result.affected_components == [
        "bench/agent",
        "bench/harbor_adapter",
        "crates/hl-worker-core",
        "harness/tools/shell",
        "recovery/patterns",
    ]


def test_failure_attribution_promotes_dependency_loop_missing_artifact():
    trial = TrialResult(
        trial_id="db-wal-recovery__missing-artifact",
        task_id="db-wal-recovery",
        task_domain=TaskDomain.DATABASE,
        task_difficulty=TaskDifficulty.HARD,
        status=TrialStatus.TIMEOUT,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "Agent execution timed out after 900.0 seconds",
            "service \"main\" is not running",
            "## ctrf.json\n"
            ">       assert os.path.exists(\"/app/recovered.json\"), "
            "\"recovered.json file does not exist\"\n"
            "E       AssertionError: recovered.json file does not exist",
        ],
        trajectory=[
            {
                "tool": "bash",
                "command": "apt-get install -y sqlite3",
                "success": False,
                "output": "Package-manager command timeout was capped at 60s",
            }
        ],
        metadata={"timeout_phase": "agent_execution", "verifier_infra_error": False},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "terminal_environment_unavailable_after_dependency_loop"
    assert result.affected_components == [
        "bench/harbor",
        "bench/network_environment",
    ]


def test_failure_attribution_preserves_csv_contract_with_ml_cv_pivot():
    trial = TrialResult(
        trial_id="sam-cell-seg__csv-heavy-import",
        task_id="sam-cell-seg",
        task_domain=TaskDomain.DATA_ENGINEERING,
        task_difficulty=TaskDifficulty.HARD,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output=(
            "FAILED ../tests/test_outputs.py::test_demo_metadata_csv_content\n"
            "df = pd.read_csv(args.csv_path)\n"
            "expected_data = {'cell-1': {'area': 12, 'mask_path': 'masks/cell-1.png'}}\n"
            "assert row['cell_id'] in expected_data"
        ),
        trajectory=[
            {
                "tool": "file_write",
                "file_path": "/app/convert_masks.py",
                "success": True,
                "output": "/app/convert_masks.py",
                "metadata": {"expected_artifact": "/app/convert_masks.py"},
            },
            {
                "tool": "bash",
                "command": "cd /app && python3 -c \"import ast; print('Import: cv2 From import: mobile_sam')\"",
                "success": True,
                "output": "AST parse OK Import: cv2 From import: mobile_sam",
            },
            {
                "tool": "bash",
                "command": "pip install torch opencv-python 2>&1 | tail -20",
                "success": False,
                "output": "ERROR: Could not find a version that satisfies the requirement torch",
            },
            {
                "tool": "bash",
                "command": "pip install numpy pandas Pillow 2>&1 | tail -20",
                "success": False,
                "output": "ERROR: No matching distribution found for numpy",
            },
            {
                "tool": "bash",
                "command": "cd /app && python3 -c \"from convert_masks import mask_to_polygon\"",
                "success": False,
                "output": "ImportError: libGL.so.1: cannot open shared object file",
            },
        ],
        artifacts=["/app/convert_masks.py"],
        metadata={"expected_artifacts": ["/app/convert_masks.py"]},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "structured_csv_table_contract"
    assert result.affected_components == [
        "bench/agent",
        "bench/harbor_adapter",
        "crates/hl-worker-core",
        "harness/tools/shell",
        "recovery/patterns",
        "harness/tools/verify",
        "verification/checks",
    ]


def test_failure_attribution_separates_environment_start_timeout():
    trial = TrialResult(
        trial_id="env-start-timeout",
        task_id="task-a",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.TIMEOUT,
        score=0.0,
        verified=False,
        error_log=["Environment start timed out after 600.0 seconds"],
        metadata={"timeout_phase": "environment_start"},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "environment_start_timeout"
    assert result.affected_components == ["bench/harbor", "bench/network_environment"]


def test_failure_attribution_separates_verifier_runtime_prepare_timeout():
    trial = TrialResult(
        trial_id="verifier-runtime-prepare-timeout",
        task_id="task-a",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.TIMEOUT,
        score=0.0,
        verified=False,
        error_log=["Verifier runtime network preparation timed out after 90 seconds"],
        metadata={
            "timeout_phase": "verifier_runtime_prepare",
            "verifier_runtime_prepare_timeout": True,
        },
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "verifier_runtime_prepare_timeout"
    assert result.affected_components == ["bench/network_environment", "bench/harbor"]


def test_failure_attribution_prefers_specific_verifier_runtime_prepare_timeout_over_infra():
    trial = TrialResult(
        trial_id="verifier-runtime-prepare-after-done",
        task_id="task-a",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.ERROR,
        score=0.0,
        verified=False,
        error_log=["Command timed out after 90 seconds"],
        trajectory=[{"type": "tool_call", "tool": "done", "success": True}],
        metadata={
            "timeout_phase": "verifier_runtime_prepare",
            "verifier_runtime_prepare_timeout": True,
            "verifier_infra_error": True,
            "post_completion_agent_exception": False,
        },
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "verifier_runtime_prepare_timeout"
    assert result.affected_components == ["bench/network_environment", "bench/harbor"]


def test_failure_attribution_extracts_structured_numeric_verifier_contracts():
    trial = TrialResult(
        trial_id="video-near-miss",
        task_id="video-processing",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "## ctrf.json\n"
            "def _test_jump_analyzer_video(video_path, takeoff_range=None, "
            "landing_range=None):\n"
            "output_path = Path('/app/output.toml')\n"
            "required_fields = ['jump_takeoff_frame_number', "
            "'jump_land_frame_number']\n"
            "video_path = '/app/example_video.mp4', takeoff_range = (50, 54)\n"
            "landing_range = (62, 64)\n"
            "Frame validation uses inclusive ranges only: provide "
            "(min_frame, max_frame)",
            "AssertionError: Takeoff frame 55 not within inclusive range [50, 54]",
        ],
        metadata={
            "verifier_infra_error": False,
            "verifier_logs": "Hit:1 http://deb.debian.org/debian bookworm InRelease",
        },
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == (
        "numeric_interval_contract,structured_output_schema_contract"
    )
    assert result.affected_components == [
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    ]


def test_failure_attribution_ignores_verifier_source_timeout_without_timeout_metadata():
    trial = TrialResult(
        trial_id="verified-semantic-timeout-text",
        task_id="cancel-async-tasks",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "## ctrf.json\n"
            "trace: stdout, stderr = proc.communicate(timeout=5)\n"
            "message: output did not match expected cleanup behavior"
        ],
        metadata={"verifier_infra_error": False},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "verifier_mismatch"
    assert result.affected_components == [
        "verification/checks",
        "harness/tools/verify",
    ]


def test_failure_attribution_prefers_verifier_assertion_over_worker_ssl_noise():
    trial = TrialResult(
        trial_id="semantic-after-worker-network-noise",
        task_id="generic-output-task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "## ctrf.json\n"
            '{"tests":[{"file_path":"test_outputs.py","raw_status":"call_failed",'
            '"trace":"E       AssertionError: output artifact was wrong"}]}',
        ],
        trajectory=[
            {
                "type": "tool_call",
                "tool": "bash",
                "success": False,
                "output": (
                    "SSLError: certificate verify failed: unable to get local "
                    "issuer certificate"
                ),
            }
        ],
        metadata={"verifier_infra_error": False},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "verifier_mismatch"
    assert result.affected_components == [
        "verification/checks",
        "harness/tools/verify",
    ]


def test_failure_attribution_keeps_explicit_verifier_infra_precedence():
    trial = TrialResult(
        trial_id="explicit-verifier-infra",
        task_id="generic-output-task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "## ctrf.json\n"
            '{"tests":[{"file_path":"test_outputs.py","raw_status":"call_failed",'
            '"trace":"E       AssertionError: verifier setup artifact missing"}]}',
        ],
        metadata={
            "verifier_infra_error": True,
            "verifier_logs": "curl: (60) SSL certificate problem\nuvx: command not found",
        },
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "verifier_environment_error"
    assert result.affected_components == ["bench/harbor", "verification/checks"]


def test_failure_attribution_routes_verifier_cache_permission_to_infra():
    trial = TrialResult(
        trial_id="verifier-cache-permission",
        task_id="mailman",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.HARD,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "## test-stdout.txt\n"
            "  x Failed to download and build `mailman==3.3.8`\n"
            "  |- Failed to write to the distribution cache\n"
            "  `- failed to rename file from /tmp/hl-verifier-cache/uv/.tmp2XLEC3 "
            "to /tmp/hl-verifier-cache/uv/archive-v0/oxEyQsybl_cOPz6B3GE5T: "
            "Permission denied (os error 13)\n"
            "error: Failed to spawn: `pytest`\n"
            "  Caused by: No such file or directory (os error 2)"
        ],
        metadata={"verifier_infra_error": False},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "verifier_environment_error"
    assert result.affected_components == [
        "bench/harbor",
        "bench/network_environment",
        "verification/checks",
    ]


def test_failure_attribution_prefers_verified_reward_over_provider_warning_noise():
    trial = TrialResult(
        trial_id="verified-fail-with-provider-warning",
        task_id="generic-output-task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output='{"reward": 0.0}',
        error_log=[
            "OpenAIException warning: transient 502 Bad Gateway while polling logs",
        ],
        harbor_stderr=(
            "client warning: auth_unavailable while refreshing provider telemetry\n"
            "Temporary failure resolving metrics endpoint"
        ),
        metadata={"verifier_infra_error": False},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "verifier_mismatch"
    assert result.affected_components == [
        "verification/checks",
        "harness/tools/verify",
    ]


def test_failure_attribution_routes_worker_dependency_noise_to_shell_layer():
    trial = TrialResult(
        trial_id="worker-dependency-noise",
        task_id="generic-build-task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=False,
        error_log=["Worker command failed before verifier evidence was available"],
        trajectory=[
            {
                "type": "tool_call",
                "tool": "bash",
                "success": False,
                "output": (
                    "pip install example\n"
                    "SSLError: certificate verify failed: unable to get local "
                    "issuer certificate\n"
                    "ModuleNotFoundError: No module named 'example'"
                ),
            }
        ],
        metadata={"verifier_infra_error": False},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "dependency_issue"
    assert result.affected_components == ["tools/shell", "recovery/patterns"]


def test_failure_attribution_separates_exception_after_done_from_dependency_noise():
    trial = TrialResult(
        trial_id="done-then-timeout",
        task_id="generic-output-task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.ERROR,
        score=0.0,
        verified=False,
        error_log=["Command timed out after 90 seconds"],
        trajectory=[
            {
                "type": "tool_call",
                "tool": "bash",
                "success": False,
                "error": "pip install dependency failed",
            },
            {"type": "tool_call", "tool": "done", "success": True},
        ],
        metadata={
            "post_completion_agent_exception": True,
            "agent_exception_type": "RuntimeError",
            "agent_exception_message": "Command timed out after 90 seconds",
        },
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "post_completion_agent_exception"
    assert result.affected_components == [
        "bench/harbor",
        "bench/harbor_adapter",
        "bench/agent",
    ]


def test_failure_attribution_infers_legacy_exception_after_done_from_trajectory():
    trial = TrialResult(
        trial_id="legacy-done-then-timeout",
        task_id="generic-output-task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.ERROR,
        score=0.0,
        verified=False,
        error_log=["Command timed out after 90 seconds"],
        trajectory=[
            {
                "type": "tool_call",
                "tool": "bash",
                "success": False,
                "error": "pip install dependency failed",
            },
            {"type": "tool_call", "tool": "done", "success": True},
        ],
        metadata={"agent_exception_type": "RuntimeError"},
    )

    result = FailureAttributor().analyze(trial)

    assert result.failure_category == "post_completion_agent_exception"
    assert result.affected_components == [
        "bench/harbor",
        "bench/harbor_adapter",
        "bench/agent",
    ]


class TestFileSystemMemory:
    def test_constructor_does_not_create_runtime_directories(self, tmp_path):
        memory_path = tmp_path / "trials"

        memory = FileSystemMemory(base_path=str(memory_path))

        assert memory.base_path == str(memory_path)
        assert not memory_path.exists()
        assert memory.list_trials() == []
        assert memory.get_latest_summary() is None
        assert memory.list_patches() == []

    def test_record_and_retrieve_trial(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        trial = TrialResult(
            trial_id="test_001",
            task_id="test::hello::1.0",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
        )
        memory.record_trial(trial)
        retrieved = memory.get_trial("test_001")
        assert retrieved.trial_id == "test_001"
        assert retrieved.status == TrialStatus.PASSED

    def test_list_trials(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        for i in range(3):
            trial = TrialResult(
                trial_id=f"test_{i:03d}",
                task_id=f"task_{i}",
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.EASY,
                status=TrialStatus.PASSED,
            )
            memory.record_trial(trial)
        trials = memory.list_trials()
        assert len(trials) == 3

    def test_list_trials_filtered(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.record_trial(TrialResult(
            trial_id="a", task_id="task_1",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
        ))
        memory.record_trial(TrialResult(
            trial_id="b", task_id="task_2",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
        ))
        filtered = memory.list_trials(task_id="task_1")
        assert len(filtered) == 1

    def test_list_trials_filtered_prefers_harbor_task_prefix(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.record_trial(TrialResult(
            trial_id="target__001", task_id="target",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
        ))

        unrelated = tmp_path / "runs" / "other__001"
        unrelated.mkdir(parents=True)
        (unrelated / "result.json").write_text("{not valid json")

        assert memory.list_trials(task_id="target") == ["target__001"]

    def test_record_summary(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        summary = TrialSummary(
            summary_id="sum_001",
            total_tasks=10,
            passed=8,
            failed=2,
            overall_score=0.8,
        )
        memory.record_summary(summary)
        latest = memory.get_latest_summary()
        assert latest is not None
        assert latest.overall_score == 0.8

    def test_get_latest_summary_skips_campaign_reports(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        summary = TrialSummary(summary_id="sum_001", total_tasks=1, overall_score=0.8)
        memory.record_summary(summary)
        (tmp_path / "summaries" / "zzz_campaign.json").write_text(
            json.dumps({"campaign_id": "local", "score_history": []})
        )

        latest = memory.get_latest_summary()

        assert latest is not None
        assert latest.summary_id == "sum_001"

    def test_get_latest_summary_empty(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        assert memory.get_latest_summary() is None

    def test_regression_snapshot(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        snapshot = RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
            required_assertions=["Must pass"],
        )
        memory.save_regression("task_solved", snapshot)
        retrieved = memory.get_regression_snapshot("task_solved")
        assert retrieved is not None
        assert retrieved.task_id == "task_solved"

    def test_check_regression_passed(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.save_regression("task_solved", RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
        ))
        result = TrialResult(
            trial_id="t", task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
        )
        assert memory.check_regression("task_solved", result) is False

    def test_check_regression_failed(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.save_regression("task_solved", RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
        ))
        result = TrialResult(
            trial_id="t", task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
            score=0.0,
        )
        assert memory.check_regression("task_solved", result) is True

    def test_check_regression_accepts_verified_reward_pass_with_timeout(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.save_regression("task_solved", RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
        ))
        result = TrialResult(
            trial_id="t", task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.TIMEOUT,
            score=1.0,
            verified=True,
            metadata={"completion_hygiene_warning": True},
        )
        assert memory.check_regression("task_solved", result) is False

    def test_check_regression_ignores_non_stable_snapshot(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.save_regression("task_solved", RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
            validation_status="invalidated",
            invalidation_reason="post-regression failed",
        ))
        result = TrialResult(
            trial_id="t", task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
            score=0.0,
        )
        assert memory.check_regression("task_solved", result) is False

    def test_check_regression_only_compares_same_model_scope(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        pro_scope = (
            model_scope_from_config(
                {
                    "provider": "openai_compatible",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-v4-pro",
                    "reasoning_effort": "max",
                    "max_output_tokens": "8000",
                }
            )
        )
        flash_scope = (
            model_scope_from_config(
                {
                    "provider": "openai_compatible",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-v4-flash",
                    "reasoning_effort": "max",
                    "max_output_tokens": "8000",
                }
            )
        )
        memory.save_regression(
            "task_solved",
            RegressionSnapshot(
                task_id="task_solved",
                harness_version="0.1.0",
                model_scope=pro_scope,
                validation_status="stable",
            ),
        )
        flash_result = TrialResult(
            trial_id="t",
            task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
            score=0.0,
            metadata={
                "model_config": {
                    "provider": "openai_compatible",
                    "base_url_host": "api.deepseek.com",
                    "model": "deepseek-v4-flash",
                    "reasoning_effort": "max",
                    "max_output_tokens": "8000",
                }
            },
        )

        assert memory.check_regression("task_solved", flash_result) is False

        memory.save_regression(
            "task_solved",
            RegressionSnapshot(
                task_id="task_solved",
                harness_version="0.1.0",
                model_scope=flash_scope,
                validation_status="stable",
            ),
        )

        assert memory.check_regression("task_solved", flash_result) is True

    def test_mark_and_invalidate_regression_snapshot(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.save_regression("task_solved", RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
            source_summary_id="summary_001",
            validation_status="pending",
        ))

        assert memory.mark_regression_stable(
            "task_solved",
            source_summary_id="other",
        ) is False
        assert memory.mark_regression_stable(
            "task_solved",
            source_summary_id="summary_001",
        ) is True
        assert memory.get_regression_snapshot("task_solved").validation_status == "stable"

        assert memory.invalidate_regression(
            "task_solved",
            source_summary_id="summary_001",
            reason="post-regression failed",
        ) is True
        snapshot = memory.get_regression_snapshot("task_solved")
        assert snapshot.validation_status == "invalidated"
        assert snapshot.invalidation_reason == "post-regression failed"

    def test_record_regression_run_and_transient_cooldown(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        memory.save_regression(
            "task_solved",
            RegressionSnapshot(
                task_id="task_solved",
                harness_version="0.1.0",
                validation_status="stable",
            ),
        )
        result = TrialResult(
            trial_id="t",
            task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
            verified=True,
            wall_time_seconds=12.5,
        )

        assert memory.record_regression_run("task_solved", result) is True
        assert memory.record_regression_transient_failure(
            "task_solved",
            reason="retry passed",
            cooldown_seconds=60,
        ) is True
        snapshot = memory.get_regression_snapshot("task_solved")
        assert snapshot.regression_runs == 1
        assert snapshot.last_regression_status == "passed"
        assert snapshot.last_regression_wall_time_seconds == 12.5
        assert snapshot.regression_transient_failures == 1
        assert snapshot.regression_cooldown_until is not None

    def test_check_regression_nonexistent(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        result = TrialResult(
            trial_id="t", task_id="never_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
        )
        assert memory.check_regression("never_solved", result) is False

    def test_save_and_list_patches(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        patch = HarnessPatch(
            component_name="prompts/system",
            before_version="0.1.0",
            after_version="0.1.1",
            file_path="harness/prompts/system.py",
            diff="- old\n+ new",
            rationale="Fix timeout guidance",
        )
        memory.save_patch(patch)
        patches = memory.list_patches()
        assert len(patches) == 1

    def test_record_trial_writes_failure_attribution(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        trial = TrialResult(
            trial_id="failed_001",
            task_id="task_failed",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
            score=0.0,
            error_log=["bash: rg: command not found"],
            tool_calls=[{"tool": "bash", "success": False, "error": "command not found"}],
        )

        memory.record_trial(trial)

        trial_dir = tmp_path / "runs" / "failed_001"
        feedback = json.loads((trial_dir / "feedback.json").read_text())
        handoff = (trial_dir / "handoff.md").read_text()
        assert feedback["failure_category"] == "dependency_issue"
        assert feedback["affected_components"] == ["tools/shell", "recovery/patterns"]
        assert feedback["component_confidence"]["tools/shell"] == 0.7
        assert "- Failure category: dependency_issue" in handoff

    def test_record_trial_attributes_verifier_environment_error(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        trial = TrialResult(
            trial_id="failed_verifier_env",
            task_id="fix-git",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
            score=0.0,
            verified=True,
            verifier_output='{"reward": 0.0}',
            metadata={
                "verifier_infra_error": True,
                "verifier_logs": "curl: (60) SSL certificate problem\nuvx: command not found",
            },
        )

        memory.record_trial(trial)

        feedback = json.loads(
            (tmp_path / "runs" / "failed_verifier_env" / "feedback.json").read_text()
        )
        assert feedback["failure_category"] == "verifier_environment_error"
        assert feedback["affected_components"] == ["bench/harbor", "verification/checks"]

    def test_attach_codex_update_copies_artifacts_to_trial_dir(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        memory.record_trial(
            TrialResult(
                trial_id="failed_001",
                task_id="task_failed",
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.EASY,
                status=TrialStatus.FAILED,
            )
        )
        artifacts_dir = tmp_path / "codex"
        artifacts_dir.mkdir()
        packet = artifacts_dir / "packet.json"
        events = artifacts_dir / "events.jsonl"
        final = artifacts_dir / "final.json"
        diff = artifacts_dir / "git.diff"
        packet.write_text('{"packet_id":"p1"}')
        events.write_text('{"event":"started"}\n')
        final.write_text('{"status":"edited"}')
        diff.write_text("diff --git a/bench/agent.py b/bench/agent.py\n")

        memory.attach_codex_update(
            "failed_001",
            packet_path=packet,
            events_path=events,
            final_message_path=final,
            diff_path=diff,
        )

        trial_dir = tmp_path / "trials" / "runs" / "failed_001"
        assert json.loads((trial_dir / "codex_update_packet.json").read_text()) == {
            "packet_id": "p1"
        }
        assert "started" in (trial_dir / "codex_events.jsonl").read_text()
        manifest = json.loads((trial_dir / "codex_update_manifest.json").read_text())
        assert manifest["copied"]["codex_update_packet.json"] is True
        assert manifest["sources"]["packet_path"] == str(packet)

    def test_save_component_lesson_dedupes_identical_body(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        lesson = "Repeated failure category `verifier_mismatch` on verification/checks."

        memory.save_component_lesson("verification_checks", lesson)
        memory.save_component_lesson("verification_checks", lesson)
        memory.save_component_lesson("verification_checks", lesson)

        text = (memory.component_lessons_dir / "verification_checks.md").read_text()
        # The lesson body must appear exactly once despite three saves.
        assert text.count(lesson) == 1

    def test_save_component_lesson_keeps_distinct_lessons(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))

        memory.save_component_lesson("tools_shell", "First distinct lesson body.")
        memory.save_component_lesson("tools_shell", "Second distinct lesson body.")

        text = (memory.component_lessons_dir / "tools_shell.md").read_text()
        assert "First distinct lesson body." in text
        assert "Second distinct lesson body." in text

    def test_save_component_lesson_dedupe_ignores_trial_id_and_whitespace(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        lesson = "Guard bloat detected in worker loop; prefer generic recovery."

        memory.save_component_lesson("tools_shell", lesson, source_trial_id="t_aaa")
        memory.save_component_lesson(
            "tools_shell", lesson + "\n", source_trial_id="t_bbb"
        )

        text = (memory.component_lessons_dir / "tools_shell.md").read_text()
        # Same body under different trial ids / trailing whitespace collapses to one.
        assert text.count(lesson) == 1

    def test_save_component_lesson_refreshes_recurring_lesson(self, tmp_path):
        # A recurring lesson should move to the end (most recent) with an updated
        # occurrence count rather than accumulate duplicate blocks.
        memory = FileSystemMemory(base_path=str(tmp_path))
        recurring = "Recurring: verifier_mismatch needs held-out check."
        other = "One-off: fix docker CA."

        memory.save_component_lesson("verification_checks", recurring)
        memory.save_component_lesson("verification_checks", other)
        memory.save_component_lesson("verification_checks", recurring)

        text = (memory.component_lessons_dir / "verification_checks.md").read_text()
        assert text.count(recurring) == 1
        assert text.count(other) == 1
        # The refreshed recurring lesson is now after the one-off entry.
        assert text.index(recurring) > text.index(other)

    def test_save_component_lesson_bounds_retention(self, tmp_path):
        # Distinct lessons that keep recurring (each with a slightly different
        # body) must not grow the playbook without bound; keep the most-recent N.
        memory = FileSystemMemory(base_path=str(tmp_path))
        cap = memory.component_lesson_max_entries
        assert cap >= 5

        for i in range(cap + 10):
            memory.save_component_lesson(
                "verification_checks",
                f"verifier_mismatch on tasks batch-{i}",
            )

        text = (memory.component_lessons_dir / "verification_checks.md").read_text()
        block_count = text.count("\n## ")
        assert block_count == cap
        # The oldest entries are dropped; the newest is retained.
        assert "batch-0" not in text
        assert f"batch-{cap + 9}" in text
