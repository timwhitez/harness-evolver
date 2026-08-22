"""T2/T3 regression: worker_loop multi-layer mapping + dirty-baseline exemption."""

import sys

from meta.codex_update import (
    CodexUpdateEngine,
    _report_categories_for_changed_files,
)
from meta.reviewer import PatchReviewResult

# Reuse the shared contract fixture helpers from the main suite.
from tests.test_meta_codex_update import (
    contract_report_fields,
    contract_report_script_lines,
    failed_trial,
    _init_repo,
)


def test_worker_loop_maps_to_multiple_functional_layers():
    categories = _report_categories_for_changed_files(
        ["crates/hl-worker-core/src/main.rs"]
    )
    # T2: the single-file Worker core carries planning/tool/recovery/etc, so all
    # those functional layers are acceptable report categories.
    assert "planning" in categories
    assert "recovery" in categories
    assert "verification" in categories
    assert "adapter" in categories


def test_report_gate_accepts_planning_layer_for_worker_core(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    changed = ["crates/hl-worker-core/src/main.rs"]
    report = {
        "status": "edited",
        "summary": "refined planning loop in worker core",
        "changed_files": changed,
        "validation_commands": ["pytest tests/ -v"],
        "component_type": "planning",
        **contract_report_fields(),
    }
    report["implementation_scope"] = {
        "primary_layer": "planning",
        "architectural_change_considered": True,
        "structural_files_changed": changed,
        "why_prompt_only_is_sufficient": "not a prompt-only update",
    }

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=changed),
        exit_code=0,
        final_report=report,
        required_validation_commands=["pytest tests/ -v"],
    )

    assert review.accepted is True
    assert not any(
        "primary_layer or component_type must match" in reason
        for reason in review.reasons
    )


def test_report_gate_exempts_dirty_baseline_from_over_report(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    changed = ["bench/agent.py"]
    report = {
        "status": "edited",
        "summary": "bounded worker edit",
        # Codex saw AGENTS.md dirty in git status and listed it, but the isolated
        # delta only touched bench/agent.py.
        "changed_files": ["bench/agent.py", "AGENTS.md"],
        "validation_commands": ["pytest tests/ -v"],
        "component_type": "adapter",
        **contract_report_fields(),
    }
    report["implementation_scope"] = {
        "primary_layer": "adapter",
        "architectural_change_considered": True,
        "structural_files_changed": ["bench/agent.py", "AGENTS.md"],
        "why_prompt_only_is_sufficient": "not a prompt-only update",
    }

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=changed),
        exit_code=0,
        final_report=report,
        required_validation_commands=["pytest tests/ -v"],
        ignore_files=["AGENTS.md"],
    )

    assert review.accepted is True
    assert not any("includes files not changed" in r for r in review.reasons)
    assert not any(
        "structural_files_changed includes files not changed" in r
        for r in review.reasons
    )


def test_report_gate_still_flags_genuine_over_report(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    changed = ["bench/agent.py"]
    report = {
        "status": "edited",
        "summary": "bounded worker edit",
        "changed_files": ["bench/agent.py", "hl/loop.py"],
        "validation_commands": ["pytest tests/ -v"],
        "component_type": "adapter",
        **contract_report_fields(),
    }
    report["implementation_scope"] = {
        "primary_layer": "adapter",
        "architectural_change_considered": True,
        "structural_files_changed": ["bench/agent.py"],
        "why_prompt_only_is_sufficient": "not a prompt-only update",
    }

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=changed),
        exit_code=0,
        final_report=report,
        required_validation_commands=["pytest tests/ -v"],
        ignore_files=["AGENTS.md"],
    )

    # hl/loop.py is not in the baseline ignore set, so it is still over-reported.
    assert review.accepted is False
    assert any("includes files not changed" in r for r in review.reasons)


def test_run_update_exempts_untouched_dirty_baseline_file(tmp_path):
    _init_repo(tmp_path)
    validation = tmp_path / "validation_ok.py"
    validation.write_text("print('validation ok')\n")
    validation_command = f"{sys.executable} {validation}"
    # Dirty, untracked baseline file that Codex will list but not modify.
    (tmp_path / "NOTES.md").write_text("scratch notes\n")

    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "final_path = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
        "pathlib.Path('bench/agent.py').write_text('original\\nCODEX_EDIT = True\\n')\n"
        "final_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "final_path.write_text(json.dumps({\n"
        "  'status': 'edited',\n"
        "  'summary': 'bounded worker edit with dirty baseline note',\n"
        "  'changed_files': ['bench/agent.py', 'NOTES.md'],\n"
        f"  'validation_commands': [{validation_command!r}],\n"
        "  'component_type': 'adapter',\n"
        + contract_report_script_lines()
        + "}))\n"
    )
    fake_codex.chmod(0o755)

    engine = CodexUpdateEngine(
        repo_root=tmp_path,
        codex_bin=str(fake_codex),
        events_dir=tmp_path / "diffs",
    )
    result = engine.run_update(
        failures=[failed_trial()],
        current_harness={"version": "x"},
        required_validation_commands=[validation_command],
    )

    assert result.review.accepted is True, result.review.reasons
    assert "NOTES.md" not in result.review.changed_files
    assert not any(
        "includes files not changed" in reason for reason in result.review.reasons
    )
