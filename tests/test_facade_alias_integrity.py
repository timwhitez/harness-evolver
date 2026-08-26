from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_base"),
    [
        ("bench.harbor", "bench._harbor_issue5_logic"),
        ("bench._harbor_issue6_base", "bench._harbor_issue7_identity_base"),
        ("bench._harbor_issue7_base", "bench._harbor_issue9_base"),
        ("bench._harbor_issue6_recovery_base", "bench._harbor_issue6_aggregate_base"),
        ("bench._harbor_issue5_logic", "bench._harbor_issue5_base"),
        ("bench._harbor_issue6_aggregate_base", "bench._harbor_issue6_base"),
        ("bench._harbor_adapter_issue4_base", "bench._harbor_adapter_issue16_base"),
        ("bench.agent", "bench._agent_issue8_base"),
        ("bench._harbor_issue5_base", "bench._harbor_issue6_recovery_base"),
        ("bench._harbor_issue7_identity_base", "bench._harbor_issue7_base"),
        ("harness.config", "harness._config_issue9_base"),
        ("harness.recovery.retry", "harness.recovery._retry_issue20_base"),
        ("harness.tools._search_issue13_base", "harness.tools._search_issue4_fixed_base"),
        ("harness.tools.search", "harness.tools._search_issue13_base"),
        ("harness.tools.shell", "harness.tools._shell_issue3_base"),
        (
            "harness.tools.process_runner",
            "harness.tools._process_runner_issue19_namespace_base",
        ),
        (
            "harness.tools._process_runner_issue19_namespace_base",
            "harness.tools._process_runner_issue19_base",
        ),
        ("hl.types", "hl._types_issue21_base"),
        ("meta.reviewer", "meta._reviewer_issue15_base"),
        ("scripts.run_trial", "scripts._run_trial_issue11_base"),
        ("scripts._run_trial_issue11_base", "scripts._run_trial_issue12_base"),
    ],
)
def test_facade_reexport_preserves_immediate_base_module(
    module_name: str,
    expected_base: str,
) -> None:
    module = import_module(module_name)

    assert module._base.__name__ == expected_base


def test_facade_reexport_preserves_non_base_module_alias() -> None:
    module = import_module("bench._harbor_adapter_issue13_audit")

    assert module._original.__name__ == "bench._harbor_adapter_issue13_impl"
