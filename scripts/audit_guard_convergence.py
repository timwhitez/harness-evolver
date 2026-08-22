#!/usr/bin/env python3
"""Audit the fixed guard-reduction safety net and convergence budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hl.guard_convergence import build_guard_convergence_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the fixed guard-convergence eval set and guard budget."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--trials-config", default="config/trials.yaml")
    parser.add_argument(
        "--fixed-eval-report",
        default=None,
        help="Optional campaign report JSON containing score_history for convergence checks.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless the safety net is configured and the convergence stop condition is met.",
    )
    args = parser.parse_args()

    audit = build_guard_convergence_audit(
        repo_root=Path(args.repo_root),
        trials_config_path=Path(args.trials_config),
        fixed_eval_report=Path(args.fixed_eval_report) if args.fixed_eval_report else None,
    )
    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        _print_text(audit)

    if args.strict and not audit["convergence"]["converged"]:
        return 1
    return 0


def _print_text(audit: dict[str, object]) -> None:
    fixed_eval = audit["fixed_eval"]
    guard_budget = audit["guard_budget"]
    convergence = audit["convergence"]
    print(f"Ready for guard reduction: {str(audit['ready_for_guard_reduction']).lower()}")
    print(
        "Fixed eval: "
        f"{fixed_eval['task_count']} tasks, {fixed_eval['domain_count']} domains, "
        f"baseline={fixed_eval['baseline_score']}, "
        f"minimum_accept={fixed_eval['minimum_accept_score']}"
    )
    print(
        "Guard budget: "
        f"current={guard_budget['current_total_guard_surface']}, "
        f"baseline={guard_budget['baseline_total_guard_surface']}, "
        f"target={guard_budget['target_total_guard_surface']}, "
        f"reduction={guard_budget['reduction_from_baseline']}"
    )
    print(f"Converged: {str(convergence['converged']).lower()}")
    print(f"Stop condition: {convergence['stop_condition']}")


if __name__ == "__main__":
    raise SystemExit(main())
