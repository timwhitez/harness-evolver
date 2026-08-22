#!/usr/bin/env python3
"""Audit guard hits and strict benefit evidence from campaign analysis reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hl.guard_hits import build_guard_hit_benefit_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify guard policies by hit count and strict fail-to-pass evidence."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--analysis-root", default="trials/analysis")
    parser.add_argument(
        "--campaign-summary",
        default=None,
        help="Optional campaign JSON used for same-task fail-to-pass benefit detection.",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Limit analysis summaries to trials/analysis/<campaign-id>/summary_*/summary.json.",
    )
    parser.add_argument(
        "--max-summaries",
        type=int,
        default=None,
        help="Use only the last N matching analysis summaries.",
    )
    parser.add_argument("--pass-score", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    audit = build_guard_hit_benefit_audit(
        repo_root=Path(args.repo_root),
        analysis_root=Path(args.analysis_root),
        campaign_summary_path=Path(args.campaign_summary)
        if args.campaign_summary
        else None,
        campaign_id=args.campaign_id,
        max_summaries=args.max_summaries,
        pass_score=args.pass_score,
    )
    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        _print_text(audit)
    return 0


def _print_text(audit: dict[str, object]) -> None:
    counts = audit["classification_counts"]
    print(f"Analysis summaries: {audit['analysis_summary_count']}")
    print(f"Guard policies: {audit['guard_policy_count']}")
    print(
        "Classifications: "
        f"zero-hit={counts['zero_hit']}, "
        f"hit-zero-benefit={counts['hit_zero_benefit']}, "
        f"beneficial={counts['beneficial']}"
    )
    for label, key in (
        ("Zero-hit guards", "zero_hit_guards"),
        ("Hit but zero strict benefit", "hit_zero_benefit_guards"),
        ("Strict beneficial guards", "beneficial_guards"),
    ):
        values = audit.get(key) or []
        preview = ", ".join(str(item) for item in values[:12])
        suffix = "" if len(values) <= 12 else f", ... (+{len(values) - 12})"
        print(f"{label}: {preview}{suffix}" if preview else f"{label}: none")


if __name__ == "__main__":
    raise SystemExit(main())
