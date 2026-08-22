#!/usr/bin/env python3
"""Build a mission-style debug packet from campaign evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from meta.missions import MissionPlanner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a validation-first mission debug packet for the outer HL loop"
    )
    parser.add_argument(
        "--campaign-summary",
        default="trials/summaries/full-scale-deepseek_campaign.json",
        help="Campaign summary JSON produced by scripts/run_campaign.py",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--max-features",
        type=int,
        default=6,
        help=(
            "Compatibility/audit field for older configs. Mission debug emits "
            "all feature candidates; this value is not a stop condition."
        ),
    )
    parser.add_argument(
        "--covered-signature",
        action="append",
        default=[],
        help=(
            "Mechanism signature (failure_category / mechanism slug) already "
            "covered by existing Worker gates/tests. Candidates matching it are "
            "filtered out to avoid re-proposing solved capabilities (repeatable). "
            "If every candidate is covered, none are dropped and "
            "evidence_summary.all_candidates_covered is set so the outer loop can "
            "skip a speculative Codex update."
        ),
    )
    args = parser.parse_args()

    packet = MissionPlanner().from_campaign_file(
        args.campaign_summary,
        max_features=args.max_features,
        covered_mechanism_signatures=args.covered_signature or None,
    )
    payload = packet.model_dump_json(indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload)

    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
