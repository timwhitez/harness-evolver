#!/usr/bin/env python3
"""Compare two trials and show score changes.

Usage:
  python scripts/compare_trials.py trial_001 trial_002
  python scripts/compare_trials.py --latest
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Compare two HL trials")
    parser.add_argument("trial_a", nargs="?", help="First trial ID")
    parser.add_argument("trial_b", nargs="?", help="Second trial ID")
    parser.add_argument("--latest", action="store_true", help="Compare two most recent trials")
    parser.add_argument("--memory-path", type=str, default="trials",
                        help="Path to trial memory store")
    parser.add_argument("--trajectory", action="store_true", help="Compare saved attempt trajectories")
    parser.add_argument("--json", action="store_true", help="Print structured JSON")
    parser.add_argument("--attempt-a", help="Exact saved attempt ID for the first trial")
    parser.add_argument("--attempt-b", help="Exact saved attempt ID for the second trial")
    parser.add_argument("--expand", action="store_true",
                        help="Explicitly display local redacted event parameters/outputs")
    args = parser.parse_args()
    if not args.trajectory and (args.attempt_a or args.attempt_b or args.expand):
        parser.error("--attempt-a, --attempt-b and --expand require --trajectory")

    from hl.memory import FileSystemMemory

    memory_path = Path(args.memory_path)
    memory = FileSystemMemory(base_path=str(memory_path))

    if args.latest:
        all_trials = memory.list_trials()
        if len(all_trials) < 2:
            if args.json:
                print(json.dumps({"status": "selection_required",
                                  "errors": ["Need at least 2 trials to compare"]}))
                return 2
            print("Need at least 2 trials to compare")
            return
        args.trial_a = all_trials[-2]
        args.trial_b = all_trials[-1]

    if not args.trial_a or not args.trial_b:
        parser.print_help()
        return

    if args.trajectory:
        from bench.trajectory_compare import compare_trajectories

        report = compare_trajectories(
            memory.runs_dir / args.trial_a / "result.json",
            memory.runs_dir / args.trial_b / "result.json",
            attempt_a=args.attempt_a, attempt_b=args.attempt_b, expand=args.expand,
        )
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            _print_trajectory(report)
        return 2 if report["status"] in {
            "input_error", "identity_mismatch", "selection_required"} else 0

    try:
        trial_a = memory.get_trial(args.trial_a)
        trial_b = memory.get_trial(args.trial_b)
    except (ValueError, OSError) as error:
        if args.json:
            print(json.dumps({"status": "input_error", "errors": ["unreadable or invalid trial input"]}))
            return 2
        if isinstance(error, FileNotFoundError):
            print(f"Error: {error}")
            return
        raise

    delta = trial_b.score - trial_a.score
    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"

    if args.json:
        from bench.trajectory_compare import _safe

        print(json.dumps(_safe({
            "trial_a": trial_a.trial_id, "trial_b": trial_b.trial_id,
            "task_a": trial_a.task_id, "task_b": trial_b.task_id,
            "score_a": trial_a.score, "score_b": trial_b.score, "score_delta": delta,
            "status_a": trial_a.status.value, "status_b": trial_b.status.value,
            "wall_time_a": trial_a.wall_time_seconds, "wall_time_b": trial_b.wall_time_seconds,
        }), indent=2, ensure_ascii=False))
        return 0

    print(f"Comparing {args.trial_a} → {args.trial_b}")
    print(f"  Task: {trial_a.task_id}")
    print(f"  Score: {trial_a.score:.4f} → {trial_b.score:.4f} ({direction} {abs(delta):.4f})")
    print(f"  Status: {trial_a.status.value} → {trial_b.status.value}")
    print(f"  Wall time: {trial_a.wall_time_seconds:.1f}s → {trial_b.wall_time_seconds:.1f}s")

    if trial_b.error_log and trial_b.status.value == "failed":
        print("\n  New errors:")
        for err in trial_b.error_log[:5]:
            print(f"    - {err}")


def _print_trajectory(report):
    print(f"{report['description']}: {report['status']}")
    print(f"Projection: {report['projection_version']}")
    for label, attempts in report["attempts"].items():
        print(f"{label} available attempts: " + ", ".join(item["attempt_id"] for item in attempts))
    for error in report["errors"]:
        print(f"Error: {error}")
    difference = report["first_difference"]
    if difference:
        print(f"Position {difference['position']}: " + ", ".join(difference["fields"]))
    for label in ("left", "right"):
        if label not in report:
            continue
        side = report[label]
        print(f"{label}: {side['identity']} coverage={side['coverage']}")
        print(f"  Metrics: {json.dumps(side['metrics'], ensure_ascii=False)}")
        if side["gaps"]:
            print("  Gaps: " + ", ".join(side["gaps"]))
        for event in side["events"]:
            print(f"  Event {event['event_index']} ({event['source']}): "
                  + json.dumps(event["projection"], ensure_ascii=False))
            if event["unknown_fields"]:
                print("    Unknown: " + ", ".join(event["unknown_fields"]))
    if report.get("metric_differences"):
        print("Metric differences: " + ", ".join(report["metric_differences"]))


if __name__ == "__main__":
    raise SystemExit(main())
