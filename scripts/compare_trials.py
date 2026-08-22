#!/usr/bin/env python3
"""Compare two trials and show score changes.

Usage:
  python scripts/compare_trials.py trial_001 trial_002
  python scripts/compare_trials.py --latest
"""

import argparse
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
    args = parser.parse_args()

    from hl.memory import FileSystemMemory

    memory_path = Path(args.memory_path)
    memory = FileSystemMemory(base_path=str(memory_path))

    if args.latest:
        all_trials = memory.list_trials()
        if len(all_trials) < 2:
            print("Need at least 2 trials to compare")
            return
        args.trial_a = all_trials[-2]
        args.trial_b = all_trials[-1]

    if not args.trial_a or not args.trial_b:
        parser.print_help()
        return

    try:
        trial_a = memory.get_trial(args.trial_a)
        trial_b = memory.get_trial(args.trial_b)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    delta = trial_b.score - trial_a.score
    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"

    print(f"Comparing {args.trial_a} → {args.trial_b}")
    print(f"  Task: {trial_a.task_id}")
    print(f"  Score: {trial_a.score:.4f} → {trial_b.score:.4f} ({direction} {abs(delta):.4f})")
    print(f"  Status: {trial_a.status.value} → {trial_b.status.value}")
    print(f"  Wall time: {trial_a.wall_time_seconds:.1f}s → {trial_b.wall_time_seconds:.1f}s")

    if trial_b.error_log and trial_b.status.value == "failed":
        print(f"\n  New errors:")
        for err in trial_b.error_log[:5]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
