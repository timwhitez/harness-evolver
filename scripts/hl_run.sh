#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env.local ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env.local
  set +a
fi

task_args=()
has_task_arg=false
for arg in "$@"; do
  case "$arg" in
    --task|--task=*|--tasks|--tasks=*|--task-file|--task-file=*|--task-selection|--task-selection=*|--task-set|--task-set=*|--task-index|--task-index=*|--task-indices|--task-indices=*)
      has_task_arg=true
      ;;
  esac
done
if [[ "$has_task_arg" == false && -n "${HL_TASK:-}" ]]; then
  task_args=(--task "$HL_TASK")
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  set +e
  python scripts/setup_wizard.py --non-interactive --dry-run
  SETUP_STATUS=$?
  set -e
  python scripts/run_campaign.py "${task_args[@]}" --dry-run "$@"
  exit "$SETUP_STATUS"
fi

if [[ "${1:-}" == "--once" ]]; then
  shift
  python scripts/run_campaign.py "${task_args[@]}" "$@"
  exit $?
fi

python - <<'PY'
from pathlib import Path

from hl.loop import HLLoop

print("HL campaign controller is ready.")
print("Use --once for one Worker trial or --dry-run for preflight command validation.")
print(f"Local config present: {Path('config/local.yaml').exists()}")
loop = HLLoop()
print(loop.get_progress())
PY
