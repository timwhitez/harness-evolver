# Docker Resource Safety

HarnessEvolver runs real Harbor/TerminalBench jobs. Docker build, pull,
extract, and task containers can overlap with other local agents and exhaust WSL
memory or swap. The checked-in defaults now prefer single-task execution and
bounded containers.

## Current Policy

- `config/trials.yaml` defaults `execution.round_task_concurrency` to `1`.
- `config/trials.yaml` defaults `regression.task_concurrency` to `1`.
- `docker_resources` defaults to `memory=4g`, `memory_swap=4g`, `cpus=2`,
  `pids_limit=1024`, HL labels, and `json-file` log rotation options.
- Harbor commands receive `--override-cpus`, `--override-memory-mb`, and
  explicit `--delete`.
- The custom Docker compose environment injects `mem_limit`, `memswap_limit`,
  `cpus`, `pids_limit`, labels, and per-container `json-file` log options.
- The custom environment replaces Harbor's default
  `docker compose down --rmi all --volumes --remove-orphans` with
  `docker compose down --remove-orphans`, then prunes only stopped containers
  with `com.harness-evolver.managed=true` and
  `com.harness-evolver.cleanup=task`.
- `scripts/network_preflight.py --full` uses resource-limited `docker run`
  containers with `--rm`.
- Large prebuilt image pulls are opt-in because `docker pull` and layer extract
  cannot be memory-capped with container flags.

Volumes are never deleted by the runner or cleanup script. Do not delete volumes
unless volume names, purpose, owners, and data-loss risk have been listed and
explicitly confirmed.

## Safe Before/After Check

Run these before a benchmark:

```bash
docker ps -a --filter label=com.harness-evolver.managed=true
docker system df -v
docker stats --no-stream
```

Run a single task:

```bash
python scripts/run_trial.py \
  --path terminal-bench-tasks/terminal-bench \
  --task fix-git \
  --worker-role worker_deepseek \
  --docker-memory 4g \
  --docker-memory-swap 4g \
  --docker-cpus 2 \
  --docker-pids-limit 1024
```

Run the same checks after the task:

```bash
docker ps -a --filter label=com.harness-evolver.managed=true
docker system df -v
docker stats --no-stream
```

Clean stopped HL task containers only:

```bash
python scripts/docker_cleanup.py --mode containers --execute
```

## Limited Concurrency

Concurrent benchmark tasks must prove that the projected per-container memory
peak stays within 60% of current WSL `MemAvailable`. The proof is enforced by
`scripts/run_campaign.py` and `scripts/regression_check.py` before launching
parallel Harbor jobs.

Example with two 2g task containers:

```bash
grep MemAvailable /proc/meminfo
python scripts/run_campaign.py \
  --dry-run \
  --task fix-git \
  --task vulnerable-secret \
  --round-task-concurrency 2 \
  --worker-role worker_deepseek \
  --docker-memory 2g \
  --docker-memory-swap 2g \
  --docker-cpus 2 \
  --docker-pids-limit 1024
```

The dry-run JSON includes `docker_concurrency_budget` with:

- `memavailable_mb`
- `allowed_peak_mb`
- `projected_peak_mb`
- `within_60_percent_memavailable`

If the projected peak is too high, the command fails before Harbor starts.

## Cleanup Modes

Preview cleanup commands:

```bash
python scripts/docker_cleanup.py --mode containers --dry-run
python scripts/docker_cleanup.py --mode conservative --dry-run
python scripts/docker_cleanup.py --mode aggressive --dry-run
```

Execute conservative cleanup:

```bash
python scripts/docker_cleanup.py --mode conservative --execute
```

Execute aggressive cleanup only when you accept slower future builds:

```bash
python scripts/docker_cleanup.py --mode aggressive --execute
```

Mode behavior:

- `containers` removes only stopped containers with HL task labels.
- `conservative` also prunes Docker builder cache older than 168h and dangling
  images.
- `aggressive` prunes builder cache older than 24h and unused images older than
  168h.
- No mode runs `docker system prune -a --volumes`, `docker volume prune`, or
  `docker volume rm`.

List volumes without deleting them:

```bash
python scripts/docker_cleanup.py --list-volumes --json
```

## Docker Daemon Log Rotation

The compose override sets per-container `json-file` log rotation for HL
containers. For host-wide protection, configure Docker daemon logging as root or
administrator:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "20m",
    "max-file": "3"
  }
}
```

Apply through `/etc/docker/daemon.json` or Docker Desktop's daemon settings,
then restart Docker. This is outside the repository and is not changed by the
scripts.
