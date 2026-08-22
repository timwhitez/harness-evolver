# HarnessEvolver

HarnessEvolver is an experimental, self-owned coding-agent harness for
[TerminalBench 2.0](https://www.tbench.ai/). It uses a Heuristic Learning (HL)
loop to turn verified benchmark failures into small, reviewable Worker and
harness improvements.

The evaluated agent is this repository's own Worker loop. Codex, Claude Code,
and other coding agents may be used only outside a benchmark run to analyze
evidence and propose a bounded change to this repository; they are never the
agent being evaluated.

## Status and scope

This is research software. It is useful for experimenting with a transparent
Worker/harness architecture, evidence collection, and guarded policy updates;
it does **not** claim a published TerminalBench score or SOTA result.

The repository intentionally does not vendor TerminalBench task definitions,
reference solutions, Harbor jobs, local models, credentials, or campaign
artifacts. Obtain compatible TerminalBench tasks and Harbor tooling separately,
then point the commands below at your local task checkout.

## Architecture

The main components are:

| Area | Responsibility |
| --- | --- |
| `crates/hl-worker-core/` | The JSONL-driven Worker core: task context, tool calls, todo discipline, recovery, compaction, and completion gates. |
| `bench/` | Harbor adapter, task catalog, trial parser, and Python bridge to the Worker. |
| `harness/` | Editable prompts, tools, planning, context, recovery, and verification policy. |
| `hl/` | Trial memory, goals, regression snapshots, attribution, and the outer HL loop. |
| `meta/` | Evidence packaging, deterministic diff review, guarded Codex updates, and mission-debug suggestions. |
| `scripts/` | Explicit command-line entry points for setup, trials, campaigns, regressions, and reports. |

See [the architecture guide](docs/architecture.md) for the data flow and update
acceptance contract.

## Requirements

- Python 3.11 or later
- Docker and Docker Compose v2 for real Harbor-backed trials
- A compatible Harbor CLI installation and authentication for real trials
- A local TerminalBench task checkout (the default path is
  `terminal-bench-tasks/terminal-bench`)
- A model-provider credential supplied through the environment for any
  non-dry-run Worker invocation
- The Codex CLI only when using the optional external HL updater

## Quick start

Create an isolated Python environment, install development dependencies, and
run the deterministic suite:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

Inspect the local prerequisites without starting a benchmark job:

```bash
python scripts/setup_wizard.py --non-interactive --dry-run
python scripts/workspace_report.py --json
```

Once the task checkout and provider configuration are available, verify command
construction before a live run:

```bash
python scripts/run_trial.py \
  --path terminal-bench-tasks/terminal-bench \
  --task vulnerable-secret \
  --dry-run

python scripts/run_campaign.py \
  --dry-run \
  --tasks fix-git,vulnerable-secret \
  --worker-role worker_deepseek
```

Remove `--dry-run` only after the provider, Docker, Harbor, and task checkout
have been checked locally. Harbor/verifier output—not a Worker self-report—is
the source of truth for a task pass or failure.

## Configuration and secrets

[`config/models.yaml`](config/models.yaml) contains shareable role templates;
it records only environment-variable names. Create local overrides in
`config/local.yaml` and put credentials in `.env.local` or your process
environment. Both files are ignored by Git.

```bash
cp .env.example .env.local
```

For an OpenAI-compatible endpoint, a local override can look like this:

```yaml
models:
  roles:
    worker:
      provider: openai_compatible
      base_url: https://provider.example/v1
      api_key_env: WORKER_API_KEY
      model: provider-model-id
```

Never commit API keys, cookies, bearer tokens, account exports, trial logs, or
local provider URLs. Optional Codex configuration homes and private mirrors
belong in `config/local.yaml`, never in the tracked templates.

## Running and validating changes

Use the narrowest relevant checks while developing, then run the full suite
before proposing a change:

```bash
pytest tests/test_models_and_worker_policy.py -q
pytest tests/ -q
python scripts/run_campaign.py --dry-run --task fix-git --mission-debug
python scripts/regression_check.py --task fix-git --lane smoke
```

Real trial and campaign outputs are intentionally written below `trials/` and
Harbor job directories. They are ignored because they can be large and may
contain operational metadata. Preserve them locally when they are needed for a
specific regression or updater decision; do not add them to source commits.

## Benchmark integrity

The Worker may use only the task workspace, visible task materials, approved
tools, and task-provided checks while solving. Do not:

- edit TerminalBench tests, verifiers, task definitions, or reference solutions;
- give the evaluated Worker external coding agents or benchmark-specific
  solution material;
- inspect hidden verifier data, Harbor internals, or benchmark source while a
  task is being solved;
- treat a model's completion message as a verified pass.

The outer updater accepts at most one bounded Worker/harness improvement slice
at a time and requires a deterministic diff review plus the applicable tests
and regression checks before accepting it.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change. Project-agent
instructions are in [AGENTS.md](AGENTS.md); [CLAUDE.md](CLAUDE.md) keeps the
same repository rules available to Claude Code users.

Please report security-sensitive issues using the process in
[SECURITY.md](SECURITY.md), not in a public issue.

## License

This project is released under the [MIT License](LICENSE).
