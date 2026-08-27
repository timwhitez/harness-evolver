<div align="center">

# HarnessEvolver

**A verifier-grounded, self-improving coding-agent harness for TerminalBench 2.0.**

[![Research Preview](https://img.shields.io/badge/status-research_preview-7c3aed)](#status-and-scope)
[![TerminalBench 2.0](https://img.shields.io/badge/benchmark-TerminalBench_2.0-2563eb)](https://www.tbench.ai/)
[![Rust Worker](https://img.shields.io/badge/worker-Rust-ce422b?logo=rust&logoColor=white)](crates/hl-worker-core/)
[![Python Orchestration](https://img.shields.io/badge/orchestration-Python-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![MIT License](https://img.shields.io/badge/license-MIT-059669)](LICENSE)

[Quick start](#quick-start) · [Architecture](docs/architecture.md) ·
[Configuration](#configuration-and-secrets) · [Contributing](CONTRIBUTING.md)

</div>

HarnessEvolver runs its own Rust Worker against TerminalBench tasks, treats
Harbor/verifier output as ground truth, and converts verified failures into
small, reviewable harness improvements through a Heuristic Learning (HL) loop.

> [!IMPORTANT]
> The evaluated agent is this repository's Worker loop and harness. Codex,
> Claude Code, and other coding agents may analyze evidence and propose bounded
> repository changes only outside benchmark runs; they are never substituted
> for the Worker being evaluated.

## Why HarnessEvolver?

| Principle | Enforced contract |
| --- | --- |
| **Self-owned agent** | Task execution stays inside the Rust Worker and repository harness. |
| **Verifier-grounded** | A model completion is not a pass; Harbor/verifier evidence decides outcomes. |
| **Auditable learning** | Trials, trajectories, failure packets, diffs, and regression decisions remain traceable. |
| **Bounded evolution** | Each updater cycle proposes one focused policy slice, followed by deterministic review and regression gates. |

```text
TerminalBench task → Rust Worker → Harbor verifier → failure evidence
                           ↑                              ↓
                     accepted policy ← review + regression ← bounded update
```

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
- A stable Rust toolchain, or a current prebuilt `hl-worker-core` binary, for
  Worker runs
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
cargo test --manifest-path crates/hl-worker-core/Cargo.toml
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
