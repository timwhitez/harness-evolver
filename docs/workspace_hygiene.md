# Workspace Hygiene

This repository intentionally separates source policy, local private config,
runtime evidence, and disposable generated files. Use this document when
deciding whether a dirty worktree entry should be committed, preserved locally,
or removed.

## Categories

### Fixed Baseline

These files define the current project baseline and should be committed when the
implementation is accepted:

- Project instructions and docs: `AGENTS.md`, `CLAUDE.md`, `README.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, `docs/`.
- Versioned config templates: `config/default.yaml`, `config/models.yaml`,
  `config/trials.yaml`, `config/benchmark.yaml`, `.env.example`.
- Source packages: `bench/`, `harness/`, `hl/`, `meta/`, `harness_evolver/`.
- Operator scripts: `scripts/`.
- Test contracts: `tests/`.
- Build metadata: `pyproject.toml`, `.gitignore`.

Fixed baseline does not mean behavior will never change. It means the file is
part of the versioned project truth and should not remain as an untracked
scratch file.

### Iterative Policy

These files are source-controlled, but expected to evolve from Harbor evidence
and regression results:

- Worker policy/runtime: `crates/hl-worker-core/`, `bench/agent.py`,
  `bench/harbor.py`. Runtime prompt/task-context, entrypoint, memory-hint,
  and completion policy live in `crates/hl-worker-core/`; `bench/agent.py`
  remains the Harbor/LiteLLM/tool bridge.
- Harness policy: `harness/recovery/`, `harness/context/`, `harness/tools/`.
  `harness/prompts/` is retained as template/historical utility code rather
  than the active Worker prompt execution path.
- HL policy and submit gates: `hl/loop.py`, `hl/goals.py`,
  `hl/compression.py`, `hl/submit.py`.
- Meta updater and mission debug: `meta/codex_update.py`, `meta/packager.py`,
  `meta/reviewer.py`, `meta/missions.py`.
- Runtime policy config: `config/models.yaml`, `config/trials.yaml`.

Changes here are still commit-worthy, but each future edit should be tied to a
bounded improvement slice and verified by tests, Harbor evidence, or regression
snapshots.

### Local Private Config

These files are machine-local and must not be committed:

- `.env.local`, `.env`, other `*.env` files.
- `config/local.yaml`, `config/local.yml`.
- `.claude/settings.local.json`, `.codex/`, and similar machine-local
  editor/agent state.
- Raw API keys, tokens, cookies, and auth files.

Tracked files may reference environment variable names such as
`OPENAI_API_KEY` or `DEEPSEEK_API_KEY`, but must not contain raw secret values.

### Runtime Evidence

These directories are intentionally ignored and should be preserved unless the
operator explicitly prunes old runs:

- `trials/runs/`
- `trials/summaries/`
- `trials/diffs/`
- `trials/goals/`
- `trials/memory/`
- `trials/regressions/`
- `trials/submissions/`
- `jobs/`
- `terminal-bench-tasks/`

They are not part of the git commit, but they are often the source of truth for
Harbor/verifier results and submit gates. Do not delete them just to make `git
status` shorter.

### Temporary Generated Files

These can be removed when cleaning the workspace:

- `__pycache__/`
- `*.pyc`, `*.pyo`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `*.egg-info/` from local editable installs
- `.coverage`, `htmlcov/`
- `*.log`, `*.tmp`, `*.bak`, `*.orig`, `coverage.xml`, and `.tox/`
- editor swap files and OS metadata.

## Submit-Gate Implication

The submit gate intentionally requires a clean git tree. To reach that state:

1. Commit fixed baseline and accepted iterative policy changes.
2. Keep local private config ignored.
3. Keep runtime evidence ignored and available for audit.
4. Remove only disposable temporary generated files.
5. Re-run `python scripts/workspace_report.py` and `git status --short`.
