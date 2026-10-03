# Contributing

Thank you for improving HarnessEvolver.

## Scope

Changes must preserve the project's central boundary: the benchmarked agent is
the self-owned Worker and harness in this repository. External coding agents may
help review or update that code outside a benchmark run, but they must never be
substituted for the Worker.

Do not commit TerminalBench task definitions, tests, reference solutions,
Harbor job contents, trial artifacts, credentials, local provider settings, or
downloaded reference material.

## Development workflow

1. Read README.md, docs/architecture.md, and AGENTS.md.
2. Start with a bounded problem statement and identify the source, test, and
   validation contract that cover it.
3. Keep behavioral changes small. Add or update deterministic tests for shared
   behavior, configuration schemas, runner/adapter code, or policy changes.
4. Run the narrowest relevant checks and then the full suite:

   ~~~bash
   pytest tests/test_models_and_worker_policy.py -q
   pytest tests/ -q
   git diff --check
   ~~~

5. For Harbor-facing changes, run an applicable dry-run command before any
   live task. A live pass is valid only when Harbor/verifier artifacts confirm
   it.
6. Commit one coherent, validated change. Keep unrelated cleanup and generated
   files out of the commit.

## Offline test contracts

Patch a re-exported entry point in its defining module (for example,
`sys.modules[main.__module__]`), and fail unexpected subprocess calls in network
fixtures. A timeout result can be simulated without launching a real preflight.

Worker task loops have no time, round, or attempt budget. Bounded process waits
are permitted for cancellation/error cleanup, and terminal stdout EOF handling
has its own bounded protocol deadline. Test those scopes separately; a descendant
holding stderr must not accidentally also hold stdout in a stderr-only fixture.

Publication tests should inject faults at actual syscall boundaries and provide
versioned publication receipts. After an exchange race, preserve indeterminate
recovery evidence rather than requiring destructive rollback. Non-object Harbor
job JSON is a structured error; singleton metric recovery cannot verify a pass.
Existing multi-attempt recovery still requires independent verifier evidence.

## Configuration and artifacts

Use .env.local and config/local.yaml for credentials, private endpoints,
mirrors, and machine-specific Codex settings. They are ignored by Git. Store
only variable names and redacted descriptions in tracked documentation.

trials/, jobs/, and local TerminalBench task checkouts are runtime evidence,
not source. Follow docs/workspace_hygiene.md when classifying a dirty worktree
entry. Do not delete runtime evidence simply to make git status clean.

## Benchmark integrity

Keep every change general. Do not add task-id-specific solution logic, hidden
verifier knowledge, or benchmark-specific answer material. Do not change
official task tests, verifiers, task definitions, resource limits, or timeouts
to improve an evaluation result.

## Pull requests

Describe the problem, the affected component, the tests you ran, and any
remaining validation limit. A review should be able to distinguish source
evidence, local test evidence, and real Harbor/verifier evidence.
