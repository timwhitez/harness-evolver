# HarnessEvolver Architecture

Complete architecture documentation for the Heuristic Learning framework.

## Overview

HarnessEvolver applies [Heuristic Learning (HL)](https://trinkle23897.github.io/learning-beyond-gradients/) to the harness engineering problem for TerminalBench 2.0 coding agents.

The core innovation: **do not train neural networks. Run this repository's own
TerminalBench Worker loop, then let Codex operate outside the benchmark as the
HL updater that edits the Worker/harness from verified failure evidence.**

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    HL Optimization Loop                       │
│                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  RUN     │──▶│ COLLECT  │──▶│ ANALYZE  │──▶│  EDIT    │ │
│  │  tasks   │   │ feedback │   │ failures │   │ harness  │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────┬────┘ │
│       ▲                                              │      │
│       │           ┌──────────┐                      │      │
│       └───────────│ VERIFY   │◀─────────────────────┘      │
│                   │ regression│                              │
│                   └──────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. HL Core (`hl/`)

The engine that drives the optimization loop.

| File | Purpose |
|------|---------|
| `protocol.py` | Abstract base classes: Policy, StateProvider, FeedbackChannel, MemoryStore, UpdateEngine |
| `types.py` | Pydantic data models: TrialResult, TrialSummary, HarnessPatch, etc. |
| `system.py` | HeuristicSystem: central registry wiring all components together |
| `loop.py` | HLLoop: the outer optimization loop (run→collect→analyze→edit→verify→repeat) |
| `memory.py` | FileSystemMemory: JSON-based storage for trials, summaries, regressions, patches |
| `coupling.py` | CouplingTracker: measures edit impact, triggers compression |
| `goals.py` | Persistent campaign goal and budget accounting |
| `compression.py` | Absorb/compress trigger planning while preserving raw evidence |
| `submit.py` | One-shot submit gate with explicit opt-in and duplicate prevention |

### 2. Harness (`harness/`)

The components being optimized — each is an independently versioned, editable "Policy."

| Sub-package | Purpose | Key SOTA Pattern |
|------------|---------|-----------------|
| `prompts/` | System, task, recovery prompt templates | Three-tier separation (Factory Droid) |
| `tools/` | Tool definitions and execution | 6 core tools, todo/goal/verify, correction hints |
| `planning/` | Planning strategies | Todo enforcement + progressive thinking (ForgeCode) |
| `context/` | Context management | 5-layer compaction + isolation (Claude Code) |
| `entrypoint/` | Task entry-point discovery | Semantic scanning + env awareness (ForgeCode + Droid) |
| `recovery/` | Error recovery patterns | Known-failure → recovery mapping (learned over time) |
| `verification/` | Post-task verification | Pre-submission checks + self-testing |

### 3. TerminalBench Integration (`bench/`)

| File | Purpose |
|------|---------|
| `harbor.py` | HarborRunner: wraps `harbor run` CLI |
| `harbor_adapter.py` | Harbor custom agent entrypoint for the self-owned Worker |
| `agent.py` | Python HLAgent shell: bridges LiteLLM/Harbor tools into the Rust Worker core and adapts the Rust result to `TrialResult` |
| `tasks.py` | TaskCatalog: local TerminalBench task loader and status tracking |
| `scoring.py` | Scoring: verifier output parsing + component impact |
| `trajectory.py` | TrajectoryReader: ATIF trajectory analysis |

### 3a. Rust Worker Core (`crates/hl-worker-core/`)

| File | Purpose |
|------|---------|
| `src/main.rs` | JSONL-driven Worker core for prompt/task context initialization, bounded entrypoint scan, same-task memory hints, turns, tool-call dispatch, todo/completion gates, recovery hints, context compaction, and trajectory ordering |

### 4. Meta-Agent (`meta/`)

| File | Purpose |
|------|---------|
| `codex_update.py` | Codex-backed UpdateEngine using `codex exec --json` |
| `missions.py` | Mission-style meta debug packets for external-loop feature selection and validation contracts |
| `packager.py` | Failure packet builder with allowed paths, verifier output, regression contracts |
| `reviewer.py` | Deterministic diff review, forbidden-path checks, rollback helpers |
| `agent.py` | Legacy scaffold MetaAgent retained for deterministic analyzer tests |
| `prompts.py` | Specialized Orchestrator prompts (analysis, edit, compression) |
| `analysis.py` | FailureAnalyzer: pattern→component mapping |
| `suggestion.py` | ImprovementSuggester: structured patch generation |
| `editor.py` | HarnessEditor: safe edit with backup, validation, rollback |

### Mission Debug Layer

Factory's Missions architecture is used here only as a meta-pattern: long work
is split into milestones/features, validation is explicit, and the orchestrator
can create follow-up fix features after validators report gaps. In this repo,
that pattern becomes `meta/missions.py` and `scripts/mission_debug.py`.
Campaign runs can also emit it with `scripts/run_campaign.py --mission-debug`
after the campaign summary is written.

The mission debug layer is intentionally deterministic. It reads existing
campaign summaries or failure trials and emits:

- validation contracts that must pass before an edit is accepted;
- bounded feature candidates for the external HL loop to choose from;
- controls such as `adjust_worker_role`, `adjust_run_cap`, and
  `request_more_evidence`;
- blocked actions that preserve benchmark integrity and the self-owned Worker
  boundary.

It does not spawn external workers, edit benchmark tasks, or mark a campaign
complete. Codex can receive the packet as context through `meta/packager.py`,
but Codex still edits this repository's Worker loop/harness only after the
normal diff review and regression gates.

## Data Flow

```
TerminalBench Task
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Harbor       │────▶│ HL Worker     │────▶│ Harbor      │
│ custom agent │     │ TAOR loop     │     │ verifier    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌──────────────┐              │
                    │ TrialResult   │◀─────────────┘
                    │ + artifacts   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐     ┌─────────────┐
                    │ Codex packet  │────▶│ codex exec   │
                    │ + contracts   │     │ bounded edit │
                    └──────────────┘     └──────┬──────┘
                                                 │
                    ┌──────────────┐              │
                    │ PatchReviewer │◀─────────────┘
                    │ + regression  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Regression    │
                    │ Check        │
                    └──────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                  PASS          FAIL → ROLLBACK
```

## Design Decisions

1. **Rust Worker core with Python boundary adapters**: Harbor and LiteLLM remain Python-facing, but the dynamically updated Worker loop and harness agent policy live in Rust. Python code only bridges model calls, Harbor environment tools, and the existing `TrialResult` contract.
2. **File-system memory**: Explicit JSON, not vector DB. Inspectable, diffable, git-trackable.
3. **Verifier-grounded**: Worker readiness is not a pass; Harbor/verifier reward is the pass signal.
4. **Prompt-driven but gated**: Codex receives structured packets, but deterministic code reviews diff scope and validation.
5. **Serial-first**: One HL improvement slice at a time, with bounded Harbor task concurrency inside a round when configured. Rounds, Codex updates, and regression checks remain serial.
6. **Every edit recorded**: Codex events, final message, diff, review, and regression status land in trials/diffs/.
7. **Compression required**: CouplingTracker and CompressionEngine flag clutter without deleting raw evidence.
8. **Reference learning is translated, not copied**: External harness projects can inform Codex update strategy, but adopted ideas must become this repo's own Worker/harness interfaces, tests, and validation contracts.

## Prediction, Frontier, And Validation Loop

Codex-backed updates are no longer only reviewed as source diffs. Each accepted
update leaves a falsifiable manifest and is compared against later evidence:

- `meta/packager.py` requires final-report `prediction` fields:
  `expected_fixed_task_classes`, `risk_task_classes`, `expected_metric_delta`,
  `confidence`, and `falsification_window`.
- `meta/codex_update.py` writes `change_manifest.json` beside `git.diff`. The
  manifest captures failure evidence, root cause/generalization, targeted fix,
  changed-file component layers, predicted impact, and the host validation
  ladder used to accept the patch.
- `scripts/run_campaign.py` maintains a same-model per-task frontier under
  `trials/summaries/<campaign>_frontier_<model-scope>.json`. The frontier
  tracks best score, best trial, best packet, last score, attempts, pass/fail
  volatility, and packet-linked regressions.
- The next round after an accepted update writes `change_evaluation.json`
  beside the manifest. It records task flips, prediction hits, prediction
  misses, and whether rollback was recommended or applied.
- Host validation is dynamic. Packet-required commands still run, and the
  updater derives additional checks from changed files: Python compile checks,
  changed-test pytest, campaign dry-runs for meta/script edits, Harbor dry-run
  smoke for Worker/runtime edits, and regression dry-runs when the regression
  gate itself changes. Campaign-scoped Codex updates additionally inject the
  active same-model solved-task regression command, preserving the current
  memory path, model/role, lane, and jobs directory before a patch is accepted.

Worker context compaction uses the serialized UTF-8 JSON size of messages and
tool schemas as a byte estimate, including tool arguments and retained reasoning;
it does not claim an exact token count. The existing compaction threshold remains
configurable. Normal compaction keeps recent complete interaction units. A
new complete tool round keeps every existing bounded head/tail projection until
a provider request accepts it, even when it exceeds the recent-result pruning
window. Transient retries retain this protection; delivered older rounds still
use the original window. The advisory threshold preserves the original short
and recent-history guards and partial-progress behavior; it never drops the
undelivered round, even when the request remains above target. Only a structured
provider context-overflow rejection permits omission: recovery removes older
units first, then may omit the entire pending call/result group with a
`worker_tool_delivery` model notice and a `tool_result_delivery_incomplete`
trajectory event (call IDs, `context_overflow` reason, and byte estimates).
Irreducible system/task/work-state input uses `context_input_unfit`.
This does not raise the context target or replay tools. A
structured provider context-overflow response forces reduction even in short
histories: system/task instructions survive, assistant calls and their parallel
tool results are removed together, and meaningful todo/verification state is
retained. Omitted history is archived in trajectory events, without claiming a
semantic summary. Recovery retries only after a strictly smaller estimated
request; irreducible input returns `context_input_unfit` for external model/input
reconfiguration. This diagnoses a rejected input, not a turn/time/token budget.
Transient provider errors retain their existing recovery path.

Trial reports also preserve efficiency evidence parsed from Harbor artifacts:
token usage, cache tokens, cost when available, turns, API calls, provider
latency, API error counts, and cache-hit ratio. Campaign reports aggregate
these metrics by domain, difficulty, task type, and whole campaign so future
updates can optimize reliability and cost, not only pass/fail.

Campaign final and checkpoint reports retain invocation provenance in
`reproducibility`: `trials_config` is the supplied trials path, while
`models_config_path`, `worker_role`, and `worker_role_source` come directly from
the configuration resolver. `config_scope=report_invocation` applies even when
the report aggregates stored trials; these fields do not identify historical
trial inputs. CLI-only model requests retain `models_config_path=null`.
`models_config_priority` remains a discovery search rule, labelled by
`models_config_priority_scope=discovery_search_order`, rather than a selected
source. Callers without source metadata emit `null` instead of inferring files.

Git fields describe `git_scope=report_generation_checkout`: queries run with
explicit `cwd=git_query_cwd`, the absolute invocation directory, which may be a
subdirectory of the checkout. Commit and status describe the enclosing checkout;
`git_observed_at` records collection start in UTC. `git_dirty` is `false` for a
successful clean query, `true` for dirty, and `null` when unavailable. Each
failed query leaves a bounded reason in `git_errors`, using only an exit code
or exception class; commit and status failures remain independent. No raw
configuration or environment dumps, Git diff, status paths, or error text is
copied into these fields. Existing endpoint redaction remains in effect.
Readers of older reports must treat absent fields as unknown, not clean or a
default configuration. Report generation preserves trial/component snapshots
and patch lineage; these observations are not an execution fingerprint or an
atomic Git snapshot and do not change score, Mission, roadmap, or submit gates.

Worker metrics schema `worker_metrics_v1` preserves optional `TrialResult.turn_count`
and `metadata.turn_count`: the number of Rust model-request attempts, including
failed requests and Worker recovery retries. Bootstrap is turn 0 and does not
increment this counter. This is separate from Harbor's generic `n_turns` and
provider API-call counts: LiteLLM's internal HTTP retries are not measured here.
Missing historical counters remain `null`; explicit zero remains known zero.
Harbor parsing and attempt aggregation retain the Worker counter; an aggregate
with a missing attempt stays unknown. Existing `n_turns` campaign metrics keep
their previous meaning.

`TrialResult.tool_calls` contains logical tool outcomes, including failures,
local argument/policy rejections, internal completion verification, and one
bootstrap scan request outcome when bash is available. It is not a physical
dispatch count. The scan has `phase=bootstrap` and is reported separately from
the model-loop policy history, preserving checkpoint and completion behavior.
It still has one `entrypoint_scan` trajectory event. Valid final results replace
the bridge's live mirror, avoiding duplicate outcomes. Interrupted runs preserve
only observed request attempts and completed outcomes with
`metadata.worker_metrics_observation.status=partial`; an in-flight tool can be
absent. The Rust final payload explicitly declares `worker_metrics_schema=worker_metrics_v1`;
older payloads without that declaration keep legacy tool coverage even when
their turn counter is known. Harbor retains the counter/observation and the
existing trajectory format; it does not reconstruct a missing tool history. Token
usage remains independently optional, and none of these counters changes
score, verified status, or Harbor/verifier evidence requirements.

The bootstrap producer requests bounded `pwd`, `ls`, and `find . -maxdepth 2`
discovery in the current task workspace. It does not request discovery of hidden
verifier paths. Default and Harbor shell tools authorize the entire command
before execution. A denied bootstrap request remains one failed logical outcome
with its policy error; its presence in history does not prove execution.

Harbor coroutine cancellation saves the bridge's observed counter and completed
outcomes immediately in context metadata and a `worker_metrics_snapshot` trajectory
event, marked `partial`. It does not await an in-flight Python model/tool call.
A late result can refine this observation only for the same adapter run UUID and
exact task ID, still as partial; stale callbacks and trajectory sinks cannot
overwrite a subsequent run. Missing or malformed top-level job JSON with one
exact surviving trial enriches only Worker metrics while keeping the inherited
score, status, and verifier decision. Cross-record task identity rules still apply.
Cancellation intent is armed before the executor starts and checked after command
resolution, after process registration, and before processing further bridge events;
an early cancellation cannot resume model/tool dispatch when setup later finishes.

Worker usage schema `worker_usage_v1_exclusive_input` defines `input` as
non-cache-hit input, `cache` as cache-hit input, and `output` as generated
tokens. The Python/LiteLLM boundary subtracts the known cached subset from
inclusive `prompt_tokens`; matching aliases count once. Raw Anthropic-shaped
`input_tokens` plus top-level `cache_read_input_tokens` (without LiteLLM prompt
fields) retain exclusive input semantics and include observed cache-creation
tokens in non-cache-hit input. Missing, invalid, or conflicting
counters stay unknown, and a missing call prevents an exact aggregate counter.
Harbor metadata retains the schema and diagnostics; attempt/campaign reports
label coverage and observed subtotals. Cache ratios require known input and
cache. Campaign means use v1 observations and report their sample count. Legacy
and v1 input counters cannot be combined under one denominator. Existing files
without the schema remain readable and labelled `legacy`; they are not silently
recalculated without original usage evidence. The inclusive LiteLLM contract is
documented in its [prompt caching guide](https://docs.litellm.ai/docs/completion/prompt_caching).

The campaign runner also writes a durable analysis layer:

- `trials/analysis/<campaign>/<summary>/overview.md` summarizes failure buckets,
  top affected components, timeout phases, candidate update classes, and raw
  traceability links.
- `trials/analysis/<campaign>/<summary>/detail/<task>.md` stores task-level
  status, score, verifier/error tails, and artifact pointers.
- `meta/packager.py` includes recent overview tails in `campaign_context` so
  Codex can start from compressed evidence instead of re-deriving every claim
  from raw logs.

Two low-cost runner policies add signal before another edit:

- Partial-pass diagnostics rerun only tasks that already have both pass and fail
  history, using `codex_update.partial_pass_diagnostic_k` as audit/reporting
  metadata rather than a diagnostic-attempt count or loop stop condition. This
  helps distinguish stochastic pass@k tasks from consistent pass@1 policy
  failures without changing official leaderboard attempts.
- Task-selection controls (`--random-count`), legacy cap metadata
  (`--max-tasks`, `tasks.max_tasks_per_trial`, `--run-task-cap`), mission debug
  feature counts, and rate-limit concurrency restore waits are scheduling,
  packet-size, audit, or throughput controls only. Random-count and legacy cap
  metadata do not truncate the campaign task pool or per-round task list. These
  fields do not stop or truncate the master loop, Codex update sub-agent,
  diagnostic/context sub-agents, validation/regression sub-agent,
  mission-debug sub-agent, or Worker loop.
- The no-limit contract applies to every loop owner: master, Codex update
  sub-agent, diagnostic/context sub-agents, validation/regression sub-agent,
  mission-debug sub-agent, and Worker task loop. Legacy or compatibility fields
  such as `iterations`, `max_features`, `max_turns`, timeouts, cooldowns, caps,
  counts, and budgets remain audit, scheduling, packet-size, recovery, or
  single-operation metadata only; they must not terminate or skip any master or
  sub-agent loop.
- `failure_class_attempts` records failure class, component layer, packet id,
  mission-selected candidate id when available, acceptance, and next evaluation
  result. Future Codex packets discourage repeated unsuccessful class/layer or
  mission-candidate attempts, while supported attempts preserve the exact
  mission candidate id for evidence-backed extension.

## External Harness References

Codex work packets include a `harness_reference_contract` so repeated poor
updates can borrow from current harness-engineering practice without drifting
into benchmark delegation or task-specific overfitting.

| Source | Local reference | What to transfer |
|--------|-----------------|------------------|
| Agentic Harness Engineering | `/tmp/harness-evolver-refs/agentic-harness-engineering` | Component/experience/decision observability, evidence-root-cause-fix-impact manifests, and next-iteration falsification of predicted flips |
| Meta-Harness | `/tmp/harness-evolver-refs/meta-harness` | Same-model baseline/frontier comparison, smoke-to-hard-to-full bring-up ladders, and trial-level cost/token/turn/cache metrics |
| TACO | `/tmp/harness-evolver-refs/TACO` | Observation compression rules, uncovered-output detection, reusable rule pools, reproducible freeze/local-only modes, and structure-preserving trajectory shortening |
| OpenClacky | `/tmp/harness-evolver-refs/openclacky` | Cache-stable prompts/tool schemas, small extensibility surfaces, insert-then-compress, layered context-overflow recovery, and direct repo search/read over stale indexes |
| Claude Code large-codebase practices | Official blog | Lightweight `AGENTS.md`/repo context, progressive-disclosure skills, deterministic hooks/checks, specialized sub-contexts, and periodic guidance pruning |
| Self-Harness article | `https://mp.weixin.qq.com/s/sgP8m1nnW7JhsDT7Ki7nVw` | Weakness mining, bounded harness proposal, proposal validation, and same-model self-improvement loop framing |
| Browser Use bitter lesson | `https://browser-use.com/posts/bitter-lesson-agent-frameworks` | Small loop, broad browser/action surface, explicit done-style completion, and context pruning for bulky transient observations |
| pi coding agent | `https://mariozechner.at/posts/2025-11-30-pi-coding-agent/` | Minimal tool surface, visible provider/context handoff, split tool result observability, and session artifacts that can be post-processed |

Reference use is gated by the same update contract as local fixes: it must tie
to campaign evidence, preserve the self-owned Worker boundary, avoid copying
reference agents, avoid changing benchmark definitions, keep same-model
comparisons, and leave concise plus detailed update memory.

Research focus areas are explicit in each Codex work packet: prefer broad
shell/file action space with restrictions learned from evidence, preserve the
Worker `done` plus verifier completion contract, keep bulky context pruning
observable, and expose provider/context/update handoff decisions in artifacts.

Web references also carry fetch constraints. In particular, `mp.weixin.qq.com`
articles must be fetched with GET plus a WeChat mobile client User-Agent
because generic crawler User-Agents and HEAD-only probes can receive an
environment verification page instead of the article body. Preflight must treat
that verification body as unavailable evidence, not a successful article fetch.

## Running Tests

```bash
# Install
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_hl_loop.py -v

# With coverage
pytest tests/ -v --cov=hl --cov=harness --cov=bench --cov=meta
```
