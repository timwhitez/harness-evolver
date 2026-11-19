# Frozen Experiment Plan: Paired Updater Attribution Evaluation (#167)

Status: **PLAN_FROZEN / REAL_EXPERIMENT_NOT_RUN** (live Worker API keys such as `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `WORKER_API_KEY` are not configured in the current environment; offline tests do not substitute for real model/Harbor runs).

## 1. Objective and Single Method Variable

Evaluate whether adding the on-demand competing-attribution and discriminating-evidence guidance from Issue #166 improves Codex updater failure attribution and subsequent held-out task outcomes, without changing Worker task prompts or building a new evaluation harness.

- **Control arm (`baseline`)**: Existing `WorkPacketBuilder` + `CodexUpdateEngine.run_update` without the #166 competing-attribution method text.
- **Treatment arm (`competing_attribution`)**: Identical `WorkPacketBuilder` + `CodexUpdateEngine.run_update` with the #166 on-demand competing-attribution guidance in `memory_record.detailed`, `update_memory_contract`, `cross_round_update_contract`, and `update_search_policy`.
- **Base projection**: Both arms share the same #163, #165, and #168 fixes (strict incomplete-snapshot detection, explicit `baseline_unavailable_*` provenance, and scoped `change_evaluation` applicability).

## 2. Frozen Inputs, Pairing, and Worktree Isolation

- **Pre-registered packet selection**: Select development failure packets spanning independent ground-truth cause classes:
  1. Misleading prior / stale hypothesis
  2. Infrastructure or environment failure
  3. Worker/harness policy defect
  4. Healthy neighbor requiring `noop` / no patch
- **Hidden ground truth**: Do not inject ground-truth cause labels into packets seen by either arm.
- **Per-pair isolation**: Every `(packet, arm, replicate)` slot starts from the identical frozen Git base commit, identical failure artifacts, identical frontier snapshot, and identical allowed memory snapshot, executed in an isolated writable `git worktree` and isolated `trials/` output root. Cross-arm contamination is prohibited.
- **Frozen configurations**: Freeze the effective Codex updater model/reasoning/sandbox config, Worker role (`--worker-role`), Harbor/Docker settings, and tool/environment interface across both arms. Record actual packet prompt/contract hashes and effective model configs as delivery proof before scoring.

## 3. Primary Analysis Unit and Effective Code Evaluation

- **Primary denominator**: All pre-registered `packet x arm x replicate` units (never filtered to accepted-patch subsets).
- **Attribution evaluation (stage 1)**: Score primary attribution accuracy, justified `noop`/`unknown`, false-positive edits on healthy neighbors, and selected component layer before outcome evaluation.
- **Task & held-out evaluation (stage 2)**:
  - Accepted patch (`review.accepted` + host validation pass): evaluate the patched worktree via `scripts/run_trial.py` and `scripts/regression_check.py --holdout-mode held_out` (`hl/regression_split.py`).
  - `noop` / rejected / rolled-back: evaluate the verified uncontaminated or cleanly rolled-back frozen baseline (reusing a baseline receipt only when pre-registered and identical across code, input, environment, and scoring identity). Unaccepted patches are never executed.
  - Unconfirmed code state or infrastructure failure: keep the slot as `unknown` (never coerce to `0` or drop from the denominator), report paired availability rate, and label any conditional subset comparison explicitly.

## 4. Resource Accounting and Stop Policy

- Resource policies and external stopping conditions are pre-registered outside `CodexUpdateEngine` and `HLLoop` without introducing internal round, turn, or token stop conditions into any no-limit loop.
- Account full wall-clock time, retries, waits, failed runs, updater tokens/calls, Worker tokens/calls, and validation/Harbor costs per slot.
- Record all artifacts (`codex_update_packet.json`, `codex_events.jsonl`, `final_message.json`, `git.diff`, `review.json`, `change_manifest.json`, `update_summary.md`, and Harbor trial/verifier outputs) with explicit cross-links. Do not auto-promote default policy or alter rollback/leaderboard submit gates.
