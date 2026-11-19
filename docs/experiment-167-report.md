# Experiment #167 Report: Paired Updater Attribution & Held-Out Evaluation

Status: **REAL_EXPERIMENT_COMPLETED** (Codex updater + Harbor Worker + held-out/held-in splits executed with `deepseek-flash` via CPA gateway).

## 1. Frozen Protocol & Delivery Proof

- **Single method variable**: #166 on-demand competing-attribution and discriminating-evidence guidance in `meta/packager.py` (`competing_attribution` arm at `78eb511` vs `baseline` arm at `a73afa3`, with identical #163/#165/#168 code and only the 4 #166 guidance blocks reverted in the isolated control branch).
- **Frozen updater & Worker model**: `deepseek-flash` (`provider=deepseek` for `codex exec`, `openai_compatible` for Harbor Worker, `reasoning_effort=high`, `max_output_tokens=8000`).
- **Pre-registered denominator**: 4 frozen packets x 2 arms x 1 replicate = **8 slots** (zero subset filtering).
- **Regression splits (`hl/regression_split.py`, `fraction=0.5`, `seed=0`)**:
  - `held_out` (`D_out`): `vulnerable-secret`, `git-leak-recovery`
  - `held_in` (`D_in`): `fix-git`

| Slot | Arm | `has_competing_guidance` | `skip_codex_update` | Selected Candidate |
|---|---|---|---|---|
| `pkt_01_misleading_prior__baseline__rep1` | `baseline` | `False` | `False` | `mission-attributed-verifier-mismatch` |
| `pkt_01_misleading_prior__competing_attribution__rep1` | `competing_attribution` | `True` | `False` | `mission-attributed-verifier-mismatch` |
| `pkt_02_infra_failure__baseline__rep1` | `baseline` | `False` | `False` | `mission-attributed-task-misunderstanding` |
| `pkt_02_infra_failure__competing_attribution__rep1` | `competing_attribution` | `True` | `False` | `mission-attributed-task-misunderstanding` |
| `pkt_03_policy_defect__baseline__rep1` | `baseline` | `False` | `False` | `mission-attributed-timeout` |
| `pkt_03_policy_defect__competing_attribution__rep1` | `competing_attribution` | `True` | `False` | `mission-attributed-timeout` |
| `pkt_04_healthy_neighbor__baseline__rep1` | `baseline` | `False` | `False` | `mission-regression-contract-hardening` |
| `pkt_04_healthy_neighbor__competing_attribution__rep1` | `competing_attribution` | `True` | `False` | `mission-regression-contract-hardening` |

## 2. Arm Summary (Full Pre-Registered Denominator: N=4 per Arm)

| Metric | Baseline (`a73afa3`) | Competing Attribution (`78eb511`) |
|---|---|---|
| Pre-registered slots | 4 | 4 |
| Stage-1 primary attribution correct | 4/4 | 4/4 |
| Competing-cause & discriminating-evidence documented (conflicting packets) | 1/2 (`pkt_02` only) | 2/2 (`pkt_01`, `pkt_02`) |
| Healthy-neighbor false-positive edits | 0/1 | 0/1 |
| Justified `noop` / `rejected` (clean/rolled-back baseline) | 3/4 | 3/4 |
| Accepted patches | 1/4 | 1/4 |
| Unknown / unconfirmed code state | 0/4 | 0/4 |
| Target task verifier pass (effective code, known) | 3/4 | 3/4 |
| `held_out` regression pass (`vulnerable-secret`, `git-leak-recovery`) | 4/4 | 4/4 |
| `held_in` regression pass (`fix-git`) | 4/4 | 4/4 |
| Codex updater total wall time (s) | 1458.6 | 1870.9 |
| Live Stage-2 Harbor wall time (s) | 332.7 | 351.3 |
| Codex tokens (input / cached / output) | 15,143,882 / 14,817,024 / 94,395 | 20,017,058 / 19,668,352 / 109,165 |

## 3. Per-Slot Results (Attribution, Effective Code, Verifier, Holdout, Cost)

| Slot | Status / Accepted | Effective Code | Stage-1 Attribution | Target Task Result | Held-Out (`D_out`) | Held-In (`D_in`) | Codex Wall / Out Tokens |
|---|---|---|---|---|---|---|---|
| `pkt_01_misleading_prior__baseline__rep1` | `edited` / `True` | `accepted_patch` (`harness/tools/shell.py`) | `True` (no explicit competing/prior note) | `passed` (`fix-git__x4wCR7k`, score=1.0, live) | `passed` (`git-leak-recovery__LEeJvBD`, `vulnerable-secret__tgHDMkQ`, 1.0, live) | `passed` (`fix-git__n2iFnRS`, 1.0, live) | 524.9s / 30,637 |
| `pkt_01_misleading_prior__competing_attribution__rep1` | `edited` / `True` | `accepted_patch` (`harness/tools/verify.py`) | `True` (explicit competing causes + discriminating obs) | `passed` (`fix-git__vFirniu`, score=1.0, live) | `passed` (`git-leak-recovery__bhApWU6`, `vulnerable-secret__e6Fp688`, 1.0, live) | `passed` (`fix-git__LacLa4K`, 1.0, live) | 743.5s / 37,358 |
| `pkt_02_infra_failure__baseline__rep1` | `noop` / `False` | `clean_baseline` | `True` | `passed` (`fix-git__Be369p4`, 1.0, frozen baseline) | `passed` (`vulnerable-secret__wHqr6HM`, `git-leak-recovery__YXVo4eA`, 1.0, frozen baseline) | `passed` (`fix-git__Be369p4`, 1.0, frozen baseline) | 128.5s / 13,209 |
| `pkt_02_infra_failure__competing_attribution__rep1` | `noop` / `False` | `clean_baseline` | `True` (explicit 3 competing causes + falsification test) | `passed` (`fix-git__Be369p4`, 1.0, frozen baseline) | `passed` (`vulnerable-secret__wHqr6HM`, `git-leak-recovery__YXVo4eA`, 1.0, frozen baseline) | `passed` (`fix-git__Be369p4`, 1.0, frozen baseline) | 232.1s / 20,869 |
| `pkt_03_policy_defect__baseline__rep1` | `rejected` / `False` | `clean_baseline` (reverted cleanly) | `True` | `timeout_cancelled` (`sanitize-git-repo__2fVcYtd`, 0.0, frozen baseline) | `passed` (frozen baseline) | `passed` (frozen baseline) | 757.3s / 44,706 |
| `pkt_03_policy_defect__competing_attribution__rep1` | `rejected` / `False` | `clean_baseline` (reverted cleanly) | `True` | `timeout_cancelled` (`sanitize-git-repo__2fVcYtd`, 0.0, frozen baseline) | `passed` (frozen baseline) | `passed` (frozen baseline) | 817.1s / 41,577 |
| `pkt_04_healthy_neighbor__baseline__rep1` | `noop` / `False` | `clean_baseline` | `True` (0 false edit) | `passed` (`fix-git__Be369p4`, 1.0, frozen baseline) | `passed` (frozen baseline) | `passed` (frozen baseline) | 47.9s / 5,843 |
| `pkt_04_healthy_neighbor__competing_attribution__rep1` | `noop` / `False` | `clean_baseline` | `True` (0 false edit) | `passed` (frozen baseline) | `passed` (frozen baseline) | `passed` (frozen baseline) | 78.1s / 9,361 |

## 4. Live Stage-2 Harbor Token & Wall-Clock Breakdown (`accepted_patch` slots)

- **`pkt_01_misleading_prior__baseline__rep1`**:
  - `fix-git__x4wCR7k` (target): `72.8s`, tokens `9,714 in / 110,208 cache / 4,321 out` -> `reward=1.0`
  - `fix-git__n2iFnRS` (`held_in`): `59.2s`, tokens `10,297 in / 88,832 cache / 3,247 out` -> `reward=1.0`
  - `git-leak-recovery__LEeJvBD` (`held_out`): `124.8s`, tokens `12,778 in / 136,064 cache / 9,750 out` -> `reward=1.0`
  - `vulnerable-secret__tgHDMkQ` (`held_out`): `75.8s`, tokens `20,469 in / 112,384 cache / 5,333 out` -> `reward=1.0`
- **`pkt_01_misleading_prior__competing_attribution__rep1`**:
  - `fix-git__vFirniu` (target): `125.1s`, tokens `15,274 in / 176,640 cache / 13,140 out` -> `reward=1.0`
  - `fix-git__LacLa4K` (`held_in`): `61.1s`, tokens `10,960 in / 93,696 cache / 2,790 out` -> `reward=1.0`
  - `git-leak-recovery__bhApWU6` (`held_out`): `97.6s`, tokens `11,015 in / 129,792 cache / 6,761 out` -> `reward=1.0`
  - `vulnerable-secret__e6Fp688` (`held_out`): `67.5s`, tokens `21,751 in / 121,984 cache / 3,541 out` -> `reward=1.0`

## 5. Findings, Caveats, and Non-Promotion

- **Main outcome (full denominator N=4 per arm)**: Both arms achieved `4/4` primary attribution accuracy, `0/1` healthy-neighbor false edits, `3/4` target-task verifier pass, and `4/4` held-out regression pass under the frozen `deepseek-flash` configuration.
- **Qualitative difference (`#166` treatment vs baseline)**: `competing_attribution` documented explicit competing explanations, commit/packet provenance, and discriminating observations in `memory_record.detailed` on `2/2` conflicting packets (`pkt_01`, `pkt_02`), versus `1/2` for `baseline`, and chose a verification-semantic gate in `harness/tools/verify.py` rather than a shell `git reflog` hint on `pkt_01` (conditional subset observation on `accepted_patch`, `1` slot per arm). This came at ~1.28x Codex wall time (`1870.9s` vs `1458.6s`) and ~1.16x Codex output tokens (`109,165` vs `94,395`).
- **Operational diagnostics surfaced during execution**:
  1. `pkt_03_policy_defect`: Both arms independently diagnosed the real Worker completion-gate loop in `crates/hl-worker-core/src/main.rs` (`sanitize-git-repo` blocked after `csv.DictReader` verification) and produced passing patches, which `report.mission_scope` cleanly rejected/reverted because `bench/resources/hl-worker-core/src/main.rs` parity requires updating a mirror outside `mission-attributed-timeout.allowed_edit_paths`.
  2. `scripts/regression_check.py` CLI gate exited `1` on `snapshot_validation_failed` due to a task-ID/port normalization mismatch between Harbor's bare task filter (`fix-git`) vs `TrialResult.task_id` (`terminal-bench/fix-git`) and `:24635` port inclusion vs portless `model_config.base_url_host`, while all 6 underlying Harbor verifier trials passed (`reward=1.0`, `verified=True`).
- **No automatic policy promotion**: Default policies, rollback logic, and leaderboard submit gates remain unchanged.
