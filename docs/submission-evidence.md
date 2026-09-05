# Submission evidence contract

`SubmitGate` defaults to disabled. With upload and integrity checks enabled,
`check()` validates the **exact job selected for upload**, not just caller-supplied
score/count summaries. This is a TerminalBench single-step normalized-reward
submission policy; it is not a general replacement for all Harbor job types.

## Supported completed-job layouts

Legacy jobs can put records in `result.json -> trial_results`. Harbor 0.22 jobs
can store records in `<trial_name>/result.json`, with `n_total_trials`, an explicit
completed-trial count in `stats`, and a valid `finished_at` in the top-level file.
An empty or absent inline list permits that native layout; a malformed non-list
or malformed inline record does not permit fallback. When both sources exist,
their trial identities, outcomes, and task/model scope must agree exactly.

Every record needs a safe trial name, a task name, and either finite verifier
rewards in `[0, 1]` or a structured exception (type and message). Provided timing,
identity and configuration fields must have valid shapes. Duplicate attempts,
incomplete job counts, mixed model/dataset scopes, contradictory same-name task
identities, and multi-step jobs fail explicitly. Legacy inline records may omit
newer timing/identity fields; the gate cannot establish provenance those records
never captured. It does not cryptographically authenticate verifier output.

## Score and coverage

Policy `terminalbench-task-mean-v1` computes each task's mean over **all recorded
attempts**, then averages those task means. Recorded errors contribute zero;
ordinary verifier failures also remain in the denominator. A stale positive
reward attached to an error does not turn the error into a pass. A `reward` key
is primary; otherwise the mean of normalized named rewards is used. This gate
does not silently apply research-time infrastructure exclusions or pass@k.

The supplied score must match within the project's four-decimal reporting
precision. The raw evidence score must independently meet the trigger, so
rounding cannot raise it across the threshold. The task count and any supplied
attempt map must exactly match. The minimum attempts requirement is checked
against actual records even when the caller omits its map. Boolean/string/fractional
counts and non-finite/out-of-range scores are rejected rather than coerced,
including at the submit CLI boundary.

## ATIF and artifact checks

For passing attempts, `require_atif_trajectory` requires a complete document
accepted by the installed official `harbor.models.trajectories.Trajectory`
model. `schema_version` must be explicit. Sequential steps and tool-observation
references are checked by that model. Missing parser dependencies fail closed.

`agent/trajectory.json` is preferred over a sibling native event log named
`trajectory.jsonl`. A malformed preferred document cannot be rescued by another
candidate. A legacy `.jsonl` candidate must contain one complete ATIF object;
a raw `tool_call` event or a byte of arbitrary text is not ATIF. External
continuations/subagent files must first be consolidated; the gate never fetches
remote references. Embedded references must resolve within the supplied document.
Valid failing attempts do not require a successful-agent trajectory.

Existing agent-artifact integrity rules remain enabled. Text inspection does not
silently skip oversized, unreadable or invalid-UTF-8 artifacts. JSON duplicate
keys and non-finite values fail explicitly. Reads are descriptor-anchored through
the existing Linux O_PATH/procfs checked opener. Symlinks, special-file entries,
and hard-linked evidence files cannot supply evidence. The whole job inventory
is bounded to 50,000 entries and depth 64; at most 10,000 trials, 64 MiB per
inspected file and 256 MiB total read bytes are permitted. Exceeding these limits
requires an explicit policy decision, never a false successful partial scan.

## Submission lifecycle and concurrency boundary

A successful check records its recomputed summary, policy identifier, and a
fingerprint of the inventory and inspected file hashes. `submit_once()` retains
the exclusive durable intent introduced for duplicate prevention. After acquiring
that claim it reinspects the job before invoking upload. Changes abort upload
and keep the intent for operator reconciliation; no automatic retry occurs.

**The job directory must remain immutable through validation and upload.**
No-follow descriptor reads, before/after inventories and a second inspection are
not an atomic filesystem snapshot against an uncooperative concurrent writer.
Callers must freeze job production and restrict mutation before submission. This
change does not claim to solve malicious same-UID mutation during the uploader's
own later reads.

Dry-run checks do not claim or upload. `harbor_upload=False` keeps its explicit
local skip semantics. `require_integrity_scan=False` remains an explicit bypass
for controlled/non-production workflows; it does not certify evidence and the
intent records that it was disabled. Malformed boolean settings cannot disable
checks accidentally. Submission evidence is included in CLI JSON and intent/result
records; tests must use fixture executables or mocked uploaders, never a real
leaderboard upload.

## Focused regression commands

```bash
python -m pytest -q tests/test_submit_evidence.py tests/test_submit_cli_validation.py
python -m pytest -q tests/test_submit_durable_claim.py
python -m pytest -q tests/test_goal_submit_compression.py -k submit
```

Run the full repository suite in a complete development checkout before a
production rollout. Focused fixture tests are not evidence of benchmark quality,
real Harbor upload acceptance, or cross-platform integration.
