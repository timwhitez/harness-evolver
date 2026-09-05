# File publication and recovery contract

## Supported boundary

The secure filesystem tools require Linux `O_PATH`, `O_NOFOLLOW`, a trusted
`/proc/self/fd`, local-filesystem `flock`, and `renameat2` support. Missing
primitives fail closed. They have not been certified on network/distributed
filesystems. Normal text reads first pin an inode with `O_PATH`, validate type
and link count without a readable device/FIFO open, and then reopen that owned
kernel descriptor. Symlinks, special devices, and multiply-linked content
sources are rejected. Pure overwrite may de-alias a regular hard link without
reading or modifying the sibling inode.

Publication provides serialization **for cooperating writers** that all acquire
an exclusive lock on the same parent directory. Parent directories, transaction
entries, and procfs must not be under arbitrary noncooperating mutation.
An advisory lock is not a sandbox or protection against an uncooperative
same-UID process. Filesystem isolation or a trusted external broker must enforce
that stronger boundary in an adversarial environment.

`RENAME_EXCHANGE` is not inode compare-and-swap. A reader can observe complete
candidate bytes at the exchange before later validation finishes. The tools do
not claim linearizable conditional writes against arbitrary concurrent writers,
nor crash-atomic rollback. This corrects the stronger claims in historical
exchange-and-rollback PRs. A detected external post-exchange mismatch is reported
as **indeterminate**, not an untouched-target rejection, and never initiates an
exchange-back that could erase a newer update.

## Outcomes

Public local and Harbor write/edit operations carry the same versioned outcome:

| `publication_state` | `atomic_replace` | Meaning |
|---|---|---|
| `not_published` | `false` | This invocation did not publish a new entry. |
| `published` | `true` | Publication and required identity validation completed; cleanup/durability may still have warnings. |
| `indeterminate` | `null` | Publication may be visible, or the transport omitted valid outcome evidence. Reconcile before another mutation. |

`cleanup_warning`, `durability_warning`, `directory_fsync`, `no_auto_retry`,
`recovery_directory`, and relative `recovery_entries` distinguish publication
from later housekeeping. A failed old-inode unlink is **successful publication
with a cleanup warning**, not a failed write that should be retried. The legacy
`atomic_write_text_nofollow()` returns durability as a bool after confirmed
publication; on failure/uncertainty its `SafePathError.publication_outcome` carries
the facts. Public tools use the structured API directly.

An initially absent target uses no-replace publication. Existing targets are
validated under the cooperating-writer lock and exchanged. Type/identity/content
changes detected after that exchange preserve both the displaced entry and a
transaction intent. No rollback is attempted. Intent and payload synchronization
occur before publication; post-publication parent synchronization is explicit.
This is not a power-failure proof for every filesystem or an exactly-once client
acknowledgment protocol. A caller losing its response must reconcile actual
state even when normal cleanup already removed the intent.

## Recovery

A per-target `.hl-publish-<target-digest>.intent` is created exclusively and
persisted before publication. It records the target, random candidate/displaced
entry, observed identity, and candidate content digest. A crash, unresolved
exchange, or cleanup failure can leave this intent. It is a durable recovery
record, **not an expiring lock**. Subsequent tool publications to that target are
blocked until an operator reconciles it. Do not delete it because its PID is
absent or because it is old. Do not blindly replay append or edit operations.

To reconcile, first stop or isolate all writers of the directory. Preserve a
copy of the intent, current target, and all listed recovery entries without
following symlinks or read-opening special devices. Compare identities, content
digests, and the intended change. Decide whether the operation was applied,
requires a deliberate new edit, or needs manual restoration. Only then remove
owned transaction entries and sync the directory. The library deliberately has
no automatic "clear intent and retry" command.

The caller must not mutate or delete recovery entries outside this controlled
procedure. A process with arbitrary directory-write rights can defeat any
advisory protocol; this implementation does not claim otherwise.

## Regression evidence

The focused local tests exercise the exact shared publication function both as
a host import and as embedded source, generated Harbor write/edit/snapshot
scripts through real subprocesses, special-file descriptor acquisition, two
external replacements, cooperative contention, process exit immediately after
exchange, pre/post-publication fsync and unlink faults, and truthful public-tool
metadata. They use temporary fixtures, not live device writes or benchmark
submission. Environment SDK and unchanged heavyweight policy imports can be
replaced by explicit local test boundaries during partial-checkout testing;
this is not evidence of a complete repository or live Harbor test run.
