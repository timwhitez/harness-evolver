"""Harbor runner with exact, cross-record task identity matching.

The complete Issue #7 implementation is retained in
:mod:`bench._harbor_issue7_identity_base`.  This final facade closes the last
ambiguity boundary: a simple requested task name must not select the first of
multiple result records whose structured paths have the same basename but
identify different tasks.
"""

from __future__ import annotations

from typing import Any

from bench import _harbor_issue7_identity_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)


def _canonical_result_identity(
    result: dict[str, Any],
) -> tuple[str, str] | None:
    """Return one unambiguous typed identity for a raw Harbor result."""

    identity = _base._task_identity(result)
    if identity.paths:
        if len(identity.paths) != 1:
            return None
        return ("path", next(iter(identity.paths)))
    if len(identity.names) != 1:
        return None
    return ("name", next(iter(identity.names)))


def _matching_trial_results(
    trial_results: list[dict[str, Any]],
    task_id: str,
) -> list[dict[str, Any]]:
    """Return all attempts only when their shared task identity is unique.

    Multiple records are expected when Harbor runs more than one attempt. They
    are accepted only when every matching record carries the same canonical
    identity. Structured path evidence is authoritative and may not be mixed
    with name-only evidence. Thus two datasets containing ``shared-task`` can
    never become order-dependent merely because the caller requested the
    basename.
    """

    candidates: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    for result in trial_results:
        if not isinstance(result, dict):
            continue
        identity = _base._task_identity(result)
        if not _base._identity_matches_requested(identity, task_id):
            continue
        canonical = _canonical_result_identity(result)
        if canonical is None:
            continue
        candidates.append(result)
        identities.append(canonical)

    if not candidates:
        return []

    kinds = {kind for kind, _ in identities}
    values = {value for _, value in identities}
    if len(kinds) != 1 or len(values) != 1:
        return []
    return candidates


class HarborRunner(_base.HarborRunner):
    """Require one canonical task identity across every selected attempt."""

    def _matching_trial_results(
        self,
        trial_results: list[dict[str, Any]],
        task_id: str,
    ) -> list[dict[str, Any]]:
        return _matching_trial_results(trial_results, task_id)

    def _select_trial_result(
        self,
        trial_results: list[dict[str, Any]],
        task_id: str,
    ) -> dict[str, Any] | None:
        matches = self._matching_trial_results(trial_results, task_id)
        return matches[0] if matches else None
