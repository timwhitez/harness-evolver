"""Check Worker task-loop boundaries separately from terminal cleanup."""

from __future__ import annotations

import ast


def assert_worker_waits_are_cleanup_only(source: str) -> None:
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def ancestors(node):
        while node in parents:
            node = parents[node]
            yield node

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        lineage = list(ancestors(node))
        owner = next(
            (
                parent
                for parent in lineage
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        timeout_position = {"wait": 0, "communicate": 1}.get(node.func.attr)
        positional_timeout = timeout_position is not None and len(node.args) > timeout_position
        keyword_timeout = any(k.arg == "timeout" for k in node.keywords)
        if timeout_position is not None and (positional_timeout or keyword_timeout):
            assert owner is not None and owner.name == "_terminate_process", (
                "a bounded process wait must be cancellation/error cleanup, not a task-loop limit"
            )
        if node.func.attr == "_terminate_process":
            cleanup = any(isinstance(parent, ast.ExceptHandler) for parent in lineage)
            cleanup |= any(
                isinstance(parent, ast.Try)
                and any(node in ast.walk(item) for item in parent.finalbody)
                for parent in lineage
            )
            assert owner is not None and (
                owner.name == "cancel_current_run" or (owner.name == "_run_rust_core" and cleanup)
            ), "process termination must not be a normal Worker loop stop condition"
