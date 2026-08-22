"""Task-local todo tools for Worker execution discipline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from harness.tools.base import ToolDef, ToolResult, ToolSchema

TodoStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class TodoItem:
    id: str
    content: str
    status: TodoStatus = "pending"


@dataclass
class TodoStore:
    """In-memory task-local todo state."""

    items: list[TodoItem] = field(default_factory=list)

    def replace(self, raw_items: list[dict[str, Any]]) -> None:
        items: list[TodoItem] = []
        in_progress = 0
        for idx, raw in enumerate(raw_items, start=1):
            status = raw.get("status", "pending")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"invalid todo status: {status}")
            if status == "in_progress":
                in_progress += 1
            content = str(raw.get("content") or "").strip()
            if not content:
                raise ValueError("todo content cannot be empty")
            items.append(
                TodoItem(
                    id=str(raw.get("id") or idx),
                    content=content,
                    status=status,
                )
            )
        if in_progress > 1:
            raise ValueError("only one todo may be in_progress")
        self.items = items

    def has_blocking_items(self) -> bool:
        return any(item.status in ("pending", "in_progress") for item in self.items)

    def pending_summary(self) -> str:
        return "\n".join(
            f"- [{item.status}] {item.id}: {item.content}"
            for item in self.items
            if item.status in ("pending", "in_progress")
        )


@dataclass
class TodoReadTool(ToolDef):
    name: str = "todo_read"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = "Read the current task-local todo list."
    store: TodoStore = field(default_factory=TodoStore)

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **_: Any) -> ToolResult:
        return ToolResult(
            success=True,
            output=json.dumps([item.__dict__ for item in self.store.items], indent=2),
            metadata={"blocking": self.store.has_blocking_items()},
        )


@dataclass
class TodoWriteTool(ToolDef):
    name: str = "todo_write"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Replace the task-local todo list. Use statuses pending, in_progress, "
        "and completed. Keep at most one item in_progress."
    )
    store: TodoStore = field(default_factory=TodoStore)

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["items"],
            },
        )

    def execute(self, items: list[dict[str, Any]], **_: Any) -> ToolResult:
        try:
            self.store.replace(items)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(
            success=True,
            output=f"todo list updated ({len(self.store.items)} item(s))",
            metadata={"blocking": self.store.has_blocking_items()},
        )
