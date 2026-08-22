"""Tool definitions — the agent's action space.

Every tool is a Policy in HL terms.  Tool descriptions, schemas,
and execution logic are all independently editable by the meta-agent.

Key design insight (from Claude Code):
  "Bash as universal adapter" — one powerful tool beats 100 narrow ones.
  The core tools are: Bash, Read, Edit, Write, Grep, Glob.
"""

from harness.tools.base import ToolDef, ToolSchema, ToolResult
from harness.tools.registry import ToolRegistry
from harness.tools.shell import ShellTool
from harness.tools.file_read import FileReadTool
from harness.tools.file_edit import FileEditTool
from harness.tools.file_write import FileWriteTool
from harness.tools.search import GrepTool, GlobTool
from harness.tools.todo import TodoReadTool, TodoStore, TodoWriteTool
from harness.tools.goal import GoalReadTool
from harness.tools.verify import VerifyTool
from harness.tools.correction import ToolFailureTracker

__all__ = [
    "ToolDef",
    "ToolSchema",
    "ToolResult",
    "ToolRegistry",
    "ShellTool",
    "FileReadTool",
    "FileEditTool",
    "FileWriteTool",
    "GrepTool",
    "GlobTool",
    "TodoReadTool",
    "TodoStore",
    "TodoWriteTool",
    "GoalReadTool",
    "VerifyTool",
    "ToolFailureTracker",
]
