"""Task prompt — wraps TerminalBench task instructions for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskPrompt:
    """Task instruction wrapper with context injection.

    Wraps the raw TerminalBench instruction.md with:
    - Environment context (OS, installed packages, git state)
    - Task metadata (domain, difficulty)
    - Behavioral guidance for benchmark mode
    """

    name: str = "task_prompt"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)

    template: str = """## Task

{{ instruction }}

{% if environment_context %}
## Environment Context
{{ environment_context }}
{% endif %}

## Task Metadata
- Domain: {{ domain }}
- Difficulty: {{ difficulty }}
- Task ID: {{ task_id }}

## Guidance
- Work in the current directory unless instructed otherwise.
- All required files and dependencies are already present in the container.
- Read the instruction carefully and follow it exactly.
- Verify your solution before declaring completion.
- Use todo_write before substantial multi-step work and keep at most one item in_progress.
- Completion means ready for Harbor verification; signal it by calling the done tool, and Harbor/verifier still decides pass/fail.
- Keep each operation bounded while continuing the task. Prefer commands that finish quickly and produce inspectable output; if a path is not converging, pivot to a smaller or faster strategy instead of waiting on a long command.
- If the task requires specific files, directories, commands, or report formats, create the expected artifact shape early and check it exists before deep optimization.
- After meaningful edits, run a bounded visible verifier/local check by mid-run and again near completion. Treat failing output as the next debugging target.
- Do not access the Terminal-Bench website, Terminal-Bench GitHub repository, Harbor internals, Terminal-Bench internals, hidden verifier files, official solutions, or benchmark definitions.
- Only the master HL orchestrator may create sub-agents. Do not start Codex, OpenAI Codex/openai-codex, Claude, ForgeCode, Factory Droid, Factory/factory, Droid/droid, Gemini, OpenCode, Aider, Amp, Cursor Agent, or another external coding-agent CLI from the Worker or a sub-agent, and do not create nested sub-agents through scripts, prompts, configs, or tools.
{% if verification_guidance %}
{{ verification_guidance }}
{% endif %}
{% if previous_errors %}
## Previous Attempts (Learn From These)
{% for err in previous_errors %}
- {{ err }}
{% endfor %}
{% endif %}
"""

    def render(self, context: dict[str, Any]) -> str:
        from jinja2 import Template

        t = Template(self.template)
        return t.render(**context)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if "{{ instruction }}" not in self.template:
            errors.append("missing {{ instruction }} placeholder")
        return errors

    def raw_content(self) -> str:
        return self.template
