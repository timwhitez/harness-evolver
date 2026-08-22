"""Recovery prompt — injected when the agent encounters errors.

Error-specific guidance templates that help the agent recover from
common failure patterns without losing context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecoveryPrompt:
    """Error recovery guidance injected into the agent context.

    When the agent encounters a known error pattern, this prompt
    is injected to guide recovery.  Patterns are learned over time
    by the meta-agent from failure trajectories.
    """

    name: str = "recovery_prompt"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)

    template: str = """## Error Recovery Guidance

An error was encountered. Before retrying, consider:

1. **Read the error message carefully** — what exactly failed?
2. **Check the current state** — is the file system in the expected state?
3. **Don't repeat the same approach** — if it failed once, try a different strategy.
4. **Verify intermediate steps** — add checks before assuming something worked.

{% if error_type %}
Error type: {{ error_type }}
{% endif %}

{% if known_pattern %}
Known pattern: {{ known_pattern }}
Suggested recovery: {{ recovery_strategy }}
{% endif %}
"""

    def render(self, context: dict[str, Any]) -> str:
        from jinja2 import Template

        t = Template(self.template)
        return t.render(**context)

    def validate(self) -> list[str]:
        return []

    def raw_content(self) -> str:
        return self.template
