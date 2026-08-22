"""Base prompt template class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jinja2 import Template


@dataclass
class PromptTemplate:
    """A versioned, renderable prompt template.

    Uses Jinja2 for templating.  Templates are versioned and
    content-hashed so the meta-agent can track changes.
    """

    name: str
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    template: str = ""
    _rendered_cache: str = field(default="", repr=False)

    def render(self, context: dict[str, Any]) -> str:
        """Render the template with given context variables."""
        if not self.template:
            return ""
        t = Template(self.template)
        result = t.render(**context)
        self._rendered_cache = result
        return result

    def validate(self) -> list[str]:
        """Check template syntax."""
        errors: list[str] = []
        if not self.template.strip():
            errors.append("template is empty")
        try:
            Template(self.template)
        except Exception as e:
            errors.append(f"template syntax error: {e}")
        return errors

    def raw_content(self) -> str:
        return self.template
