"""HarnessRegistry — typed registry for all harness components.

Provides type-safe registration and lookup of harness components
with dependency resolution and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from hl.protocol import Policy

P = TypeVar("P", bound=Policy)


@dataclass
class HarnessRegistry:
    """Typed registry for harness components with dependency ordering.

    Components are registered by category and name.  The registry
    ensures dependency consistency and provides ordered iteration
    for context rendering.
    """

    _components: dict[str, Policy] = field(default_factory=dict)
    _categories: dict[str, list[str]] = field(default_factory=dict)

    def register(
        self, component: Policy, category: str = "default"
    ) -> None:
        key = f"{category}/{component.name}"
        self._components[key] = component
        if category not in self._categories:
            self._categories[category] = []
        if component.name not in self._categories[category]:
            self._categories[category].append(component.name)

    def get(self, category: str, name: str) -> Policy | None:
        key = f"{category}/{name}"
        return self._components.get(key)

    def get_all(self, category: str | None = None) -> list[Policy]:
        if category:
            names = self._categories.get(category, [])
            return [self._components[f"{category}/{n}"] for n in names]
        return list(self._components.values())

    def render_all(
        self, context: dict[str, Any], category: str | None = None
    ) -> str:
        """Render all components in dependency order."""
        components = self.get_all(category)
        # Topological sort by dependencies
        ordered = self._topo_sort(components)
        parts = []
        for comp in ordered:
            rendered = comp.render(context)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts)

    def validate_all(self) -> list[str]:
        """Return list of validation errors across all components."""
        errors: list[str] = []
        for key, comp in self._components.items():
            comp_errors = comp.validate()
            for err in comp_errors:
                errors.append(f"[{key}] {err}")

        # Check dependency integrity
        all_names = {c.name for c in self._components.values()}
        for comp in self._components.values():
            for dep in comp.dependencies:
                if dep not in all_names:
                    errors.append(
                        f"[{comp.name}] missing dependency: {dep}"
                    )
        return errors

    def _topo_sort(self, components: list[Policy]) -> list[Policy]:
        """Topological sort by dependency count (simplified Kahn)."""
        name_to_comp = {c.name: c for c in components}
        in_degree = {c.name: len([d for d in c.dependencies if d in name_to_comp])
                     for c in components}
        queue = [c for c in components if in_degree[c.name] == 0]
        result: list[Policy] = []

        while queue:
            c = queue.pop(0)
            result.append(c)
            # Decrease in-degree of dependents
            for other in components:
                if c.name in other.dependencies:
                    in_degree[other.name] -= 1
                    if in_degree[other.name] == 0 and other not in result:
                        queue.append(other)

        # Append any remaining (circular deps are rendered last)
        for c in components:
            if c not in result:
                result.append(c)

        return result

    def __len__(self) -> int:
        return len(self._components)
