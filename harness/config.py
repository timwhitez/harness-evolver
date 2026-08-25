"""Versioned harness configuration.

HarnessConfig is the single source of truth for the current harness state.
Every component version is tracked.  The config is serialized as YAML
and versioned alongside the code in git.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


OPENAI_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
OPENAI_COMPATIBLE_REASONING_EFFORTS = OPENAI_REASONING_EFFORTS | {"max"}
ANTHROPIC_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}
ANTHROPIC_LIKE_PROVIDERS = {"anthropic", "openrouter", "forge", "forgecode"}
OPENAI_NATIVE_PROVIDERS = {"openai", "azure_openai"}
OPENAI_COMPATIBLE_PROVIDERS = {"local", "openai_compatible"}
OPENAI_LIKE_PROVIDERS = OPENAI_NATIVE_PROVIDERS | OPENAI_COMPATIBLE_PROVIDERS


class ReasoningConfig(BaseModel):
    """Provider-neutral reasoning policy.

    OpenAI-style providers use ``effort``. Anthropic-like providers may also
    use ``max_tokens`` for extended thinking budgets. ``exclude`` is kept as a
    provider-neutral switch for providers that support hiding reasoning traces.
    """

    effort: str | None = "none"
    max_tokens: int | None = None
    exclude: bool = False

    @field_validator("effort")
    @classmethod
    def validate_effort_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.lower()
        allowed = OPENAI_REASONING_EFFORTS | ANTHROPIC_REASONING_EFFORTS
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported reasoning effort {value!r}. Allowed: {sorted(allowed)}"
            )
        return normalized

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("reasoning.max_tokens must be non-negative")
        return value


class RoleModelConfig(BaseModel):
    """Model/provider settings for one role in the HL system."""

    provider: str | None = None
    runner: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    model: str
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    max_output_tokens: int | None = None
    # Single model-provider request timeout. This bounds one LiteLLM call only;
    # the Worker records timeout errors and continues recovery instead of using
    # this as a master, sub-agent, or Worker loop stop condition.
    timeout_seconds: int | None = None
    max_retries: int | None = None
    sandbox: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str | None) -> str | None:
        return value.lower() if value else value

    @model_validator(mode="after")
    def validate_provider_reasoning(self) -> "RoleModelConfig":
        provider = self.provider or self.runner or "openai"
        provider = provider.lower()
        effort = self.reasoning.effort or "none"

        if provider in OPENAI_NATIVE_PROVIDERS and effort not in OPENAI_REASONING_EFFORTS:
            raise ValueError(
                f"Provider {provider!r} does not accept reasoning effort {effort!r}; "
                f"use one of {sorted(OPENAI_REASONING_EFFORTS)}"
            )

        if (
            provider in OPENAI_COMPATIBLE_PROVIDERS
            and effort not in OPENAI_COMPATIBLE_REASONING_EFFORTS
        ):
            raise ValueError(
                f"Provider {provider!r} does not accept reasoning effort {effort!r}; "
                f"use one of {sorted(OPENAI_COMPATIBLE_REASONING_EFFORTS)}"
            )

        if provider in ANTHROPIC_LIKE_PROVIDERS and effort not in ANTHROPIC_REASONING_EFFORTS:
            raise ValueError(
                f"Provider {provider!r} does not accept reasoning effort {effort!r}; "
                f"use one of {sorted(ANTHROPIC_REASONING_EFFORTS)}"
            )

        if provider in OPENAI_LIKE_PROVIDERS and self.reasoning.max_tokens is not None:
            raise ValueError(
                "reasoning.max_tokens is reserved for Anthropic/OpenRouter/Forge-style "
                "thinking budgets; use max_output_tokens for OpenAI-style output length"
            )

        return self

    def redacted(self) -> dict[str, Any]:
        """Return a log-safe representation without secret values."""
        data = self.model_dump(mode="json")
        if self.api_key_env:
            data["api_key_env"] = self.api_key_env
            data["api_key"] = "<from env>"
        if self.base_url:
            data["base_url_host"] = self.base_url.split("//")[-1].split("/")[0]
        data.pop("extra", None)
        return data

    def litellm_kwargs(self) -> dict[str, Any]:
        """Build kwargs for LiteLLM without exposing raw secrets."""
        kwargs: dict[str, Any] = {"model": self._litellm_model_name()}
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.api_key_env and os.environ.get(self.api_key_env):
            kwargs["api_key"] = os.environ[self.api_key_env]
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        if self.max_retries is not None:
            kwargs["num_retries"] = self.max_retries
        if self.reasoning.effort and self.reasoning.effort != "none":
            kwargs["reasoning_effort"] = self.reasoning.effort
            if self._is_deepseek_openai_compatible():
                kwargs["allowed_openai_params"] = ["reasoning_effort", "thinking"]
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if self.reasoning.max_tokens is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.reasoning.max_tokens}
        kwargs.update(self.extra)
        return kwargs

    def _litellm_model_name(self) -> str:
        provider = (self.provider or "").lower()
        if provider in {"local", "openai_compatible"} and "/" not in self.model:
            return f"openai/{self.model}"
        return self.model

    def _is_deepseek_openai_compatible(self) -> bool:
        provider = (self.provider or "").lower()
        base_url = (self.base_url or "").lower()
        model = self.model.lower()
        return (
            provider in {"local", "openai_compatible"}
            and ("deepseek" in base_url or model.startswith("deepseek-"))
        )


class ProviderRequestDefaults(BaseModel):
    """Validated defaults that have a complete path into provider requests."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: StrictInt | None = Field(
        default=None,
        validation_alias=AliasChoices("timeout_seconds", "timeout"),
    )
    max_retries: StrictInt | None = None

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("defaults.timeout must be positive")
        return value

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("defaults.max_retries must be non-negative")
        return value


class ModelsConfig(BaseModel):
    """Role-indexed provider configuration loaded from ``config/models.yaml``."""

    roles: dict[str, RoleModelConfig] = Field(default_factory=dict)
    defaults: ProviderRequestDefaults = Field(default_factory=ProviderRequestDefaults)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelsConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        if isinstance(data.get("models"), dict):
            data = data["models"]
        if "roles" not in data:
            legacy_roles = {
                key: value
                for key, value in data.items()
                if isinstance(value, dict) and key != "defaults"
            }
            data = {"roles": legacy_roles, "defaults": data.get("defaults", {})}
        return cls(**data)

    def get_role(self, role: str) -> RoleModelConfig:
        if role not in self.roles:
            raise KeyError(f"Model role {role!r} is not configured")

        configured = self.roles[role]
        updates: dict[str, Any] = {}
        if configured.timeout_seconds is None and self.defaults.timeout_seconds is not None:
            updates["timeout_seconds"] = self.defaults.timeout_seconds
        if configured.max_retries is None and self.defaults.max_retries is not None:
            updates["max_retries"] = self.defaults.max_retries
        return configured.model_copy(update=updates) if updates else configured

    def redacted(self) -> dict[str, Any]:
        return {
            "roles": {name: self.get_role(name).redacted() for name in self.roles},
            "defaults": self.defaults.model_dump(mode="json"),
        }


class ComponentRef(BaseModel):
    """Reference to a harness component with its version."""

    name: str
    path: str
    version: str
    content_hash: str
    dependencies: list[str] = Field(default_factory=list)
    enabled: bool = True


class HarnessConfig(BaseModel):
    """Full harness configuration — the versioned state of all components."""

    version: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Component registrations
    prompts: dict[str, ComponentRef] = Field(default_factory=dict)
    tools: dict[str, ComponentRef] = Field(default_factory=dict)
    planning: dict[str, ComponentRef] = Field(default_factory=dict)
    context: dict[str, ComponentRef] = Field(default_factory=dict)
    entrypoint: dict[str, ComponentRef] = Field(default_factory=dict)
    recovery: dict[str, ComponentRef] = Field(default_factory=dict)
    verification: dict[str, ComponentRef] = Field(default_factory=dict)

    # Global settings
    model: str = "claude-sonnet-4-6"
    # Compatibility/progress reference only. ``0`` disables turn-count based
    # late-run heuristics; Worker loops do not have a turn-count stop condition.
    # Legacy configs may still provide ``max_turns``; new serialized configs use
    # the explicit audit-only field name.
    max_turns_audit: int = Field(
        default=0,
        validation_alias=AliasChoices("max_turns_audit", "max_turns"),
    )
    # Legacy single provider-request timeout reference. This is not a Worker,
    # sub-agent, or master loop deadline.
    timeout_seconds: int = 1800
    reasoning: ReasoningConfig = Field(
        default_factory=lambda: ReasoningConfig(effort="xhigh", max_tokens=16000)
    )
    thinking_budget: int = 16000  # Backward-compatible alias; prefer reasoning.max_tokens.

    @classmethod
    def create_default(cls) -> HarnessConfig:
        return cls(version="0.1.0")

    def get_all_components(self) -> dict[str, ComponentRef]:
        """Flatten all component categories into one dict."""
        all_refs: dict[str, ComponentRef] = {}
        for category in [
            "prompts",
            "tools",
            "planning",
            "context",
            "entrypoint",
            "recovery",
            "verification",
        ]:
            for name, ref in getattr(self, category).items():
                all_refs[f"{category}/{name}"] = ref
        return all_refs

    def get_enabled_components(self) -> dict[str, ComponentRef]:
        return {k: v for k, v in self.get_all_components().items() if v.enabled}

    def bump_version(self, component_name: str) -> str:
        """Increment version for a component after edit."""
        ref = self._find_component(component_name)
        if ref is None:
            raise KeyError(f"Component {component_name} not found")
        parts = ref.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        ref.version = ".".join(parts)
        self.updated_at = datetime.now()
        return ref.version

    def update_hash(self, component_name: str, content: str) -> str:
        """Update the content hash for a component."""
        ref = self._find_component(component_name)
        if ref is None:
            raise KeyError(f"Component {component_name} not found")
        ref.content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return ref.content_hash

    def _find_component(self, name: str) -> ComponentRef | None:
        for category in [
            "prompts",
            "tools",
            "planning",
            "context",
            "entrypoint",
            "recovery",
            "verification",
        ]:
            refs = getattr(self, category)
            if name in refs:
                return refs[name]
        return None

    def to_yaml(self) -> str:
        import yaml

        return yaml.dump(self.model_dump(mode="json"), default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HarnessConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            f.write(self.to_yaml())