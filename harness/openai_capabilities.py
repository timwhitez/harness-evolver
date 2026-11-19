"""Model- and endpoint-aware OpenAI reasoning & API-surface capability rules.

Centralizes capability resolution for OpenAI-native providers (``openai``,
``azure_openai``) so validation and outbound LiteLLM request construction share
one authoritative source of truth:

- Distinguishes unspecified effort (``None``), explicit off (``"none"``), and
  each non-zero effort level (``minimal``, ``low``, ``medium``, ``high``,
  ``xhigh``, ``max``).
- Validates against specific model capabilities instead of a provider-wide
  fixed set; never blindly opens ``max`` or ``minimal`` to all OpenAI models
  and never silently downgrades unsupported levels (for example ``none`` ->
  ``low`` on mandatory-thinking models).
- Validates Responses vs Chat Completions and tool-calling compatibility before
  sending requests, including LiteLLM 1.98.x compatibility handling for GPT-6.x
  models and ``reasoning_effort="max"`` over the Responses bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


ALL_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
DEFAULT_UNKNOWN_OPENAI_EFFORTS: frozenset[str] = frozenset(
    {"none", "low", "medium", "high", "xhigh"}
)
VALID_OPENAI_APIS: frozenset[str] = frozenset({"chat", "responses"})

_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class OpenAIModelCapabilities:
    """Resolved reasoning and API-surface capabilities for one OpenAI model."""

    base_model: str
    supported_efforts: frozenset[str]
    supports_chat_completions: bool = True
    supports_responses: bool = True
    requires_responses_for_tools: bool = False
    supports_tools: bool = True
    default_effort: str | None = None
    known_model: bool = True
    needs_allowed_reasoning_param: bool = False


def parse_openai_model_spec(model: str) -> tuple[str, str | None]:
    """Return ``(normalized_base_model, explicit_api_from_model_prefix)``.

    Recognizes optional provider prefixes (``openai/``, ``azure/``,
    ``azure_openai/``) and explicit Responses prefixes (``responses/``,
    ``openai/responses/``, ``azure/responses/``).
    """

    raw = str(model or "").strip()
    explicit_api: str | None = None
    parts = [part for part in raw.split("/") if part]
    if parts and parts[0].lower() in {"openai", "azure", "azure_openai"}:
        parts = parts[1:]
    if parts and parts[0].lower() == "responses":
        explicit_api = "responses"
        parts = parts[1:]
    base_model = "/".join(parts) if parts else raw
    return base_model, explicit_api


def _strip_date_suffix(model_lower: str) -> str:
    return _DATE_SUFFIX_RE.sub("", model_lower)


def _litellm_model_cost_entry(base_model: str) -> dict[str, object] | None:
    try:
        import litellm  # type: ignore[import-untyped]
    except Exception:
        return None
    cost_map = getattr(litellm, "model_cost", None)
    if not isinstance(cost_map, dict):
        return None
    for candidate in (base_model, base_model.lower(), f"openai/{base_model.lower()}"):
        entry = cost_map.get(candidate)
        if isinstance(entry, dict) and entry.get("litellm_provider") in {None, "openai"}:
            return entry
    return None


def _capabilities_from_litellm_entry(
    base_model: str,
    entry: dict[str, object],
) -> OpenAIModelCapabilities:
    supports_reasoning = bool(entry.get("supports_reasoning"))
    mode = str(entry.get("mode") or "chat").lower()
    supports_tools = entry.get("supports_function_calling") is not False
    supports_chat = mode != "responses"

    if not supports_reasoning:
        supported_efforts: frozenset[str] = frozenset()
    else:
        efforts = {"medium", "high"}
        if entry.get("supports_low_reasoning_effort") is not False:
            efforts.add("low")
        if entry.get("supports_none_reasoning_effort") is True:
            efforts.add("none")
        if entry.get("supports_minimal_reasoning_effort") is True:
            efforts.add("minimal")
        if entry.get("supports_xhigh_reasoning_effort") is True:
            efforts.add("xhigh")
        if entry.get("supports_max_reasoning_effort") is True:
            efforts.add("max")
        supported_efforts = frozenset(efforts)

    model_lower = base_model.lower()
    requires_responses_for_tools = (not supports_chat) or model_lower.startswith("gpt-6.2")
    needs_allowed_param = model_lower.startswith("gpt-6")
    default_effort_raw = entry.get("default_reasoning_effort")
    default_effort = str(default_effort_raw).lower() if default_effort_raw else None

    return OpenAIModelCapabilities(
        base_model=base_model,
        supported_efforts=supported_efforts,
        supports_chat_completions=supports_chat,
        supports_responses=True,
        requires_responses_for_tools=requires_responses_for_tools,
        supports_tools=supports_tools,
        default_effort=default_effort,
        known_model=True,
        needs_allowed_reasoning_param=needs_allowed_param,
    )


def resolve_openai_model_capabilities(model: str) -> OpenAIModelCapabilities:
    """Resolve exact effort and API-surface capabilities for an OpenAI model."""

    base_model, _ = parse_openai_model_spec(model)
    lower = base_model.lower()
    canonical = _strip_date_suffix(lower)

    # 1. Non-reasoning OpenAI models (never accept reasoning_effort).
    if (
        canonical.startswith(("gpt-4", "gpt-3.5", "chatgpt-4o"))
        or canonical.startswith("gpt-5-chat")
        or canonical.startswith("gpt-5-search-api")
    ):
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset(),
            supports_chat_completions=True,
            supports_responses=True,
            requires_responses_for_tools=False,
            supports_tools=not (canonical == "gpt-5-chat"),
            known_model=True,
        )

    # 2. GPT-6.2 family (supports max; mandatory thinking -> rejects none/minimal;
    #    tools require the Responses API path).
    if canonical.startswith("gpt-6.2"):
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"low", "medium", "high", "xhigh", "max"}),
            supports_chat_completions=True,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            default_effort="medium",
            known_model=True,
            needs_allowed_reasoning_param=True,
        )

    # 3. GPT-6 / GPT-6.1 families.
    if canonical.startswith(("gpt-6-", "gpt-6.1-")) or canonical in {"gpt-6", "gpt-6.1"}:
        entry = _litellm_model_cost_entry(base_model)
        if entry is not None:
            return _capabilities_from_litellm_entry(base_model, entry)
        supports_none = canonical in {"gpt-6-luna", "gpt-6-sol", "gpt-6.1-terra"}
        efforts = {"low", "medium", "high", "xhigh", "max"}
        if supports_none:
            efforts.add("none")
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset(efforts),
            supports_chat_completions=True,
            supports_responses=True,
            requires_responses_for_tools=False,
            supports_tools=True,
            known_model=True,
            needs_allowed_reasoning_param=True,
        )

    # 4. GPT-5.x Pro & Codex Responses-only models.
    if canonical == "gpt-5.5-pro":
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"medium", "high", "xhigh"}),
            supports_chat_completions=False,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            known_model=True,
        )
    if canonical == "gpt-5.4-pro":
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"low", "medium", "high", "xhigh"}),
            supports_chat_completions=False,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            known_model=True,
        )
    if canonical == "gpt-5.2-pro":
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"minimal", "low", "medium", "high", "xhigh"}),
            supports_chat_completions=False,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            known_model=True,
        )
    if canonical == "gpt-5-pro":
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"minimal", "low", "medium", "high"}),
            supports_chat_completions=False,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            known_model=True,
        )
    if canonical.startswith("gpt-5.3-codex"):
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"none", "low", "medium", "high", "xhigh"}),
            supports_chat_completions=False,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            known_model=True,
        )
    if canonical.startswith(("gpt-5.2-codex", "gpt-5.1-codex-max")):
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"low", "medium", "high", "xhigh"}),
            supports_chat_completions=False,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            known_model=True,
        )
    if canonical.startswith(("gpt-5-codex", "gpt-5.1-codex")):
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"low", "medium", "high"}),
            supports_chat_completions=False,
            supports_responses=True,
            requires_responses_for_tools=True,
            supports_tools=True,
            known_model=True,
        )

    # 5. GPT-5.2 / 5.4 / 5.5 / 5.6 standard models (support none..xhigh, reject minimal/max).
    if canonical.startswith(("gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.2")):
        default_effort = "medium" if canonical.startswith("gpt-5.5") else "none"
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"none", "low", "medium", "high", "xhigh"}),
            supports_chat_completions=True,
            supports_responses=True,
            requires_responses_for_tools=False,
            supports_tools=True,
            default_effort=default_effort,
            known_model=True,
        )

    # 6. GPT-5.1 standard (supports none..high, rejects minimal/xhigh/max).
    if canonical.startswith("gpt-5.1"):
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"none", "low", "medium", "high"}),
            supports_chat_completions=True,
            supports_responses=True,
            requires_responses_for_tools=False,
            supports_tools=True,
            default_effort="none",
            known_model=True,
        )

    # 7. GPT-5 base / mini / nano (supports minimal..high; mandatory thinking -> rejects none/xhigh/max).
    if canonical in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"minimal", "low", "medium", "high"}),
            supports_chat_completions=True,
            supports_responses=True,
            requires_responses_for_tools=False,
            supports_tools=True,
            default_effort="medium",
            known_model=True,
        )

    # 8. O-series reasoning models (o1, o3, o3-mini, o4-mini, o*-pro, o*-deep-research).
    if re.match(r"^o[134](?:$|[-.])", canonical):
        is_deep_research = "deep-research" in canonical
        is_responses_only = is_deep_research or canonical.endswith("-pro")
        return OpenAIModelCapabilities(
            base_model=base_model,
            supported_efforts=frozenset({"low", "medium", "high"}),
            supports_chat_completions=not is_responses_only,
            supports_responses=True,
            requires_responses_for_tools=is_responses_only,
            supports_tools=not is_deep_research,
            default_effort="medium",
            known_model=True,
        )

    # 9. Consult installed LiteLLM model_cost map for any other known OpenAI model.
    entry = _litellm_model_cost_entry(base_model)
    if entry is not None:
        return _capabilities_from_litellm_entry(base_model, entry)

    # 10. Unknown OpenAI-native model: conservative baseline (never blindly opens max/minimal).
    return OpenAIModelCapabilities(
        base_model=base_model,
        supported_efforts=DEFAULT_UNKNOWN_OPENAI_EFFORTS,
        supports_chat_completions=True,
        supports_responses=True,
        requires_responses_for_tools=False,
        supports_tools=True,
        known_model=False,
    )


def validate_openai_effort(
    *,
    provider: str,
    model: str,
    effort: str | None,
    api: str | None = None,
) -> OpenAIModelCapabilities:
    """Validate OpenAI model + API surface + explicit effort before any request."""

    caps = resolve_openai_model_capabilities(model)
    _, prefix_api = parse_openai_model_spec(model)
    if api is not None and prefix_api is not None and api != prefix_api:
        raise ValueError(
            f"Conflicting OpenAI API selection for model {model!r}: "
            f"api={api!r} contradicts model prefix {prefix_api!r}"
        )
    effective_api = api or prefix_api

    if effective_api == "chat" and not caps.supports_chat_completions:
        raise ValueError(
            f"OpenAI model {caps.base_model!r} does not support Chat Completions "
            f"(provider={provider!r}); use the Responses API instead"
        )

    if effort is None:
        return caps

    if not caps.supported_efforts:
        raise ValueError(
            f"OpenAI model {caps.base_model!r} (provider={provider!r}) does not support "
            f"reasoning effort (received {effort!r}); omit reasoning.effort"
        )

    if effort not in caps.supported_efforts:
        allowed = sorted(caps.supported_efforts)
        if effort == "none":
            raise ValueError(
                f"OpenAI model {caps.base_model!r} (provider={provider!r}) requires reasoning "
                f"and does not support reasoning effort 'none'; allowed: {allowed}"
            )
        raise ValueError(
            f"OpenAI model {caps.base_model!r} (provider={provider!r}) does not support "
            f"reasoning effort {effort!r}; allowed: {allowed}"
        )

    return caps


def resolve_openai_request_api(
    *,
    provider: str,
    model: str,
    api: str | None = None,
    with_tools: bool = False,
) -> tuple[str, OpenAIModelCapabilities]:
    """Validate request-time tool/API capability and choose ``'chat'`` or ``'responses'``."""

    caps = resolve_openai_model_capabilities(model)
    _, prefix_api = parse_openai_model_spec(model)
    effective_api = api or prefix_api

    if with_tools and not caps.supports_tools:
        raise ValueError(
            f"OpenAI model {caps.base_model!r} (provider={provider!r}) does not support "
            "tool/function calling"
        )

    if effective_api == "chat":
        if not caps.supports_chat_completions:
            raise ValueError(
                f"OpenAI model {caps.base_model!r} (provider={provider!r}) does not support "
                "Chat Completions; use the Responses API"
            )
        if with_tools and caps.requires_responses_for_tools:
            raise ValueError(
                f"OpenAI model {caps.base_model!r} (provider={provider!r}) with tools "
                "requires the Responses API and is unsupported on Chat Completions"
            )
        return "chat", caps

    if effective_api == "responses":
        return "responses", caps

    # Auto-select Responses when required by model or by model+tools capability.
    if not caps.supports_chat_completions or (with_tools and caps.requires_responses_for_tools):
        return "responses", caps

    return "chat", caps
