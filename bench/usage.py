"""Worker usage v1: input excludes cache hits; cache is a known input subset.

Provider prompt/input aliases describe inclusive input at the LiteLLM boundary.
Missing or invalid counters stay unknown, including an unobserved cache count.
"""

from typing import Any

USAGE_SCHEMA = "worker_usage_v1_exclusive_input"


def normalize_worker_usage(usage: Any) -> tuple[dict[str, int], dict[str, Any]]:
    def get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    diagnostics = []

    def select(name, values):
        present = [value for value in values if value is not None]
        if not present:
            diagnostics.append(f"missing_{name}")
            return None
        if any(type(value) is not int or value < 0 or value > 2**63 - 1 for value in present):
            diagnostics.append(f"invalid_{name}")
            return None
        if any(value != present[0] for value in present[1:]):
            diagnostics.append(f"conflicting_{name}")
            return None
        return present[0]

    prompt = get(usage, "prompt_tokens")
    details = get(usage, "prompt_tokens_details")
    raw_anthropic = prompt is None and details is None and get(usage, "cache_read_input_tokens") is not None
    inclusive = select("input", [prompt, get(usage, "input_tokens")])
    output = select("output", [get(usage, "completion_tokens"), get(usage, "output_tokens")])
    cache = select("cache", [
        get(details, "cached_tokens"),
        get(usage, "cache_read_input_tokens"),
    ])
    counts = {}
    if output is not None:
        counts["output"] = output
    if cache is not None and inclusive is not None:
        if raw_anthropic:
            creation = get(usage, "cache_creation_input_tokens")
            creation = 0 if creation is None else select("cache_creation", [creation])
            if creation is not None and inclusive + creation <= 2**63 - 1:
                counts.update(input=inclusive + creation, cache=cache)
            elif creation is not None:
                diagnostics.append("invalid_input_total")
        elif cache > inclusive:
            diagnostics.append("cache_exceeds_input")
        else:
            counts.update(input=inclusive - cache, cache=cache)
    unknown = sorted({"input", "cache", "output"} - counts.keys())
    invalid = any(not reason.startswith("missing_") for reason in diagnostics)
    return counts, {
        "schema": USAGE_SCHEMA,
        "status": "invalid" if invalid else "incomplete" if unknown else "complete",
        "unknown_fields": unknown,
        "diagnostics": diagnostics,
        "source_input_semantics": "raw_anthropic_exclusive" if raw_anthropic else "litellm_inclusive",
    }


def usage_coverage(trials) -> dict[str, Any]:
    """Label observed aggregate subtotals; never reconstruct legacy counters."""
    trials = list(trials)
    schemas = {trial.metadata.get("token_usage_observation", {}).get("schema", "legacy")
               for trial in trials}
    return {
        "schema": next(iter(schemas)) if len(schemas) == 1 else "mixed_or_empty",
        "trials": len(trials),
        "known_trials": {key: sum(key in trial.token_usage for trial in trials)
                         for key in ("input", "cache", "output")},
        "totals_are_observed_subtotals": True,
    }


def compatible_usage_totals(trials, totals):
    """Mixed legacy/v1 input has no reconstructible common denominator."""
    if usage_coverage(trials)["schema"] == "mixed_or_empty":
        return {key: value for key, value in totals.items() if key == "output"}
    return totals
