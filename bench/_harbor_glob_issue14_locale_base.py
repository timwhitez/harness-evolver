"""Executable C-locale path preflight for the final glob facade."""

from __future__ import annotations

from bench import _harbor_glob_issue14_semantic_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

# `[! -~]` contains an unquoted literal space and is rejected by Bash's parser
# in a `case` pattern. Under the already-forced C locale, `[:print:]` denotes
# exactly printable ASCII, so its negation provides the intended byte-safe
# preflight without introducing a parser error.
_runtime_module = _base._base
_runtime_module._GLOB_FALLBACK_SCRIPT = _runtime_module._GLOB_FALLBACK_SCRIPT.replace(
    "*[! -~]*)",
    "*[![:print:]]*)",
)

HarborGlobTool = _base.HarborGlobTool
HLWorkerHarborAgent = _base.HLWorkerHarborAgent
_unsupported_glob_reason = _base._unsupported_glob_reason
_unsupported_glob_root_reason = _base._unsupported_glob_root_reason
