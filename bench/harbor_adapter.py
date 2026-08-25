"""Harbor adapter facade stacking exact glob fallback on canonical file tools."""

from __future__ import annotations

from bench import _harbor_adapter_issue14_base as _base
from bench._harbor_glob_issue14_order_guard import HarborGlobTool

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_registry_module = _base._registry_module
_registry_module.HarborGlobTool = HarborGlobTool
HLWorkerHarborAgent = _registry_module.HLWorkerHarborAgent
