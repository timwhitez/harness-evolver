"""Public Harbor adapter facade preserving the full stacked tool registry.

The counting/error-aware grep implementation lives in
:mod:`bench._harbor_adapter_issue13_impl`; the strict host-side audit wrapper in
:mod:`bench._harbor_adapter_issue13_audit` validates all configured bounds before
any environment access. Earlier wrappers updated only the immediate compatibility
module, while the actual ``HLWorkerHarborAgent`` class was defined one layer
deeper. Its registry therefore continued to instantiate the previous grep class.
This facade re-exports the complete inherited surface and patches the module that
owns the registry method.
"""

from __future__ import annotations

from bench import _harbor_adapter_issue13_audit as _impl

# PR #38's compatibility module delegates its agent class to the canonical
# adapter module inherited from PR #41. Method globals are resolved in that
# defining module, not in the intermediate wrapper.
_registry_module = _impl._base._base

# Preserve the historical public import surface while allowing each stacked
# layer to override the classes it owns. This avoids hiding file/search tool
# classes behind a narrow facade.
for _module in (_registry_module, _impl._base, _impl):
    for _name, _value in vars(_module).items():
        if not (_name.startswith("__") and _name.endswith("__")):
            globals()[_name] = _value

HarborGrepTool = _impl.HarborGrepTool
_registry_module.HarborGrepTool = HarborGrepTool
HLWorkerHarborAgent = _registry_module.HLWorkerHarborAgent