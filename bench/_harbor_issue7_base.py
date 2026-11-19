"""Harbor orchestration with secret-safe endpoint host metadata.

The audit-baseline runner remains byte-for-byte in
:mod:`bench._harbor_issue9_base`; only URL-to-host metadata extraction is
replaced here.
"""

from __future__ import annotations

from bench import _harbor_issue9_base as _base
from harness.url_safety import canonical_endpoint_host

for _name in dir(_base):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_base, _name))


def _base_url_host(self: object, base_url: str) -> str:
    host = canonical_endpoint_host(base_url)
    return host or "<invalid>"


_base.HarborRunner._base_url_host = _base_url_host
HarborRunner = _base.HarborRunner
