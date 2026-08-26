"""Current-main compatibility surface for the reliable grep integration."""

from __future__ import annotations

import harness.tools._search_issue4_fixed_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)
