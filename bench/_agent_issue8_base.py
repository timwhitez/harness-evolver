"""HLAgent facade that launches the packaged Rust Worker runtime."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from bench import _agent_bridge as _base
from bench.runtime_resources import worker_crate_root


# Preserve the complete Python bridge surface while overriding only Worker
# runtime discovery. This keeps existing private test and adapter imports stable.
for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)


def _worker_binary_name() -> str:
    return "hl-worker-core.exe" if os.name == "nt" else "hl-worker-core"


class HLAgent(_base.HLAgent):
    """Use a packaged, writable Worker crate instead of a source-checkout path."""

    def _rust_worker_command(self) -> list[str]:
        explicit = os.environ.get("HL_WORKER_RUST_BIN")
        if explicit:
            try:
                binary = Path(explicit).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"HL_WORKER_RUST_BIN does not reference an accessible file: {explicit}"
                ) from exc
            if not binary.is_file():
                raise RuntimeError(f"HL_WORKER_RUST_BIN does not reference a file: {binary}")
            if os.name != "nt" and not os.access(binary, os.X_OK):
                raise RuntimeError(f"HL_WORKER_RUST_BIN is not executable: {binary}")
            return [str(binary)]

        crate_root = worker_crate_root()
        binary_name = _worker_binary_name()
        for profile in ("release", "debug"):
            candidate = crate_root / "target" / profile / binary_name
            if (
                candidate.is_file()
                and (os.name == "nt" or os.access(candidate, os.X_OK))
                and self._rust_worker_binary_is_current(candidate, crate_root)
            ):
                return [str(candidate)]

        manifest = crate_root / "Cargo.toml"
        cargo = shutil.which("cargo")
        rustup = shutil.which("rustup")
        if rustup is not None:
            command = [rustup, "run", "stable", "cargo"]
        elif cargo is not None:
            command = [cargo]
        else:
            raise RuntimeError(
                "The packaged Rust Worker is not built and Cargo is unavailable. "
                "Install a Rust toolchain or set HL_WORKER_RUST_BIN to a compatible "
                "hl-worker-core executable."
            )
        command.extend(
            [
                "run",
                "--quiet",
                "--locked",
                "--manifest-path",
                str(manifest),
                "--",
            ]
        )
        return command
