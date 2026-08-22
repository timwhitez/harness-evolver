#!/usr/bin/env python3
"""Bootstrap the initial harness configuration.

Creates the default harness config and initializes the component registry.
Run this once after installing the project.

Usage:
  python scripts/bootstrap_harness.py
  python scripts/bootstrap_harness.py --output config/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Bootstrap initial harness config")
    parser.add_argument("--output", type=str, default="config/default.yaml",
                        help="Output path for harness config")
    args = parser.parse_args()

    from harness.config import HarnessConfig, ComponentRef

    config = HarnessConfig(version="0.1.0")

    # Register runtime prompt policy components. The active Worker prompt and
    # task-context policy lives in the Rust Worker core; Python prompt modules
    # are retained as historical/template utilities.
    for name, path in [
        ("worker_policy", "crates/hl-worker-core/src/main.rs"),
        ("recovery", "harness/prompts/recovery.py"),
    ]:
        config.prompts[name] = ComponentRef(
            name=name, path=path, version="0.1.0", content_hash="initial"
        )

    # Register all tool components
    for name, path in [
        ("bash", "harness/tools/shell.py"),
        ("read", "harness/tools/file_read.py"),
        ("edit", "harness/tools/file_edit.py"),
        ("write", "harness/tools/file_write.py"),
        ("grep", "harness/tools/search.py"),
        ("glob", "harness/tools/search.py"),
    ]:
        config.tools[name] = ComponentRef(
            name=name, path=path, version="0.1.0", content_hash="initial"
        )

    config.save(args.output)
    print(f"Harness config bootstrapped to: {args.output}")
    print(f"Components registered: {len(config.get_all_components())}")

    # Verify the config can be loaded back
    loaded = HarnessConfig.from_yaml(args.output)
    print(f"Verified: config loads correctly (version {loaded.version})")


if __name__ == "__main__":
    main()
