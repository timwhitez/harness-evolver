"""Regression coverage for lightweight ``bench`` package imports."""

import subprocess
import sys


def test_network_environment_import_does_not_eagerly_load_litellm():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bench.network_environment; "
                "assert 'litellm' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert completed.stderr == ""


def test_bench_public_exports_remain_discoverable_without_importing_them():
    import bench

    assert set(bench.__all__) <= set(dir(bench))


def test_codex_update_import_does_not_eagerly_load_legacy_meta_agent():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import meta.codex_update; "
                "assert 'meta.agent' not in sys.modules; "
                "assert 'litellm' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert completed.stderr == ""
