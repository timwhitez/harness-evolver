#!/usr/bin/env python3
"""Network preflight checks for Harbor/TerminalBench campaigns."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.network_environment import (  # noqa: E402
    AptMirrorConfig,
    DEFAULT_DEBIAN_MIRROR,
    DEFAULT_DEBIAN_SECURITY_MIRROR,
    DEFAULT_DOCKER_HUB_MIRROR,
    DEFAULT_DOWNLOAD_URL_REWRITES,
    DEFAULT_PREBUILT_DOCKER_HUB_MIRROR,
    DEFAULT_PYPI_INDEX_URL,
    DEFAULT_PYPI_TRUSTED_HOST,
    DEFAULT_UBUNTU_MIRROR,
    effective_docker_image_reference,
    rewrite_download_url,
)
from hl.web_references import is_wechat_article_url, request_headers_for_url  # noqa: E402
from scripts.run_trial import (  # noqa: E402
    _apply_docker_resource_defaults,
    add_docker_resource_args,
    docker_run_resource_args,
)


# An empty value means Docker should use its normal registry configuration.
DEFAULT_DOCKER_MIRROR = ""
DEFAULT_DOCKER_IMAGES = [
    "ubuntu:24.04",
    "python:3.13-slim-bookworm",
    "python:3.11",
    "python:3.11-slim",
    "python:3.10-slim-bookworm",
    "debian:13.0-slim",
    "debian:bullseye-slim",
]
DEFAULT_PREBUILT_DOCKER_IMAGES = [
    "alexgshaw/qemu-startup:20251031",
    "alexgshaw/custom-memory-heap-crash:20251031",
    "alexgshaw/hf-model-inference:20251031",
]
DEFAULT_PREBUILT_DOCKER_PULL_IMAGES: list[str] = []
DEBIAN_SAMPLE_PATH = "/dists/stable/InRelease"
UBUNTU_SAMPLE_PATH = "/pool/main/h/hello/hello_2.10-3build1_amd64.deb"
ALPINE_SAMPLE_URL = (
    "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/"
    "alpine-extended-3.19.0-x86_64.iso"
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    repair: str = ""
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Harbor network preflight checks")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", dest="mode", action="store_const", const="quick")
    mode.add_argument("--full", dest="mode", action="store_const", const="full")
    parser.set_defaults(mode="quick")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--docker-mirror", default=DEFAULT_DOCKER_MIRROR)
    parser.add_argument("--docker-hub-mirror", default=DEFAULT_DOCKER_HUB_MIRROR)
    parser.add_argument(
        "--prebuilt-docker-hub-mirror",
        default=DEFAULT_PREBUILT_DOCKER_HUB_MIRROR,
    )
    parser.add_argument(
        "--docker-image-override",
        action="append",
        default=None,
        help=(
            "Override an implicit Docker Hub image as source=target; repeatable. "
            "Values from config/trials.yaml network.docker_image_overrides are also used."
        ),
    )
    parser.add_argument(
        "--docker-image",
        action="append",
        default=None,
        help="Implicit Docker Hub image to pull through the mirror; repeatable",
    )
    parser.add_argument(
        "--prebuilt-docker-image",
        action="append",
        default=None,
        help=(
            "Prebuilt task Docker Hub image to inspect through the prebuilt "
            "mirror; repeatable"
        ),
    )
    parser.add_argument(
        "--prebuilt-docker-pull-image",
        action="append",
        default=None,
        help=(
            "Prebuilt task Docker Hub image to pull through the prebuilt mirror; "
            "repeatable. Defaults can also come from config/trials.yaml "
            "network.prebuilt_docker_pull_images."
        ),
    )
    parser.add_argument(
        "--prebuilt-docker-pull-timeout",
        type=int,
        default=None,
        help=(
            "Timeout in seconds for prebuilt task image pulls. Defaults to "
            "config/trials.yaml network.prebuilt_docker_pull_timeout_seconds "
            "or the general --timeout."
        ),
    )
    parser.add_argument("--debian-mirror", default=DEFAULT_DEBIAN_MIRROR)
    parser.add_argument("--debian-security-mirror", default=DEFAULT_DEBIAN_SECURITY_MIRROR)
    parser.add_argument("--ubuntu-mirror", default=DEFAULT_UBUNTU_MIRROR)
    parser.add_argument(
        "--download-url-rewrite",
        action="append",
        default=None,
        help=(
            "Rewrite direct Dockerfile download URLs as source=target; repeatable. "
            "Values from config/trials.yaml network.download_url_rewrites are also used."
        ),
    )
    parser.add_argument("--pypi-index-url", default=DEFAULT_PYPI_INDEX_URL)
    parser.add_argument("--pypi-trusted-host", default=DEFAULT_PYPI_TRUSTED_HOST)
    parser.add_argument(
        "--skip-docker-pull",
        action="store_true",
        help="Skip the docker pull smoke check",
    )
    add_docker_resource_args(parser)
    args = parser.parse_args()
    docker_image_overrides: dict[str, str] = {}
    download_url_rewrites: dict[str, str] = dict(DEFAULT_DOWNLOAD_URL_REWRITES)
    trials_config = _load_trials_config(Path("config/trials.yaml"))
    config = trials_config.get("network", {}) if isinstance(trials_config, dict) else {}
    if not isinstance(config, dict):
        config = {}
    _apply_docker_resource_defaults(
        args,
        trials_config.get("docker_resources", {})
        if isinstance(trials_config.get("docker_resources", {}), dict)
        else {},
        parser,
        Path("config/trials.yaml"),
    )
    if config:
        if args.docker_mirror == DEFAULT_DOCKER_MIRROR and config.get("docker_mirror"):
            args.docker_mirror = str(config["docker_mirror"])
        if args.docker_hub_mirror == DEFAULT_DOCKER_HUB_MIRROR and config.get("docker_hub_mirror"):
            args.docker_hub_mirror = str(config["docker_hub_mirror"])
        if (
            args.prebuilt_docker_hub_mirror == DEFAULT_PREBUILT_DOCKER_HUB_MIRROR
            and config.get("prebuilt_docker_hub_mirror")
        ):
            args.prebuilt_docker_hub_mirror = str(config["prebuilt_docker_hub_mirror"])
        docker_image_overrides = _image_overrides_from_config(config)
        if args.debian_mirror == DEFAULT_DEBIAN_MIRROR and config.get("debian_mirror"):
            args.debian_mirror = str(config["debian_mirror"])
        if (
            args.debian_security_mirror == DEFAULT_DEBIAN_SECURITY_MIRROR
            and config.get("debian_security_mirror")
        ):
            args.debian_security_mirror = str(config["debian_security_mirror"])
        if args.ubuntu_mirror == DEFAULT_UBUNTU_MIRROR and config.get("ubuntu_mirror"):
            args.ubuntu_mirror = str(config["ubuntu_mirror"])
        download_url_rewrites = _url_rewrites_from_config(config)
        if args.pypi_index_url == DEFAULT_PYPI_INDEX_URL and config.get("pypi_index_url"):
            args.pypi_index_url = str(config["pypi_index_url"])
        if args.pypi_trusted_host == DEFAULT_PYPI_TRUSTED_HOST and config.get("pypi_trusted_host"):
            args.pypi_trusted_host = str(config["pypi_trusted_host"])
        if args.prebuilt_docker_pull_timeout is None and config.get(
            "prebuilt_docker_pull_timeout_seconds"
        ):
            try:
                args.prebuilt_docker_pull_timeout = int(
                    config["prebuilt_docker_pull_timeout_seconds"]
                )
            except (TypeError, ValueError):
                args.prebuilt_docker_pull_timeout = None
    for item in args.docker_image_override or []:
        source, separator, target = item.partition("=")
        if separator and source.strip() and target.strip():
            docker_image_overrides[source.strip()] = target.strip()
    for item in args.download_url_rewrite or []:
        source, separator, target = item.partition("=")
        if separator and source.strip() and target.strip():
            download_url_rewrites[source.strip()] = target.strip()

    report = run_preflight(
        mode=args.mode,
        timeout=args.timeout,
        docker_mirror=args.docker_mirror,
        docker_hub_mirror=args.docker_hub_mirror,
        prebuilt_docker_hub_mirror=args.prebuilt_docker_hub_mirror,
        docker_image_overrides=docker_image_overrides,
        download_url_rewrites=download_url_rewrites,
        docker_images=args.docker_image or DEFAULT_DOCKER_IMAGES,
        prebuilt_docker_images=_configured_list(
            args.prebuilt_docker_image,
            config.get("prebuilt_docker_images") if config else None,
            DEFAULT_PREBUILT_DOCKER_IMAGES,
        ),
        prebuilt_docker_pull_images=_configured_list(
            args.prebuilt_docker_pull_image,
            config.get("prebuilt_docker_pull_images") if config else None,
            DEFAULT_PREBUILT_DOCKER_PULL_IMAGES,
        ),
        prebuilt_docker_pull_timeout=args.prebuilt_docker_pull_timeout,
        debian_mirror=args.debian_mirror,
        debian_security_mirror=args.debian_security_mirror,
        ubuntu_mirror=args.ubuntu_mirror,
        pypi_index_url=args.pypi_index_url,
        pypi_trusted_host=args.pypi_trusted_host,
        skip_docker_pull=args.skip_docker_pull,
        docker_run_args=docker_run_resource_args(args, parser),
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


def run_preflight(
    *,
    mode: str = "quick",
    timeout: int = 120,
    docker_mirror: str = DEFAULT_DOCKER_MIRROR,
    docker_hub_mirror: str = DEFAULT_DOCKER_HUB_MIRROR,
    prebuilt_docker_hub_mirror: str = DEFAULT_PREBUILT_DOCKER_HUB_MIRROR,
    docker_image_overrides: dict[str, str] | None = None,
    download_url_rewrites: dict[str, str] | None = None,
    docker_images: list[str] | None = None,
    prebuilt_docker_images: list[str] | None = None,
    prebuilt_docker_pull_images: list[str] | None = None,
    prebuilt_docker_pull_timeout: int | None = None,
    debian_mirror: str = DEFAULT_DEBIAN_MIRROR,
    debian_security_mirror: str = DEFAULT_DEBIAN_SECURITY_MIRROR,
    ubuntu_mirror: str = DEFAULT_UBUNTU_MIRROR,
    pypi_index_url: str = DEFAULT_PYPI_INDEX_URL,
    pypi_trusted_host: str = DEFAULT_PYPI_TRUSTED_HOST,
    skip_docker_pull: bool = False,
    docker_run_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run preflight checks and return a JSON-serializable report."""

    checks: list[CheckResult] = []
    checks.append(check_docker_daemon(timeout=timeout))
    normalized_docker_mirror = docker_mirror.strip()
    if normalized_docker_mirror:
        checks.append(
            check_docker_mirror(
                expected_mirror=normalized_docker_mirror,
                timeout=timeout,
            )
        )
        registry_url = f"{normalized_docker_mirror.rstrip('/')}/v2/"
    else:
        checks.append(
            CheckResult(
                "docker_mirror",
                True,
                "no Docker registry mirror configured",
            )
        )
        registry_url = "https://registry-1.docker.io/v2/"
    registry_check = check_registry_v2(
        "docker_mirror_v2",
        registry_url,
        timeout=timeout,
    )
    checks.append(registry_check)
    prebuilt_registry_check: CheckResult | None = None
    if prebuilt_docker_hub_mirror.strip():
        prebuilt_registry_check = check_registry_v2(
            "prebuilt_docker_mirror_v2",
            f"https://{prebuilt_docker_hub_mirror.rstrip('/')}/v2/",
            timeout=timeout,
        )
        checks.append(prebuilt_registry_check)
    mirror_config = AptMirrorConfig(
        docker_hub_mirror=docker_hub_mirror,
        prebuilt_docker_hub_mirror=prebuilt_docker_hub_mirror,
        docker_image_overrides=docker_image_overrides or {},
        download_url_rewrites=(
            dict(DEFAULT_DOWNLOAD_URL_REWRITES)
            if download_url_rewrites is None
            else download_url_rewrites
        ),
        debian_mirror=debian_mirror,
        debian_security_mirror=debian_security_mirror,
        ubuntu_mirror=ubuntu_mirror,
        pypi_index_url=pypi_index_url,
        pypi_trusted_host=pypi_trusted_host,
    )
    if not skip_docker_pull:
        for image in (
            docker_images if docker_images is not None else DEFAULT_DOCKER_IMAGES
        ):
            checks.append(
                check_docker_pull(
                    effective_docker_image_reference(image, mirror_config),
                    timeout=timeout,
                    source_image=image,
                    name=f"docker_pull:{image}",
                )
        )
        _downgrade_registry_probe_if_docker_pull_succeeded(checks, registry_check.name)
    prebuilt_mirror_config = AptMirrorConfig(
        docker_hub_mirror=prebuilt_docker_hub_mirror,
        docker_image_overrides=docker_image_overrides or {},
    )
    prebuilt_manifest_images = _ordered_unique(
        list(
            prebuilt_docker_images
            if prebuilt_docker_images is not None
            else DEFAULT_PREBUILT_DOCKER_IMAGES
        )
        + list(
            prebuilt_docker_pull_images
            if prebuilt_docker_pull_images is not None
            else DEFAULT_PREBUILT_DOCKER_PULL_IMAGES
        )
    )
    for image in prebuilt_manifest_images:
        checks.append(
            check_docker_manifest(
                effective_docker_image_reference(image, prebuilt_mirror_config),
                timeout=timeout,
                source_image=image,
                name=f"prebuilt_docker_manifest:{image}",
            )
        )
    if not skip_docker_pull:
        prebuilt_pull_timeout = (
            prebuilt_docker_pull_timeout
            if prebuilt_docker_pull_timeout is not None
            else timeout
        )
        for image in (
            prebuilt_docker_pull_images
            if prebuilt_docker_pull_images is not None
            else DEFAULT_PREBUILT_DOCKER_PULL_IMAGES
        ):
            checks.append(
                check_docker_pull(
                    effective_docker_image_reference(image, prebuilt_mirror_config),
                    timeout=prebuilt_pull_timeout,
                    source_image=image,
                    name=f"prebuilt_docker_pull:{image}",
                )
            )
        _downgrade_prebuilt_pull_failures_if_manifest_succeeded(checks)
    if prebuilt_registry_check is not None:
        _downgrade_registry_probe_if_prebuilt_manifest_succeeded(
            checks,
            prebuilt_registry_check.name,
        )
    checks.append(
        check_http_head(
            "debian_package_mirror",
            f"{debian_mirror.rstrip('/')}{DEBIAN_SAMPLE_PATH}",
            timeout=timeout,
        )
    )
    checks.append(
        check_http_head(
            "debian_security_mirror",
            f"{debian_security_mirror.rstrip('/')}/dists/bookworm-security/InRelease",
            timeout=timeout,
        )
    )
    checks.append(
        check_http_head(
            "ubuntu_package_mirror",
            f"{ubuntu_mirror.rstrip('/')}{UBUNTU_SAMPLE_PATH}",
            timeout=timeout,
        )
    )
    checks.append(
        check_http_head(
            "alpine_release_mirror",
            rewrite_download_url(ALPINE_SAMPLE_URL, mirror_config),
            timeout=timeout,
        )
    )
    checks.append(
        check_http_head(
            "pypi_index_mirror",
            f"{pypi_index_url.rstrip('/')}/pip/",
            timeout=timeout,
        )
    )
    if mode == "full":
        checks.append(
            check_debian_container_download(
                debian_mirror=debian_mirror,
                debian_security_mirror=debian_security_mirror,
                mirror_config=mirror_config,
                timeout=max(timeout, 900),
                docker_run_args=docker_run_args,
            )
        )
        checks.append(
            check_ubuntu_container_download(
                ubuntu_mirror=ubuntu_mirror,
                mirror_config=mirror_config,
                timeout=max(timeout, 900),
                docker_run_args=docker_run_args,
            )
        )
        checks.append(
            check_python_container_pip_download(
                pypi_index_url=pypi_index_url,
                pypi_trusted_host=pypi_trusted_host,
                mirror_config=mirror_config,
                timeout=max(timeout, 600),
                docker_run_args=docker_run_args,
            )
        )

    ok = all(check.ok for check in checks)
    return {
        "ok": ok,
        "mode": mode,
        "checks": [asdict(check) for check in checks],
        "failed_checks": [check.name for check in checks if not check.ok],
        "repair": _combined_repair(checks),
        "docker_run_args": list(docker_run_args or []),
        "prebuilt_docker_pull_default_opt_in": True,
    }


def _downgrade_registry_probe_if_docker_pull_succeeded(
    checks: list[CheckResult],
    registry_check_name: str,
) -> None:
    registry_index = next(
        (
            index
            for index, check in enumerate(checks)
            if check.name == registry_check_name
        ),
        None,
    )
    if registry_index is None or checks[registry_index].ok:
        return
    pull_successes = [
        check.name
        for check in checks
        if check.name.startswith("docker_pull:") and check.ok
    ]
    if not pull_successes:
        return
    original = checks[registry_index]
    checks[registry_index] = CheckResult(
        name=original.name,
        ok=True,
        detail=(
            original.detail
            + "; registry /v2 host probe failed, but Docker pull succeeded for "
            + ", ".join(pull_successes[:5])
            + ", so treating this as a host-DNS diagnostic instead of blocking Harbor."
        ),
        repair="",
        elapsed_seconds=original.elapsed_seconds,
        metadata={
            **original.metadata,
            "downgraded_to_diagnostic": True,
            "docker_pull_successes": pull_successes[:10],
        },
    )


def _downgrade_registry_probe_if_prebuilt_manifest_succeeded(
    checks: list[CheckResult],
    registry_check_name: str,
) -> None:
    registry_index = next(
        (
            index
            for index, check in enumerate(checks)
            if check.name == registry_check_name
        ),
        None,
    )
    if registry_index is None or checks[registry_index].ok:
        return
    manifest_successes = [
        check.name
        for check in checks
        if check.name.startswith("prebuilt_docker_manifest:") and check.ok
    ]
    if not manifest_successes:
        return
    original = checks[registry_index]
    checks[registry_index] = CheckResult(
        name=original.name,
        ok=True,
        detail=(
            original.detail
            + "; registry /v2 host probe failed, but prebuilt image manifest "
            "inspection succeeded for "
            + ", ".join(manifest_successes[:5])
            + ", so treating this as a host-DNS diagnostic instead of blocking Harbor."
        ),
        repair="",
        elapsed_seconds=original.elapsed_seconds,
        metadata={
            **original.metadata,
            "downgraded_to_diagnostic": True,
            "prebuilt_manifest_successes": manifest_successes[:10],
        },
    )


def _downgrade_prebuilt_pull_failures_if_manifest_succeeded(
    checks: list[CheckResult],
) -> None:
    manifest_successes = {
        check.metadata.get("source_image")
        for check in checks
        if check.name.startswith("prebuilt_docker_manifest:") and check.ok
    }
    for index, check in enumerate(checks):
        if check.ok or not check.name.startswith("prebuilt_docker_pull:"):
            continue
        source_image = check.metadata.get("source_image")
        if source_image not in manifest_successes:
            continue
        checks[index] = CheckResult(
            name=check.name,
            ok=True,
            detail=(
                check.detail
                + "; prebuilt image manifest is reachable, so this cache-warm "
                "pull failure is treated as non-blocking. Harbor may still pull "
                "or reuse cached layers when that task runs."
            ),
            repair="",
            elapsed_seconds=check.elapsed_seconds,
            metadata={
                **check.metadata,
                "downgraded_to_cache_warning": True,
            },
        )


def check_docker_daemon(*, timeout: int) -> CheckResult:
    start = time.monotonic()
    completed = _run(["docker", "info"], timeout=timeout)
    ok = completed.returncode == 0
    return CheckResult(
        name="docker_daemon",
        ok=ok,
        detail=_detail(completed),
        repair="Start Docker and verify `docker info` succeeds.",
        elapsed_seconds=time.monotonic() - start,
    )


def check_docker_mirror(
    *,
    expected_mirror: str,
    timeout: int,
    strict: bool = False,
) -> CheckResult:
    start = time.monotonic()
    completed = _run(
        ["docker", "info", "--format", "{{json .RegistryConfig.Mirrors}}"],
        timeout=timeout,
    )
    mirrors: list[str] = []
    ok = completed.returncode == 0
    detail = _detail(completed)
    matches_expected = False
    if completed.returncode == 0:
        try:
            mirrors = json.loads((completed.stdout or "").strip() or "[]")
        except json.JSONDecodeError:
            mirrors = []
        normalized = {mirror.rstrip("/") for mirror in mirrors}
        matches_expected = expected_mirror.rstrip("/") in normalized
        ok = matches_expected if strict else True
        detail = (
            f"configured Docker daemon mirrors: {mirrors}; "
            f"explicit image mirror: {expected_mirror}; "
            f"matches_expected={matches_expected}"
        )
    return CheckResult(
        name="docker_registry_mirror",
        ok=ok,
        detail=detail,
        repair=(
            "Set /etc/docker/daemon.json registry-mirrors to "
            f"{expected_mirror!r} and restart Docker."
            if strict
            else ""
        ),
        elapsed_seconds=time.monotonic() - start,
        metadata={
            "mirrors": mirrors,
            "expected": expected_mirror,
            "matches_expected": matches_expected,
            "strict": strict,
        },
    )


def check_docker_pull(
    image: str,
    *,
    timeout: int,
    source_image: str | None = None,
    name: str = "docker_pull",
) -> CheckResult:
    start = time.monotonic()
    completed = _run(["docker", "pull", image], timeout=timeout)
    return CheckResult(
        name=name,
        ok=completed.returncode == 0,
        detail=_detail(completed),
        repair=(
            "Fix Docker registry mirror/DNS before launching Harbor. "
            f"`docker pull {image}` must succeed."
        ),
        elapsed_seconds=time.monotonic() - start,
        metadata={"image": image, "source_image": source_image or image},
    )


def check_docker_manifest(
    image: str,
    *,
    timeout: int,
    source_image: str | None = None,
    name: str = "docker_manifest",
) -> CheckResult:
    start = time.monotonic()
    completed = _run(["docker", "manifest", "inspect", image], timeout=timeout)
    return CheckResult(
        name=name,
        ok=completed.returncode == 0,
        detail=_detail(completed),
        repair=(
            "Fix the prebuilt Docker image mirror before launching Harbor. "
            f"`docker manifest inspect {image}` must succeed."
        ),
        elapsed_seconds=time.monotonic() - start,
        metadata={"image": image, "source_image": source_image or image},
    )


def check_http_head(name: str, url: str, *, timeout: int) -> CheckResult:
    start = time.monotonic()
    method = "GET" if is_wechat_article_url(url) else "HEAD"
    try:
        request = Request(url, method=method, headers=request_headers_for_url(url))
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 0)
            body = response.read(4096) if method == "GET" else b""
        ok = 200 <= int(status) < 400
        if method == "GET" and _looks_like_wechat_verification_page(body):
            ok = False
            detail = f"GET {url} -> HTTP {status}; WeChat environment verification page"
        else:
            detail = f"{method} {url} -> HTTP {status}"
    except Exception as exc:  # noqa: BLE001 - exact network exception type varies by environment.
        ok = False
        detail = f"{method} {url} failed: {exc}"
    return CheckResult(
        name=name,
        ok=ok,
        detail=detail,
        repair="Use a reachable Debian/Ubuntu mirror or fix DNS/proxy routing.",
        elapsed_seconds=time.monotonic() - start,
        metadata={"url": url},
    )


def _looks_like_wechat_verification_page(body: bytes) -> bool:
    text = body[:4096].decode("utf-8", "ignore").lower()
    return (
        "\u73af\u5883\u5f02\u5e38" in text
        or "\u5b8c\u6210\u9a8c\u8bc1\u540e\u5373\u53ef\u7ee7\u7eed\u8bbf\u95ee" in text
    )


def check_registry_v2(name: str, url: str, *, timeout: int) -> CheckResult:
    """Check Registry v2 reachability.

    Registry endpoints commonly return HTTP 401 for unauthenticated /v2/ probes.
    That is a healthy reachability signal; `docker pull` remains the stronger
    end-to-end check.
    """

    start = time.monotonic()
    try:
        request = Request(url, method="GET", headers=request_headers_for_url(url))
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 0)
        ok = int(status) in {200, 401}
        detail = f"GET {url} -> HTTP {status}"
    except Exception as exc:  # noqa: BLE001 - urllib raises HTTPError for healthy 401s.
        status = getattr(exc, "code", None)
        ok = int(status) in {200, 401} if status is not None else False
        detail = f"GET {url} -> HTTP {status}" if status is not None else f"GET {url} failed: {exc}"
    return CheckResult(
        name=name,
        ok=ok,
        detail=detail,
        repair="Use a reachable Docker registry mirror or fix Docker DNS/proxy routing.",
        elapsed_seconds=time.monotonic() - start,
        metadata={"url": url},
    )


def check_debian_container_download(
    *,
    debian_mirror: str,
    debian_security_mirror: str,
    mirror_config: AptMirrorConfig | None = None,
    timeout: int,
    docker_run_args: list[str] | None = None,
) -> CheckResult:
    start = time.monotonic()
    script = (
        "set -eu; "
        "sed -i "
        f"s#http://deb.debian.org/debian-security#{debian_security_mirror.rstrip('/')}#g "
        "/etc/apt/sources.list.d/debian.sources; "
        "sed -i "
        f"s#http://deb.debian.org/debian#{debian_mirror.rstrip('/')}#g "
        "/etc/apt/sources.list.d/debian.sources; "
        "printf '%s\\n' 'Acquire::Retries \"5\";' "
        "'Acquire::http::Timeout \"30\";' 'Acquire::https::Timeout \"30\";' "
        "> /etc/apt/apt.conf.d/99hl-network; "
        "apt-get clean; apt-get update; apt-get install -y --download-only gnucobol3"
    )
    completed = _run(
        [
            "docker",
            "run",
            *(docker_run_args or ["--rm"]),
            effective_docker_image_reference(
                "python:3.13-slim-bookworm",
                mirror_config,
            ),
            "sh",
            "-lc",
            script,
        ],
        timeout=timeout,
    )
    return CheckResult(
        name="debian_container_download",
        ok=completed.returncode == 0,
        detail=_detail(completed),
        repair=(
            "Container APT downloads are failing even with mirror rewriting. "
            "Fix host/container DNS or choose another mirror before scoring campaigns."
        ),
        elapsed_seconds=time.monotonic() - start,
    )


def check_ubuntu_container_download(
    *,
    ubuntu_mirror: str,
    mirror_config: AptMirrorConfig | None = None,
    timeout: int,
    docker_run_args: list[str] | None = None,
) -> CheckResult:
    start = time.monotonic()
    mirror = ubuntu_mirror.rstrip("/")
    replacements = [
        "http://archive.ubuntu.com/ubuntu",
        "https://archive.ubuntu.com/ubuntu",
        "http://security.ubuntu.com/ubuntu",
        "https://security.ubuntu.com/ubuntu",
    ]
    sed_args = " ".join(
        f"-e {shlex.quote(f's#{source}#{mirror}#g')}" for source in replacements
    )
    script = (
        "set -eu; "
        "mkdir -p /etc/apt/apt.conf.d; "
        "for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list "
        "/etc/apt/sources.list.d/*.sources; do "
        "[ -f \"$file\" ] || continue; "
        f"sed -i {sed_args} \"$file\"; "
        "done; "
        "printf '%s\\n' 'Acquire::Retries \"5\";' "
        "'Acquire::http::Timeout \"30\";' 'Acquire::https::Timeout \"30\";' "
        "> /etc/apt/apt.conf.d/99hl-network; "
        "apt-get clean; apt-get update; "
        "apt-get install -y --download-only ca-certificates"
    )
    completed = _run(
        [
            "docker",
            "run",
            *(docker_run_args or ["--rm"]),
            effective_docker_image_reference(
                "ubuntu:24.04",
                mirror_config,
            ),
            "sh",
            "-lc",
            script,
        ],
        timeout=timeout,
    )
    return CheckResult(
        name="ubuntu_container_download",
        ok=completed.returncode == 0,
        detail=_detail(completed),
        repair=(
            "Ubuntu container APT bootstrap is failing. Use an HTTP apt mirror "
            "or install ca-certificates before switching apt sources to HTTPS."
        ),
        elapsed_seconds=time.monotonic() - start,
        metadata={"ubuntu_mirror": ubuntu_mirror},
    )


def check_python_container_pip_download(
    *,
    pypi_index_url: str,
    pypi_trusted_host: str,
    mirror_config: AptMirrorConfig | None = None,
    timeout: int,
    docker_run_args: list[str] | None = None,
) -> CheckResult:
    start = time.monotonic()
    command = [
        "python",
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--dry-run",
        "--index-url",
        pypi_index_url,
        "--timeout",
        "30",
        "--retries",
        "5",
    ]
    if pypi_trusted_host:
        command.extend(["--trusted-host", pypi_trusted_host])
    command.append("requests==2.32.4")
    completed = _run(
        [
            "docker",
            "run",
            *(docker_run_args or ["--rm"]),
            effective_docker_image_reference(
                "python:3.13-slim-bookworm",
                mirror_config,
            ),
            *command,
        ],
        timeout=timeout,
    )
    return CheckResult(
        name="python_container_pip_download",
        ok=completed.returncode == 0,
        detail=_detail(completed),
        repair=(
            "Container pip downloads are failing. Fix CA/proxy routing or configure "
            "a reachable pypi_index_url before scoring campaigns."
        ),
        elapsed_seconds=time.monotonic() - start,
        metadata={"pypi_index_url": pypi_index_url},
    )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout=stdout,
            stderr=f"timed out after {timeout}s\n{stderr}",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, stdout="", stderr=str(exc))
    except URLError as exc:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=str(exc))


def _detail(completed: subprocess.CompletedProcess[str], *, limit: int = 2000) -> str:
    text = "\n".join(
        part for part in (_as_text(completed.stdout), _as_text(completed.stderr)) if part
    )
    text = text.strip()
    if len(text) > limit:
        text = text[-limit:]
    return text or f"exit {completed.returncode}"


def _combined_repair(checks: list[CheckResult]) -> str:
    repairs = []
    for check in checks:
        if not check.ok and check.repair and check.repair not in repairs:
            repairs.append(check.repair)
    return " ".join(repairs)


def _print_human(report: dict[str, Any]) -> None:
    status = "ok" if report["ok"] else "failed"
    print(f"Network preflight: {status}")
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"- {mark} {check['name']}: {check['detail']}")
        if not check["ok"] and check.get("repair"):
            print(f"  repair: {check['repair']}")


def mirror_image_reference(image: str, docker_hub_mirror: str) -> str:
    if _is_explicit_registry(image):
        return image
    mirror = docker_hub_mirror.rstrip("/")
    if "/" in image:
        return f"{mirror}/{image}"
    return f"{mirror}/library/{image}"


def _is_explicit_registry(image: str) -> bool:
    if "/" not in image:
        return False
    first = image.split("/", 1)[0]
    return "." in first or ":" in first or first == "localhost"


def _configured_list(
    cli_values: list[str] | None,
    config_value: Any,
    default: list[str],
) -> list[str]:
    if cli_values is not None:
        return [str(item) for item in cli_values if str(item).strip()]
    if isinstance(config_value, list):
        return [str(item) for item in config_value if str(item).strip()]
    if isinstance(config_value, str):
        return [item.strip() for item in config_value.split(",") if item.strip()]
    return list(default)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _load_network_config(path: Path) -> dict[str, Any]:
    data = _load_trials_config(path)
    network = data.get("network") if isinstance(data, dict) else {}
    return network if isinstance(network, dict) else {}


def _load_trials_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _image_overrides_from_config(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("docker_image_overrides")
    if not isinstance(raw, dict):
        return {}
    return {
        str(source): str(target)
        for source, target in raw.items()
        if str(source).strip() and str(target).strip()
    }


def _url_rewrites_from_config(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("download_url_rewrites")
    if not isinstance(raw, dict):
        return dict(DEFAULT_DOWNLOAD_URL_REWRITES)
    return {
        str(source): str(target)
        for source, target in raw.items()
        if str(source).strip() and str(target).strip()
    }


if __name__ == "__main__":
    raise SystemExit(main())
