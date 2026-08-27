"""Network-hardened Harbor Docker environment wrappers.

The wrapper edits only Harbor's per-trial copied build context. It must never
rewrite the source TerminalBench task directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.paths import EnvironmentPaths

# Public, vendor-neutral network defaults. Private mirrors, proxy roots, and
# image overrides can be supplied through gitignored config/local.yaml.
DEFAULT_DEBIAN_MIRROR = "https://deb.debian.org/debian"
DEFAULT_DEBIAN_SECURITY_MIRROR = "https://security.debian.org/debian-security"
DEFAULT_UBUNTU_MIRROR = "https://archive.ubuntu.com/ubuntu"
DEFAULT_DOCKER_HUB_MIRROR = ""
DEFAULT_PREBUILT_DOCKER_HUB_MIRROR = ""
DEFAULT_PYPI_INDEX_URL = "https://pypi.org/simple/"
DEFAULT_PYPI_TRUSTED_HOST = ""
DEFAULT_ALPINE_MIRROR = "https://dl-cdn.alpinelinux.org/alpine"
DEFAULT_DOWNLOAD_URL_REWRITES: dict[str, str] = {}
DEFAULT_DOCKER_MEMORY = "4g"
DEFAULT_DOCKER_MEMORY_SWAP = "4g"
DEFAULT_DOCKER_CPUS = "2"
DEFAULT_DOCKER_PIDS_LIMIT = 1024
DEFAULT_DOCKER_LOG_MAX_SIZE = "20m"
DEFAULT_DOCKER_LOG_MAX_FILE = "3"
DEFAULT_DOCKER_LABELS = {
    "com.harness-evolver.managed": "true",
    "com.harness-evolver.cleanup": "task",
    "com.harness-evolver.role": "benchmark",
}
LOCAL_IMAGE_INSPECT_TIMEOUT_SECONDS = 10
PREBUILT_WARMUP_FAILURE_RECEIPT = "hl-prebuilt-warmup-failure.json"
PREBUILT_WARMUP_FAILURE_RECEIPT_SCHEMA = (
    "harness-evolver.prebuilt-warmup-failure.v1"
)


@dataclass(frozen=True)
class AptMirrorConfig:
    """Network mirror settings injected into copied Docker build contexts."""

    debian_mirror: str = DEFAULT_DEBIAN_MIRROR
    debian_security_mirror: str = DEFAULT_DEBIAN_SECURITY_MIRROR
    ubuntu_mirror: str = DEFAULT_UBUNTU_MIRROR
    docker_hub_mirror: str = DEFAULT_DOCKER_HUB_MIRROR
    prebuilt_docker_hub_mirror: str = DEFAULT_PREBUILT_DOCKER_HUB_MIRROR
    docker_image_overrides: dict[str, str] = field(default_factory=dict)
    download_url_rewrites: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DOWNLOAD_URL_REWRITES)
    )
    pypi_index_url: str = DEFAULT_PYPI_INDEX_URL
    pypi_trusted_host: str = DEFAULT_PYPI_TRUSTED_HOST
    apt_retries: int = 5
    apt_timeout_seconds: int = 30
    pip_retries: int = 5
    pip_timeout_seconds: int = 30
    prebuilt_docker_pull_timeout_seconds: int = 600
    bootstrap_ca_certificates: bool = True
    download_retry_wrapper: bool = True
    # Optional local override for a private CA. Disabled by default so a
    # user-provided trust bundle is never copied into a trial build context.
    inject_host_ca_into_build: bool = False
    host_ca_cert_bundle: str = "/etc/ssl/certs/ca-certificates.crt"


@dataclass(frozen=True)
class DockerResourceConfig:
    """Local Docker resource safety policy for Harbor task containers."""

    enabled: bool = True
    memory: str = DEFAULT_DOCKER_MEMORY
    memory_swap: str = DEFAULT_DOCKER_MEMORY_SWAP
    cpus: str = DEFAULT_DOCKER_CPUS
    pids_limit: int = DEFAULT_DOCKER_PIDS_LIMIT
    labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DOCKER_LABELS))
    log_max_size: str = DEFAULT_DOCKER_LOG_MAX_SIZE
    log_max_file: str = DEFAULT_DOCKER_LOG_MAX_FILE


class AptMirrorDockerEnvironment(DockerEnvironment):
    """DockerEnvironment that patches network sources in a copied build context.

    Harbor passes the original TerminalBench task environment directory to the
    environment constructor. To preserve benchmark integrity, this subclass first
    copies that directory under the trial directory, patches only the copy, and
    then lets Harbor build from that copied context.
    """

    def __init__(
        self,
        environment_dir: Path,
        *args: Any,
        apt_mirror_enabled: str | bool = True,
        debian_mirror: str = DEFAULT_DEBIAN_MIRROR,
        debian_security_mirror: str = DEFAULT_DEBIAN_SECURITY_MIRROR,
        ubuntu_mirror: str = DEFAULT_UBUNTU_MIRROR,
        docker_hub_mirror: str = DEFAULT_DOCKER_HUB_MIRROR,
        prebuilt_docker_hub_mirror: str = DEFAULT_PREBUILT_DOCKER_HUB_MIRROR,
        docker_image_overrides: str | dict[str, str] | None = None,
        download_url_rewrites: str | dict[str, str] | None = None,
        pypi_index_url: str = DEFAULT_PYPI_INDEX_URL,
        pypi_trusted_host: str = DEFAULT_PYPI_TRUSTED_HOST,
        apt_retries: str | int = 5,
        apt_timeout_seconds: str | int = 30,
        pip_retries: str | int = 5,
        pip_timeout_seconds: str | int = 30,
        prebuilt_docker_pull_timeout_seconds: str | int = 600,
        bootstrap_ca_certificates: str | bool = True,
        download_retry_wrapper: str | bool = True,
        inject_host_ca_into_build: str | bool = False,
        host_ca_cert_bundle: str = "/etc/ssl/certs/ca-certificates.crt",
        docker_resource_enabled: str | bool = True,
        docker_memory: str = DEFAULT_DOCKER_MEMORY,
        docker_memory_swap: str = DEFAULT_DOCKER_MEMORY_SWAP,
        docker_cpus: str | int = DEFAULT_DOCKER_CPUS,
        docker_pids_limit: str | int = DEFAULT_DOCKER_PIDS_LIMIT,
        docker_labels: str | dict[str, str] | None = None,
        docker_log_max_size: str = DEFAULT_DOCKER_LOG_MAX_SIZE,
        docker_log_max_file: str | int = DEFAULT_DOCKER_LOG_MAX_FILE,
        **kwargs: Any,
    ) -> None:
        trial_paths = kwargs.get("trial_paths")
        enabled = _truthy(apt_mirror_enabled)
        mirror_config = AptMirrorConfig(
            debian_mirror=str(debian_mirror),
            debian_security_mirror=str(debian_security_mirror),
            ubuntu_mirror=str(ubuntu_mirror),
            docker_hub_mirror=str(docker_hub_mirror),
            prebuilt_docker_hub_mirror=str(prebuilt_docker_hub_mirror),
            docker_image_overrides=_parse_image_overrides(docker_image_overrides),
            download_url_rewrites=_parse_rewrite_map(
                download_url_rewrites,
                default=DEFAULT_DOWNLOAD_URL_REWRITES,
            ),
            pypi_index_url=str(pypi_index_url),
            pypi_trusted_host=str(pypi_trusted_host),
            apt_retries=_positive_int(apt_retries, default=5),
            apt_timeout_seconds=_positive_int(apt_timeout_seconds, default=30),
            pip_retries=_positive_int(pip_retries, default=5),
            pip_timeout_seconds=_positive_int(pip_timeout_seconds, default=30),
            prebuilt_docker_pull_timeout_seconds=_positive_int(
                prebuilt_docker_pull_timeout_seconds,
                default=600,
            ),
            bootstrap_ca_certificates=_truthy(bootstrap_ca_certificates),
            download_retry_wrapper=_truthy(download_retry_wrapper),
            inject_host_ca_into_build=_truthy(inject_host_ca_into_build),
            host_ca_cert_bundle=str(
                host_ca_cert_bundle or "/etc/ssl/certs/ca-certificates.crt"
            ),
        )
        self._hl_mirror_config = mirror_config
        self._hl_resource_config = DockerResourceConfig(
            enabled=_truthy(docker_resource_enabled),
            memory=str(docker_memory or DEFAULT_DOCKER_MEMORY),
            memory_swap=str(docker_memory_swap or docker_memory or DEFAULT_DOCKER_MEMORY_SWAP),
            cpus=str(docker_cpus or DEFAULT_DOCKER_CPUS),
            pids_limit=_positive_int(docker_pids_limit, default=DEFAULT_DOCKER_PIDS_LIMIT),
            labels=_parse_docker_labels(docker_labels),
            log_max_size=str(docker_log_max_size or DEFAULT_DOCKER_LOG_MAX_SIZE),
            log_max_file=str(docker_log_max_file or DEFAULT_DOCKER_LOG_MAX_FILE),
        )
        self._hl_resource_compose_path: Path | None = None
        self._hl_verifier_runtime_setup_enabled = enabled
        self._hl_verifier_runtime_setup_done = False
        self._hl_original_prebuilt_image = ""
        self._hl_effective_prebuilt_image = ""
        self._hl_prebuilt_mirror_config = replace(
            mirror_config,
            docker_hub_mirror=mirror_config.prebuilt_docker_hub_mirror,
        )
        if enabled:
            task_env_config = kwargs.get("task_env_config")
            docker_image = getattr(task_env_config, "docker_image", None)
            if task_env_config is not None and docker_image:
                effective_image = effective_docker_image_reference(
                    str(docker_image),
                    self._hl_prebuilt_mirror_config,
                )
                if effective_image != docker_image:
                    self._hl_original_prebuilt_image = str(docker_image)
                    self._hl_effective_prebuilt_image = effective_image
                    if hasattr(task_env_config, "model_copy"):
                        kwargs["task_env_config"] = task_env_config.model_copy(
                            update={"docker_image": effective_image}
                        )
                    else:
                        task_env_config.docker_image = effective_image
        prepared_dir = Path(environment_dir)
        if enabled and trial_paths is not None:
            prepared_dir = prepare_apt_mirror_environment(
                Path(environment_dir),
                Path(trial_paths.trial_dir) / "hl_patched_environment",
                mirror_config,
            )
        super().__init__(prepared_dir, *args, **kwargs)

    @property
    def _docker_compose_paths(self) -> list[Path]:
        paths = list(super()._docker_compose_paths)
        if self._hl_resource_compose_path is None:
            return paths
        if self._hl_resource_compose_path in paths:
            return paths

        # Harbor 0.22 replaced the legacy no-network compose override with
        # egress-control overlays. Keep our resource policy after task-authored
        # overrides but before whichever network overlay the installed Harbor
        # exposes, without requiring a version-specific private sentinel.
        network_overlays = {
            overlay
            for overlay in (
                getattr(self, "_DOCKER_COMPOSE_NO_NETWORK_PATH", None),
                getattr(self, "_DOCKER_COMPOSE_EGRESS_CONTROL_PATH", None),
                getattr(self, "_egress_control_services_compose_path", None),
            )
            if overlay is not None
        }
        insert_at = next(
            (
                index
                for index, compose_path in enumerate(paths)
                if compose_path in network_overlays
            ),
            len(paths),
        )
        return [
            *paths[:insert_at],
            self._hl_resource_compose_path,
            *paths[insert_at:],
        ]

    async def start(self, force_build: bool) -> None:
        if self._hl_resource_config.enabled:
            self._hl_resource_compose_path = write_docker_resource_compose_file(
                self.trial_paths.trial_dir / "hl-docker-resources.json",
                service_names=compose_service_names(self.environment_dir),
                config=self._hl_resource_config,
                session_id=str(self.session_id),
                environment_name=str(self.environment_name),
            )
        receipt_path = (
            Path(self.trial_paths.trial_dir) / PREBUILT_WARMUP_FAILURE_RECEIPT
        )
        receipt_path.unlink(missing_ok=True)
        try:
            await self._warm_prebuilt_image_cache_if_needed(force_build)
        except RuntimeError as exc:
            try:
                _write_prebuilt_warmup_failure_receipt(
                    receipt_path,
                    kind=str(getattr(exc, "_hl_prebuilt_warmup_failure_kind", "")),
                    deterministic_access_failure=(
                        getattr(
                            exc,
                            "_hl_prebuilt_warmup_deterministic_access_failure",
                            False,
                        )
                        is True
                    ),
                )
            except OSError as receipt_exc:
                self.logger.warning(
                    "Could not persist prebuilt warmup failure provenance: %s",
                    receipt_exc,
                )
            raise
        await super().start(force_build)

    async def _warm_prebuilt_image_cache_if_needed(self, force_build: bool) -> None:
        if force_build:
            return
        image = self._hl_effective_prebuilt_image or str(self.task_env_config.docker_image or "")
        image = image.strip()
        if not image:
            return
        digest_pinned = bool(_normalized_registry_digest(image))
        digest_available = False
        if digest_pinned:
            digest_available = await asyncio.to_thread(
                _docker_image_has_registry_provenance,
                image,
                self._hl_original_prebuilt_image,
            )
        if digest_available:
            self.logger.info(
                "Registry-provenanced prebuilt image already available locally; "
                "skipping Docker pull warmup: %s",
                image,
            )
            return
        fallback = await asyncio.to_thread(
            _tag_available_prebuilt_fallback,
            image,
            self._hl_original_prebuilt_image,
        )
        if fallback:
            self.logger.info(
                "Prebuilt image warmup satisfied from local fallback: %s -> %s",
                fallback,
                image,
            )
            return
        timeout = max(1, self._hl_mirror_config.prebuilt_docker_pull_timeout_seconds)
        argv = ["docker", "pull", image]
        self.logger.info(
            "Warming prebuilt image cache before docker compose up: %s (timeout=%ss)",
            image,
            timeout,
        )
        try:
            await asyncio.to_thread(
                subprocess.run,
                argv,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            detail = _subprocess_failure_detail(exc.stdout, exc.stderr)
            raise _prebuilt_warmup_error(
                "Prebuilt Docker image cache warmup timed out after "
                f"{timeout} seconds for image {image}. Run `docker pull {image}` "
                "or `python scripts/network_preflight.py --quick` to diagnose Docker "
                f"registry access. {detail}",
                kind="prebuilt_image_cache_warmup_timeout",
                deterministic_access_failure=False,
            ) from exc
        except subprocess.CalledProcessError as exc:
            if self._hl_original_prebuilt_image and _prebuilt_pull_failure_can_try_original(
                exc.stdout,
                exc.stderr,
            ):
                original_image = self._hl_original_prebuilt_image
                self.logger.warning(
                    "Prebuilt image mirror pull failed for %s; trying original "
                    "image %s before failing environment start",
                    image,
                    original_image,
                )
                try:
                    await asyncio.to_thread(
                        _pull_and_tag_original_prebuilt_image,
                        original_image,
                        image,
                        timeout,
                    )
                    self.logger.info(
                        "Prebuilt image warmup satisfied from original image "
                        "fallback: %s -> %s",
                        original_image,
                        image,
                    )
                    return
                except RuntimeError as fallback_exc:
                    detail = _subprocess_failure_detail(exc.stdout, exc.stderr)
                    raise _prebuilt_warmup_error(
                        "Prebuilt Docker image cache warmup failed for image "
                        f"{image} with return code {exc.returncode}; original image "
                        f"fallback {original_image} also failed. Run `docker pull "
                        f"{image}` or `docker pull {original_image}` or `python "
                        "scripts/network_preflight.py --quick` to diagnose Docker "
                        f"registry access. Mirror error: {detail} "
                        f"Fallback error: {fallback_exc}",
                        kind="prebuilt_image_cache_warmup_failure",
                        deterministic_access_failure=(
                            _prebuilt_pull_failure_can_try_original(
                                "",
                                str(fallback_exc),
                            )
                        ),
                    ) from fallback_exc
            detail = _subprocess_failure_detail(exc.stdout, exc.stderr)
            raise _prebuilt_warmup_error(
                "Prebuilt Docker image cache warmup failed for image "
                f"{image} with return code {exc.returncode}. Run `docker pull {image}` "
                "or `python scripts/network_preflight.py --quick` to diagnose Docker "
                f"registry access. {detail}",
                kind="prebuilt_image_cache_warmup_failure",
                deterministic_access_failure=(
                    _prebuilt_pull_failure_can_try_original(exc.stdout, exc.stderr)
                ),
            ) from exc

    async def stop(self, delete: bool) -> None:
        """Stop Harbor containers without deleting Docker volumes.

        The base Harbor DockerEnvironment uses ``docker compose down --volumes``
        for delete=True. Local HL runs must not remove volumes implicitly, so we
        remove only containers/networks/orphans and rely on explicit cleanup
        scripts for images/build cache.
        """

        try:
            await self.prepare_logs_for_host()

            if self._keep_containers and delete:
                self.logger.warning(
                    "Both `keep_containers` and `--delete` option are set. "
                    "keep_containers takes precedence."
                )
            if self._keep_containers:
                try:
                    await self._run_docker_compose_command(["stop"])
                except Exception as exc:
                    self.logger.warning(f"Docker compose stop failed: {exc}")
                return
            try:
                if delete:
                    await self._run_docker_compose_command(
                        ["down", "--remove-orphans"]
                    )
                else:
                    await self._run_docker_compose_command(["down"])
            except Exception as exc:
                self.logger.warning(f"Docker compose down failed: {exc}")
            if delete and self._hl_resource_config.enabled:
                await self._cleanup_stopped_hl_containers()
        finally:
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
            self._cleanup_env_compose_file()
            self._cleanup_egress_control_services_compose_file()

    async def _cleanup_stopped_hl_containers(self) -> None:
        label = self._hl_resource_config.labels.get("com.harness-evolver.cleanup", "task")
        argv = [
            "docker",
            "container",
            "prune",
            "--force",
            "--filter",
            "label=com.harness-evolver.managed=true",
            "--filter",
            f"label=com.harness-evolver.cleanup={label}",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await process.communicate()
        except Exception as exc:
            self.logger.warning(f"HL stopped-container cleanup failed: {exc}")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        if self._should_prepare_verifier_runtime(command, env):
            await self._prepare_verifier_runtime(timeout_sec)
        return await super().exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    def _should_prepare_verifier_runtime(
        self,
        command: str,
        env: dict[str, str] | None,
    ) -> bool:
        if (
            not self._hl_verifier_runtime_setup_enabled
            or self._hl_verifier_runtime_setup_done
            or self._is_windows_container
        ):
            return False
        env = env or {}
        verifier_env_keys = {
            "HL_VERIFIER_NETWORK_PREPARE",
            "PIP_INDEX_URL",
            "UV_INDEX_URL",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
        }
        if verifier_env_keys & set(env):
            return True
        lowered = command.lower()
        tests_dir = str(EnvironmentPaths.for_os(self.os).tests_dir).lower()
        return tests_dir in lowered or "/tests/" in lowered

    async def _prepare_verifier_runtime(self, verifier_timeout_sec: int | None = None) -> None:
        self._hl_verifier_runtime_setup_done = True
        timeout = verifier_runtime_prepare_timeout_seconds(
            self._hl_mirror_config,
            verifier_timeout_sec=verifier_timeout_sec,
        )
        if timeout <= 0:
            self.logger.warning(
                "Skipping verifier runtime network preparation because the "
                "verifier command timeout window is too small; continuing "
                "with official verifier command."
            )
            return
        try:
            result = await super().exec(
                command=verifier_runtime_network_prepare_command(self._hl_mirror_config),
                timeout_sec=timeout,
                user="root",
            )
        except Exception as exc:
            self.logger.warning(
                "Verifier runtime network preparation failed before verifier run; "
                "continuing with official verifier command: %s",
                exc,
            )
            return
        if result.return_code != 0:
            self.logger.warning(
                "Verifier runtime network preparation exited with code %s: %s",
                result.return_code,
                (result.stdout or result.stderr or "")[-1000:],
            )


def prepare_apt_mirror_environment(
    source_dir: Path,
    destination_dir: Path,
    config: AptMirrorConfig | None = None,
) -> Path:
    """Copy an environment directory and patch network settings in the copy."""

    config = config or AptMirrorConfig()
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir, symlinks=True)

    # Copy the host CA bundle into the build context so the Dockerfile can COPY
    # and trust it before any apk/apt/pip/npm step. Only when the bundle exists;
    # a missing bundle must not emit a COPY that would break the build.
    host_ca_injected = False
    if _host_ca_bundle_available(config):
        try:
            shutil.copyfile(
                config.host_ca_cert_bundle,
                destination_dir / HL_HOST_CA_BUILD_CONTEXT_NAME,
            )
            host_ca_injected = True
        except OSError:
            host_ca_injected = False
    effective_config = config
    if config.inject_host_ca_into_build and not host_ca_injected:
        # Disable build CA injection for this context so patch_dockerfile does
        # not reference a file that was not copied.
        effective_config = replace(config, inject_host_ca_into_build=False)

    patched_files: list[str] = []
    for dockerfile in _dockerfile_candidates(destination_dir):
        original = dockerfile.read_text(errors="replace")
        patched, changed = patch_dockerfile_for_apt_mirrors(original, effective_config)
        if changed:
            dockerfile.write_text(patched)
            patched_files.append(str(dockerfile.relative_to(destination_dir)))
    for compose_file in _compose_candidates(destination_dir):
        original = compose_file.read_text(errors="replace")
        patched, changed = patch_compose_for_docker_mirrors(original, effective_config)
        if changed:
            compose_file.write_text(patched)
            patched_files.append(str(compose_file.relative_to(destination_dir)))

    marker = {
        "source_dir": str(source_dir),
        "patched_files": patched_files,
        "debian_mirror": config.debian_mirror,
        "debian_security_mirror": config.debian_security_mirror,
        "ubuntu_mirror": config.ubuntu_mirror,
        "docker_hub_mirror": config.docker_hub_mirror,
        "prebuilt_docker_hub_mirror": config.prebuilt_docker_hub_mirror,
        "docker_image_overrides": config.docker_image_overrides,
        "download_url_rewrites": config.download_url_rewrites,
        "pypi_index_url": config.pypi_index_url,
        "pypi_trusted_host": config.pypi_trusted_host,
        "apt_retries": config.apt_retries,
        "apt_timeout_seconds": config.apt_timeout_seconds,
        "pip_retries": config.pip_retries,
        "pip_timeout_seconds": config.pip_timeout_seconds,
        "prebuilt_docker_pull_timeout_seconds": config.prebuilt_docker_pull_timeout_seconds,
        "bootstrap_ca_certificates": config.bootstrap_ca_certificates,
        "download_retry_wrapper": config.download_retry_wrapper,
    }
    (destination_dir / ".hl_apt_mirror.json").write_text(json.dumps(marker, indent=2))
    return destination_dir


def write_docker_resource_compose_file(
    path: Path,
    *,
    service_names: list[str],
    config: DockerResourceConfig,
    session_id: str,
    environment_name: str,
) -> Path:
    """Write a compose override that caps and labels every known service."""

    labels = {
        **config.labels,
        "com.harness-evolver.session": session_id,
        "com.harness-evolver.environment": environment_name,
    }
    services = {}
    for service_name in service_names or ["main"]:
        services[service_name] = {
            "cpus": config.cpus,
            "mem_limit": config.memory,
            "memswap_limit": config.memory_swap,
            "pids_limit": config.pids_limit,
            "labels": labels,
            "logging": {
                "driver": "json-file",
                "options": {
                    "max-size": config.log_max_size,
                    "max-file": config.log_max_file,
                },
            },
            "deploy": {
                "resources": {
                    "limits": {
                        "cpus": config.cpus,
                        "memory": config.memory,
                        "pids": config.pids_limit,
                    }
                }
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"services": services}, indent=2))
    return path


def compose_service_names(environment_dir: Path) -> list[str]:
    names = {"main"}
    for compose_file in _compose_candidates(environment_dir):
        names.update(_read_compose_service_names(compose_file))
    return sorted(names)


def _read_compose_service_names(path: Path) -> set[str]:
    text = path.read_text(errors="replace")
    try:
        import yaml

        loaded = yaml.safe_load(text) or {}
        services = loaded.get("services") if isinstance(loaded, dict) else None
        if isinstance(services, dict):
            return {str(name) for name in services if str(name).strip()}
    except Exception:
        pass
    return _read_compose_service_names_fallback(text)


def _read_compose_service_names_fallback(text: str) -> set[str]:
    names: set[str] = set()
    in_services = False
    service_indent: int | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = raw_line.strip()
        if re.match(r"^services\s*:\s*$", stripped):
            in_services = True
            service_indent = None
            continue
        if not in_services:
            continue
        if indent == 0 and not stripped.startswith("services:"):
            break
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(?:#.*)?$", stripped)
        if not match:
            continue
        if service_indent is None:
            service_indent = indent
        if indent == service_indent:
            names.add(match.group(1))
    return names


def patch_dockerfile_for_apt_mirrors(
    dockerfile_text: str,
    config: AptMirrorConfig | None = None,
) -> tuple[str, bool]:
    """Inject mirror rewrite steps after Dockerfile FROM lines."""

    config = config or AptMirrorConfig()
    rewritten_text, rewritten = rewrite_dockerfile_base_images(dockerfile_text, config)
    rewritten_text, url_rewritten = rewrite_dockerfile_download_urls(rewritten_text, config)
    rewritten = rewritten or url_rewritten
    if not _needs_network_patch(rewritten_text):
        return rewritten_text, rewritten
    run_instruction = _network_mirror_run_instruction(config)
    env_instruction = _pip_env_instruction(config)
    ca_copy_instruction = _host_ca_copy_instruction(config)
    ca_trust_instruction = _host_ca_trust_instruction(config)
    lines = rewritten_text.splitlines(keepends=True)
    if "99hl-network" in rewritten_text:
        return rewritten_text, rewritten
    output: list[str] = []
    changed = rewritten
    for line in lines:
        output.append(line)
        if _should_inject_after_from(line):
            if not line.endswith("\n"):
                output.append("\n")
            # Establish CA trust before the mirror RUN so HTTPS apt/pip through a
            # MITM proxy can verify certificates during the mirror step itself.
            if ca_copy_instruction:
                output.append(ca_copy_instruction)
                output.append("\n")
                output.append(ca_trust_instruction)
                output.append("\n")
            output.append(run_instruction)
            output.append("\n")
            output.append(env_instruction)
            output.append("\n")
            changed = True
    if not changed:
        return dockerfile_text, False
    return "".join(output), True


def rewrite_dockerfile_base_images(
    dockerfile_text: str,
    config: AptMirrorConfig | None = None,
) -> tuple[str, bool]:
    """Rewrite implicit Docker Hub references to an explicit mirror."""

    config = config or AptMirrorConfig()
    if not config.docker_hub_mirror:
        return dockerfile_text, False

    changed = False
    stage_aliases: set[str] = set()
    rewritten_lines: list[str] = []
    for raw_line in dockerfile_text.splitlines(keepends=True):
        line, newline = _split_newline(raw_line)
        syntax_match = re.match(
            r"^(?P<prefix>\s*#\s*syntax=)(?P<image>\S+)(?P<trailing>\s*)$",
            line,
            flags=re.IGNORECASE,
        )
        if syntax_match:
            image = syntax_match.group("image")
            rewritten_image = effective_docker_image_reference(image, config)
            if rewritten_image != image:
                changed = True
            rewritten_lines.append(
                f"{syntax_match.group('prefix')}{rewritten_image}"
                f"{syntax_match.group('trailing')}{newline}"
            )
            continue

        arg_match = re.match(
            r"^(?P<prefix>\s*ARG\s+\w+=)(?P<image>[^\s#]+)(?P<trailing>.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if arg_match:
            image = arg_match.group("image")
            rewritten_image = image
            override = config.docker_image_overrides.get(image)
            if override:
                rewritten_image = override
            elif _looks_like_image_reference(image) and not _is_explicit_registry(image):
                rewritten_image = _mirror_image_reference(image, config.docker_hub_mirror)
            if rewritten_image != image:
                changed = True
            rewritten_lines.append(
                f"{arg_match.group('prefix')}{rewritten_image}"
                f"{arg_match.group('trailing')}{newline}"
            )
            continue

        match = re.match(
            r"^(?P<prefix>\s*FROM\s+)(?P<platform>--platform=\S+\s+)?"
            r"(?P<image>\S+)(?P<suffix>\s+(?:AS|as)\s+(?P<alias>\S+))?(?P<trailing>\s*)$",
            line,
        )
        if not match:
            rewritten_lines.append(raw_line)
            continue

        image = match.group("image")
        alias = match.group("alias")
        rewritten_image = effective_docker_image_reference(image, config, stage_aliases)
        if alias:
            stage_aliases.add(alias)
        if rewritten_image != image:
            changed = True
        rewritten_lines.append(
            f"{match.group('prefix')}{match.group('platform') or ''}"
            f"{rewritten_image}{match.group('suffix') or ''}"
            f"{match.group('trailing') or ''}{newline}"
        )
    return "".join(rewritten_lines), changed


def rewrite_dockerfile_download_urls(
    dockerfile_text: str,
    config: AptMirrorConfig | None = None,
) -> tuple[str, bool]:
    """Rewrite known external download URLs to configured mirrors."""

    config = config or AptMirrorConfig()
    rewritten = dockerfile_text
    changed = False
    for source, target in _normalized_rewrite_items(config.download_url_rewrites):
        if source in rewritten:
            rewritten = rewritten.replace(source, target)
            changed = True
    return rewritten, changed


def rewrite_download_url(url: str, config: AptMirrorConfig | None = None) -> str:
    """Return the URL after applying the configured literal download rewrites."""

    rewritten = url
    config = config or AptMirrorConfig()
    for source, target in _normalized_rewrite_items(config.download_url_rewrites):
        if rewritten.startswith(source):
            return f"{target}{rewritten[len(source):]}"
    return rewritten


def verifier_runtime_prepare_timeout_seconds(
    config: AptMirrorConfig | None = None,
    *,
    verifier_timeout_sec: int | None = None,
) -> int:
    """Return the single-operation timeout for verifier runtime preparation.

    The runtime preparation is best-effort infrastructure hardening. It must not
    consume the whole Harbor verifier command window, because Harbor's official
    verifier still needs enough time to run and decide pass/fail.
    """

    config = config or AptMirrorConfig()
    base_timeout = max(5, min(20, max(1, config.apt_timeout_seconds) // 2))
    if verifier_timeout_sec is None:
        return base_timeout
    try:
        verifier_timeout = int(verifier_timeout_sec)
    except (TypeError, ValueError):
        return base_timeout
    if verifier_timeout <= 0:
        return base_timeout
    if verifier_timeout <= base_timeout + 30:
        return 0
    return max(5, min(base_timeout, verifier_timeout // 4))


def verifier_runtime_network_prepare_command(config: AptMirrorConfig | None = None) -> str:
    """Return a verifier-time shell preflight for apt/pip network hardening.

    Dockerfile patching only affects image build time. TerminalBench verifiers
    often install packages inside /tests/test.sh after the container is already
    running, so the runtime container needs the same mirror and lock hygiene.
    """

    config = config or AptMirrorConfig()
    sed_args = _apt_source_sed_args(config)
    apt_conf = " ".join(
        shlex.quote(line) for line in _apt_conf_lines(config, include_lock_timeout=True)
    )
    pip_conf = " ".join(shlex.quote(line) for line in _pip_conf_lines(config))
    uv_env_path = "/root/.local/bin/env"
    lock_wait = _verifier_runtime_lock_cleanup_wait_seconds(config)
    lock_files = " ".join(
        shlex.quote(path)
        for path in [
            "/var/lib/dpkg/lock-frontend",
            "/var/lib/dpkg/lock",
            "/var/lib/apt/lists/lock",
            "/var/cache/apt/archives/lock",
        ]
    )
    return (
        "set +e; "
        "export DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none; "
        "hl_pkg_manager_running() { "
        "for comm in /proc/[0-9]*/comm; do "
        "[ -r \"$comm\" ] || continue; "
        "name=$(cat \"$comm\" 2>/dev/null || true); "
        "case \"$name\" in apt|apt-get|dpkg|unattended-upgrade|unattended-upgrades) "
        "return 0;; esac; "
        "done; return 1; "
        "}; "
        "hl_signal_pkg_managers() { "
        "signal=\"$1\"; "
        "for comm in /proc/[0-9]*/comm; do "
        "[ -r \"$comm\" ] || continue; "
        "name=$(cat \"$comm\" 2>/dev/null || true); "
        "case \"$name\" in apt|apt-get|dpkg|unattended-upgrade|unattended-upgrades) "
        "pid=${comm#/proc/}; pid=${pid%/comm}; "
        "if [ \"$signal\" = TERM ]; then kill -TERM \"$pid\" 2>/dev/null || true; "
        "else kill -KILL \"$pid\" 2>/dev/null || true; fi;; "
        "esac; "
        "done; "
        "}; "
        "mkdir -p /etc/apt/apt.conf.d /root/.local/bin "
        "/tmp/hl-verifier-cache/pip /tmp/hl-verifier-cache/uv "
        "/tmp/hl-verifier-cache/uv/archive-v0 "
        "/tmp/hl-verifier-cache/uv/wheels-v5 "
        "/tmp/hl-verifier-cache/uv/sdists-v9 "
        "/tmp/hl-verifier-cache/uv/simple-v18 "
        "/tmp/hl-verifier-cache/uv/builds-v0 "
        "/tmp/hl-verifier-cache/uv/interpreter-v4 "
        "/tmp/hl-verifier-cache/uv-python /tmp/hl-verifier-cache/uv-bin "
        "2>/dev/null || true; "
        "printf '%s\\n' preparing > /tmp/hl-verifier-network-prepared "
        "2>/dev/null || true; "
        "chmod a+rwx /tmp/hl-verifier-cache /tmp/hl-verifier-cache/pip "
        "/tmp/hl-verifier-cache/uv /tmp/hl-verifier-cache/uv/archive-v0 "
        "/tmp/hl-verifier-cache/uv/wheels-v5 /tmp/hl-verifier-cache/uv/sdists-v9 "
        "/tmp/hl-verifier-cache/uv/simple-v18 /tmp/hl-verifier-cache/uv/builds-v0 "
        "/tmp/hl-verifier-cache/uv/interpreter-v4 /tmp/hl-verifier-cache/uv-python "
        "/tmp/hl-verifier-cache/uv-bin 2>/dev/null || true; "
        "printf '%s\\n' "
        "'export PATH=/root/.local/bin:/tmp/hl-verifier-cache/uv-bin:/usr/local/bin:/usr/bin:/bin:$PATH' "
        "'export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/hl-verifier-cache/uv}' "
        f"> {uv_env_path} 2>/dev/null || true; "
        "for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list "
        "/etc/apt/sources.list.d/*.sources; do "
        "[ -f \"$file\" ] || continue; "
        f"sed -i {sed_args} \"$file\" 2>/dev/null || true; "
        "done; "
        f"printf '%s\\n' {apt_conf} > /etc/apt/apt.conf.d/99hl-network "
        "2>/dev/null || true; "
        f"printf '%s\\n' {pip_conf} > /etc/pip.conf 2>/dev/null || true; "
        "if command -v apt-get >/dev/null 2>&1; then "
        "hl_waited=0; "
        f"while hl_pkg_manager_running && [ \"$hl_waited\" -lt {lock_wait} ]; do "
        "sleep 1; hl_waited=$((hl_waited + 1)); "
        "done; "
        "if hl_pkg_manager_running; then "
        "hl_signal_pkg_managers TERM; sleep 1; "
        "if hl_pkg_manager_running; then hl_signal_pkg_managers KILL; sleep 1; fi; "
        "fi; "
        "if ! hl_pkg_manager_running; then "
        f"rm -f {lock_files} 2>/dev/null || true; "
        "fi; "
        "fi; "
        "if ! command -v uvx >/dev/null 2>&1 && [ -x /root/.local/bin/uvx ]; then "
        "true; "
        "elif ! command -v uvx >/dev/null 2>&1 "
        "&& command -v python3 >/dev/null 2>&1 "
        "&& python3 -m uv --version >/dev/null 2>&1; then "
        "printf '%s\\n' '#!/bin/sh' "
        "'export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/hl-verifier-cache/uv}' "
        "'exec python3 -m uv \"$@\"' "
        "> /root/.local/bin/uv 2>/dev/null || true; "
        "printf '%s\\n' '#!/bin/sh' "
        "'export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/hl-verifier-cache/uv}' "
        "'exec python3 -m uv tool run \"$@\"' "
        "> /root/.local/bin/uvx 2>/dev/null || true; "
        "chmod +x /root/.local/bin/uv /root/.local/bin/uvx 2>/dev/null || true; "
        "fi; "
        "printf '%s\\n' prepared > /tmp/hl-verifier-network-prepared "
        "2>/dev/null || true; "
        "exit 0"
    )


def _verifier_runtime_lock_cleanup_wait_seconds(config: AptMirrorConfig) -> int:
    """Return a short best-effort wait for stale package-manager locks."""

    return max(2, min(6, max(1, config.apt_timeout_seconds) // 5))


def patch_compose_for_docker_mirrors(
    compose_text: str,
    config: AptMirrorConfig | None = None,
) -> tuple[str, bool]:
    """Rewrite simple Docker Compose image references to the explicit mirror."""

    config = config or AptMirrorConfig()
    if not config.docker_hub_mirror:
        return compose_text, False

    changed = False
    output: list[str] = []
    for raw_line in compose_text.splitlines(keepends=True):
        line, newline = _split_newline(raw_line)
        match = re.match(
            r"^(?P<prefix>\s*image:\s*)(?P<quote>['\"]?)(?P<image>[^'\"\s#]+)"
            r"(?P=quote)(?P<trailing>\s*(?:#.*)?)$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            output.append(raw_line)
            continue
        image = match.group("image")
        rewritten_image = effective_docker_image_reference(image, config)
        if rewritten_image != image:
            changed = True
        quote = match.group("quote")
        output.append(
            f"{match.group('prefix')}{quote}{rewritten_image}{quote}"
            f"{match.group('trailing')}{newline}"
        )
    return "".join(output), changed


def _dockerfile_candidates(environment_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in environment_dir.rglob("Dockerfile*")
        if path.is_file() and ".git" not in path.parts
    )


def _compose_candidates(environment_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in environment_dir.rglob("*")
        if path.is_file()
        and path.name.lower()
        in {
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
            "compose.yaml",
        }
        and ".git" not in path.parts
    )


def _should_inject_after_from(line: str) -> bool:
    match = re.match(
        r"^\s*FROM\s+(?:--platform=\S+\s+)?(?P<image>\S+)",
        line,
        flags=re.IGNORECASE,
    )
    return bool(match and match.group("image").lower() != "scratch")


def _needs_network_patch(text: str) -> bool:
    needles = (
        "apt-get",
        "apt ",
        "/etc/apt/",
        "deb.debian.org",
        "security.debian.org",
        "archive.ubuntu.com",
        "security.ubuntu.com",
        "pip install",
        "pip3 install",
        "python -m pip",
        "uv pip",
        "pypi.org",
        "files.pythonhosted.org",
        "wget ",
        "curl ",
        "dl-cdn.alpinelinux.org",
    )
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _is_explicit_registry(image: str) -> bool:
    if "/" not in image:
        return False
    first = image.split("/", 1)[0]
    return "." in first or ":" in first or first == "localhost"


def _docker_image_exists_locally(image: str) -> bool:
    if not image:
        return False
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=LOCAL_IMAGE_INSPECT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _docker_image_has_registry_provenance(
    image: str,
    equivalent_reference: str = "",
) -> bool:
    """Return whether a local image has the exact declared registry digest.

    A locally built image can be tagged with any benchmark image name. Mere tag
    presence therefore cannot establish that it is equivalent to the declared
    prebuilt. Tag references cannot be proven from local Docker metadata because
    ``RepoDigests`` does not record which tag produced a digest. Only digest-pinned
    effective or explicit original references are eligible for local reuse.
    """

    references = (image, equivalent_reference)
    expected_digests = {
        digest
        for digest in (
            _normalized_registry_digest(reference) for reference in references
        )
        if digest
    }
    if not expected_digests:
        return False
    try:
        completed = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{json .RepoDigests}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=LOCAL_IMAGE_INSPECT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if completed.returncode != 0:
        return False
    try:
        repo_digests = json.loads(completed.stdout.strip() or "[]")
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(repo_digests, list):
        return False
    actual_digests = {
        digest
        for digest in (
            _normalized_registry_digest(str(repo_digest))
            for repo_digest in repo_digests
        )
        if digest
    }
    return bool(expected_digests & actual_digests)


def _normalized_registry_repository(reference: str) -> str:
    """Normalize a tag or digest to a registry-qualified repository name."""

    reference = reference.strip()
    if not reference or re.fullmatch(r"sha256:[0-9a-fA-F]+", reference):
        return ""
    reference = reference.split("@", 1)[0]
    if not reference:
        return ""

    if _is_explicit_registry(reference):
        registry, path = reference.split("/", 1)
    else:
        registry, path = "docker.io", reference
    if not path:
        return ""
    head, separator, leaf = path.rpartition("/")
    if ":" in leaf:
        leaf = leaf.split(":", 1)[0]
    path = f"{head}{separator}{leaf}" if separator else leaf
    if not path:
        return ""
    if "/" not in path:
        path = f"library/{path}"

    registry = registry.lower()
    if registry in {"index.docker.io", "registry-1.docker.io"}:
        registry = "docker.io"
    return f"{registry}/{path.lower()}"


def _normalized_registry_digest(reference: str) -> str:
    """Normalize a registry digest while preserving its content identity."""

    reference = reference.strip()
    repository, separator, digest = reference.partition("@")
    if not separator or not re.fullmatch(
        r"[a-zA-Z0-9_+.-]+:[0-9a-fA-F]+",
        digest,
    ):
        return ""
    normalized_repository = _normalized_registry_repository(repository)
    if not normalized_repository:
        return ""
    return f"{normalized_repository}@{digest.lower()}"


def _tag_available_prebuilt_fallback(image: str, original_image: str = "") -> str:
    """Tag a local equivalent image to the effective prebuilt reference.

    Only an explicit original reference recorded during mirror rewriting is
    considered, and only when it is digest-pinned. Tag-only and inferred
    cross-registry candidates cannot establish content identity and must be
    pulled from their declared registry.
    """

    image = image.strip()
    original_image = original_image.strip()
    if (
        not image
        or not original_image
        or image == original_image
        or not _normalized_registry_digest(original_image)
    ):
        return ""
    if _docker_image_has_registry_provenance(original_image) and _docker_tag_image(
        original_image,
        image,
    ):
        return original_image
    return ""


def _docker_tag_image(source: str, target: str) -> bool:
    if not source or not target or source == target:
        return False
    try:
        completed = subprocess.run(
            ["docker", "tag", source, target],
            check=False,
            capture_output=True,
            text=True,
            timeout=LOCAL_IMAGE_INSPECT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and _docker_image_exists_locally(target)


def _prebuilt_pull_failure_can_try_original(stdout: Any, stderr: Any) -> bool:
    """Return True when a mirror pull failure is a deterministic access/not-found
    error that the original Docker Hub image can plausibly resolve instead.

    This is intentionally generic: it keys off registry denial/not-found markers,
    not any specific task, image, or mirror host.
    """
    text = _subprocess_failure_detail(stdout, stderr).lower()
    return any(
        marker in text
        for marker in [
            "403 forbidden",
            "pull access denied",
            "denied: requested access",
            "manifest unknown",
            "repository does not exist",
            "not found",
        ]
    )


def _prebuilt_warmup_error(
    message: str,
    *,
    kind: str,
    deterministic_access_failure: bool,
) -> RuntimeError:
    error = RuntimeError(message)
    error._hl_prebuilt_warmup_failure_kind = kind  # type: ignore[attr-defined]
    error._hl_prebuilt_warmup_deterministic_access_failure = (  # type: ignore[attr-defined]
        deterministic_access_failure
    )
    return error


def _write_prebuilt_warmup_failure_receipt(
    path: Path,
    *,
    kind: str,
    deterministic_access_failure: bool,
) -> None:
    """Persist host-owned provenance before the task container can start."""

    if kind not in {
        "prebuilt_image_cache_warmup_failure",
        "prebuilt_image_cache_warmup_timeout",
    }:
        return

    payload = {
        "schema": PREBUILT_WARMUP_FAILURE_RECEIPT_SCHEMA,
        "kind": kind,
        "source": "apt_mirror_docker_environment_start",
        "deterministic_access_failure": deterministic_access_failure,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{secrets.token_hex(8)}")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        content = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _pull_and_tag_original_prebuilt_image(source: str, target: str, timeout: int) -> None:
    """Pull the original prebuilt image and tag it to the effective reference.

    Raises RuntimeError with diagnostics if the pull or tag fails.
    """
    if not source or not target or source == target:
        raise RuntimeError("original image fallback has no distinct source image")
    try:
        completed = subprocess.run(
            ["docker", "pull", source],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _subprocess_failure_detail(exc.stdout, exc.stderr)
        raise RuntimeError(
            f"docker pull {source} timed out after {timeout} seconds. {detail}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = _subprocess_failure_detail(exc.stdout, exc.stderr)
        raise RuntimeError(
            f"docker pull {source} failed with return code {exc.returncode}. {detail}"
        ) from exc
    if completed.returncode != 0:
        detail = _subprocess_failure_detail(completed.stdout, completed.stderr)
        raise RuntimeError(
            f"docker pull {source} failed with return code {completed.returncode}. {detail}"
        )
    if not _docker_tag_image(source, target):
        raise RuntimeError(f"docker tag {source} {target} failed after original pull")


def _mirror_image_reference(image: str, docker_hub_mirror: str) -> str:
    mirror = docker_hub_mirror.rstrip("/")
    normalized = _strip_docker_hub_registry(image)
    image = normalized or image
    if "/" in image:
        return f"{mirror}/{image}"
    return f"{mirror}/library/{image}"


def effective_docker_image_reference(
    image: str,
    config: AptMirrorConfig | None = None,
    stage_aliases: set[str] | None = None,
) -> str:
    """Return the exact image reference Harbor should use after hardening."""

    config = config or AptMirrorConfig()
    override = config.docker_image_overrides.get(image)
    if override:
        return override
    if not config.docker_hub_mirror:
        return image
    if _should_mirror_image_reference(image, stage_aliases):
        return _mirror_image_reference(image, config.docker_hub_mirror)
    return image


def _should_mirror_image_reference(
    image: str,
    stage_aliases: set[str] | None = None,
) -> bool:
    if not image or image.lower() == "scratch" or image.startswith(("$", "{")):
        return False
    if stage_aliases and image in stage_aliases:
        return False
    return not _is_explicit_registry(image) or _is_docker_hub_registry(image)


def _looks_like_image_reference(value: str) -> bool:
    if not value or value.startswith(("$", "{")):
        return False
    if "://" in value:
        return False
    return "/" in value or ":" in value


def _is_docker_hub_registry(image: str) -> bool:
    if "/" not in image:
        return False
    first = image.split("/", 1)[0].lower()
    return first in {"docker.io", "index.docker.io", "registry-1.docker.io"}


def _strip_docker_hub_registry(image: str) -> str | None:
    if not _is_docker_hub_registry(image):
        return None
    return image.split("/", 1)[1]


def _split_newline(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


HL_HOST_CA_BUILD_CONTEXT_NAME = ".hl-host-ca.crt"
HL_HOST_CA_CONTAINER_PATH = "/usr/local/share/ca-certificates/hl-host-ca.crt"


def _host_ca_bundle_available(config: AptMirrorConfig) -> bool:
    if not config.inject_host_ca_into_build:
        return False
    bundle = str(config.host_ca_cert_bundle or "").strip()
    return bool(bundle) and Path(bundle).is_file()


def _host_ca_copy_instruction(config: AptMirrorConfig) -> str:
    if not _host_ca_bundle_available(config):
        return ""
    return f"COPY {HL_HOST_CA_BUILD_CONTEXT_NAME} {HL_HOST_CA_CONTAINER_PATH}"


def _host_ca_trust_instruction(config: AptMirrorConfig) -> str:
    """Trust the injected host CA before any apk/apt/pip/npm step.

    Covers Debian/Ubuntu (update-ca-certificates), Alpine (append to the system
    bundle, installing ca-certificates first if update-ca-certificates is
    missing), and sets pip/node/git CA env vars. Never downgrades HTTPS to HTTP.
    """
    if not _host_ca_bundle_available(config):
        return ""
    system_bundle = "/etc/ssl/certs/ca-certificates.crt"
    env_pairs = _host_ca_env_pairs(config)
    env_instruction = ""
    if env_pairs:
        env_instruction = "ENV " + " ".join(
            f"{key}={_docker_env_quote(value)}" for key, value in env_pairs.items()
        )
    run_instruction = (
        "RUN set -eux; "
        "if command -v update-ca-certificates >/dev/null 2>&1; then "
        "update-ca-certificates; "
        "elif command -v apk >/dev/null 2>&1; then "
        "apk add --no-cache ca-certificates >/dev/null 2>&1 || true; "
        "update-ca-certificates >/dev/null 2>&1 || true; "
        "fi; "
        # Always also append to the merged bundle so tools that read it directly
        # (pip, curl, git) trust the CA even where update-ca-certificates is absent.
        f"if [ -f {system_bundle} ]; then "
        f"cat {HL_HOST_CA_CONTAINER_PATH} >> {system_bundle}; fi"
    )
    if env_instruction:
        return run_instruction + "\n" + env_instruction
    return run_instruction


def _host_ca_env_pairs(config: AptMirrorConfig) -> dict[str, str]:
    if not _host_ca_bundle_available(config):
        return {}
    system_bundle = "/etc/ssl/certs/ca-certificates.crt"
    return {
        "PIP_CERT": system_bundle,
        "NODE_EXTRA_CA_CERTS": HL_HOST_CA_CONTAINER_PATH,
        "REQUESTS_CA_BUNDLE": system_bundle,
        "SSL_CERT_FILE": system_bundle,
        "GIT_SSL_CAINFO": system_bundle,
    }


def _network_mirror_run_instruction(config: AptMirrorConfig) -> str:
    sed_args = _apt_source_sed_args(config)
    apt_conf = " ".join(
        shlex.quote(line) for line in _apt_conf_lines(config, include_lock_timeout=False)
    )
    pip_conf = " ".join(shlex.quote(line) for line in _pip_conf_lines(config))
    wget_conf = " ".join(shlex.quote(line) for line in _wget_conf_lines(config))
    wget_retry_wrapper = ""
    if config.download_retry_wrapper:
        wget_retry_wrapper = _download_retry_wrapper_instruction(config)
    ca_bootstrap = ""
    if config.bootstrap_ca_certificates:
        ca_bootstrap = (
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get update; "
            "apt-get install -y --no-install-recommends ca-certificates; "
            "rm -rf /var/lib/apt/lists/*; "
            "fi; "
        )
    return (
        "RUN set -eux; "
        "mkdir -p /etc/apt/apt.conf.d; "
        "for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list "
        "/etc/apt/sources.list.d/*.sources; do "
        "[ -f \"$file\" ] || continue; "
        f"sed -i {sed_args} \"$file\"; "
        "done; "
        f"printf '%s\\n' {apt_conf} > /etc/apt/apt.conf.d/99hl-network; "
        f"{ca_bootstrap}"
        f"printf '%s\\n' {wget_conf} > /etc/hl-wgetrc; "
        f"{wget_retry_wrapper}"
        f"printf '%s\\n' {pip_conf} > /etc/pip.conf"
    )


def _apt_source_replacements(config: AptMirrorConfig) -> list[tuple[str, str]]:
    return [
        ("http://deb.debian.org/debian-security", config.debian_security_mirror),
        ("https://deb.debian.org/debian-security", config.debian_security_mirror),
        ("http://security.debian.org/debian-security", config.debian_security_mirror),
        ("https://security.debian.org/debian-security", config.debian_security_mirror),
        ("http://deb.debian.org/debian", config.debian_mirror),
        ("https://deb.debian.org/debian", config.debian_mirror),
        ("http://archive.ubuntu.com/ubuntu", config.ubuntu_mirror),
        ("https://archive.ubuntu.com/ubuntu", config.ubuntu_mirror),
        ("http://security.ubuntu.com/ubuntu", config.ubuntu_mirror),
        ("https://security.ubuntu.com/ubuntu", config.ubuntu_mirror),
    ]


def _apt_source_sed_args(config: AptMirrorConfig) -> str:
    return " ".join(
        f"-e {shlex.quote(f's#{source}#{target}#g')}"
        for source, target in _apt_source_replacements(config)
    )


def _apt_conf_lines(
    config: AptMirrorConfig,
    *,
    include_lock_timeout: bool,
) -> list[str]:
    lines = [
        f'Acquire::Retries "{config.apt_retries}";',
        f'Acquire::http::Timeout "{config.apt_timeout_seconds}";',
        f'Acquire::https::Timeout "{config.apt_timeout_seconds}";',
    ]
    if include_lock_timeout:
        lock_timeout = max(30, min(180, config.apt_timeout_seconds * 2))
        lines.append(f'DPkg::Lock::Timeout "{lock_timeout}";')
    return lines


def _pip_conf_lines(config: AptMirrorConfig) -> list[str]:
    lines = [
        "[global]",
        f"index-url = {config.pypi_index_url}",
        f"timeout = {config.pip_timeout_seconds}",
        f"retries = {config.pip_retries}",
    ]
    if config.pypi_trusted_host:
        lines.append(f"trusted-host = {config.pypi_trusted_host}")
    return lines


def _wget_conf_lines(config: AptMirrorConfig) -> list[str]:
    return [
        f"tries = {config.apt_retries}",
        f"timeout = {config.apt_timeout_seconds}",
        "retry_connrefused = on",
    ]


def _download_retry_wrapper_instruction(config: AptMirrorConfig) -> str:
    script_lines = [
        "#!/bin/sh",
        'if [ "${HL_NETWORK_WRAPPER_DISABLE:-}" = "1" ]; then',
        '  for candidate in /usr/bin/wget /bin/wget; do',
        '    if [ -x "$candidate" ]; then exec "$candidate" "$@"; fi',
        "  done",
        '  echo "hl wget wrapper: real wget not found" >&2',
        "  exit 127",
        "fi",
        'real_wget="${HL_REAL_WGET:-}"',
        'if [ -z "$real_wget" ]; then',
        '  for candidate in /usr/local/bin/wget.real /usr/bin/wget /bin/wget; do',
        '    if [ -x "$candidate" ] && [ "$candidate" != "$0" ]; then',
        '      real_wget="$candidate"',
        "      break",
        "    fi",
        "  done",
        "fi",
        'if [ -z "$real_wget" ]; then',
        '  echo "hl wget wrapper: real wget not found" >&2',
        "  exit 127",
        "fi",
        f'attempts="${{HL_DOWNLOAD_RETRIES:-{config.apt_retries}}}"',
        'delay="${HL_DOWNLOAD_RETRY_DELAY:-3}"',
        f'timeout="${{HL_WGET_TIMEOUT:-{config.apt_timeout_seconds}}}"',
        "count=1",
        "while :; do",
        '  if "$real_wget" --help 2>&1 | grep -q -- "--waitretry"; then',
        '    "$real_wget" --tries=1 --timeout="$timeout" --waitretry="$delay" '
        '--retry-connrefused "$@"',
        "  else",
        '    "$real_wget" "$@"',
        "  fi",
        "  status=$?",
        '  if [ "$status" -eq 0 ] || [ "$count" -ge "$attempts" ]; then',
        '    exit "$status"',
        "  fi",
        '  case "$status" in',
        "    4|5|8)",
        '      sleep "$delay"',
        "      ;;",
        "    *)",
        '      exit "$status"',
        "      ;;",
        "  esac",
        "  count=$((count + 1))",
        "done",
    ]
    script = " ".join(shlex.quote(line) for line in script_lines)
    return (
        "mkdir -p /usr/local/bin; "
        "if [ -x /usr/local/bin/wget ] && [ ! -e /usr/local/bin/wget.real ]; "
        "then mv /usr/local/bin/wget /usr/local/bin/wget.real; fi; "
        f"printf '%s\\n' {script} > /usr/local/bin/wget; "
        "chmod +x /usr/local/bin/wget; "
    )


def _pip_env_instruction(config: AptMirrorConfig) -> str:
    parts = {
        "PIP_INDEX_URL": config.pypi_index_url,
        "PIP_DEFAULT_TIMEOUT": str(config.pip_timeout_seconds),
        "PIP_RETRIES": str(config.pip_retries),
        "UV_INDEX_URL": config.pypi_index_url,
        "WGETRC": "/etc/hl-wgetrc",
        "HL_DOWNLOAD_RETRIES": str(config.apt_retries),
        "HL_WGET_TIMEOUT": str(config.apt_timeout_seconds),
    }
    if config.pypi_trusted_host:
        parts["PIP_TRUSTED_HOST"] = config.pypi_trusted_host
    return "ENV " + " ".join(
        f"{key}={_docker_env_quote(value)}" for key, value in parts.items() if value
    )


def _docker_env_quote(value: str) -> str:
    return json.dumps(value)


def _truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _positive_int(value: str | int, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _subprocess_failure_detail(stdout: Any, stderr: Any) -> str:
    parts: list[str] = []
    stdout_text = str(stdout or "").strip()
    stderr_text = str(stderr or "").strip()
    if stdout_text:
        parts.append(f"Stdout: {stdout_text[-1000:]}")
    if stderr_text:
        parts.append(f"Stderr: {stderr_text[-1000:]}")
    return " ".join(parts) if parts else "No docker pull output captured."


def _parse_image_overrides(value: str | dict[str, str] | None) -> dict[str, str]:
    return _parse_rewrite_map(value, default=None)


def _parse_docker_labels(value: str | dict[str, str] | None) -> dict[str, str]:
    labels = dict(DEFAULT_DOCKER_LABELS)
    if value is None:
        return labels
    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return labels
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pairs = [item.strip() for item in text.split(",") if item.strip()]
            parsed = dict(item.split("=", 1) for item in pairs if "=" in item)
    if not isinstance(parsed, dict):
        return labels
    for key, label_value in parsed.items():
        key_text = str(key).strip()
        value_text = str(label_value).strip()
        if key_text:
            labels[key_text] = value_text
    return labels


def _parse_rewrite_map(
    value: str | dict[str, str] | None,
    *,
    default: dict[str, str] | None,
) -> dict[str, str]:
    if value is None:
        return dict(default or {})
    if isinstance(value, dict):
        return {
            str(source): str(target)
            for source, target in value.items()
            if str(source).strip()
            and str(target).strip()
            and str(source).strip() != str(target).strip()
        }
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {
            str(source): str(target)
            for source, target in parsed.items()
            if str(source).strip()
            and str(target).strip()
            and str(source).strip() != str(target).strip()
        }
    overrides: dict[str, str] = {}
    for item in re.split(r"[\n,]+", text):
        source, separator, target = item.partition("=")
        if separator and source.strip() and target.strip() and source.strip() != target.strip():
            overrides[source.strip()] = target.strip()
    return overrides


def _normalized_rewrite_items(rewrites: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(
        (
            (source.rstrip("/"), target.rstrip("/"))
            for source, target in rewrites.items()
            if source.strip() and target.strip() and source.rstrip("/") != target.rstrip("/")
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
