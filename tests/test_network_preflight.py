import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

import bench.network_environment as net_env
from bench.network_environment import (
    DEFAULT_ALPINE_MIRROR,
    PREBUILT_WARMUP_FAILURE_RECEIPT,
    PREBUILT_WARMUP_FAILURE_RECEIPT_SCHEMA,
    AptMirrorConfig,
    AptMirrorDockerEnvironment,
    effective_docker_image_reference,
    patch_compose_for_docker_mirrors,
    patch_dockerfile_for_apt_mirrors,
    prepare_apt_mirror_environment,
    rewrite_download_url,
    verifier_runtime_network_prepare_command,
    verifier_runtime_prepare_timeout_seconds,
)
from hl.web_references import DEFAULT_WEB_USER_AGENT, WECHAT_ARTICLE_USER_AGENT
from scripts import network_preflight


def test_web_reference_user_agent_uses_wechat_client_for_mp_articles():
    headers = network_preflight.request_headers_for_url(
        "https://mp.weixin.qq.com/s/sgP8m1nnW7JhsDT7Ki7nVw"
    )
    assert "MicroMessenger" in headers["User-Agent"]
    assert "Process/tools" in headers["User-Agent"]
    assert headers["Referer"] == "https://mp.weixin.qq.com/"
    assert headers["Accept-Language"] == "zh-CN,zh;q=0.9"
    assert "text/html" in headers["Accept"]
    assert (
        network_preflight.request_headers_for_url("https://example.com/article")["User-Agent"]
        == DEFAULT_WEB_USER_AGENT
    )
    assert "Referer" not in network_preflight.request_headers_for_url(
        "https://example.com/article"
    )


def test_check_http_head_sends_wechat_user_agent_for_mp_articles(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return b"<html><h1>Self-Harness article</h1></html>"

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["user_agent"] = request.get_header("User-agent")
        captured["referer"] = request.get_header("Referer")
        captured["accept_language"] = request.get_header("Accept-language")
        return FakeResponse()

    monkeypatch.setattr(network_preflight, "urlopen", fake_urlopen)

    result = network_preflight.check_http_head(
        "wechat_article",
        "https://mp.weixin.qq.com/s/sgP8m1nnW7JhsDT7Ki7nVw",
        timeout=3,
    )

    assert result.ok is True
    assert captured["url"] == "https://mp.weixin.qq.com/s/sgP8m1nnW7JhsDT7Ki7nVw"
    assert captured["method"] == "GET"
    assert captured["timeout"] == 3
    assert captured["user_agent"] == WECHAT_ARTICLE_USER_AGENT
    assert "Process/tools" in captured["user_agent"]
    assert "MicroMessenger" in captured["user_agent"]
    assert captured["referer"] == "https://mp.weixin.qq.com/"
    assert captured["accept_language"] == "zh-CN,zh;q=0.9"


def test_check_http_head_rejects_wechat_verification_page(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return "\u73af\u5883\u5f02\u5e38\uff0c\u5b8c\u6210\u9a8c\u8bc1\u540e\u5373\u53ef\u7ee7\u7eed\u8bbf\u95ee".encode()

    monkeypatch.setattr(network_preflight, "urlopen", lambda request, *, timeout: FakeResponse())

    result = network_preflight.check_http_head(
        "wechat_article",
        "https://mp.weixin.qq.com/s/sgP8m1nnW7JhsDT7Ki7nVw",
        timeout=3,
    )

    assert result.ok is False
    assert "WeChat environment verification page" in result.detail


def test_patch_dockerfile_for_apt_mirrors_injects_after_from_without_source_edit():
    text = "FROM python:3.13-slim-bookworm\nRUN apt-get update && apt-get install -y gnucobol3\n"

    patched, changed = patch_dockerfile_for_apt_mirrors(
        text,
        AptMirrorConfig(
            debian_mirror="https://mirror.local/debian",
            debian_security_mirror="https://mirror.local/debian-security",
            ubuntu_mirror="https://mirror.local/ubuntu",
        ),
    )

    assert changed is True
    assert "https://mirror.local/debian" in patched
    assert "FROM python:3.13-slim-bookworm" in patched
    assert "Acquire::Retries" in patched
    assert "PIP_INDEX_URL" in patched
    assert "/etc/pip.conf" in patched
    assert patched.index("RUN set -eux") < patched.index("RUN apt-get update")


def test_patch_dockerfile_preserves_multistage_alias_references():
    text = (
        "FROM ubuntu:24.04 as build\n"
        "RUN apt-get update\n"
        "FROM build as target\n"
        "RUN echo ok\n"
    )

    patched, changed = patch_dockerfile_for_apt_mirrors(text)

    assert changed is True
    assert "FROM ubuntu:24.04 as build" in patched
    assert "FROM build as target" in patched


def test_default_apt_mirrors_use_public_https_endpoints():
    text = "FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y ca-certificates\n"

    patched, changed = patch_dockerfile_for_apt_mirrors(text)

    assert changed is True
    assert "FROM ubuntu:24.04" in patched
    assert "https://archive.ubuntu.com/ubuntu" in patched
    assert "https://deb.debian.org/debian" in patched
    assert "command -v apt-get" in patched
    assert "ca-certificates" in patched


def test_patch_dockerfile_rewrites_syntax_and_arg_images_without_touching_variables():
    text = (
        "# syntax=docker/dockerfile:1.7\n"
        "ARG BASE=python:3.11-slim\n"
        "ARG RUNTIME=${BASE}\n"
        "FROM ${BASE} as build\n"
        "RUN pip install requests==2.32.4\n"
        "FROM scratch\n"
    )

    patched, changed = patch_dockerfile_for_apt_mirrors(text)

    assert changed is True
    assert "# syntax=docker/dockerfile:1.7" in patched
    assert "ARG BASE=python:3.11-slim" in patched
    assert "ARG RUNTIME=${BASE}" in patched
    assert "FROM ${BASE} as build" in patched
    assert "FROM scratch" in patched
    assert "PIP_INDEX_URL" in patched
    assert "PIP_TRUSTED_HOST" not in patched


def test_patch_compose_for_docker_mirrors_rewrites_simple_image_references():
    text = (
        "services:\n"
        "  db:\n"
        "    image: postgres:16\n"
        "  cache:\n"
        "    image: 'redis:7' # keep comment\n"
        "  local:\n"
        "    image: registry.local/tool:latest\n"
    )

    patched, changed = patch_compose_for_docker_mirrors(text)

    assert changed is False
    assert "image: postgres:16" in patched
    assert "image: 'redis:7' # keep comment" in patched
    assert "image: registry.local/tool:latest" in patched


def test_prepare_apt_mirror_environment_patches_only_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dockerfile = source / "Dockerfile"
    original = "FROM ubuntu:24.04\nRUN apt-get update\n"
    dockerfile.write_text(original)

    destination = prepare_apt_mirror_environment(source, tmp_path / "dest")

    assert dockerfile.read_text() == original
    assert "https://archive.ubuntu.com/ubuntu" in (destination / "Dockerfile").read_text()
    assert (destination / ".hl_apt_mirror.json").exists()


def test_patch_dockerfile_injects_host_ca_before_mirror_run():
    text = "FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y curl\n"

    patched, changed = patch_dockerfile_for_apt_mirrors(
        text,
        AptMirrorConfig(inject_host_ca_into_build=True),
    )

    assert changed is True
    # The host CA must be copied and trusted before the mirror RUN so HTTPS
    # apt/pip through a MITM proxy can verify certificates.
    assert "COPY .hl-host-ca.crt" in patched
    ca_index = patched.index(".hl-host-ca.crt")
    mirror_index = patched.index("99hl-network")
    assert ca_index < mirror_index, "CA trust must be established before mirror RUN"
    # Trust-store update for both Debian/Ubuntu and Alpine, plus pip/node env.
    assert "update-ca-certificates" in patched
    assert "PIP_CERT" in patched
    assert "NODE_EXTRA_CA_CERTS" in patched


def test_patch_dockerfile_no_host_ca_when_disabled():
    text = "FROM ubuntu:24.04\nRUN apt-get update\n"

    patched, _ = patch_dockerfile_for_apt_mirrors(
        text,
        AptMirrorConfig(inject_host_ca_into_build=False),
    )

    assert ".hl-host-ca.crt" not in patched


def test_prepare_apt_mirror_environment_copies_host_ca_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN apt-get update\n")
    host_ca = tmp_path / "host-ca.crt"
    host_ca.write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")

    destination = prepare_apt_mirror_environment(
        source,
        tmp_path / "dest",
        AptMirrorConfig(
            inject_host_ca_into_build=True,
            host_ca_cert_bundle=str(host_ca),
        ),
    )

    copied = destination / ".hl-host-ca.crt"
    assert copied.is_file()
    assert copied.read_text() == host_ca.read_text()
    assert "COPY .hl-host-ca.crt" in (destination / "Dockerfile").read_text()


def test_prepare_apt_mirror_environment_skips_host_ca_when_bundle_missing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM ubuntu:24.04\nRUN apt-get update\n")

    destination = prepare_apt_mirror_environment(
        source,
        tmp_path / "dest",
        AptMirrorConfig(
            inject_host_ca_into_build=True,
            host_ca_cert_bundle=str(tmp_path / "does-not-exist.crt"),
        ),
    )

    # No CA file to copy -> do not emit a COPY that would break the build.
    assert not (destination / ".hl-host-ca.crt").exists()
    assert "COPY .hl-host-ca.crt" not in (destination / "Dockerfile").read_text()


def test_patch_dockerfile_applies_configured_image_overrides():
    text = "FROM debian:13.0-slim\nRUN apt-get update\n"

    patched, changed = patch_dockerfile_for_apt_mirrors(
        text,
        AptMirrorConfig(
            docker_hub_mirror="docker.1ms.run",
            docker_image_overrides={
                "debian:13.0-slim": "docker.m.daocloud.io/library/debian:13.0-slim",
            },
        ),
    )

    assert changed is True
    assert "FROM docker.m.daocloud.io/library/debian:13.0-slim" in patched
    assert "docker.1ms.run/library/debian:13.0-slim" not in patched


def test_effective_docker_image_reference_can_disable_default_mirror():
    assert (
        effective_docker_image_reference(
            "alexgshaw/qemu-startup:20251031",
            AptMirrorConfig(docker_hub_mirror=""),
        )
        == "alexgshaw/qemu-startup:20251031"
    )


def test_apt_mirror_environment_rewrites_configured_prebuilt_docker_image(monkeypatch, tmp_path):
    captured = {}

    def fake_init(self, environment_dir, *args, **kwargs):
        captured["environment_dir"] = environment_dir
        captured["task_env_config"] = kwargs["task_env_config"]

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/qemu-startup:20251031",
        ),
        prebuilt_docker_hub_mirror="registry.example",
    )

    assert captured["environment_dir"] == tmp_path
    assert (
        captured["task_env_config"].docker_image
        == "registry.example/alexgshaw/qemu-startup:20251031"
    )
    assert env._hl_original_prebuilt_image == "alexgshaw/qemu-startup:20251031"
    assert (
        env._hl_effective_prebuilt_image
        == "registry.example/alexgshaw/qemu-startup:20251031"
    )


def test_prebuilt_cache_warmup_runs_before_compose_up(monkeypatch, tmp_path):
    calls = []

    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: calls.append(("info", args)))

    async def fake_start(self, force_build):
        calls.append(("super_start", force_build))

    def fake_run(argv, **kwargs):
        calls.append(("pull", argv, kwargs.get("timeout")))
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 1, "", "missing")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/qemu-startup:20251031",
        ),
        prebuilt_docker_pull_timeout_seconds=42,
        prebuilt_docker_hub_mirror="registry.example",
    )

    asyncio.run(env.start(force_build=False))

    assert calls[0][0] == "info"
    assert calls[1] == (
        "pull",
        ["docker", "pull", "registry.example/alexgshaw/qemu-startup:20251031"],
        42,
    )
    assert calls[2] == ("super_start", False)
    assert all(
        call[1][:3] != ["docker", "image", "inspect"]
        for call in calls
        if call[0] == "pull"
    )


def test_resource_overlay_precedes_current_harbor_egress_overlays(
    monkeypatch, tmp_path
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")
    trial_dir = tmp_path / "trial"

    monkeypatch.setattr(
        DockerEnvironment,
        "_egress_control_kernel_support",
        staticmethod(lambda: True),
    )
    env = AptMirrorDockerEnvironment(
        environment_dir,
        environment_name="fix-git",
        session_id="session-1",
        trial_paths=TrialPaths(trial_dir),
        task_env_config=EnvironmentConfig(),
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
        apt_mirror_enabled=False,
    )
    resource_overlay = trial_dir / "hl-docker-resources.json"
    egress_services_overlay = trial_dir / "egress-services.json"
    env._hl_resource_compose_path = resource_overlay
    env._egress_control_services_compose_path = egress_services_overlay

    paths = env._docker_compose_paths

    assert resource_overlay in paths
    assert paths.index(resource_overlay) < paths.index(
        env._DOCKER_COMPOSE_EGRESS_CONTROL_PATH
    )
    assert paths.index(resource_overlay) < paths.index(egress_services_overlay)


def test_stop_cleans_current_harbor_temporary_compose_files(monkeypatch, tmp_path):
    calls = []
    temp_roots = []
    compose_attrs = (
        ("_mounts_compose_temp_dir", "_mounts_compose_path"),
        ("_resources_compose_temp_dir", "_resources_compose_path"),
        ("_env_compose_temp_dir", "_env_compose_path"),
        (
            "_egress_control_services_compose_temp_dir",
            "_egress_control_services_compose_path",
        ),
    )

    def fake_init(self, environment_dir, *args, **kwargs):
        self._keep_containers = False
        self.logger = SimpleNamespace(
            warning=lambda *args, **kwargs: calls.append(("warning", args)),
            debug=lambda *args, **kwargs: calls.append(("debug", args)),
        )
        for temp_attr, path_attr in compose_attrs:
            handle = tempfile.TemporaryDirectory()
            root = Path(handle.name)
            path = root / "compose.json"
            path.write_text("{}")
            temp_roots.append(root)
            setattr(self, temp_attr, handle)
            setattr(self, path_attr, path)

    async def fake_prepare_logs(self):
        calls.append(("prepare_logs",))

    async def fake_compose(self, argv):
        calls.append(("compose", argv))

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "prepare_logs_for_host", fake_prepare_logs)
    monkeypatch.setattr(
        DockerEnvironment,
        "_run_docker_compose_command",
        fake_compose,
    )
    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(),
        docker_resource_enabled=False,
    )

    asyncio.run(env.stop(delete=True))

    assert ("prepare_logs",) in calls
    assert ("compose", ["down", "--remove-orphans"]) in calls
    assert all(not root.exists() for root in temp_roots)
    for temp_attr, path_attr in compose_attrs:
        assert getattr(env, temp_attr) is None
        assert getattr(env, path_attr) is None


def test_prebuilt_cache_warmup_skips_pull_with_registry_provenance(
    monkeypatch, tmp_path
):
    calls = []

    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: calls.append(("info", args)))

    async def fake_start(self, force_build):
        calls.append(("super_start", force_build))

    def fake_run(argv, **kwargs):
        calls.append(("run", argv))
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                '["alexgshaw/qemu-startup@sha256:aaaa"]',
                "",
            )
        raise AssertionError("docker pull should not run when image inspect succeeds")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/qemu-startup@sha256:aaaa",
        ),
        prebuilt_docker_pull_timeout_seconds=42,
        prebuilt_docker_hub_mirror="registry.example",
    )

    asyncio.run(env.start(force_build=False))

    assert any(
        call[0] == "run" and call[1][:3] == ["docker", "image", "inspect"]
        for call in calls
    )
    assert any(call[0] == "info" and "already available locally" in call[1][0] for call in calls)
    assert calls[-1] == ("super_start", False)


def test_prebuilt_cache_warmup_pulls_when_local_tag_has_no_registry_provenance(
    monkeypatch, tmp_path
):
    calls = []
    target = "registry.example/alexgshaw/qemu-startup:20251031"

    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(
            info=lambda *args, **kwargs: calls.append(("info", args)),
            warning=lambda *args, **kwargs: calls.append(("warning", args)),
        )

    async def fake_start(self, force_build):
        calls.append(("super_start", force_build))

    def fake_run(argv, **kwargs):
        calls.append(("run", argv, kwargs.get("timeout")))
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, "[]", "")
        if argv[:3] == ["docker", "images", "--format"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/qemu-startup:20251031",
        ),
        prebuilt_docker_pull_timeout_seconds=42,
        prebuilt_docker_hub_mirror="registry.example",
    )

    asyncio.run(env.start(force_build=False))

    assert ("run", ["docker", "pull", target], 42) in calls
    assert calls[-1] == ("super_start", False)


def test_prebuilt_registry_provenance_requires_matching_repository(monkeypatch):
    def fake_run(argv, **kwargs):
        assert argv[-2:] == ["--format", "{{json .RepoDigests}}"]
        return subprocess.CompletedProcess(
            argv,
            0,
            '["docker.io/alexgshaw/qemu-startup@sha256:aaaa"]',
            "",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert net_env._docker_image_has_registry_provenance(
        "registry.example/alexgshaw/qemu-startup:20251031",
        "alexgshaw/qemu-startup:20251031",
    ) is False
    assert net_env._docker_image_has_registry_provenance(
        "registry.example/alexgshaw/other-task:20251031"
    ) is False
    assert net_env._docker_image_has_registry_provenance(
        "alexgshaw/qemu-startup@sha256:aaaa"
    ) is True
    assert net_env._docker_image_has_registry_provenance(
        "alexgshaw/qemu-startup@sha256:bbbb"
    ) is False


def test_prebuilt_cache_warmup_tags_original_image_before_pull(monkeypatch, tmp_path):
    calls = []

    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: calls.append(("info", args)))

    async def fake_start(self, force_build):
        calls.append(("super_start", force_build))

    def fake_run(argv, **kwargs):
        calls.append(("run", argv, kwargs.get("timeout")))
        if argv[:3] == ["docker", "image", "inspect"]:
            image = argv[3]
            if image == "alexgshaw/mailman@sha256:aaaa":
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    '["alexgshaw/mailman@sha256:aaaa"]',
                    "",
                )
            if image == "registry.example/alexgshaw/mailman@sha256:aaaa" and any(
                call[1] == [
                    "docker",
                    "tag",
                    "alexgshaw/mailman@sha256:aaaa",
                    "registry.example/alexgshaw/mailman@sha256:aaaa",
                ]
                for call in calls
            ):
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    '["alexgshaw/mailman@sha256:aaaa"]',
                    "",
                )
            return subprocess.CompletedProcess(argv, 1, "", "missing")
        if argv[:2] == ["docker", "tag"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError("docker pull should not run when local fallback can be tagged")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/mailman@sha256:aaaa",
        ),
        prebuilt_docker_pull_timeout_seconds=42,
        prebuilt_docker_hub_mirror="registry.example",
    )

    asyncio.run(env.start(force_build=False))

    assert (
        "run",
        [
            "docker",
            "tag",
            "alexgshaw/mailman@sha256:aaaa",
            "registry.example/alexgshaw/mailman@sha256:aaaa",
        ],
        10,
    ) in calls
    assert any(call[0] == "info" and "local fallback" in call[1][0] for call in calls)
    assert calls[-1] == ("super_start", False)


def test_prebuilt_cache_warmup_does_not_infer_cross_registry_fallback(
    monkeypatch, tmp_path
):
    calls = []
    target = "ghcr.io/acme/task:v1"

    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(
            info=lambda *args, **kwargs: calls.append(("info", args))
        )

    async def fake_start(self, force_build):
        calls.append(("super_start", force_build))

    def fake_run(argv, **kwargs):
        calls.append(("run", argv, kwargs.get("timeout")))
        if argv[:3] == ["docker", "image", "inspect"]:
            image = argv[3]
            if image == "acme/task:v1":
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    '["docker.io/acme/task@sha256:aaaa"]',
                    "",
                )
            if image == target and any(
                call[1] == ["docker", "tag", "acme/task:v1", target]
                for call in calls
            ):
                return subprocess.CompletedProcess(argv, 0, "[]", "")
            return subprocess.CompletedProcess(argv, 1, "", "missing")
        if argv[:2] in (["docker", "tag"], ["docker", "pull"]):
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)
    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(docker_image=target),
        prebuilt_docker_pull_timeout_seconds=42,
    )

    asyncio.run(env.start(force_build=False))

    assert ("run", ["docker", "pull", target], 42) in calls
    assert all(
        call[1][:2] != ["docker", "tag"] for call in calls if call[0] == "run"
    )
    assert calls[-1] == ("super_start", False)


def test_prebuilt_cache_warmup_does_not_promote_task_built_image(
    monkeypatch, tmp_path
):
    calls = []

    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: calls.append(("info", args)))

    async def fake_start(self, force_build):
        calls.append(("super_start", force_build))

    def fake_run(argv, **kwargs):
        calls.append(("run", argv, kwargs.get("timeout")))
        if argv[:3] == ["docker", "image", "inspect"]:
            image = argv[3]
            if image == "mailman__iljz4f8-main:latest":
                return subprocess.CompletedProcess(argv, 0, "[]", "")
            if image == "registry.example/alexgshaw/mailman:20251031" and any(
                call[1] == [
                    "docker",
                    "tag",
                    "mailman__iljz4f8-main:latest",
                    "registry.example/alexgshaw/mailman:20251031",
                ]
                for call in calls
            ):
                return subprocess.CompletedProcess(argv, 0, "[]", "")
            return subprocess.CompletedProcess(argv, 1, "", "missing")
        if argv[:3] == ["docker", "images", "--format"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "other__abc-main:latest\nmailman__iljz4f8-main:latest\n",
                "",
            )
        if argv[:2] == ["docker", "tag"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/mailman:20251031",
        ),
        prebuilt_docker_pull_timeout_seconds=42,
        prebuilt_docker_hub_mirror="registry.example",
    )

    asyncio.run(env.start(force_build=False))

    assert (
        "run",
        ["docker", "pull", "registry.example/alexgshaw/mailman:20251031"],
        42,
    ) in calls
    assert all(
        call[1][:2] != ["docker", "tag"] for call in calls if call[0] == "run"
    )
    assert calls[-1] == ("super_start", False)


def test_prebuilt_cache_warmup_timeout_raises_clear_runtime_error(monkeypatch, tmp_path):
    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    async def fake_start(self, force_build):
        raise AssertionError("super().start should not run after warmup timeout")

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="pulling layer")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="alexgshaw/hf-model-inference:20251031",
        ),
        prebuilt_docker_pull_timeout_seconds=7,
        prebuilt_docker_hub_mirror="registry.example",
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(env.start(force_build=False))

    message = str(excinfo.value)
    assert "Prebuilt Docker image cache warmup timed out after 7 seconds" in message
    assert "registry.example/alexgshaw/hf-model-inference:20251031" in message
    assert "python scripts/network_preflight.py --quick" in message
    receipt = json.loads(
        (tmp_path / "trial" / PREBUILT_WARMUP_FAILURE_RECEIPT).read_text()
    )
    assert receipt == {
        "schema": PREBUILT_WARMUP_FAILURE_RECEIPT_SCHEMA,
        "kind": "prebuilt_image_cache_warmup_timeout",
        "source": "apt_mirror_docker_environment_start",
        "deterministic_access_failure": False,
    }


def test_prebuilt_cache_warmup_pulls_original_when_mirror_denies(monkeypatch, tmp_path):
    # When the configured mirror returns a deterministic access/not-found error
    # (e.g. 403 Forbidden), fall back to pulling the original Docker Hub image
    # and tag it to the effective mirror reference instead of failing the
    # environment start.
    calls = []

    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(
            info=lambda *args, **kwargs: calls.append(("info", args)),
            warning=lambda *args, **kwargs: calls.append(("warning", args)),
        )

    async def fake_start(self, force_build):
        calls.append(("super_start", force_build))

    mirror_ref = "registry.example/alexgshaw/write-compressor:20251031"
    original_ref = "alexgshaw/write-compressor:20251031"

    def fake_run(argv, **kwargs):
        calls.append(("run", tuple(argv)))
        if argv[:2] == ["docker", "pull"] and argv[2] == mirror_ref:
            raise subprocess.CalledProcessError(
                1,
                argv,
                output="",
                stderr=(
                    "Error response from daemon: unknown: failed to resolve "
                    f'reference "{mirror_ref}": unexpected status from HEAD '
                    "request: 403 Forbidden"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_exists(image):
        # Nothing is available locally until the original image is pulled+tagged.
        return False

    def fake_tag(source, target):
        calls.append(("tag", source, target))
        return True

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(net_env, "_docker_image_exists_locally", fake_exists)
    monkeypatch.setattr(net_env, "_docker_tag_image", fake_tag)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(docker_image=original_ref),
        prebuilt_docker_pull_timeout_seconds=7,
        prebuilt_docker_hub_mirror="registry.example",
    )

    asyncio.run(env.start(force_build=False))

    # Original image was pulled after the mirror denial, then tagged to the
    # effective mirror reference, and compose start proceeded.
    assert ("run", ("docker", "pull", original_ref)) in calls
    assert ("tag", original_ref, mirror_ref) in calls
    assert ("super_start", False) in calls


def test_prebuilt_cache_warmup_original_fallback_failure_reports_both(monkeypatch, tmp_path):
    # If the original-image fallback also fails, the raised error must retain
    # both the mirror and the fallback diagnostics.
    def fake_init(self, environment_dir, *args, **kwargs):
        self.environment_dir = environment_dir
        self.task_env_config = kwargs["task_env_config"]
        self.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
        self.session_id = "sess"
        self.environment_name = "env"
        self.logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )

    async def fake_start(self, force_build):
        raise AssertionError("super().start should not run when fallback fails")

    mirror_ref = "registry.example/alexgshaw/write-compressor:20251031"
    original_ref = "alexgshaw/write-compressor:20251031"

    def fake_run(argv, **kwargs):
        if argv[:2] == ["docker", "pull"] and argv[2] == mirror_ref:
            raise subprocess.CalledProcessError(
                1, argv, output="", stderr="403 Forbidden"
            )
        if argv[:2] == ["docker", "pull"] and argv[2] == original_ref:
            raise subprocess.CalledProcessError(
                1, argv, output="", stderr="original hub unreachable"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(net_env, "_docker_image_exists_locally", lambda image: False)
    monkeypatch.setattr(net_env, "_docker_tag_image", lambda s, t: False)

    env = AptMirrorDockerEnvironment(
        tmp_path,
        task_env_config=EnvironmentConfig(docker_image=original_ref),
        prebuilt_docker_pull_timeout_seconds=7,
        prebuilt_docker_hub_mirror="registry.example",
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(env.start(force_build=False))

    message = str(excinfo.value)
    assert mirror_ref in message
    assert original_ref in message
    receipt = json.loads(
        (tmp_path / "trial" / PREBUILT_WARMUP_FAILURE_RECEIPT).read_text()
    )
    assert receipt["kind"] == "prebuilt_image_cache_warmup_failure"
    assert receipt["deterministic_access_failure"] is False


def test_verifier_runtime_prepare_timeout_does_not_skip_verifier(monkeypatch, tmp_path):
    calls = []

    def fake_init(self, environment_dir, *args, **kwargs):
        self._is_windows_container = False
        self._env_paths = SimpleNamespace(tests_dir="/tests")
        self.logger = SimpleNamespace(warning=lambda *args, **kwargs: calls.append(("warn", args)))

    async def fake_exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        calls.append(("exec", command, timeout_sec, user))
        if "hl-verifier-network-prepared" in command:
            raise RuntimeError("Command timed out after 90 seconds")
        return SimpleNamespace(return_code=0, stdout="verifier ran", stderr="")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "exec", fake_exec)
    env = AptMirrorDockerEnvironment(tmp_path, task_env_config=EnvironmentConfig())

    result = asyncio.run(
        env.exec(
            command="python3 /tests/test_outputs.py",
            env={"HL_VERIFIER_NETWORK_PREPARE": "1"},
            timeout_sec=90,
        )
    )

    assert result.return_code == 0
    exec_calls = [call for call in calls if call[0] == "exec"]
    assert len(exec_calls) == 2
    assert "hl-verifier-network-prepared" in exec_calls[0][1]
    assert exec_calls[0][2] == 15
    assert exec_calls[1] == ("exec", "python3 /tests/test_outputs.py", 90, None)
    assert any(call[0] == "warn" for call in calls)


def test_verifier_detection_uses_current_harbor_environment_paths(
    monkeypatch, tmp_path
):
    def fake_init(self, environment_dir, *args, **kwargs):
        self._is_windows_container = False
        self.task_env_config = kwargs["task_env_config"]

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    env = AptMirrorDockerEnvironment(tmp_path, task_env_config=EnvironmentConfig())

    assert not hasattr(env, "_env_paths")
    assert env._should_prepare_verifier_runtime(
        "chown -R 0:0 /logs/agent", None
    ) is False
    assert env._should_prepare_verifier_runtime(
        "python3 /tests/test_outputs.py", None
    ) is True


def test_verifier_runtime_prepare_skips_when_verifier_window_is_too_small(
    monkeypatch, tmp_path
):
    calls = []

    def fake_init(self, environment_dir, *args, **kwargs):
        self._is_windows_container = False
        self._env_paths = SimpleNamespace(tests_dir="/tests")
        self.logger = SimpleNamespace(warning=lambda *args, **kwargs: calls.append(("warn", args)))

    async def fake_exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        calls.append(("exec", command, timeout_sec, user))
        return SimpleNamespace(return_code=0, stdout="verifier ran", stderr="")

    monkeypatch.setattr(DockerEnvironment, "__init__", fake_init)
    monkeypatch.setattr(DockerEnvironment, "exec", fake_exec)
    env = AptMirrorDockerEnvironment(tmp_path, task_env_config=EnvironmentConfig())

    result = asyncio.run(
        env.exec(
            command="python3 /tests/test_outputs.py",
            env={"HL_VERIFIER_NETWORK_PREPARE": "1"},
            timeout_sec=30,
        )
    )

    assert result.return_code == 0
    exec_calls = [call for call in calls if call[0] == "exec"]
    assert exec_calls == [("exec", "python3 /tests/test_outputs.py", 30, None)]
    assert any(
        "Skipping verifier runtime network preparation" in call[1][0]
        for call in calls
        if call[0] == "warn"
    )


def test_verifier_runtime_prepare_timeout_reserves_verifier_time():
    config = AptMirrorConfig(apt_timeout_seconds=30)

    assert verifier_runtime_prepare_timeout_seconds(config, verifier_timeout_sec=90) == 15
    assert verifier_runtime_prepare_timeout_seconds(config, verifier_timeout_sec=30) == 0
    assert verifier_runtime_prepare_timeout_seconds(config, verifier_timeout_sec=None) == 15


def test_patch_dockerfile_rewrites_explicit_docker_hub_domains():
    text = (
        "FROM docker.io/library/ubuntu:24.04\n"
        "RUN apt-get update\n"
        "FROM registry-1.docker.io/library/python:3.11-slim as py\n"
        "RUN pip install requests\n"
    )

    patched, changed = patch_dockerfile_for_apt_mirrors(text)

    assert changed is True
    assert "FROM docker.io/library/ubuntu:24.04" in patched
    assert "FROM registry-1.docker.io/library/python:3.11-slim as py" in patched


def test_patch_dockerfile_rewrites_configured_direct_download_urls():
    text = (
        "FROM debian:bullseye-slim\n"
        "RUN apt-get update && apt-get install -y wget ca-certificates\n"
        "RUN wget -q https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/"
        "alpine-extended-3.19.0-x86_64.iso -O /app/alpine.iso\n"
    )

    patched, changed = patch_dockerfile_for_apt_mirrors(text)

    assert changed is True
    assert DEFAULT_ALPINE_MIRROR in patched
    assert "https://dl-cdn.alpinelinux.org/alpine" in patched
    assert "/etc/hl-wgetrc" in patched
    assert "WGETRC=\"/etc/hl-wgetrc\"" in patched
    assert "retry_connrefused = on" in patched


def test_patch_dockerfile_does_not_precreate_packaged_wgetrc():
    text = (
        "FROM python:3.13-slim-bookworm\n"
        "RUN apt-get update && apt-get install -y wget\n"
    )

    patched, changed = patch_dockerfile_for_apt_mirrors(text)

    assert changed is True
    assert "> /etc/wgetrc" not in patched
    assert "> /etc/hl-wgetrc" in patched
    assert "WGETRC=\"/etc/hl-wgetrc\"" in patched


def test_patch_dockerfile_wraps_wget_with_bounded_retry():
    text = (
        "FROM python:3.13-slim-bookworm\n"
        "RUN apt-get update && apt-get install -y wget\n"
        "RUN wget -P /app/data https://example.invalid/data.parquet\n"
    )

    patched, changed = patch_dockerfile_for_apt_mirrors(text)

    assert changed is True
    assert "printf '%s\\n'" in patched
    assert "/usr/local/bin/wget" in patched
    assert "/usr/local/bin/wget.real" in patched
    assert "HL_DOWNLOAD_RETRIES" in patched
    assert "--tries=1" in patched
    assert 'grep -q -- "--waitretry"' in patched
    assert "4|5|8)" in patched
    assert patched.index("/usr/local/bin/wget") < patched.index("RUN wget -P /app/data")


def test_patch_dockerfile_can_disable_wget_retry_wrapper():
    text = (
        "FROM python:3.13-slim-bookworm\n"
        "RUN apt-get update && apt-get install -y wget\n"
        "RUN wget -P /app/data https://example.invalid/data.parquet\n"
    )

    patched, changed = patch_dockerfile_for_apt_mirrors(
        text,
        AptMirrorConfig(download_retry_wrapper=False),
    )

    assert changed is True
    assert "/usr/local/bin/wget" not in patched
    assert "/etc/hl-wgetrc" in patched


def test_verifier_runtime_prepare_rewrites_apt_sources_and_handles_locks():
    command = verifier_runtime_network_prepare_command(
        AptMirrorConfig(
            debian_mirror="http://mirror.local/debian",
            debian_security_mirror="http://mirror.local/debian-security",
            ubuntu_mirror="http://mirror.local/ubuntu",
            pypi_index_url="https://mirror.local/simple/",
            pypi_trusted_host="mirror.local",
            apt_timeout_seconds=20,
        )
    )

    assert "http://mirror.local/debian-security" in command
    assert "http://mirror.local/debian" in command
    assert "http://mirror.local/ubuntu" in command
    assert 'DPkg::Lock::Timeout "40";' in command
    assert "/var/lib/dpkg/lock-frontend" in command
    assert "hl_pkg_manager_running" in command
    assert "hl_signal_pkg_managers TERM" in command
    assert "while hl_pkg_manager_running && [ \"$hl_waited\" -lt 4 ]" in command
    assert "sleep 1; hl_waited=$((hl_waited + 1))" in command
    assert "dpkg --configure -a" not in command
    assert "apt-get update" not in command
    assert "apt-get -y -f install" not in command
    assert "apt-get install" not in command
    assert "https://mirror.local/simple/" in command
    assert "trusted-host = mirror.local" in command
    assert "/root/.local/bin/env" in command
    assert "/tmp/hl-verifier-cache/uv-python" in command
    mkdir_segment = command.split("mkdir -p ", 1)[1].split(
        " 2>/dev/null || true;", 1
    )[0]
    for cache_path in [
        "/tmp/hl-verifier-cache/uv/archive-v0",
        "/tmp/hl-verifier-cache/uv/wheels-v5",
        "/tmp/hl-verifier-cache/uv/sdists-v9",
        "/tmp/hl-verifier-cache/uv/simple-v18",
        "/tmp/hl-verifier-cache/uv/builds-v0",
        "/tmp/hl-verifier-cache/uv/interpreter-v4",
    ]:
        assert cache_path in mkdir_segment
        assert cache_path in command
    assert "chmod a+rwx /tmp/hl-verifier-cache" in command
    assert "uv==0.9.5" not in command
    assert "python3 -m pip install" not in command
    assert "python3 -m uv --version" in command
    assert "python3 -m uv tool run" in command
    assert "hl-verifier-network-prepared" in command


def test_verifier_runtime_prepare_avoids_network_installs_before_verifier():
    command = verifier_runtime_network_prepare_command(AptMirrorConfig())

    forbidden_runtime_install_steps = [
        "apt-get update",
        "apt-get -y -f install",
        "apt-get install",
        "dpkg --configure -a",
        "python3 -m pip install",
        "pip install",
        "uv==",
    ]
    for step in forbidden_runtime_install_steps:
        assert step not in command
    assert "timeout" not in command.lower().split("hl-verifier-network-prepared", 1)[0]
    assert "python3 -m uv --version" in command
    assert "python3 -m uv tool run" in command
    assert "loop stop condition" not in command


def test_rewrite_download_url_applies_longest_configured_prefix():
    config = AptMirrorConfig(
        download_url_rewrites={
            "https://example.com": "https://mirror.invalid/root",
            "https://example.com/specific": "https://mirror.invalid/specific",
        }
    )

    rewritten = rewrite_download_url("https://example.com/specific/file.tar.gz", config)

    assert rewritten == "https://mirror.invalid/specific/file.tar.gz"


def test_network_preflight_skips_daemon_mirror_check_when_unconfigured(monkeypatch):
    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://wrong.mirror/"]', "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(timeout=1)

    assert report["ok"] is True
    assert report["failed_checks"] == []
    mirror_check = next(check for check in report["checks"] if check["name"] == "docker_mirror")
    assert mirror_check["ok"] is True
    assert mirror_check["detail"] == "no Docker registry mirror configured"


def test_network_preflight_downgrades_v2_probe_when_docker_pull_succeeds(monkeypatch):
    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://wrong.mirror/"]', "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=False,
            detail=f"{url} failed",
            repair="Use a reachable Docker registry mirror.",
        ),
    )

    report = network_preflight.run_preflight(timeout=1)

    assert report["ok"] is True
    assert report["failed_checks"] == []
    v2 = next(check for check in report["checks"] if check["name"] == "docker_mirror_v2")
    assert v2["metadata"]["downgraded_to_diagnostic"] is True
    assert "docker_pull:ubuntu:24.04" in v2["metadata"]["docker_pull_successes"]


def test_network_preflight_fails_when_registry_mirror_and_pulls_fail(monkeypatch):
    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://wrong.mirror/"]', "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 1, "", "pull failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=False,
            detail=f"{url} failed",
            repair="Use a reachable Docker registry mirror.",
        ),
    )

    report = network_preflight.run_preflight(
        timeout=1,
        docker_images=["ubuntu:24.04"],
        prebuilt_docker_pull_images=[],
    )

    assert report["ok"] is False
    assert report["failed_checks"] == ["docker_mirror_v2", "docker_pull:ubuntu:24.04"]
    assert "reachable Docker registry mirror" in report["repair"]


def test_network_preflight_checks_configured_direct_download_rewrite(monkeypatch):
    checked_urls = []

    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.m.daocloud.io/"]', "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    def fake_head(name, url, *, timeout):
        checked_urls.append((name, url))
        return network_preflight.CheckResult(name=name, ok=True, detail=url)

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(network_preflight, "check_http_head", fake_head)
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(timeout=1, skip_docker_pull=True)

    assert report["ok"] is True
    assert (
        "debian_package_mirror",
        "https://deb.debian.org/debian/dists/stable/InRelease",
    ) in checked_urls
    assert (
        "alpine_release_mirror",
        "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/"
        "alpine-extended-3.19.0-x86_64.iso",
    ) in checked_urls


def test_network_preflight_pulls_default_task_base_images(monkeypatch):
    pulled = []
    manifests = []

    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.m.daocloud.io/"]', "")
        if argv[:2] == ["docker", "pull"]:
            pulled.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        if argv[:3] == ["docker", "manifest", "inspect"]:
            manifests.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "manifest", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(timeout=1)

    assert report["ok"] is True
    assert "ubuntu:24.04" in pulled
    assert "python:3.13-slim-bookworm" in pulled
    assert "debian:13.0-slim" in pulled
    assert "alexgshaw/qemu-startup:20251031" in manifests
    assert "alexgshaw/custom-memory-heap-crash:20251031" in manifests
    assert "alexgshaw/hf-model-inference:20251031" in manifests
    assert "alexgshaw/hf-model-inference:20251031" not in pulled
    assert report["prebuilt_docker_pull_default_opt_in"] is True


def test_network_preflight_uses_configured_prebuilt_pull_images(monkeypatch):
    pulled = []

    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.m.daocloud.io/"]', "")
        if argv[:2] == ["docker", "pull"]:
            pulled.append((argv[-1], timeout))
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        if argv[:3] == ["docker", "manifest", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, "manifest", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(
        timeout=5,
        docker_images=[],
        prebuilt_docker_images=[],
        prebuilt_docker_pull_images=["alexgshaw/hf-model-inference:20251031"],
        prebuilt_docker_pull_timeout=1800,
    )

    assert report["ok"] is True
    assert pulled == [("alexgshaw/hf-model-inference:20251031", 1800)]


def test_network_preflight_respects_shorter_prebuilt_pull_timeout(monkeypatch):
    pulled = []

    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.m.daocloud.io/"]', "")
        if argv[:2] == ["docker", "pull"]:
            pulled.append((argv[-1], timeout))
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        if argv[:3] == ["docker", "manifest", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, "manifest", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(
        timeout=180,
        docker_images=[],
        prebuilt_docker_images=[],
        prebuilt_docker_pull_images=["alexgshaw/hf-model-inference:20251031"],
        prebuilt_docker_pull_timeout=60,
    )

    assert report["ok"] is True
    assert pulled == [("alexgshaw/hf-model-inference:20251031", 60)]


def test_network_preflight_reports_timeout_bytes_without_crashing(monkeypatch):
    def fake_subprocess_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(
            argv,
            kwargs["timeout"],
            output=b"partial pull output",
            stderr=b"partial pull error",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    result = network_preflight.check_docker_pull(
        "docker.1panel.live/alexgshaw/hf-model-inference:20251031",
        timeout=1,
        name="prebuilt_docker_pull:alexgshaw/hf-model-inference:20251031",
    )

    assert result.ok is False
    assert "partial pull output" in result.detail
    assert "timed out after 1s" in result.detail
    assert "partial pull error" in result.detail


def test_network_preflight_downgrades_prebuilt_cache_pull_when_manifest_succeeds(monkeypatch):
    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.m.daocloud.io/"]', "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 124, "", "timed out")
        if argv[:3] == ["docker", "manifest", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, "manifest", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(
        timeout=1,
        docker_images=[],
        prebuilt_docker_images=["alexgshaw/hf-model-inference:20251031"],
        prebuilt_docker_pull_images=["alexgshaw/hf-model-inference:20251031"],
    )

    assert report["ok"] is True
    pull = next(
        check
        for check in report["checks"]
        if check["name"] == "prebuilt_docker_pull:alexgshaw/hf-model-inference:20251031"
    )
    assert pull["ok"] is True
    assert pull["metadata"]["downgraded_to_cache_warning"] is True
    assert "non-blocking" in pull["detail"]


def test_network_preflight_manifests_pull_only_prebuilt_images_before_downgrade(monkeypatch):
    manifests = []

    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.m.daocloud.io/"]', "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 124, "", "timed out")
        if argv[:3] == ["docker", "manifest", "inspect"]:
            manifests.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "manifest", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(
        timeout=1,
        docker_images=[],
        prebuilt_docker_images=[],
        prebuilt_docker_pull_images=["alexgshaw/mteb-leaderboard:20251031"],
    )

    assert report["ok"] is True
    assert manifests == ["alexgshaw/mteb-leaderboard:20251031"]
    manifest = next(
        check
        for check in report["checks"]
        if check["name"] == "prebuilt_docker_manifest:alexgshaw/mteb-leaderboard:20251031"
    )
    pull = next(
        check
        for check in report["checks"]
        if check["name"] == "prebuilt_docker_pull:alexgshaw/mteb-leaderboard:20251031"
    )
    assert manifest["ok"] is True
    assert pull["ok"] is True
    assert pull["metadata"]["downgraded_to_cache_warning"] is True


def test_network_preflight_blocks_prebuilt_pull_when_manifest_fails(monkeypatch):
    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.m.daocloud.io/"]', "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(argv, 124, "", "timed out")
        if argv[:3] == ["docker", "manifest", "inspect"]:
            return subprocess.CompletedProcess(argv, 1, "", "manifest failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(
        timeout=1,
        docker_images=[],
        prebuilt_docker_images=["alexgshaw/hf-model-inference:20251031"],
        prebuilt_docker_pull_images=["alexgshaw/hf-model-inference:20251031"],
    )

    assert report["ok"] is False
    assert "prebuilt_docker_manifest:alexgshaw/hf-model-inference:20251031" in report[
        "failed_checks"
    ]
    assert "prebuilt_docker_pull:alexgshaw/hf-model-inference:20251031" in report[
        "failed_checks"
    ]


def test_network_preflight_pulls_effective_image_overrides(monkeypatch):
    pulled = []

    def fake_run(argv, *, timeout):
        if argv[:2] == ["docker", "info"] and "--format" not in argv:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 0, '["https://docker.1ms.run/"]', "")
        if argv[:2] == ["docker", "pull"]:
            pulled.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "pulled", "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)
    monkeypatch.setattr(
        network_preflight,
        "check_http_head",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )
    monkeypatch.setattr(
        network_preflight,
        "check_registry_v2",
        lambda name, url, *, timeout: network_preflight.CheckResult(
            name=name,
            ok=True,
            detail=url,
        ),
    )

    report = network_preflight.run_preflight(
        timeout=1,
        docker_mirror="https://docker.1ms.run",
        docker_hub_mirror="docker.1ms.run",
        docker_images=["debian:13.0-slim"],
        prebuilt_docker_pull_images=[],
        docker_image_overrides={
            "debian:13.0-slim": "docker.m.daocloud.io/library/debian:13.0-slim",
        },
    )

    assert report["ok"] is True
    assert pulled == ["docker.m.daocloud.io/library/debian:13.0-slim"]


def test_full_preflight_ubuntu_download_uses_http_apt_mirror(monkeypatch):
    captured = {}

    def fake_run(argv, *, timeout):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(network_preflight, "_run", fake_run)

    result = network_preflight.check_ubuntu_container_download(
        ubuntu_mirror="http://mirrors.aliyun.com/ubuntu",
        mirror_config=AptMirrorConfig(),
        timeout=1,
    )

    assert result.ok is True
    assert captured["argv"][:3] == ["docker", "run", "--rm"]
    assert "ubuntu:24.04" in captured["argv"]
    script = captured["argv"][-1]
    assert "http://mirrors.aliyun.com/ubuntu" in script
    assert "https://mirrors.aliyun.com/ubuntu" not in script
    assert "ca-certificates" in script
