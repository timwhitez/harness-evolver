from types import SimpleNamespace

from scripts.setup_wizard import build_env_example, build_local_config, worker_api_key_checks


def test_setup_wizard_preserves_existing_multi_provider_roles():
    args = SimpleNamespace(
        task_path="terminal-bench-tasks/terminal-bench",
        worker_provider="openai_compatible",
        worker_base_url="https://api.deepseek.com/v1",
        worker_api_key_env="DEEPSEEK_API_KEY",
        worker_model="deepseek-v4-pro",
        worker_reasoning_effort="max",
        worker_timeout_seconds=900,
        worker_max_retries=5,
        codex_model="gpt-5.4",
        codex_sandbox="danger-full-access",
        codex_timeout_seconds=None,
        submit_threshold=0.80,
        submit_visibility="private",
        submit_share_org=["TimWhite-AGI"],
        submit_share_user=["timwhitez"],
        submit_share_yes=True,
    )
    existing = {
        "models": {
            "roles": {
                "worker": {
                    "provider": "openai_compatible",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "model": "deepseek-v4-pro",
                    "reasoning": {"effort": "none"},
                },
                "worker_gpt": {
                    "provider": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-5.4",
                    "reasoning": {"effort": "xhigh"},
                },
            }
        }
    }

    config = build_local_config(args, existing_config=existing)

    assert config["models"]["roles"]["worker"]["model"] == "deepseek-v4-pro"
    assert config["models"]["roles"]["worker_gpt"]["base_url"] == (
        "https://api.openai.com/v1"
    )
    assert config["models"]["roles"]["orchestrator"]["runner"] == "codex"
    assert config["models"]["roles"]["orchestrator"]["model"] == "gpt-5.4"
    assert config["models"]["roles"]["orchestrator"]["sandbox"] == "danger-full-access"
    assert config["models"]["roles"]["orchestrator"]["timeout_seconds"] is None
    assert config["benchmark"]["path"] == "terminal-bench-tasks/terminal-bench"
    assert config["submit"]["enabled"] is False
    assert config["submit"]["trigger_score"] == 0.80
    assert config["submit"]["share_orgs"] == ["TimWhite-AGI"]
    assert config["submit"]["share_users"] == ["timwhitez"]
    assert config["submit"]["share_yes"] is True


def test_setup_wizard_env_example_lists_role_env_names_without_secrets():
    args = SimpleNamespace(worker_api_key_env="OPENAI_API_KEY")
    config = {
        "models": {
            "roles": {
                "worker": {"api_key_env": "DEEPSEEK_API_KEY"},
                "worker_gpt": {"api_key_env": "OPENAI_API_KEY"},
            }
        }
    }

    env_example = build_env_example(args, config)

    assert "DEEPSEEK_API_KEY=\n" in env_example
    assert "OPENAI_API_KEY=\n" in env_example
    assert "CODEX_API_KEY=\n" in env_example
    assert "sk-" not in env_example


def test_setup_wizard_checks_default_worker_api_key_without_printing_secret(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-secret")
    config = {
        "models": {
            "roles": {
                "worker": {"api_key_env": "DEEPSEEK_API_KEY"},
                "worker_gpt": {"api_key_env": "OPENAI_API_KEY"},
            }
        }
    }

    checks = worker_api_key_checks(config)

    assert checks == [
        {
            "name": "api_key:DEEPSEEK_API_KEY",
            "ok": True,
            "detail": "DEEPSEEK_API_KEY=<redacted>",
            "repair": "Set DEEPSEEK_API_KEY in .env.local or the process environment.",
        }
    ]
    assert "sk-test-secret" not in str(checks)


def test_setup_wizard_reports_missing_default_worker_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = {"models": {"roles": {"worker": {"api_key_env": "DEEPSEEK_API_KEY"}}}}

    checks = worker_api_key_checks(config)

    assert checks[0]["ok"] is False
    assert checks[0]["detail"] == "DEEPSEEK_API_KEY missing"
