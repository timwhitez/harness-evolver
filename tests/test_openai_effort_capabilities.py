"""Tests for Issue #171: OpenAI model-aware reasoning effort & Responses capability validation."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import litellm
import pytest
from pydantic import ValidationError

from bench.agent import HLAgent
from harness.config import HarnessConfig, ModelsConfig, ReasoningConfig, RoleModelConfig
from harness.tools.registry import ToolRegistry
from hl.model_scope import model_scope_from_config, model_scope_matches


class _LoopbackCaptureServer:
    """In-process local loopback HTTP server capturing exact outbound Chat/Responses JSON."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                body = json.loads(raw) if raw else {}
                outer.requests.append({"path": self.path, "body": body})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if "/responses" in self.path:
                    resp = {
                        "id": "resp_local_1",
                        "object": "response",
                        "created_at": 1700000000,
                        "status": "completed",
                        "error": None,
                        "incomplete_details": None,
                        "instructions": None,
                        "max_output_tokens": None,
                        "model": body.get("model", "gpt-6.2-astra"),
                        "output": [
                            {
                                "id": "fc_local_1",
                                "type": "function_call",
                                "call_id": "call_local_1",
                                "name": "ping",
                                "arguments": "{}",
                                "status": "completed",
                            }
                        ],
                        "parallel_tool_calls": True,
                        "previous_response_id": None,
                        "reasoning": body.get("reasoning") or {"effort": None, "summary": None},
                        "temperature": 1.0,
                        "text": {"format": {"type": "text"}},
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "truncation": "disabled",
                        "usage": {
                            "input_tokens": 10,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 4,
                            "output_tokens_details": {"reasoning_tokens": 2},
                            "total_tokens": 14,
                        },
                        "user": None,
                        "metadata": {},
                    }
                else:
                    resp = {
                        "id": "chatcmpl_local_1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": body.get("model", "gpt-5.5"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "ok",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                self.wfile.write(json.dumps(resp).encode("utf-8"))

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def loopback_server():
    server = _LoopbackCaptureServer()
    try:
        yield server
    finally:
        server.close()


class TestOpenAIEffortValidationMatrix:
    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            # GPT-6.2 accepts low..max and omitted (None)
            ("gpt-6.2-astra", None),
            ("gpt-6.2-astra", "low"),
            ("gpt-6.2-astra", "medium"),
            ("gpt-6.2-astra", "high"),
            ("gpt-6.2-astra", "xhigh"),
            ("gpt-6.2-astra", "max"),
            ("gpt-6.2-sol", "max"),
            # GPT-5 base/mini/nano accept minimal..high and omitted (None)
            ("gpt-5", None),
            ("gpt-5", "minimal"),
            ("gpt-5", "low"),
            ("gpt-5", "medium"),
            ("gpt-5", "high"),
            ("gpt-5-mini", "minimal"),
            ("gpt-5-nano", "minimal"),
            # GPT-5.1 accepts none..high
            ("gpt-5.1", None),
            ("gpt-5.1", "none"),
            ("gpt-5.1", "low"),
            ("gpt-5.1", "medium"),
            ("gpt-5.1", "high"),
            # GPT-5.2 / 5.4 / 5.5 / 5.6 accept none..xhigh
            ("gpt-5.2", "none"),
            ("gpt-5.2", "xhigh"),
            ("gpt-5.4", "none"),
            ("gpt-5.4", "xhigh"),
            ("gpt-5.5", None),
            ("gpt-5.5", "none"),
            ("gpt-5.5", "medium"),
            ("gpt-5.5", "xhigh"),
            ("gpt-5.6", "none"),
            # GPT-5.5-pro accepts medium..xhigh and omitted
            ("gpt-5.5-pro", None),
            ("gpt-5.5-pro", "medium"),
            ("gpt-5.5-pro", "xhigh"),
            # Mandatory reasoning O-series accepts low..high and omitted
            ("o3", None),
            ("o3", "low"),
            ("o3", "high"),
            ("o4-mini", None),
            ("o4-mini", "medium"),
            # Non-reasoning models accept omitted effort
            ("gpt-4o", None),
            ("gpt-4.1", None),
        ],
    )
    def test_supported_openai_model_effort_combinations_accepted(
        self, model: str, effort: str | None
    ) -> None:
        reasoning = {"effort": effort} if effort is not None else {}
        cfg = RoleModelConfig(provider="openai", model=model, reasoning=reasoning)
        assert cfg.reasoning.effort == effort

    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            # Mandatory-thinking GPT-6.2 rejects none and minimal (never maps none -> low)
            ("gpt-6.2-astra", "none"),
            ("gpt-6.2-astra", "minimal"),
            ("gpt-6.2-sol", "none"),
            # Mandatory-thinking O-series rejects none, minimal, xhigh, max
            ("o3", "none"),
            ("o3", "minimal"),
            ("o3", "xhigh"),
            ("o3", "max"),
            ("o4-mini", "none"),
            ("o4-mini", "max"),
            ("o1", "none"),
            # GPT-5 base rejects none, xhigh, max
            ("gpt-5", "none"),
            ("gpt-5", "xhigh"),
            ("gpt-5", "max"),
            # GPT-5.1 rejects minimal, xhigh, max
            ("gpt-5.1", "minimal"),
            ("gpt-5.1", "xhigh"),
            ("gpt-5.1", "max"),
            # GPT-5.4 / 5.5 do NOT blindly accept max or minimal
            ("gpt-5.4", "max"),
            ("gpt-5.4", "minimal"),
            ("gpt-5.5", "max"),
            ("gpt-5.5", "minimal"),
            # GPT-5.5-pro rejects none, minimal, low, max
            ("gpt-5.5-pro", "none"),
            ("gpt-5.5-pro", "low"),
            ("gpt-5.5-pro", "max"),
            # Non-reasoning OpenAI models reject any explicit effort including none
            ("gpt-4o", "none"),
            ("gpt-4o", "low"),
            ("gpt-4.1", "high"),
            # Unknown OpenAI model must NOT blindly open max or minimal
            ("gpt-unknown-native", "max"),
            ("gpt-unknown-native", "minimal"),
        ],
    )
    def test_unsupported_openai_model_effort_combinations_rejected(
        self, model: str, effort: str
    ) -> None:
        with pytest.raises(ValidationError):
            RoleModelConfig(
                provider="openai",
                model=model,
                reasoning={"effort": effort},
            )

    def test_explicit_none_vs_omitted_in_litellm_kwargs(self) -> None:
        omitted = RoleModelConfig(provider="openai", model="gpt-5.5")
        explicit_none = RoleModelConfig(
            provider="openai",
            model="gpt-5.5",
            reasoning={"effort": "none"},
        )
        assert omitted.reasoning.effort is None
        assert explicit_none.reasoning.effort == "none"

        assert omitted.litellm_kwargs() == {"model": "gpt-5.5"}
        assert explicit_none.litellm_kwargs() == {
            "model": "gpt-5.5",
            "reasoning_effort": "none",
        }

    def test_responses_and_tool_capability_validation_before_send(self) -> None:
        # Responses-only model rejects explicit api="chat" at config validation time
        with pytest.raises(ValidationError):
            RoleModelConfig(
                provider="openai",
                model="gpt-5.5-pro",
                api="chat",
                reasoning={"effort": "high"},
            )

        # GPT-6.2 with tools auto-routes to Responses path preserving max
        astra_auto = RoleModelConfig(
            provider="openai",
            model="gpt-6.2-astra",
            reasoning={"effort": "max"},
        )
        kw_tools = astra_auto.litellm_kwargs(with_tools=True)
        assert kw_tools["model"] == "openai/responses/gpt-6.2-astra"
        assert kw_tools["reasoning_effort"] == {"effort": "max"}
        assert "reasoning_effort" in kw_tools["allowed_openai_params"]

        # GPT-6.2 pinned to api="chat" MUST reject tool requests before sending
        astra_chat = RoleModelConfig(
            provider="openai",
            model="gpt-6.2-astra",
            api="chat",
            reasoning={"effort": "max"},
        )
        assert astra_chat.litellm_kwargs(with_tools=False)["model"] == "gpt-6.2-astra"
        with pytest.raises(ValueError, match="Responses API"):
            astra_chat.litellm_kwargs(with_tools=True)

        # Deep-research model without function calling rejects tools before sending
        deep = RoleModelConfig(
            provider="openai",
            model="o3-deep-research",
            reasoning={"effort": "medium"},
        )
        with pytest.raises(ValueError, match="tool/function calling"):
            deep.litellm_kwargs(with_tools=True)


class TestOutboundLoopbackTransportPayloads:
    """Verify actual outbound Chat & Responses HTTP JSON via installed LiteLLM over loopback."""

    def test_openai_chat_and_responses_outbound_json_via_litellm(
        self,
        loopback_server: _LoopbackCaptureServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # 1. gpt-5.5 + explicit none -> Chat Completions JSON preserves reasoning_effort="none"
        loopback_server.requests.clear()
        cfg_none = RoleModelConfig(
            provider="openai",
            base_url=loopback_server.base_url,
            api_key_env="TEST_OPENAI_KEY",
            model="gpt-5.5",
            reasoning={"effort": "none"},
        )
        litellm.completion(
            **cfg_none.litellm_kwargs(),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/chat/completions"
        assert req["body"]["model"] == "gpt-5.5"
        assert req["body"]["reasoning_effort"] == "none"

        # 2. gpt-5.5 + omitted effort -> Chat Completions JSON omits reasoning_effort
        loopback_server.requests.clear()
        cfg_omitted = RoleModelConfig(
            provider="openai",
            base_url=loopback_server.base_url,
            api_key_env="TEST_OPENAI_KEY",
            model="gpt-5.5",
        )
        litellm.completion(
            **cfg_omitted.litellm_kwargs(),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/chat/completions"
        assert "reasoning_effort" not in req["body"]

        # 3. gpt-5 + minimal -> Chat Completions JSON preserves reasoning_effort="minimal"
        loopback_server.requests.clear()
        cfg_minimal = RoleModelConfig(
            provider="openai",
            base_url=loopback_server.base_url,
            api_key_env="TEST_OPENAI_KEY",
            model="gpt-5",
            reasoning={"effort": "minimal"},
        )
        litellm.completion(
            **cfg_minimal.litellm_kwargs(),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/chat/completions"
        assert req["body"]["reasoning_effort"] == "minimal"

        # 4. gpt-6.2-astra + max + tools via Worker bridge -> POST /v1/responses with reasoning.effort="max"
        loopback_server.requests.clear()
        cfg_astra = RoleModelConfig(
            provider="openai",
            base_url=loopback_server.base_url,
            api_key_env="TEST_OPENAI_KEY",
            model="gpt-6.2-astra",
            reasoning={"effort": "max"},
            max_output_tokens=256,
        )
        agent = HLAgent(
            config=HarnessConfig.create_default(),
            tool_registry=ToolRegistry(),
            role_config=cfg_astra,
        )
        bridge_kwargs = agent._completion_kwargs_for_bridge(
            messages=[{"role": "user", "content": "run ping"}],
            tool_schemas=tool_schemas,
        )
        resp = litellm.completion(**bridge_kwargs)
        payload = agent._llm_response_payload(resp)
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/responses"
        assert req["body"]["model"] == "gpt-6.2-astra"
        assert req["body"]["reasoning"] == {"effort": "max"}
        assert req["body"]["max_output_tokens"] == 256
        assert req["body"]["tool_choice"] == "auto"
        assert len(req["body"]["tools"]) == 1
        assert payload["message"]["tool_calls"][0]["name"] == "ping"

        # 5. gpt-5.5 + explicit none on Responses API -> POST /v1/responses with reasoning.effort="none"
        loopback_server.requests.clear()
        cfg_resp_none = RoleModelConfig(
            provider="openai",
            api="responses",
            base_url=loopback_server.base_url,
            api_key_env="TEST_OPENAI_KEY",
            model="gpt-5.5",
            reasoning={"effort": "none"},
        )
        litellm.completion(
            **cfg_resp_none.litellm_kwargs(with_tools=True),
            messages=[{"role": "user", "content": "hi"}],
            tools=tool_schemas,
            tool_choice="auto",
        )
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/responses"
        assert req["body"]["reasoning"] == {"effort": "none"}

        # 6. gpt-6.2-astra + omitted effort + tools -> POST /v1/responses with NO reasoning key
        loopback_server.requests.clear()
        cfg_astra_omitted = RoleModelConfig(
            provider="openai",
            base_url=loopback_server.base_url,
            api_key_env="TEST_OPENAI_KEY",
            model="gpt-6.2-astra",
        )
        litellm.completion(
            **cfg_astra_omitted.litellm_kwargs(with_tools=True),
            messages=[{"role": "user", "content": "hi"}],
            tools=tool_schemas,
            tool_choice="auto",
        )
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/responses"
        assert "reasoning" not in req["body"]

    def test_compatible_custom_gateway_preserves_declared_semantics(
        self,
        loopback_server: _LoopbackCaptureServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_GW_KEY", "gw-secret")

        # Compatible gateway with explicit none -> sends reasoning_effort="none", no thinking override
        loopback_server.requests.clear()
        gw_none = RoleModelConfig(
            provider="openai_compatible",
            base_url=loopback_server.base_url,
            api_key_env="TEST_GW_KEY",
            model="deepseek-flash",
            reasoning={"effort": "none"},
        )
        litellm.completion(
            **gw_none.litellm_kwargs(),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/chat/completions"
        assert req["body"]["reasoning_effort"] == "none"
        assert "thinking" not in req["body"]

        # Compatible gateway with omitted effort -> omits reasoning_effort and thinking
        loopback_server.requests.clear()
        gw_omitted = RoleModelConfig(
            provider="openai_compatible",
            base_url=loopback_server.base_url,
            api_key_env="TEST_GW_KEY",
            model="custom-proxy-model",
        )
        litellm.completion(
            **gw_omitted.litellm_kwargs(),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert len(loopback_server.requests) == 1
        assert "reasoning_effort" not in loopback_server.requests[0]["body"]

        # Compatible gateway with max on DeepSeek -> preserves reasoning_effort="max" & thinking enabled
        loopback_server.requests.clear()
        gw_deepseek_max = RoleModelConfig(
            provider="openai_compatible",
            base_url=loopback_server.base_url,
            api_key_env="TEST_GW_KEY",
            model="deepseek-v4-pro",
            reasoning={"effort": "max"},
        )
        litellm.completion(
            **gw_deepseek_max.litellm_kwargs(),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert len(loopback_server.requests) == 1
        req = loopback_server.requests[0]
        assert req["path"] == "/v1/chat/completions"
        assert req["body"]["reasoning_effort"] == "max"
        assert req["body"]["thinking"] == {"type": "enabled"}


class TestRoleConfigSerializationIdentityRegressions:
    def test_repo_models_yaml_old_roles_load_and_roundtrip(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        models = ModelsConfig.from_yaml(repo_root / "config" / "models.yaml")

        for role_name in ("worker", "worker_gpt", "worker_deepseek", "orchestrator", "validator"):
            role = models.get_role(role_name)
            dumped = role.model_dump(mode="json")
            restored = RoleModelConfig.model_validate(dumped)
            assert restored == role
            assert restored.model_dump(mode="json") == dumped

    def test_new_role_configs_roundtrip_and_preserve_distinct_identity(self) -> None:
        omitted = RoleModelConfig(provider="openai", model="gpt-5.5")
        explicit_none = RoleModelConfig(
            provider="openai",
            model="gpt-5.5",
            reasoning={"effort": "none"},
        )
        astra_max = RoleModelConfig(
            provider="openai",
            model="gpt-6.2-astra",
            reasoning={"effort": "max"},
        )
        gpt5_minimal = RoleModelConfig(
            provider="openai",
            model="gpt-5",
            reasoning={"effort": "minimal"},
        )

        for cfg in (omitted, explicit_none, astra_max, gpt5_minimal):
            dumped = cfg.model_dump(mode="json")
            restored = RoleModelConfig.model_validate(dumped)
            assert restored == cfg
            assert restored.reasoning.effort == cfg.reasoning.effort

        # Omitted effort vs explicit "none" must NOT serialize identically or collide in scope!
        assert omitted.model_dump(mode="json") != explicit_none.model_dump(mode="json")
        assert omitted.redacted() != explicit_none.redacted()
        assert omitted.redacted()["reasoning"]["effort"] is None
        assert explicit_none.redacted()["reasoning"]["effort"] == "none"

        scope_omitted = model_scope_from_config(
            {
                "provider": omitted.provider,
                "model": omitted.model,
                "reasoning_effort": omitted.reasoning.effort,
            }
        )
        scope_none = model_scope_from_config(
            {
                "provider": explicit_none.provider,
                "model": explicit_none.model,
                "reasoning_effort": explicit_none.reasoning.effort,
            }
        )
        assert scope_omitted != scope_none
        assert model_scope_matches(scope_omitted, scope_none) is False

    def test_harbor_adapter_preserves_omitted_vs_explicit_none(self, tmp_path: Path) -> None:
        from bench.harbor_adapter import HLWorkerHarborAgent

        agent_omitted = HLWorkerHarborAgent(
            logs_dir=tmp_path / "logs_omitted",
            model_name="gpt-6.2-astra",
            provider="openai",
            reasoning_effort=None,
        )
        built_omitted = agent_omitted._build_agent(ToolRegistry())
        assert built_omitted.role_config is not None
        assert built_omitted.role_config.reasoning.effort is None

        agent_none = HLWorkerHarborAgent(
            logs_dir=tmp_path / "logs_none",
            model_name="gpt-5.5",
            provider="openai",
            reasoning_effort="none",
        )
        built_none = agent_none._build_agent(ToolRegistry())
        assert built_none.role_config is not None
        assert built_none.role_config.reasoning.effort == "none"
        assert built_none.role_config.litellm_kwargs()["reasoning_effort"] == "none"

    def test_reasoning_config_default_effort_is_unspecified(self) -> None:
        rc = ReasoningConfig()
        assert rc.effort is None
