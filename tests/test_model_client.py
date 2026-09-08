from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from werewolf_game.core.errors import ModelClientError
from werewolf_game.agents.llm.client import (
    ModelClient,
    build_model_payload,
    extract_json,
    extract_tool_calls,
    extract_token_usage,
    get_model_config,
    is_model_configured,
    serialize_tool_result,
)


class ModelClientTest(unittest.TestCase):
    @staticmethod
    def configured_client(**overrides: object) -> ModelClient:
        config = get_model_config(
            "api",
            {
                "OPENAI_MODEL": "demo-model",
                "OPENAI_BASE_URL": "https://example.test/v1/",
                "OPENAI_API_KEY": "server-key",
                "MODEL_API_MODE": "responses",
            },
        )
        return ModelClient(replace(config, **overrides))

    def test_responses_configuration_matches_reference_environment_names(self) -> None:
        config = get_model_config(
            "api",
            {
                "OPENAI_MODEL": "demo-model",
                "OPENAI_BASE_URL": "https://example.test/v1/",
                "OPENAI_API_KEY": "server-key",
                "MODEL_API_MODE": "responses",
                "MODEL_API_PATH": "/responses",
            },
        )
        self.assertEqual(config.protocol, "responses")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertTrue(is_model_configured(config))
        payload = build_model_payload(
            "system", [{"role": "user", "content": "hello"}], 500, config
        )
        self.assertEqual(payload["model"], "demo-model")
        self.assertEqual(payload["instructions"], "system")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["input"], [{"role": "user", "content": "hello"}])
        self.assertNotIn("reasoning", payload)

    def test_task_and_meta_profiles_use_separate_model_settings(self) -> None:
        env = {
            "OPENAI_MODEL": "gpt-5.5",
            "OPENAI_BASE_URL": "https://example.test/v1/",
            "OPENAI_API_KEY": "server-key",
            "MODEL_API_MODE": "responses",
            "WEREWOLF_TASK_MODEL": "deepseek-v4-flash",
            "WEREWOLF_TASK_REASONING_EFFORT": "none",
            "WEREWOLF_META_MODEL": "gpt-5.5",
        }
        task = get_model_config("api", env, profile="task")
        meta = get_model_config("api", env, profile="meta")

        self.assertEqual(task.model, "deepseek-v4-flash")
        self.assertEqual(task.reasoning_effort, "none")
        self.assertEqual(meta.model, "gpt-5.5")
        self.assertIsNone(meta.reasoning_effort)

        task_payload = build_model_payload("system", [], 500, task)
        meta_payload = build_model_payload("system", [], 500, meta)
        self.assertEqual(task_payload["reasoning"], {"effort": "none"})
        self.assertNotIn("reasoning", meta_payload)

        messages_task = get_model_config(
            "api",
            {**env, "MODEL_API_MODE": "messages"},
            profile="task",
        )
        messages_meta = get_model_config(
            "api",
            {**env, "MODEL_API_MODE": "messages"},
            profile="meta",
        )
        self.assertEqual(
            build_model_payload("system", [], 500, messages_task)["thinking"],
            {"type": "disabled"},
        )
        self.assertNotIn("thinking", build_model_payload("system", [], 500, messages_meta))

        chat_task = get_model_config(
            "api",
            {**env, "MODEL_API_MODE": "chat_completions"},
            profile="task",
        )
        chat_meta = get_model_config(
            "api",
            {**env, "MODEL_API_MODE": "chat_completions"},
            profile="meta",
        )
        self.assertEqual(
            build_model_payload("system", [], 500, chat_task)["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("chat_template_kwargs", build_model_payload("system", [], 500, chat_meta))

    def test_json_extraction_accepts_fenced_and_embedded_output(self) -> None:
        self.assertEqual(extract_json('```json\n{"kind":"pass"}\n```'), {"kind": "pass"})
        self.assertEqual(
            extract_json('Explanation: {"kind":"day_vote","target_id":"p2"}'),
            {"kind": "day_vote", "target_id": "p2"},
        )

    def test_token_usage_supports_responses_and_chat_field_names(self) -> None:
        self.assertEqual(
            extract_token_usage(
                {"usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}}
            ),
            {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        )
        self.assertEqual(
            extract_token_usage(
                {"usage": {"prompt_tokens": 20, "completion_tokens": 5}}
            ),
            {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        )
        self.assertEqual(extract_token_usage({"output_text": "{}"}), {})

    def test_usage_limit_retries_immediately_for_account_pool_routing(self) -> None:
        client = self.configured_client(max_retries=0, usage_limit_retries=2)
        limited = ModelClientError("The usage limit has been reached")
        limited.status_code = 429  # type: ignore[attr-defined]
        responses: list[object] = [limited, limited, {"output_text": '{"kind":"pass"}'}]

        def request(_: object) -> dict:
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value  # type: ignore[return-value]

        with (
            patch.object(client, "_request_json", side_effect=request) as mocked_request,
            patch("werewolf_game.agents.llm.client.time.sleep") as mocked_sleep,
        ):
            result = client.complete_json(system="system", messages=[], max_tokens=32)

        self.assertEqual(result, {"kind": "pass"})
        self.assertEqual(mocked_request.call_count, 3)
        mocked_sleep.assert_not_called()

    def test_invalid_model_json_uses_the_generic_retry_policy(self) -> None:
        client = self.configured_client(max_retries=1, usage_limit_retries=0)
        responses: list[dict] = [
            {"output_text": "这不是 JSON"},
            {"output_text": '{"kind":"pass"}'},
        ]

        with (
            patch.object(client, "_request_json", side_effect=responses) as mocked_request,
            patch("werewolf_game.agents.llm.client.time.sleep") as mocked_sleep,
        ):
            result = client.complete_json(system="system", messages=[], max_tokens=32)

        self.assertEqual(result, {"kind": "pass"})
        self.assertEqual(mocked_request.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_completed_model_response_keeps_usage_as_non_schema_attributes(self) -> None:
        client = self.configured_client(max_retries=0)
        with patch.object(
            client,
            "_request_json",
            return_value={
                "output_text": '{"kind":"pass"}',
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        ):
            result = client.complete_json(system="system", messages=[], max_tokens=32)

        self.assertEqual(result, {"kind": "pass"})
        self.assertEqual(result.token_usage, {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13})

    def test_tool_payload_and_tool_loop_are_bounded(self) -> None:
        client = self.configured_client(max_retries=0)
        tool = {
            "name": "read_current_round_dialogue",
            "description": "读取本轮对话",
            "parameters": {"type": "object", "properties": {}},
        }
        payload = build_model_payload(
            "system",
            [{"role": "user", "content": "hello"}],
            32,
            client.config,
            tools=[tool],
        )
        self.assertEqual(payload["tools"][0]["name"], tool["name"])
        self.assertEqual(
            extract_tool_calls(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": tool["name"],
                                            "arguments": "{}",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                "chat_completions",
            )[0]["name"],
            tool["name"],
        )
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": tool["name"],
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": '{"kind":"pass"}'}}]},
        ]
        with patch.object(client, "_request_json", side_effect=responses) as request:
            result = client.complete_json(
                system="system",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=32,
                tools=[tool],
                tool_executor=lambda _name, _arguments: {"dialogue": "x" * 100},
                max_tool_calls=1,
                max_tool_result_tokens=20,
            )
        self.assertEqual(result, {"kind": "pass"})
        self.assertEqual(request.call_count, 2)
        self.assertLessEqual(len(serialize_tool_result({"x": "y" * 100}, 20)), 20)
