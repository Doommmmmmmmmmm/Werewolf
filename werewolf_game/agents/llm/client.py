"""零第三方依赖的模型 API 客户端。

兼容参考项目使用的 Responses、Messages 和 Chat Completions 三种协议。密钥只从
进程环境或 .env 读取，绝不写进对局记录。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ...core.errors import ModelClientError


SUPPORTED_PROTOCOLS = frozenset({"responses", "messages", "chat_completions"})
SUPPORTED_AGENT_PROFILES = frozenset({"task", "meta"})


def _normalized_tool_schema(tool: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """读取框架内部统一的工具描述。"""

    name = str(tool.get("name", "")).strip()
    if not name:
        raise ValueError("工具描述缺少 name")
    description = str(tool.get("description", "")).strip()
    parameters = tool.get("parameters") or tool.get("input_schema")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    return name, description, dict(parameters)


def _provider_tools(
    tools: list[dict[str, Any]] | None, protocol: str
) -> list[dict[str, Any]]:
    """把统一工具描述转换成三种 API 协议各自的形状。"""

    if not tools:
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        name, description, parameters = _normalized_tool_schema(tool)
        if protocol == "responses":
            converted.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            )
        elif protocol == "messages":
            converted.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": parameters,
                }
            )
        else:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
    return converted


@dataclass(frozen=True)
class ModelConfig:
    mode: str
    model: str
    base_url: str
    protocol: str
    api_path: str
    api_key: str
    requires_api_key: bool
    enable_thinking: bool
    reasoning_effort: str | None
    max_retries: int
    retry_delay_ms: int
    timeout_seconds: float
    max_output_tokens: int
    responses_max_output_tokens: int
    # 额度不足通常来自上游号池中的单个账号。与普通网络重试分开计数，
    # 以便下一次请求有机会被路由到另一个可用账号。
    usage_limit_retries: int = 12

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.api_path.lstrip('/')}"


class ModelResponse(dict[str, Any]):
    """保留调用诊断的 JSON 字典，不向模型动作 schema 注入额外字段。"""

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        api_attempts: int,
        generic_retries: int,
        usage_limit_retries: int,
        token_usage: Mapping[str, int | None] | None = None,
        prompt_stats: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(value)
        self.api_attempts = int(api_attempts)
        self.generic_retries = int(generic_retries)
        self.usage_limit_retries = int(usage_limit_retries)
        # usage 仅作为对象属性保留，绝不混入玩家动作 JSON，避免影响既有 schema。
        self.token_usage = dict(token_usage or {})
        self.input_tokens = self.token_usage.get("input_tokens")
        self.output_tokens = self.token_usage.get("output_tokens")
        self.total_tokens = self.token_usage.get("total_tokens")
        # Prompt 预算诊断只作为对象属性保留，不注入模型返回的动作 JSON。
        self.prompt_stats = dict(prompt_stats or {})


def load_dotenv(file_path: str | Path | None = None) -> bool:
    """载入简单 KEY=VALUE 格式 .env；已存在的环境变量优先。"""

    path = Path(file_path or Path.cwd() / ".env")
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
    return True


def normalize_protocol(value: str | None, fallback: str = "responses") -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"chat", "openai", "chat_completion", "chat_completions"}:
        return "chat_completions"
    return normalized if normalized in SUPPORTED_PROTOCOLS else fallback


def get_model_config(
    mode: str = "api",
    env: Mapping[str, str] | None = None,
    *,
    profile: str | None = None,
) -> ModelConfig:
    """从环境变量读取模型配置，并可按 Task / Meta Agent 分流。

    ``task`` 默认使用轻量、无思考的游戏模型；``meta`` 默认继承原有 API
    模型配置，以避免策略更新与局内行动共用同一个模型选择。
    """

    values = env or os.environ
    self_hosted = str(mode).lower() == "self_hosted"

    def first(*names: str) -> str:
        for name in names:
            value = values.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def boolean(name: str, fallback: bool) -> bool:
        value = values.get(name)
        if value is None or not str(value).strip():
            return fallback
        return str(value).strip().lower() == "true"

    def number(name: str, fallback: int, minimum: int, maximum: int) -> int:
        try:
            value = int(str(values.get(name, "")).strip())
        except (TypeError, ValueError):
            return fallback
        return max(minimum, min(maximum, value))

    normalized_profile = str(profile or "").strip().lower()
    if normalized_profile and normalized_profile not in SUPPORTED_AGENT_PROFILES:
        raise ValueError(f"不支持的模型 profile：{profile}")

    if normalized_profile == "task":
        profile_model = first("WEREWOLF_TASK_MODEL") or "qwen3.8-flash"
        profile_reasoning_effort = (
            first("WEREWOLF_TASK_REASONING_EFFORT") or "none"
        )
    elif normalized_profile == "meta":
        profile_model = first("WEREWOLF_META_MODEL")
        profile_reasoning_effort = first("WEREWOLF_META_REASONING_EFFORT") or None
    else:
        profile_model = ""
        profile_reasoning_effort = None

    if self_hosted:
        protocol = normalize_protocol(first("SELF_HOSTED_MODEL_API_MODE"), "chat_completions")
        model = profile_model or first("SELF_HOSTED_MODEL_NAME")
        base_url = first("SELF_HOSTED_MODEL_BASE_URL")
        api_key = first("SELF_HOSTED_MODEL_API_KEY")
        api_path = first("SELF_HOSTED_MODEL_API_PATH")
        requires_api_key = boolean("SELF_HOSTED_MODEL_REQUIRES_API_KEY", False)
        enable_thinking = boolean("SELF_HOSTED_MODEL_ENABLE_THINKING", False)
        response_limit = number("SELF_HOSTED_MODEL_RESPONSES_MAX_OUTPUT_TOKENS", 0, 0, 32000)
    else:
        protocol = normalize_protocol(first("API_MODEL_API_MODE", "MODEL_API_MODE"), "responses")
        model = profile_model or first("API_MODEL_NAME", "OPENAI_MODEL", "MODEL_ID")
        base_url = first("API_MODEL_BASE_URL", "OPENAI_BASE_URL", "MODEL_BASE_URL")
        api_key = first("API_MODEL_API_KEY", "OPENAI_API_KEY", "MODEL_API_KEY")
        api_path = first("API_MODEL_API_PATH", "MODEL_API_PATH")
        requires_api_key = True
        # The Task profile explicitly disables thinking.  Meta keeps the
        # provider's original behavior unless its protocol has a dedicated
        # ``reasoning_effort`` override.
        enable_thinking = normalized_profile == "meta"
        response_limit = number("API_MODEL_RESPONSES_MAX_OUTPUT_TOKENS", 0, 0, 32000)
        if response_limit == 0:
            response_limit = number("RESPONSES_MAX_OUTPUT_TOKENS", 0, 0, 32000)

    default_path = {
        "responses": "/responses",
        "messages": "/messages",
        "chat_completions": "/v1/chat/completions",
    }[protocol]
    return ModelConfig(
        mode="self_hosted" if self_hosted else "api",
        model=model,
        base_url=base_url.rstrip("/"),
        protocol=protocol,
        api_path=api_path or default_path,
        api_key=api_key,
        requires_api_key=requires_api_key,
        enable_thinking=enable_thinking,
        reasoning_effort=profile_reasoning_effort,
        max_retries=number("MODEL_MAX_RETRIES", 2, 0, 4),
        retry_delay_ms=number("MODEL_RETRY_DELAY_MS", 300, 100, 5000),
        timeout_seconds=number("MODEL_TIMEOUT_MS", 45000, 1000, 180000) / 1000,
        max_output_tokens=number("MAX_OUTPUT_TOKENS", 1000, 1, 32000),
        responses_max_output_tokens=response_limit,
        usage_limit_retries=number("MODEL_USAGE_LIMIT_RETRIES", 12, 0, 100),
    )


def is_model_configured(config: ModelConfig) -> bool:
    return bool(
        config.model
        and config.base_url
        and (not config.requires_api_key or config.api_key)
    )


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    for message in messages:
        content = message.get("content", {})
        result.append(
            {
                "role": "assistant" if message.get("role") == "assistant" else "user",
                "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            }
        )
    return result


def build_model_payload(
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    config: ModelConfig,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = normalize_messages(messages)
    if config.protocol == "responses":
        payload: dict[str, Any] = {
            "model": config.model,
            "instructions": system,
            "input": normalized,
            "store": False,
        }
        if config.responses_max_output_tokens > 0:
            payload["max_output_tokens"] = config.responses_max_output_tokens
        if config.reasoning_effort:
            payload["reasoning"] = {"effort": config.reasoning_effort}
        provider_tools = _provider_tools(tools, config.protocol)
        if provider_tools:
            payload["tools"] = provider_tools
        return payload
    if config.protocol == "messages":
        payload = {
            "model": config.model,
            "system": system,
            "messages": normalized,
            "max_tokens": int(max_tokens or config.max_output_tokens),
            "stream": False,
        }
        # DeepSeek / Anthropic-compatible gateways use this field rather than
        # chat_template_kwargs to suppress hidden thinking text.  Only the
        # Task profile requests ``reasoning_effort=none``; Meta keeps the
        # provider's original default behavior.
        if config.reasoning_effort == "none":
            payload["thinking"] = {"type": "disabled"}
        provider_tools = _provider_tools(tools, config.protocol)
        if provider_tools:
            payload["tools"] = provider_tools
        return payload
    payload = {
        "model": config.model,
        "messages": [{"role": "system", "content": system}, *normalized],
        "max_tokens": int(max_tokens or config.max_output_tokens),
        "stream": False,
    }
    if not config.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    provider_tools = _provider_tools(tools, config.protocol)
    if provider_tools:
        payload["tools"] = provider_tools
    return payload


def _prompt_char_count(system: str, messages: list[dict[str, Any]]) -> int:
    """计算一次模型请求的保守输入长度（以 Unicode 字符计）。"""

    total = len(str(system or ""))
    for message in messages:
        if isinstance(message, Mapping):
            total += len(str(message.get("content") or ""))
    return total


def _enforce_prompt_budget(
    system: str, messages: list[dict[str, Any]], max_prompt_chars: int | None
) -> None:
    if max_prompt_chars is None:
        return
    limit = max(1, int(max_prompt_chars))
    actual = _prompt_char_count(system, messages)
    if actual > limit:
        error = ModelClientError(f"Task-Agent prompt 超过外部上限：{actual}>{limit} 字符")
        error.prompt_stats = {
            "prompt_measurement_count": 1,
            "prompt_chars_total": actual,
            "prompt_chars_max": actual,
            "prompt_chars_min": actual,
            "prompt_remaining_chars": 0,
            "prompt_turns": 1,
        }
        raise error


def extract_text(response: dict[str, Any]) -> str:
    chat_content = response.get("choices", [{}])[0].get("message", {}).get("content") if response.get("choices") else None
    if isinstance(chat_content, str):
        return chat_content
    if isinstance(chat_content, list):
        return "\n".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in chat_content
        )
    content = response.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"output_text", "text"}
        )
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                texts.append(str(part.get("text", "")))
    return "\n".join(texts)


def _tool_arguments(value: object) -> dict[str, Any]:
    """把不同供应商返回的工具参数统一成对象。"""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def extract_tool_calls(
    response: Mapping[str, Any], protocol: str | None = None
) -> list[dict[str, Any]]:
    """提取 Responses、Messages 和 Chat Completions 的工具调用。"""

    calls: list[dict[str, Any]] = []

    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type", ""))
            if item_type not in {"function_call", "tool_call", "tool_use"}:
                continue
            name = item.get("name")
            arguments = item.get("arguments", item.get("input", {}))
            if not name:
                continue
            calls.append(
                {
                    "id": str(item.get("call_id", item.get("id", ""))),
                    "name": str(name),
                    "arguments": _tool_arguments(arguments),
                }
            )

    content = response.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type", ""))
            if item_type not in {"tool_use", "function_call", "tool_call"}:
                continue
            name = item.get("name")
            if not name:
                continue
            calls.append(
                {
                    "id": str(item.get("id", item.get("call_id", ""))),
                    "name": str(name),
                    "arguments": _tool_arguments(item.get("input", item.get("arguments", {}))),
                }
            )

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        if isinstance(message, Mapping):
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for item in tool_calls:
                    if not isinstance(item, Mapping):
                        continue
                    function = item.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    name = function.get("name")
                    if not name:
                        continue
                    calls.append(
                        {
                            "id": str(item.get("id", "")),
                            "name": str(name),
                            "arguments": _tool_arguments(function.get("arguments", {})),
                        }
                    )
            function_call = message.get("function_call")
            if isinstance(function_call, Mapping) and function_call.get("name"):
                calls.append(
                    {
                        "id": "",
                        "name": str(function_call["name"]),
                        "arguments": _tool_arguments(function_call.get("arguments", {})),
                    }
                )

    # A few gateways put tool calls at the top level. Avoid duplicating calls
    # already found in the protocol-specific fields.
    top_level = response.get("tool_calls")
    if isinstance(top_level, list):
        for item in top_level:
            if not isinstance(item, Mapping):
                continue
            function = item.get("function", item)
            if not isinstance(function, Mapping) or not function.get("name"):
                continue
            candidate = {
                "id": str(item.get("id", "")),
                "name": str(function["name"]),
                "arguments": _tool_arguments(function.get("arguments", function.get("input", {}))),
            }
            if candidate not in calls:
                calls.append(candidate)
    return calls


def serialize_tool_result(value: object, max_tokens: int) -> str:
    """序列化并截断工具结果。

    当前没有额外 tokenizer 依赖，采用 Unicode 字符作为保守预算单位；对中文
    内容这不会超过指定的 token 上限，虽可能比真实 tokenizer 更节省一些空间。
    """

    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = json.dumps({"value": str(value)}, ensure_ascii=False)
    limit = max(1, int(max_tokens))
    if len(text) <= limit:
        return text
    marker = "…[已截断]"
    if len(marker) >= limit:
        return marker[:limit]
    return text[: limit - len(marker)] + marker


def _is_tool_unsupported_error(error: Exception) -> bool:
    """判断网关是否明确表示不支持工具字段。"""

    status_code = int(getattr(error, "status_code", 0) or 0)
    if status_code not in {400, 404, 405, 422}:
        return False
    message = str(error).lower()
    tool_words = ("tool", "function_call", "function call", "input_schema")
    unsupported_words = ("unsupported", "not support", "unknown", "invalid", "unrecognized")
    return any(word in message for word in tool_words) and any(
        word in message for word in unsupported_words
    )


def extract_json(text: str) -> dict[str, Any] | None:
    """兼容模型偶尔加上的 Markdown 围栏或前后解释。"""

    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_token_usage(response: Mapping[str, Any]) -> dict[str, int | None]:
    """提取兼容 Responses / Chat Completions 的非敏感 token 用量。

    不同兼容服务使用 ``input_tokens`` / ``output_tokens`` 或
    ``prompt_tokens`` / ``completion_tokens``。服务没有返回 ``usage`` 时返回
    空字典；调用方据此明确标为不可用，而不会杜撰估算值。
    """

    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return {}

    def token_count(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, bool):
                continue
            try:
                normalized = int(value)
            except (TypeError, ValueError):
                continue
            if normalized >= 0:
                return normalized
        return None

    input_tokens = token_count("input_tokens", "prompt_tokens", "input_token_count")
    output_tokens = token_count(
        "output_tokens", "completion_tokens", "output_token_count"
    )
    total_tokens = token_count("total_tokens", "total_token_count")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return {}
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class ModelClient:
    """同步 HTTP 客户端；LLM Participant 会在线程中调用它，保持投票并发。"""

    def __init__(self, config: ModelConfig) -> None:
        if not is_model_configured(config):
            raise ModelClientError("模型配置不完整，请检查服务端环境变量")
        self.config = config

    @classmethod
    def from_env(
        cls,
        mode: str | None = None,
        *,
        profile: str | None = None,
    ) -> "ModelClient":
        load_dotenv()
        return cls(
            get_model_config(
                mode or os.environ.get("DEFAULT_MODEL_MODE", "api"),
                profile=profile,
            )
        )

    def complete_json(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], object] | None = None,
        max_tool_calls: int = 0,
        max_tool_result_tokens: int = 800,
        max_prompt_chars: int | None = None,
    ) -> dict[str, Any]:
        """完成一次 JSON 调用，并按异常类型采用不同的重试策略。

        ``usage limit`` / HTTP 429 在当前接入中通常表示号池挑到了不可用账号。
        这类错误不等待退避，直接重新发起请求以触发上游重新路由；其余可恢复错误
        仍使用原有的有限线性退避，避免无限等待真正故障的服务。
        """

        if tools and tool_executor is not None and int(max_tool_calls) > 0:
            return self._complete_json_with_tools(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                tools=tools,
                tool_executor=tool_executor,
                max_tool_calls=max_tool_calls,
                max_tool_result_tokens=max_tool_result_tokens,
                max_prompt_chars=max_prompt_chars,
            )

        prompt_chars = _prompt_char_count(system, messages)
        _enforce_prompt_budget(system, messages, max_prompt_chars)
        generic_retries = 0
        usage_limit_retries = 0
        total_attempts = 0
        while True:
            total_attempts += 1
            try:
                payload = build_model_payload(system, messages, max_tokens, self.config)
                response = self._request_json(payload)
                parsed = extract_json(extract_text(response))
                if parsed is None:
                    raise ModelClientError("模型响应不是有效 JSON")
                return ModelResponse(
                    parsed,
                    api_attempts=total_attempts,
                    generic_retries=generic_retries,
                    usage_limit_retries=usage_limit_retries,
                    token_usage=extract_token_usage(response),
                    prompt_stats={
                        "prompt_measurement_count": 1,
                        "prompt_chars_total": prompt_chars,
                        "prompt_chars_max": prompt_chars,
                        "prompt_chars_min": prompt_chars,
                        "prompt_remaining_chars": max(0, int(max_prompt_chars or 0) - prompt_chars),
                        "prompt_tool_calls": 0,
                        "prompt_turns": 1,
                    },
                )
            except Exception as error:  # 保留服务端原始错误供 Runner 记录
                if self._is_usage_limit(error):
                    if usage_limit_retries < self.config.usage_limit_retries:
                        usage_limit_retries += 1
                        # 号池场景下立即重试，避免无意义地等待同一账号冷却。
                        continue
                    self._attach_retry_context(
                        error,
                        total_attempts=total_attempts,
                        generic_retries=generic_retries,
                        usage_limit_retries=usage_limit_retries,
                    )
                    raise

                if generic_retries >= self.config.max_retries or not self._retryable(error):
                    self._attach_retry_context(
                        error,
                        total_attempts=total_attempts,
                        generic_retries=generic_retries,
                        usage_limit_retries=usage_limit_retries,
                    )
                    raise

                generic_retries += 1
                time.sleep((self.config.retry_delay_ms / 1000) * generic_retries)

    def _complete_json_with_tools(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], object],
        max_tool_calls: int,
        max_tool_result_tokens: int,
        max_prompt_chars: int | None = None,
    ) -> ModelResponse:
        """执行一个极小的工具循环。

        工具结果以新的用户消息回传，而不是把完整历史留在客户端会话中；这样既
        兼容三种当前协议，也保证每次 Task-Agent 行动结束后会话可丢弃。
        """

        conversation = [dict(message) for message in messages]
        allowed_tool_names = {
            str(tool.get("name", "")) for tool in tools if tool.get("name")
        }
        tool_calls_used = 0
        tools_enabled = True
        total_attempts = 0
        total_generic_retries = 0
        total_usage_limit_retries = 0
        usage_totals: dict[str, int] = {}
        prompt_chars_total = 0
        prompt_chars_max = 0
        prompt_chars_min: int | None = None
        prompt_measurement_count = 0

        # One initial call, at most one final call after each tool response, and
        # one guard turn if a model ignores the exhausted tool budget.
        max_turns = max(3, int(max_tool_calls) + 2)
        for _turn in range(max_turns):
            current_prompt_chars = _prompt_char_count(system, conversation)
            prompt_measurement_count += 1
            prompt_chars_total += current_prompt_chars
            prompt_chars_max = max(prompt_chars_max, current_prompt_chars)
            prompt_chars_min = (
                current_prompt_chars
                if prompt_chars_min is None
                else min(prompt_chars_min, current_prompt_chars)
            )
            _enforce_prompt_budget(system, conversation, max_prompt_chars)
            payload = build_model_payload(
                system,
                conversation,
                max_tokens,
                self.config,
                tools=tools if tools_enabled else None,
            )
            try:
                response, attempts, generic_retries, usage_retries = (
                    self._request_with_retries(payload)
                )
            except Exception as error:
                # Some OpenAI-compatible gateways expose the model but not
                # function tools. Keep the minimum game runnable by retrying
                # the original request once without a tool declaration; the
                # fallback is limited to explicit unsupported-tool errors and
                # never masks network or quota failures.
                if _is_tool_unsupported_error(error) and _turn == 0:
                    return self.complete_json(
                        system=system,
                        messages=messages,
                        max_tokens=max_tokens,
                        max_prompt_chars=max_prompt_chars,
                    )  # type: ignore[return-value]
                raise
            total_attempts += attempts
            total_generic_retries += generic_retries
            total_usage_limit_retries += usage_retries
            for key, value in extract_token_usage(response).items():
                if value is not None:
                    usage_totals[key] = usage_totals.get(key, 0) + int(value)

            calls = extract_tool_calls(response, self.config.protocol)
            if calls and tools_enabled:
                remaining = max(0, int(max_tool_calls) - tool_calls_used)
                if remaining == 0:
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "工具调用预算已用尽。请不要再调用工具，直接返回最终行动 JSON。"
                            ),
                        }
                    )
                    tools_enabled = False
                    continue

                result_items: list[dict[str, Any]] = []
                for call in calls[:remaining]:
                    tool_calls_used += 1
                    name = str(call.get("name", ""))
                    if name not in allowed_tool_names:
                        result: object = {"error": f"不支持的工具：{name}"}
                    else:
                        try:
                            result = tool_executor(
                                name, call.get("arguments") or {}
                            )
                        except Exception as error:  # pragma: no cover - defensive boundary
                            result = {"error": f"工具执行失败：{type(error).__name__}"}
                    result_items.append(
                        {
                            "name": name,
                            "result": serialize_tool_result(
                                result, max_tool_result_tokens
                            ),
                        }
                    )
                if len(calls) > remaining:
                    result_items.append(
                        {
                            "name": "预算提示",
                            "result": "本次其余工具调用未执行：已达到调用次数上限。",
                        }
                    )
                conversation.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "tool_results": result_items,
                                "instruction": "请结合工具结果直接返回最终行动 JSON。",
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                if tool_calls_used >= int(max_tool_calls):
                    tools_enabled = False
                continue

            parsed = extract_json(extract_text(response))
            if parsed is not None:
                return ModelResponse(
                    parsed,
                    api_attempts=total_attempts,
                    generic_retries=total_generic_retries,
                    usage_limit_retries=total_usage_limit_retries,
                    token_usage=usage_totals,
                    prompt_stats={
                        "prompt_measurement_count": prompt_measurement_count,
                        "prompt_chars_total": prompt_chars_total,
                        "prompt_chars_max": prompt_chars_max,
                        "prompt_chars_min": prompt_chars_min or 0,
                        "prompt_remaining_chars": max(
                            0, int(max_prompt_chars or 0) - prompt_chars_max
                        ),
                        "prompt_tool_calls": tool_calls_used,
                        "prompt_turns": prompt_measurement_count,
                    },
                )
            raise ModelClientError("模型响应不是有效 JSON，且没有可处理的工具调用")

        raise ModelClientError("模型工具调用超过允许的会话轮数")

    def _request_with_retries(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], int, int, int]:
        """发送一次请求并返回该请求的重试诊断。"""

        generic_retries = 0
        usage_limit_retries = 0
        total_attempts = 0
        while True:
            total_attempts += 1
            try:
                return (
                    self._request_json(payload),
                    total_attempts,
                    generic_retries,
                    usage_limit_retries,
                )
            except Exception as error:
                if self._is_usage_limit(error):
                    if usage_limit_retries < self.config.usage_limit_retries:
                        usage_limit_retries += 1
                        continue
                    self._attach_retry_context(
                        error,
                        total_attempts=total_attempts,
                        generic_retries=generic_retries,
                        usage_limit_retries=usage_limit_retries,
                    )
                    raise
                if generic_retries >= self.config.max_retries or not self._retryable(error):
                    self._attach_retry_context(
                        error,
                        total_attempts=total_attempts,
                        generic_retries=generic_retries,
                        usage_limit_retries=usage_limit_retries,
                    )
                    raise
                generic_retries += 1
                time.sleep((self.config.retry_delay_ms / 1000) * generic_retries)

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(self.config.endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                text = response.read().decode("utf-8")
        except HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(text)
                message = detail.get("error", {}).get("message") or detail.get("message")
            except json.JSONDecodeError:
                message = None
            failure = ModelClientError(message or f"模型 API 返回 HTTP {error.code}")
            failure.status_code = error.code  # type: ignore[attr-defined]
            raise failure from error
        except URLError as error:
            raise ModelClientError(str(error.reason)) from error
        try:
            response_value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ModelClientError("模型 API 返回的不是 JSON") from error
        if not isinstance(response_value, dict):
            raise ModelClientError("模型 API 返回结构不正确")
        return response_value

    @staticmethod
    def _retryable(error: Exception) -> bool:
        status_code = int(getattr(error, "status_code", 0) or 0)
        if status_code in {408, 409, 429} or status_code >= 500:
            return True
        message = str(error).lower()
        return any(
            phrase in message
            for phrase in (
                "timeout",
                "timed out",
                "connection reset",
                "refused",
                "invalid json",
                "模型响应不是有效 json",
                "模型 api 返回的不是 json",
            )
        )

    @staticmethod
    def _is_usage_limit(error: Exception) -> bool:
        """识别号池可通过重发请求恢复的额度/配额异常。"""

        status_code = int(getattr(error, "status_code", 0) or 0)
        if status_code == 429:
            return True
        message = str(error).lower()
        return any(
            phrase in message
            for phrase in (
                "usage limit",
                "quota exceeded",
                "quota exhausted",
                "insufficient quota",
                "额度限制",
                "额度已用尽",
                "配额不足",
                "配额已用尽",
            )
        )

    @staticmethod
    def _attach_retry_context(
        error: Exception,
        *,
        total_attempts: int,
        generic_retries: int,
        usage_limit_retries: int,
    ) -> None:
        """给最终异常附上诊断信息；异常记录层会把它写进完整日志。"""

        try:
            error.api_attempts = total_attempts  # type: ignore[attr-defined]
            error.generic_retries = generic_retries  # type: ignore[attr-defined]
            error.usage_limit_retries = usage_limit_retries  # type: ignore[attr-defined]
        except Exception:
            # 第三方异常对象未必允许添加属性；不影响原始异常的传播。
            pass
