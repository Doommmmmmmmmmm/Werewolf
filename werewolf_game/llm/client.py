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
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..errors import ModelClientError


SUPPORTED_PROTOCOLS = frozenset({"responses", "messages", "chat_completions"})
SUPPORTED_AGENT_PROFILES = frozenset({"task", "meta"})


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
        profile_model = first("WEREWOLF_TASK_MODEL") or "deepseek-v4-flash"
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
        return payload
    payload = {
        "model": config.model,
        "messages": [{"role": "system", "content": system}, *normalized],
        "max_tokens": int(max_tokens or config.max_output_tokens),
        "stream": False,
    }
    if not config.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


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
        self, *, system: str, messages: list[dict[str, Any]], max_tokens: int
    ) -> dict[str, Any]:
        """完成一次 JSON 调用，并按异常类型采用不同的重试策略。

        ``usage limit`` / HTTP 429 在当前接入中通常表示号池挑到了不可用账号。
        这类错误不等待退避，直接重新发起请求以触发上游重新路由；其余可恢复错误
        仍使用原有的有限线性退避，避免无限等待真正故障的服务。
        """

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
