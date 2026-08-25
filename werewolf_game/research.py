"""Meta-Agent 可选的外部研究接口。

默认没有启用任何网络调用。用户可以注入自己的异步 ``search`` provider，或配置一个
返回 JSON 的兼容检索端点。网页内容只会变成带哈希的研究来源，不能直接覆盖 Harness。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import asyncio


@dataclass(frozen=True)
class ResearchSource:
    title: str
    url: str
    summary: str
    retrieved_at: str
    content_sha256: str
    credibility: str = "unrated"
    provider: str = "unknown"

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "retrieved_at": self.retrieved_at,
            "content_sha256": self.content_sha256,
            "credibility": self.credibility,
            "provider": self.provider,
        }


class SearchProvider(Protocol):
    async def search(self, query: str, *, max_results: int = 5) -> list[ResearchSource]:
        """返回已脱敏、可审计的研究来源。"""


class JsonSearchProvider:
    """调用用户配置的 JSON 检索端点。

    请求体采用 ``query`` / ``max_results``；响应兼容 ``results``、``data`` 或直接数组。
    这是工具适配层，不把任何 provider 的正文当成系统指令。
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = "",
        provider_name: str = "json_endpoint",
        timeout_seconds: float = 15.0,
        default_credibility: str = "unrated",
    ) -> None:
        endpoint = str(endpoint or "").strip()
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("检索 endpoint 必须是 http(s) URL")
        self.endpoint = endpoint
        self.api_key = str(api_key or "")
        self.provider_name = provider_name
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.default_credibility = default_credibility

    async def search(self, query: str, *, max_results: int = 5) -> list[ResearchSource]:
        normalized_query = " ".join(str(query or "").split())[:500]
        if not normalized_query:
            return []
        return await asyncio.to_thread(self._search_sync, normalized_query, max_results)

    def _search_sync(self, query: str, max_results: int) -> list[ResearchSource]:
        body = json.dumps(
            {"query": query, "max_results": max(1, min(20, int(max_results)))},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(f"检索 provider 返回 HTTP {error.code}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"检索 provider 请求失败：{error}") from error
        return self._normalise_results(payload)

    def _normalise_results(self, payload: object) -> list[ResearchSource]:
        if isinstance(payload, Mapping):
            raw_results = payload.get("results", payload.get("data", []))
        else:
            raw_results = payload
        if not isinstance(raw_results, list):
            return []
        now = datetime.now(timezone.utc).isoformat()
        result: list[ResearchSource] = []
        for item in raw_results[:20]:
            if not isinstance(item, Mapping):
                continue
            title = _text(item.get("title", item.get("name")), 300)
            url = _text(item.get("url", item.get("link")), 1000)
            summary = _text(item.get("summary", item.get("snippet", item.get("content"))), 2400)
            if not title and not url and not summary:
                continue
            digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()
            result.append(
                ResearchSource(
                    title=title or "未命名来源",
                    url=url,
                    summary=summary,
                    retrieved_at=now,
                    content_sha256=digest,
                    credibility=_text(item.get("credibility"), 80) or self.default_credibility,
                    provider=self.provider_name,
                )
            )
        return result


def _text(value: object, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


async def call_search_provider(provider: Any, query: str, *, max_results: int = 5) -> list[ResearchSource]:
    """兼容异步和同步 provider，供 Meta-Agent 使用。"""

    method = getattr(provider, "search", None)
    if not callable(method):
        raise ValueError("research provider 必须提供 search 方法")
    if inspect.iscoroutinefunction(method):
        value = await method(query, max_results=max_results)
    else:
        value = await asyncio.to_thread(method, query, max_results=max_results)
    result: list[ResearchSource] = []
    for item in value or []:
        if isinstance(item, ResearchSource):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(
                ResearchSource(
                    title=_text(item.get("title"), 300),
                    url=_text(item.get("url"), 1000),
                    summary=_text(item.get("summary"), 2400),
                    retrieved_at=_text(item.get("retrieved_at"), 80)
                    or datetime.now(timezone.utc).isoformat(),
                    content_sha256=_text(item.get("content_sha256"), 128)
                    or hashlib.sha256(
                        _text(item.get("summary"), 2400).encode("utf-8")
                    ).hexdigest(),
                    credibility=_text(item.get("credibility"), 80) or "unrated",
                    provider=_text(item.get("provider"), 100) or "injected",
                )
            )
    return result[:20]
