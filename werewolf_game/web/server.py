"""零依赖的 asyncio HTTP/1.1 + SSE 服务器。

只实现 demo 需要的最小子集：GET 静态文件、GET SSE 流、POST JSON。使用
`asyncio.start_server` 让 HTTP 和 GameRunner 共用同一个事件循环，避免线程与
事件循环之间搬运 Future 带来的取消与竞态问题。

安全：默认只绑定 127.0.0.1 且没有任何认证。绑到对外地址意味着任何人都能抢占
真人席位并读到该席位的私密身份信息。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from ..agents.participants.human import KIND_LABELS, ROLE_LABELS
from ..season2.config import Season2Config
from .session import SessionManager, SessionOptions

LOGGER = logging.getLogger("werewolf.web")

STATIC_ROOT = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 256 * 1024
MAX_HEADER_BYTES = 64 * 1024

STATUS_TEXT = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    500: "Internal Server Error",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class HttpError(Exception):
    """带 HTTP 状态码的请求错误。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Request:
    def __init__(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> None:
        self.method = method
        split = urlsplit(target)
        self.path = unquote(split.path)
        self.query = parse_qs(split.query)
        self.headers = dict(headers)
        self.body = body

    def json(self) -> dict[str, Any]:
        if not self.body:
            return {}
        try:
            value = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HttpError(400, f"请求体不是合法 JSON：{error}") from error
        if not isinstance(value, dict):
            raise HttpError(400, "请求体必须是 JSON 对象")
        return value

    def query_int(self, name: str, default: int = 0) -> int:
        values = self.query.get(name)
        if not values:
            return default
        try:
            return int(values[0])
        except (TypeError, ValueError):
            return default


class WebDemoServer:
    """把 SessionManager 暴露为 HTTP 接口。"""

    def __init__(
        self,
        manager: SessionManager,
        *,
        host: str = "127.0.0.1",
        port: int = 8192,
    ) -> None:
        self.manager = manager
        self.host = host
        self.port = port
        self._server: asyncio.base_events.Server | None = None

    async def serve_forever(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        addresses = ", ".join(
            f"http://{sock.getsockname()[0]}:{sock.getsockname()[1]}"
            for sock in self._server.sockets or []
        )
        print(f"狼人杀 Web demo 已启动：{addresses}")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            print(
                "警告：服务器绑定在非本机地址且没有认证；任何可访问该地址的人都能"
                "占用真人席位并看到该席位的私密身份。"
            )
        async with self._server:
            await self._server.serve_forever()

    # ------------------------------------------------------------------
    # HTTP 解析与分发。
    # ------------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await self._read_request(reader)
            if request is None:
                return
            await self._dispatch(request, writer)
        except HttpError as error:
            await self._write_json(writer, error.status, {"error": error.message})
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        except Exception as error:  # 不让单个连接错误拖垮服务器
            LOGGER.exception("处理请求失败")
            try:
                await self._write_json(writer, 500, {"error": str(error)})
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_request(self, reader: asyncio.StreamReader) -> Request | None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError as error:
            raise HttpError(413, "请求头过大") from error
        except asyncio.IncompleteReadError:
            return None
        if len(head) > MAX_HEADER_BYTES:
            raise HttpError(413, "请求头过大")

        lines = head.decode("latin-1").split("\r\n")
        request_line = lines[0].split(" ")
        if len(request_line) < 2:
            raise HttpError(400, "请求行不合法")
        method, target = request_line[0].upper(), request_line[1]

        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()

        body = b""
        length = headers.get("content-length")
        if length:
            try:
                size = int(length)
            except ValueError as error:
                raise HttpError(400, "Content-Length 不合法") from error
            if size > MAX_BODY_BYTES:
                raise HttpError(413, "请求体过大")
            if size > 0:
                body = await reader.readexactly(size)
        return Request(method, target, headers, body)

    async def _dispatch(self, request: Request, writer: asyncio.StreamWriter) -> None:
        path = request.path.rstrip("/") or "/"

        if path == "/" and request.method == "GET":
            await self._write_static(writer, "index.html")
            return
        if path == "/api/meta" and request.method == "GET":
            await self._write_json(writer, 200, self._meta())
            return
        if path == "/api/games":
            if request.method != "POST":
                raise HttpError(405, "该路径只接受 POST")
            await self._create_game(request, writer)
            return

        if path.startswith("/api/games/"):
            parts = [item for item in path.split("/") if item]
            # /api/games/{id}/{action}
            if len(parts) == 4:
                session_id, action = parts[2], parts[3]
                await self._session_route(session_id, action, request, writer)
                return
            raise HttpError(404, "未知接口")

        if request.method == "GET":
            await self._write_static(writer, path.lstrip("/"))
            return
        raise HttpError(404, "未知接口")

    async def _session_route(
        self,
        session_id: str,
        action: str,
        request: Request,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            session = self.manager.get(session_id)
        except KeyError as error:
            raise HttpError(404, str(error)) from error

        if action == "state" and request.method == "GET":
            await self._write_json(
                writer, 200, session.state_view(request.query_int("since"))
            )
            return
        if action == "events" and request.method == "GET":
            await self._stream_events(session, request, writer)
            return
        if action == "action" and request.method == "POST":
            payload = request.json()
            request_id = str(payload.get("request_id") or "")
            action_payload = payload.get("action")
            if not isinstance(action_payload, dict):
                raise HttpError(400, "action 必须是 JSON 对象")
            try:
                session.human.submit(request_id, action_payload)
            except ValueError as error:
                raise HttpError(409, str(error)) from error
            await self._write_json(writer, 200, {"accepted": True})
            return
        if action == "result" and request.method == "GET":
            await self._write_json(
                writer, 200, {"result": session.result_view(), "error": session.error}
            )
            return
        if action == "pause" and request.method == "POST":
            await session.pause()
            await self._write_json(writer, 200, {"paused": True})
            return
        if action == "resume" and request.method == "POST":
            await session.resume()
            await self._write_json(writer, 200, {"paused": False})
            return
        if action == "abandon" and request.method == "POST":
            await session.close()
            self.manager.sessions.pop(session_id, None)
            await self._write_json(writer, 200, {"abandoned": True})
            return
        if action == "replay" and request.method == "GET":
            try:
                await self._write_json(writer, 200, session.replay_view())
            except ValueError as error:
                raise HttpError(409, str(error)) from error
            return
        raise HttpError(404, "未知接口")

    async def _create_game(
        self, request: Request, writer: asyncio.StreamWriter
    ) -> None:
        payload = request.json()
        overrides = payload.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise HttpError(400, "overrides 必须是对象")
        options = SessionOptions(
            mode=str(payload.get("mode") or "random"),
            seat=str(payload.get("seat") or "p1"),
            role=payload.get("role") or None,
            strategy=str(payload.get("strategy") or "best_win_rate"),
            overrides={str(k): str(v) for k, v in overrides.items()},
        )
        try:
            session = self.manager.create(options)
        except (ValueError, KeyError) as error:
            raise HttpError(400, str(error)) from error
        await self._write_json(writer, 200, session.describe())

    def _meta(self) -> dict[str, Any]:
        """前端渲染所需的静态元数据；中文标签复用终端真人玩家的映射表。

        规则说明直接取引擎的 `public_rule_summary()`，避免前端硬编码一份
        可能与实际规则不一致的文案。该投影本身就是"开局即应知"的公开配置。
        """

        return {
            "kind_labels": dict(KIND_LABELS),
            "role_labels": dict(ROLE_LABELS),
            "player_count": self.manager.config.evaluation.player_count,
            "roles": self.manager.available_roles(),
            "rules": self.manager.rule_summary(),
            "strategies": self.manager.strategy_options(),
            "modes": [
                {"mode": "random", "label": "随机模式"},
                {"mode": "choose", "label": "自选模式"},
            ],
        }

    # ------------------------------------------------------------------
    # SSE。
    # ------------------------------------------------------------------

    async def _stream_events(
        self,
        session: Any,
        request: Request,
        writer: asyncio.StreamWriter,
    ) -> None:
        headers = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/event-stream; charset=utf-8",
            "Cache-Control: no-cache",
            "Connection: close",
            "X-Accel-Buffering: no",
        ]
        writer.write(("\r\n".join(headers) + "\r\n\r\n").encode("utf-8"))
        await writer.drain()

        cursor = request.query_int("since")
        while True:
            view = session.state_view(cursor)
            cursor = view["event_cursor"]
            payload = json.dumps(view, ensure_ascii=False)
            writer.write(f"data: {payload}\n\n".encode("utf-8"))
            try:
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                return
            if view["finished"]:
                return
            await session.wait_for_change()

    # ------------------------------------------------------------------
    # 响应写出。
    # ------------------------------------------------------------------

    async def _write_json(
        self, writer: asyncio.StreamWriter, status: int, payload: Mapping[str, Any]
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self._write_response(
            writer, status, body, "application/json; charset=utf-8"
        )

    async def _write_static(self, writer: asyncio.StreamWriter, name: str) -> None:
        # 只允许读取 static 目录内的文件，拒绝任何越界路径。
        candidate = (STATIC_ROOT / name).resolve()
        try:
            candidate.relative_to(STATIC_ROOT)
        except ValueError as error:
            raise HttpError(404, "文件不存在") from error
        if not candidate.is_file():
            raise HttpError(404, "文件不存在")
        content_type = CONTENT_TYPES.get(
            candidate.suffix.lower(), "application/octet-stream"
        )
        await self._write_response(writer, 200, candidate.read_bytes(), content_type)

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        headers = [
            f"HTTP/1.1 {status} {STATUS_TEXT.get(status, 'OK')}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body)}",
            "Connection: close",
        ]
        writer.write(("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body)
        await writer.drain()


async def serve(
    *,
    config: Season2Config,
    host: str = "127.0.0.1",
    port: int = 8192,
    record_directory: str | Path | None = None,
    model_client: Any | None = None,
) -> None:
    manager = SessionManager(
        config=config, record_directory=record_directory, model_client=model_client
    )
    server = WebDemoServer(manager, host=host, port=port)
    try:
        await server.serve_forever()
    finally:
        await manager.close_all()
