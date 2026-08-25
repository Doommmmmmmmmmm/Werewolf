"""命令行人工玩家适配器。

人工玩家和 LLM、脚本玩家共享同一个 ``Participant`` 接口。它只负责把行动包
翻译成易读的终端提示，再把玩家输入翻译回结构化行动；身份、目标合法性和结算
仍由 ``GameEngine`` 负责。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .base import Participant


ROLE_LABELS = {
    "wolf": "狼人",
    "villager": "平民",
    "seer": "预言家",
    "witch": "女巫",
    "guard": "守卫",
    "hunter": "猎人",
    "idiot": "白痴",
}

KIND_LABELS = {
    "speak": "发言",
    "pass": "跳过",
    "wolf_kill_vote": "狼人刀人",
    "seer_inspect": "预言家查验",
    "witch_heal": "女巫使用解药",
    "witch_poison": "女巫使用毒药",
    "guard_protect": "守卫守护",
    "hunter_shoot": "猎人开枪",
    "day_vote": "白天投票",
    "last_words": "发表遗言",
    "sheriff_candidate": "参加警长竞选",
    "sheriff_vote": "警长投票",
    "sheriff_speech_order": "调整发言顺序",
    "sheriff_badge_transfer": "传递警徽",
    "sheriff_badge_destroy": "撕毁警徽",
}

ALIASES = {
    "说话": "speak",
    "发言": "speak",
    "speak": "speak",
    "跳过": "pass",
    "过": "pass",
    "pass": "pass",
    "刀": "wolf_kill_vote",
    "狼人投票": "wolf_kill_vote",
    "wolf_kill_vote": "wolf_kill_vote",
    "验": "seer_inspect",
    "查验": "seer_inspect",
    "seer_inspect": "seer_inspect",
    "救": "witch_heal",
    "解药": "witch_heal",
    "witch_heal": "witch_heal",
    "毒": "witch_poison",
    "毒药": "witch_poison",
    "witch_poison": "witch_poison",
    "守": "guard_protect",
    "守护": "guard_protect",
    "guard_protect": "guard_protect",
    "枪": "hunter_shoot",
    "开枪": "hunter_shoot",
    "hunter_shoot": "hunter_shoot",
    "投票": "day_vote",
    "投": "day_vote",
    "day_vote": "day_vote",
    "遗言": "last_words",
    "last_words": "last_words",
    "上警": "sheriff_candidate",
    "sheriff_candidate": "sheriff_candidate",
    "警长投票": "sheriff_vote",
    "竞选投票": "sheriff_vote",
    "sheriff_vote": "sheriff_vote",
    "顺序": "sheriff_speech_order",
    "发言顺序": "sheriff_speech_order",
    "sheriff_speech_order": "sheriff_speech_order",
    "传徽": "sheriff_badge_transfer",
    "传警徽": "sheriff_badge_transfer",
    "sheriff_badge_transfer": "sheriff_badge_transfer",
    "撕徽": "sheriff_badge_destroy",
    "撕警徽": "sheriff_badge_destroy",
    "sheriff_badge_destroy": "sheriff_badge_destroy",
}


class HumanParticipant(Participant):
    """通过终端输入行动的人工玩家。

    ``input_fn`` 和 ``output_fn`` 可注入测试或网页适配层；默认使用终端的
    ``input`` / ``print``。输入支持 JSON，也支持简写，例如：

    ``发言 三号这轮的查杀逻辑需要重新核对``
    ``投票 p7``
    ``刀 p10``
    ``pass``
    """

    # ``input()`` 在线程中等待时无法被 asyncio 安全取消。若把它和模型玩家
    # 共用 Runner 的行动超时，超时后的旧输入线程会继续占用 stdin，并可能吃掉
    # 下一轮真人输入。真人席位因此默认不设 Runner 级超时；网页等非阻塞实现
    # 如需超时，可在实例上显式设置一个正数覆盖该值。
    decision_timeout_seconds: float | None = None

    def __init__(
        self,
        player_id: str,
        *,
        input_fn: Callable[[str], str | Awaitable[str]] = input,
        output_fn: Callable[[str], None] = print,
        show_events: int = 8,
    ) -> None:
        super().__init__(player_id)
        self.input_fn = input_fn
        self._threaded_input = input_fn is input
        self.output_fn = output_fn
        self.show_events = max(0, int(show_events))
        self._role_announced = False
        self.latest_public_sync: dict[str, Any] | None = None

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收每夜公开状态同步；不显示秘密信息。"""

        self.latest_public_sync = sync_packet
        state = sync_packet.get("public_state") or {}
        players = state.get("players") or []
        if players:
            status = "、".join(
                f"{item.get('id')}({'存活' if item.get('alive') else '死亡'})"
                for item in players
            )
            self.output_fn(
                f"\n[夜间公开同步] 第 {sync_packet.get('game', {}).get('round', '?')} 轮：{status}"
            )

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        self._display_packet(turn_packet)
        while True:
            # 真实终端输入使用线程，避免阻塞 Runner 的事件循环；测试和网页
            # 适配器可注入 async callable 或普通的非阻塞 callable。
            value = (
                await asyncio.to_thread(self.input_fn, "行动> ")
                if self._threaded_input
                else self.input_fn("行动> ")
            )
            if isinstance(value, BaseException):
                raise value
            raw = await value if hasattr(value, "__await__") else value
            try:
                action = self.parse_input(raw, turn_packet["request"])
                self._validate_local(action, turn_packet["request"])
                return action
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self.output_fn(f"输入无效：{error}。请按照上面的格式重新输入。")

    @classmethod
    def parse_input(
        cls, raw: str, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        """解析 JSON 或终端简写，但不替代引擎做最终校验。"""

        text = str(raw or "").strip()
        if not text or text.lower() in {"pass", "p", "跳过", "过"}:
            return {"kind": "pass"}
        if text.startswith("{"):
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("JSON 输入必须是对象")
            return dict(value)

        command, separator, remainder = text.partition(" ")
        kind = ALIASES.get(command.strip().lower(), ALIASES.get(command.strip()))
        if kind is None:
            raise ValueError(f"未知行动“{command}”，请使用合法 kind 或中文简写")
        remainder = remainder.strip() if separator else ""
        if kind in {"speak", "last_words"}:
            if not remainder:
                raise ValueError("发言行动需要在命令后提供文字")
            return {"kind": kind, "text": remainder}
        if kind == "sheriff_speech_order":
            order = remainder.lower()
            target = {
                "正序": "next",
                "下一位": "next",
                "next": "next",
                "逆序": "previous",
                "上一位": "previous",
                "previous": "previous",
            }.get(order)
            if target is None:
                raise ValueError("发言顺序请输入“正序/逆序”")
            return {"kind": kind, "target_id": target}
        if kind in {"pass", "sheriff_candidate", "sheriff_badge_destroy"}:
            if remainder:
                raise ValueError("该行动不需要目标")
            return {"kind": kind}
        if not remainder:
            raise ValueError("该行动需要目标，例如“投票 p7”")
        target = remainder.split()[0]
        return {"kind": kind, "target_id": target}

    @staticmethod
    def _validate_local(action: dict[str, Any], request: Mapping[str, Any]) -> None:
        kind = str(action.get("kind", ""))
        allowed_actions = request.get("allowed_actions", [])
        allowed = next(
            (item for item in allowed_actions if item.get("kind") == kind), None
        )
        if allowed is None:
            raise ValueError(f"当前阶段不允许“{kind}”")
        target_ids = allowed.get("target_ids")
        target = action.get("target_id")
        if target_ids is not None:
            if target_ids and target not in target_ids:
                raise ValueError(
                    f"目标不合法，可选目标：{'、'.join(str(item) for item in target_ids)}"
                )
            if not target_ids and target:
                raise ValueError("该行动不需要目标")
        elif target:
            raise ValueError("该行动不需要目标")
        if kind in {"speak", "last_words"}:
            speech = str(action.get("text", "")).strip()
            if not speech:
                raise ValueError("发言不能为空")
            if len(speech) > int(allowed.get("max_chars", 10**9)):
                raise ValueError(f"发言不能超过 {allowed['max_chars']} 字")
            if allowed.get("require_chinese") and not any(
                "\u3400" <= char <= "\u9fff" for char in speech
            ):
                raise ValueError("发言必须包含中文")
            if not allowed.get("allow_latin_letters", True) and any(
                ("A" <= char <= "Z") or ("a" <= char <= "z") for char in speech
            ):
                raise ValueError("发言不能包含英文字母")

    def _display_packet(self, packet: Mapping[str, Any]) -> None:
        private = packet.get("private_information") or {}
        if not self._role_announced and private.get("role"):
            role = str(private["role"])
            self.output_fn(
                f"\n你的身份：{ROLE_LABELS.get(role, role)}；"
                f"阵营：{'狼人' if private.get('team') == 'wolf' else '好人'}"
            )
            teammates = private.get("wolf_teammates") or []
            if teammates:
                self.output_fn(f"狼人队友：{'、'.join(map(str, teammates))}")
            self._role_announced = True

        request = packet.get("request") or {}
        self.output_fn(
            f"\n第 {packet.get('game', {}).get('round', packet.get('round', '?'))} 轮 · "
            f"阶段：{packet.get('game', {}).get('phase', packet.get('phase', '?'))}"
        )
        state = packet.get("public_state") or {}
        players = state.get("players") or []
        if players:
            status = "、".join(
                f"{item.get('id')}({'存活' if item.get('alive') else '死亡'})"
                for item in players
            )
            self.output_fn(f"玩家状态：{status}")
        allowed = request.get("allowed_actions") or []
        labels = []
        for item in allowed:
            kind = str(item.get("kind"))
            target_ids = item.get("target_ids")
            suffix = ""
            if target_ids:
                suffix = f"，目标：{'、'.join(map(str, target_ids))}"
            if item.get("max_chars"):
                suffix = f"，最多{item['max_chars']}字" + suffix
            labels.append(f"{kind}（{KIND_LABELS.get(kind, kind)}{suffix}）")
        self.output_fn("可选行动：" + "；".join(labels))
        events = packet.get("visible_events") or []
        if self.show_events and events:
            self.output_fn("最近可见事件：")
            for event in events[-self.show_events :]:
                payload = event.get("payload") or {}
                self.output_fn(
                    f"  #{event.get('seq')} {event.get('type')} "
                    + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))[:500]
                )
