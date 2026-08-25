"""把事件记录投影为公开记录或可读 Markdown 复盘。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ROLE_LABELS = {
    "wolf": "狼人",
    "villager": "平民",
    "seer": "预言家",
    "witch": "女巫",
    "guard": "守卫",
    "hunter": "猎人",
    "idiot": "白痴",
}
TEAM_LABELS = {"wolf": "狼人阵营", "village": "好人阵营"}
WITCH_ACTION_LABELS = {"witch_heal": "使用解药", "witch_poison": "使用毒药"}


def escape_markdown(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def player_index(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        player["id"]: player
        for player in record.get("metadata", {}).get("players", [])
    }


def player_name(player_id: str | None, players: dict[str, dict[str, Any]]) -> str:
    if not player_id:
        return "无人"
    player = players.get(player_id)
    if player is None:
        return str(player_id)
    return (
        f"{player_id}（{player['name']}）"
        if player.get("name") and player["name"] != player_id
        else player_id
    )


def role_name(role: str | None) -> str:
    return ROLE_LABELS.get(str(role), str(role or "未知"))


def team_name(team: str | None) -> str:
    return TEAM_LABELS.get(str(team), str(team or "未知"))


def _new_round(round_number: int) -> dict[str, Any]:
    return {
        "round": round_number,
        "wolf_speeches": [],
        "night_actions": [],
        "night_hunter_reactions": [],
        "dawn": None,
        "deaths": [],
        "sheriff_election": {
            "started": None,
            "candidates": [],
            "speeches": [],
            "passes": [],
            "votes": [],
            "events": [],
        },
        "day_speech_order": None,
        "night_last_words": [],
        "night_last_words_skipped": [],
        "night_badge_events": [],
        "day_speeches": [],
        "day_passes": [],
        "votes": [],
        "vote_result": None,
        "public_vote_outcome": None,
        "eliminated": None,
        "idiot_reveals": [],
        "day_hunter_reactions": [],
        "day_last_words": [],
        "day_last_words_skipped": [],
        "day_badge_events": [],
        "public_syncs": [],
    }


def rounds_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """按回合把扁平事件流整理为展示友好的结构。"""

    rounds: dict[int, dict[str, Any]] = {}

    def get_round(number: int) -> dict[str, Any]:
        if number not in rounds:
            rounds[number] = _new_round(number)
        return rounds[number]

    current_round = 1
    day_started = False
    for event in record.get("events", []):
        payload = event.get("payload", {})
        event_type = event.get("type")
        if event_type == "NIGHT_STARTED":
            current_round = int(payload.get("round", current_round))
            day_started = False
            get_round(current_round)
            continue
        if event_type == "DAY_STARTED":
            current_round = int(payload.get("round", current_round))
            day_started = True
            get_round(current_round)
            continue
        if event_type == "DAWN_ANNOUNCED":
            current_round = int(payload.get("round", current_round))
            get_round(current_round)["dawn"] = payload
            continue

        current = get_round(current_round)
        if event_type == "PLAYER_SPOKE":
            if event.get("channel") == "wolf":
                current["wolf_speeches"].append(payload)
            elif payload.get("stage") == "sheriff_election":
                current["sheriff_election"]["speeches"].append(payload)
            else:
                current["day_speeches"].append(payload)
        elif event_type == "PLAYER_PASSED" and event.get("channel") != "wolf":
            if payload.get("stage") == "sheriff_election":
                current["sheriff_election"]["passes"].append(payload)
            elif day_started:
                current["day_passes"].append(payload)
        elif event_type == "WOLF_TARGET_SELECTED":
            current["night_actions"].append({"kind": "狼人目标", **payload})
        elif event_type == "SEER_RESULT":
            current["night_actions"].append(
                {
                    "kind": "预言家查验",
                    "player_id": (event.get("recipients") or [None])[0],
                    **payload,
                }
            )
        elif event_type == "WITCH_ACTION_CONFIRMED":
            current["night_actions"].append(
                {
                    "kind": WITCH_ACTION_LABELS.get(payload.get("kind"), payload.get("kind", "女巫行动")),
                    "player_id": (event.get("recipients") or [None])[0],
                    **payload,
                }
            )
        elif event_type == "GUARD_ACTION_CONFIRMED":
            current["night_actions"].append(
                {
                    "kind": "守卫守护",
                    "player_id": (event.get("recipients") or [None])[0],
                    **payload,
                }
            )
        elif event_type in {"HUNTER_SHOT_FIRED", "HUNTER_SHOT_SKIPPED"}:
            reaction = {
                "kind": "猎人开枪" if event_type == "HUNTER_SHOT_FIRED" else "猎人放弃开枪",
                **payload,
            }
            if payload.get("trigger_phase") == "night_resolve":
                current["night_hunter_reactions"].append(reaction)
            else:
                current["day_hunter_reactions"].append(reaction)
        elif event_type == "VOTE_CAST":
            current["votes"].append(payload)
        elif event_type == "SHERIFF_ELECTION_STARTED":
            current["sheriff_election"]["started"] = payload
        elif event_type == "SHERIFF_CANDIDATES_ANNOUNCED":
            current["sheriff_election"]["candidates"] = list(payload.get("candidate_ids", []))
        elif event_type == "SHERIFF_VOTE_CAST":
            current["sheriff_election"]["votes"].append(payload)
        elif event_type.startswith("SHERIFF_ELECTION_") or event_type == "SHERIFF_ELECTED":
            current["sheriff_election"]["events"].append({"type": event_type, **payload})
        elif event_type == "DAY_SPEECH_ORDER_SET":
            current["day_speech_order"] = payload
        elif event_type == "PLAYER_LAST_WORDS":
            key = "night_last_words" if payload.get("trigger_phase") == "night_resolve" else "day_last_words"
            current[key].append(payload)
        elif event_type == "PLAYER_LAST_WORDS_SKIPPED":
            key = "night_last_words_skipped" if payload.get("trigger_phase") == "night_resolve" else "day_last_words_skipped"
            current[key].append(payload)
        elif event_type in {"SHERIFF_BADGE_TRANSFERRED", "SHERIFF_BADGE_DESTROYED"}:
            key = "night_badge_events" if payload.get("trigger_phase") == "night_resolve" else "day_badge_events"
            current[key].append({"type": event_type, **payload})
        elif event_type == "DAY_VOTE_RESOLVED":
            current["vote_result"] = payload
        elif event_type in {"VOTE_TIE", "NO_ONE_ELIMINATED"}:
            current["public_vote_outcome"] = {"type": event_type, **payload}
        elif event_type == "PLAYER_DIED":
            current["deaths"].append(payload)
        elif event_type == "PLAYER_ELIMINATED":
            current["eliminated"] = payload
        elif event_type == "IDIOT_REVEALED":
            current["idiot_reveals"].append(payload)

    for event in record.get("runner_events", []):
        if event.get("type") == "PUBLIC_STATE_SYNC":
            get_round(int(event["round"]))["public_syncs"].append(event)
    return [rounds[key] for key in sorted(rounds)]


def project_public_record(audit_record: dict[str, Any]) -> dict[str, Any]:
    """剥离种子、底牌、私密事件和审计状态，生成可交给真人的文件。"""

    def public_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
        if not snapshot:
            return None
        return {
            "latest_event_seq": snapshot.get("latest_event_seq"),
            "public_state": deepcopy(snapshot.get("public_state")),
        }

    return {
        "schema_version": 1,
        "record_type": "public",
        "record_id": audit_record.get("record_id"),
        "created_at": audit_record.get("created_at"),
        "updated_at": audit_record.get("updated_at"),
        "metadata": {
            "game_id": audit_record.get("metadata", {}).get("game_id"),
            "players": deepcopy(audit_record.get("metadata", {}).get("players", [])),
            "training_round": audit_record.get("metadata", {}).get("training_round"),
            "game_index": audit_record.get("metadata", {}).get("game_index"),
        },
        "events": [
            deepcopy(event)
            for event in audit_record.get("events", [])
            if event.get("visibility") == "public"
        ],
        "latest_snapshot": public_snapshot(audit_record.get("latest_snapshot")),
        "final_snapshot": public_snapshot(audit_record.get("final_snapshot")),
    }


def _final_public_state(record: dict[str, Any]) -> dict[str, Any]:
    final_snapshot = record.get("final_snapshot") or {}
    latest_snapshot = record.get("latest_snapshot") or {}
    return final_snapshot.get("public_state") or latest_snapshot.get("public_state") or {}


def _role_assignments(record: dict[str, Any]) -> dict[str, str]:
    deal = next(
        (event for event in record.get("events", []) if event.get("type") == "ROLE_ASSIGNMENTS_CREATED"),
        None,
    )
    if deal:
        return dict(deal.get("payload", {}).get("assignments", {}))
    final_snapshot = record.get("final_snapshot") or {}
    return dict(final_snapshot.get("audit_state", {}).get("roles", {}))


def _render_night_action(action: dict[str, Any], players: dict[str, dict[str, Any]]) -> str:
    kind = action.get("kind")
    if kind == "预言家查验":
        return (
            f"预言家 {player_name(action.get('player_id'), players)} 查验 "
            f"{player_name(action.get('target_id'), players)}，结果为：{team_name(action.get('team'))}"
        )
    if kind == "狼人目标":
        return f"狼人共同目标：{player_name(action.get('target_id'), players)}"
    return (
        f"{kind}：{player_name(action.get('player_id'), players)}"
        + (f" → {player_name(action.get('target_id'), players)}" if action.get("target_id") else "")
    )


def _render_hunter_reaction(
    reaction: dict[str, Any], players: dict[str, dict[str, Any]]
) -> str:
    hunter = player_name(reaction.get("player_id"), players)
    if reaction.get("kind") == "猎人放弃开枪":
        return f"猎人 {hunter} 放弃开枪。"
    return f"猎人 {hunter} 开枪带走 {player_name(reaction.get('target_id'), players)}。"


def _append_hunter_reactions(
    lines: list[str], reactions: list[dict[str, Any]], players: dict[str, dict[str, Any]]
) -> None:
    if not reactions:
        return
    lines.extend(["### 猎人反应", ""])
    lines.extend(f"- {_render_hunter_reaction(reaction, players)}" for reaction in reactions)
    lines.append("")


def _append_sheriff_election(
    lines: list[str], election: dict[str, Any], players: dict[str, dict[str, Any]]
) -> None:
    if not election.get("started"):
        return
    lines.extend(["### 警长竞选", ""])
    candidates = election.get("candidates", [])
    if candidates:
        lines.append(f"上警玩家：{'、'.join(player_name(item, players) for item in candidates)}。")
    else:
        lines.append("无人上警。")
    for speech in election.get("speeches", []):
        lines.append(
            f"- {player_name(speech.get('player_id'), players)}：{escape_markdown(speech.get('text') or '（跳过）')}"
        )
    for passed in election.get("passes", []):
        lines.append(f"- {player_name(passed.get('player_id'), players)}：竞选发言选择跳过。")
    if election.get("votes"):
        lines.extend(["", "| 投票玩家 | 投给 |", "| --- | --- |"])
        for vote in election["votes"]:
            lines.append(
                f"| {escape_markdown(player_name(vote.get('voter_id'), players))} | {escape_markdown(player_name(vote.get('target_id'), players))} |"
            )
    for event in election.get("events", []):
        event_type = event.get("type")
        if event_type == "SHERIFF_ELECTED":
            lines.append(
                f"警长当选：{player_name(event.get('player_id'), players)}（{event.get('vote_weight')} 票）。"
            )
        elif event_type == "SHERIFF_ELECTION_TIED":
            labels = "、".join(player_name(item, players) for item in event.get("candidate_ids", []))
            lines.append(f"警长竞选平票，以下候选人进入 PK：{labels}。")
        elif event_type == "SHERIFF_ELECTION_FAILED":
            lines.append("警长竞选再次平票或无人投票，本局无警长。")
    lines.append("")


def _append_last_words_and_badges(
    lines: list[str],
    *,
    last_words: list[dict[str, Any]],
    last_words_skipped: list[dict[str, Any]],
    badge_events: list[dict[str, Any]],
    players: dict[str, dict[str, Any]],
) -> None:
    if badge_events:
        lines.extend(["### 警徽交接", ""])
        for event in badge_events:
            if event.get("type") == "SHERIFF_BADGE_TRANSFERRED":
                lines.append(
                    f"- {player_name(event.get('from_player_id'), players)} 将警徽传给 {player_name(event.get('to_player_id'), players)}。"
                )
            else:
                lines.append(f"- {player_name(event.get('player_id'), players)} 撕毁警徽。")
        lines.append("")
    if last_words or last_words_skipped:
        lines.extend(["### 遗言", ""])
        for speech in last_words:
            lines.append(
                f"- {player_name(speech.get('player_id'), players)}：{escape_markdown(speech.get('text') or '（跳过）')}"
            )
        for skipped in last_words_skipped:
            lines.append(f"- {player_name(skipped.get('player_id'), players)}：放弃遗言。")
        lines.append("")


def _append_day_speech_order(
    lines: list[str], order: dict[str, Any] | None, players: dict[str, dict[str, Any]]
) -> None:
    if not order:
        return
    direction = "下一位起顺序" if order.get("direction") == "next" else "上一位起逆序"
    reference = (
        f"死者 {order.get('reference_seat')} 号"
        if order.get("reference_is_dawn_death")
        else "1 号"
    )
    ordered_names = "、".join(
        player_name(item, players) for item in order.get("speaker_ids", [])
    )
    lines.extend([f"发言顺序：从{reference}{direction}，依次为 {ordered_names}。", ""])


def _append_votes(lines: list[str], votes: list[dict[str, Any]], players: dict[str, dict[str, Any]]) -> None:
    weighted = any(float(vote.get("weight", 1)) != 1 for vote in votes)
    if weighted:
        lines.extend(["| 投票玩家 | 投给 | 票权 |", "| --- | --- | ---: |"])
    else:
        lines.extend(["| 投票玩家 | 投给 |", "| --- | --- |"])
    for vote in votes:
        row = (
            f"| {escape_markdown(player_name(vote.get('voter_id'), players))} | {escape_markdown(player_name(vote.get('target_id'), players))} |"
        )
        if weighted:
            row += f" {vote.get('weight', 1)} |"
        lines.append(row)


def render_audit_markdown(record: dict[str, Any]) -> str:
    """完整管理员复盘，包含发牌和夜间私密行动。"""

    players = player_index(record)
    state = _final_public_state(record)
    roles = _role_assignments(record)
    lines = [
        "# 狼人杀完整复盘",
        "",
        "> 审计版：包含底牌、狼人私聊和夜间私密行动。请仅在可信环境中保存和阅读。",
        "",
        "## 基本信息",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| 对局 ID | {escape_markdown(record.get('metadata', {}).get('game_id') or record.get('record_id'))} |",
        f"| 随机种子 | {escape_markdown(record.get('metadata', {}).get('seed') or '未记录')} |",
        f"| 获胜阵营 | {escape_markdown(team_name(state.get('winner')))} |",
        f"| 结束轮次 | {escape_markdown(state.get('round') or '未结束')} |",
        "",
        "## 发牌与身份",
        "",
        "| 座位 | 玩家 | 身份 |",
        "| --- | --- | --- |",
    ]
    for player in sorted(players.values(), key=lambda item: item["seat"]):
        lines.append(
            f"| {player['seat']} | {escape_markdown(player_name(player['id'], players))} | {escape_markdown(role_name(roles.get(player['id'])))} |"
        )
    lines.append("")

    _append_model_token_usage(lines, record.get("model_token_usage"))

    for round_data in rounds_from_record(record):
        number = round_data["round"]
        lines.extend([f"## 第 {number} 夜", ""])
        if round_data["public_syncs"]:
            recipients = round_data["public_syncs"][0].get("recipients", [])
            lines.extend([f"公开状态已同步给：{'、'.join(player_name(item, players) for item in recipients)}。", ""])
        lines.extend(["### 狼人私聊", ""])
        if not round_data["wolf_speeches"]:
            lines.extend(["本夜没有狼人私聊记录。", ""])
        else:
            for speech in round_data["wolf_speeches"]:
                lines.append(f"- {player_name(speech.get('player_id'), players)}：{escape_markdown(speech.get('text') or '（跳过）')}")
            lines.append("")
        lines.extend(["### 夜间行动（审计）", ""])
        if not round_data["night_actions"]:
            lines.extend(["无可见夜间行动记录。", ""])
        else:
            lines.extend(f"- {_render_night_action(action, players)}" for action in round_data["night_actions"])
            lines.append("")
        _append_sheriff_election(lines, round_data["sheriff_election"], players)
        _append_public_day_sections(lines, round_data, players)

    lines.extend(["## 结局", "", f"获胜阵营：{team_name(state.get('winner'))}。", ""])
    return "\n".join(lines)


def _append_model_token_usage(lines: list[str], usage: object) -> None:
    """在管理员复盘中以简洁表格展示每局玩家行动的 API usage。"""

    if not isinstance(usage, dict):
        return
    availability = {
        "complete": "完整",
        "partial": "部分可得（含未报告或重试请求）",
        "unavailable": "服务未返回 usage",
        "not_applicable": "本局未调用模型",
    }.get(str(usage.get("availability") or ""), "未知")
    lines.extend(
        [
            "## 模型 Token 使用（仅玩家行动）",
            "",
            "| 项目 | 数值 |",
            "| --- | ---: |",
            f"| 可用性 | {availability} |",
            f"| 输入 Token | {usage.get('input_tokens', 0)} |",
            f"| 输出 Token | {usage.get('output_tokens', 0)} |",
            f"| 合计 Token | {usage.get('total_tokens', 0)} |",
            f"| 已报告 usage 的响应 | {usage.get('reported_usage_response_count', 0)} |",
            f"| API 尝试 | {usage.get('api_attempt_count', 0)} |",
            "",
        ]
    )


def _append_public_day_sections(
    lines: list[str], round_data: dict[str, Any], players: dict[str, dict[str, Any]]
) -> None:
    number = round_data["round"]
    lines.extend(["### 天亮结果", ""])
    dawn = round_data["dawn"]
    if not dawn:
        lines.extend(["本轮未进入天亮结算。", ""])
    else:
        dead = dawn.get("dead_player_ids", [])
        message = (
            f"死亡玩家：{'、'.join(player_name(item, players) for item in dead)}。"
            if dead
            else "平安夜，无玩家死亡。"
        )
        lines.extend([message, ""])
    _append_hunter_reactions(lines, round_data["night_hunter_reactions"], players)
    _append_last_words_and_badges(
        lines,
        last_words=round_data["night_last_words"],
        last_words_skipped=round_data["night_last_words_skipped"],
        badge_events=round_data["night_badge_events"],
        players=players,
    )
    lines.extend([f"## 第 {number} 天", "", "### 公开发言", ""])
    _append_day_speech_order(lines, round_data["day_speech_order"], players)
    if not round_data["day_speeches"] and not round_data["day_passes"]:
        lines.extend(["无白天发言记录。", ""])
    else:
        for speech in round_data["day_speeches"]:
            lines.append(f"- {player_name(speech.get('player_id'), players)}：{escape_markdown(speech.get('text') or '（跳过）')}")
        for passed in round_data["day_passes"]:
            lines.append(f"- {player_name(passed.get('player_id'), players)}：选择跳过发言。")
        lines.append("")
    lines.extend(["### 投票", ""])
    if not round_data["votes"]:
        lines.extend(["无投票记录。", ""])
        _append_hunter_reactions(lines, round_data["day_hunter_reactions"], players)
        _append_last_words_and_badges(
            lines,
            last_words=round_data["day_last_words"],
            last_words_skipped=round_data["day_last_words_skipped"],
            badge_events=round_data["day_badge_events"],
            players=players,
        )
        return
    _append_votes(lines, round_data["votes"], players)
    lines.append("")
    if round_data["eliminated"]:
        lines.extend([f"出局玩家：{player_name(round_data['eliminated'].get('player_id'), players)}。", ""])
    elif round_data["idiot_reveals"]:
        reveal = round_data["idiot_reveals"][-1]
        label = player_name(reveal.get("player_id"), players)
        vote_notice = "，失去投票权" if reveal.get("lost_vote_right") else "，仍保有投票权"
        lines.extend([f"{label} 亮明白痴身份，免于出局{vote_notice}。", ""])
    elif len((round_data["vote_result"] or {}).get("tied_targets", [])) > 1:
        tied = round_data["vote_result"]["tied_targets"]
        lines.extend([f"投票平票：{'、'.join(player_name(item, players) for item in tied)}。", ""])
    else:
        lines.extend(["本轮无人出局。", ""])
    _append_hunter_reactions(lines, round_data["day_hunter_reactions"], players)
    _append_last_words_and_badges(
        lines,
        last_words=round_data["day_last_words"],
        last_words_skipped=round_data["day_last_words_skipped"],
        badge_events=round_data["day_badge_events"],
        players=players,
    )


def render_public_markdown(record: dict[str, Any]) -> str:
    """真人可读的公开复盘，不含任何隐私事件。"""

    players = player_index(record)
    state = _final_public_state(record)
    lines = [
        "# 狼人杀公开记录",
        "",
        "> 本记录只包含游戏公开信息；不含发牌、狼人私聊、私密夜间目标或私密技能结果。",
        "",
        "## 基本信息",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| 对局 ID | {escape_markdown(record.get('metadata', {}).get('game_id') or record.get('record_id'))} |",
        f"| 获胜阵营 | {escape_markdown(team_name(state.get('winner')))} |",
        f"| 结束轮次 | {escape_markdown(state.get('round') or '进行中')} |",
        "",
        "## 参与玩家",
        "",
    ]
    for player in sorted(players.values(), key=lambda item: item["seat"]):
        lines.append(f"- {player['seat']} 号：{player_name(player['id'], players)}")
    lines.append("")

    for round_data in rounds_from_record(record):
        number = round_data["round"]
        lines.extend([f"## 第 {number} 夜", ""])
        _append_sheriff_election(lines, round_data["sheriff_election"], players)
        lines.extend(["### 天亮结果", ""])
        dawn = round_data["dawn"]
        if not dawn:
            lines.extend(["本轮尚未进入天亮结算。", ""])
        else:
            dead = dawn.get("dead_player_ids", [])
            if not dead:
                lines.extend(["平安夜，无玩家死亡。", ""])
            else:
                labels = []
                for player_id in dead:
                    death = next(
                        (item for item in round_data["deaths"] if item.get("player_id") == player_id),
                        {},
                    )
                    role = death.get("role")
                    labels.append(
                        player_name(player_id, players)
                        + (f"（{role_name(role)}）" if role else "")
                    )
                lines.extend([f"死亡玩家：{'、'.join(labels)}。", ""])
        _append_hunter_reactions(lines, round_data["night_hunter_reactions"], players)
        _append_last_words_and_badges(
            lines,
            last_words=round_data["night_last_words"],
            last_words_skipped=round_data["night_last_words_skipped"],
            badge_events=round_data["night_badge_events"],
            players=players,
        )
        _append_public_day_sections_without_dawn(lines, round_data, players)

    roles = _public_roles(record)
    if roles:
        lines.extend(["## 终局身份揭示", "", "| 玩家 | 身份 |", "| --- | --- |"])
        for player in sorted(players.values(), key=lambda item: item["seat"]):
            lines.append(
                f"| {escape_markdown(player_name(player['id'], players))} | {escape_markdown(role_name(roles.get(player['id'])))} |"
            )
        lines.append("")
    lines.extend(["## 结局", "", f"获胜阵营：{team_name(state.get('winner'))}。", ""])
    return "\n".join(lines)


def _append_public_day_sections_without_dawn(
    lines: list[str], round_data: dict[str, Any], players: dict[str, dict[str, Any]]
) -> None:
    number = round_data["round"]
    lines.extend([f"## 第 {number} 天", "", "### 公开发言", ""])
    _append_day_speech_order(lines, round_data["day_speech_order"], players)
    if not round_data["day_speeches"] and not round_data["day_passes"]:
        lines.extend(["无白天发言记录。", ""])
    else:
        for speech in round_data["day_speeches"]:
            lines.append(f"- {player_name(speech.get('player_id'), players)}：{escape_markdown(speech.get('text') or '（跳过）')}")
        for passed in round_data["day_passes"]:
            lines.append(f"- {player_name(passed.get('player_id'), players)}：选择跳过发言。")
        lines.append("")
    lines.extend(["### 投票", ""])
    if not round_data["votes"]:
        lines.extend(["无投票记录。", ""])
        _append_hunter_reactions(lines, round_data["day_hunter_reactions"], players)
        _append_last_words_and_badges(
            lines,
            last_words=round_data["day_last_words"],
            last_words_skipped=round_data["day_last_words_skipped"],
            badge_events=round_data["day_badge_events"],
            players=players,
        )
        return
    _append_votes(lines, round_data["votes"], players)
    lines.append("")
    if round_data["eliminated"]:
        role = round_data["eliminated"].get("role")
        label = player_name(round_data["eliminated"].get("player_id"), players)
        lines.extend([f"出局玩家：{label}{f'（{role_name(role)}）' if role else ''}。", ""])
    elif round_data["idiot_reveals"]:
        reveal = round_data["idiot_reveals"][-1]
        label = player_name(reveal.get("player_id"), players)
        vote_notice = "，失去投票权" if reveal.get("lost_vote_right") else "，仍保有投票权"
        lines.extend([f"{label} 亮明白痴身份，免于出局{vote_notice}。", ""])
    elif (round_data["public_vote_outcome"] or {}).get("type") == "VOTE_TIE":
        tied = round_data["public_vote_outcome"].get("tied_target_ids", [])
        lines.extend([f"投票平票：{'、'.join(player_name(item, players) for item in tied)}。", ""])
    else:
        lines.extend(["本轮无人出局。", ""])
    _append_hunter_reactions(lines, round_data["day_hunter_reactions"], players)
    _append_last_words_and_badges(
        lines,
        last_words=round_data["day_last_words"],
        last_words_skipped=round_data["day_last_words_skipped"],
        badge_events=round_data["day_badge_events"],
        players=players,
    )


def _public_roles(record: dict[str, Any]) -> dict[str, str]:
    finished = next(
        (event for event in record.get("events", []) if event.get("type") == "GAME_FINISHED"),
        None,
    )
    if finished:
        return dict(finished.get("payload", {}).get("roles", {}))
    return dict(_final_public_state(record).get("roles_revealed", {}))
