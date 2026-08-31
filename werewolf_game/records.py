"""对局记录存储：同时维护管理员审计版和公开版。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from .replay import project_public_record, render_audit_markdown, render_public_markdown


GAMES_PER_ROUND = 10


def safe_file_part(value: object) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "game")).strip("_")
    return cleaned or "game"


class FileGameRecordStore:
    """每次 append 都原子更新审计 JSON/Markdown 和公开 JSON/Markdown。"""

    def __init__(self, directory: str | Path = "records", record_id: str | None = None) -> None:
        self.directory = Path(directory).resolve()
        self.record_id = record_id
        self.audit_json_path: Path | None = None
        self.audit_markdown_path: Path | None = None
        self.public_json_path: Path | None = None
        self.public_markdown_path: Path | None = None
        self.record: dict[str, Any] | None = None

    def start(self, metadata: dict[str, Any]) -> Path:
        if self.record is not None:
            assert self.audit_json_path is not None
            return self.audit_json_path
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.directory.chmod(0o700)
        except PermissionError:
            pass
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now_utc.microsecond // 1000:03d}Z"
        self.record_id = self.record_id or f"{safe_file_part(metadata.get('game_id'))}-{timestamp}-{os.getpid()}"
        stem = safe_file_part(self.record_id)
        self.audit_json_path = self.directory / f"{stem}.json"
        self.audit_markdown_path = self.directory / f"{stem}.md"
        self.public_json_path = self.directory / f"{stem}.public.json"
        self.public_markdown_path = self.directory / f"{stem}.public.md"
        now = datetime.now(timezone.utc).isoformat()
        self.record = {
            "schema_version": 1,
            "record_id": self.record_id,
            "created_at": now,
            "updated_at": now,
            "metadata": deepcopy(metadata),
            "events": [],
            "runner_events": [],
            "model_token_usage": None,
            "latest_snapshot": None,
            "final_snapshot": None,
        }
        self._write_all()
        return self.audit_json_path

    def append(
        self,
        *,
        events: list[dict[str, Any]],
        runner_events: list[dict[str, Any]],
        snapshot: dict[str, Any],
        model_token_usage: dict[str, Any] | None = None,
    ) -> None:
        if self.record is None:
            raise RuntimeError("必须先调用 start()")
        known_sequences = {event.get("seq") for event in self.record["events"]}
        for event in events:
            if event.get("seq") not in known_sequences:
                self.record["events"].append(deepcopy(event))
                known_sequences.add(event.get("seq"))
        self.record["runner_events"].extend(deepcopy(runner_events))
        if model_token_usage is not None:
            self.record["model_token_usage"] = deepcopy(model_token_usage)
        self.record["latest_snapshot"] = deepcopy(snapshot)
        if snapshot.get("public_state", {}).get("status") == "finished":
            self.record["final_snapshot"] = deepcopy(snapshot)
        self.record["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_all()

    def paths(self) -> dict[str, str | None]:
        return {
            "record_path": str(self.audit_json_path) if self.audit_json_path else None,
            "record_markdown_path": str(self.audit_markdown_path) if self.audit_markdown_path else None,
            "public_record_path": str(self.public_json_path) if self.public_json_path else None,
            "public_record_markdown_path": str(self.public_markdown_path) if self.public_markdown_path else None,
        }

    def _write_all(self) -> None:
        assert self.record is not None
        assert self.audit_json_path and self.audit_markdown_path
        assert self.public_json_path and self.public_markdown_path
        self._write_atomic(self.audit_json_path, json.dumps(self.record, ensure_ascii=False, indent=2) + "\n")
        self._write_atomic(self.audit_markdown_path, render_audit_markdown(self.record))
        public_record = project_public_record(self.record)
        self._write_atomic(self.public_json_path, json.dumps(public_record, ensure_ascii=False, indent=2) + "\n")
        self._write_atomic(self.public_markdown_path, render_public_markdown(public_record))

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)


class RoundGameRecordStore(FileGameRecordStore):
    """按 round 和 game 编号写入稳定命名的对局文件。"""

    def __init__(
        self,
        directory: str | Path = "records",
        *,
        round_index: int,
        game_index: int,
    ) -> None:
        self.round_index = self._validated_index(round_index, "round_index")
        self.game_index = self._validated_index(game_index, "game_index")
        if self.game_index >= GAMES_PER_ROUND:
            raise ValueError(f"game_index 必须在 0–{GAMES_PER_ROUND - 1} 之间")
        super().__init__(
            directory=directory,
            record_id=f"round{self.round_index}-game{self.game_index}",
        )

    @property
    def round_directory(self) -> Path:
        return self.round_directory_for(self.directory, self.round_index)

    @property
    def public_directory(self) -> Path:
        return self.round_directory / "public"

    @property
    def full_directory(self) -> Path:
        return self.round_directory / "full"

    @property
    def log_directory(self) -> Path:
        return self.round_directory / "log"

    @classmethod
    def round_directory_for(cls, directory: str | Path, round_index: int) -> Path:
        index = cls._validated_index(round_index, "round_index")
        return Path(directory).resolve() / f"round{index}"

    @classmethod
    def next_available(cls, directory: str | Path = "records") -> tuple[int, int]:
        """寻找下一个未占用的 ``roundN/gameM`` 槽位。"""

        root = Path(directory).resolve()
        round_index = 0
        while True:
            log_directory = cls.round_directory_for(root, round_index) / "log"
            for game_index in range(GAMES_PER_ROUND):
                if not (log_directory / f"full-game{game_index}.json").exists():
                    return round_index, game_index
            round_index += 1

    def start(self, metadata: dict[str, Any]) -> Path:
        if self.record is not None:
            assert self.audit_json_path is not None
            return self.audit_json_path

        for path in (
            self.round_directory,
            self.public_directory,
            self.full_directory,
            self.log_directory,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.chmod(0o700)
            except PermissionError:
                pass

        game_name = f"game{self.game_index}"
        self.audit_json_path = self.log_directory / f"full-{game_name}.json"
        self.audit_markdown_path = self.full_directory / f"{game_name}.md"
        self.public_json_path = self.log_directory / f"public-{game_name}.json"
        self.public_markdown_path = self.public_directory / f"{game_name}.md"
        occupied_paths = (
            self.audit_json_path,
            self.audit_markdown_path,
            self.public_json_path,
            self.public_markdown_path,
        )
        if any(path.exists() for path in occupied_paths):
            raise FileExistsError(
                f"round{self.round_index}/game{self.game_index} 已有记录；请选择新的编号"
            )
        now = datetime.now(timezone.utc).isoformat()
        record_metadata = deepcopy(metadata)
        record_metadata["round"] = self.round_index
        record_metadata["game_index"] = self.game_index
        self.record = {
            "schema_version": 1,
            "record_id": self.record_id,
            "created_at": now,
            "updated_at": now,
            "metadata": record_metadata,
            "events": [],
            "runner_events": [],
            "model_token_usage": None,
            "latest_snapshot": None,
            "final_snapshot": None,
        }
        self._write_all()
        return self.audit_json_path

    @staticmethod
    def _validated_index(value: int, name: str) -> int:
        try:
            index = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 必须是非负整数") from error
        if index < 0 or str(index) != str(value).strip():
            raise ValueError(f"{name} 必须是非负整数")
        return index
