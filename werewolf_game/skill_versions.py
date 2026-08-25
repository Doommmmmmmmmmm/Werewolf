"""保存每个训练 round 使用和产出的角色 skill 版本。"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .constants import ALL_ROLES
from .prompts import RoleStrategyStore
from .records import GAMES_PER_TRAINING_ROUND, FileGameRecordStore, RoundGameRecordStore


SKILL_VERSION_SCHEMA = 1
_ROUND_DIRECTORY = re.compile(r"round(\d+)$")


def _markdown_content(value: str) -> str:
    """将快照内容规范为唯一的 UTF-8 文件形式。"""

    return str(value).strip() + "\n"


def _sha256(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


class RoundSkillVersionStore:
    """维护一个 round 的输入/输出角色 skill 快照。

    输入快照是在本 round 第一局开始前实际提供给玩家 Agent 的角色档案；
    输出快照则是在十局复盘后会提供给下一 round 的档案。每个快照同时包含
    固定 ``base.md`` 和可更新 ``strategy.md``，使回放可独立复现。
    """

    def __init__(self, directory: str | Path = "records", *, round_index: int) -> None:
        self.directory = Path(directory).resolve()
        self.round_index = self._validated_round_index(round_index)
        self.round_directory = RoundGameRecordStore.round_directory_for(
            self.directory, self.round_index
        )
        self.skill_directory = self.round_directory / "skill"
        self.log_directory = self.round_directory / "log"
        self.manifest_path = self.log_directory / "skill-version.json"

    def ensure_input(
        self,
        *,
        roles: Iterable[str],
        strategy_store: RoleStrategyStore | None = None,
    ) -> dict[str, Any]:
        """保存本 round 的输入版本；已有快照绝不覆盖。"""

        store = strategy_store or RoleStrategyStore()
        normalized_roles = self._normalized_roles(roles)
        manifest = self._read_manifest()
        entries = manifest["roles"]
        changed = False

        for role in normalized_roles:
            existing = entries.get(role, {}).get("input")
            if isinstance(existing, dict):
                self._require_snapshot_files(existing, role=role, stage="input")
                continue

            profile = store.profile(role)
            base_content = _markdown_content(profile.base)
            strategy_content = _markdown_content(profile.strategy)
            previous = self._latest_output(role)
            version, source = self._input_version(
                previous=previous,
                base_content=base_content,
                strategy_content=strategy_content,
            )
            entries.setdefault(role, {})["input"] = self._write_snapshot(
                role=role,
                stage="input",
                version=version,
                source=source,
                base_content=base_content,
                strategy_content=strategy_content,
            )
            changed = True

        if changed:
            self._write_manifest(manifest)
        return deepcopy(manifest)

    def capture_output(
        self,
        *,
        roles: Iterable[str],
        strategy_store: RoleStrategyStore | None = None,
        review_results: Mapping[str, Mapping[str, Any]] | None = None,
        refresh_failed_review_outputs: bool = False,
    ) -> dict[str, Any]:
        """保存复盘后的输出版本；同一 round 只会写入一次。"""

        store = strategy_store or RoleStrategyStore()
        normalized_roles = self._normalized_roles(roles)
        manifest = self.ensure_input(roles=normalized_roles, strategy_store=store)
        entries = manifest["roles"]
        review_results = review_results or {}
        changed = False

        for role in normalized_roles:
            existing = entries.get(role, {}).get("output")
            if isinstance(existing, dict):
                self._require_snapshot_files(existing, role=role, stage="output")
                previous_source = existing.get("source", {})
                was_successfully_reviewed = isinstance(previous_source, dict) and bool(
                    previous_source.get("review_updated")
                )
                if not refresh_failed_review_outputs or was_successfully_reviewed:
                    continue

            input_entry = entries[role]["input"]
            profile = store.profile(role)
            base_content = _markdown_content(profile.base)
            strategy_content = _markdown_content(profile.strategy)
            has_changed = (
                input_entry.get("base_sha256") != _sha256(base_content)
                or input_entry.get("strategy_sha256") != _sha256(strategy_content)
            )
            input_version = self._entry_version(input_entry, role=role, stage="input")
            review = review_results.get(role, {})
            review_updated = bool(review.get("updated"))
            if has_changed:
                output_version = input_version + 1
                change_source = "role_review" if review_updated else "external_change"
            else:
                output_version = input_version
                change_source = "unchanged"
            output_entry = self._write_snapshot(
                role=role,
                stage="output",
                version=output_version,
                source={
                    "kind": change_source,
                    "input_version": input_version,
                    "review_updated": review_updated,
                    "review_error": str(review.get("error") or "") or None,
                },
                base_content=base_content,
                strategy_content=strategy_content,
            )
            entries[role]["output"] = output_entry
            changed = True

        if changed:
            self._write_manifest(manifest)
        return deepcopy(manifest)

    def is_round_complete(self) -> bool:
        """仅检查十局完整记录是否都已落盘。"""

        return all(
            (self.log_directory / f"full-game{game_index}.json").exists()
            for game_index in range(GAMES_PER_TRAINING_ROUND)
        )

    def _write_snapshot(
        self,
        *,
        role: str,
        stage: str,
        version: int,
        source: dict[str, Any],
        base_content: str,
        strategy_content: str,
    ) -> dict[str, Any]:
        role_directory = self.skill_directory / stage / role
        self._ensure_directories(role_directory)
        base_path = role_directory / "base.md"
        strategy_path = role_directory / "strategy.md"
        FileGameRecordStore._write_atomic(base_path, base_content)
        FileGameRecordStore._write_atomic(strategy_path, strategy_content)
        return {
            "version": version,
            "source": source,
            "base_path": str(base_path.relative_to(self.round_directory)),
            "strategy_path": str(strategy_path.relative_to(self.round_directory)),
            "base_sha256": _sha256(base_content),
            "strategy_sha256": _sha256(strategy_content),
        }

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "schema_version": SKILL_VERSION_SCHEMA,
                "round_index": self.round_index,
                "roles": {},
            }
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"skill 版本 manifest 无法读取：{self.manifest_path}") from error
        if not isinstance(manifest, dict):
            raise RuntimeError(f"skill 版本 manifest 格式错误：{self.manifest_path}")
        if manifest.get("round_index") != self.round_index:
            raise RuntimeError(f"skill 版本 manifest 的 round 不匹配：{self.manifest_path}")
        if not isinstance(manifest.get("roles"), dict):
            raise RuntimeError(f"skill 版本 manifest 缺少 roles：{self.manifest_path}")
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self._ensure_directories(self.skill_directory, self.log_directory)
        FileGameRecordStore._write_atomic(
            self.manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    def _latest_output(self, role: str) -> tuple[int, dict[str, Any]] | None:
        """查找当前 round 之前，该角色最后一个已保存的输出版本。"""

        if not self.directory.exists():
            return None
        latest: tuple[int, dict[str, Any]] | None = None
        for candidate_round in self.directory.iterdir():
            if not candidate_round.is_dir():
                continue
            match = _ROUND_DIRECTORY.fullmatch(candidate_round.name)
            if match is None:
                continue
            candidate_index = int(match.group(1))
            if candidate_index >= self.round_index:
                continue
            candidate_manifest = candidate_round / "log" / "skill-version.json"
            if not candidate_manifest.exists():
                continue
            try:
                data = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"历史 skill 版本 manifest 无法读取：{candidate_manifest}"
                ) from error
            output = data.get("roles", {}).get(role, {}).get("output")
            if not isinstance(output, dict):
                continue
            if latest is None or candidate_index > latest[0]:
                latest = (candidate_index, output)
        return latest

    def _input_version(
        self,
        *,
        previous: tuple[int, dict[str, Any]] | None,
        base_content: str,
        strategy_content: str,
    ) -> tuple[int, dict[str, Any]]:
        if previous is None:
            return 0, {"kind": "initial"}
        previous_round, previous_entry = previous
        previous_version = self._entry_version(
            previous_entry, role="历史角色", stage="output"
        )
        unchanged = (
            previous_entry.get("base_sha256") == _sha256(base_content)
            and previous_entry.get("strategy_sha256") == _sha256(strategy_content)
        )
        if unchanged:
            return previous_version, {
                "kind": "previous_round",
                "round_index": previous_round,
                "version": previous_version,
            }
        return previous_version + 1, {
            "kind": "external_change",
            "round_index": previous_round,
            "previous_version": previous_version,
        }

    def _require_snapshot_files(
        self, entry: dict[str, Any], *, role: str, stage: str
    ) -> None:
        for field in ("base_path", "strategy_path"):
            relative_path = entry.get(field)
            if not isinstance(relative_path, str):
                raise RuntimeError(f"{role} 的 {stage} skill 快照缺少 {field}")
            path = (self.round_directory / relative_path).resolve()
            try:
                path.relative_to(self.round_directory.resolve())
            except ValueError as error:
                raise RuntimeError(f"{role} 的 {stage} skill 快照路径非法") from error
            if not path.exists():
                raise RuntimeError(f"{role} 的 {stage} skill 快照不存在：{path}")

    @staticmethod
    def _entry_version(entry: dict[str, Any], *, role: str, stage: str) -> int:
        try:
            version = int(entry["version"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{role} 的 {stage} skill 版本格式错误") from error
        if version < 0:
            raise RuntimeError(f"{role} 的 {stage} skill 版本不能为负数")
        return version

    @staticmethod
    def _normalized_roles(roles: Iterable[str]) -> tuple[str, ...]:
        normalized = tuple(sorted({str(role) for role in roles}))
        invalid = [role for role in normalized if role not in ALL_ROLES]
        if invalid:
            raise ValueError(f"不支持的角色 skill：{'、'.join(invalid)}")
        return normalized

    @staticmethod
    def _validated_round_index(value: int) -> int:
        try:
            index = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("round_index 必须是非负整数") from error
        if index < 0 or str(index) != str(value).strip():
            raise ValueError("round_index 必须是非负整数")
        return index

    @staticmethod
    def _ensure_directories(*paths: Path) -> None:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.chmod(0o700)
            except PermissionError:
                pass
