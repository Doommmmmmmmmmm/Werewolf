"""狼人杀游戏引擎和参与者接口。"""

from .core.engine import GameEngine
from .agents.llm import ModelRequestCoordinator
from .prompts import RoleProfile, RoleProfileStore, load_role_profile
from .recording.recorder import LlmPublicNarrator, PublicRecorder
from .recording.records import FileGameRecordStore, RoundGameRecordStore
from .core.rules import RuleSet, create_default_rules, create_rules_for_player_count
from .core.runner import GameRunner
from .agents.task_agent import TaskAgent
from .agents.participants import (
    HumanParticipant,
    LlmParticipant,
    Participant,
    ScriptedParticipant,
    TaskAgentParticipant,
)

__all__ = [
    "GameEngine",
    "GameRunner",
    "HumanParticipant",
    "LlmParticipant",
    "TaskAgent",
    "TaskAgentParticipant",
    "LlmPublicNarrator",
    "ModelRequestCoordinator",
    "Participant",
    "PublicRecorder",
    "RoleProfile",
    "RoleProfileStore",
    "RoundGameRecordStore",
    "FileGameRecordStore",
    "RuleSet",
    "ScriptedParticipant",
    "create_default_rules",
    "create_rules_for_player_count",
    "load_role_profile",
]
