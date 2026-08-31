"""狼人杀游戏引擎和参与者接口。"""

from .engine import GameEngine
from .llm import ModelRequestCoordinator
from .prompts import RoleProfile, RoleProfileStore, load_role_profile
from .recorder import LlmPublicNarrator, PublicRecorder
from .records import FileGameRecordStore, RoundGameRecordStore
from .rules import RuleSet, create_default_rules, create_rules_for_player_count
from .runner import GameRunner
from .task_agent import TaskAgent
from .participants import (
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
