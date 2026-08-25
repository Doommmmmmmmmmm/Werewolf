"""人类、脚本或 LLM 玩家使用的统一接口。"""

from .base import Participant, ScriptedParticipant
from .human import HumanParticipant
from .llm import LlmParticipant, TaskAgentParticipant

__all__ = [
    "HumanParticipant",
    "LlmParticipant",
    "TaskAgentParticipant",
    "Participant",
    "ScriptedParticipant",
]
