"""狼人杀多 Agent 框架的 Python 入口。"""

from .engine import GameEngine
from .harness import (
    HarnessFileStore,
    HarnessRuntime,
    HarnessSpec,
    HarnessStaticEvaluation,
    TaskAgentHarness,
    TacticalCard,
    default_harness_for_profile,
    evaluate_harness,
)
from .evaluation import (
    SkillTestGameRecordStore,
    SkillTestRunner,
    SkillTestScenario,
    SkillTestSource,
)
from .llm import ModelRequestCoordinator
from .prompts import FrozenRoleStrategyStore, RoleStrategyStore
from .recorder import LlmPublicNarrator, PublicRecorder
from .records import RoundGameRecordStore
from .review import RoleStrategyReviewer
from .research import JsonSearchProvider, ResearchSource, SearchProvider
from .meta_agent import (
    CandidateAssessment,
    HarnessArchiveStore,
    MetaAgentConfig,
    MetaAgentResult,
    MetaAgent,
    ReplayAnalysis,
    TaskAgentMetaAgent,
)
from .rules import RuleSet, create_default_rules, create_rules_for_player_count
from .runner import GameRunner
from .skill_versions import RoundSkillVersionStore
from .training import GameRoundRunner, review_completed_round
from .participants import (
    HumanParticipant,
    LlmParticipant,
    Participant,
    ScriptedParticipant,
    TaskAgentParticipant,
)

__all__ = [
    "GameEngine",
    "HarnessFileStore",
    "HarnessRuntime",
    "HarnessSpec",
    "HarnessStaticEvaluation",
    "TaskAgentHarness",
    "TacticalCard",
    "default_harness_for_profile",
    "evaluate_harness",
    "GameRoundRunner",
    "GameRunner",
    "HumanParticipant",
    "LlmParticipant",
    "TaskAgentParticipant",
    "LlmPublicNarrator",
    "ModelRequestCoordinator",
    "PublicRecorder",
    "RoleStrategyReviewer",
    "JsonSearchProvider",
    "ResearchSource",
    "SearchProvider",
    "CandidateAssessment",
    "HarnessArchiveStore",
    "MetaAgentConfig",
    "MetaAgentResult",
    "MetaAgent",
    "ReplayAnalysis",
    "TaskAgentMetaAgent",
    "RoleStrategyStore",
    "FrozenRoleStrategyStore",
    "Participant",
    "RoundGameRecordStore",
    "RoundSkillVersionStore",
    "SkillTestGameRecordStore",
    "SkillTestRunner",
    "SkillTestScenario",
    "SkillTestSource",
    "ScriptedParticipant",
    "RuleSet",
    "create_default_rules",
    "create_rules_for_player_count",
    "review_completed_round",
]
