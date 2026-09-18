"""
agent_signal.py — AgentSignal: the structured return type for all agent decisions.

AgentStatus values drive orchestrator routing:
  COMPLETE  → return to orchestrator for next-agent decision
  ESCALATE  → route immediately to escalation_agent
  BLOCKED   → route immediately to escalation_agent (unrecoverable error)

AgentSignal is serialized via to_state_dict() into state["last_agent_signal"]
so the orchestrator and fast-path router can inspect it without importing agents.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    COMPLETE = "complete"
    BLOCKED = "blocked"
    ESCALATE = "escalate"


class AgentSignal(BaseModel):
    status: AgentStatus
    resolved_intents: List[str] = Field(default_factory=list)
    new_intent_detected: Optional[str] = None
    closure_requested: bool = False
    context_updates: Dict[str, Any] = Field(default_factory=dict)
    proactive_offer_available: bool = False
    escalation_reason: Optional[str] = None
    reasoning: Optional[str] = None

    def to_state_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_state_dict(cls, d: dict) -> "AgentSignal":
        if not d:
            return cls(status=AgentStatus.COMPLETE)
        try:
            return cls(**d)
        except Exception:
            return cls(status=AgentStatus.COMPLETE)
