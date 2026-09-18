"""
agent.py — Graceful call closure. Uses pick() for varied messages.
"""

from __future__ import annotations

from typing import Optional

from agent.core.agent import BaseAgent
from agent.logger import get_logger
from agent.state import State
from agent.utils import name_part as get_name_part
from agent.utils import pick

_DONE_KEYWORDS = {
    "no",
    "nope",
    "nah",
    "no thank",
    "no thanks",
    "that's all",
    "thats all",
    "that's it",
    "thats it",
    "nothing else",
    "nothing more",
    "all good",
    "all set",
    "i'm good",
    "im good",
    "i'm done",
    "im done",
    "done",
    "good",
    "ok",
    "okay",
    "great",
    "perfect",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "have a good",
    "have a great",
}

_INTENT_KEYWORDS = {
    "provider": "provider_services",
    "doctor": "provider_services",
    "network": "provider_services",
    "specialist": "provider_services",
    "in-network": "provider_services",
    "claim": "claim_services",
    "claims": "claim_services",
    "reimbursement": "claim_services",
    "adjustment": "claim_services",
    "denial": "claim_services",
    "eob": "claim_services",
    "benefit": "benefits_inquiry",
    "benefits": "benefits_inquiry",
    "coverage": "benefits_inquiry",
    "deductible": "benefits_inquiry",
    "care coach": "care_wellness",
    "wellness": "care_wellness",
    "reward": "care_wellness",
    "rewards": "care_wellness",
    "incentive": "care_wellness",
    "points": "care_wellness",
}

_QUESTION_TEMPLATES = [
    "Is there anything else I can help you with today{name_part}?",
    "Is there anything else I can assist you with{name_part}?",
    "Do you have any other questions{name_part}?",
    "Is there anything else on your mind{name_part}?",
    "What else can I help you with today{name_part}?",
]

_GOODBYE_MESSAGE = [
    "Thank you for calling Sagility Health. Have a great day{name_part}!",
    "Thank you for calling{name_part}. Take care and have a wonderful day!",
    "Thanks for calling Sagility Health{name_part}. Be safe and enjoy the rest of your day!",
    "It was a pleasure helping you today{name_part}. Have a great rest of your day!",
    "Thank you{name_part}. Take care — have a wonderful day!",
]

logger = get_logger(__name__)


class ClosureAgent(BaseAgent):
    AGENT_NAME = "closure_agent"

    async def run(self, state: State) -> dict:
        np = get_name_part(state)
        goodbye = pick(_GOODBYE_MESSAGE).format(name_part=np)
        result = self.signal_complete(
            state,
            message=goodbye,
            closure_requested=True,
            reasoning="Closure confirmed by prior agent — delivering goodbye",
        )
        result["next_node"] = "END"
        return result

    def _classify(self, text: str) -> tuple[bool, Optional[str]]:
        lower = text.lower().strip()
        for keyword, intent in _INTENT_KEYWORDS.items():
            if keyword in lower:
                return False, intent
        done = any(lower == kw or (" " in kw and lower.startswith(kw)) for kw in _DONE_KEYWORDS)
        return done, None


async def closure_agent(state: State) -> dict:
    return await ClosureAgent.from_state(state).execute(state)
