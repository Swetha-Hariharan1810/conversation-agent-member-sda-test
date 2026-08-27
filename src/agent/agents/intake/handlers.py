"""
handlers.py — Unclear intent handling for IntakeAgent.

Uses LLM 2 (Gemini, generate_recovery_message) to produce a natural
recovery sentence. Falls back to static strings on exception.
"""

from __future__ import annotations

from agent.agents.intake.constants import (
    INTENT_SLOT,
    LOG_INTENT_UNCLEAR,
    LOG_MAX_RETRY,
    LOG_OUT_OF_SCOPE,
    MAX_CLARIFICATION_ATTEMPTS,
    OUT_OF_SCOPE_REASON,
    UNCLEAR_ESCALATION,
    UNCLEAR_INTENT_REASON,
)
from agent.logger import get_logger
from agent.state import State
from agent.utils import _last_user_msg

logger = get_logger(__name__)


def _get_clarification_attempts(state: State) -> int:
    slot = (state.get("slot_attempts") or {}).get(INTENT_SLOT, {})
    intent_attempts = slot if isinstance(slot, int) else slot.get("attempt_count", 0)
    offtopic_turns = state.get("offtopic_global_count") or 0
    return intent_attempts + offtopic_turns


async def handle_unclear_intent(agent, state: State, result=None) -> dict:
    attempts = _get_clarification_attempts(state)

    if attempts >= MAX_CLARIFICATION_ATTEMPTS:
        logger.warning(LOG_MAX_RETRY)
        return agent.signal_escalate(
            state=state,
            message=UNCLEAR_ESCALATION,
            reason=UNCLEAR_INTENT_REASON,
            initiator="Agent",
        )

    agent.slot_fail(INTENT_SLOT, is_asr=False)
    logger.info(LOG_INTENT_UNCLEAR, extra={"attempt": attempts + 1})
    messages = list(state.get("messages") or [])

    # if attempts == 0:
    #     from agent.agents.intake.constants import UNCLEAR_FIRST_ATTEMPT_MSGS
    #     from agent.utils import pick

    #     msg = pick(UNCLEAR_FIRST_ATTEMPT_MSGS)
    #     return agent.ask_member(state, msg)

    from agent.llm.response_generator import generate_recovery_message

    if attempts == 0:
        # First failure — pure open question, no hint yet
        # Caller may know exactly what they want, just said it vaguely
        label_override = "the caller's reason for calling today — ask warmly and openly"
    else:
        # Second failure — caller genuinely does not know what this service offers
        # Natural hint is appropriate now, but must not sound like a phone tree menu
        # Frame it as "I can help with X or Y — what brings you in?" not "Press 1 for X"
        label_override = (
            "the caller's reason for calling — they seem unsure what this service offers. "
            "Mention naturally that you can help with finding an in-network doctor "
            "or following up on a health insurance claim, then invite them to share "
            "what they need. Keep it warm and conversational, not a menu."
        )
    msg = await generate_recovery_message(
        slot_name="intent",
        attempt=attempts + 1,
        guard="RETRY",
        last_messages=messages[-4:],
        slot_label_override=label_override,
        caller_name=None,
        confirmed_slots={},
        user_utterance=_last_user_msg(messages),
    )
    return agent.ask_member(state, msg)


def _match_out_of_scope_routing(utterance: str) -> tuple[str, str, str]:
    """
    Match the caller's utterance against the keyword routing table.
    Returns (matched_phrase, team, number).
    Falls back to defaults if no keyword matches.
    Runs in <0.1ms — no LLM call, no token cost.
    """
    from agent.agents.intake.constants import (
        OUT_OF_SCOPE_FALLBACK_NUMBER,
        OUT_OF_SCOPE_FALLBACK_TEAM,
        OUT_OF_SCOPE_KEYWORD_ROUTING,
    )

    t = (utterance or "").lower()
    for keyword, team, number in OUT_OF_SCOPE_KEYWORD_ROUTING:
        if keyword in t:
            return keyword, team, number
    return "", OUT_OF_SCOPE_FALLBACK_TEAM, OUT_OF_SCOPE_FALLBACK_NUMBER


async def handle_out_of_scope_intent(agent, state: State, result=None) -> dict:
    """
    Called when the extraction LLM classifies intent as out_of_scope.

    Team and phone number are resolved by keyword-matching the caller's
    last utterance — no LLM extraction needed, no silent fallback risk.
    Routes directly to END (not escalation_agent) so the caller hears
    exactly one message, not the out-of-scope message followed by the
    escalation agent's ref number message.
    """
    import random

    from agent.agents.intake.constants import (
        OUT_OF_SCOPE_MSG_TEMPLATES,
    )
    from agent.utils import _last_user_msg

    logger.warning(LOG_OUT_OF_SCOPE)

    # Get the caller's last utterance from state messages
    messages = list(state.get("messages") or [])
    last_user = _last_user_msg(messages)

    # Keyword match — O(n) over a small fixed list, <0.1ms
    keyword, team, number = _match_out_of_scope_routing(last_user)

    # Build topic description from matched keyword or generic fallback
    if keyword:
        # Make it sound natural: "billing" → "your billing question"
        topic_description = f"your {keyword} question"
    else:
        topic_description = "your request"

    msg = random.choice(OUT_OF_SCOPE_MSG_TEMPLATES).format(
        topic_description=topic_description,
        team=team,
        number=number,
    )

    # Route directly to END — do not go through escalation_agent.
    # The message already says "I'll connect you now" and gives the number.
    # escalation_agent would add a second message with a reference number.
    result = agent.ask_member(state, msg)
    result["next_node"] = "END"
    result["escalation_reason"] = OUT_OF_SCOPE_REASON
    result["is_interrupt"] = False
    return result


def _extract_provider_type_from_utterance(utterance: str) -> str:
    """
    Extract a readable provider type label from the caller's raw utterance
    for the {provider_type} placeholder in the escalation message.

    Uses word-boundary regex matching so short keywords like "ent" never
    falsely match inside common words (e.g. "efficient", "different").
    Falls back to "this provider type" if nothing recognisable is found.
    """
    import re

    _UNSUPPORTED_KEYWORDS: list[tuple[str, str]] = [
        ("oncologist", "Oncologist"),
        ("neurologist", "Neurologist"),
        ("radiologist", "Radiologist"),
        ("ophthalmologist", "Ophthalmologist"),
        ("urologist", "Urologist"),
        ("psychiatrist", "Psychiatrist"),
        ("psychologist", "Psychologist"),
        ("podiatrist", "Podiatrist"),
        ("gastroenterologist", "Gastroenterologist"),
        ("rheumatologist", "Rheumatologist"),
        ("endocrinologist", "Endocrinologist"),
        ("nephrologist", "Nephrologist"),
        ("pulmonologist", "Pulmonologist"),
        ("hematologist", "Hematologist"),
        ("allergist", "Allergist"),
        ("immunologist", "Immunologist"),
        ("pain management", "Pain Management Specialist"),
        ("physical therapist", "Physical Therapist"),
        ("occupational therapist", "Occupational Therapist"),
        ("speech therapist", "Speech Therapist"),
        ("obgyn", "OB-GYN"),
        ("ob-gyn", "OB-GYN"),
        ("gynecologist", "Gynecologist"),
        ("obstetrician", "Obstetrician"),
        # "ent" is 3 chars — word-boundary match prevents false hits inside
        # common words like "efficient", "different", "prevent", etc.
        ("otolaryngologist", "Otolaryngologist"),
        ("plastic surgeon", "Plastic Surgeon"),
        ("oral surgeon", "Oral Surgeon"),
        ("vascular", "Vascular Specialist"),
        ("surgeon", "Surgeon"),
        ("dentist", "Dentist"),
        ("optometrist", "Optometrist"),
        ("chiropractor", "Chiropractor"),
        ("audiologist", "Audiologist"),
        ("therapist", "Therapist"),
        ("ent", "ENT Specialist"),
    ]
    t = (utterance or "").lower()
    for keyword, label in _UNSUPPORTED_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", t):
            return label
    return "this provider type"


async def handle_unsupported_provider_type(agent, state: State, result=None) -> dict:
    """
    Called when the extraction LLM classifies intent as provider_type_unsupported.

    Immediately routes to escalation_agent with a pre-message that names the
    unsupported specialty and lists the five supported types. No verification
    runs — the member hears one clear message before being transferred.

    Design: uses signal_escalate() (not ask_member + END) so escalation_agent
    appends a reference number and the call ends through the normal transfer path.
    This is identical to the pattern in provider_search_agent.
    """
    import random

    from agent.agents.intake.constants import (
        LOG_PROVIDER_TYPE_UNSUPPORTED,
        PROVIDER_TYPE_UNSUPPORTED_ESCALATION,
        PROVIDER_TYPE_UNSUPPORTED_REASON,
    )
    from agent.utils import _last_user_msg

    logger.warning(LOG_PROVIDER_TYPE_UNSUPPORTED)

    messages = list(state.get("messages") or [])
    last_user = _last_user_msg(messages)

    provider_type = _extract_provider_type_from_utterance(last_user)
    msg = random.choice(PROVIDER_TYPE_UNSUPPORTED_ESCALATION).format(provider_type=provider_type)

    return agent.signal_escalate(
        state=state,
        message=msg,
        reason=PROVIDER_TYPE_UNSUPPORTED_REASON,
        initiator="Agent",
    )
