"""
user_simulator.py — Simulates user responses via LLM call.

THE CORE RULE:
  The simulator must be COMPLETELY INDEPENDENT of the ground truth builder.
  They must never share code, never call the same function, never look at
  the same data.  If they are independent and agree, the score is 1.0.
  If the agent drifts off-script, they will disagree, the judge fires,
  and the score reflects actual quality.

HOW IT WORKS:
  Given the live AI message and the member entity data, an LLM generates
  a realistic caller response.  The LLM is given:
    - The AI's actual message (what the agent just said)
    - Member data (name, DOB, member ID, etc.) as grounding facts
    - The scenario description (e.g. "this is a clarification scenario
      where the caller initially hesitates on the ZIP code")
    - A persona: a real health insurance member calling for help

  The LLM does NOT see:
    - The static transcript
    - The ground truth
    - The slot classification
    - Any other eval infrastructure

  This means:
    - When the agent asks "Can I get your first name?" the simulator
      says "emily" (or "Emily" or "it's Emily") based on the entity.
    - When the agent asks "May I have your Member ID?" the simulator
      says "m nine zero seven five zero three".
    - When the scenario calls for a correction, the simulator provides
      the wrong value first (guided by the scenario description).
    - When the agent asks something unexpected, the simulator responds
      naturally to whatever was actually said.

  The judge then compares this to the transcript ground truth.  Agreement
  = the agent is behaving correctly.  Disagreement = something is wrong
  with either the agent or the simulator, and the score will reflect it.

SCENARIO GUIDANCE:
  Each scenario gets a short description injected into the simulator
  prompt.  This is the ONLY way the simulator knows to behave differently
  across scenarios.  It does NOT share any data with ground_truth_builder.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from agent.logger import get_logger

logger = get_logger(__name__)

_SIMULATOR_MAX_TOKENS = 80


SCENARIO_PERSONAS: Dict[str, str] = {
    "pcp_happy_path": (
        "You are a cooperative caller. You answer every question clearly and "
        "correctly the first time. You confirm things when asked."
    ),
    "pcp_clarification_zip": (
        "You are a cooperative caller, but when the agent reads back your ZIP "
        "code you hesitate — say something like 'Hmm, let me think about that' "
        "or 'I'm not sure'. After the agent re-confirms the ZIP, you agree it "
        "is correct. For all other questions, answer normally."
    ),
    # "pcp_correction_first_name": (
    #     "You are a cooperative caller. When the agent first asks for your first "
    #     "name, say 'emily'. When the agent next asks for your last name, instead "
    #     "correct yourself: say your first name is actually Emilia (E-M-I-L-I-A) "
    #     "and your last name is Carter. After that, answer all other questions "
    #     "normally."
    # ),
    "pcp_correction_member_id": (
        "You are a cooperative caller. When first asked for your Member ID, "
        "give a slightly wrong one: say 'm nine oh seven five oh two' (wrong). "
        "When the agent then asks for your date of birth, correct yourself "
        "first: say 'Sorry, that's wrong — it's m nine zero seven five zero "
        "three'. Then give your date of birth. Answer all other questions "
        "normally."
    ),
    "pcp_clarification_fax": (
        "You are a cooperative caller. When the agent reads back your fax "
        "number, express doubt — say something like 'I'm not sure that's the "
        "right number'. After the agent asks for the correct fax number, "
        "provide it. For all other questions, answer normally."
    ),
    "claim_adjustment_happy_path": (
        "You are James Wilson, a cooperative health insurance member following up on a claim adjustment. "
        "Your member ID is M310188 (spoken: m three one zero one eight eight). "
        "Your date of birth is July 30, 1977 (spoken: Thirtieth of July, nineteen seventy seven). "
        "Your phone on file is 512-555-6101 — confirm it when asked. "
        "Your reference number is 42695817. "
        "Your email is james.wilson@gmail.com — confirm it when asked. "
        "When asked how you want to provide records, say your doctor will send them. "
        "When offered an upload link to your email, say yes please. "
        "When the upload link is confirmed sent and you are offered Personal Guide outreach, say yes, please arrange that. "  # noqa: E501
        "For the FIRST notification preference question (about provider outreach status), say you want updates by phone (SMS). "  # noqa: E501
        "Confirm the phone number 512-555-6101 when it is read back to you. "
        "When asked how long it will take, ask 'how long will it take to finalize the request?'. "
        "For the SECOND notification preference question (about progress updates or timeline updates), say 'email them to me'. "  # noqa: E501
        "At the follow-up 'anything else' question, ask where you can see rewards from your annual checkup. "
        "Then say no that is it, thanks."
    ),
    "claim_adjustment_no_proceed": (
        "You are Michael Brown, a cooperative health insurance member following up on a claim adjustment. "
        "Your member ID is M662130 (spoken: m six six two one three zero). "
        "Your date of birth is March 18, 1986 (spoken: eighteenth of March, nineteen eighty six). "
        "Your phone on file is 312-555-2201 — confirm it when asked. "
        "Your reference number is 12491584. "
        "When asked how to provide records, say you will send them. "
        "When offered an upload link, decline it (say no thanks). "
        "When offered Personal Guide outreach, decline it (say no i dont want to proceed)."
    ),
    "claim_adjustment_upload_only": (
        "You are a cooperative caller doing a claim follow-up. "
        "When asked about records, say you'll upload them yourself. "
        "Accept the upload link. Confirm your email when asked. "
        "Decline Personal Guide outreach. "
        "Choose email for notifications. Confirm email on file. "
        "When asked about timeline, ask how long it takes. "
        "Choose SMS for the second notification preference. "
        "Say no more questions at follow-up."
    ),
    "claim_adjustment_guide_only": (
        "You are a cooperative caller doing a claim follow-up. "
        "When asked about records, immediately say you want the Personal Guide to contact your doctor. "
        "Give explicit consent for Personal Guide outreach. "
        "Choose SMS for notifications. Confirm phone on file. "
        "When asked about timeline, ask how long it takes. "
        "Choose email for the second notification. "
        "Say no more questions at follow-up."
    ),
}

_DEFAULT_PERSONA = "You are a cooperative caller. Answer every question clearly and correctly."


def _build_simulator_prompt(entity_data: dict, scenario_tag: str) -> str:
    persona = SCENARIO_PERSONAS.get(scenario_tag, _DEFAULT_PERSONA)

    if scenario_tag.startswith("claim_"):
        facts = (
            f"Your first name is {entity_data.get('first_name', 'James')}. "
            f"Your last name is {entity_data.get('last_name', 'Wilson')}. "
            f"Your member ID is {entity_data.get('member_id', 'M310188')} "
            f"(spoken aloud: {entity_data.get('member_id_spoken', 'm three one zero one eight eight')}). "
            f"Your date of birth is {entity_data.get('date_of_birth', 'July 30, 1977')} "
            f"(spoken: {entity_data.get('dob_spoken', 'Thirtieth of July, nineteen seventy seven')}). "
            f"Your phone number on file is {entity_data.get('phone_number', '512-555-6101')} — confirm it when asked. "  # noqa: E501
            f"Your reference number is {entity_data.get('reference_number', '42695817')}. "
            f"Your email on file is {entity_data.get('email', 'james.wilson@gmail.com')} — confirm it when asked."  # noqa: E501
        )
    else:
        facts = (
            f"Your first name is {entity_data.get('first_name', 'Emily')}. "
            f"Your last name is {entity_data.get('last_name', 'Carter')}. "
            f"Your member ID is {entity_data.get('member_id', 'M907503')} "
            f"(spoken aloud: m nine zero seven five zero three). "
            f"Your date of birth is April 12, 1988 "
            f"(spoken: April twelfth nineteen eighty-eight). "
            f"You are calling for yourself (the plan holder). "
            f"You are looking for a Primary Care Physician. "
            f"Your ZIP code is {entity_data.get('zip_code', '12139')}. "
            f"Your fax number is {entity_data.get('fax_number', '6175554199')} "
            f"(spoken: six one seven five five five four one nine nine). "
            f"You would like the provider list sent to your fax. "
            f"When asked about benefits or Care Coach, say yes."
        )

    return (
        "You are roleplaying as a health insurance member on a phone call.\n\n"
        f"Your facts:\n{facts}\n\n"
        f"Your behaviour for this call:\n{persona}\n\n"
        "Instructions:\n"
        "- Respond ONLY as the caller would speak. No narration, no stage "
        "directions, no quotation marks.\n"
        "- Keep responses short (1-2 sentences maximum).\n"
        "- Speak naturally as a real caller would over the phone.\n"
        "- Do not explain your reasoning.\n"
        "- Do not say anything other than your response to the agent."
    )


def _call_simulator_llm(ai_message: str, system_prompt: str) -> str:
    try:
        from langchain_openai import AzureChatOpenAI

        from agent.llm.config import Config

        llm = AzureChatOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.OPENAI_API_VERSION,
            azure_deployment=Config.WORKER_DEPLOYMENT,
            temperature=0.3,
            max_tokens=_SIMULATOR_MAX_TOKENS,
            streaming=False,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Agent said: {ai_message}\n\nYour response:",
            },
        ]

        result = llm.invoke(messages)
        text = (result.content or "").strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1].strip()

        return text or "Okay."

    except Exception:
        logger.exception("simulate_user_response: LLM call failed")
        return "Okay."


async def simulate_user_response_async(
    ai_message: str,
    entity,
    flow: str = "pcp",
    scenario_tag: str = "",
    turn_counters: Optional[Dict[Tuple[str, str], int]] = None,
) -> str:
    """
    Async: generates a realistic caller response via LLM.
    Completely independent of build_dynamic_ground_truth.
    """
    try:
        entity_data = entity.model_dump() if hasattr(entity, "model_dump") else dict(entity)
    except Exception:
        entity_data = {}

    system_prompt = _build_simulator_prompt(entity_data, scenario_tag)

    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _call_simulator_llm,
        ai_message,
        system_prompt,
    )


def simulate_user_response(
    ai_message: str,
    entity,
    flow: str = "pcp",
    scenario_tag: str = "",
    turn_counters: Optional[Dict[Tuple[str, str], int]] = None,
) -> str:
    """
    Sync wrapper.
    MUST NEVER call build_dynamic_ground_truth or read the static transcript.
    """
    try:
        entity_data = entity.model_dump() if hasattr(entity, "model_dump") else dict(entity)
    except Exception:
        entity_data = {}

    system_prompt = _build_simulator_prompt(entity_data, scenario_tag)
    return _call_simulator_llm(ai_message, system_prompt)
