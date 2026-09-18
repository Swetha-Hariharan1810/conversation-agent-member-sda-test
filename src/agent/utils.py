import importlib.resources
import random
import re
import re as _re
from functools import lru_cache
from typing import Any

from agent.core.constants import WAIT_PATTERNS
from agent.logger import get_logger

logger = get_logger(__name__)


# Maintainer notes inside a prompt file. Markdown hides them from a reader;
# nothing hid them from the model, which was handed every word of them on every
# turn. _followup_contract.md opens with thirty-five lines explaining which
# transcript made its rules necessary — useful to the next person to edit it,
# and to the extraction model exactly the kind of competing text the rules
# below it are trying to stand out from.
_PROMPT_COMMENT_RE = _re.compile(r"<!--.*?-->", _re.DOTALL)


@lru_cache(maxsize=32)
def read_prompt(file_path: str) -> str:
    """Read a prompt file, minus anything written for a human reader."""
    try:
        raw = importlib.resources.files("agent").joinpath("prompts", file_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Prompt not found: %s — %s", file_path, e)
        return ""
    return _PROMPT_COMMENT_RE.sub("", raw).strip()


def clean_asr_input(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\b(um+|uh+|er+|hmm+)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def human_join(items: list, *, final: str = "or") -> str:
    items = [i.strip() for i in items if i and i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {final} {items[1]}"
    return f"{', '.join(items[:-1])}, {final} {items[-1]}"


def _last_assistant_msg(messages: list) -> str:
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role in ("assistant", "ai"):
            return (content or "").strip()
    return ""


def _last_user_msg(messages: list) -> str:
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role in ("user", "human"):
            return (content or "").strip()
    return ""


def _last_agent_question(messages: list) -> str:
    """The question the last agent turn left on the table, if it asked one.

    A guard that declines an off-topic request has to hand the caller back to
    something. The question already asked is that something — and it is the
    only source that is right for every step, including the yes/no offers
    ("would you also like the office-visit benefits?") that have no field to
    ask for and that a generated re-ask has been seen to invent an ask for.
    """
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if str(role).lower() not in ("assistant", "ai"):
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (content or "").strip()) if s.strip()]
        questions = [s for s in sentences if s.endswith("?")]
        return questions[-1] if questions else ""
    return ""


def build_history(messages: list, n: int = None) -> list[str]:
    """Build a compact turn-by-turn history for LLM context."""
    if n is None:
        from agent.core.constants import HISTORY_WINDOW_SIZE

        n = HISTORY_WINDOW_SIZE
    result = []

    for m in messages[-n:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")

        normalized_role = str(role).lower()

        if normalized_role in {"user", "human"}:
            speaker = "Caller"
        elif normalized_role in {"assistant", "ai"}:
            speaker = "AI"
        else:
            speaker = normalized_role or "Unknown"

        result.append(f"{speaker}: {content}")

    return result


# Fallback only — primary transfer detection is handled by the LLM guard in guards.py.
_TRANSFER_PATTERNS: dict[str, list[str]] = {
    "explicit_transfer": [
        "transfer me",
        "transfer to",
        "connect me to",
        "put me through",
        "patch me through",
    ],
    "human_agent_request": [
        "human agent",
        "live agent",
        "real agent",
        "actual agent",
        "live person",
        "real person",
        "actual person",
        "human being",
    ],
    "speak_to_someone": [
        "speak to someone",
        "speak to a person",
        "speak to an agent",
        "speak to a human",
        "speak to a representative",
        "speak to a rep",
        "speak to a supervisor",
        "speak to a manager",
        "speak to an operator",
    ],
    "talk_to_someone": [
        "talk to someone",
        "talk to a person",
        "talk to an agent",
        "talk to a human",
        "talk to a representative",
        "talk to a rep",
        "talk to a supervisor",
        "talk to a manager",
        "talk to an operator",
        "get a person",
        "get an agent",
        "get a human",
        "get a supervisor",
        "talk to representative",
        "talk to rep",
    ],
    "end_or_exit": [
        "end call",
        "end the call",
        "hang up",
        "just cancel",
        "cancel this",
    ],
    "frustration_signals": [
        "you're not helping",
        "youre not helping",
        "you are not helping",
        "this isn't working",
        "this is not working",
        "this isnt working",
        "no one is helping",
        "nobody is helping",
    ],
}


def detect_transfer_request(state: Any) -> bool:
    msgs = state.get("messages", []) if isinstance(state, dict) else []
    if not msgs:
        return False
    last = msgs[-1]
    content = (last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")).lower()
    all_patterns = [p for group in _TRANSFER_PATTERNS.values() for p in group]
    return any(p in content for p in all_patterns)


# The side-question rules, composed into every extraction prompt from ONE file.
#
# They used to be written out inside each of the three headers, three different
# ways — and header_core.md did not describe answered_with_followup at all — so
# whether a caller's question was heard depended on which header the slot they
# were on happened to use. Nineteen agent prompts and three headers cannot be
# kept in step by hand, so they are no longer asked to be: the contract is read
# from one place and every builder below composes the same copy.
_FOLLOWUP_CONTRACT = "extraction/_followup_contract.md"

# The read-back rules, composed into every extraction prompt from ONE file.
#
# Same story as the follow-up contract above, one layer over: the rules were
# written inside delivery_management.md after a caller was read their own stale
# fax back, and the six other prompts that collect a confirmation slot never got
# them. So "yeah, but actually I want to change it" was a decline on the fax and
# a confirmation on the ZIP, and the ZIP turn carried on to delivery with the
# value the caller was replacing still on file.
_CONFIRMATION_CONTRACT = "extraction/_confirmation_contract.md"

# The turn classification and the return shape, composed into every extraction
# prompt from ONE file.
#
# The third of these, and the largest: EVENT_TYPE, WAIT, CROSS-CALL REQUESTS,
# LOCKED FIELDS, NEEDS FREEFORM RESPONSE, CANNOT PROVIDE, FALLBACK PIVOT and a
# RETURN block used to be written out per header, in three dialects, for eleven
# schema fields — six of which restated each other. What the model reports now
# is one intent and one target, described once. See the file's own header
# comment for what came out and why.
_INTENT_CONTRACT = "extraction/_intent_contract.md"

# Three prompts are bound to a different structured-output schema, so the
# intent contract's return shape is not theirs to fill. They are composed
# without it.
#
# ssn_fallback.md returns SsnFallbackResult; follow_up.md and
# follow_up_claims.md return FollowUpResult, which classifies a request with no
# slot being collected around it and so reports its own request_kind.
# Each already describes its own fields; what they used to get on top was a
# header RETURN block for a schema they do not use.
_NON_WORKER_RESULT_PROMPTS: frozenset[str] = frozenset(
    {
        "extraction/ssn_fallback.md",
        "extraction/follow_up.md",
        "extraction/follow_up_claims.md",
    }
)


def _intent_contract_for(agent_prompt_file: str) -> str:
    """The shared turn contract, or nothing for a prompt on another schema."""
    if agent_prompt_file in _NON_WORKER_RESULT_PROMPTS:
        return ""
    return read_prompt(_INTENT_CONTRACT)


@lru_cache(maxsize=36)
def build_extraction_prompt(agent_prompt_file: str) -> str:
    """
    System prompt for LLM 1 (get_extraction_llm).
    Combines extraction header + the shared follow-up contract + agent-specific
    rules only. No global behavioural rules — extraction LLM does not need them.
    """
    global_prompt = read_prompt("system/global_extraction.md")
    header = read_prompt("extraction/header.md")
    followup = read_prompt(_FOLLOWUP_CONTRACT)
    intent = _intent_contract_for(agent_prompt_file)
    confirmation = read_prompt(_CONFIRMATION_CONTRACT)
    agent = read_prompt(agent_prompt_file)
    parts = (global_prompt, header, followup, intent, confirmation, agent)
    return "\n\n---\n\n".join(p for p in parts if p) + "\n\n"


@lru_cache(maxsize=36)
def build_extraction_prompt_core(agent_prompt_file: str) -> str:
    """
    Minimal extraction prompt for agents that only need guard detection
    and simple field extraction (no spelling handling).

    Use for: intake, benefits, care_wellness.

    Still the smallest of the three tiers, but no longer the one that skips the
    side-question rules: those are composed in from the shared contract, the
    same copy every other tier gets. Dropping them here is what made a caller's
    question audible on one slot and inaudible on the next.
    """
    global_prompt = read_prompt("system/global_extraction.md")
    core_header = read_prompt("extraction/header_core.md")
    followup = read_prompt(_FOLLOWUP_CONTRACT)
    intent = _intent_contract_for(agent_prompt_file)
    confirmation = read_prompt(_CONFIRMATION_CONTRACT)
    agent = read_prompt(agent_prompt_file)
    parts = (global_prompt, core_header, followup, intent, confirmation, agent)
    return "\n\n---\n\n".join(p for p in parts if p) + "\n\n"


@lru_cache(maxsize=36)
def build_extraction_prompt_extraction(agent_prompt_file: str) -> str:
    """
    Mid-tier extraction prompt for agents that collect structured slot
    values and need the grounding and confidence rules, but not the full
    verification machinery (SPELL_CONFIRM).

    Use for: provider_search, delivery_management, follow_up.
    Input tokens: ~380-500 (vs ~1050-1200 with full header).
    """
    global_prompt = read_prompt("system/global_extraction.md")
    extraction_header = read_prompt("extraction/header_extraction.md")
    followup = read_prompt(_FOLLOWUP_CONTRACT)
    intent = _intent_contract_for(agent_prompt_file)
    confirmation = read_prompt(_CONFIRMATION_CONTRACT)
    agent = read_prompt(agent_prompt_file)
    parts = (global_prompt, extraction_header, followup, intent, confirmation, agent)
    return "\n\n---\n\n".join(p for p in parts if p) + "\n\n"


@lru_cache(maxsize=36)
def build_generation_prompt(guard: str = "RETRY") -> str:
    """
    System prompt for LLM 2 (response generator) and orchestrator.
    Used by response_generator.py and app_graph.py (warmup).

    Assembled per guard label (Phase 5): global rules + shared recovery base
    (identity, tone, variation, slot discipline, hard rules) + the matching
    events/<guard>.md section. Unknown guards fall back to the RETRY section.
    All file reads are cached (read_prompt lru_cache + this function's own).
    """
    global_prompt = read_prompt("system/global_generation.md")
    base_prompt = read_prompt("generation/recovery_base.md")
    event_file = f"generation/events/{(guard or 'RETRY').lower()}.md"
    event_prompt = read_prompt(event_file) or read_prompt("generation/events/retry.md")
    return f"{global_prompt}\n\n---\n\n{base_prompt}\n\n---\n\n{event_prompt}"


@lru_cache(maxsize=16)
def build_system_prompt(agent_prompt_file: str) -> str:
    """
    Backward-compatible shim. Routes to the correct builder by path prefix.

    BUG FIX: previously called build_generation_prompt(agent_prompt_file)
    but that function takes no arguments — would raise TypeError at runtime.
    """
    if agent_prompt_file.startswith("generation/"):
        return build_generation_prompt()  # no argument
    return build_extraction_prompt(agent_prompt_file)


# ---------------------------------------------------------------------------
# Humanized message variation helpers (merged from utils_humanize.py)
# ---------------------------------------------------------------------------


# Openers a message carries because it is usually the whole turn. Two pools
# concatenated then open twice:
#
#   MSG_DOCTOR_DIRECT_ACK  "Sure, that's fine."
#   MSG_UPLOAD_OFFER       "Sure. I can also send a secure link..."
#   spoken                 "Sure, that's fine.  Sure. I can also send..."
#
# One in six combinations of those two pools stutters, and another reads "Sure,
# that's fine. Thank You. I can also...". The openers are right when the offer
# stands alone and wrong the moment something acknowledges ahead of it, so they
# are dropped from the second half rather than removed from the pools.
_LEADING_OPENER_RE = _re.compile(
    r"^(?:sure|of\s+course|certainly|absolutely|okay|ok|alright|"
    r"thank\s+you|thanks|great|perfect|no\s+problem)\b[\s,.!—-]*",
    _re.IGNORECASE,
)


def join_turn(first: str, second: str, *, separator: str = "\n\n") -> str:
    """Join two spoken messages without opening twice.

    ``first`` keeps its opener; ``second`` loses one if it has one, because
    something has already acknowledged for it. Either side being empty returns
    the other unchanged, so a call site does not have to check.
    """
    lead = (first or "").strip()
    tail = (second or "").strip()
    if not lead:
        return tail
    if not tail:
        return lead
    trimmed = _LEADING_OPENER_RE.sub("", tail, count=1).lstrip()
    # Only take the trim when something is left to say — an offer that is
    # nothing but its opener keeps it.
    if trimmed:
        tail = trimmed[0].upper() + trimmed[1:] if trimmed[0].islower() else trimmed
    return f"{lead}{separator}{tail}"


def pick(pool) -> str:
    """Randomly select from a pool, or return the string as-is."""
    if isinstance(pool, list):
        return random.choice(pool) if pool else ""
    return pool or ""


def name_part(source) -> str:
    """
    Return ', FirstName' if a first name is known, else ''.
    source: State dict, ConversationContext dict, or plain name string.
    """
    if isinstance(source, dict):
        name = source.get("caller_first_name") or source.get("first_name") or ""
    elif isinstance(source, str):
        name = source
    else:
        name = ""
    name = (name or "").strip().title()
    return f", {name}" if name else ""


# ---------------------------------------------------------------------------
# Spoken-form helpers for AI/voice messages
# APPEND these two functions to src/agent/utils.py (re is already imported).
#
# Requirement: any email address or website spoken in an AI message must be
# fully spelled out in words — "@" → "at" and every "." → "dot" — so TTS
# reads the address verbatim instead of pronouncing it as a word.
# ---------------------------------------------------------------------------


def speak_email(email: str | None) -> str:
    """
    Convert an email address to its fully spoken form for AI messages.

        "jane.doe@example.com" → "jane dot doe at example dot com"

    Replaces "@" with " at " AND every "." with " dot ".
    Use this ONLY for the spoken/display string — never store or write
    the spoken form back to state or Salesforce.
    """
    if not email:
        return ""
    spoken = email.strip().replace("@", " at ").replace(".", " dot ")
    return re.sub(r"\s+", " ", spoken).strip()


def speak_url(url: str | None) -> str:
    """
    Convert a website URL to its fully spoken form for AI messages.

        "www.mysagilityhealth.com" → "www dot mysagilityhealth dot com"
        "https://example.com/portal" → "example dot com slash portal"

    Strips the scheme, then spells out "." as "dot" and "/" as "slash".
    Use this ONLY for the spoken/display string.
    """
    if not url:
        return ""
    spoken = url.strip().replace("https://", "").replace("http://", "")
    spoken = spoken.rstrip("/")
    spoken = spoken.replace(".", " dot ").replace("/", " slash ")
    return re.sub(r"\s+", " ", spoken).strip()


# ---------------------------------------------------------------------------
# "Cannot provide" detector — keyword-based, zero latency, no LLM cost.
#
# Returns True when the caller is explicitly stating they do NOT have the
# requested value (as opposed to giving a wrong answer or being garbled).
#
# Called in _collect_slot (core/slot_manager.py) BEFORE slot_fail() so
# that "I don't have it" → immediate escalation, not 3 pointless retries.
# Also called directly in claim_adjustment_agent.py for the manual
# reference_number loop.
#
# Design notes:
#   - All patterns require a first-person ownership phrase so plain "no"
#     (a legitimate confirmation response) is never caught.
#   - Compiled once at import time — <1µs per call at runtime.
#
# What "first-person ownership" has to mean here. A match ends the call: every
# call site in _collect_slot escalates on it without a retry, so a pattern that
# also fits an ordinary sentence hangs up on a caller who was cooperating. Two
# rules keep that from happening, and both are load-bearing:
#
#   - A verb of loss or non-receipt names WHAT was lost, from the nouns that
#     carry an identifier (_IDENTIFIER_OBJECT). "I lost my card" is a denial;
#     "I lost my job" is why they are calling, and an unrestricted "my" cannot
#     tell the two apart.
#   - "I don't know" counts alone, as the whole turn. Bare, it is filler at
#     least as often as refusal — "I don't know, is it the one ending in
#     5309?" is an answer — so it denies only when the turn says nothing else.
#     Qualified by an object ("I don't know the claim number") it is a denial
#     wherever it appears.
# ---------------------------------------------------------------------------

# Nouns that stand for the thing being asked for. A caller who has lost one of
# these has lost the identifier; a caller who has lost anything else has told
# us about their life. Kept deliberately concrete — no "thing", no "stuff".
_IDENTIFIER_OBJECT = (
    r"(?:it|that|this|one|those|them"
    r"|(?:my|the|a|an|any)\s+(?:\w+\s+){0,2}"
    r"(?:card|cards|i\.?d\.?|ids|member\s*i\.?d\.?|number|numbers|letter|letters"
    r"|paper|papers|paperwork|document|documents|documentation|form|forms|statement"
    r"|statements|bill|bills|mail|envelope|policy|plan|reference|claim|insurance"
    r"|info|information)"
    r")"
)

_CANNOT_PROVIDE_PATTERNS: list = [
    _re.compile(p, _re.IGNORECASE)
    for p in [
        # "don't / doesn't have" variants
        r"\bi\s+(?:do\s+not|don'?t)\s+have\b",
        r"\bi\s+don'?t\s+have\s+(it|that|my\b|the\b|a\b)",
        r"\bdon'?t\s+have\s+(it|that)\b",
        # adverb-inserted: "I just / simply / really / actually don't have/know"
        r"\bi\s+(?:just|simply|really|actually|honestly|unfortunately|currently)\s+don'?t\s+(have|know)\b",
        # standalone "don't have the/a/my/that/any X" — e.g. "I just don't have the reference number"
        r"\bdon'?t\s+have\s+(?:the|a|my|that|any)\b",
        # hedged refusals: "I don't think / believe I have/know", "I'm not sure I have"
        r"\bi\s+don'?t\s+think\s+i\s+(have|know)\b",
        r"\bi\s+don'?t\s+believe\s+i\s+(have|know)\b",
        r"\bi'?m\s+not\s+(sure|certain)\s+(if\s+)?i\s+(have|know)\b",
        # "I don't think I received/got one" — optional "ever" between subject and verb
        r"\bi\s+don'?t\s+think\s+i\s+(?:ever\s+)?(received|got|was\s+given)\b",
        r"\bhe\s+doesn'?t\s+have\b",
        r"\bshe\s+doesn'?t\s+have\b",
        # "don't / doesn't know" variants. Qualified by an object — including
        # "where my card is" — this is a denial wherever it sits in the turn.
        r"\bi\s+don'?t\s+know\s+(it|that|my\b|the\b|what|where)",
        # Bare "I don't know", and only when it is the whole turn. Politeness
        # and hedges are not "something else"; a follow-on clause is, and it
        # is usually the caller answering ("I don't know, is it 512-555-6101?").
        r"^\W*(?:(?:i'?m\s+|i\s+am\s+)?sorry[\s,.\-]*)?"
        r"i\s+(?:really\s+|honestly\s+|truly\s+|just\s+|actually\s+)?"
        r"do(?:\s+not|n'?t)\s+know"
        r"(?:[\s,.\-]*(?:it|that|this|offhand|off\s+hand|right\s+now|sorry|unfortunately))*"
        r"[\s,.!\-]*$",
        # "can't remember / recall / find"
        r"\bcan'?t\s+(remember|recall|find)\s+(it|that|my\b|the\b)",
        r"\bi\s+can'?t\s+(remember|recall|find)\b",
        # "don't remember / recall"
        r"\bi\s+don'?t\s+(remember|recall)\b",
        # "haven't memorised / got it"
        r"\bhaven'?t\s+(memorized?|memorised?|got\s+it)\b",
        # "never received / got it" — what was never received has to be the
        # identifier. "I never received the provider list you faxed" is the
        # reason for the call, not an inability to answer the question.
        r"\bi\s+never\s+(?:received|got)\s+" + _IDENTIFIER_OBJECT + r"\b",
        # physical absence — same rule on the object.
        r"\bi\s+(?:lost|misplaced)\s+" + _IDENTIFIER_OBJECT + r"\b",
        # First-person, and the place it was left is named. The bare "left it"
        # this replaces matched "she left it with the doctor" and "we left it
        # at that"; "I left it" alone needs no complement, but "I left it
        # blank on the form" is a caller describing a form, not a denial.
        r"\bi\s+left\s+it\s+(?:at\b|in\s+(?:the|my)\b|back\s+(?:at|home)\b"
        r"|behind\b|there\b|home\b|with\s+(?:my|the)\b)",
        r"^\W*i\s+left\s+it\W*$",
        r"\bi\s+don'?t\s+carry\s+(?:it|that|one\b|my\b|the\b)",
        r"^\W*not\s+with\s+me\W*$",
        # access / availability
        r"\bdon'?t\s+have\s+access\b",
        r"\bnot\s+(available|with\s+me|here)\s+right\s+now\b",
        r"\bi'?m\s+unable\s+to\s+provide\b",
        # "no, I don't have it" — leading negation + inability
        r"^no[,.]?\s+i\s+don'?t\s+have\b",
        r"^no[,.]?\s+i\s+don'?t\s+know\b",
        # "don't have that information"
        r"\bdon'?t\s+have\s+that\s+information\b",
        # "don't have [X] handy / on me / with me"
        r"\bdon'?t\s+have\s+\w[\w\s]*(handy|on\s+me|with\s+me)\b",
        # "it's not available / with me / here"
        r"\bit'?s\s+not\s+(available|with\s+me|here)\b",
    ]
]


def detect_cannot_provide(text: str | None) -> bool:
    """
    Return True when the caller is explicitly stating they cannot or do not
    have the value being requested.

    Examples that return True:
      "I don't have it"           "I don't have my member ID"
      "I don't know it"           "I can't remember"
      "I lost my card"            "It's not with me right now"
      "I left it at home"         "No, I don't have it"
      "I don't have that info"    "I haven't got it"
      "I never received one"      "I can't find it"

    Examples that return False (normal retry / confirmation flow):
      "no"            "that's wrong"     "M110781"
      "nope"          "I think it's..."  "can you repeat"
      "I moved"       "yes"              "april twelfth"

    Every pattern is first-person, and the ones built on a verb of loss or
    non-receipt also name what was lost, from _IDENTIFIER_OBJECT. A match
    escalates the call with no retry (see the four call sites in
    _collect_slot), so a pattern that also fits an ordinary sentence — "I lost
    my job", "I never received the list you faxed" — costs a caller who was
    answering. Widen this list only with an object attached.
    """
    if not text:
        return False
    return any(pat.search(text.strip()) for pat in _CANNOT_PROVIDE_PATTERNS)


# ---------------------------------------------------------------------------
# WAIT detection — "give me a minute", "hold on", "let me grab my card"
#
# Regex fallback for the WAIT event in _collect_slot (core/slot_manager.py):
# fires when the extraction LLM reports turn_intent "wait" OR mislabels a
# wait as ambiguous. Compiled once at import time.
# ---------------------------------------------------------------------------

_WAIT_PATTERNS: list = [_re.compile(p, _re.IGNORECASE) for p in WAIT_PATTERNS]


# Spelled-out digits, for callers who read a number aloud ("four five one
# nine"). Dictated numbers are normalised elsewhere; here we only need to know
# that digits were spoken.
_SPOKEN_DIGITS: frozenset[str] = frozenset(
    "zero oh one two three four five six seven eight nine ten "
    "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
    "nineteen twenty thirty forty fifty sixty seventy eighty ninety "
    "hundred double triple".split()
)

# An identifier the caller might have read out alongside a wait phrase: a token
# mixing letters and digits (M451982), or a run of digits.
_VALUE_TOKEN_RE = _re.compile(r"[a-z]\d{3,}|\d{3,}[a-z]|\d", _re.IGNORECASE)


def _looks_like_a_value(text: str) -> bool:
    """Does ``text`` carry something value-shaped, as opposed to prose?

    Used by detect_wait_request to tell "hold on, it's M451982" (a value the
    extractor should have) from "hold on, let me dig out the letter" (a caller
    narrating their search). Deliberately about shape, never about length.
    """
    if sum(c.isdigit() for c in text) >= 4:
        return True
    if _re.search(r"[a-z]\d{3,}|\d{3,}[a-z]", text, _re.IGNORECASE):
        return True
    spoken = [w for w in _re.findall(r"[a-z]+", text.lower()) if w in _SPOKEN_DIGITS]
    return len(spoken) >= 3


def detect_wait_request(text: str | None) -> bool:
    """
    Return True when the caller is asking for time to find or think about
    the value — NOT answering and NOT refusing.

    Examples that return True:
      "give me a minute"      "hold on, let me grab my card"
      "one second"            "let me check"
      "wait"                  "just a sec"        "bear with me"

    Examples that return False:
      "hold on, it's M451982"   — a plausible value follows; extraction wins
      "I don't have my card"    — cannot-provide outranks wait
      "M110781"                 — plain answer, no wait phrase

    Precedence rules:
      1. detect_cannot_provide() outranks wait — "I don't have it" must
         route to the cannot-provide escalation, never a wait ack.
      2. If, after removing every matched wait phrase, a plausible slot VALUE
         remains, return False and let extraction handle the turn — the value
         wins. "Plausible value" means value-shaped: digits, spelled-out
         digits, or an alphanumeric identifier.

    Rule 2 used to count words instead: three or more word tokens left over
    meant "a value follows". Word count is the wrong proxy, and it read the
    commonest wait turn there is as an answer —

        Caller  hold on, let me dig out the letter... one second
        AI      Sorry, I didn't catch that — could you repeat the reference
                number?

    "hold on" and "one second" both matched; the leftover "let me dig out the
    letter" is six words, so the guard vetoed the wait and the caller burned a
    retry for narrating their search. Six words of English is the opposite of
    a slot value — a reference number is eight digits, a member ID is M plus
    six. Prose after "hold on" is the caller telling you they are looking.
    """
    if not text:
        return False
    # A correction/update/redo/replay request outranks wait: "hold on, my ZIP
    # changed" is a correction turn, not a hold request. Import inside the
    # function — request_detection depends on core constants and must never
    # pull in agent.utils (cycle safety).
    from agent.core.request_detection import detect_request

    if detect_request(text) is not None:
        return False
    if detect_cannot_provide(text):
        return False
    lowered = text.lower().strip()
    remainder = lowered
    for pat in _WAIT_PATTERNS:
        remainder = pat.sub(" ", remainder)
    if remainder == lowered:
        return False  # no wait phrase matched
    if _looks_like_a_value(remainder):
        return False  # a value follows the wait phrase — extraction decides
    return True
