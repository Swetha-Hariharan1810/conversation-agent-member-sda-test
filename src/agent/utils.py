import importlib.resources
import random
import re
import re as _re
from functools import lru_cache
from typing import Any

from agent.core.constants import WAIT_PATTERNS
from agent.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=32)
def read_prompt(file_path: str) -> str:
    try:
        return (
            importlib.resources.files("agent")
            .joinpath("prompts", file_path)
            .read_text(encoding="utf-8")
            .strip()
        )
    except Exception as e:
        logger.warning("Prompt not found: %s — %s", file_path, e)
        return ""


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


@lru_cache(maxsize=36)
def build_extraction_prompt(agent_prompt_file: str) -> str:
    """
    System prompt for LLM 1 (get_extraction_llm).
    Combines extraction header + agent-specific rules only.
    No global behavioural rules — extraction LLM does not need them.
    """
    global_prompt = read_prompt("system/global_extraction.md")
    header = read_prompt("extraction/header.md")
    agent = read_prompt(agent_prompt_file)
    return f"{global_prompt}\n\n---\n\n{header}\n\n---\n\n{agent}\n\n"


@lru_cache(maxsize=36)
def build_extraction_prompt_core(agent_prompt_file: str) -> str:
    """
    Minimal extraction prompt for agents that only need guard detection
    and simple field extraction (no corrections, no spelling handling).

    Use for: intake, benefits, care_wellness.
    Input tokens: ~200-300 (vs ~1050-1450 with full header).
    """
    global_prompt = read_prompt("system/global_extraction.md")
    core_header = read_prompt("extraction/header_core.md")
    agent = read_prompt(agent_prompt_file)
    return f"{global_prompt}\n\n---\n\n{core_header}\n\n---\n\n{agent}\n\n"


@lru_cache(maxsize=36)
def build_extraction_prompt_extraction(agent_prompt_file: str) -> str:
    """
    Mid-tier extraction prompt for agents that collect structured slot
    values and need the confidence rule and basic event_type, but not
    the full verification machinery (SPELL_CONFIRM, CORRECTED, LOCKED).

    Use for: provider_search, delivery_management, follow_up.
    Input tokens: ~380-500 (vs ~1050-1200 with full header).
    """
    global_prompt = read_prompt("system/global_extraction.md")
    extraction_header = read_prompt("extraction/header_extraction.md")
    agent = read_prompt(agent_prompt_file)
    return f"{global_prompt}\n\n---\n\n{extraction_header}\n\n---\n\n{agent}\n\n"


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


def pick(pool) -> str:
    """Randomly select from a pool, or return the string as-is."""
    if isinstance(pool, list):
        return random.choice(pool) if pool else ""
    return pool or ""


def _quick_yes_no(text: str) -> str:
    """
    Fast keyword check for unambiguous yes/no responses.
    Returns 'yes', 'no', or '' (ambiguous — needs LLM).

    Used as fast-path before LLM extraction for any yes/no decision.
    Avoids LLM call for clear responses, improving latency.

    Rules:
    - Word boundary matching to avoid false positives
    - Checks negation before YES patterns
      ("not sure" must not match YES via "sure")
    - Returns '' for anything ambiguous — caller decides via LLM
    """
    t = text.lower().strip()
    if not t:
        return ""

    # Check negation first — blocks false YES matches
    _NEGATION = re.compile(r"\b(not|don't|doesn't|can't|won't|never)\b")
    has_negation = bool(_NEGATION.search(t))

    # Clear YES — only when no leading negation
    if not has_negation:
        _YES = re.compile(
            r"\b(yes|yeah|yep|yup|sure|please|transfer|"
            r"connect|go ahead|ok|okay|absolutely)\b"
        )
        if _YES.search(t):
            return "yes"
        if t.startswith(("yes ", "yeah ", "sure ", "please ")):
            return "yes"

    # Clear NO
    _NO = re.compile(
        r"\b(no|nope|nah|never mind|nevermind|"
        r"continue|stay|keep going|help me)\b"
    )
    if _NO.search(t):
        return "no"
    if t.startswith(("no ", "nope ", "nah ")):
        return "no"

    return ""  # ambiguous — needs LLM


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
# ---------------------------------------------------------------------------

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
        # "don't / doesn't know" variants
        r"\bi\s+don'?t\s+know\s+(it|that|my\b|the\b|what)",
        r"\bi\s+don'?t\s+know\b",  # bare "I don't know"
        # "can't remember / recall / find"
        r"\bcan'?t\s+(remember|recall|find)\s+(it|that|my\b|the\b)",
        r"\bi\s+can'?t\s+(remember|recall|find)\b",
        # "don't remember / recall"
        r"\bi\s+don'?t\s+(remember|recall)\b",
        # "haven't memorised / got it"
        r"\bhaven'?t\s+(memorized?|memorised?|got\s+it)\b",
        # "never received / got it"
        # r"\bi\s+never\s+(received|got)\s+(it|that|one|my\b)",
        r"\bi\s+never\s+(received|got)\b",
        # physical absence
        r"\bi\s+(lost|misplaced)\s+(it|that|my\b|the\b)",
        r"\b(not with me|left it|don'?t carry)\b",
        r"\bi\s+left\s+it\s+(at\s+home|behind|there)\b",
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

    False positives are extremely low: all patterns require first-person
    ownership language ("I don't have", "I lost", "it's not with me").
    """
    if not text:
        return False
    return any(pat.search(text.strip()) for pat in _CANNOT_PROVIDE_PATTERNS)


# ---------------------------------------------------------------------------
# WAIT detection — "give me a minute", "hold on", "let me grab my card"
#
# Regex fallback for the WAIT event in _collect_slot (core/slot_manager.py):
# fires when the extraction LLM returns event_type "wait" OR mislabels a
# wait as ambiguous. Compiled once at import time.
# ---------------------------------------------------------------------------

_WAIT_PATTERNS: list = [_re.compile(p, _re.IGNORECASE) for p in WAIT_PATTERNS]


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
      2. If, after removing every matched wait phrase, a plausible slot-value
         continuation remains (>= 3 word tokens or >= 4 digits), return False
         and let extraction handle the turn — the value wins.
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
    word_tokens = _re.findall(r"[a-z']+", remainder)
    digit_count = sum(c.isdigit() for c in remainder)
    if len(word_tokens) >= 3 or digit_count >= 4:
        return False  # plausible slot-value continuation — extraction decides
    return True
