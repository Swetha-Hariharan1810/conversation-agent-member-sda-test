import logging
import re

from agent.utils import build_history

logger = logging.getLogger(__name__)

# Patterns that trigger Azure OpenAI's jailbreak content filter. None of
# these appear in legitimate caller speech (name, DOB, member ID, etc.).
_INJECTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,40}\b(?:instructions?|rules?|prompts?|directives?|context)\b"
    r"|\byou\s+are\s+now\b"
    r"|\bpretend\s+(?:you\s+are|to\s+be)\b"
    r"|\bact\s+as\s+(?:a\b|if\b)"
    r"|\bnew\s+(?:system\s+)?instructions?\b"
    r"|\bsystem\s+prompt\b"
    r"|\bjailbreak\b"
    r"|\bdan\s*mode\b"
    r"|\[INST\]|\[\/INST\]",
    re.IGNORECASE,
)


def _sanitize_utterance(text: str) -> str:
    """Replace prompt-injection phrases in caller speech with [*]."""
    if not text:
        return text
    sanitized = _INJECTION_RE.sub("[*]", text)
    if sanitized != text:
        logger.warning("Prompt-injection pattern detected in caller utterance; sanitized before LLM call")
    return sanitized


def remaining_slots(order: list[str], current: str) -> list[str]:
    """Slice of ``order`` from ``current`` onward — the pending slots this turn.

    Falls back to the full order when ``current`` is not in it (e.g. a
    transient confirmation sub-slot not listed in the agent's static order).
    """
    if current in order:
        return list(order[order.index(current) :])
    return list(order)


def build_worker_input(
    system_prompt: str,
    awaiting_slot: str,
    last_agent_message: str,
    last_user_message: str,
    *,
    confirmed_slots: dict | None = None,
    pending_slots: list[str] | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> list[dict]:
    """
    Build the message list for LLM 1 (extraction model).

    Parameters
    ----------
    system_prompt:
        The system prompt for the extraction model.
    awaiting_slot:
        The single slot currently being collected.
    last_agent_message:
        The question the agent just asked. Used to build history when
        recent_messages is absent.
    last_user_message:
        The caller's response. Used to build history when recent_messages
        is absent.
    confirmed_slots:
        Dict of slot name → value for all slots already confirmed. Only
        entries with non-empty string values are included in the prompt.
        Omitted entirely when None or empty.
    pending_slots:
        Slot names still to be collected later in this call, in order.
        Rendered as a "Pending:" context line so the extraction LLM can
        classify follow-up questions as parkable (followup_disposition
        "park"). Omitted entirely when None or empty.
    attempt:
        How many collection attempts have been made for awaiting_slot so far.
    recent_messages:
        Recent conversation turns. Each entry is a dict with "role" and
        "content" keys. Up to the last 6 messages are used.
    """
    # Sanitize caller utterances against prompt-injection before embedding.
    last_user_message = _sanitize_utterance(last_user_message)
    if recent_messages:
        recent_messages = [
            {**m, "content": _sanitize_utterance(m["content"])}
            if isinstance(m, dict) and str(m.get("role", "")).lower() in {"user", "human"}
            else m
            for m in recent_messages
        ]

    # Build conversation history — priority: recent_messages > individual params
    history_block = ""
    if recent_messages:
        n = 2 if attempt >= 2 else 4
        history_block = "\n".join(build_history(recent_messages, n=n))
        history_block += "\n"
    elif last_agent_message or last_user_message:
        history_block = f"Agent: {last_agent_message}\nCaller: {last_user_message}\n\n"

    # Build context lines
    context_lines = [
        f"Currently asking for: {awaiting_slot}",
    ]
    if confirmed_slots:
        filled = {k: _sanitize_utterance(v) for k, v in confirmed_slots.items() if isinstance(v, str) and v}
        if filled:
            confirmed_str = ", ".join(f"{k}={v}" for k, v in filled.items())
            context_lines.append(f"Confirmed: {confirmed_str}")
    if pending_slots:
        context_lines.append(f"Pending: {', '.join(pending_slots)}")

    # Explicitly surface the most recent caller utterance so the extraction
    # LLM does not have to re-parse it from the history block. This prevents
    # ambiguous classification on spoken-digit or multi-word values when
    # attempt count is elevated and the model becomes overly conservative.
    if last_user_message:
        context_lines.append(f"Caller just said: {last_user_message}")

    user_content = history_block + "\n".join(context_lines)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
