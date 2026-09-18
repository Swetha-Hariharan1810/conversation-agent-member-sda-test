import uuid


def new_conversation_id() -> str:
    return str(uuid.uuid4())[:8]


def extract_last_ai_message(messages: list) -> str:
    """Return content of the most recent assistant message.
    Handles both plain dicts and LangChain Message objects (AIMessage.type == 'ai').
    """
    for m in reversed(messages):
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        else:
            # LangChain Message objects: AIMessage has type="ai", not role="assistant"
            role = getattr(m, "type", "") or getattr(m, "role", "")
            content = getattr(m, "content", "")

        if role in ("assistant", "ai"):
            return content.strip() if isinstance(content, str) else str(content).strip()
    return ""
