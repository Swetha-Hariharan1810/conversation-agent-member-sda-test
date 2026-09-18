"""
Transcript cursor — position-tracked ground-truth lookup over the static
reference transcripts in static_transcripts/.

Given the live agent message and the current cursor position, returns the
scripted human reply from the matching point of the reference transcript so
the user simulator can answer exactly as the canonical conversation did.

FIXES (stability — these were the dominant sources of "random" eval failures):

  1. SIMILARITY METRIC. The old score was |A∩B| / max(|A|,|B|) over raw word
     sets with CONFIRM_THRESHOLD=0.25. Agents frequently combine two scripted
     messages into one turn (e.g. the dispatch confirmation + the benefits
     offer, and now the ZIP-aware dispatch message which is even longer).
     The combined message inflates max(|A|,|B|), the score drops below 0.25,
     the cursor MISSES, and the runner falls back to the slot ground truth —
     introducing turn-to-turn variance that looked random.
     New score: CONTAINMENT |A∩B| / min(|A|,|B|) over *content* tokens
     (stopwords stripped), so a reference line fully contained in a longer
     combined live message scores ~1.0. Threshold raised to 0.5, with a
     minimum of 2 shared content tokens to keep short generic lines from
     matching everything.

  2. CURSOR DESYNC. The old lookup took the FIRST position within
     LOOKAHEAD=4 whose score cleared the threshold. Generic lines ("Is there
     anything else I can help you with?") appear several times per
     transcript, so a generic live message could clear the threshold at a
     *later* position first, jump the cursor forward, and permanently desync
     the rest of the conversation — every subsequent turn then fell back or
     mismatched. New lookup is a windowed BEST-match with current-position
     priority: the current position wins unless a lookahead position beats
     it by AHEAD_MARGIN (0.2). Re-asks of the previous question (slot
     retries) are served from position-1 WITHOUT advancing the cursor.

  3. MULTI-LINE PARSING. _parse_transcript dropped continuation lines of
     multi-line "ai:" messages (the claim transcripts contain them), which
     shrank the reference text and weakened matching exactly on the longest,
     most information-dense turns. Continuation lines are now appended to
     the preceding message.

Public API (unchanged):
    _SCENARIO_FILE_MAP
    _parse_transcript(text) -> list[(role, message)]
    load(scenario_tag)      -> list[(ai_message, human_reply)]
    get_ground_truth(scenario_tag, ai_message, cursor) -> (reply | None, new_cursor)
    reset(scenario_tag=None)
"""

from __future__ import annotations

import re
from pathlib import Path

_TRANSCRIPT_DIR = Path(__file__).parent / "static_transcripts"

_SCENARIO_FILE_MAP: dict[str, str] = {
    "pcp_happy_path": "pcp_happy_path.txt",
    "pcp_clarification_zip": "pcp_clarification_zip.txt",
    # "pcp_correction_first_name": "pcp_correction_first_name.txt",
    "pcp_correction_member_id": "pcp_correction_member_id.txt",
    "pcp_clarification_fax": "pcp_clarification_fax.txt",
    "claim_adjustment_happy_path": "claim_adjustment_happy_path.txt",
    "claim_adjustment_no_proceed": "claim_adjustment_no_proceed.txt",
    "claim_adjustment_upload_only": "claim_adjustment_upload_only.txt",
    "claim_adjustment_guide_only": "claim_adjustment_guide_only.txt",
}

# Matching parameters — see module docstring for the rationale.
CONFIRM_THRESHOLD = 0.5  # containment score required to accept a position
AHEAD_MARGIN = 0.2  # how much better a lookahead position must score to win
LOOKAHEAD = 4  # pairs scanned beyond the current position
LOOKBACK = 1  # pairs scanned behind (re-asked question → same reply)
MIN_SHARED_TOKENS = 2  # minimum shared content tokens (unless a side has 1)

# High-frequency function words stripped before scoring. Content tokens are
# what actually distinguish one agent prompt from another.
_STOPWORDS = frozenset(
    """
    the a an is are was were to for of your you i we can could may please and
    or that this it in on with do did me my our us be have has will would
    like get just so as at by if any what how
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9@.\-]+")


def _normalize_token(token: str) -> str:
    """Strip punctuation glue so equivalent tokens compare equal.

    Formatted numbers must match their plain form: the transcripts write
    "617-555-4199" while the live agent may read back "6175554199" — without
    normalization the containment score for the fax/phone read-back drops
    below threshold and the cursor misses. Trailing sentence punctuation
    captured by the token regex ("12139.") is stripped for the same reason.
    """
    token = token.strip(".-")
    if any(ch.isdigit() for ch in token):
        token = token.replace("-", "").replace(".", "")
    return token


def _content_tokens(text: str) -> frozenset[str]:
    """Lowercased word set with punctuation and stopwords removed."""
    return frozenset(
        norm
        for t in _TOKEN_RE.findall(text.lower())
        if (norm := _normalize_token(t)) and norm not in _STOPWORDS
    )


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    """|A∩B| / min(|A|,|B|): 1.0 when the smaller side is fully contained."""
    if not a or not b:
        return 0.0
    shared = len(a & b)
    smaller = min(len(a), len(b))
    if shared < MIN_SHARED_TOKENS and smaller >= MIN_SHARED_TOKENS:
        return 0.0
    return shared / smaller


def _parse_transcript(text: str) -> list[tuple[str, str]]:
    """Parse 'ai:' / 'human:' lines into (role, message) tuples.

    Lines that do not begin with a role prefix are CONTINUATIONS of the
    previous message and are appended to it (the old parser silently dropped
    them, truncating multi-line ai turns in the claim transcripts).
    """
    turns: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("ai:"):
            turns.append(("ai", line[3:].strip()))
        elif lowered.startswith("human:"):
            turns.append(("human", line[6:].strip()))
        elif turns:
            role, msg = turns[-1]
            turns[-1] = (role, f"{msg} {line}")
        # a continuation before any role line is malformed — ignore it
    return turns


def _pair_turns(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Collapse the turn list into (ai_message, human_reply) pairs.

    Consecutive ai messages (no human in between) are merged — the live
    agent emits them as one combined turn anyway. A trailing ai message with
    no reply (the closing line) is kept with an empty reply so it can still
    anchor matching, but get_ground_truth never returns an empty reply.
    """
    pairs: list[tuple[str, str]] = []
    pending_ai: str | None = None
    for role, msg in turns:
        if role == "ai":
            pending_ai = msg if pending_ai is None else f"{pending_ai} {msg}"
        else:
            if pending_ai is not None:
                pairs.append((pending_ai, msg))
                pending_ai = None
            # a human turn with no preceding ai (shouldn't occur) is skipped
    if pending_ai is not None:
        pairs.append((pending_ai, ""))
    return pairs


_cache: dict[str, list[tuple[str, str]]] = {}
_token_cache: dict[str, list[frozenset[str]]] = {}


def load(scenario_tag: str) -> list[tuple[str, str]]:
    """Load (and cache) the (ai, human) pairs for a scenario."""
    if scenario_tag not in _cache:
        filename = _SCENARIO_FILE_MAP.get(scenario_tag)
        if filename is None:
            raise KeyError(f"No static transcript registered for scenario {scenario_tag!r}")
        text = (_TRANSCRIPT_DIR / filename).read_text(encoding="utf-8")
        pairs = _pair_turns(_parse_transcript(text))
        _cache[scenario_tag] = pairs
        _token_cache[scenario_tag] = [_content_tokens(ai) for ai, _ in pairs]
    return _cache[scenario_tag]


def reset(scenario_tag: str | None = None) -> None:
    """Drop cached transcripts (all of them, or one scenario's)."""
    if scenario_tag is None:
        _cache.clear()
        _token_cache.clear()
    else:
        _cache.pop(scenario_tag, None)
        _token_cache.pop(scenario_tag, None)


def get_ground_truth(scenario_tag: str, ai_message: str, cursor: int) -> tuple[str | None, int]:
    """Return (human_reply, new_cursor) for the live agent message.

    Windowed best-match with current-position priority:
      * score the current cursor position and every position up to LOOKAHEAD
        ahead;
      * the current position is accepted when it clears CONFIRM_THRESHOLD,
        unless some lookahead position beats it by at least AHEAD_MARGIN
        (the agent genuinely skipped ahead, e.g. a combined turn swallowed a
        scripted question);
      * if the current position misses but the previous position matches,
        the agent re-asked its last question (slot retry) — the previous
        reply is returned WITHOUT advancing the cursor;
      * otherwise (None, cursor) — the caller falls back to the slot ground
        truth and the cursor stays put instead of being corrupted.
    """
    pairs = load(scenario_tag)
    if not pairs:
        return None, cursor
    cursor = max(0, min(cursor, len(pairs)))  # clamp
    tokens = _token_cache[scenario_tag]
    live = _content_tokens(ai_message)

    def score(i: int) -> float:
        return _containment(live, tokens[i])

    current = score(cursor) if cursor < len(pairs) else 0.0

    best_ahead, best_ahead_idx = 0.0, -1
    for i in range(cursor + 1, min(cursor + 1 + LOOKAHEAD, len(pairs))):
        s = score(i)
        if s > best_ahead:
            best_ahead, best_ahead_idx = s, i

    # Current position wins on a tie or anything short of the margin.
    if current >= CONFIRM_THRESHOLD and not (
        best_ahead >= CONFIRM_THRESHOLD and best_ahead >= current + AHEAD_MARGIN
    ):
        reply = pairs[cursor][1]
        return (reply or None), cursor + 1

    # The agent skipped ahead — accept the decisively better position.
    if best_ahead >= CONFIRM_THRESHOLD and best_ahead >= current + AHEAD_MARGIN:
        reply = pairs[best_ahead_idx][1]
        return (reply or None), best_ahead_idx + 1

    # Re-ask of the previous question (slot retry): same reply, no advance.
    for i in range(cursor - 1, max(cursor - 1 - LOOKBACK, -1), -1):
        if score(i) >= CONFIRM_THRESHOLD:
            reply = pairs[i][1]
            return (reply or None), cursor

    return None, cursor
