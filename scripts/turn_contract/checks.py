"""
checks.py — turn-level sanity checks over a call transcript.

Step 0 of the conversation-quality plan: the safety net. These checks are pure
Python — no LLM, no graph, no network — so they run in milliseconds and can gate
every prompt or flow change.

Each check answers one question about a single agent turn (or a short window of
turns) that a human reviewer would answer instantly, and that the eval harness
cannot see because it scores conversational *similarity*, not self-consistency:

  contradiction        the turn declines something and then does it
  stray_filler         a contentless fragment welded into the middle of a turn
  wrong_decline        the turn declines a change the capability registry allows
  declined_open_option the caller named a supported option and was declined
  partial_readback     a value was read back for confirmation before it was complete
  repeat_readback      the same slot was read back twice with different values
  two_asks             one turn asks for two different slots
  duplicate_turn       the agent said exactly what it just said

Grounding: `wrong_decline` consults the production registry in
agent.core.slot_ownership, so it fails whenever the spoken decline disagrees
with what the code would actually have done. That is the Transcript 3 bug.

Usage:
    from scripts.turn_contract.checks import run_all_checks
    violations = run_all_checks(transcript)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

# The package lives outside src/, so make `agent` importable when run directly.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


Role = Literal["ai", "human"]
Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Turn:
    role: Role
    text: str
    # Optional annotations. Supplied by seed transcripts; a live capture can
    # fill them from state. Every check degrades gracefully when they are absent.
    awaiting_slot: str = ""
    open_slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transcript:
    name: str
    turns: tuple[Turn, ...]
    note: str = ""


@dataclass(frozen=True)
class Violation:
    check: str
    severity: Severity
    turn_index: int
    detail: str
    excerpt: str = ""


# ── Vocabulary ───────────────────────────────────────────────────────────────
# Kept deliberately narrow. A false positive here costs a real change being
# blocked, so every pattern below was taken from an observed production turn.

# Phrases that say "this cannot be done here".
_DECLINE_RE = re.compile(
    r"\ba representative (?:would|will) (?:need|have) to\b"
    r"|\b(?:would|will) need a representative\b"
    r"|\bneed to speak (?:with|to) a representative\b"
    r"|\bis(?:n't| not) something (?:i|we|this line) can\b"
    r"|\b(?:can(?:'|no)?t|unable to) do that\b"
    r"|\bnot able to\b",
    re.IGNORECASE,
)

# Phrases that say "this is being done, now, by me".
_GRANT_RE = re.compile(
    r"\blet me (?:update|change|correct|fix|get)\b"
    r"|\bi(?:'ve| have) (?:updated|changed|corrected|sent|refreshed)\b"
    r"|\bi(?:'ll| will) (?:update|change|send|get)\b"
    r"|\ball set\b"
    r"|\bcould you (?:give|provide) me your\b",
    re.IGNORECASE,
)

# A sentence that is nothing but an acknowledgement token.
_FILLER_ONLY_RE = re.compile(
    r"^(?:definitely|sure|of course|certainly|absolutely|okay|ok|alright|great|"
    r"perfect|no problem|got it|understood|thank you|thanks)[\s,.!—-]*$",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Spoken ways each slot gets named. Used to work out what a decline was about.
# This is the seed of the option/label table that Step 2 makes authoritative —
# today the same facts are spread across _SLOT_LABELS, the agent prompts, the
# normalizers and SLOT_ASK_SYNONYMS.
_SLOT_TERMS: dict[str, tuple[str, ...]] = {
    "zip_code": ("zip code", "zip"),
    "fax": ("fax number", "fax"),
    "email": ("email address", "email"),
    "member_id": ("member id", "member number"),
    "dob": ("date of birth", "birth date"),
    "first_name": ("first name",),
    "last_name": ("last name",),
    "phone_number": ("phone number",),
    "relationship": ("subscriber", "dependent"),
    "delivery_method": ("delivery method",),
    "notification_method": ("notification channel", "notification method"),
}

# Values a caller can choose for a closed-set slot, and how they say them.
# Seeded from records_coordination.md. Step 2 moves this onto the slot
# definitions and renders it into the extraction prompt.
_SLOT_OPTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "upload_method": {
        "member_upload": ("upload", "secure link", "i'll send", "i can send them"),
        "doctor_direct": (
            "doctor send",
            "doctor to send",
            "doctor's office send",
            "ask my doctor",
            "have my doctor",
            "provider send",
        ),
        "personal_guide": ("contact my doctor", "reach out to my doctor", "on my behalf"),
    },
}

# How long a complete value is, per slot. Seeded from slots/validators.py.
# Step 3 moves this onto the slot definitions as `expected_shape`.
_EXPECTED_DIGITS: dict[str, int] = {
    "reference_number": 8,
    "zip_code": 5,
    "ssn": 9,
}

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_SPOKEN_DIGIT_RE = re.compile(r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)\b", re.IGNORECASE)

# Slots whose ask legitimately names several values in one breath:
# "fax or email" is one question with two options, not two asks.
_CHOICE_GROUPS: dict[str, frozenset[str]] = {
    "delivery_method": frozenset({"fax", "email"}),
    "notification_method": frozenset({"fax", "email"}),
}


# A turn that reads a value back and asks the caller to confirm it.
_READBACK_RE = re.compile(
    r"\b(?:is|that)\b[^.?!]*\b(?:correct|right)\b\s*\?"
    r"|\bis that the right\b"
    r"|\band that .{0,40}\bis\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split((text or "").strip()) if s.strip()]


def _spoken_digits(text: str) -> str:
    """Digits in a phrase, reading spoken number words as digits."""
    converted = _SPOKEN_DIGIT_RE.sub(lambda m: _DIGIT_WORDS[m.group(0).lower()], text or "")
    return re.sub(r"\D", "", converted)


def _mentions(text: str, terms: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(t in lowered for t in terms)


def _targets_in(text: str) -> list[str]:
    """Slot names this text appears to be talking about."""
    return [slot for slot, terms in _SLOT_TERMS.items() if _mentions(text, terms)]


# ── Checks ───────────────────────────────────────────────────────────────────


def check_contradiction(turns: Sequence[Turn]) -> list[Violation]:
    """One turn that declines something and then does it.

        "a representative would need to make that change to your ZIP code.
         Sure — let me update your zip code first."

    Two producers wrote that turn and neither could see the other.
    """
    out: list[Violation] = []
    for i, t in enumerate(turns):
        if t.role != "ai":
            continue
        declines = [s for s in _sentences(t.text) if _DECLINE_RE.search(s)]
        grants = [s for s in _sentences(t.text) if _GRANT_RE.search(s)]
        if not declines or not grants:
            continue
        for d in declines:
            d_targets = set(_targets_in(d))
            for g in grants:
                g_targets = set(_targets_in(g))
                shared = d_targets & g_targets
                # Same named target, or a bare decline immediately followed by
                # an action — both are the caller being told no and yes at once.
                if shared or (not d_targets and g_targets):
                    which = ", ".join(sorted(shared)) if shared else ", ".join(sorted(g_targets))
                    out.append(
                        Violation(
                            "contradiction",
                            "error",
                            i,
                            f"turn declines and then acts on the same target ({which})",
                            f"{d}  ||  {g}",
                        )
                    )
                    break
            else:
                continue
            break
    return out


def check_stray_filler(turns: Sequence[Turn]) -> list[Violation]:
    """A contentless fragment in the middle of a turn — a joined-on remnant.

    "...refreshed your provider list for that area. Definitely. The fax
     number we have on file is 6171234199."
    """
    out: list[Violation] = []
    for i, t in enumerate(turns):
        if t.role != "ai":
            continue
        sents = _sentences(t.text)
        for pos, s in enumerate(sents):
            if pos == 0:
                continue  # an opener is fine
            if _FILLER_ONLY_RE.match(s):
                out.append(
                    Violation(
                        "stray_filler",
                        "error",
                        i,
                        f"contentless fragment at sentence {pos + 1} of {len(sents)}",
                        s,
                    )
                )
    return out


def check_wrong_decline(turns: Sequence[Turn]) -> list[Violation]:
    """A decline about something the capability registry says is doable.

    Consults the production table, so this fails exactly when the spoken words
    disagree with what the code would have done.
    """
    try:
        from agent.core.slot_ownership import get_ownership
    except Exception:  # pragma: no cover - registry import is best effort
        return []

    out: list[Violation] = []
    for i, t in enumerate(turns):
        if t.role != "ai" or not _DECLINE_RE.search(t.text):
            continue
        for slot in _targets_in(t.text):
            own = get_ownership(slot)
            if own is not None and own.updatable != "human_only":
                out.append(
                    Violation(
                        "wrong_decline",
                        "error",
                        i,
                        f"declined {slot!r}, but SLOT_OWNERSHIP says updatable={own.updatable!r}"
                        + (f" (owner {own.agent})" if own.agent else ""),
                        next((s for s in _sentences(t.text) if _DECLINE_RE.search(s)), t.text),
                    )
                )
    return out


def check_declined_open_option(turns: Sequence[Turn]) -> list[Violation]:
    """The caller named a supported option and the agent declined it.

    Looks back over the recent open slots, so an answer that arrives a turn
    late — after an ASR cutoff — is still recognised as an answer.
    """
    out: list[Violation] = []
    for i, t in enumerate(turns):
        if t.role != "ai" or not _DECLINE_RE.search(t.text):
            continue
        prior = turns[i - 1] if i else None
        if prior is None or prior.role != "human":
            continue
        # Which slots were live around this point in the call?
        window = turns[max(0, i - 6) : i + 1]
        live: set[str] = set()
        for w in window:
            live.update(w.open_slots)
            if w.awaiting_slot:
                live.add(w.awaiting_slot)
        for slot in live:
            for value, phrasings in _SLOT_OPTIONS.get(slot, {}).items():
                if _mentions(prior.text, phrasings):
                    out.append(
                        Violation(
                            "declined_open_option",
                            "error",
                            i,
                            f"caller chose {slot}={value!r}, a supported option, and was declined",
                            prior.text.strip(),
                        )
                    )
    return out


def check_partial_readback(turns: Sequence[Turn]) -> list[Violation]:
    """A value read back for confirmation before it was complete.

    Caller  "It is four two six nine."          (4 of 8 digits)
    AI      "And that reference number is four two six nine?"
    """
    out: list[Violation] = []
    for i, t in enumerate(turns):
        if t.role != "ai" or not _READBACK_RE.search(t.text):
            continue
        slot = t.awaiting_slot
        expected = _EXPECTED_DIGITS.get(slot)
        if not expected:
            continue
        digits = _spoken_digits(t.text)
        if digits and len(digits) < expected:
            out.append(
                Violation(
                    "partial_readback",
                    "error",
                    i,
                    f"read back {len(digits)} of {expected} digits for {slot!r} — "
                    "should have asked for the remainder",
                    t.text.strip(),
                )
            )
    return out


def check_repeat_readback(turns: Sequence[Turn]) -> list[Violation]:
    """The same slot read back for confirmation more than once."""
    seen: dict[str, int] = {}
    out: list[Violation] = []
    for i, t in enumerate(turns):
        if t.role != "ai" or not _READBACK_RE.search(t.text) or not t.awaiting_slot:
            continue
        seen[t.awaiting_slot] = seen.get(t.awaiting_slot, 0) + 1
        if seen[t.awaiting_slot] > 1:
            out.append(
                Violation(
                    "repeat_readback",
                    "warning",
                    i,
                    f"{t.awaiting_slot!r} read back {seen[t.awaiting_slot]} times in one call",
                    t.text.strip(),
                )
            )
    return out


def _collapse_choices(asked: set[str]) -> set[str]:
    """Fold the members of a single choice into the one slot they belong to.

    "Shall I deliver it by fax or email?" names two slot terms and is one ask.
    Only collapses when the named terms are wholly contained in one group, so a
    turn that really does ask for a fax number and an email address still fails.
    """
    for slot, members in _CHOICE_GROUPS.items():
        if len(asked) > 1 and asked <= members:
            return {slot}
    return asked


def check_two_asks(turns: Sequence[Turn]) -> list[Violation]:
    """One turn asking for two different slots."""
    out: list[Violation] = []
    for i, t in enumerate(turns):
        if t.role != "ai":
            continue
        asked: set[str] = set()
        for s in _sentences(t.text):
            if "?" not in s:
                continue
            asked.update(_collapse_choices(set(_targets_in(s))))
        asked = _collapse_choices(asked)
        if len(asked) > 1:
            out.append(
                Violation(
                    "two_asks",
                    "error",
                    i,
                    f"turn asks for {len(asked)} slots: {', '.join(sorted(asked))}",
                    t.text.strip(),
                )
            )
    return out


def check_duplicate_turn(turns: Sequence[Turn]) -> list[Violation]:
    """The agent repeating itself verbatim."""
    out: list[Violation] = []
    previous_ai = ""
    for i, t in enumerate(turns):
        if t.role != "ai":
            continue
        normalized = " ".join(t.text.split()).lower()
        if normalized and normalized == previous_ai:
            out.append(
                Violation("duplicate_turn", "warning", i, "identical to the previous agent turn", t.text)
            )
        previous_ai = normalized
    return out


ALL_CHECKS = (
    check_contradiction,
    check_stray_filler,
    check_wrong_decline,
    check_declined_open_option,
    check_partial_readback,
    check_repeat_readback,
    check_two_asks,
    check_duplicate_turn,
)


def run_all_checks(transcript: Transcript, *, only: Optional[Iterable[str]] = None) -> list[Violation]:
    """Every check over one transcript, ordered by turn."""
    wanted = set(only) if only else None
    found: list[Violation] = []
    for check in ALL_CHECKS:
        name = check.__name__.removeprefix("check_")
        if wanted and name not in wanted:
            continue
        found.extend(check(transcript.turns))
    return sorted(found, key=lambda v: (v.turn_index, v.check))
