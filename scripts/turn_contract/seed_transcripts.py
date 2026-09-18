"""
seed_transcripts.py — the observed production failures, as regression fixtures.

Each transcript here is a real call excerpt that went wrong. They are the
acceptance criteria for the conversation-quality work: a change is only
finished when the checks that fire on these stop firing, and no check starts
firing on a transcript that was previously clean.

`awaiting_slot` / `open_slots` annotate what the call was collecting at that
point. A live capture can fill them from state; here they are written by hand
from the flow definitions.
"""

from __future__ import annotations

from scripts.turn_contract.checks import Transcript, Turn

# ── Transcript 1 — ASR cutoff leads to a wrong decline ───────────────────────
# "Yeah. Sure. But" was finalised mid-clause by the ASR. The agent took it as a
# complete yes and moved on, so the caller's real continuation — "Can I ask my
# doctor to send them over?" — arrived one turn late, when the only question
# being listened for was upload_consent (a yes/no about the email link).
#
# doctor_direct is one of four defined values of upload_method, and
# records_coordination.md lists "you can contact my doctor" as an example of
# it. The system declined an option it supports natively.
CUTOFF_WRONG_DECLINE = Transcript(
    name="t1_cutoff_wrong_decline",
    note="ASR cutoff pushes the answer one turn late; it lands on a yes/no slot and is declined",
    turns=(
        Turn(
            "ai",
            "Could you provide the reference number for your adjustment?",
            awaiting_slot="reference_number",
        ),
        Turn("human", "It is four two six nine five eight one seven."),
        Turn(
            "ai",
            "Got it. The request is currently Review. The most recent update on file is dated "
            "2026-01-21. To move forward, we'll need a complete copy of the medical records for "
            "this adjustment. Are you able to provide those?",
            awaiting_slot="upload_method",
            open_slots=("upload_method",),
        ),
        Turn("human", "Yeah. Sure. But"),  # ← ASR finalised mid-clause
        Turn(
            "ai",
            "Of course — that works as well. I can also send a secure link to your email so you "
            "can upload the records yourself. Would you like me to send that over?",
            awaiting_slot="upload_consent",
            open_slots=("upload_consent", "upload_method"),
        ),
        Turn("human", "Can I ask my doctor to send them over?"),
        Turn(
            "ai",
            "That's a great question, and a representative will need to make that change for you.",
            awaiting_slot="upload_consent",
            open_slots=("upload_consent", "upload_method"),
        ),
        Turn("human", "Yeah. Please do that."),
    ),
)


# ── Transcript 2 — a partial value confirmed as complete ─────────────────────
# validate_reference_number requires exactly 8 digits. The caller gave 4. Rather
# than asking for the rest, the system read the partial back for confirmation,
# then read the whole thing back again after the second fragment arrived.
PARTIAL_REFERENCE_NUMBER = Transcript(
    name="t2_partial_reference_number",
    note="4 of 8 digits treated as a complete value; two confirmation round-trips for one number",
    turns=(
        Turn("ai", "Thank you. Is your phone number 512-555-6101?", awaiting_slot="phone_confirmed"),
        Turn("human", "Yes. That is correct."),
        Turn(
            "ai",
            "May I have the reference number of the adjustment request?",
            awaiting_slot="reference_number",
        ),
        Turn("human", "It is four two six nine."),
        Turn("ai", "And that reference number is four two six nine?", awaiting_slot="reference_number"),
        Turn("human", "five eight one seven."),
        Turn(
            "ai",
            "And that reference number is four two six nine five eight one seven?",
            awaiting_slot="reference_number",
        ),
        Turn("human", "Yes. That is correct."),
    ),
)


# ── Transcript 3 — one turn that declines and grants the same request ────────
# SLOT_OWNERSHIP["zip_code"] is route_to_owner — documented as "honored NOW,
# never park or say 'later'". The side-question layer was handed the request
# with no resolution and guessed a decline; the detour opener separately wrote
# the grant; ask_member joined them. Two turns later a generated fragment
# ("Definitely.") is welded into the middle of a static message.
DECLINE_AND_GRANT = Transcript(
    name="t3_decline_and_grant",
    note="two text producers in one turn: a decline and a grant of the same request",
    turns=(
        Turn("human", "Calling for myself."),
        Turn("ai", "Thank you. Your ZIP code is 12139, correct?", awaiting_slot="zip_confirmed"),
        Turn("human", "Yeah."),
        Turn(
            "ai",
            "I have a list of in-network providers in your area ready to send. "
            "Shall I deliver it by fax or email?",
            awaiting_slot="delivery_method",
        ),
        Turn("human", "And you send it to my fax, please?"),
        Turn(
            "ai", "I'll send it to 6171234199 — is that the right fax number?", awaiting_slot="fax_confirmed"
        ),
        Turn("human", "Yeah. That's right. But I just realized my ZIP code's wrong. Can I change it?"),
        Turn(
            "ai",
            "Got it, Emily — a representative would need to make that change to your ZIP code. "
            "Sure — let me update your zip code first. Could you give me your five-digit ZIP code?",
            awaiting_slot="zip_code",
        ),
        Turn("human", "Yeah. It's one two three one nine."),
        Turn(
            "ai",
            "All set — I've updated your ZIP code to 12319 and refreshed your provider list for "
            "that area. Definitely. The fax number we have on file is 6171234199. Is this correct?",
            awaiting_slot="fax_confirmed",
        ),
        Turn("human", "Yeah. That's right."),
    ),
)


# ── Transcript 2, as the flow produces it after the partial-value fix ────────
# The same call, with agent.slots.shapes in place: four digits is measured
# against the reference number's declared eight, so it is held as a fragment
# and the caller is asked only for what is missing. The second piece completes
# the value and the flow moves on — one exchange for the remainder instead of
# two confirmation round-trips, and no fragment ever offered for a yes.
#
# The agent line here is what build_remainder_prompt emits (it draws from a
# two-line pool; this is one of them).
PARTIAL_REFERENCE_NUMBER_FIXED = Transcript(
    name="t2_partial_reference_number_fixed",
    note="the same call once a partial value is asked for rather than confirmed",
    turns=(
        Turn("ai", "Thank you. Is your phone number 512-555-6101?", awaiting_slot="phone_confirmed"),
        Turn("human", "Yes. That is correct."),
        Turn(
            "ai",
            "May I have the reference number of the adjustment request?",
            awaiting_slot="reference_number",
        ),
        Turn("human", "It is four two six nine."),
        Turn(
            "ai",
            "So far I have four two six nine. Could you give me the last four digits?",
            awaiting_slot="reference_number",
        ),
        Turn("human", "five eight one seven."),
        Turn(
            "ai",
            "Got it. The request is currently Review. The most recent update on file is dated 2026-01-21.",
            awaiting_slot="upload_method",
        ),
        Turn("human", "Okay, thank you."),
    ),
)


# ── A clean control ──────────────────────────────────────────────────────────
# Nothing should fire on this. It exists so a check that starts flagging
# ordinary conversation is caught immediately.
CLEAN_CONTROL = Transcript(
    name="t0_clean_control",
    note="an ordinary exchange — every check must stay silent",
    turns=(
        Turn("ai", "Thank you for calling Sagility Health. How can I help today?", awaiting_slot="intent"),
        Turn("human", "I need a list of in-network providers."),
        Turn("ai", "Absolutely. To get started, what's your first name?", awaiting_slot="first_name"),
        Turn("human", "Monique."),
        Turn("ai", "Thank you, Monique. And your last name?", awaiting_slot="last_name"),
        Turn("human", "Customer."),
        Turn("ai", "May I have your Member ID?", awaiting_slot="member_id"),
        Turn("human", "It's M nine zero seven five zero three."),
        Turn("ai", "Thank you. And your date of birth?", awaiting_slot="dob"),
        Turn("human", "April twelfth, nineteen eighty eight."),
    ),
)


ALL_TRANSCRIPTS = (
    CLEAN_CONTROL,
    CUTOFF_WRONG_DECLINE,
    PARTIAL_REFERENCE_NUMBER,
    PARTIAL_REFERENCE_NUMBER_FIXED,
    DECLINE_AND_GRANT,
)

# What each seed transcript is expected to expose today. The runner compares
# against this, so the suite fails both when a known bug stops being detected
# and when a fix lands without this file being updated.
EXPECTED_VIOLATIONS: dict[str, set[str]] = {
    "t0_clean_control": set(),
    "t1_cutoff_wrong_decline": {"declined_open_option"},
    "t2_partial_reference_number": {"partial_readback", "repeat_readback"},
    # The recorded calls above are frozen evidence — fixing the code does not
    # change what was said on those calls, so their expectations never move.
    # This is the shape the same call takes now, and nothing may fire on it.
    "t2_partial_reference_number_fixed": set(),
    # repeat_readback is a true positive here: fax_confirmed is confirmed at
    # turn 5, then re-confirmed at turn 9 after the ZIP detour although the fax
    # never changed — the caller is asked the same thing twice.
    "t3_decline_and_grant": {"contradiction", "wrong_decline", "stray_filler", "repeat_readback"},
}
