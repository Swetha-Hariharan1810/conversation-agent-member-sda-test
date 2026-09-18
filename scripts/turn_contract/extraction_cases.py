"""
extraction_cases.py — the turn-level extraction contract, as data.

One case = one caller utterance in a known state, plus the fields the
extraction LLM must return for it. Graph-free: no agent, no Salesforce, no
routing. One LLM call per case.

Each case carries a `status`:

  "contract"  — the current design can express this, and it must pass today.
                A failure is a regression.
  "gap"       — the target behaviour, which the current schema/prompt cannot
                express yet. Failing is expected; the `fixed_by` field names
                the plan step that should close it. These are the spec for
                that step, written before the work starts.

`expect` is a SUBSET match: only the named fields are compared, so a case
asserts one thing and stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["contract", "gap"]


@dataclass(frozen=True)
class ExtractionCase:
    id: str
    prompt_file: str
    awaiting_slot: str
    utterance: str
    expect: dict[str, Any]
    status: Status = "contract"
    fixed_by: str = ""
    note: str = ""
    last_agent_message: str = ""
    confirmed: dict[str, str] = field(default_factory=dict)
    pending: tuple[str, ...] = ()
    attempt: int = 0
    history: tuple[dict[str, str], ...] = ()


_VERIF_CLAIMS = "extraction/verification_claims.md"
_VERIF_PROVIDER = "extraction/verification_provider.md"
_CLAIM_ADJ = "extraction/claim_adjustment.md"
_RECORDS = "extraction/records_coordination.md"
_DELIVERY = "extraction/delivery_management.md"


CASES: tuple[ExtractionCase, ...] = (
    # ── Baseline: the boring turns must stay boring ──────────────────────────
    ExtractionCase(
        id="dob_clean",
        note="a plain, complete answer — no freeform response needed",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="dob",
        last_agent_message="And your date of birth?",
        utterance="April twelfth nineteen eighty eight.",
        confirmed={"first_name": "Monique", "last_name": "Customer"},
        expect={
            "extracted.dob": "04/12/1988",
            "event_type": "answered",
            "needs_freeform_response": False,
        },
    ),
    ExtractionCase(
        id="member_id_spoken_digits",
        note="spoken digits must normalise, not fall to ambiguous",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="member_id",
        last_agent_message="May I have your Member ID?",
        utterance="My member ID is m nine zero seven five zero three.",
        confirmed={"first_name": "Monique", "last_name": "Customer"},
        expect={"extracted.member_id": "M907503", "event_type": "answered"},
    ),
    ExtractionCase(
        id="name_spelled_wins",
        note="spelled letters are authoritative over the spoken form",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="last_name",
        last_agent_message="Could you please provide your last name?",
        utterance="Ried, R-e-e-d.",
        confirmed={"first_name": "Monique"},
        expect={"extracted.last_name": "Reed"},
    ),
    ExtractionCase(
        id="wait_request",
        note="asking for time is not a failed attempt",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="member_id",
        last_agent_message="May I have your Member ID?",
        utterance="Hold on, let me grab my card.",
        expect={"event_type": "wait"},
    ),
    ExtractionCase(
        id="wait_then_value",
        note="a value in the same breath as a wait — the value wins",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="member_id",
        last_agent_message="May I have your Member ID?",
        utterance="Hold on... okay, it's M451982.",
        expect={"event_type": "answered", "extracted.member_id": "M451982"},
    ),
    ExtractionCase(
        id="cannot_provide",
        note="semantic, not a phrase list",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="member_id",
        last_agent_message="May I have your Member ID?",
        utterance="I never received a card.",
        expect={"cannot_provide": True},
    ),
    ExtractionCase(
        id="fallback_pivot_ssn",
        note="naming another identifier outranks a denial",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="member_id",
        last_agent_message="May I have your Member ID?",
        utterance="I don't have the member ID, can I use my social?",
        expect={"fallback_pivot": "ssn", "cannot_provide": False},
    ),
    ExtractionCase(
        id="bare_correction",
        note="a correction with a value, no answer to the awaiting slot",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="dob",
        last_agent_message="And your date of birth?",
        utterance="Actually, my last name is Smith.",
        confirmed={"first_name": "Monique", "last_name": "Customer", "member_id": "M907503"},
        expect={"event_type": "corrected", "corrections.last_name": "Smith"},
    ),
    ExtractionCase(
        id="plain_non_answer_stays_cheap",
        note="a contentless turn must NOT request the generation LLM",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="last_name",
        last_agent_message="Could you please provide your last name?",
        utterance="what?",
        expect={"needs_freeform_response": False},
    ),
    ExtractionCase(
        id="request_cue_needs_prose",
        note="a real request must NOT get 'I didn't catch that'",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="first_name",
        last_agent_message="Can I get your first name, please?",
        utterance="Please check my claim status today.",
        expect={"needs_freeform_response": True},
    ),
    # ── Transcript 3: answer + update request in one breath ──────────────────
    ExtractionCase(
        id="t3_fax_yes_plus_zip_change",
        note="confirms the fax AND asks to change the ZIP — both halves must survive",
        prompt_file=_DELIVERY,
        awaiting_slot="fax_confirmed",
        last_agent_message="I'll send it to 6171234199 — is that the right fax number?",
        utterance="Yeah. That's right. But I just realized my ZIP code's wrong. Can I change it?",
        confirmed={"first_name": "Emily", "zip_code": "12139", "fax": "6171234199"},
        pending=("delivery_method",),
        expect={
            "event_type": "answered_with_followup",
            "update_target": "zip_code",
            "needs_freeform_response": True,
        },
    ),
    # ── Transcript 1: the cutoff and its consequence ─────────────────────────
    ExtractionCase(
        id="t1_truncated_utterance",
        status="gap",
        fixed_by="Step 1 — fragment merge / incomplete-utterance detection",
        note="ASR finalised mid-clause. Today this reads as a clean yes and the call moves on. "
        "It should be recognised as incomplete so the turn waits instead of advancing.",
        prompt_file=_RECORDS,
        awaiting_slot="upload_method",
        last_agent_message=(
            "To move forward, we'll need a complete copy of the medical records for this "
            "adjustment. Are you able to provide those?"
        ),
        utterance="Yeah. Sure. But",
        expect={"event_type": "ambiguous"},
    ),
    ExtractionCase(
        id="t1_late_option_answer",
        status="gap",
        fixed_by="Step 1 (open-question window) + Step 2 (options in the prompt)",
        note="The caller names a supported upload_method while the awaiting slot is the "
        "upload_consent yes/no. Today it becomes an unownable side question and is declined.",
        prompt_file=_RECORDS,
        awaiting_slot="upload_consent",
        last_agent_message=(
            "I can also send a secure link to your email so you can upload the records "
            "yourself. Would you like me to send that over?"
        ),
        utterance="Can I ask my doctor to send them over?",
        pending=("upload_method",),
        expect={"extracted.upload_method": "doctor_direct"},
    ),
    # ── Transcript 2: the partial value ──────────────────────────────────────
    ExtractionCase(
        id="t2_partial_reference_number",
        status="gap",
        fixed_by="Step 3 — completeness as a first-class field",
        note="4 of the 8 digits. Today there is no way to say 'partial', so it becomes a "
        "value and gets read back. The contract wants it reported as incomplete.",
        prompt_file=_CLAIM_ADJ,
        awaiting_slot="reference_number",
        last_agent_message="May I have the reference number of the adjustment request?",
        utterance="It is four two six nine.",
        expect={"completeness": "partial"},
    ),
    ExtractionCase(
        id="t2_complete_reference_number",
        note="the full eight digits — the control for the case above",
        prompt_file=_CLAIM_ADJ,
        awaiting_slot="reference_number",
        last_agent_message="Could you provide the reference number for your adjustment?",
        utterance="It is four two six nine five eight one seven.",
        expect={"extracted.reference_number": "42695817", "event_type": "answered"},
    ),
    # ── Step 2: the accepted answers are in the prompt ───────────────────────
    ExtractionCase(
        id="upload_method_named_as_a_question",
        note="the transcript-1 utterance, asked on its own slot. doctor_direct is a "
        "defined value and the prompt now says so — asking permission is how people "
        "pick an option.",
        prompt_file=_RECORDS,
        awaiting_slot="upload_method",
        last_agent_message=(
            "To move forward, we'll need a complete copy of the medical records for this "
            "adjustment. Are you able to provide those?"
        ),
        utterance="Can I ask my doctor to send them over?",
        expect={"extracted.upload_method": "doctor_direct"},
    ),
    ExtractionCase(
        id="upload_method_personal_guide_beats_doctor",
        note="both name the provider; the caller is asking US to go and get them",
        prompt_file=_RECORDS,
        awaiting_slot="upload_method",
        last_agent_message="Are you able to provide a copy of the medical records?",
        utterance="Could you contact my doctor's office and get them yourselves?",
        expect={"extracted.upload_method": "personal_guide"},
    ),
    ExtractionCase(
        id="delivery_method_chosen_as_a_question",
        note="transcript 3, turn 5 — a choice phrased as a question",
        prompt_file=_DELIVERY,
        awaiting_slot="delivery_method",
        last_agent_message=(
            "I have a list of in-network providers in your area ready to send. "
            "Shall I deliver it by fax or email?"
        ),
        utterance="And you send it to my fax, please?",
        confirmed={"first_name": "Emily", "zip_code": "12139"},
        expect={"extracted.delivery_method": "fax", "event_type": "answered"},
    ),
    ExtractionCase(
        id="closed_set_is_not_forced",
        note="the accepted-answers block must not make the model pick one. An empty "
        "slot is recoverable; a wrong one that reaches confirmation is not.",
        prompt_file=_DELIVERY,
        awaiting_slot="delivery_method",
        last_agent_message="Shall I deliver it by fax or email?",
        utterance="How long does it usually take to arrive?",
        expect={"extracted.delivery_method": None},
    ),
    # ── Guards must keep working ─────────────────────────────────────────────
    ExtractionCase(
        id="guard_transfer",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="first_name",
        last_agent_message="Can I get your first name, please?",
        utterance="Just put me through to a real person.",
        expect={"guard": "TRANSFER_REQUEST"},
    ),
    ExtractionCase(
        id="smalltalk_is_not_offtopic",
        note="a pleasantry must not burn a retry attempt",
        prompt_file=_VERIF_CLAIMS,
        awaiting_slot="first_name",
        last_agent_message="Can I get your first name, please?",
        utterance="How are you doing today?",
        expect={"guard": "NONE"},
    ),
)
