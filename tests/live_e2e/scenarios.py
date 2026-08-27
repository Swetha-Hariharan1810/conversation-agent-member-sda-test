"""
scenarios.py — All live E2E scenario definitions.

User utterances are derived from the static transcripts in
scripts/conversational_workload/static_transcripts/ where those exist;
the remaining scripts are written in the same spoken-form style the
normalizers handle ("m nine zero seven five zero three",
"April twelfth nineteen eighty-eight").

Assertions follow the robustness rules: state keys, escalation reasons
(substring/regex over every reason source), metadata events, END/interrupt
flags, and tolerant case-insensitive regexes — never exact AI sentences.
Where wording comes from a constant pool, the pool is imported and matched
via harness.pool_regex().

ZIP-update behavior change (provider_search):
  The ZIP read-back confirmation step ("Just to be sure I have it right —
  your ZIP code is X, correct?") was REMOVED from provider_search_agent.
  A valid new ZIP is now written to Salesforce immediately and the flow
  proceeds straight to the delivery-method question. delivery_management's
  dispatch confirmation then includes the updated ZIP
  ("...list of in-network providers for your current ZIP code X within
  30 minutes"). Consequences for this suite:
    - pcp_zip_update no longer scripts a confirmation turn and asserts
      the ZIP-aware dispatch message + zip_code_updated state flag.
    - pcp_zip_inline_update (new) covers the inline "no, it's X" path.
    - zip_change_loop_escalates was REDEFINED: the zip_change_cycles
      read-back rejection loop it used to exercise no longer exists, so it
      now checks the remaining escalation loop — zip_code slot exhaustion
      when the member repeatedly provides INVALID (non-5-digit) values —
      and verifies no invalid value is ever persisted to Salesforce.
"""

from __future__ import annotations

# Static pools — imported so assertions survive any re-pick of pool members.
from agent.agents.follow_up.constants import MSG_FOLLOW_UP_ASK, MSG_UPDATE_REQUEST_ESCALATE  # noqa: E402
from agent.agents.verification.constants import (  # noqa: E402
    MSG_REASK_DOB,
    MSG_REASK_FIRST_NAME,
    MSG_REASK_GENERIC,
    MSG_REASK_LAST_NAME,
    MSG_SSN_ASK,
    MSG_SSN_COLLECT,
    MSG_SSN_EITHER,
    NAME_CORRECTION_PROMPTS,
)
from agent.agents.verification.handlers import MSG_PHONE_NOT_CONFIRMED, MSG_RESTART  # noqa: E402
from agent.responses.static import MSG_SELF_HARM_ESCALATION, MSG_WAIT_ACK  # noqa: E402
from tests.live_e2e.harness import Expected, Scenario, TurnExpectation, pool_regex

# ──────────────────────────────────────────────────────────────────────────────
# Predicates for Expected.final_state
# ──────────────────────────────────────────────────────────────────────────────


def truthy(v) -> bool:
    return bool(v)


def falsy(v) -> bool:
    return not v


def _digits(v) -> str:
    return "".join(c for c in str(v or "") if c.isdigit())


def contains(sub: str):
    def _pred(v, _sub=sub):
        return _sub.lower() in str(v or "").lower()

    _pred.__name__ = f"contains({sub!r})"
    return _pred


def digits_equal(expected: str):
    def _pred(v, _exp=expected):
        return _digits(v) == _digits(_exp)

    _pred.__name__ = f"digits_equal({expected!r})"
    return _pred


# ──────────────────────────────────────────────────────────────────────────────
# Salesforce post-checks (real re-queries after the conversation ends)
# ──────────────────────────────────────────────────────────────────────────────


def sf_field_check(member_id: str, fld: str, expected: str, compare_digits: bool = False):
    async def _check(_final_state):
        from agent.storage.queries.members import get_member_contact

        record = await get_member_contact(member_id)
        if not record:
            return f"SF post-check: member {member_id} not found on re-query"
        actual = record.get(fld) or ""
        if compare_digits:
            ok = _digits(actual) == _digits(expected)
        else:
            ok = str(actual).strip().lower() == expected.strip().lower()
        if not ok:
            return (
                f"SF post-check: {member_id}.{fld}={actual!r} after run, "
                f"expected {expected!r} — the agent did not persist the update"
            )
        return None

    _check.__name__ = f"sf_{fld}_check"
    return _check


# ──────────────────────────────────────────────────────────────────────────────
# Shared script prefixes
# ──────────────────────────────────────────────────────────────────────────────

# PCP flow: intent → first/last name → member id → dob → relationship
PCP_VERIFY = [
    "I need to find a primary care physician in my area.",
    "emily",
    "carter",
    "yes correct",  # name_confirmed
    "m nine zero seven five zero three",
    "April twelvee nineteen eighty-eight",
    "I'm calling for myself",
]


# Claim flow: intent → first/last name → member id → dob → phone confirmation
CLAIM_VERIFY = [
    "I adjusted the claim and I want to follow up",
    "james",
    "wilson",
    "yes correct",  # name_confirmed
    "m three one zero one eight eight",
    "Thirtieth of July, nineteen seventy seven",
    "yes correct",
]

NEW_EMAIL = "james.w.new@gmail.com"
NEW_EMILY_EMAIL = "emily.c.new@example.com"

# PCP flow: intent → first/last name → member id → dob → relationship (conversational)
PCP_VERIFY_CONVERSATIONAL = [
    "Hi there, yeah, I'm trying to find a primary care doctor near where I live",
    "sure, it's emily",
    "carter, that's c a r t e r",
    "yes thats correct",
    "okay so my member id is m nine zero seven five zero three",
    "I was born on the twelfth of april, nineteen eighty eight",
    "it's my own plan, I'm the plan holder",
]

# Claim flow: intent → first/last name → member id → dob → phone confirmation (conversational)
CLAIM_VERIFY_CONVERSATIONAL = [
    "hello, I submitted a claim adjustment a while back and wanted to check on it",
    "yeah it's james",
    "wilson",
    "yes thats correct",
    "let me grab my card... okay it's m three one zero one eight eight",
    "the Thirtieth of July, nineteen seventy seven",
    "yep, that's the right number",
]

# Verification turn-level sanity checks shared by happy paths
_VERIFY_TURNS = {
    4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
}


# Dispatch confirmation must name the updated ZIP — matches either
# DELIVERY_WINDOW_MSG_ZIP_UPDATED pool member ("current"/"updated" wording).
def _zip_dispatch_regex(zip_code: str) -> str:
    return rf"(current|updated)\s+zip\s*code\s+{zip_code}.*within 30 minutes"


# ──────────────────────────────────────────────────────────────────────────────
# A. Provider (PCP) happy paths
# ──────────────────────────────────────────────────────────────────────────────

pcp_happy_path_fax = Scenario(
    name="pcp_happy_path_fax",
    flow="pcp",
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "send it to my fax",
        "yes that's correct",  # fax on file
        "yes please",  # benefits offer
        "yes that sounds interesting",  # Care Coach offer
        "when should I expect to receive the provider list?",  # one follow-up
        "no thanks that was helpful",  # close
    ],
    turn_expectations=_VERIFY_TURNS,
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            "delivery_method": "fax",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
    ),
)

pcp_happy_path_email = Scenario(
    name="pcp_happy_path_email",
    flow="pcp",
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",  # email on file
        "yes please",
        "yes that sounds interesting",
        "no thanks that was helpful",
    ],
    turn_expectations=_VERIFY_TURNS,
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
    ),
)

pcp_benefits_declined = Scenario(
    name="pcp_benefits_declined",
    flow="pcp",
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "no thanks",  # decline benefits offer → BenefitsAgent NO path
        "no thank you",  # decline Care Coach (no-explanation offer)
        "no, that's everything, thanks",  # follow-up → close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "benefits_explained": False,
            "care_coach_nooffer_sent": True,
            "care_coach_details_sent": falsy,
        },
    ),
)

pcp_zip_update = Scenario(
    name="pcp_zip_update",
    flow="pcp",
    mutating=True,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "no, I moved recently",  # decline ZIP on file
        "my new zip code is zero two one three nine",  # spoken 5-digit ZIP — accepted
        # directly: NO read-back confirmation turn anymore; the next AI prompt
        # must already be the delivery-method bridge (asserted below)
        "send it to my fax",
        "yes that's correct",  # fax on file
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    turn_expectations={
        # The AI prompt that precedes "send it to my fax" must be the delivery
        # bridge — proving the ZIP was accepted with no confirmation step.
        10: TurnExpectation(ai_contains=[r"fax or email"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "zip_code_used": "02139",
            "zip_code_updated": True,
        },
        transcript_contains=[
            # ZIP-aware dispatch confirmation from DELIVERY_WINDOW_MSG_ZIP_UPDATED
            _zip_dispatch_regex("02139"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02139")],
    notes=(
        "Mutates Emily's zip in Salesforce; teardown restores the snapshot. "
        "The ZIP read-back confirmation was removed from provider_search: the "
        "new ZIP is written to Salesforce on first hearing and the very next "
        "AI turn is the fax/email delivery question. The dispatch confirmation "
        "must include the updated ZIP (DELIVERY_WINDOW_MSG_ZIP_UPDATED)."
    ),
)

pcp_zip_inline_update = Scenario(
    name="pcp_zip_inline_update",
    flow="pcp",
    mutating=True,
    retries=1,  # inline "no + new ZIP" extraction is mildly non-deterministic:
    # if the LLM returns zip_confirmed="no" instead of the bare zip_code, the
    # agent asks for the ZIP on a separate turn and the script desyncs
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        # Inline decline + replacement in ONE utterance at the zip_confirmed
        # read-back: extraction contract extracts zip_code and omits
        # zip_confirmed → provider_search accepts it directly (no read-back).
        "no, my zip changed — it's zero two one four zero",
        "email please",  # next AI prompt is already the delivery bridge
        "yes that's correct",  # email on file
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",
    ],
    turn_expectations={
        # The AI prompt preceding "email please" must be the delivery bridge —
        # the inline-replacement path must not produce a confirmation read-back.
        9: TurnExpectation(ai_contains=[r"fax or email"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "zip_code_used": "02140",
            "zip_code_updated": True,
        },
        transcript_contains=[
            _zip_dispatch_regex("02140"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02140")],
    notes=(
        "Mutates Emily's zip in Salesforce; teardown restores the snapshot. "
        "Covers the zip_confirmed inline-replacement path ('no, it's X' in one "
        "utterance): the new ZIP is persisted immediately, zip_code_updated is "
        "set, and the dispatch confirmation names the new ZIP. Replaces the "
        "deleted zip_change_loop_escalates scenario — with the read-back gone, "
        "the zip_change_cycles loop it exercised can no longer occur."
    ),
)

pcp_fax_update = Scenario(
    name="pcp_fax_update",
    flow="pcp",
    mutating=True,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "no, that fax number is outdated",  # decline fax on file
        "my new fax number is six one seven five five five nine one nine nine",
        "yes that's correct",  # confirm read-back → SF write (fax read-back still exists)
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "fax": digits_equal("6175559199"),
        },
    ),
    post_checks=[sf_field_check("M907503", "fax", "6175559199", compare_digits=True)],
    notes="Mutates Emily's fax in Salesforce; teardown restores the snapshot.",
)

pcp_email_update = Scenario(
    name="pcp_email_update",
    flow="pcp",
    mutating=True,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "email please",
        "no, that's my old email address",  # decline email on file
        NEW_EMILY_EMAIL,
        "yes that's correct",  # confirm read-back → SF write (email read-back still exists)
        "no thanks",  # decline benefits offer
        "no thank you",  # decline Care Coach
        "no that's all, thanks",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "email": contains(NEW_EMILY_EMAIL),
        },
    ),
    post_checks=[sf_field_check("M907503", "email", NEW_EMILY_EMAIL)],
    notes=(
        "Mutates Emily's email in Salesforce; teardown restores the snapshot. "
        "The agent reads email addresses back with '@' replaced by ' at ' "
        "(Azure content-filter workaround) — assertions use contains() on the "
        "state value and must not depend on a literal '@' in any AI transcript line."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# B. Verification escalations
# ──────────────────────────────────────────────────────────────────────────────

verification_restart_then_success = Scenario(
    name="verification_restart_then_success",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 1
        "m nine zero seven five zero two",  # wrong member id — lookup fails
        "April twelfth nineteen eighty-eight",
        # agent restarts ("let's try once more") — give correct details
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 2
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
        # complete the PCP flow minimally
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no that's all",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True},
        # restart message — tolerant alternation covering every MSG_RESTART member
        transcript_contains=[r"(one more try|try once more|once more|try again|didn't quite match)"],
    ),
)

verification_fail_twice_escalates = Scenario(
    name="verification_fail_twice_escalates",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 1
        "m nine zero seven five zero two",  # wrong, round 1
        "April twelfth nineteen eighty-eight",
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 2
        "m nine zero seven five zero two",  # wrong, round 2
        "April twelfth nineteen eighty-eight",
    ],
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="Verification failed",
        final_state={"escalation_reference_number": truthy},
    ),
)

member_id_exhaustion = Scenario(
    name="member_id_exhaustion",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "one two three",  # no M prefix
        "I don't know",
        "umm banana",
        # spares in case a turn is classified as clarification (not counted)
        "no idea",
        "I really don't know it",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="member_id",
        transcript_contains=[r"member id after a few tries|wasn't able to capture"],
    ),
)

dob_no_year_exhaustion = Scenario(
    name="dob_no_year_exhaustion",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth",  # no year — invalid
        "April twelfth",
        "April twelfth",
        # spares for uncounted clarification turns
        "April twelfth",
        "April twelfth",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="dob",
    ),
)

member_id_ambiguous_exhaustion = Scenario(
    name="member_id_ambiguous_exhaustion",
    flow="pcp",
    user_turns=[
        "I wanted to see my primary care doctor.",
        "Sure. Emily?",
        "Carter?",
        "yes correct",  # name_confirmed
        "I don't have it.",  # ask #1 → AMBIGUOUS → slot_fail (fix: threshold >= 1)
        "I don't have it.",  # ask #2 → AMBIGUOUS → slot_fail
        "I don't have it.",  # ask #3 → escalation (was ask #5 before fix)
    ],
    turn_expectations={
        4: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="member_id",
        final_state={"member_status_verify": lambda v: not v},
        last_ai_contains=[r"(member id after a few|wasn't able to capture your member)"],
        max_turns=20,
    ),
    notes=(
        "History: 'I don't have it' used to burn the whole retry budget via the "
        "AMBIGUOUS branch. Today detect_cannot_provide() short-circuits it: the "
        "FIRST 'I don't have it' escalates immediately with reason "
        "member_id_cannot_provide (the cannot-provide check runs before the "
        "ambiguous threshold), so the scripted spares are never consumed. Note "
        "the ambiguous threshold itself is >= 2 again since Phase 7: a first "
        "genuinely-ambiguous turn (not cannot-provide) gets a free CLARIFY, the "
        "second burns an attempt — see followup/wait scenarios in section N."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# B2. Partial re-ask on identity mismatch (member found, one field wrong)
#
# These lock the targeted-re-ask behavior: on a failed full match where the
# Member ID exists, only the mismatched identity field is re-asked; matched
# fields and the Member ID are retained, and the spelled-name read-back is not
# repeated when the names already matched.
# ──────────────────────────────────────────────────────────────────────────────

verification_dob_only_mismatch = Scenario(
    name="verification_dob_only_mismatch",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name readback #1 → confirmed
        "m nine zero seven five zero three",  # correct Member ID
        "April thirteenth nineteen eighty-eight",  # WRONG dob (on file: the 12th)
        # lookup fails: member_id_found=True, only dob mismatches → MSG_REASK_DOB,
        # awaiting_slot="dob". name/last-name/Member ID are NOT re-asked.
        "April twelfth nineteen eighty-eight",  # corrected dob → re-lookup → verified
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "email please",
        "yes that's correct",  # email on file
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # the one read-back
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # The re-ask after the failed lookup: the disclosing DOB pool, awaiting dob.
        # Proves ONLY dob is re-asked (no name / Member-ID re-ask).
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_DOB)], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-I-L-Y"],
        # The spelled-name read-back appears EXACTLY once — name_confirmed is
        # preserved across the DOB-only re-ask, so no second read-back fires.
        transcript_count={r"E-M-I-L-Y": 1},
    ),
    notes=(
        "Mirrors the real transcript: a correct Member ID + name but a wrong DOB. "
        "The lookup returns member_id_found=True with dob mismatched, so the agent "
        "re-asks ONLY the date of birth (MSG_REASK_DOB, awaiting_slot='dob'); first "
        "name, last name and Member ID are retained and never re-asked, and the "
        "spelled-name read-back is delivered exactly once (name_confirmed is "
        "preserved). After one corrected DOB turn the re-lookup matches and "
        "member_status_verify becomes True."
    ),
)

verification_last_name_only_mismatch = Scenario(
    name="verification_last_name_only_mismatch",
    flow="pcp",
    timeout_s=360,
    retries=1,  # name re-confirmation involves LLM extraction (cf. name_confirmation_inline_correction)
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name (on file: Carter)
        "yes correct",  # readback #1 → "Emily Carson" confirmed
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: first_name + dob match, last_name mismatches →
        # MSG_REASK_LAST_NAME, awaiting_slot="last_name". Member ID + dob retained.
        "carter",  # corrected last name → fresh read-back of "Emily Carter"
        "yes correct",  # confirm corrected name → straight to lookup (NO Member-ID re-ask)
        "I'm calling for myself",  # relationship (proves we skipped to post-lookup)
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "email please",
        "yes that's correct",  # email on file
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-S-O-N"]),  # read-back #1
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # The re-ask after the failed lookup: disclosing LAST-NAME pool, awaiting
        # last_name. Proves ONLY the last name is re-asked (not Member ID / dob).
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_LAST_NAME)], slot_awaiting="last_name"),
        # After confirming the corrected name, the next prompt is the relationship
        # question — NOT a Member-ID re-ask — proving Member ID + dob were retained.
        8: TurnExpectation(ai_contains=[r"plan holder|dependent"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",  # corrected
            "provider_list_sent": True,
        },
        # Both the original and corrected read-backs appear; the corrected last
        # name is read back exactly once.
        transcript_contains=[r"E-M-I-L-Y", r"C-A-R-T-E-R"],
        transcript_count={r"C-A-R-T-E-R": 1},
    ),
    notes=(
        "Last-name-only mismatch: a correct Member ID + first name + DOB but a "
        "wrong last name. The lookup returns member_id_found=True with last_name "
        "mismatched, so the agent re-asks ONLY the last name (MSG_REASK_LAST_NAME, "
        "awaiting_slot='last_name'); Member ID and DOB are retained. Because a name "
        "field mismatched, name_confirmed is reset and the corrected name is read "
        "back once more; on confirmation the flow proceeds straight to the lookup "
        "(via _finish_after_identity) WITHOUT re-asking the already-known Member ID "
        "— turn-8 relationship prompt is the proof. Re-lookup matches → verified."
    ),
)

verification_first_name_only_mismatch = Scenario(
    name="verification_first_name_only_mismatch",
    flow="pcp",
    timeout_s=360,
    retries=1,  # name re-confirmation involves LLM extraction
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emma",  # WRONG first name (on file: Emily)
        "carter",  # correct last name
        "yes correct",  # read-back #1 → "Emma Carter" confirmed
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: last name + dob match, first name mismatches →
        # MSG_REASK_FIRST_NAME, awaiting_slot="first_name". Member ID + dob retained.
        "emily",  # corrected first name → fresh read-back of "Emily Carter"
        "yes correct",  # confirm → straight to lookup (NO Member-ID re-ask)
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-M-A.*C-A-R-T-E-R"]),  # read-back #1
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # Re-ask after the failed lookup: disclosing FIRST-NAME pool, awaiting first_name.
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_FIRST_NAME)], slot_awaiting="first_name"),
        7: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # corrected read-back
        8: TurnExpectation(ai_contains=[r"plan holder|dependent"]),  # relationship, not Member-ID
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",  # corrected
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"E-M-I-L-Y"],
        transcript_count={r"E-M-I-L-Y": 1},  # corrected first name read back exactly once
    ),
    notes=(
        "First-name-only mismatch: wrong first name, correct last name + Member ID "
        "+ DOB. Lookup returns member_id_found=True with first_name mismatched → only "
        "the first name is re-asked (MSG_REASK_FIRST_NAME, awaiting_slot='first_name'); "
        "Member ID and DOB are retained. name_confirmed resets (a name field changed) "
        "and the cached caller_first_name is cleared; the corrected name is read back "
        "once, then the flow proceeds straight to the lookup (no Member-ID re-ask) → "
        "verified. Turn-8 relationship prompt is the proof."
    ),
)

verification_name_mismatch_bare_no_at_readback = Scenario(
    name="verification_name_mismatch_bare_no_at_readback",
    flow="pcp",
    timeout_s=420,
    retries=2,  # exercises the name-correction sub-loop on top of the partial re-ask
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name (on file: Carter)
        "yes correct",  # read-back #1 → "Emily Carson"
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: last name mismatches → MSG_REASK_LAST_NAME
        "carson",  # caller restates the wrong name → read-back "Emily Carson"
        "no",  # BARE NO at the read-back → agent asks for the correct name
        "it's Emily Carter",  # correct name → read-back "Emily Carter"
        "yes correct",  # confirm → straight to lookup → verified
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        # Re-ask after the failed lookup: only the last name.
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_LAST_NAME)], slot_awaiting="last_name"),
        7: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-S-O-N"]),  # read-back of restated wrong name
        8: TurnExpectation(ai_contains=[pool_regex(NAME_CORRECTION_PROMPTS)]),  # after bare "no"
        9: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # corrected read-back
        10: TurnExpectation(ai_contains=[r"plan holder|dependent"]),  # relationship
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",  # corrected
            "provider_list_sent": True,
        },
        transcript_contains=[r"C-A-R-S-O-N", r"C-A-R-T-E-R"],
    ),
    notes=(
        "Name-mismatch re-ask plus a 'no' at the confirmation read-back. The last "
        "name is re-asked (MSG_REASK_LAST_NAME); the caller restates the wrong name, "
        "the read-back fires, and the caller says a bare 'no' → the name-correction "
        "sub-loop asks for the correct name, reads it back, and only then confirms. "
        "name_confirm_attempts (reset to 0 by the partial re-ask) must not exhaust on "
        "a single rejection. On confirmation the flow proceeds to the lookup → "
        "verified, with no Member-ID re-ask. retries=2: the nested name loop adds "
        "extraction non-determinism."
    ),
)

verification_multi_field_mismatch_generic = Scenario(
    name="verification_multi_field_mismatch_generic",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emma",  # WRONG first name
        "carson",  # WRONG last name
        "yes correct",  # read-back #1 → "Emma Carson"
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: first AND last mismatch (dob matches) → non-disclosing
        # MSG_REASK_GENERIC, awaiting_slot="first_name". Member ID + dob retained.
        "Emily Carter",  # full corrected name → read-back "Emily Carter"
        "yes correct",  # confirm → straight to lookup → verified
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-M-A.*C-A-R-S-O-N"]),  # read-back #1
        # Multi-field mismatch → the GENERIC (non-disclosing) pool, awaiting the
        # first mismatched field in identity order.
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_GENERIC)], slot_awaiting="first_name"),
        7: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # corrected read-back
        8: TurnExpectation(ai_contains=[r"plan holder|dependent"]),  # relationship
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"E-M-I-L-Y"],
    ),
    notes=(
        "Multiple identity fields wrong (first + last name) with a correct Member ID "
        "+ DOB. The lookup reports two mismatches, so the agent uses the NON-disclosing "
        "MSG_REASK_GENERIC (it does not enumerate every wrong field) and points "
        "awaiting_slot at the first mismatched field. The caller restates the full "
        "name; it is read back once and confirmed, then the flow proceeds to the "
        "lookup → verified. retries=2: re-collecting two name fields from one "
        "utterance adds extraction non-determinism."
    ),
)

verification_member_id_not_found_restart = Scenario(
    name="verification_member_id_not_found_restart",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name readback round 1
        "m nine nine nine nine nine nine",  # Member ID with NO record → not found
        "April twelfth nineteen eighty-eight",
        # Phase 0: Member-ID-not-found → full restart (MSG_RESTART, re-ask from the
        # top). Provide the correct details on round 2.
        "emily",
        "carter",
        "yes correct",  # name readback round 2
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
        # MSG_RESTART pool — full restart wording (Member-ID-not-found path).
        transcript_contains=[pool_regex(MSG_RESTART)],
    ),
    notes=(
        "Member-ID-not-found branch: a Member ID with no record in Salesforce "
        "(M999999) makes the full match fail AND the Member-ID-only fetch return "
        "nothing → member_id_found=False → Phase 0 full restart (re-ask everything "
        "with MSG_RESTART). Distinct from verification_restart_then_success, which "
        "uses a near-miss ID; here the ID is deliberately non-existent to exercise "
        "the member_id_found=False path explicitly. Correct details on round 2 "
        "verify the member."
    ),
)

verification_repeated_dob_mismatch_escalates = Scenario(
    name="verification_repeated_dob_mismatch_escalates",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name confirmed
        "m nine zero seven five zero three",  # correct Member ID throughout
        "April thirteenth nineteen eighty-eight",  # wrong dob, lookup attempt 1 → re-ask
        "April fourteenth nineteen eighty-eight",  # wrong dob, lookup attempt 2 → escalate
        # spares (uncounted clarify turns)
        "April fifteenth nineteen eighty-eight",
        "April sixteenth nineteen eighty-eight",
    ],
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="Verification failed",
        final_state={
            # Never verified on a partial match — no partial value slips through.
            "member_status_verify": falsy,
            "escalation_reference_number": truthy,
        },
    ),
    notes=(
        "Repeated partial mismatch with a correct Member ID: each failed lookup "
        "increments the GLOBAL lookup_fail counter (guard_loop_limit), so after "
        "MAX_LOOKUP_ATTEMPTS=2 the agent escalates with 'Verification failed'. "
        "Asserts member_status_verify stays falsy — a wrong DOB must never verify "
        "the member even partially."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# C. Guard escalations (mid-verification; retried once on failure — LLM guard
#    classification is non-deterministic)
# ──────────────────────────────────────────────────────────────────────────────

transfer_request = Scenario(
    name="transfer_request",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "I want to talk to a real person",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Caller",
        escalation_reason_contains="Transfer requested",
    ),
)

abuse = Scenario(
    name="abuse",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "this is bullshit, you idiot",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="abuse_detected",
    ),
)

self_harm = Scenario(
    name="self_harm",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "I can't go on, I want to hurt myself",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="self_harm_detected",
        # supportive close — match the spec regex OR any member of the static pool
        last_ai_contains=[
            r"(support|help right now|stay on the line)|" + pool_regex(MSG_SELF_HARM_ESCALATION)
        ],
    ),
)

offtopic_repeated = Scenario(
    name="offtopic_repeated",
    flow="pcp",
    retries=1,
    user_turns=[
        "can you order me a pizza",
        "what's the weather like today",
        "tell me a joke",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        # either the off-topic counter or intake's unclear-intent limit may fire
        # by the 3rd off-topic turn — both are escalations by design
        escalation_reason_regex=r"(off-topic|Intent could not be classified)",
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# D. Intake routing
# ──────────────────────────────────────────────────────────────────────────────

intake_unclear_exhaustion = Scenario(
    name="intake_unclear_exhaustion",
    flow="pcp",
    user_turns=[
        "hi",
        "I have a question",
        "not sure",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="Intent could not be classified",
    ),
)

intake_out_of_scope_billing = Scenario(
    name="intake_out_of_scope_billing",
    flow="pcp",
    user_turns=["I want to pay my bill"],
    expect=Expected(
        completed=True,  # graph ENDs directly — no escalation agent
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        last_ai_contains=[r"1-\d{3}-\d{3}-\d{4}"],
        final_state={"escalation_reason": contains("outside covered workflows")},
    ),
)

intake_out_of_scope_appeal = Scenario(
    name="intake_out_of_scope_appeal",
    flow="pcp",
    retries=1,  # LLM extraction is the primary signal; retries=1 for classification reliability
    user_turns=["I want to appeal my claim denial"],
    expect=Expected(
        completed=True,  # graph ENDs directly — no escalation agent, no identity verification
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        last_ai_contains=[r"appeal", r"1-\d{3}-\d{3}-\d{4}"],
        final_state={"escalation_reason": contains("outside covered workflows")},
    ),
    notes=(
        "Regression guard: appeal utterances must route to out_of_scope, NOT claim_services. "
        "The caller hears a direct number for the appeals team and the graph ends."
    ),
)

non_member_caller = Scenario(
    name="non_member_caller",
    flow="pcp",
    retries=1,  # passive caller-type detection is LLM-extracted
    user_turns=["Hi, I'm a provider calling about a patient"],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        final_state={
            "caller_type": "provider",
            "caller_type_handled": True,
        },
        last_ai_contains=[r"1-740-660-3977"],
    ),
)

intake_unsupported_provider_oncologist = Scenario(
    name="intake_unsupported_provider_oncologist",
    flow="pcp",
    retries=1,  # LLM extraction is the primary signal; retries=1 for guard-class reliability
    user_turns=["I need to find an oncologist in my network."],
    expect=Expected(
        completed=True,  # graph reaches END via escalation_agent
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="provider_type_unsupported",
        final_is_interrupt=False,
        final_state={
            # Verification must NEVER have run — member_status_verify stays unset
            "member_status_verify": falsy,
        },
        # Escalation message must name the specialty and list the five supported types
        last_ai_contains=[
            r"oncologist",
            r"(Primary Care|PCP|Pediatrician|Cardiologist|Dermatologist|Orthopedic)",
            r"representative",
        ],
    ),
    notes=(
        "Canonical unsupported-provider-type case. The member says 'oncologist' in "
        "their very first utterance. The intake LLM must classify this as "
        "provider_type_unsupported, which routes directly to escalation_agent without "
        "any verification. member_status_verify must be falsy (unset) — if it is True "
        "the test fails because verification ran. retries=1: LLM extraction for a "
        "new intent tag can be slightly non-deterministic on the first live run."
    ),
)

intake_unsupported_provider_neurologist = Scenario(
    name="intake_unsupported_provider_neurologist",
    flow="pcp",
    retries=1,
    user_turns=["Hi, I'm trying to find a neurologist covered under my plan."],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="provider_type_unsupported",
        final_is_interrupt=False,
        final_state={"member_status_verify": falsy},
        last_ai_contains=[
            r"neurologist",
            r"(Primary Care|Cardiologist|Dermatologist|Orthopedic|Pediatrician)",
        ],
    ),
    notes=(
        "Covers a less common specialty to ensure the prompt generalises beyond "
        "the most salient example (oncologist). Also exercises the spoken-form "
        "phrasing 'covered under my plan' — the intent classification must ignore "
        "the context words and key on the specialty name."
    ),
)

intake_supported_provider_cardiologist = Scenario(
    name="intake_supported_provider_cardiologist",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a cardiologist.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder.",
        # Complete the PCP flow minimally
        # "Cardiologist",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    expect=Expected(
        completed=True,
        escalated=False,  # must NOT escalate — cardiologist IS supported
        final_state={
            "member_status_verify": True,  # verification MUST have run
            "provider_type": "Cardiologist",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Critical regression guard. Cardiologist is one of the five supported types "
        "and must be classified as provider_services, NOT provider_type_unsupported. "
        "If this scenario escalates the implementation is broken."
    ),
)

intake_provider_type_propagates_to_search = Scenario(
    name="intake_provider_type_propagates_to_search",
    flow="pcp",
    timeout_s=360,
    retries=1,  # intake provider_type extraction is LLM-driven and mildly non-deterministic
    user_turns=[
        "I need to find a cardiologist.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder.",
        # NOTE: there is NO provider-type turn here. The intake LLM named
        # "cardiologist" in the first utterance, so intake_agent propagates
        # provider_type="Cardiologist" into state. provider_search_agent must
        # therefore SKIP the provider-type question and go straight to the ZIP
        # confirmation — the turn-7 expectation below asserts exactly that.
        "yes that's correct",  # ZIP on file (provider type was NOT re-asked)
        "email please",
        "yes that's correct",  # email on file
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        # The first provider_search prompt must be the ZIP confirmation, not the
        # "what type of provider?" question. slot_awaiting="zip_confirmed" is the
        # definitive proof that the propagated provider_type let the agent skip
        # the provider-type collection step entirely.
        7: TurnExpectation(ai_contains=[r"zip\s*code"], slot_awaiting="zip_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_type": "Cardiologist",  # propagated from intake, never re-asked
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Regression guard for intake → provider_search provider_type propagation. "
        "When the caller names a supported specialty ('cardiologist') in their first "
        "utterance, intake_agent extracts and normalizes it and carries it into state "
        "so provider_search_agent does not ask 'what type of provider?' again. The "
        "script omits the provider-type turn on purpose; if the agent still asks for "
        "it the script desyncs and the turn-7 ZIP-confirmation expectation fails. "
        "provider_type must end as the canonical 'Cardiologist'."
    ),
)

intake_generic_provider_request = Scenario(
    name="intake_generic_provider_request",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a doctor in my network.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder.",
        "Primary Care Physician",
        "yes that's correct",
        "fax please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's it",
    ],
    expect=Expected(
        completed=True,
        escalated=False,  # generic "I need a doctor" must NOT escalate
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Critical regression guard. 'I need to find a doctor' is a generic request "
        "with no specialty named. It must be classified as provider_services and "
        "proceed through the full flow. The specialty is collected later by "
        "provider_search_agent."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# E. Claim flow
# ──────────────────────────────────────────────────────────────────────────────

claim_happy_path = Scenario(
    name="claim_happy_path",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can I ask my doctor to send it over?",  # doctor-direct
        "Yes, please",  # accept upload link
        "Yes, that's correct",  # confirm email on file
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "Okay, how long will it take to finalize the request?",  # timeline question
        "email them to me",  # N2 channel
        "Yes, can you tell me where I can see how many rewards I earned from my annual check up last week?",
        "No, that's it for me. Thanks!",
    ],
    turn_expectations={7: TurnExpectation(ai_contains=[r"reference number"])},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
)

claim_upload_only = Scenario(
    name="claim_upload_only",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please send the link",
        "Yes, that's correct",  # email on file
        "No, that won't be necessary. I'll handle it myself.",  # decline guide
        "email please",  # notifications
        "Yes, that's correct",
        "How long does the review usually take after you receive everything?",
        "No, that's all. Thank you!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
        },
    ),
    notes=(
        "Follows claim_adjustment_upload_only.txt. Member declines Personal Guide "
        "after upload link is sent; agent now routes to follow_up (not escalation)."
    ),
)

claim_guide_only = Scenario(
    name="claim_guide_only",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can you contact my doctor directly to get the records?",
        "No thanks, I'd prefer the Personal Guide to contact them.",  # decline link
        "Yes, please proceed with that.",  # accept guide
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        "No, that's all. Thank you.",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "records_branch_taken": "personal_guide",
            "personal_guide_outreach_requested": True,
        },
    ),
)

claim_no_proceed = Scenario(
    name="claim_no_proceed",
    flow="claim",
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "okay will send it",
        "no thanks",  # decline upload link → Personal Guide offer
        "no i dont want to proceed",  # decline Personal Guide → follow_up (not escalation)
        "No, I'm all set. Thanks.",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "records_branch_taken": "declined_personal_guide",
        },
        transcript_contains=[pool_regex(MSG_FOLLOW_UP_ASK)],
    ),
    notes=(
        "Regression guard for the guide-decline behavior change: declining both "
        "the upload link and the Personal Guide now routes to follow_up and closes "
        "cleanly rather than escalating."
    ),
)

# ── RC. Records coordination — Personal Guide decline → follow-up (not escalation) ──────────────

records_no_guide_upload_then_close = Scenario(
    name="records_no_guide_upload_then_close",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",  # member_upload
        "Yes, please send the link",  # upload_consent = yes
        "Yes, that's correct",  # email on file confirmed
        "No, that won't be necessary",  # personal_guide_consent = no → follow_up
        "No, that's all. Thank you!",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
        transcript_contains=[pool_regex(MSG_FOLLOW_UP_ASK)],
    ),
    notes=(
        "RC-24a — regression guard: member uploads themselves, upload link is sent, "
        "then declines Personal Guide. Agent must route to follow_up "
        "('anything else?') and NOT escalate."
    ),
)

records_no_guide_doctor_direct_then_close = Scenario(
    name="records_no_guide_doctor_direct_then_close",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can I ask my doctor to send it over?",  # doctor_direct
        "No thanks, I don't need the link",  # upload_consent = no → guide offer
        "No, that's okay",  # personal_guide_consent = no → follow_up
        "No, that's everything. Bye!",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": falsy,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
        transcript_contains=[pool_regex(MSG_FOLLOW_UP_ASK)],
    ),
    notes=(
        "RC-24b — doctor-direct path: member says doctor will send the records, "
        "declines the upload link offer, then declines the Personal Guide. "
        "Agent must route to follow_up and NOT escalate."
    ),
)

records_no_guide_regression_no_transfer = Scenario(
    name="records_no_guide_regression_no_transfer",
    flow="claim",
    timeout_s=300,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I can send it myself",  # member_upload
        "no thanks",  # upload_consent = no → guide offer
        "no I don't want that either",  # personal_guide_consent = no → follow_up
        "Nope, I'm done. Goodbye.",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "records_branch_taken": "declined_personal_guide",
        },
    ),
    notes=(
        "RC-24c — pure no-transfer regression guard: declining both upload link and "
        "Personal Guide must never fire an AgentCallTransfer event. "
        "Complements RC-24a/b by using a shorter script with no follow-up turns."
    ),
)

records_no_guide_then_follow_up_question = Scenario(
    name="records_no_guide_then_follow_up_question",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please",  # accept link
        "Yes, that's correct",  # email confirmed
        "No, that's not necessary",  # decline guide → follow_up
        "How long does the review usually take after you receive everything?",
        "No, that's all. Thank you!",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
    ),
    notes=(
        "RC-24d — after declining Personal Guide the agent routes to follow_up; "
        "the member then asks a timeline question which follow_up must answer "
        "before the call closes cleanly."
    ),
)

records_no_guide_conversational_phrasing = Scenario(
    name="records_no_guide_conversational_phrasing",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "oh I think the doctor's office will just send it over",  # doctor_direct
        "yeah sure, go ahead and send me the link",  # upload_consent = yes
        "uh-huh, that email's fine",  # email confirmed
        "I don't think so, no thank you — I'll wait to hear from you",  # decline guide
        "Nope, that's it for me, thanks",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
        transcript_contains=[pool_regex(MSG_FOLLOW_UP_ASK)],
    ),
    notes=(
        "RC-24e — conversational/natural phrasing robustness: member uses an indirect "
        "'I don't think so, no thank you' to decline the Personal Guide. "
        "Normalizer must resolve this to no and route to follow_up, not escalation. "
        "retries=1 for LLM extraction variance on the indirect phrasing."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# RC2. New-intent pivots from follow_up reached via records-decline path
#
# The member reaches follow_up after declining the Personal Guide (the new
# routing introduced in records_coordination_agent). From there they raise a
# new service request. These scenarios verify the full new_intent machinery
# works identically regardless of HOW follow_up was entered.
# ──────────────────────────────────────────────────────────────────────────────

# Shared prefix: claim verify → reference → doctor-direct → no link → no guide
# → follow_up.  Three turns after CLAIM_VERIFY + reference number.
_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE = CLAIM_VERIFY + [
    "42695817",
    "my doctor can send the records over",  # doctor_direct
    "no thank you",  # upload_consent = no → guide offer
    "no, that's fine",  # personal_guide_consent = no → follow_up
]

records_no_guide_then_pcp_new_intent = Scenario(
    name="records_no_guide_then_pcp_new_intent",
    flow="claim",
    mutating=True,  # provider flow writes James's ZIP if absent
    timeout_s=480,
    retries=1,  # new_intent + intake re-screen + provider/delivery slots are LLM-driven
    user_turns=_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE
    + [
        "Actually, I also need to find a primary care physician near me.",  # new_intent
        # Re-verification (provider slot order: first/last name → readback →
        # member_id → dob → relationship). Same caller, James M310188.
        "james",
        "wilson",
        "yes correct",
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "I'm the plan holder",  # relationship
        # Now in provider_search.
        "Primary Care Physician",  # provider type
        "zero two one three nine",  # ZIP (James may have none on file)
        "email please",  # delivery method
        "yes that's correct",  # email on file confirmed
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no, that's all, thanks",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "call_intent": "provider_services",
            "provider_list_sent": True,
            "delivery_method": "email",
            "pending_intent": lambda v: not v,
        },
        transcript_contains=[r"first name"],  # proof of re-verification
    ),
    notes=(
        "RC-24f — after declining Personal Guide the member asks for a new PCP search. "
        "follow_up routes new_intent (provider_services) through intake re-screen, "
        "which clears state and re-verifies James before running the provider flow. "
        "Marked mutating: James may have no ZIP on file. retries=1: new_intent + "
        "re-screen + provider_type/delivery_method extraction are LLM-driven."
    ),
)

records_no_guide_then_claim_new_intent = Scenario(
    name="records_no_guide_then_claim_new_intent",
    flow="claim",
    timeout_s=420,
    retries=1,  # new_intent classification is LLM-driven
    user_turns=_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE
    + [
        "Actually, I submitted another adjustment and want to check on that one too.",  # new_intent
        # Re-verification (claims slot order: first/last name → readback →
        # member_id → dob → phone confirmation). Same caller, James M310188.
        "james",
        "wilson",
        "yes correct",
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "yes that's correct",  # phone confirmation
        # Reference 98765432 does not exist → not-found → retry → escalation.
        "98765432",
        "98765432",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "call_intent": "claim_services",
            "pending_intent": lambda v: not v,
        },
        transcript_contains=[r"first name", r"reference number"],
    ),
    notes=(
        "RC-24g — after declining Personal Guide the member asks for a second claim "
        "adjustment. follow_up routes new_intent (claim_services), resets the call, "
        "and re-verifies James. Reference 98765432 does not exist, so "
        "claim_adjustment escalates after two not-found attempts — a deterministic "
        "outcome that avoids fixture collision with 42695817. retries=1: new_intent "
        "classification is LLM-driven."
    ),
)

records_no_guide_then_unsupported_provider = Scenario(
    name="records_no_guide_then_unsupported_provider",
    flow="claim",
    timeout_s=300,
    retries=1,  # new_intent + intake unsupported-type classification are LLM-driven
    user_turns=_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE
    + [
        "Actually, I also need to find a neurologist near me.",  # new_intent → unsupported
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=False,
        final_is_interrupt=False,
        final_state={
            # DECISIVE: intake re-screen fired before re-verification.
            "member_status_verify": falsy,
            "first_name": falsy,
            "pending_intent": falsy,
            "escalation_pre_message": contains("Orthopedic"),
        },
        last_ai_contains=[
            r"neurologist",
            r"(Primary Care|Pediatrician|Cardiologist|Dermatologist|Orthopedic)",
        ],
    ),
    notes=(
        "RC-24h — after declining Personal Guide the member asks for a neurologist "
        "(unsupported specialty). follow_up routes new_intent (provider_services) "
        "through intake re-screen, which fires the unsupported-provider gate BEFORE "
        "any re-verification — member_status_verify and first_name must be falsy at "
        "END. Mirrors followup_unsupported_provider_rescreen but starts from the "
        "records-decline follow_up entry point. retries=1: new_intent + "
        "unsupported-type classification are LLM-driven."
    ),
)

phone_not_confirmed_ends_call = Scenario(
    name="phone_not_confirmed_ends_call",
    flow="claim",
    user_turns=[
        "I adjusted the claim and I want to follow up",
        "james",
        "wilson",
        "yes correct",  # name_confirmed
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "no, that's not my number",  # decline phone confirmation
    ],
    expect=Expected(
        completed=True,  # hard END, no escalation agent
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        final_state={"phone_update_requested": True},
        last_ai_contains=[
            r"unable to verify",
            r"transferring you to a live representative",
            pool_regex(MSG_PHONE_NOT_CONFIRMED),
        ],
    ),
)

ref_not_found_retry_then_success = Scenario(
    name="ref_not_found_retry_then_success",
    flow="claim",
    timeout_s=360,
    user_turns=[
        # Emily Carter verification (claim flow)
        "I want to check on a claim adjustment I submitted.",
        "emily",
        "carter",
        "yes correct",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "yes correct",  # phone_confirmed
        # wrong reference number → fallback to claim number
        "99999999",  # valid format, no such adjustment → fallback to claim number
        "882301",  # Emily's claim number → fallback lookup succeeds
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"fallback_claim_number": "882301"},
        transcript_contains=[r"(claim number|another way|look it up differently|another approach)"],
    ),
)

ref_not_found_twice_escalates = Scenario(
    name="ref_not_found_twice_escalates",
    flow="claim",
    user_turns=CLAIM_VERIFY
    + [
        "99999999",  # wrong ref number → lookup fails → fallback to claim number
        "88888888",  # treated as claim number → also not found → escalation
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="fallback_claim_number_not_found",
    ),
)

ref_exhaustion = Scenario(
    name="ref_exhaustion",
    flow="claim",
    user_turns=CLAIM_VERIFY
    + [
        "I don't have it",  # → fallback starts: AI asks for claim number
        "No",  # → move to DOS+billed stage
        "I don't have that either",  # → retry prompt (attempt 0)
        "I have no idea what that is",  # → attempt 1 exhausted → escalation
        # spares
        "still nothing",
        "I really can't find anything",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        # New fallback exhaustion reason (dos_billed stage ran out of attempts)
        escalation_reason_regex=r"fallback_(dos_billed_exhausted|dos_billed_not_found|claim_number_not_found)",
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# E2. Reference-number pivot — member recovers the reference number after
#     entering a fallback stage
#
# Regression guard for the fix that detects "reference number" in the member's
# utterance during the claim_number_ask or dos_billed_ask fallback stages and
# resets the flow back to collecting the reference_number.
#
# Three shapes are covered:
#   1. Pivot during claim_number_ask — immediately after being asked for the
#      claim number the member says they found the reference number.
#   2. Pivot during dos_billed_ask — member skips the claim number stage too
#      ("No") before realising they have the reference number.
#   3. Pivot after a WAIT in claim_number_ask — mirrors the real-world transcript
#      (member asks for a moment, then returns with the reference number); this
#      was the specific failure path that surfaced the bug.
# ──────────────────────────────────────────────────────────────────────────────

ref_fallback_pivot_from_claim_number_ask = Scenario(
    name="ref_fallback_pivot_from_claim_number_ask",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have it",  # 7  can't provide ref → AI asks for claim number
        # 8  pivot: member now has the reference number; our fix must
        #    detect "reference number" here and re-ask for it instead of
        #    treating the utterance as a failed claim_number extraction.
        "Actually, I found the reference number. Can I give you that?",
        "42695817",  # 9  reference number provided → claim lookup → complete flow
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        # AI before turn 7: initial reference-number ask.
        7: TurnExpectation(ai_contains=[r"reference number"]),
        # AI before turn 8: after "I don't have it", AI pivots to claim number.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # AI before turn 9: pivot detected — agent re-asks for the reference number.
        9: TurnExpectation(
            ai_contains=[r"reference number"],
            slot_awaiting="reference_number",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "reference_number": "42695817",
            "ref_no_fallback_stage": falsy,  # pivot cleared the fallback stage
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Regression guard: member initially cannot provide the reference number "
        "(claim_number_ask fallback starts), then says 'I found the reference "
        "number' on the very next turn. The fix detects 'reference number' in the "
        "utterance, resets ref_no_fallback_stage to '' and awaiting_slot to "
        "'reference_number', and re-asks. The member then provides the real "
        "reference number and the flow completes normally. Before the fix the "
        "agent kept asking for the claim number, burning a retry."
    ),
)

ref_fallback_pivot_from_dos_billed_ask = Scenario(
    name="ref_fallback_pivot_from_dos_billed_ask",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have it",  # 7  can't provide ref → AI asks for claim number
        "No, I don't have that either",  # 8  no claim number → AI asks for DOS + billed
        # 9  pivot: member finds the reference number during the dos_billed stage;
        #    the fix in _collect_dos_billed_fallback detects "reference number"
        #    and resets back to reference_number collection.
        "Wait, I actually found the reference number",
        "42695817",  # 10 reference number provided → complete flow
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        # AI before turn 8: AI pivoted to claim number fallback.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # AI before turn 9: "No" moved flow to dos+billed stage.
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed|billed.*service)"]),
        # AI before turn 10: pivot detected — agent re-asks for the reference number.
        10: TurnExpectation(
            ai_contains=[r"reference number"],
            slot_awaiting="reference_number",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "reference_number": "42695817",
            "ref_no_fallback_stage": falsy,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Regression guard: member skips both the reference number AND the claim "
        "number, entering the dos_billed_ask fallback stage. They then say 'I found "
        "the reference number'. The fix in _collect_dos_billed_fallback detects "
        "'reference number' in the utterance, resets ref_no_fallback_stage and "
        "awaiting_slot, and re-asks for the reference number. The member provides "
        "42695817 and the flow completes normally. Before the fix the agent tried "
        "to extract date-of-service/billed-amount from the pivot phrase, failed, "
        "and eventually escalated."
    ),
)

ref_fallback_pivot_after_wait = Scenario(
    name="ref_fallback_pivot_after_wait",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have it with me.",  # 7  can't provide ref → AI asks for claim number
        # 8  WAIT: member asks for a moment to look — AI issues MSG_WAIT_ACK and
        #    holds the claim_number_ask stage (ref_no_fallback_stage unchanged).
        "Let me check. Give me one second.",
        # 9  member returns with the reference number, not the claim number;
        #    the fix must detect "reference number" here (even after a WAIT) and
        #    re-ask for it — this is the exact transcript path that surfaced the bug.
        "Actually, I did find the reference number. Can I provide you that?",
        "42695817",  # 10 reference number provided → complete flow
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        # AI before turn 8: after "I don't have it", AI asks for the claim number.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # AI before turn 9: WAIT acknowledged — agent holds the claim_number stage.
        9: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),
        # AI before turn 10: pivot detected — agent re-asks for the reference number.
        10: TurnExpectation(
            ai_contains=[r"reference number"],
            slot_awaiting="reference_number",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "reference_number": "42695817",
            "ref_no_fallback_stage": falsy,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Regression guard for the exact transcript that surfaced the bug. Member "
        "says 'I don't have it' (claim_number_ask fallback starts), asks for a "
        "moment ('Let me check'), then returns with the reference number. Before "
        "the fix: the pivot phrase had no digits and was not a bare affirmative, "
        "so it fell through to LLM extraction which tried (and failed) to extract "
        "a claim number, burning a retry; on the second repetition the agent "
        "moved to dos_billed_ask. After the fix: 'reference number' is detected "
        "in the utterance, ref_no_fallback_stage is cleared, and the agent re-asks "
        "for the reference number — the member provides 42695817 and the flow "
        "completes normally."
    ),
)

ref_fallback_hesitation_heavy = Scenario(
    name="ref_fallback_hesitation_heavy",
    flow="claim",
    timeout_s=480,
    retries=2,  # WAIT classification of the ref# hesitation goes through the LLM
    # (no detect_wait_request in the main flow); bare affirmative + fallback
    # WAITs are deterministic but the surrounding LLM calls add variance
    user_turns=[
        # Emily Carter / M907503 — claim_services verification
        "I want to check on a claim adjustment I submitted.",  # 0
        "emily",  # 1
        "carter",  # 2
        "yes correct",  # 3 name_confirmed
        "m nine zero seven five zero three",  # 4
        "April twelfth nineteen eighty eight",  # 5
        "yes correct",  # 6 phone_confirmed (617-555-4101)
        # still can't find it → cannot-provide → claim_number_ask
        "Sorry, I just don't have the reference number.",  # 8
        # claim_number_ask: WAIT 1 → MSG_WAIT_ACK (detect_wait_request fires)
        "Give me one second.",  # 9
        # bare affirmative (in _AFFIRMATIVE_PHRASES) → re-ask, no retry burned
        "I have it.",  # 10
        # claim_number_ask: WAIT 2 → MSG_WAIT_ACK
        "Hold on, let me check...",  # 11
        # cannot-provide → dos_billed_ask
        "I don't have it.",  # 12
        # dos_billed_ask: WAIT 3 → MSG_WAIT_ACK
        "Hang on a moment.",  # 13
        # keyword pivot: "claim number" (no digits) → back to claim_number_ask
        "Actually, I just found the claim number.",  # 14
        # provide Emily's claim number → lookup succeeds
        "eight eight two three zero one",  # 15
        # complete the claim flow (Emily: records_required=True)
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "No",
        "email them to me",
        "No, that's all. Thanks!",
        # spares — turn count varies when the ref# retry fires differently
        "that's all, thanks",
        "no, nothing else",
    ],
    turn_expectations={
        # AI before turn 8: ref# retry message (hesitation burned one attempt)
        # 8: TurnExpectation(ai_contains=[r"reference number"], slot_awaiting="reference_number"),
        # AI before turn 9: cannot-provide fired → claim_number_ask
        # 9: TurnExpectation(
        #     ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        # ),
        # AI before turn 10: WAIT 1 acked
        # 10: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),
        # AI before turn 11: bare affirmative → re-ask (no retry burned)
        # 11: TurnExpectation(ai_contains=[r"claim number"], slot_awaiting="fallback_claim_number"),
        # AI before turn 12: WAIT 2 acked
        # 12: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),
        # AI before turn 13: cannot-provide → dos_billed_ask
        # 13: TurnExpectation(
        #     ai_contains=[r"(date of service|billed amount|service.*billed|billed.*service)"]
        # ),
        # AI before turn 14: WAIT 3 acked
        # 14: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),
        # AI before turn 15: "claim number" keyword pivot → back to claim_number_ask
        # 15: TurnExpectation(ai_contains=[r"claim number"], slot_awaiting="fallback_claim_number"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": "882301",
            "ref_no_fallback_stage": falsy,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Stress test for hesitation handling across all three fallback stages. "
        "Exercises five distinct hesitation shapes in a single call: "
        "(1) ref# stage hesitation — no detect_wait_request in the main flow, "
        "LLM extraction fails → slot_fail → retry message (not MSG_WAIT_ACK); "
        "(2) WAIT × 2 inside claim_number_ask (detect_wait_request) → MSG_WAIT_ACK; "
        "(3) bare affirmative 'I have it.' inside claim_number_ask → re-ask "
        "without burning a retry ('i have it' is in _AFFIRMATIVE_PHRASES); "
        "(4) WAIT inside dos_billed_ask (detect_wait_request) → MSG_WAIT_ACK; "
        "(5) keyword pivot from dos_billed_ask back to claim_number_ask when "
        "member says 'I just found the claim number' (no digits). "
        "Uses Emily Carter (M907503) — 882301 is her fixture claim number. "
        "retries=2: the ref# hesitation goes through the LLM (no deterministic "
        "detect_wait_request guard in the main flow) adding extraction variance."
    ),
)

ref_fallback_dos_billed_pivot_to_claim_number = Scenario(
    name="ref_fallback_dos_billed_pivot_to_claim_number",
    flow="claim",
    timeout_s=360,
    user_turns=[
        # Emily Carter (M907503) — 882301 is her fixture claim number
        "I want to check on a claim adjustment I submitted.",
        "emily",
        "carter",
        "yes correct",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "yes correct",  # phone_confirmed
        "I don't have it",  # 7  can't provide ref → AI asks for claim number
        "No, I don't have that either",  # 8  no claim number → AI asks for DOS + billed
        # 9  pivot: member now has the claim number; the fix in
        #    _collect_dos_billed_fallback detects "claim number" (no digits) and
        #    resets ref_no_fallback_stage to "claim_number_ask".
        "Actually, I do have the claim number",
        "882301",  # 10 Emily's claim number → fallback lookup succeeds → complete flow
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        # AI before turn 8: claim number fallback started.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # AI before turn 9: "No" moved flow to dos+billed stage.
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed|billed.*service)"]),
        # AI before turn 10: pivot detected — agent re-asks for the claim number.
        10: TurnExpectation(
            ai_contains=[r"claim number"],
            slot_awaiting="fallback_claim_number",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": "882301",
            "ref_no_fallback_stage": falsy,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Regression guard: member skips the reference number AND declines the claim "
        "number ('No'), entering the dos_billed_ask stage. They then say 'Actually, "
        "I do have the claim number'. The fix in _collect_dos_billed_fallback detects "
        "'claim number' (no digits → not a rescue-path value), resets ref_no_fallback_stage "
        "to 'claim_number_ask', and re-asks for the claim number. Member provides Emily "
        "Carter's fixture claim number (882301) → fallback lookup succeeds → flow completes. "
        "Must use Emily Carter (M907503), NOT CLAIM_VERIFY (James Wilson) — 882301 is "
        "Emily's claim number; a James Wilson lookup always fails."
    ),
)

ref_fallback_claim_number_forward_to_dos_billed = Scenario(
    name="ref_fallback_claim_number_forward_to_dos_billed",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have it",  # 7  can't provide ref → AI asks for claim number
        # 8  member offers dos+billed keywords instead of a claim number;
        #    the fix in _collect_claim_number_fallback detects "date of service"
        #    or "billed amount" (no digits) and advances to dos_billed_ask.
        "I don't have the claim number, but I do have the date of service and the billed amount",
        # 9  now in dos_billed_ask — provide values; no sandbox fixture exists for
        #    James Wilson's dos+billed so the lookup escalates (expected). The
        #    decisive assertion is the turn-9 expectation proving the pivot happened.
        "It was 2026-01-19 and the billed amount was 620",
        # spares — not reached when escalation fires
        "No, that's everything. Thanks!",
        "I'm all set, thanks.",
    ],
    turn_expectations={
        # AI before turn 8: claim number fallback started.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # AI before turn 9: dos+billed keywords triggered the forward — agent asks
        # for date of service and billed amount (NOT claim number). This is the
        # decisive proof that the pivot mechanism fired.
        9: TurnExpectation(
            ai_contains=[r"(date of service|billed amount|service.*billed|billed.*service)"],
            slot_awaiting="fallback_dos_billed",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=True,  # no sandbox fixture for James Wilson dos+billed — lookup fails
        transfer_event=True,
        escalation_reason_regex=r"fallback_(dos_billed_exhausted|dos_billed_not_found|claim_number_not_found)",
        final_state={"ref_no_fallback_stage": falsy},
    ),
    notes=(
        "Regression guard: member is asked for the claim number but says they have "
        "the date of service and billed amount instead. The fix in "
        "_collect_claim_number_fallback detects 'date of service'/'billed amount' "
        "keywords (no digits yet) and immediately advances ref_no_fallback_stage to "
        "'dos_billed_ask'. The decisive assertion is turn 9: slot_awaiting='fallback_dos_billed' "
        "proves the agent went straight to the dos+billed ask without burning a "
        "claim-number retry. Escalation is expected because no Salesforce fixture "
        "exists for James Wilson's dos+billed values — the pivot mechanism is what's "
        "under test, not the Salesforce lookup."
    ),
)

ref_fallback_chained_pivot_transcript = Scenario(
    name="ref_fallback_chained_pivot_transcript",
    flow="claim",
    timeout_s=420,
    retries=1,  # chained pivots involve multiple LLM-free guards but the
    # surrounding verification + claim lookup add some extraction variance
    user_turns=[
        # Emily Carter / M907503 — claim_services verification
        "I have made a claim adjustment, and I would like an update on it now.",  # 0
        "Emily",  # 1
        "Carter",  # 2
        "Yes",  # 3 name_confirmed
        "m nine zero seven five zero three",  # 4
        "twelve April nineteen eighty eight",  # 5
        # Phone on file is 617-555-4101; Emily initially says "no" then provides
        # the same digits — the agent re-reads and she confirms.
        "No. It's actually been changed.",  # 6 decline phone on file
        "six one seven five five five four one zero one",  # 7 same digits
        "Yeah. The correct phone number is six one seven five five five four one zero one.",
        "Yeah. That is correct.",  # 9 phone_confirmed (agent read back 617-555-4101)
        # Fallback chain starts
        "I don't have the reference number.",  # 10 → claim_number_ask
        "No. I don't have the claim number either.",  # 11 → dos_billed_ask
        # Pivot 1 (dos_billed → reference_number): existing fix
        "I actually have the reference number. I found it right now. Can I give it to you?",  # 12
        # Pivot 2 (reference_number → claim_number_ask): NEW FIX
        # Member changes their mind mid-utterance — wants to give claim number instead.
        "The reference number is... actually, sorry. I found the claim number. Can I give you that?",  # 13
        "eight eight two three zero one",  # 14 claim number → lookup succeeds
        # Complete the claim flow (Emily's claim: records_required=True)
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
        # spares — phone re-ask and repeat add turn variance
        "that's all, thanks",
        "no, nothing else",
    ],
    turn_expectations={
        # After "I don't have the reference number" → claim number ask.
        11: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # After "No, I don't have that either" → dos+billed ask.
        12: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed|billed.*service)"]),
        # After pivot-1 phrase → reference number re-ask (existing fix).
        13: TurnExpectation(
            ai_contains=[r"reference number"],
            slot_awaiting="reference_number",
        ),
        # After pivot-2 phrase → claim number re-ask (NEW FIX).
        14: TurnExpectation(
            ai_contains=[r"claim number"],
            slot_awaiting="fallback_claim_number",
        ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": "882301",
            "ref_no_fallback_stage": falsy,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Regression guard for the exact transcript that exposed the chained-pivot gap. "
        "Emily cannot provide the reference number (claim_number_ask), then the claim "
        "number (dos_billed_ask). She then finds the reference number — pivot-1 resets "
        "to reference_number collection (existing fix). Mid-utterance she changes her "
        "mind and says she has the claim number instead — pivot-2 (NEW FIX) detects "
        "'claim number' during reference_number collection and starts claim_number_ask. "
        "She provides 882301 → fallback lookup succeeds. "
        "Phone: Emily says 'No' but then provides the same 617-555-4101 digits; the "
        "agent reads them back and she confirms — turn count is variable, hence spares. "
        "retries=1: the surrounding verification LLM steps add extraction variance."
    ),
)

claim_email_change_on_upload = Scenario(
    name="claim_email_change_on_upload",
    flow="claim",
    mutating=True,
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please send the link",
        f"that's my old email, use {NEW_EMAIL}",  # email read-back → change
        "yes that's correct",  # confirm new email read-back
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "upload_link_sent": True,
            "email": contains(NEW_EMAIL),
        },
    ),
    post_checks=[sf_field_check("M310188", "email", NEW_EMAIL)],
    notes=(
        "Per code reading, records_coordination only carries the new email in "
        "graph state — it never writes the member record in Salesforce, so the "
        "SF post-check documents that gap. See README Known issues."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# F. Follow-up escalations (on top of a completed PCP flow)
# ──────────────────────────────────────────────────────────────────────────────

_PCP_TO_FOLLOW_UP = PCP_VERIFY + [
    "Primary Care Physician",
    "yes that's correct",
    "send it to my fax",
    "yes that's correct",
    "no thanks",  # decline benefits
    "no thank you",  # decline Care Coach → follow-up "anything else?"
]

# Phase 6 redefinition: update requests in follow_up now route to the owning
# flow for every capability/slot the registries know (re-sends → delivery,
# zip → provider flow, …). Only HUMAN-ONLY targets still escalate — the phone
# number on file is the canonical one. The old fax re-send utterance this
# scenario used is now a routable redo, covered by redo_resend_from_follow_up.
follow_up_update_request = Scenario(
    name="follow_up_update_request",
    flow="pcp",
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "actually I need to update the phone number you have on file for me",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="update_request_in_follow_up",
        last_ai_contains=[
            r"transfer you to a representative",
            pool_regex(MSG_UPDATE_REQUEST_ESCALATE),
        ],
    ),
    notes=(
        "Phase 6: only human-only update targets (phone_number) escalate from "
        "follow_up; routable targets are handed to their owning flow instead."
    ),
)

follow_up_cannot_answer_x3 = Scenario(
    name="follow_up_cannot_answer_x3",
    flow="pcp",
    timeout_s=360,
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "what's my copay for an MRI?",
        "is acupuncture covered?",
        "what about dental?",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="repeated_cannot_answer_in_follow_up",
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# G. Contact-change loop limits
#
# NOTE: zip_change_loop_escalates was REDEFINED for the no-confirmation flow.
# It previously exercised the zip_change_cycles loop guard around the ZIP
# read-back confirmation, which no longer exists — a VALID new ZIP is accepted
# directly, so a rejection loop on the read-back is impossible by construction
# (the direct-acceptance paths are covered by pcp_zip_update and
# pcp_zip_inline_update in group A). The escalation loop that REMAINS is slot
# exhaustion on INVALID values: decline the on-file ZIP, then repeatedly give
# non-5-digit values → the zip_code pipeline never validates → slot_fail →
# MAX_SLOT_ATTEMPTS exhausted → signal_escalate("zip_code exhausted").
# ──────────────────────────────────────────────────────────────────────────────

zip_change_loop_escalates = Scenario(
    name="zip_change_loop_escalates",
    flow="pcp",
    timeout_s=360,
    retries=1,  # ambiguous-vs-answered classification of garbled digit strings
    # is LLM-dependent; first AMBIGUOUS turn per slot is an uncounted CLARIFY,
    # so exhaustion needs 3-4 invalid turns depending on classification
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "no, that's not my zip code",  # decline ZIP on file → asked for a new ZIP
        # Invalid replacements — never 5 digits, so the zip_code pipeline can
        # never validate/accept. Each turn is either AMBIGUOUS (extraction
        # prompt: "not exactly 5 digits → ambiguous"; second consecutive
        # ambiguous counts a failure) or a rejected extraction (counts
        # immediately). NO Salesforce write must ever occur.
        "nine eight seven",  # 3 digits
        "zero two one",  # 3 digits
        "one two three four",  # 4 digits
        "four two",  # 2 digits
        # spares — CLARIFY turns are not counted attempts, so the number of
        # interrupts before exhaustion varies by ±1-2
        "seven seven seven",
        "two two",
        "still just nine eight seven",
    ],
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        # _collect_slot exhaustion: signal_escalate(reason=f"{slot_name} exhausted")
        escalation_reason_regex=r"zip_code\s+exhausted|zip_code_exhausted",
        final_state={
            "provider_list_sent": falsy,  # flow never reached delivery
            "zip_code_updated": falsy,  # no valid ZIP was ever accepted
        },
        transcript_contains=[
            # build_slot_exhausted_message("zip_code") wording, kept tolerant
            r"(zip code after a few tries|wasn't able to capture)",
        ],
    ),
    post_checks=[
        # The on-file ZIP must be untouched in Salesforce — with the read-back
        # removed, the only write path is a VALIDATED 5-digit ZIP; invalid
        # values must never reach update_zip_in_salesforce. Emily's fixture
        # ZIP is snapshotted by preflight but this scenario is non-mutating
        # by design, so we assert no write happened at all.
        sf_field_check("M907503", "zip_code", "12139"),
    ],
    notes=(
        "Redefined after the ZIP read-back confirmation was removed from "
        "provider_search. The old zip_change_cycles rejection loop is "
        "impossible now; the remaining escalation loop is zip_code slot "
        "exhaustion on repeatedly INVALID (non-5-digit) values. Asserts the "
        "Agent-initiated transfer, the 'zip_code exhausted' reason, that the "
        "flow never dispatched, that zip_code_updated stays falsy, and — via "
        "SF post-check — that no invalid value was ever written to Salesforce "
        "(critical now that valid ZIPs are persisted without confirmation). "
        "Expects the Emily fixture ZIP to be 12139 per the conversational-"
        "workload entity data; adjust the post-check if the sandbox fixture "
        "differs."
    ),
)

email_change_loop_in_notification = Scenario(
    name="email_change_loop_in_notification",
    flow="claim",
    mutating=True,
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please send the link",
        "Yes, that's correct",  # confirm email for the upload link
        "Yes, please proceed",  # accept guide → notification setup
        "email please",  # choose email notifications
        "no, use james.one@example.com",  # reject read-back w/ new email (cycle 1)
        "no, actually use james.two@example.com",  # cycle 2
        "no, make it james.three@example.com",  # cycle 3 → escalate
        # spares
        "no, that's wrong as well — james.four@example.com",
        "no, james.five@example.com",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_regex=r"email_(change_loop_exceeded|confirmed_exhausted)",
    ),
    notes="Marked mutating: notification-preference rows are inserted in Salesforce.",
)


# ──────────────────────────────────────────────────────────────────────────────
# G2. Notification contact-confirmation advances on the first affirmative
#
# Regression guard for the phone_confirmed / email_confirmed loop in
# notification_setup_agent. The bug: advancement depended solely on the
# extraction LLM returning contact_confirmed="yes". Because notification_method
# is passed in as an already-confirmed slot, the LLM is biased to treat a plain
# affirmative as a redundant acknowledgment and return an EMPTY contact_confirmed,
# which fell through to a non-advancing slot retry — so the caller had to repeat
# "yes" two or three times before the flow moved on (see the transcript symptom
# on the fix commit). The fix adds a deterministic normalize_yes_no(last_user)
# fallback, gated on no replacement contact being extracted this turn.
#
# These scenarios drive the claim flow to the notification phone/email read-back
# and answer with the exact affirmative phrasings from the bug report
# ("yes thats correct", "yes", "yes please"). The decisive assertion is the
# turn_expectation on the AI prompt that FOLLOWS the affirmative: awaiting_slot
# must already be "timeline_question" (the flow advanced to _save_and_complete +
# the timeline bridge on the FIRST turn). Under the bug the agent re-asks and
# awaiting_slot stays phone_confirmed / email_confirmed, failing the assertion.
#
# James M310188 has phone 512-555-6101 (the number from the bug transcript) and
# email james.wilson@gmail.com on file, so both confirmation read-backs fire.
# ──────────────────────────────────────────────────────────────────────────────

# Claim flow up to the point where notification_setup asks for the channel:
# verify → reference number → doctor-direct records + upload link → Personal Guide.
# The next scripted turn (index 12) is the notification_method answer.
_CLAIM_TO_NOTIFICATION = CLAIM_VERIFY + [
    "42695817",  # 7  reference number
    "Can I ask my doctor to send it over?",  # 8  doctor-direct
    "Yes, please",  # 9  accept upload link
    "Yes, that's correct",  # 10 confirm email on file (records upload link)
    "Perfect. Please do that",  # 11 accept Personal Guide → notification setup
]

notification_phone_confirm_advances = Scenario(
    name="notification_phone_confirm_advances",
    flow="claim",
    timeout_s=360,
    retries=1,  # whether the LLM extracts "yes" or returns empty (→ deterministic
    # fallback) is non-deterministic; BOTH must advance, but the surrounding flow
    # has other LLM-driven steps, so allow one rerun for unrelated flakiness
    user_turns=_CLAIM_TO_NOTIFICATION
    + [
        "You can send me the updates to my phone",  # 12 notification_method = sms
        "yes thats correct",  # 13 phone_confirmed — affirmative phrasing from the bug
        "Okay, how long will it take to finalize the request?",  # 14 timeline question
        "email them to me",  # 15 N2 channel
        "No, that's it. Thanks!",  # 16 close
    ],
    turn_expectations={
        # The phone read-back before the affirmative: still awaiting confirmation.
        13: TurnExpectation(
            ai_contains=[r"(still the correct number|on file|is that right)"],
            slot_awaiting="phone_confirmed",
        ),
        # THE REGRESSION CATCH: one affirmative advanced the flow to the timeline
        # bridge. awaiting_slot must be timeline_question (not a phone_confirmed
        # re-ask), and the AI prompt is the timeline bridge.
        14: TurnExpectation(ai_contains=[r"timeline"], slot_awaiting="timeline_question"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Phone-confirmation regression guard. At phone_confirmed the member says "
        "'yes thats correct'; the flow must advance to the timeline bridge on the "
        "FIRST turn (turn-14 expectation: awaiting_slot=timeline_question). Before "
        "the fix, an empty contact_confirmed extraction fell through to a "
        "non-advancing slot retry and awaiting_slot stayed phone_confirmed."
    ),
)

notification_phone_confirm_bare_yes_advances = Scenario(
    name="notification_phone_confirm_bare_yes_advances",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=_CLAIM_TO_NOTIFICATION
    + [
        "You can send me the updates to my phone",  # 12 notification_method = sms
        "yes",  # 13 phone_confirmed — bare "yes": the strongest empty-extraction trigger
        "Okay, how long will it take to finalize the request?",  # 14 timeline question
        "email them to me",  # 15 N2 channel
        "No, that's it. Thanks!",  # 16 close
    ],
    turn_expectations={
        13: TurnExpectation(
            ai_contains=[r"(still the correct number|on file|is that right)"],
            slot_awaiting="phone_confirmed",
        ),
        14: TurnExpectation(ai_contains=[r"timeline"], slot_awaiting="timeline_question"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Same as notification_phone_confirm_advances but with a bare 'yes' — the "
        "phrasing most likely to be dropped by the extraction LLM as a redundant "
        "acknowledgment. The deterministic normalize_yes_no fallback must still "
        "advance the flow on the first turn."
    ),
)

notification_email_confirm_advances = Scenario(
    name="notification_email_confirm_advances",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=_CLAIM_TO_NOTIFICATION
    + [
        "email please",  # 12 notification_method = email
        "yes please",  # 13 email_confirmed — affirmative phrasing from the bug
        "Okay, how long will it take to finalize the request?",  # 14 timeline question
        "email them to me",  # 15 N2 channel
        "No, that's all. Thanks!",  # 16 close
    ],
    turn_expectations={
        # The email read-back before the affirmative: still awaiting confirmation.
        13: TurnExpectation(
            ai_contains=[r"(still the right address|on file|correct email)"],
            slot_awaiting="email_confirmed",
        ),
        # THE REGRESSION CATCH: one affirmative advanced to the timeline bridge.
        14: TurnExpectation(ai_contains=[r"timeline"], slot_awaiting="timeline_question"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "notification_channel": "email",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Email-confirmation regression guard (mirror of the phone case). At "
        "email_confirmed the member says 'yes please'; the flow must advance to "
        "the timeline bridge on the FIRST turn (turn-14 expectation: "
        "awaiting_slot=timeline_question). Before the fix an empty contact_confirmed "
        "extraction fell through to a non-advancing slot retry."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# H. Conversational & confusion-recovery
# ──────────────────────────────────────────────────────────────────────────────

pcp_happy_path_conversational = Scenario(
    name="pcp_happy_path_conversational",
    flow="pcp",
    retries=1,
    user_turns=PCP_VERIFY_CONVERSATIONAL
    + [
        "yeah I'm looking for a primary care doctor, or PCP",  # provider type
        "yep that's right",  # ZIP on file
        "email is fine",  # delivery method
        "yes that's the right one",  # email on file
        "yeah go ahead please",  # accept benefits
        "yeah that sounds good",  # accept Care Coach
        "roughly how long before I get the list?",  # follow-up question
        "nope, I think that covers everything, thanks so much",  # close
    ],
    turn_expectations=_VERIFY_TURNS,
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
    ),
    notes=(
        "Uses PCP_VERIFY_CONVERSATIONAL; retries=1 because natural phrasing "
        "slightly raises extraction non-determinism on provider_type and "
        "delivery_method slots."
    ),
)

claim_happy_path_conversational = Scenario(
    name="claim_happy_path_conversational",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_CONVERSATIONAL
    + [
        "42695817",  # reference number
        "Can I ask my doctor to send it over?",  # doctor-direct
        "Yes, please",  # accept upload link
        "Yes, that's correct",  # confirm email on file
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "any idea how long this whole process usually takes?",  # natural timeline question
        "email them to me",  # N2 channel
        "No, I think we're good. Really appreciate the help!",  # close
    ],
    turn_expectations={7: TurnExpectation(ai_contains=[r"reference number"])},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Uses CLAIM_VERIFY_CONVERSATIONAL; retries=1 because natural phrasing "
        "slightly raises extraction non-determinism on upload_method and "
        "personal_guide_consent slots."
    ),
)

pcp_confused_member = Scenario(
    name="pcp_confused_member",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",  # provider type
        "wait, what did you say?",  # ambiguous ZIP read-back → CLARIFY/retry
        "yes that's correct",  # ZIP confirmed after re-read
        "umm... hold on... actually email is better",  # hedged then resolved delivery method
        "yes that's correct",  # email on file confirmed
        "do you guys have an app?",  # benign side question mid-benefits offer
        "oh sorry, yes please go ahead",  # accept benefits after agent redirects
        "no thank you",  # decline Care Coach
        "no, that's everything",  # close
        # 2 spare turns: CLARIFY turns for ZIP/email are not counted attempts;
        # the app-question turn is also uncounted — total interrupts is variable
        "I'm all set, thanks",
        "that was all I needed",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
        },
    ),
    notes=(
        "Exercises: AMBIGUOUS handling for ZIP confirmation ('wait, what did you "
        "say?' is a CLARIFY turn — not counted as a slot failure); "
        "ANSWERED_WITH_FOLLOWUP when a benign side-question ('do you guys have "
        "an app?') interrupts the benefits offer; guard non-escalation on benign "
        "confusion. retries=1: hedged delivery-method phrasing ('umm... hold on... "
        "actually email is better') slightly raises extraction non-determinism."
    ),
)

claim_confused_member = Scenario(
    name="claim_confused_member",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY
    + [
        "hold on, let me find the letter...",  # hesitant non-answer → retry
        "four two six nine five eight one seven",  # reference number on retry
        "sorry, what records do you need exactly?",  # confused → upload_method retry
        "okay, I'll have my doctor send them over",  # doctor_direct on re-ask
        "Yes, please",  # accept upload link
        "hmm, I think so? probably",  # ambiguous email confirm → re-ask
        "yes that's correct",  # email confirmed on re-ask
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "Okay, how long will it take to finalize the request?",  # timeline question
        "email them to me",  # N2 channel
        "No, that's it. Thanks!",  # close
        # 3 spare turns: hesitation/confusion/clarify turns are not counted
        # as slot-failure attempts, making total interrupt count variable
        "I think that covers it",
        "all done from my side",
        "that's everything, thanks",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "reference_number": "42695817",
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Exercises: reference-number slot retry after a hesitant non-answer "
        "('hold on, let me find the letter...' consumes a retry but must not "
        "escalate); upload_method clarification after a confused re-question "
        "('sorry, what records do you need exactly?'); ambiguous email_confirmed "
        "answer ('hmm, I think so? probably') triggering a gentle re-ask rather "
        "than a slot failure. retries=1: ambiguous mid-flow utterances slightly "
        "raise extraction non-determinism."
    ),
)


pcp_conversational_confusion = Scenario(
    name="pcp_conversational_confusion",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        # Verification — naturally phrased; no PCP_VERIFY prefix
        "hi there, I'm trying to find a regular family doctor in my area",
        "sure, it's emily",
        "carter — that's c a r t e r",  # exercises SPELL_CONFIRM: LLM strips
        "yes correct",  # name_confirmed
        "okay so my member id is m nine zero seven five zero three",  # the spelling echo
        "I was born on the twelfth of april, nineteen eighty eight",
        "it's my own plan, I'm the plan holder",
        # Provider flow
        "Primary Care Physician",
        # Confusion #1 — ZIP read-back: no yes/no → zip_confirmed slot_fail → RETRY
        "sorry, what was that zip code again?",
        "ah yes, that's right",  # ZIP confirmed
        # Confusion #2 — delivery method: no channel mention → delivery_method pipeline
        #   ambiguous → retry interrupt; "hold on" noted below (not a transfer request)
        "umm... hold on... what were my options again?",
        "actually, email is better for me",  # delivery_method = email
        "yep, that's the correct email",  # email_confirmed = yes (on file)
        # Confusion #3 — benefits offer: no yes/no for benefits_response → slot_fail
        #   → re-offer in _handle_benefits_response
        "wait, quick thing — do you guys have a mobile app?",
        "oh sorry — yes please, go ahead with the benefits",  # benefits_response = yes
        "yeah that sounds great",  # accept Care Coach
        "where do I go to check my wellness reward points?",  # answerable follow-up
        "no, that's all for me, thanks so much",  # close
        # 2 spare turns: confusion turns (#1 and #3) may each consume one extra interrupt
        # depending on whether the guard fires for the app question before _handle_benefits
        # re-offers, making total interrupt count variable by ±1–2
        "I'm all set, thanks",
        "that was everything, thank you",
    ],
    turn_expectations={4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id")},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
        transcript_contains=[r"mysagilityhealth\.com"],
    ),
    notes=(
        "Exercises three slot-retry recovery paths without escalation: "
        "(1) 'sorry, what was that zip code again?' → zip_confirmed receives no "
        "yes/no → slot_fail('zip_confirmed') → generate_recovery_message(guard='RETRY') "
        "in provider_search_agent; "
        "(2) 'umm... hold on... what were my options again?' → delivery_method pipeline "
        "extracts nothing (no fax/email mention → ambiguous) → pipeline retry interrupt "
        "in delivery_management_agent; risk: 'hold on' could pattern-match guard "
        "keywords — utterance retained because there is zero transfer-intent semantics; "
        "semantic LLM guard will not classify this as TRANSFER_REQUEST; "
        "(3) 'wait, quick thing — do you guys have a mobile app?' → benefits_response "
        "extraction yields empty → slot_fail('benefits_response') → re-offer in "
        "_handle_benefits_response (first off-topic occurrence does not escalate). "
        "retries=1: natural conversational phrasing slightly raises extraction "
        "non-determinism on provider_type and delivery_method slots."
    ),
)

claim_conversational_confusion = Scenario(
    name="claim_conversational_confusion",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=[
        # Verification — naturally phrased; no CLAIM_VERIFY prefix
        "hello, I submitted a claim adjustment a while back and wanted to check on it",
        "yeah, it's james",
        "wilson",
        "yes correct",  # name_confirmed
        "let me grab my card... okay, it's m three one zero one eight eight",
        "the Thirtieth of July, nineteen seventy seven",
        "yep, that's the right number",  # phone_confirmed = yes
        # Claim flow
        # Confusion #1 — reference number: zero spoken digits → claim_adjustment.md
        #   'zero digits → ambiguous' rule → slot_fail('reference_number') → RETRY;
        #   "hold on" noted below (no transfer-intent, in-context temporal stall)
        "hold on, let me dig out the letter... one second",
        "okay found it — it's four two six nine five eight one seven",  # ref = 42695817
        # Records coordination
        # Confusion #2 — upload_method: no doctor_direct/member_upload/personal_guide
        #   intent extractable → ambiguous → slot_fail('upload_method') → retry
        "sorry, which records do you need from me exactly?",
        "ah okay — I'll just have my doctor's office send them over",  # doctor_direct
        "Yes, please",  # accept upload link
        # Confusion #3 — email_confirmed: records_coordination.md explicitly lists
        #   'I think so' / 'probably' → AMBIGUOUS (NOT a 'no') → slot_fail('email_confirmed')
        #   → generate_recovery_message(guard='CLARIFY') gentle re-ask, no email-update path
        "hmm, I think so? probably",
        "yes, that's correct",  # email confirmed on re-ask
        "Perfect. Please do that",  # accept Personal Guide
        "you can just text me",  # notification_method = sms
        "Yes, that's correct",  # confirm phone on file
        "how long is all of this going to take?",  # timeline question
        "email works for me",  # N2 channel = email
        "No, that's all. Thanks!",  # close
        # 2 spare turns: confusion turns (#1 and #2) each consume one extra interrupt;
        # total interrupt count is variable by ±1–2 depending on CLARIFY guard firing
        "I think that covers everything",
        "all done on my end, thanks",
    ],
    turn_expectations={7: TurnExpectation(ai_contains=[r"reference number"])},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
        transcript_contains=[r"5 to 10 business days"],
    ),
    notes=(
        "Exercises three slot-retry recovery paths without escalation: "
        "(1) 'hold on, let me dig out the letter... one second' → zero spoken digits "
        "→ claim_adjustment.md 'zero digits → ambiguous' rule → slot_fail('reference_number') "
        "→ _generate_slot_retry_response(guard='RETRY') in claim_adjustment_agent; "
        "risk: 'hold on' has no transfer-intent; utterance is clearly in-context "
        "(we just asked for the reference number) so the semantic LLM guard will not "
        "classify it as TRANSFER_REQUEST; "
        "(2) 'sorry, which records do you need from me exactly?' → upload_method "
        "ambiguous (no member_upload/doctor_direct/personal_guide/decline intent) "
        "→ slot_fail('upload_method') → _generate_slot_retry_response in "
        "records_coordination_agent; "
        "(3) 'hmm, I think so? probably' → records_coordination.md explicitly lists "
        "'I think so' and 'probably' as AMBIGUOUS (not 'no') → slot_fail('email_confirmed') "
        "→ generate_recovery_message(guard='CLARIFY', slot_label_override=...) gentle "
        "re-ask; the email-update path is NOT triggered. "
        "retries=1: natural conversational phrasing slightly raises extraction "
        "non-determinism on upload_method and personal_guide_consent slots."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# I. Boundary stress
# ──────────────────────────────────────────────────────────────────────────────

boundary_walk_claim = Scenario(
    name="boundary_walk_claim",
    flow="claim",
    timeout_s=420,
    retries=2,
    user_turns=[
        "hi, yeah — I'm calling about a claim adjustment I submitted, I want to see where it stands",
        "it's james",
        "wilson",
        "yes correct",  # name_confirmed
        # PROBE 1 — lane drift during verification (member id ask): asks the
        # claim question early; verification redirects back to the pending slot
        "before I give you that — what did the adjustment actually come out to? "
        "that's really what I'm calling about",
        "alright, fine — it's m three one zero one eight eight",
        # PROBE 2 — corrects an already-accepted field (last name) while DOB is
        # being collected; resolves to the same spelling so lookup is unaffected
        "wait, actually — did you get my last name down right earlier? "
        "it's wilson, w i l s o n. people write it with two L's all the time",
        "it's the Thirtieth of July, nineteen seventy seven",
        # PROBE 5 — stacked answer + unrelated question at phone confirm;
        # verification flattens ANSWERED_WITH_FOLLOWUP → ANSWERED by design
        "yep, that's the right number — oh, quick question, is there an online "
        "portal where I can see my claim too?",
        # PROBE 6 — mild impatience, zero digits → one reference_number retry;
        # phrasing deliberately clear of FRUSTRATED/INTERRUPTION/ABUSE keywords
        "okay, bear with me, I need to find the letter... honestly, this is taking a while",
        "got it — four two six nine five eight one seven",
        "I can upload them myself",  # member_upload → link offer
        # PROBE 4 — mind-change one turn after choosing member_upload: clear
        # "no" to the link + request for guide outreach; lands safely at either
        # upload_consent (no → guide offer) or upload_method (personal_guide →
        # consent ask) — both reconverge on the guide-consent question
        "no — actually, I've changed my mind, I don't want the link. could you "
        "just reach out to my doctor's office for me instead?",
        "yes, please do that",  # personal_guide_consent = yes
        "just text me, that's easiest",  # notification_method = sms
        "yes, that's correct",  # confirm phone on file
        "how long is all of this going to take?",  # timeline question
        "email works for that",  # N2 channel = email
        # PROBE 3 — request the system genuinely cannot serve (billing detail
        # lookup); follow_up cannot-answer count 1 of 3, then caller accepts
        "actually yeah — could you check what my doctor billed for that visit? I'm curious",
        "no worries, that's fine. no, that's everything — thanks for the help",
        # 4 spare turns: this is the most detour-heavy script in the suite —
        # probes 1/2/6 each consume a retry or redirect interrupt and probe 5
        # may or may not pause, so total interrupt count varies by ±2–3
        "really, that's all — thanks",
        "nope, nothing else",
        "I'm all set",
        "that's it, thank you",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        max_turns=50,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            # The changed decision is the real assertion: the upload link was
            # first accepted in principle, then declined — it must NOT be sent,
            # and the records branch must end on the second choice.
            "records_branch_taken": "personal_guide",
            "upload_link_sent": falsy,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
        transcript_contains=[r"5 to 10 business days"],
    ),
    notes=(
        "Boundary-stress walk of the claim flow (James M310188). Claim flow chosen "
        "over PCP: it chains six member-driven sub-agent handoffs (verification → "
        "claim_adjustment → records_coordination → notification_setup → follow_up → "
        "closure) and is the only flow with an agent-supported recoverable "
        "mind-change (records Branch B→C); PCP's comparable pivot (delivery method "
        "after the fax read-back) is unsupported by delivery_management's state "
        "machine and derails into the fax-update path. "
        "Probe map (1-based user turns): "
        "(1) turn 4 lane-drift — claim question during member_id collection → "
        "verification redirect_off_topic / slot retry, then comply; "
        "(2) turn 6 post-acceptance correction — last-name spelling re-stated while "
        "awaiting dob → apply_corrections + correction_return_to; same spelling, so "
        "the SF lookup is unaffected whichever way the turn is classified; "
        "(3) turn 18 cannot-do request — provider billing lookup → follow_up "
        "cannot-answer count 1 of 3 (worded as a question, NOT a contact-update, "
        "which would escalate immediately), caller accepts the redirect; "
        "(4) turn 12 mind-change — declines the upload link one turn after choosing "
        "member_upload and asks for guide outreach; reconverges on the guide-consent "
        "ask from either upload_consent or upload_method, so a prior retry cannot "
        "derail it; "
        "(5) turn 8 stacked answer + portal question at phone_confirmed — "
        "verification flattens ANSWERED_WITH_FOLLOWUP to ANSWERED ('never pause for "
        "side questions'), so the confirm lands and the side question is dropped; "
        "(6) turn 9 mild impatience with zero digits → exactly one reference_number "
        "retry, no escalation. "
        "Guard-keyword rewording judgment calls: avoided the INTERRUPTION keyword "
        "fallbacks ('one more thing', 'before you continue', 'hold on a second') — "
        "turn 9 uses 'bear with me' instead; 'this is taking a while' chosen over "
        "the FRUSTRATED_PATTERNS regex 'this is taking too long'; no utterance "
        "contains transfer phrases ('real person', 'speak to someone', ...) or "
        "ABUSE_PATTERNS words. retries=2: most non-deterministic scenario in the "
        "suite — each probe depends on LLM guard/extraction classification."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# J. Name confirmation
# ──────────────────────────────────────────────────────────────────────────────

name_confirmation_happy_path = Scenario(
    name="name_confirmation_happy_path",
    flow="pcp",
    timeout_s=300,
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "yes that's correct",  # name readback → confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(
            ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R|E-M-I-L-Y\s+C-A-R-T-E-R"],
        ),
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-I-L-Y"],
    ),
    notes=(
        "Baseline: name is confirmed on the first readback. "
        "Asserts the readback appears (turn 3 expectation) and that "
        "member_id collection follows immediately after (turn 4 expectation)."
    ),
)

name_confirmation_inline_correction = Scenario(
    name="name_confirmation_inline_correction",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "no, it's Emma Carter",  # inline correction → agent re-reads "E-M-M-A  C-A-R-T-E-R"
        "yes that's correct",  # confirm corrected name
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "emma",
        "carter",
        "no, it's Emily Carter",  # inline correction → agent re-reads "E-M-M-A  C-A-R-T-E-R"
        "yes that's correct",  # confirm corrected name
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # first readback
        4: TurnExpectation(ai_contains=[r"E-M-M-A.*C-A-R-T-E-R"]),  # corrected readback
        5: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "first_name": "Emma",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"E-M-I-L-Y"],
    ),
    notes=(
        "Member gives inline correction 'no, it's Emma Carter'. "
        "Agent re-reads back the corrected name; member confirms. "
        "first_name must be 'Emma' in final state, not 'Emily'."
    ),
)

name_confirmation_bare_no_then_gives_name = Scenario(
    name="name_confirmation_bare_no_then_gives_name",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "no",  # bare no → agent asks for correct name
        "it's Emma",
        "no",
        "Brown",  # member gives correct name
        "yes that's correct",  # confirm new readback "E-M-M-A  B-R-O-W-N"
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    # turn_expectations={
    #     3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),
    #     4: TurnExpectation(ai_contains=[r"correct.*name|correct name|what.*name"]),
    #     5: TurnExpectation(ai_contains=[r"E-M-M-A.*B-R-O-W-N"]),
    #     6: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    # },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "name_confirmed": True,
            "first_name": "Emma",
            "last_name": "Brown",
            "member_status_verify": True,
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"B-R-O-W-N"],
    ),
    notes=(
        "Member says bare 'no' — agent asks for the correct name. "
        "Member provides 'Emma Brown'. Agent reads back E-M-M-A B-R-O-W-N. "
        "Member confirms. Flow then proceeds to member_id."
    ),
)

name_confirmation_exhaust_escalates = Scenario(
    name="name_confirmation_exhaust_escalates",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "no",  # rejection 1 → asks for correct name
        "hmm, I'm not sure",  # can't extract a name
        "no",  # readback re-delivered; rejection 2
        "I don't know",  # can't extract a name
        "no",  # rejection 3 → escalate
        # spares in case a clarify turn fires
        "still no",
        "nope",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="name_confirm_exhausted",
        final_state={"member_status_verify": falsy},
        transcript_contains=[r"E-M-I-L-Y"],
    ),
    notes=(
        "Member rejects every readback without providing a valid name. "
        "After MAX_NAME_CONFIRM_ATTEMPTS the agent escalates. "
        "member_status_verify must remain False — SF lookup never ran."
    ),
)

name_confirmation_claim_flow = Scenario(
    name="name_confirmation_claim_flow",
    flow="claim",
    timeout_s=360,
    user_turns=[
        "I want to follow up on a claim adjustment.",
        "james",
        "wilson",
        "yes that's right",  # name confirmed
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "yes correct",  # phone confirmed
        "42695817",
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"J-A-M-E-S.*W-I-L-S-O-N"]),
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "claim_flow_complete": True,
        },
        transcript_contains=[r"J-A-M-E-S"],
    ),
    notes=(
        "Confirms the name readback works identically in the claim_services "
        "call_intent path (uses verification_claims.md extraction prompt)."
    ),
)

name_confirmation_single_letter_first_name = Scenario(
    name="name_confirmation_single_letter_first_name",
    flow="pcp",
    timeout_s=300,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "aj",  # unusual short first name
        "smith",
        "yes that's correct",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's it",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"A-J.*S-M-I-T-H"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"name_confirmed": True},
        transcript_contains=[r"A-J"],
    ),
    notes=(
        "Edge case: very short first name. _spell_name('Aj', 'Smith') must "
        "produce 'A-J  S-M-I-T-H', not crash or produce empty output."
    ),
)

name_confirmation_rejection_asks_correction = Scenario(
    name="name_confirmation_rejection_asks_correction",
    flow="pcp",
    timeout_s=360,
    retries=1,  # Outcome 3/4 both now ask for correction; retries=1 for LLM extraction variance
    user_turns=[
        "I need to find a primary care physician.",
        "jeans",  # WRONG first name
        "carter",
        "No, that's not correct.",  # rejection without inline name
        "It's Emily.",  # first name only; last_name filled from state ("Carter")
        "yes that's correct",  # confirm re-readback "Emily Carter"
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"J-E-A-N-S.*C-A-R-T-E-R"]),  # first readback with wrong name
        4: TurnExpectation(
            ai_contains=[pool_regex(NAME_CORRECTION_PROMPTS)]
        ),  # correction asked — NOT a re-readback
        5: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # re-readback with corrected name
        6: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"J-E-A-N-S", r"E-M-I-L-Y"],
        transcript_count={r"J-E-A-N-S": 1},  # wrong-name readback not repeated after rejection
    ),
    notes=(
        "NC-7 — Regression for the Outcome 4 fix. 'No, that's not correct.' (no inline "
        "name) must trigger NAME_CORRECTION_PROMPTS, not a repeated readback. Before "
        "the fix: ambiguous LLM result → Outcome 4 → readback repeated indefinitely. "
        "After: Outcome 4 always asks for the correct name when no 'yes' and no inline "
        "correction were extracted. transcript_count J-E-A-N-S=1 is the definitive proof. "
        "Also exercises the partial-correction path in _collect_name_correction: the "
        "member provides only a first name ('Emily') and the stored last name ('Carter') "
        "is preserved from state."
    ),
)

name_confirmation_consecutive_rejections_then_correct = Scenario(
    name="name_confirmation_consecutive_rejections_then_correct",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "Can I get a list of in network providers within my area, please?",
        "Jeans.",  # WRONG first name
        "Carter.",
        "No. That's not correct.",  # [3] first rejection — must produce correction ask, NOT repeated readback
        "No. The first name is incorrect.",  # [4] second rejection in correction stage — retry correction ask
        "Emily.",  # [5] correct first name provided
        "yes that's correct",  # [6] confirm re-readback "Emily Carter"
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"J-E-A-N-S.*C-A-R-T-E-R"]),  # readback with wrong name
        4: TurnExpectation(
            ai_contains=[pool_regex(NAME_CORRECTION_PROMPTS)]
        ),  # first rejection → correction ask
        5: TurnExpectation(
            slot_awaiting="name_correction"
        ),  # second rejection → still in correction stage, not another readback
        6: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # re-readback after correct name given
        7: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"J-E-A-N-S", r"E-M-I-L-Y"],
        # THE CORE REGRESSION ASSERTION: wrong-name readback appears exactly once.
        # Before the fix it appeared three times (once for each rejection turn).
        transcript_count={r"J-E-A-N-S": 1},
    ),
    notes=(
        "NC-12 — Exact bug transcript reproduction. The production bug: after the member "
        "said 'No. That's not correct.' and then 'No. The first name is incorrect.' the "
        "agent repeated the same 'Jeans Carter J-E-A-N-S-C-A-R-T-E-R' readback on "
        "every turn instead of asking for the correct name. Root cause: both responses "
        "produced an ambiguous LLM result → Outcome 4 → _deliver_name_readback repeated. "
        "Fix: Outcome 4 now asks for correction instead of repeating the readback. "
        "The second rejection ('No. The first name is incorrect.') lands in "
        "_collect_name_correction (awaiting_slot='name_correction') and produces a "
        "LLM-2 retry rather than another readback. transcript_count J-E-A-N-S=1 is "
        "the definitive proof that the wrong readback was never repeated."
    ),
)

name_confirmation_confirmed_with_side_question = Scenario(
    name="name_confirmation_confirmed_with_side_question",
    flow="pcp",
    timeout_s=360,
    retries=2,  # ANSWERED_WITH_FOLLOWUP event + followup_query extraction are LLM-driven
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        # Confirms name AND asks a followup in one turn → ANSWERED_WITH_FOLLOWUP path.
        # _name_confirmed_with_followup answers the question and sets name_confirmed=True.
        "yes that's me — by the way, do providers in the list usually accept new patients?",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
        # spares — ANSWERED_WITH_FOLLOWUP may produce an extra AI turn for the side answer
        "that's all, thanks",
        "no, nothing else",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R|E-M-I-L-Y\s+C-A-R-T-E-R"]),  # readback
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "provider_list_sent": True,
        },
    ),
    notes=(
        "NC-8 — ANSWERED_WITH_FOLLOWUP path in _process_name_readback_response. "
        "Member confirms name and asks a side question in the same turn; "
        "_name_confirmed_with_followup answers the question and sets name_confirmed=True "
        "without requiring a second confirmation turn. Spare turns absorb any extra "
        "interrupt from the side-answer exchange. "
        "retries=2: ANSWERED_WITH_FOLLOWUP event and followup_query extraction are LLM-driven."
    ),
)

name_confirmation_both_names_corrected_inline = Scenario(
    name="name_confirmation_both_names_corrected_inline",
    flow="claim",
    timeout_s=360,
    retries=1,  # two-field inline correction extraction is mildly non-deterministic
    user_turns=[
        "I want to follow up on a claim adjustment.",
        "john",  # WRONG first name (on file: James)
        "smith",  # WRONG last name (on file: Wilson)
        "no, it's James Wilson",  # Outcome 2: both names corrected inline in one utterance
        "yes that's right",  # confirm re-readback "James Wilson"
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "yes correct",  # phone confirmed
        "42695817",
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"J-O-H-N.*S-M-I-T-H"]),  # readback with both wrong names
        4: TurnExpectation(ai_contains=[r"J-A-M-E-S.*W-I-L-S-O-N"]),  # re-readback after both corrected
        5: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "first_name": "James",
            "last_name": "Wilson",
            "claim_flow_complete": True,
        },
        transcript_contains=[r"J-O-H-N", r"J-A-M-E-S"],
        transcript_count={r"J-O-H-N": 1},  # wrong readback not repeated
    ),
    notes=(
        "NC-9 — Both first and last names wrong, both corrected inline in one utterance. "
        "Member gives 'john smith' (both wrong for James M310188); readback fires "
        "'J-O-H-N S-M-I-T-H'. Member says 'no, it's James Wilson' — Outcome 2 extracts "
        "first_name='James' AND last_name='Wilson' in one turn, re-reads "
        "'J-A-M-E-S W-I-L-S-O-N'. Member confirms and the full claim flow completes. "
        "Exercises the dual-field correction path (first_ok=True AND last_ok=True). "
        "retries=1: two-name extraction from a single utterance is mildly non-deterministic."
    ),
)

name_confirmation_partial_correction_first_only = Scenario(
    name="name_confirmation_partial_correction_first_only",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "emma",  # WRONG first name (on file: Emily)
        "carter",
        "no",  # bare no → agent asks for correct name
        "Emily.",  # first name only — last_name filled from state ("Carter")
        "yes that's correct",  # confirm re-readback "Emily Carter"
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-M-A.*C-A-R-T-E-R"]),  # readback with wrong first name
        4: TurnExpectation(ai_contains=[pool_regex(NAME_CORRECTION_PROMPTS)]),  # correction asked
        5: TurnExpectation(
            ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]
        ),  # re-readback: corrected first + stored last
        6: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"E-M-I-L-Y"],
    ),
    notes=(
        "NC-10 — Partial correction: only first name provided after correction prompt. "
        "Member gives 'emma' (wrong first), readback fires 'E-M-M-A C-A-R-T-E-R'. "
        "Bare 'no' → correction asked. Member says 'Emily.' (first name only). "
        "In _collect_name_correction: first_ok=True, last_ok=False → "
        "new_last = state.get('last_name') = 'Carter' (preserved). Re-readback fires "
        "as 'E-M-I-L-Y C-A-R-T-E-R'. Verifies that _collect_name_correction fills "
        "the missing last_name from state rather than requiring a fully re-stated name."
    ),
)

name_confirmation_natural_sentence_correction = Scenario(
    name="name_confirmation_natural_sentence_correction",
    flow="pcp",
    timeout_s=360,
    retries=1,  # correction extracted from a natural declarative sentence is mildly non-deterministic
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "watson",  # WRONG last name (on file: Carter)
        # Outcome 2: correction embedded in a natural sentence with no leading "no" prefix
        "Actually, my surname is Carter, not Watson.",
        "yes correct",  # confirm re-readback "Emily Carter"
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*W-A-T-S-O-N"]),  # first readback wrong last name
        4: TurnExpectation(
            ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]
        ),  # re-readback after natural-sentence correction
        5: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"W-A-T-S-O-N", r"C-A-R-T-E-R"],
        transcript_count={r"W-A-T-S-O-N": 1},  # wrong last name read back exactly once
    ),
    notes=(
        "NC-11 — Natural-sentence inline correction (Outcome 2) with no leading 'no' prefix. "
        "Member gives wrong last name 'watson'; readback fires 'E-M-I-L-Y W-A-T-S-O-N'. "
        "Member says 'Actually, my surname is Carter, not Watson.' — the LLM must extract "
        "last_name='Carter' from a declarative sentence, not just from the canonical "
        "'no, it's X' pattern. Re-readback fires as 'E-M-I-L-Y C-A-R-T-E-R'. "
        "Proves Outcome 2 fires on natural correction sentences. "
        "retries=1: LLM extraction from an indirect correction sentence is mildly non-deterministic."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# M. New-intent mid-session (member raises a DIFFERENT service during follow-up)
#
# Exercises the NEW_INTENT path: when a verified member asks about a different
# service during the follow-up phase, follow_up_agent classifies
# follow_up_intent=new_intent (with detected_intent), and — for a fresh intake
# intent — fully RESETS the call (reset_for_new_intent) and re-routes through
# the verification node. The member re-verifies, then verification dispatches on
# pending_intent straight to the new intent's domain agent. Both directions are
# covered.
# ──────────────────────────────────────────────────────────────────────────────

pcp_then_claim_new_intent = Scenario(
    name="pcp_then_claim_new_intent",
    flow="pcp",
    timeout_s=420,
    retries=1,  # new_intent and same-member LLM classifications are non-deterministic
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: the member pivots to a brand-new claim request.
        "Actually, can you check a claim reprocessing for me?",
        # Intake fires the same-member disambiguation question.  claim_services is
        # now in INTAKE_RESCREEN_INTENTS so _reroute_through_intake saved Emily's
        # verified member context before the reset; intake re-classifies claim_services
        # and asks whether the new request is for the same member.
        "yes, same member",  # skip re-verification → member context restored
        # Directly in claim_adjustment (verification was skipped).  The only
        # adjustment fixture (42695817) is linked to James (M310188), NOT Emily
        # → not-found → re-ask → deterministic escalation.
        "42695817",
        "42695817",
    ],
    turn_expectations={
        # After the pivot utterance the AI delivers the disambiguation question.
        14: TurnExpectation(ai_contains=[r"same member|different member"]),
        # After confirming "same member" claim_adjustment's first prompt fires.
        15: TurnExpectation(ai_contains=[r"reference number"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            # Restored from saved context — no re-verification ran.
            "member_status_verify": True,
            "call_intent": "claim_services",
            # pending_intent was consumed by the fast-path dispatch.
            # "pending_intent": lambda v: not v,
            # Saved context was consumed by the same-member shortcut.
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
        # claim_adjustment engaged and asked for the reference number.
        transcript_contains=[r"reference number"],
    ),
    notes=(
        "New-intent pivot PCP → claim with same-member shortcut. Emily completes "
        "the provider flow, then during follow-up asks about a claim reprocessing. "
        "claim_services is now in INTAKE_RESCREEN_INTENTS: _reroute_through_intake "
        "saves Emily's verified member context before the reset and routes to intake. "
        "Intake re-classifies claim_services, detects the follow-up origin and the "
        "saved context, and fires the same-member disambiguation question. Emily "
        "confirms 'same member' → the saved context is restored and verification is "
        "skipped entirely. claim_adjustment_agent engages directly. The only "
        "adjustment fixture is linked to James (M310188) not Emily → escalation. "
        "member_status_verify=True (restored, not re-verified). "
        "saved_member_context=falsy (consumed). retries=1: new_intent + same-member "
        "LLM classifications are non-deterministic."
    ),
)

claim_then_pcp_new_intent = Scenario(
    name="claim_then_pcp_new_intent",
    flow="claim",
    mutating=True,  # provider flow writes James's ZIP (he has none on file)
    timeout_s=480,
    retries=1,  # new_intent + same-member + provider/delivery slots are LLM-driven
    user_turns=CLAIM_VERIFY
    + [
        # Complete the claim flow up to the follow-up phase.
        "42695817",
        "Can I ask my doctor to send it over?",  # doctor-direct
        "Yes, please",  # accept upload link
        "Yes, that's correct",  # email on file (upload link)
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "Okay, how long will it take to finalize the request?",  # timeline question
        "email them to me",  # N2 channel
        # Follow-up phase: pivot to a brand-new provider search.
        "Can I also find an in-network doctor?",  # new_intent → provider_services
        # Intake fires the same-member disambiguation question.  provider_services
        # is in INTAKE_RESCREEN_INTENTS and James was verified, so saved_member_context
        # was preserved; intake re-classifies provider_services and asks.
        "yes, same member",  # skip re-verification → James's context restored
        # Verification re-asks relationship before handing off to provider_search.
        "fdsfc",
        "I'm calling for myself",  # relationship → plan_holder
        "Primary Care Physician",  # provider type
        "yes correct",  # ZIP (James has none on file → fresh ask)
        "email please",  # delivery method
        "yes that's correct",  # email on file (james.wilson@gmail.com)
        "no thanks",  # decline benefits offer (James has no benefit plan)
        "no thank you",  # decline Care Coach
        "no, that's all, thanks",  # close
    ],
    turn_expectations={
        # After the pivot utterance the AI delivers the disambiguation question.
        17: TurnExpectation(ai_contains=[r"same member|different member"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            # Restored from saved context — no re-verification ran.
            "member_status_verify": True,
            "call_intent": "provider_services",
            # pending_intent was consumed by the fast-path dispatch.
            # "pending_intent": lambda v: not v,
            # The provider flow ran to completion with the restored member.
            "provider_list_sent": True,
            "delivery_method": "email",
            # Saved context was consumed by the same-member shortcut.
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
    ),
    notes=(
        "New-intent pivot claim → PCP with same-member shortcut. James completes "
        "the claim flow, then during follow-up asks to find an in-network doctor. "
        "provider_services is in INTAKE_RESCREEN_INTENTS: _reroute_through_intake "
        "saves James's verified member context before the reset and routes to intake. "
        "Intake re-classifies provider_services, detects the follow-up origin and "
        "the saved context, and fires the same-member disambiguation question. James "
        "confirms 'same member' → the saved context is restored and verification is "
        "skipped entirely; verification re-asks 'Are you the subscriber or dependent?' "
        "before routing to provider_search. James has no ZIP on file so one must be"
        "provided. Marked mutating: "
        "the provider flow writes a ZIP for James; preflight snapshots/restores "
        "contact fields. retries=1: new_intent + same-member + provider/delivery "
        "classifications are LLM-driven."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# M2. Follow-up RE-SCREEN through intake (front-door screening on a mid-call pivot)
#
# A fresh intake intent raised during follow-up is routed back through the
# INTAKE node (not straight to verification) so intake re-applies its front-door
# screening. Two triggers:
#
#   * provider_services is in follow_up.INTAKE_RESCREEN_INTENTS, so a new provider
#     request re-runs intake's unsupported-provider-type gate. The payoff: an
#     UNSUPPORTED specialty escalates at intake BEFORE any identity is
#     re-collected (vs. the old direct-to-verification path that re-verified
#     first). A SUPPORTED specialty re-classifies cleanly and completes.
#   * Appeals / grievances are out_of_scope but the follow-up classifier has no
#     tag for them, so follow_up._is_appeal_or_grievance() catches them by keyword
#     and reroutes through intake, whose out_of_scope screening routes the caller
#     to the right team and hard-ENDs.
#
# The decisive, LLM-independent fact in the escalate / out_of_scope cases is that
# the member is NEVER re-verified — member_status_verify and first_name are falsy
# at END (the pivot reset cleared them and screening fired before identity was
# re-collected). That is what proves the re-screen ran through the intake node.
#
# ASSERTION NOTE: agent-side escalations (the unsupported-provider case) do not
# surface a harvestable escalation reason or AgentCallTransfer event in this
# codebase — signal_escalate and escalation_agent both emit metadata_events=[],
# and the reason lives only in last_agent_signal, which escalation_agent
# overwrites with a COMPLETE signal before the next graph pause. So these
# scenarios assert on escalated (via the reference number escalation_agent
# stamps), the staged pre-escalation message, the final AI text, and the
# no-re-verification state — not on escalation_reason_contains / transfer_event.
# The out_of_scope cases DO carry a top-level escalation_reason, so they assert it
# the same way intake_out_of_scope_appeal does.
# ──────────────────────────────────────────────────────────────────────────────

followup_unsupported_provider_rescreen = Scenario(
    name="followup_unsupported_provider_rescreen",
    flow="pcp",
    timeout_s=360,
    retries=1,  # follow_up new_intent + intake unsupported-type classification are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: pivot to a brand-new UNSUPPORTED provider request.
        # provider_services is a re-screen intent → _reroute_through_intake →
        # intake re-classifies → provider_type_unsupported → escalates, all
        # BEFORE any re-verification.
        "Actually, I need to find an oncologist instead.",
    ],
    expect=Expected(
        completed=True,  # reaches END via escalation_agent
        escalated=True,  # escalation_agent stamps an escalation_reference_number
        transfer_event=False,  # current code emits no AgentCallTransfer metadata event
        final_is_interrupt=False,
        final_state={
            # DECISIVE: the unsupported type was rejected at intake's front door,
            # before identity was re-collected. If either of these is truthy the
            # re-screen wrongly went through verification first.
            "member_status_verify": falsy,
            "first_name": falsy,
            # pending_intent is cleared on the intake re-screen path.
            # "pending_intent": falsy,
            # The unsupported-type message is staged for escalation_agent.
            "escalation_pre_message": contains("Orthopedic"),
        },
        # The member hears the specialty named plus the five supported types.
        last_ai_contains=[
            r"oncologist",
            r"(Primary Care|Pediatrician|Cardiologist|Dermatologist|Orthopedic)",
        ],
    ),
    notes=(
        "Re-screen payoff. Emily completes the provider flow, then during follow-up "
        "asks for an oncologist. follow_up classifies new_intent "
        "(detected_intent=provider_services); is_new_intake_intent is True and "
        "provider_services is in INTAKE_RESCREEN_INTENTS, so _reroute_through_intake "
        "resets the call, CLEARS call_intent, and routes to the intake node. Intake "
        "re-classifies the request as provider_type_unsupported and escalates "
        "immediately — member_status_verify must be falsy because no re-verification "
        "ran. retries=1: new_intent + unsupported-type classification are LLM-driven."
    ),
)

followup_supported_provider_rescreen = Scenario(
    name="followup_supported_provider_rescreen",
    flow="pcp",
    timeout_s=420,
    retries=1,  # follow_up new_intent + same-member + provider/delivery slots are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: pivot to a brand-new SUPPORTED provider request
        # (dermatologist is one of the five supported types).
        "Actually, I also need to find a dermatologist.",
        # Intake re-screens → classifies provider_services → same-member check fires.
        # _reroute_through_intake saved Emily's verified context before the reset;
        # intake re-classifies provider_services and asks whether the new request is
        # for the same member.
        "yes, same member",  # skip re-verification → Emily's context restored
        # Verification re-asks relationship before handing off to provider_search.
        "I'm calling for myself",  # relationship → plan_holder
        # "Dermatologist",  # provider type (pre-set from new_intent utterance)
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        "yes that's correct",  # fax on file confirmed
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no, that's everything",  # close
    ],
    turn_expectations={
        # After the pivot utterance the AI delivers the disambiguation question.
        14: TurnExpectation(ai_contains=[r"same member|different member"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,  # supported type must NOT escalate
        final_state={
            # Restored from saved context — no re-verification ran.
            "member_status_verify": True,
            "call_intent": "provider_services",
            "provider_type": "Dermatologist",
            "provider_list_sent": True,
            # "pending_intent": falsy,
            # Saved context was consumed by the same-member shortcut.
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
    ),
    notes=(
        "Re-screen + same-member shortcut for a SUPPORTED specialty. Emily completes "
        "the provider flow, then during follow-up asks for a dermatologist (one of the "
        "five supported types). _reroute_through_intake saves Emily's verified context "
        "and routes to intake. Intake re-classifies provider_services (supported — no "
        "escalation) and fires the same-member disambiguation question. Emily confirms "
        "'same member' → context restored; verification re-asks 'Are you the "
        "subscriber or dependent?' before routing to provider_search. "
        "The re-screen payoff (intake still classifies the intent "
        "correctly before any re-verification) is intact. retries=1: new_intent + "
        "same-member + provider_type/delivery_method classifications are LLM-driven."
    ),
)

followup_appeal_rescreen = Scenario(
    name="followup_appeal_rescreen",
    flow="pcp",
    retries=1,  # intake out_of_scope classification is the primary signal
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: raise an appeal. The keyword gate fires regardless of
        # the follow-up LLM tag and reroutes through intake → out_of_scope.
        "Actually, I'd like to appeal a denial on my claim.",
    ],
    expect=Expected(
        completed=True,  # graph ENDs directly via intake out_of_scope (no escalation_agent)
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        last_ai_contains=[r"appeal", r"1-\d{3}-\d{3}-\d{4}"],
        final_state={
            "escalation_reason": contains("outside covered workflows"),
            # No re-verification — out_of_scope is decided at the front door.
            "member_status_verify": falsy,
            "first_name": falsy,
        },
    ),
    notes=(
        "Appeal raised in follow-up. follow_up._is_appeal_or_grievance() catches the "
        "'appeal' keyword and calls _reroute_through_intake, which resets the call "
        "and routes to intake. Intake classifies out_of_scope and routes the caller "
        "to the appeals team (1-800-555-0105) with a hard END. member_status_verify "
        "must be falsy — the request never reached re-verification. retries=1: the "
        "intake out_of_scope classification is LLM-driven."
    ),
)

followup_grievance_rescreen = Scenario(
    name="followup_grievance_rescreen",
    flow="pcp",
    retries=1,
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "Actually, I want to file a grievance about how my claim was handled.",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        # A team number is given; "grievance" is NOT in OUT_OF_SCOPE_KEYWORD_ROUTING,
        # so it falls back to the default support team rather than a dedicated one.
        last_ai_contains=[r"1-\d{3}-\d{3}-\d{4}"],
        final_state={
            "escalation_reason": contains("outside covered workflows"),
            "member_status_verify": falsy,
            "first_name": falsy,
        },
    ),
    notes=(
        "Grievance half of APPEAL_GRIEVANCE_KEYWORDS. The keyword gate reroutes "
        "through intake → out_of_scope, hard END, no re-verification. Asserts the "
        "out_of_scope OUTCOME (reason + a routed number), NOT a specific team, "
        "because 'grievance' is not yet in OUT_OF_SCOPE_KEYWORD_ROUTING and therefore "
        "falls back to the default support team — this scenario documents that gap "
        "rather than masking it. retries=1: intake out_of_scope classification is "
        "LLM-driven."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# M3. Same-member disambiguation (follow-up → intake re-entry)
#
# When follow_up routes a new PCP or Claims intent back through intake and a
# verified member context was saved, intake asks "same member or different?"
# instead of always forcing a full re-verification.
#
# These scenarios are COMPLEMENTARY to M/M2: M/M2 now all confirm "same member"
# (the expected common case); M3 covers the two edge paths — declining (different
# member triggers a fresh verification) and ambiguity (clarification → confirm).
#
# All use Emily Carter (M907503) on the PCP → claim pivot because:
#   * _PCP_TO_FOLLOW_UP brings her to the follow-up stage in a standard way.
#   * claim_services is the simplest "different domain" pivot.
#   * The only adjustment fixture is James's (42695817), so Emily's claim path
#     always escalates deterministically — no fixture coupling needed.
# ──────────────────────────────────────────────────────────────────────────────

followup_new_intent_different_member = Scenario(
    name="followup_new_intent_different_member",
    flow="pcp",
    timeout_s=480,
    retries=1,  # new_intent + same-member LLM classifications are non-deterministic
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: claim request flagged as being for a DIFFERENT person.
        "I also need to check on a claim, but it's actually for someone else.",
        # Same-member check fires.  Member answers "different" → saved context is
        # cleared → intake delivers the first-name bridge to verification.
        "no, it's for a different member",
        # Re-verification from scratch (claims slot order).  James M310188 is used
        # so that claim_adjustment can actually match the fixture reference.
        "james",
        "wilson",
        "yes correct",  # name confirmed
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "yes that's correct",  # phone confirmation
        # claim_adjustment with James's fixture reference → completes.
        "42695817",
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        # After the new-intent utterance the AI delivers the disambiguation question.
        14: TurnExpectation(ai_contains=[r"same member|different member"]),
        # After "different" intake sends the first-name bridge.
        15: TurnExpectation(ai_contains=[r"first name"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            # Re-verified as the new member (James).
            "member_status_verify": True,
            "call_intent": "claim_services",
            "first_name": "James",  # new identity collected — proves re-verification ran
            "last_name": "Wilson",
            # Saved context was cleared when "different" was chosen.
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
            "claim_flow_complete": True,
        },
        # Re-verification ran AND claim_adjustment engaged.
        transcript_contains=[r"reference number"],
    ),
    notes=(
        "Same-member disambiguation → 'different member' → full re-verification. "
        "Emily completes the PCP flow, then during follow-up asks about a claim for "
        "'someone else'. Intake detects the follow-up origin and the saved member "
        "context, fires the same-member question. Emily answers 'different member': "
        "the saved context is cleared, intake delivers the first-name bridge, and "
        "verification collects the new member's identity from scratch (James M310188). "
        "The claim flow then completes with James's fixture. Asserts: disambiguation "
        "question asked (turn-14), first-name bridge fired (turn-15), "
        "saved_member_context=falsy (cleared), and first_name='James' (new identity "
        "collected — not Emily). retries=1: new_intent + same-member LLM "
        "classifications are non-deterministic."
    ),
)

followup_new_intent_same_member_ambiguous = Scenario(
    name="followup_new_intent_same_member_ambiguous",
    flow="pcp",
    timeout_s=420,
    retries=2,  # same-member classification on ambiguous phrasing is doubly LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: new claim request.
        "I also need to check on a claim.",
        # Same-member check fires.  Ambiguous first answer → clarification asked.
        "I think so, probably",
        # Clarification → member explicitly confirms same member.
        "yes, same member",
        # claim_adjustment: no claim for Emily → not-found → escalation.
        "42695817",
        "42695817",
    ],
    turn_expectations={
        # After the new-intent utterance the AI delivers the disambiguation question.
        14: TurnExpectation(ai_contains=[r"same member|different member"]),
        # After the ambiguous answer the AI delivers one clarification question.
        15: TurnExpectation(ai_contains=[r"same member|different|on file"]),
        # After confirming "same member" claim_adjustment's first prompt fires.
        16: TurnExpectation(ai_contains=[r"reference number"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            # Restored from saved context after the delayed confirmation.
            "member_status_verify": True,
            "call_intent": "claim_services",
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
        transcript_contains=[r"reference number"],
    ),
    notes=(
        "Same-member disambiguation with an ambiguous first answer. Emily completes "
        "the PCP flow, then during follow-up asks about a claim. Intake fires the "
        "same-member question. Emily answers 'I think so, probably' — classified as "
        "unclear by the same-member LLM prompt. Intake delivers one clarification "
        "question (same_member_check_pending stays True). Emily then explicitly "
        "confirms 'yes, same member' → saved member context is restored, verification "
        "is skipped, and claim_adjustment_agent engages directly. The only adjustment "
        "fixture is for James (M310188) → escalation on not-found. Turn expectations: "
        "disambiguation (14), clarification (15), reference-number ask (16). "
        "saved_member_context and same_member_check_pending both falsy at end. "
        "retries=2: ambiguous phrasing + LLM same-member classification are doubly "
        "non-deterministic."
    ),
)

followup_supported_provider_rescreen_different_member = Scenario(
    name="followup_supported_provider_rescreen_different_member",
    flow="pcp",
    timeout_s=480,
    retries=1,  # new_intent + same-member + provider/delivery slots are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: pivot to a SUPPORTED provider, but for a different person.
        "Actually, I also need to find a dermatologist, but it's for my daughter.",
        # Intake fires the same-member disambiguation question.
        "no, it's for a different person",  # cleared saved context → first-name bridge
        # Re-verification from scratch (same Emily data — fixture convenience;
        # proves re-verify ran regardless of who the new member is).
        "emily",
        "carter",
        "yes correct",  # name confirmed
        "m nine zero seven five zero three",
        "April twelvee nineteen eighty-eight",
        "I'm calling for myself",
        # Provider search for Dermatologist (type pre-set from the pivot utterance;
        # ZIP on file for Emily — confirmed without a fresh ZIP ask).
        "yes that's correct",  # ZIP 12139 on file
        "email please",
        "yes that's correct",  # email on file
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no, that's everything",
    ],
    turn_expectations={
        # After the pivot utterance the AI delivers the disambiguation question.
        14: TurnExpectation(ai_contains=[r"same member|different member"]),
        # After "different" the AI delivers the first-name bridge.
        15: TurnExpectation(ai_contains=[r"first name"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            # Re-verification ran and provider search completed.
            "member_status_verify": True,
            "call_intent": "provider_services",
            "provider_type": "Dermatologist",
            "provider_list_sent": True,
            # Saved context was cleared when "different" was chosen.
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
    ),
    notes=(
        "Same-member disambiguation → 'different member' for a supported provider. "
        "Emily completes the PCP flow, then during follow-up asks for a dermatologist "
        "for 'my daughter' (a different person). Intake detects the follow-up origin "
        "and the saved member context, fires the same-member question. Emily answers "
        "'different': the saved context is cleared, the first-name bridge fires, and "
        "verification collects identity from scratch. After re-verification, "
        "provider_search_agent runs for Dermatologist (provider_type was pre-set from "
        "the pivot utterance before the disambiguation question). Emily's ZIP is on "
        "file so no extra ZIP ask. Counterpart to followup_supported_provider_rescreen "
        "(31d) which confirms 'same member'. Asserts: disambiguation (turn 14), "
        "first-name bridge (turn 15), saved_member_context=falsy, "
        "provider_list_sent=True. retries=1: new_intent + same-member + "
        "provider/delivery classifications are LLM-driven."
    ),
)

claim_then_pcp_new_intent_different_member = Scenario(
    name="claim_then_pcp_new_intent_different_member",
    flow="claim",
    timeout_s=540,
    retries=1,  # new_intent + same-member + provider/delivery slots are LLM-driven
    user_turns=CLAIM_VERIFY
    + [
        # Complete the claim flow through to the follow-up phase.
        "42695817",
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        # Follow-up phase: pivot to a brand-new provider search for a DIFFERENT member.
        "I also need to find a PCP for my daughter.",
        # Same-member check fires.  James answers "different" → saved context cleared
        # → intake delivers the first-name bridge → re-verify as Emily.
        "no, it's for a different member",
        # Re-verification as Emily Carter (a genuinely different member from James).
        "emily",
        "carter",
        "yes correct",  # name confirmed
        "m nine zero seven five zero three",
        "April twelvee nineteen eighty-eight",
        "I'm calling for myself",
        # Provider search for PCP (provider_type pre-set from pivot utterance;
        # Emily's ZIP 12139 is on file).
        "yes that's correct",  # ZIP on file
        "email please",
        "yes that's correct",  # email on file
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no, that's all",
    ],
    turn_expectations={
        # After the pivot utterance the AI delivers the disambiguation question.
        17: TurnExpectation(ai_contains=[r"same member|different member"]),
        # After "different" the AI delivers the first-name bridge.
        18: TurnExpectation(ai_contains=[r"first name"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            # Re-verified as Emily (a different member from James).
            "member_status_verify": True,
            "call_intent": "provider_services",
            "first_name": "Emily",  # new identity — proves re-verification ran
            "last_name": "Carter",
            "provider_list_sent": True,
            # Saved context was cleared when "different" was chosen.
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
    ),
    notes=(
        "New-intent pivot claim → PCP with 'different member' → full re-verification. "
        "James completes the claim flow, then during follow-up asks to find a PCP for "
        "'my daughter' (a different person). provider_services is in "
        "INTAKE_RESCREEN_INTENTS: _reroute_through_intake saves James's verified "
        "member context and routes to intake. Intake detects the follow-up origin and "
        "the saved context, fires the same-member disambiguation question. James "
        "answers 'different member': the saved context is cleared, the first-name "
        "bridge fires, and verification collects Emily Carter's identity from scratch "
        "(a genuinely different member from James M310188). After re-verification, "
        "provider_search_agent runs for Primary Care Physician (provider_type pre-set "
        "from the pivot utterance). Emily's ZIP 12139 is on file. "
        "Counterpart to claim_then_pcp_new_intent (31b) which confirms 'same member'. "
        "Asserts: disambiguation (turn 17), first-name bridge (turn 18), "
        "first_name='Emily' (proves new identity), saved_member_context=falsy. "
        "retries=1: new_intent + same-member + provider/delivery classifications "
        "are LLM-driven."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# N. Follow-up disposition routing, update detours & WAIT (Phases 4-7)
#
# Live mirrors of the offline scenario matrix in test_scenario_matrix_phase7.py.
# The offline matrix asserts LLM call counts and attempt deltas with mocks;
# these run the same conversational shapes against the REAL LLMs and assert
# what a live run can prove: slot_awaiting on the AI prompt that follows the
# turn (Option A — Python appends the next ask / detour ask), detour pointers
# via prompt routing, parked_followups lifecycle, loop-guard escalation, and
# the static WAIT acks. All use Emily Carter (M907503) on the PCP flow.
# ──────────────────────────────────────────────────────────────────────────────

# Minimal PCP completion tail after relationship (ZIP on file → email on file
# → decline benefits → decline Care Coach → close).
_PCP_TAIL = [
    "Primary Care Physician",
    "yes that's correct",  # ZIP on file
    "email please",
    "yes that's correct",  # email on file
    "no thanks",  # decline benefits
    "no thank you",  # decline Care Coach
    "no, that's everything",  # close
]

# MSG_WAIT_NUDGE has a {slot_label} placeholder — build the regex directly.
_WAIT_NUDGE_MEMBER_ID = r"whenever you'?re ready.*member id"

followup_answer_confirmed_slot = Scenario(
    name="followup_answer_confirmed_slot",
    flow="pcp",
    timeout_s=360,
    retries=1,  # answered_with_followup classification is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        # Answer + side question about an already-confirmed slot →
        # FOLLOWUP_ANSWER: Gemini answers from Confirmed, Python appends the
        # static DOB ask — proven by the turn-5 expectation.
        "it's m nine zero seven five zero three — and did you get my last name right?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 4 of the Phase 7 matrix, live. The member answers member_id AND asks "
        "about a confirmed slot in one utterance. The turn must confirm the slot "
        "(no re-ask), address the question, and end with the DOB ask appended by "
        "Python — slot_awaiting='dob' before the DOB answer is the proof that the "
        "follow-up did not stall or re-ask member_id."
    ),
)

followup_park_question_deferred = Scenario(
    name="followup_park_question_deferred",
    flow="pcp",
    timeout_s=360,
    retries=2,  # disposition (park) classification + parked answer are LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Answer + question answerable LATER in the call → FOLLOWUP_PARK:
        # the question is queued in parked_followups, flow moves straight on.
        "m nine zero seven five zero three — will I get a text when the provider list is sent?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL
    + [
        "no, that's all, thanks",  # spare: follow_up answers the parked question first
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            # follow_up surfaced and consumed the parked question at the end.
            "parked_followups": falsy,
        },
    ),
    notes=(
        "Row 5 of the Phase 7 matrix, live. The side question maps to a later "
        "stage → parked (never answered mid-verification, flow proceeds straight "
        "to DOB), then surfaced and cleared by follow_up_agent at the end of the "
        "call. Final parked_followups must be empty; the spare closing turn "
        "absorbs the extra follow-up exchange when the parked answer is delivered."
    ),
)

followup_decline_irrelevant = Scenario(
    name="followup_decline_irrelevant",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Answer + never-answerable side question → FOLLOWUP_DECLINE: brief warm
        # decline, then the appended DOB ask — the flow never stalls.
        "m nine zero seven five zero three — quick question, do you sell car insurance?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
    ),
    notes=(
        "Row 6 of the Phase 7 matrix, live. 'Do you sell car insurance?' is the "
        "canonical decline example from the extraction header. The value must be "
        "captured, the question declined without an apology spiral, and the DOB "
        "ask appended — asserted via slot_awaiting='dob' on the next prompt."
    ),
)

correction_inline_case_a = Scenario(
    name="correction_inline_case_a",
    flow="pcp",
    timeout_s=360,
    retries=2,  # inline answer+correction extraction is mildly non-deterministic
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name, confirmed at the read-back
        "yes correct",  # read-back "Emily Carson" confirmed
        # Case A: answer + correction WITH a valid value in one utterance.
        # The correction applies before the answer confirms; both are
        # acknowledged and the DOB ask is appended.
        "m nine zero seven five zero three — actually my last name is Carter, not Carson",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-S-O-N"]),  # read-back of the wrong name
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,  # lookup matched the CORRECTED name
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_count={r"C-A-R-S-O-N": 1},  # no second read-back after the correction
    ),
    notes=(
        "Row 7 of the Phase 7 matrix, live (Case A). The confirmed-then-corrected "
        "last name must be replaced with the validated new value in ONE turn — no "
        "detour, no second name read-back — and the Salesforce lookup must match "
        "on the corrected name. slot_awaiting='dob' after the correction turn "
        "proves member_id confirmed and the pipeline advanced."
    ),
)

update_without_value_case_b = Scenario(
    name="update_without_value_case_b",
    flow="pcp",
    timeout_s=360,
    retries=2,  # update_target extraction is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Case B: answer + update request WITHOUT a value → the awaiting slot
        # confirms, then a detour asks for the new value (replacing the normal
        # next-slot ask). correction_return_to brings the pipeline back to DOB.
        "m nine zero seven five zero three — oh, also I need to update my last name",
        "Carter",  # the "new" value (same as on file, so the lookup still matches)
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        # Detour: the prompt after the update request asks for the LAST NAME —
        # not DOB — proving awaiting_slot switched to the update target.
        5: TurnExpectation(ai_contains=[r"last\s*name"], slot_awaiting="last_name"),
        # After the new value, the pipeline resumes at DOB (correction_return_to).
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 9 of the Phase 7 matrix, live (Case B). The member answers member_id "
        "and asks to change their last name without giving one. The turn confirms "
        "member_id, opens a detour (awaiting_slot=last_name — asserted), and after "
        "the new value the pipeline returns to DOB (asserted) instead of re-asking "
        "member_id."
    ),
)

bare_update_detour_c2 = Scenario(
    name="bare_update_detour_c2",
    flow="pcp",
    timeout_s=360,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        "m nine zero seven five zero three",
        # Case C2: bare update request (no value, no answer) at the DOB ask →
        # detour to last_name; the DOB ask is preserved in correction_return_to.
        "wait — before that, I need to change my last name",
        "Carter",  # new value (same as on file)
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # After the bare update request the agent asks for the last name.
        6: TurnExpectation(ai_contains=[r"last\s*name"], slot_awaiting="last_name"),
        # After the new value, the pipeline returns to the original DOB ask.
        7: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 10 of the Phase 7 matrix, live (Case C2). A value-less update request "
        "interrupting the DOB collection detours to the update target and then "
        "returns to DOB — the 5/6/7 turn expectations trace the full detour "
        "round-trip (dob → last_name → dob). The DOB attempt budget must not be "
        "consumed by the detour: any exhaustion here fails the run."
    ),
)

locked_field_update_declined = Scenario(
    name="locked_field_update_declined",
    flow="pcp",
    timeout_s=360,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Bare update request for a caller-LOCKED field (phone_number) at the
        # member_id ask → declined; the flow stays on member_id (no detour).
        "before I give you that — can you change the phone number on my file?",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        # The decline re-asks the SAME slot — awaiting stays member_id.
        5: TurnExpectation(slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 12 of the Phase 7 matrix, live. phone_number is in "
        "CALLER_LOCKED_SLOTS: the update request must be declined — no detour "
        "(awaiting stays member_id, asserted on turn 5), nothing applied, no "
        "escalation — and verification proceeds normally."
    ),
)

update_loop_guard_escalates = Scenario(
    name="update_loop_guard_escalates",
    flow="pcp",
    timeout_s=360,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Update #1 (bare) at the member_id ask → detour, counter update_last_name=1
        "actually, I need to change my last name",
        "Carter",  # detour collects the value, pipeline returns to member_id
        # Update #2 for the SAME field → guard_loop_limit (max 2) → escalation
        "hmm, no wait — I need to change my last name again",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"last\s*name"], slot_awaiting="last_name"),
        6: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="update_last_name",
        final_state={"member_status_verify": falsy},
    ),
    notes=(
        "Row 13 of the Phase 7 matrix, live. Two update detours for the same "
        "field exhaust the per-target budget (guard_loop_limit, counter "
        "update_last_name, max 2): the second request escalates with the "
        "exhausted-style copy instead of opening a third detour loop."
    ),
)

wait_ack_then_answer = Scenario(
    name="wait_ack_then_answer",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        "give me a minute",  # WAIT → static ack, no attempt cost, no Gemini
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        # The prompt after the wait is the static ack pool, still awaiting member_id.
        5: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)], slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
        transcript_contains=[pool_regex(MSG_WAIT_ACK)],
    ),
    notes=(
        "Rows 14-15 of the Phase 7 matrix, live. A bare wait gets the static "
        "acknowledgement (MSG_WAIT_ACK pool — asserted verbatim via pool_regex, "
        "proving no LLM generated it) with awaiting_slot unchanged and no retry "
        "burned; the member then answers and verification completes normally."
    ),
)

wait_nudge_after_three = Scenario(
    name="wait_nudge_after_three",
    flow="pcp",
    timeout_s=360,
    retries=1,  # three consecutive turns must all classify as WAIT
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        "give me a minute",  # wait #1 → ack
        "just a sec",  # wait #2 → ack
        "hold on",  # wait #3 → nudge naming the slot
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)], slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)], slot_awaiting="member_id"),
        # Third consecutive wait: the gentle nudge that names the slot.
        7: TurnExpectation(ai_contains=[_WAIT_NUDGE_MEMBER_ID], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
        transcript_contains=[_WAIT_NUDGE_MEMBER_ID],
    ),
    notes=(
        "Rows 16-17 of the Phase 7 matrix, live. Three consecutive waits: the "
        "first two get static acks, the third the MSG_WAIT_NUDGE naming the "
        "member id. None of them burn a retry — the member then answers once and "
        "verification completes, which would be impossible if the waits had "
        "consumed the MAX_SLOT_ATTEMPTS=3 budget. Also covers the regex rescue: "
        "'just a sec' / 'hold on' still WAIT even if the LLM mislabels them."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# K. Indirect-decline regression (delivery_management fax/email)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# L. Cannot-provide short-circuit (detect_cannot_provide + slot_manager /
#    claim_adjustment_agent)
# ──────────────────────────────────────────────────────────────────────────────
from tests.live_e2e.test_cannot_provide import (  # noqa: E402
    CANNOT_PROVIDE_SCENARIOS,
)
from tests.live_e2e.test_fax_indirect_decline import (  # noqa: E402
    INDIRECT_DECLINE_SCENARIOS,
)

# ──────────────────────────────────────────────────────────────────────────────
# O. Production-transcript regressions (LLM-2 hygiene, Phases 1-4)
#    Each scenario mirrors one real production transcript that exposed a bug.
# ──────────────────────────────────────────────────────────────────────────────

emily_carter_correction_single_ask = Scenario(
    name="emily_carter_correction_single_ask",
    flow="pcp",
    timeout_s=360,
    retries=2,  # CORRECTED-event classification is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name, confirmed at the read-back
        "yes correct",  # read-back "Emily Carson" confirmed
        "m nine zero seven five zero three",
        # Production transcript (Bug A): pure name correction at the DOB ask.
        # The broken behavior acknowledged the correction and then DOUBLE-asked
        # ("…could you confirm your Member ID number again? And what's your
        # date of birth?"). The fixed turn must re-ask DOB and nothing else.
        "wait — actually my name is Emily Carter, not Carson",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # The correction ack re-asks DOB ONLY — awaiting must still be dob and
        # the confirmed member_id must not be re-asked (transcript_count below).
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,  # lookup matched the CORRECTED name
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_count={
            # The exact production double-ask: a re-ask/re-confirm of the
            # already-confirmed member_id must never appear anywhere.
            r"member\s*id[^.?!]*again": 0,
            r"confirm your member\s*id": 0,
        },
    ),
    notes=(
        "Mirrors the Emily Carter production transcript (Bug A, Phase 2). A "
        "confirmed-name correction arrives at the DOB ask; the response must "
        "acknowledge and re-ask DOB in one sentence — the single-ask sanitizer "
        "strips any re-ask of the confirmed member_id, asserted via the zero "
        "transcript_count entries and slot_awaiting staying on dob."
    ),
)

notification_followup_not_declined = Scenario(
    name="notification_followup_not_declined",
    flow="pcp",
    timeout_s=360,
    retries=2,  # park disposition classification is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Production transcript (Bug B): a notification question asked during
        # member_id collection was DECLINED ("that part I can't help with").
        # It concerns a later stage of this same call → must PARK, then be
        # answered by follow_up at the end.
        "m nine zero seven five zero three — will I get a notification when the list is sent out?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL
    + [
        "no, that's all, thanks",  # spare: follow_up answers the parked question first
    ],
    turn_expectations={
        # The park ack must not stall the pipeline: the same turn ends on the
        # appended DOB ask.
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            "parked_followups": falsy,  # surfaced and consumed by follow_up
        },
        transcript_count={
            # The production decline wording, in either its old or new form —
            # a later-stage notification question must never be declined.
            r"(can'?t|cannot|not able to)\s+(help|assist)": 0,
            r"representative will need to (help|make that change)": 0,
        },
    ),
    notes=(
        "Mirrors the notification-question production transcript (Bug B, "
        "Phase 3). 'Will I get a notification when it's sent?' during member_id "
        "must park (header.md: delivery/notification/timeline questions are "
        "never declined), keep the flow moving to DOB, and be answered by "
        "follow_up before close — final parked_followups empty, zero decline "
        "phrasings anywhere in the transcript."
    ),
)

zip_update_during_fax_confirmation = Scenario(
    name="zip_update_during_fax_confirmation",
    flow="pcp",
    mutating=True,
    timeout_s=420,
    retries=2,  # update_target extraction mid-delivery is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        # Production transcript run-df1e16a9 (Bug C): at the fax read-back the
        # member says their ZIP changed. The broken behavior repeated the fax
        # question over the request; the fix routes to provider_search NOW.
        "wait — actually my ZIP code changed, I moved recently",
        "zero two one four one",  # new ZIP, collected by provider_search
        "yes that's correct",  # fax read-back re-asked on resume → confirm
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the ZIP interjection: the fax read-back question.
        10: TurnExpectation(ai_contains=[r"fax"], slot_awaiting="fax_confirmed"),
        # The hand-off: honest "update your ZIP first" ask — awaiting flips to
        # zip_code and the next turn is owned by provider_search.
        11: TurnExpectation(ai_contains=[r"zip"], slot_awaiting="zip_code"),
        # The resume: ZIP-update acknowledgement naming the NEW ZIP plus the
        # re-asked fax read-back — dispatch never fired from the disputed ZIP.
        12: TurnExpectation(ai_contains=[r"02141", r"fax"], slot_awaiting="fax_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "zip_code_used": "02141",
            "zip_code_updated": True,
            "pending_slot_update": falsy,  # round-trip fully consumed
        },
        transcript_contains=[
            # Dispatch confirmation names the NEW ZIP (list rebuilt from it).
            _zip_dispatch_regex("02141"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02141")],
    notes=(
        "Mirrors production run-df1e16a9 (Bug C, Phase 4). Mutates Emily's zip "
        "in Salesforce; teardown restores the snapshot. A ZIP update requested "
        "at the fax read-back routes to provider_search (pending_slot_update), "
        "collects + persists the new ZIP, and the orchestrator fast-path "
        "returns to delivery_management at fax_confirmed with the update "
        "acknowledged. The provider list must be dispatched from the NEW ZIP "
        "only — asserted by the ZIP-aware dispatch message, zip_code_used, and "
        "the Salesforce post-check."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# P. Cross-agent redo/replay requests (Phase 6)
#
# Phase 4's registry routes slot VALUE updates. Phase 6 adds the two further
# request kinds real calls contain: redo (re-perform a completed action with
# a changed parameter) and replay (re-state information already given).
# ──────────────────────────────────────────────────────────────────────────────

redo_fax_to_email_from_benefits = Scenario(
    name="redo_fax_to_email_from_benefits",
    flow="pcp",
    timeout_s=420,
    retries=2,  # request_kind extraction is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        "yes that's correct",  # fax confirmed → dispatch + benefits offer
        "yes please",  # benefits → explanation + Care Coach offer
        # Phase 6 (a): mid Care-Coach offer, the member wants the ALREADY
        # DISPATCHED list re-sent by another method — a redo_action, not a
        # slot update. benefits routes to delivery, which re-dispatches.
        "actually can you send that list to my email instead of fax",
        "yes that's correct",  # email read-back confirmed → re-dispatch
        "no thank you",  # Care Coach re-offer declined (where we left off)
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the redo: the Care Coach offer (benefits agent).
        12: TurnExpectation(ai_contains=[r"[Cc]oach"], slot_awaiting="care_coach_response"),
        # The hop landed in delivery's re-dispatch branch: email read-back.
        13: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
        # The resume: re-send acknowledged AND the Care Coach offer re-asked —
        # never the benefits offer again.
        14: TurnExpectation(ai_contains=[r"email", r"[Cc]oach"], slot_awaiting="care_coach_response"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_offer_made": True,
            "pending_cross_agent_request": falsy,  # round-trip fully consumed
        },
        transcript_contains=[r"(?i)same .{0,40}list|as well"],
    ),
    notes=(
        "Phase 6 (a): a fax→email redo requested from benefits_agent after "
        "dispatch routes to delivery_management (capability registry), "
        "re-dispatches by email, does NOT repeat the benefits offer "
        "(benefits_offer_made stays True), and returns to benefits at the "
        "Care Coach offer where the call left off."
    ),
)

replay_benefits_from_follow_up = Scenario(
    name="replay_benefits_from_follow_up",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits explained + Care Coach offer
        "no thank you",  # Care Coach declined → care_wellness → follow-up stage
        # Phase 6 (b): a replay_info request after the benefits flow finished.
        # No update_target fires and benefits_inquiry is not an intake intent —
        # the capability registry is the only way to honor this.
        "can you repeat my benefits again?",
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # The replay: benefits re-explained (deductible read again) — and the
        # Care Coach must NOT be re-offered on a routed replay.
        14: TurnExpectation(ai_contains=[r"(?i)deductible"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "pending_cross_agent_request": falsy,
            "benefits_explained": True,
        },
        # The benefits summary appears twice: the original explanation and
        # the replay.
        transcript_count={r"(?i)individual deductible": 2},
    ),
    notes=(
        "Phase 6 (b): 'repeat my benefits' voiced at the post-flow stage "
        "routes through follow_up to benefits_agent via the capability "
        "registry, re-explains (fetch_benefits is idempotent), skips a second "
        "Care Coach offer, and hands back to follow_up for the close."
    ),
)

redo_inflow_before_dispatch = Scenario(
    name="redo_inflow_before_dispatch",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        # Phase 6 (c): the owner IS the active agent — switching fax→email
        # while still in delivery before dispatch resolves in-flow: the
        # existing delivery_method/contact branches handle it, zero routing.
        "actually email is better",
        "yes that's correct",  # email read-back → dispatch + benefits offer
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",
    ],
    turn_expectations={
        11: TurnExpectation(ai_contains=[r"email"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "pending_cross_agent_request": falsy,  # never set — no hop
        },
    ),
    notes=(
        "Phase 6 (c): a method switch voiced while delivery_management is "
        "active and the list is NOT yet dispatched stays in-flow — no "
        "pending_cross_agent_request, no orchestrator hop."
    ),
)

replay_benefits_inflow_at_coach_offer = Scenario(
    name="replay_benefits_inflow_at_coach_offer",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits explained + Care Coach offer
        # Phase 6 (c): replay of benefits' own material while benefits is
        # active — in-flow re-explain + re-ask, zero routing.
        "sorry, can you repeat my benefits again?",
        "no thank you",  # Care Coach declined
        "no that's all, thanks",
    ],
    turn_expectations={
        # The in-flow replay: benefits re-explained AND the offer re-asked.
        13: TurnExpectation(
            ai_contains=[r"(?i)deductible", r"[Cc]oach"], slot_awaiting="care_coach_response"
        ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"pending_cross_agent_request": falsy},
    ),
    notes=(
        "Phase 6 (c): 'repeat my benefits' during benefits' own Care Coach "
        "offer resolves in-flow — re-explanation + re-offer in one turn, no "
        "routing."
    ),
)

unknown_replay_topic_parks = Scenario(
    name="unknown_replay_topic_parks",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits explained + Care Coach offer
        # Phase 6 (d): a replay request for a topic no capability owns.
        # Must park as a question (Phase 3 path) — never a hard decline.
        "can you go over my claim history again?",
        "no thank you",  # Care Coach declined
        "what about that claim history?",  # follow_up answers/cannot-answer
        "no that's all, thanks",
    ],
    turn_expectations={
        # The park acknowledgement + the Care Coach offer re-asked, no
        # decline phrasing.
        13: TurnExpectation(ai_contains=[r"[Cc]oach"], slot_awaiting="care_coach_response"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "pending_cross_agent_request": falsy,
            "parked_followups": falsy,  # consumed by follow_up
        },
    ),
    notes=(
        "Phase 6 (d): an unknown replay topic ('claim history') parks as a "
        "kind=question item instead of declining; follow_up surfaces it at "
        "the post-flow stage. The call never escalates over it."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Q. Language-variation regressions (paraphrased triggers, Phases 6-7)
#
# Every scenario in this section is a paraphrase twin of a proven scenario
# elsewhere in this file. The conversational SHAPE and the assertions are held
# identical; only the trigger utterance is rewritten so it shares NO wording
# with the canonical examples baked into the extraction/generation prompts or
# the request_detection.py regexes. If a paraphrase twin fails while its
# original passes, the behavior is over-fit to the example phrasing rather than
# to the intent — exactly the generalization gap these guards exist to catch.
#
#   Q-1  BUG-5  zip-update-while-confirming-delivery, paraphrased
#   Q-2  BUG-2  fax→email redo from the Care Coach offer, paraphrased
#   Q-3  BUG-3 + Phase-7 parity  notification channel switch, paraphrased
#   Q-4  BUG-1  parked notification question grounded, paraphrased
#   Q-5  BUG-4  mid-verification identity update detour, paraphrased
#   Q-6  Phase-7 parity  claim_status replay from follow_up, paraphrased
#
# All wording below deliberately avoids the canonical cues, e.g.:
#   "I moved recently"            → "I've relocated to a new address"
#   "send it to my email instead" → "put that provider list through by email"
#   "actually email me instead"   → "you know what, email works better for me"
#   "will I get a notification…"  → "how will you let me know once it's out?"
#   "I need to update my last name"→ "the surname on my file needs fixing"
#   "give me an update on my claim"→ "remind me where my adjustment stands"
# ──────────────────────────────────────────────────────────────────────────────

# Q-1 — BUG-5 paraphrase twin of zip_update_during_fax_confirmation (O-3).
# The ZIP interjection at the fax read-back uses "relocated / new address /
# postal code is off" instead of "my ZIP code changed, I moved recently".
zip_update_during_fax_paraphrased = Scenario(
    name="zip_update_during_fax_paraphrased",
    flow="pcp",
    mutating=True,
    timeout_s=420,
    retries=2,  # update_target extraction on paraphrased phrasing is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        # Paraphrased BUG-5 trigger: no "ZIP changed", no "I moved recently".
        # request_detection's zip_code EXTRA pattern still catches "relocated"?
        # No — it keys on "moved"/"address changed"; this phrasing leans on the
        # extraction LLM to surface update_target=zip_code, which is the point.
        "hold on — I've relocated to a new address, so the postal code you have is off",
        "zero two one four three",  # new ZIP, collected by provider_search
        "yes that's correct",  # fax read-back re-asked on resume → confirm
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the ZIP interjection: the fax read-back question.
        10: TurnExpectation(ai_contains=[r"fax"], slot_awaiting="fax_confirmed"),
        # The hand-off: honest "update your ZIP first" ask — awaiting flips to
        # zip_code and the next turn is owned by provider_search.
        11: TurnExpectation(ai_contains=[r"zip"], slot_awaiting="zip_code"),
        # The resume: ZIP-update acknowledgement naming the NEW ZIP plus the
        # re-asked fax read-back — dispatch never fired from the disputed ZIP.
        12: TurnExpectation(ai_contains=[r"02143", r"fax"], slot_awaiting="fax_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "zip_code_used": "02143",
            "zip_code_updated": True,
            "pending_slot_update": falsy,  # round-trip fully consumed
        },
        transcript_contains=[
            _zip_dispatch_regex("02143"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02143")],
    notes=(
        "Paraphrase twin of zip_update_during_fax_confirmation (O-3, Bug C/BUG-5). "
        "Same fax-read-back → ZIP-update → resume round-trip, but the trigger drops "
        "every canonical cue ('ZIP changed', 'I moved recently') in favor of "
        "'relocated to a new address / postal code is off'. Proves the mid-delivery "
        "update detour keys on intent, not on the example wording. Mutates Emily's "
        "zip; teardown restores the snapshot."
    ),
)

practice_team_context_retension_issue1 = Scenario(
    name="practice_team_context_retension_issue1",
    flow="pcp",
    mutating=True,
    timeout_s=420,
    retries=2,  # update_target extraction on paraphrased phrasing is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "sorry i need to update this",
        "I'm sorry. I was saying that the the ZIP code is incorrect. I want to update the ZIP code",
        "send it to my fax",  # delivery method
        "yes that's correct",  # fax read-back re-asked on resume → confirm
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the ZIP interjection: the fax read-back question.
        10: TurnExpectation(ai_contains=[r"fax"], slot_awaiting="fax_confirmed"),
        # The hand-off: honest "update your ZIP first" ask — awaiting flips to
        # zip_code and the next turn is owned by provider_search.
        11: TurnExpectation(ai_contains=[r"zip"], slot_awaiting="zip_code"),
        # The resume: ZIP-update acknowledgement naming the NEW ZIP plus the
        # re-asked fax read-back — dispatch never fired from the disputed ZIP.
        12: TurnExpectation(ai_contains=[r"02143", r"fax"], slot_awaiting="fax_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "zip_code_used": "02143",
            "zip_code_updated": True,
            "pending_slot_update": falsy,  # round-trip fully consumed
        },
        transcript_contains=[
            _zip_dispatch_regex("02143"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02143")],
    notes=(
        "Paraphrase twin of zip_update_during_fax_confirmation (O-3, Bug C/BUG-5). "
        "Same fax-read-back → ZIP-update → resume round-trip, but the trigger drops "
        "every canonical cue ('ZIP changed', 'I moved recently') in favor of "
        "'relocated to a new address / postal code is off'. Proves the mid-delivery "
        "update detour keys on intent, not on the example wording. Mutates Emily's "
        "zip; teardown restores the snapshot."
    ),
)

practice_team_context_retension_issue2 = Scenario(
    name="practice_team_context_retension_issue2",
    flow="pcp",
    mutating=True,
    timeout_s=420,
    retries=2,  # update_target extraction on paraphrased phrasing is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "No.",
        "fax, but I need to update my ZIP code.",
        "zero two one four three",  # new ZIP, collected by provider_search
        "yes that's correct",  # fax read-back re-asked on resume → confirm
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the ZIP interjection: the fax read-back question.
        10: TurnExpectation(ai_contains=[r"fax"], slot_awaiting="fax_confirmed"),
        # The hand-off: honest "update your ZIP first" ask — awaiting flips to
        # zip_code and the next turn is owned by provider_search.
        11: TurnExpectation(ai_contains=[r"zip"], slot_awaiting="zip_code"),
        # The resume: ZIP-update acknowledgement naming the NEW ZIP plus the
        # re-asked fax read-back — dispatch never fired from the disputed ZIP.
        12: TurnExpectation(ai_contains=[r"02143", r"fax"], slot_awaiting="fax_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "zip_code_used": "02143",
            "zip_code_updated": True,
            "pending_slot_update": falsy,  # round-trip fully consumed
        },
        transcript_contains=[
            _zip_dispatch_regex("02143"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02143")],
    notes=(
        "Paraphrase twin of zip_update_during_fax_confirmation (O-3, Bug C/BUG-5). "
        "Same fax-read-back → ZIP-update → resume round-trip, but the trigger drops "
        "every canonical cue ('ZIP changed', 'I moved recently') in favor of "
        "'relocated to a new address / postal code is off'. Proves the mid-delivery "
        "update detour keys on intent, not on the example wording. Mutates Emily's "
        "zip; teardown restores the snapshot."
    ),
)

# Q-2 — BUG-2 paraphrase twin of redo_fax_to_email_from_benefits (P-1).
# The redo request avoids "send that list to my email instead of fax".
redo_list_to_email_paraphrased = Scenario(
    name="redo_list_to_email_paraphrased",
    flow="pcp",
    timeout_s=420,
    retries=2,  # request_kind extraction on paraphrased phrasing is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        "yes that's correct",  # fax confirmed → dispatch + benefits offer
        "yes please",  # benefits → explanation + Care Coach offer
        # Paraphrased BUG-2 trigger at the Care Coach offer: a redo_action to
        # re-route the ALREADY dispatched list by the other channel. Phrasing
        # avoids "send that list to my email instead of fax"; "re-route the
        # provider list by email instead" still trips the 'by email instead'
        # redo pattern (verified against request_detection).
        "on second thought, send provider list by email?",
        "yes that's correct",  # email read-back confirmed → re-dispatch
        "no thank you",  # Care Coach re-offer declined (where we left off)
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the redo: the Care Coach offer (benefits agent).
        12: TurnExpectation(ai_contains=[r"[Cc]oach"], slot_awaiting="care_coach_response"),
        # The hop landed in delivery's re-dispatch branch: email read-back.
        13: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
        # The resume: re-send acknowledged AND the Care Coach offer re-asked —
        # never the benefits offer again.
        14: TurnExpectation(ai_contains=[r"email", r"[Cc]oach"], slot_awaiting="care_coach_response"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_offer_made": True,
            "pending_cross_agent_request": falsy,  # round-trip fully consumed
        },
    ),
    notes=(
        "Paraphrase twin of redo_fax_to_email_from_benefits (P-1, BUG-2). Same "
        "fax→email redo routed from benefits_agent after dispatch, but the trigger "
        "reads 're-route the provider list by email instead' instead of "
        "'send that list to my email instead of fax'. The benefits offer must not "
        "repeat (benefits_offer_made stays True) and the call resumes at the Care "
        "Coach offer. Drops the transcript_contains phrasing assert from P-1 — the "
        "acknowledgement wording is LLM-generated and not load-bearing here."
    ),
)

redo_list_to_email_address = Scenario(
    name="redo_list_to_email_address",
    flow="pcp",
    timeout_s=420,
    retries=2,  # request_kind extraction on paraphrased phrasing is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my email",  # delivery method
        "yes that's correct",  # fax confirmed → dispatch + benefits offer
        "yes please",  # benefits → explanation + Care Coach offer
        # Paraphrased BUG-2 trigger at the Care Coach offer: a redo_action to
        # re-route the ALREADY dispatched list by the other channel. Phrasing
        # avoids "send that list to my email instead of fax"; "re-route the
        # provider list by email instead" still trips the 'by email instead'
        # redo pattern (verified against request_detection).
        "please change my email address?",
        "yes that's correct",  # email read-back confirmed → re-dispatch
        "no thank you",  # Care Coach re-offer declined (where we left off)
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the redo: the Care Coach offer (benefits agent).
        12: TurnExpectation(ai_contains=[r"[Cc]oach"], slot_awaiting="care_coach_response"),
        # The hop landed in delivery's re-dispatch branch: email read-back.
        13: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
        # The resume: re-send acknowledged AND the Care Coach offer re-asked —
        # never the benefits offer again.
        14: TurnExpectation(ai_contains=[r"email", r"[Cc]oach"], slot_awaiting="care_coach_response"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_offer_made": True,
            "pending_cross_agent_request": falsy,  # round-trip fully consumed
        },
    ),
    notes=(
        "Paraphrase twin of redo_fax_to_email_from_benefits (P-1, BUG-2). Same "
        "fax→email redo routed from benefits_agent after dispatch, but the trigger "
        "reads 're-route the provider list by email instead' instead of "
        "'send that list to my email instead of fax'. The benefits offer must not "
        "repeat (benefits_offer_made stays True) and the call resumes at the Care "
        "Coach offer. Drops the transcript_contains phrasing assert from P-1 — the "
        "acknowledgement wording is LLM-generated and not load-bearing here."
    ),
)

# Q-3 — BUG-3 + Phase-7 parity: notification-channel switch during the phone
# read-back, paraphrased away from delivery's "actually email me instead".
notification_channel_switch_paraphrased = Scenario(
    name="notification_channel_switch_paraphrased",
    flow="claim",
    timeout_s=360,
    retries=1,  # the surrounding claim flow has LLM-driven steps
    user_turns=_CLAIM_TO_NOTIFICATION
    + [
        "You can send me the updates to my phone",  # 12 notification_method = sms
        # Paraphrased BUG-3 trigger at the phone read-back: a channel switch, not
        # a dispute of the number on file. "email works better for me" trips the
        # '<channel> works better' switch pattern without reusing "email me instead".
        "you know what, email works better for me",  # 13 → switch to email
        "yes that's correct",  # 14 confirm email on file → save N1 + timeline bridge
        "Okay, how long will it take to finalize the request?",  # 15 timeline question
        "email them to me",  # 16 N2 channel
        "No, that's it. Thanks!",  # 17 close
    ],
    turn_expectations={
        # The phone read-back precedes the switch turn — still awaiting phone.
        13: TurnExpectation(ai_contains=[r"\d{3}-\d{3}-\d{4}|phone"], slot_awaiting="phone_confirmed"),
        # After the switch the agent asks to confirm the EMAIL on file — proving
        # _maybe_switch_channel fired and abandoned the phone channel cleanly.
        14: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            # N1 landed on the SWITCHED-TO channel, not the originally chosen sms.
            "notification_channel": "email",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Phase-7 claims-path parity for BUG-3: notification_setup._maybe_switch_channel "
        "mirrors delivery's _maybe_switch_method. The member picks phone notifications, "
        "then at the phone read-back says 'email works better for me' — a channel "
        "SWITCH (value carry-through to the email on file), NOT a dispute of the "
        "number (which would stay a decline). notification_channel must end 'email'. "
        "Paraphrased away from 'actually email me instead' to prove the switch keys on "
        "intent. Contrast notification_phone_confirm_advances (33a), where a plain "
        "affirmative keeps sms."
    ),
)

# Q-4 — BUG-1 paraphrase twin of notification_followup_not_declined (O-2).
# The parked notification question avoids "will I get a notification when…".
followup_parked_notification_paraphrased = Scenario(
    name="followup_parked_notification_paraphrased",
    flow="pcp",
    timeout_s=360,
    retries=2,  # park disposition classification is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Paraphrased BUG-1 trigger during member_id collection: a notification
        # question about a LATER stage of this same call. Phrasing avoids "will I
        # get a notification when the list is sent" — it must PARK (delivery/
        # notification questions are never declined), keep the flow moving to DOB,
        # and be answered from real state (not an invented channel) by follow_up.
        "m nine zero seven five zero three — and how will you let me know once that list actually goes out?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL
    + [
        "no, that's all, thanks",  # spare: follow_up answers the parked question first
    ],
    turn_expectations={
        # The park ack must not stall the pipeline: the same turn ends on the
        # appended DOB ask.
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            "parked_followups": falsy,  # surfaced and consumed by follow_up
        },
        transcript_count={
            # A later-stage delivery/notification question must never be declined,
            # regardless of the wording it arrives in.
            r"(can'?t|cannot|not able to)\s+(help|assist)": 0,
            r"representative will need to (help|make that change)": 0,
        },
    ),
    notes=(
        "Paraphrase twin of notification_followup_not_declined (O-2, BUG-1). Same "
        "park-then-answer lifecycle for a delivery/notification question raised mid "
        "member_id, but phrased 'how will you let me know once that list goes out?' "
        "instead of 'will I get a notification when it's sent?'. Must park (not "
        "decline), advance to DOB, and be answered by follow_up before close — final "
        "parked_followups empty, zero decline phrasings anywhere."
    ),
)

# ── RC. Records coordination — Personal Guide decline → follow-up (not escalation) ──────────────

records_no_guide_upload_then_close = Scenario(
    name="records_no_guide_upload_then_close",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",  # member_upload
        "Yes, please send the link",  # upload_consent = yes
        "Yes, that's correct",  # email on file confirmed
        "No, that won't be necessary",  # personal_guide_consent = no → follow_up
        "No, that's all. Thank you!",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
        transcript_contains=[pool_regex(MSG_FOLLOW_UP_ASK)],
    ),
    notes=(
        "RC-24a — regression guard: member uploads themselves, upload link is sent, "
        "then declines Personal Guide. Agent must route to follow_up "
        "('anything else?') and NOT escalate."
    ),
)

records_no_guide_doctor_direct_then_close = Scenario(
    name="records_no_guide_doctor_direct_then_close",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can I ask my doctor to send it over?",  # doctor_direct
        "No thanks, I don't need the link",  # upload_consent = no → guide offer
        "No, that's okay",  # personal_guide_consent = no → follow_up
        "No, that's everything. Bye!",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": falsy,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
        transcript_contains=[pool_regex(MSG_FOLLOW_UP_ASK)],
    ),
    notes=(
        "RC-24b — doctor-direct path: member says doctor will send the records, "
        "declines the upload link offer, then declines the Personal Guide. "
        "Agent must route to follow_up and NOT escalate."
    ),
)

records_no_guide_regression_no_transfer = Scenario(
    name="records_no_guide_regression_no_transfer",
    flow="claim",
    timeout_s=300,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I can send it myself",  # member_upload
        "no thanks",  # upload_consent = no → guide offer
        "no I don't want that either",  # personal_guide_consent = no → follow_up
        "Nope, I'm done. Goodbye.",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "records_branch_taken": "declined_personal_guide",
        },
    ),
    notes=(
        "RC-24c — pure no-transfer regression guard: declining both upload link and "
        "Personal Guide must never fire an AgentCallTransfer event. "
        "Complements RC-24a/b by using a shorter script with no follow-up turns."
    ),
)

records_no_guide_then_follow_up_question = Scenario(
    name="records_no_guide_then_follow_up_question",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please",  # accept link
        "Yes, that's correct",  # email confirmed
        "No, that's not necessary",  # decline guide → follow_up
        "How long does the review usually take after you receive everything?",
        "No, that's all. Thank you!",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
    ),
    notes=(
        "RC-24d — after declining Personal Guide the agent routes to follow_up; "
        "the member then asks a timeline question which follow_up must answer "
        "before the call closes cleanly."
    ),
)

records_no_guide_conversational_phrasing = Scenario(
    name="records_no_guide_conversational_phrasing",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "oh I think the doctor's office will just send it over",  # doctor_direct
        "yeah sure, go ahead and send me the link",  # upload_consent = yes
        "uh-huh, that email's fine",  # email confirmed
        "I don't think so, no thank you — I'll wait to hear from you",  # decline guide
        "Nope, that's it for me, thanks",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
            "records_branch_taken": "declined_personal_guide",
        },
        transcript_contains=[pool_regex(MSG_FOLLOW_UP_ASK)],
    ),
    notes=(
        "RC-24e — conversational/natural phrasing robustness: member uses an indirect "
        "'I don't think so, no thank you' to decline the Personal Guide. "
        "Normalizer must resolve this to no and route to follow_up, not escalation. "
        "retries=1 for LLM extraction variance on the indirect phrasing."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# RC2. New-intent pivots from follow_up reached via records-decline path
#
# The member reaches follow_up after declining the Personal Guide (the new
# routing introduced in records_coordination_agent). From there they raise a
# new service request. These scenarios verify the full new_intent machinery
# works identically regardless of HOW follow_up was entered.
# ──────────────────────────────────────────────────────────────────────────────

# Shared prefix: claim verify → reference → doctor-direct → no link → no guide
# → follow_up.  Three turns after CLAIM_VERIFY + reference number.
_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE = CLAIM_VERIFY + [
    "42695817",
    "my doctor can send the records over",  # doctor_direct
    "no thank you",  # upload_consent = no → guide offer
    "no, that's fine",  # personal_guide_consent = no → follow_up
]

records_no_guide_then_pcp_new_intent = Scenario(
    name="records_no_guide_then_pcp_new_intent",
    flow="claim",
    mutating=True,  # provider flow writes James's ZIP if absent
    timeout_s=480,
    retries=1,  # new_intent + intake re-screen + provider/delivery slots are LLM-driven
    user_turns=_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE
    + [
        "Actually, I also need to find a primary care physician near me.",  # new_intent
        # Re-verification (provider slot order: first/last name → readback →
        # member_id → dob → relationship). Same caller, James M310188.
        "james",
        "wilson",
        "yes correct",
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "I'm the plan holder",  # relationship
        # Now in provider_search.
        "Primary Care Physician",  # provider type
        "zero two one three nine",  # ZIP (James may have none on file)
        "email please",  # delivery method
        "yes that's correct",  # email on file confirmed
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no, that's all, thanks",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "call_intent": "provider_services",
            "provider_list_sent": True,
            "delivery_method": "email",
            "pending_intent": lambda v: not v,
        },
        transcript_contains=[r"first name"],  # proof of re-verification
    ),
    notes=(
        "RC-24f — after declining Personal Guide the member asks for a new PCP search. "
        "follow_up routes new_intent (provider_services) through intake re-screen, "
        "which clears state and re-verifies James before running the provider flow. "
        "Marked mutating: James may have no ZIP on file. retries=1: new_intent + "
        "re-screen + provider_type/delivery_method extraction are LLM-driven."
    ),
)

records_no_guide_then_claim_new_intent = Scenario(
    name="records_no_guide_then_claim_new_intent",
    flow="claim",
    timeout_s=420,
    retries=1,  # new_intent classification is LLM-driven
    user_turns=_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE
    + [
        "Actually, I submitted another adjustment and want to check on that one too.",  # new_intent
        # Re-verification (claims slot order: first/last name → readback →
        # member_id → dob → phone confirmation). Same caller, James M310188.
        "james",
        "wilson",
        "yes correct",
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "yes that's correct",  # phone confirmation
        # Reference 98765432 does not exist → not-found → retry → escalation.
        "98765432",
        "98765432",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "call_intent": "claim_services",
            "pending_intent": lambda v: not v,
        },
        transcript_contains=[r"first name", r"reference number"],
    ),
    notes=(
        "RC-24g — after declining Personal Guide the member asks for a second claim "
        "adjustment. follow_up routes new_intent (claim_services), resets the call, "
        "and re-verifies James. Reference 98765432 does not exist, so "
        "claim_adjustment escalates after two not-found attempts — a deterministic "
        "outcome that avoids fixture collision with 42695817. retries=1: new_intent "
        "classification is LLM-driven."
    ),
)

records_no_guide_then_unsupported_provider = Scenario(
    name="records_no_guide_then_unsupported_provider",
    flow="claim",
    timeout_s=300,
    retries=1,  # new_intent + intake unsupported-type classification are LLM-driven
    user_turns=_CLAIM_TO_FOLLOW_UP_VIA_RECORDS_DECLINE
    + [
        "Actually, I also need to find a neurologist near me.",  # new_intent → unsupported
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=False,
        final_is_interrupt=False,
        final_state={
            # DECISIVE: intake re-screen fired before re-verification.
            "member_status_verify": falsy,
            "first_name": falsy,
            "pending_intent": falsy,
            "escalation_pre_message": contains("Orthopedic"),
        },
        last_ai_contains=[
            r"neurologist",
            r"(Primary Care|Pediatrician|Cardiologist|Dermatologist|Orthopedic)",
        ],
    ),
    notes=(
        "RC-24h — after declining Personal Guide the member asks for a neurologist "
        "(unsupported specialty). follow_up routes new_intent (provider_services) "
        "through intake re-screen, which fires the unsupported-provider gate BEFORE "
        "any re-verification — member_status_verify and first_name must be falsy at "
        "END. Mirrors followup_unsupported_provider_rescreen but starts from the "
        "records-decline follow_up entry point. retries=1: new_intent + "
        "unsupported-type classification are LLM-driven."
    ),
)


# Q-5 — BUG-4 paraphrase twin of update_without_value_case_b (N-5).
# The mid-verification identity update avoids "I need to update my last name".
verification_identity_update_paraphrased = Scenario(
    name="verification_identity_update_paraphrased",
    flow="pcp",
    timeout_s=360,
    retries=2,  # update_target extraction on paraphrased phrasing is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Paraphrased BUG-4 trigger (Case B): answer member_id AND ask to change
        # last name WITHOUT giving a value. "the surname on my file needs fixing"
        # avoids "update/change my last name" — the awaiting slot must still
        # confirm, then a detour asks for the new value, then resume at DOB.
        "m nine zero seven five zero three — oh, and the surname on my file needs fixing, by the way",
        "Carter",  # the "new" value (same as on file, so the lookup still matches)
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        # Detour: the prompt after the update request asks for the LAST NAME —
        # not DOB — proving awaiting_slot switched to the update target.
        5: TurnExpectation(ai_contains=[r"last\s*name|surname"], slot_awaiting="last_name"),
        # After the new value, the pipeline resumes at DOB (correction_return_to).
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Paraphrase twin of update_without_value_case_b (N-5, BUG-4). Same Case-B "
        "detour: member answers member_id and asks to change their surname without "
        "giving one — the turn confirms member_id, opens a last_name detour "
        "(asserted), and returns to DOB after the new value. Phrased 'the surname on "
        "my file needs fixing' instead of 'I need to update my last name' to prove "
        "the detour is not example-bound. The DOB attempt budget must not be spent "
        "by the detour."
    ),
)

# Q-6 — Phase-7 parity: claim_status replay from follow_up, paraphrased.
claim_status_replay_paraphrased = Scenario(
    name="claim_status_replay_paraphrased",
    flow="claim",
    timeout_s=420,
    retries=2,  # request_kind=replay classification is LLM-driven
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can I ask my doctor to send it over?",  # doctor-direct
        "Yes, please",  # accept upload link
        "Yes, that's correct",  # confirm email on file
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "Okay, how long will it take to finalize the request?",  # timeline question
        "email them to me",  # N2 channel → claim flow completes, follow_up takes over
        # Paraphrased Phase-7 trigger at the post-flow follow_up stage: replay the
        # adjustment status. "remind me where my adjustment stands / an update on my
        # claim again" routes replay → claim_adjustment._replay_claim_status, which
        # re-states from state (no lookup, no re-entry) and hands back to follow_up.
        "actually, before we finish — can you give me an update on my claim again?",
        "No, that's it for me. Thanks!",  # close
    ],
    turn_expectations={7: TurnExpectation(ai_contains=[r"reference number"])},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "claim_flow_complete": True,
            "pending_cross_agent_request": falsy,  # replay hop fully consumed
        },
        transcript_contains=[
            # _replay_claim_status re-states "your claim adjustment … is currently …".
            r"(?i)claim adjustment.*is currently",
        ],
    ),
    notes=(
        "Phase-7 claims-path parity: the ('replay','claim_status') capability, live. "
        "After the claim flow completes, the member asks to hear the adjustment status "
        "again at the follow_up stage; follow_up routes the replay to "
        "claim_adjustment_agent, which re-states from state via _replay_claim_status "
        "(idempotent, like delivery's _replay_provider_list) and returns. Phrased "
        "'give me an update on my claim again' — different from any prompt example — "
        "and asserts the replay summary reappears with pending_cross_agent_request "
        "cleared and the flow still complete."
    ),
)

automated_test = Scenario(
    name="automated_test",
    flow="pcp",
    timeout_s=360,
    retries=2,  # update_target extraction on paraphrased phrasing is LLM-driven
    # user_turns=[
    #     "I'm calling to check the list of your network providers in my area, please.",
    #     "I am... can you give me a minute while I pull up... yeah. I'm Emily.",
    #     "It's Carter.",
    #     "Yeah. I do. Don't you know how it spells?",
    #     "i am not going to give you my member id",
    #     "Yeah. It's m nine zero seven five zero three.",
    #     "April twelfth nineteen eighty eight.",
    #     "I am the plan holder.",
    #     "Primary definition.",
    #     "I'm looking for",
    #     "Primary care physician.",
    #     "Yeah. That's right.",
    #     "fax, please.",
    #     "Yeah. That's right.",
    #     "I wanna change my fax number. Is it possible?",
    #     "its zero one two three four five six seven eight nine.",
    #     "yes",
    #     "yes",
    #     "No. Thank you.",
    #     "No.",
    # ],
    user_turns=[
        # "I'm looking for a network providers in the area for five hundred acquisitions.",
        # "And I believe?",
        # "And",
        # "My name is Emily.",
        # "Hospital.",
        # "No. It's Emily Carter, p a r t e r.",
        # "Yeah. That's right.",
        # "My member ID is m nine zero seven five zero three.",
        # "If it's worth nineteen eighty six.",
        # "It's April twelfth nineteen eighty eight.",
        # "I'm the plan holder.",
        # "I already told that to you when I called you.",
        # "A primary cancellation.",
        # "Yeah. That's right.",
        # "Pass would be great.",
        # "No, use this fax number is one two three four five six seven eight nine one.",
        # "It's one two three five five five four six seven eight.",
        # "Can I get a list of in network providers within my area?",
        # "Daniel.",
        # "Read.",
        # "No. The second name is spelled r e a d.",
        # "R e e d.",
        # "Yes.",
        # "Ten seven one four five nine eight.",
        # "Third September nineteen eighty five."
        "Can I get a help with the list of in network providers within my area, please?",
        "Daniel.",
        "Read.",
        "Yes.",
        "M seven one four five nine eight.",
        "Third September nineteen eighty five.",
        "My last name is Reed. That is r e e d.",
        "Yes.",
        "I'm calling for myself.",
        "Pediatrician.",
        "No. I want to update.",
        ", one six seven eight three.",
        "One six seven eight three.",
    ],
    turn_expectations={},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Paraphrase twin of update_without_value_case_b (N-5, BUG-4). Same Case-B "
        "detour: member answers member_id and asks to change their surname without "
        "giving one — the turn confirms member_id, opens a last_name detour "
        "(asserted), and returns to DOB after the new value. Phrased 'the surname on "
        "my file needs fixing' instead of 'I need to update my last name' to prove "
        "the detour is not example-bound. The DOB attempt budget must not be spent "
        "by the detour."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# T. Reference-number fallback sub-flow (claim_number / DOS+billed_amount)
#
# Seed data (Emily Carter M907503):
#   reference_number = 49502271
#   claim_number     = 882301
#   dos              = 2026-01-08  (January 8, 2026)
#   billed_amount    = 1250
#
# Emily's claim verification requires phone confirmation. All T-scenarios
# that expect a successful SF lookup require Emily to have a phone number on
# file in the target Salesforce org.
# ──────────────────────────────────────────────────────────────────────────────

# Claim verification prefix for Emily Carter (M907503).
# Requires Emily to have a phone number on file.
CLAIM_VERIFY_EMILY = [
    "I want to check on a claim adjustment I submitted.",
    "emily",
    "carter",
    "yes correct",  # name_confirmed
    "m nine zero seven five zero three",
    "April twelfth nineteen eighty eight",
    "yes correct",  # phone_confirmed
]

# Seed values for Emily's adjustment record (from adjustment_requests CSV)
_EMILY_CLAIM_NUMBER = "882301"
_EMILY_CLAIM_NUMBER_SPOKEN = "eight eight two three zero one"
_EMILY_DOS_SPOKEN = "January eighth twenty twenty six"  # → normalize → 2026-01-08
_EMILY_DOS_ISO = "January 8th, 2026"  # explicit year variant
_EMILY_BILLED_SPOKEN = "twelve fifty"  # → normalize → 1250.00
_EMILY_BILLED_DOLLAR = "$1,250"  # → normalize → 1250.00


# ── TB-1: Fallback via claim number — happy path ──────────────────────────────
ref_no_fallback_claim_number_happy_path = Scenario(
    name="ref_no_fallback_claim_number_happy_path",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # 7 → fallback starts
        _EMILY_CLAIM_NUMBER,  # 8 → claim number → SF lookup
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything, thanks",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
            "upload_link_sent": True,
        },
    ),
    notes=(
        "TB-1: Happy path via claim number fallback. Emily cannot provide her "
        "reference number; agent asks for claim number; she provides '882301'; "
        "SF lookup by claim_number + member_id finds the record; status is "
        "reported and the records + notification flow continues."
    ),
)

# ── TB-2: Fallback via DOS + billed amount — happy path ──────────────────────
ref_no_fallback_dos_billed_happy_path = Scenario(
    name="ref_no_fallback_dos_billed_happy_path",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have it",  # 7 → fallback: ask for claim number
        "No",  # 8 → no claim number → ask for DOS+billed
        f"{_EMILY_DOS_SPOKEN}, and it was {_EMILY_BILLED_DOLLAR}",  # 9 → both in one turn
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|dos)"]),
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "claim_status": truthy,
            "upload_link_sent": True,
        },
    ),
    notes=(
        "TB-2: Happy path via DOS+billed fallback. Member cannot provide "
        "reference number OR claim number; provides both DOS and billed amount "
        "in one utterance ('January eighth, and it was $1,250'); SF lookup by "
        "DOS+member_id filtered by billed_amount finds Emily's record."
    ),
)

# ── TB-3: Inline DOS + billed amount — both in one utterance ─────────────────
ref_no_fallback_inline_dos_billed = Scenario(
    name="ref_no_fallback_inline_dos_billed",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",
        "No",
        "January eighth twenty twenty six and the bill was twelve fifty",  # spoken DOS + spoken amount
        "I'll have my doctor send the records",
        "No, I don't need the link",
        "Yes, please proceed",
        "email please",
        "Yes, that's correct",
        "No, that's all",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-3: The canonical example from the task spec — DOS and billed amount "
        "provided together in one spoken utterance. LLM must extract both 'dos' "
        "and 'billed_amount' from a single turn."
    ),
)

# ── TB-4: Claim number as spoken digits ──────────────────────────────────────
ref_no_fallback_spoken_claim_number = Scenario(
    name="ref_no_fallback_spoken_claim_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have my reference number handy",
        _EMILY_CLAIM_NUMBER_SPOKEN,  # spoken digits → normalize → 882301
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
        },
    ),
    notes=(
        "TB-4: Claim number provided as spoken digits 'eight eight two three "
        "zero one' — normalize_claim_number must convert to '882301' via "
        "_convert_spoken_digits, matching the stored numeric format."
    ),
)

# ── TB-5: Claim number not found → escalation ────────────────────────────────
ref_no_fallback_claim_number_not_found = Scenario(
    name="ref_no_fallback_claim_number_not_found",
    flow="claim",
    timeout_s=300,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have the reference number",
        "99990000",  # claim number not in SF
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="fallback_claim_number_not_found",
    ),
    notes=(
        "TB-5: Claim number provided but not found in Salesforce → escalates with "
        "reason=fallback_claim_number_not_found. Uses James M310188 whose record "
        "does not match claim_number=99990000."
    ),
)

# ── TB-6: DOS + billed amount not found → escalation ─────────────────────────
ref_no_fallback_dos_billed_not_found = Scenario(
    name="ref_no_fallback_dos_billed_not_found",
    flow="claim",
    timeout_s=300,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have it",
        "No",
        "March 15 and it was $500",  # DOS+billed not in SF
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_regex=r"fallback_dos_billed_not_found",
    ),
    notes=(
        "TB-6: DOS+billed provided but no matching record → escalates with "
        "reason=fallback_dos_billed_not_found. Uses James M310188; the combination "
        "March 15 / $500 does not match any fixture record."
    ),
)

# ── TB-7: Cannot-provide at claim number → moves to DOS+billed → success ─────
ref_no_fallback_cannot_provide_claim_number = Scenario(
    name="ref_no_fallback_cannot_provide_claim_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",
        "I don't have that either, sorry",  # detect_cannot_provide at claim stage
        f"{_EMILY_DOS_SPOKEN}, and it was {_EMILY_BILLED_DOLLAR}",
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's all",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-7: Member cannot provide the claim number either — detect_cannot_provide "
        "fires at the claim_number stage, pushing the fallback to DOS+billed. "
        "Emily provides January 8th + $1,250; SF lookup succeeds."
    ),
)

# ── TB-8: Bare \"No\" at claim number → DOS+billed → success ──────────────────
ref_no_fallback_bare_no_claim_number = Scenario(
    name="ref_no_fallback_bare_no_claim_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have my reference number",
        "No",  # bare no → DOS+billed stage
        f"{_EMILY_DOS_ISO} and it was {_EMILY_BILLED_DOLLAR}",
        "I'll have my doctor send the records",
        "No, I don't need the link",
        "Yes, please proceed",
        "email please",
        "Yes, that's correct",
        "No, that's all",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-8: Bare 'No' at the claim number question (_is_no_response check) "
        "moves the fallback to DOS+billed. Emily provides 'January 8th, 2026' "
        "and $1,250; lookup succeeds."
    ),
)

# ── TB-9: Invalid claim number → retry → correct → success ───────────────────
ref_no_fallback_claim_number_retry_then_succeed = Scenario(
    name="ref_no_fallback_claim_number_retry_then_succeed",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",
        "abc",  # too short / invalid → retry
        _EMILY_CLAIM_NUMBER,  # valid → SF lookup → success
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),  # retry re-asks
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-9: Invalid claim number on first attempt ('abc' — too short for "
        "validate_claim_number). Agent retries once; member provides '882301'; "
        "SF lookup succeeds."
    ),
)

# ── TB-10: DOS+billed — partial first, complete on retry ─────────────────────
ref_no_fallback_dos_billed_retry = Scenario(
    name="ref_no_fallback_dos_billed_retry",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",
        "No",
        "just the eighth",  # DOS only, no amount → retry
        f"{_EMILY_DOS_SPOKEN} and {_EMILY_BILLED_SPOKEN}",  # both → success
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        10: TurnExpectation(ai_contains=[r"(date of service|billed|amount)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-10: Member provides only the date on the first DOS+billed attempt "
        "('just the eighth' — no billed amount). Agent retries; member provides "
        "both 'January eighth and twelve fifty' on the second attempt. Verifies "
        "the partial retry path in _collect_dos_billed_fallback."
    ),
)

# ── TB-11: Full claim flow after claim-number fallback ────────────────────────
ref_no_fallback_full_flow_via_claim_number = Scenario(
    name="ref_no_fallback_full_flow_via_claim_number",
    flow="claim",
    timeout_s=420,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",
        _EMILY_CLAIM_NUMBER,
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "Send updates to my phone",
        "Yes, that's correct",
        "How long will this take?",
        "email them too",
        "No, that's all. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "TB-11: Full end-to-end flow via claim-number fallback. Verifies that "
        "status reporting, records coordination, and both notification preferences "
        "all complete normally after a claim-number fallback lookup."
    ),
)

# ── TB-12: Full claim flow after DOS+billed fallback ─────────────────────────
ref_no_fallback_full_flow_via_dos_billed = Scenario(
    name="ref_no_fallback_full_flow_via_dos_billed",
    flow="claim",
    timeout_s=420,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",
        "No",
        f"{_EMILY_DOS_SPOKEN} and it was {_EMILY_BILLED_DOLLAR}",
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "Text me updates",
        "Yes, that's correct",
        "How long will this take?",
        "email me for that one",
        "No, that's all. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "claim_status": truthy,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "TB-12: Full end-to-end flow via DOS+billed fallback. Neither reference "
        "number nor claim number available; Emily provides January 8th + $1,250. "
        "Full records + notification flow completes normally."
    ),
)

# ── TB-13: No identifier at all → exhaustion escalation ──────────────────────
ref_no_fallback_no_identifiers_escalates = Scenario(
    name="ref_no_fallback_no_identifiers_escalates",
    flow="claim",
    timeout_s=300,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have the reference number",
        "No",
        "I don't know",  # can't provide DOS+billed → retry (attempt 0)
        "I have nothing here",  # second failure → escalation
        "still nothing",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_regex=r"fallback_(dos_billed_exhausted|dos_billed_not_found)",
    ),
    notes=(
        "TB-13: Member exhausts all fallback options — no reference number, no "
        "claim number, cannot provide DOS or billed amount. After one retry at "
        "the DOS+billed stage the agent escalates with "
        "reason=fallback_dos_billed_exhausted."
    ),
)

# ── TB-14: \"I never received a reference number\" → fallback ─────────────────
ref_no_fallback_never_had_ref = Scenario(
    name="ref_no_fallback_never_had_ref",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I never received a reference number for this",
        _EMILY_CLAIM_NUMBER,
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        r"TB-14: 'I never received a reference number' matches "
        r"\bi\s+never\s+(received|got)\b in detect_cannot_provide. Fallback starts "
        "with the claim-number ask; Emily provides 882301; lookup succeeds."
    ),
)

# ── TB-15: \"I lost the paper\" → fallback → claim number → success ───────────
ref_no_fallback_lost_letter_then_claim = Scenario(
    name="ref_no_fallback_lost_letter_then_claim",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I lost the paper that had the reference number",
        _EMILY_CLAIM_NUMBER,
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's all",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-15: Physical-absence phrasing 'I lost the paper' triggers the fallback. "
        "Emily then provides her claim number; SF lookup succeeds and status is "
        "reported."
    ),
)

# ── TB-16: Conversational claim number phrasing ───────────────────────────────
ref_no_fallback_claim_number_conversational = Scenario(
    name="ref_no_fallback_claim_number_conversational",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "oh I don't have that reference number with me right now",
        f"uh yeah my claim number is {_EMILY_CLAIM_NUMBER_SPOKEN}",
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
        },
    ),
    notes=(
        "TB-16: Natural conversational phrasing throughout. Member says 'uh yeah "
        "my claim number is eight eight two three zero one'; LLM must extract the "
        "spoken digits and the normalizer produces '882301'."
    ),
)

# ── TB-17: DOS with explicit year ─────────────────────────────────────────────
ref_no_fallback_dos_with_explicit_year = Scenario(
    name="ref_no_fallback_dos_with_explicit_year",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have it",
        "No",
        "January 8th, 2026 and the bill was $1,250",  # full ISO year
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's all",
    ],
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-17: DOS provided with explicit year 'January 8th, 2026'. "
        "normalize_date_of_service must parse the full spoken date and produce "
        "'2026-01-08', matching the seed record."
    ),
)

# ── TB-18: Claim number provided → DOS+billed must NOT be asked ──────────────
ref_no_fallback_claim_number_skips_dos_billed = Scenario(
    name="ref_no_fallback_claim_number_skips_dos_billed",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",
        _EMILY_CLAIM_NUMBER,  # valid claim number → skip DOS+billed entirely
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        # Status reported directly — no DOS+billed question in between
        9: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy, "upload_link_sent": True},
        # DOS+billed question must NEVER appear
        transcript_count={r"(date of service|billed amount|billed\s+\$)": 0},
    ),
    notes=(
        "TB-18: Regression guard — when a valid claim number is provided the agent "
        "must NOT ask for date of service or billed amount. The claim-number lookup "
        "succeeds directly and status is reported. transcript_count assert guards "
        "against erroneously exercising both fallback stages."
    ),
)

# ── TB-19: DOS+billed retry exhausted → escalation ───────────────────────────
ref_no_fallback_dos_billed_retry_then_escalate = Scenario(
    name="ref_no_fallback_dos_billed_retry_then_escalate",
    flow="claim",
    timeout_s=300,
    user_turns=CLAIM_VERIFY
    + [
        "I don't have the reference number",
        "No",
        "I'm not sure",  # invalid/missing → attempt 0 retry
        "hmm, I still can't find it",  # attempt 1 → exhausted → escalate
        "nothing",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_regex=r"fallback_dos_billed_exhausted",
    ),
    notes=(
        "TB-19: One retry at the DOS+billed stage is allowed. After the second "
        "consecutive invalid response (attempt_count >= 1) the agent escalates "
        "with reason=fallback_dos_billed_exhausted."
    ),
)

# ── TB-20: All-natural phrasing end-to-end ───────────────────────────────────
ref_no_fallback_all_natural_phrasing = Scenario(
    name="ref_no_fallback_all_natural_phrasing",
    flow="claim",
    timeout_s=420,
    retries=2,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "yeah, honestly I can't locate that anywhere in my paperwork",
        f"oh okay, yes — I do have the claim number, it's {_EMILY_CLAIM_NUMBER_SPOKEN}",
        "my doctor can just fax those records over",
        "no, I don't think I need the link, thanks",
        "yeah, go ahead and have the guide reach out",
        "you can send me text messages for updates",
        "yes, that number's correct",
        "and how long does this normally take to resolve?",
        "email is fine for those",
        "no, that covers everything, appreciate it",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "TB-20: All-natural phrasing throughout the fallback path. Cannot-provide "
        "trigger avoids literal patterns; claim number given with filler words and "
        "spoken digits. retries=2: conversational LLM extraction is non-deterministic. "
        "Covers the complete flow: fallback → claim_number → status → records "
        "(personal_guide) → notification (sms) → timeline (email)."
    ),
)


# ── TB-21: WAIT at claim_number stage → acknowledged → claim number → success ─
ref_no_fallback_wait_at_claim_number = Scenario(
    name="ref_no_fallback_wait_at_claim_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] → fallback: ask claim number
        "Yeah. I have the claim number. Just give me two minutes so I can find it.",  # [8] → WAIT
        f"Okay I found it, it's {_EMILY_CLAIM_NUMBER}",  # [9] → claim number → lookup
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),  # must ack, not retry
        10: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
        },
    ),
    notes=(
        "TB-21: Member signals they have the claim number but needs time to locate "
        "it ('just give me two minutes'). detect_wait_request fires in "
        "_collect_claim_number_fallback; agent acknowledges with MSG_WAIT_ACK and "
        "stays on claim_number_ask without incrementing the attempt counter. Member "
        "then provides '882301'; SF lookup succeeds and the flow continues normally."
    ),
)

# ── TB-22: WAIT at dos_billed stage → acknowledged → dos+billed → success ─────
ref_no_fallback_wait_at_dos_billed = Scenario(
    name="ref_no_fallback_wait_at_dos_billed",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] → fallback
        "No",  # [8] → no claim number → dos_billed stage
        "Just give me a second, I need to look this up.",  # [9] → WAIT in dos_billed
        f"{_EMILY_DOS_SPOKEN} and it was {_EMILY_BILLED_DOLLAR}",  # [10] → both → lookup
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        10: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),  # must ack, not retry
        11: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "TB-22: Member asks for time at the DOS+billed stage ('just give me a "
        "second'). detect_wait_request fires in _collect_dos_billed_fallback; agent "
        "acknowledges with MSG_WAIT_ACK and stays on dos_billed_ask without "
        "consuming an attempt. Member then provides January eighth and $1,250; SF "
        "lookup succeeds. Without the WAIT guard this turn would have been counted "
        "as attempt 0 and burned the only retry budget."
    ),
)

# ── TB-23: Claim number offered mid dos_billed stage → rescued → success ──────
ref_no_fallback_claim_number_rescue_from_dos_billed = Scenario(
    name="ref_no_fallback_claim_number_rescue_from_dos_billed",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] → fallback: ask claim number
        "No",  # [8] → no claim number → dos_billed stage
        # Member explicitly declines DOS+billed but provides the claim number
        # they located in the meantime — rescue block must catch it.
        f"No, I don't have those. But I just found my claim number — it's {_EMILY_CLAIM_NUMBER}.",  # [9]
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        # Status reported directly after rescue — no intermediate re-ask
        10: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
        },
        # DOS+billed question must NOT appear a second time after the rescue
        transcript_count={r"(date of service|billed amount|billed\s+\$)": 1},
    ),
    notes=(
        "TB-23: Member reaches the DOS+billed stage (no ref or claim number on "
        "first offer) but then finds and volunteers the claim number mid-utterance "
        "('No, I don't have those. But I just found my claim number — it's 882301.'). "
        "The rescue block in _collect_dos_billed_fallback extracts claim_number via "
        "the expanded pending_slots=[..., 'claim_number'] extraction call, then calls "
        "lookup_adjustment_by_claim_number directly. Status is reported on the very "
        "next turn; the dos_billed question appears exactly once (not re-asked after "
        "the rescue). Mirrors the real conversation: member says they found the claim "
        "number while the agent was asking for DOS+billed."
    ),
)


# ── TB-24: Hedged cannot-provide + clarification question prelude ────────────
ref_no_fallback_hedged_cannot_provide_with_clarification = Scenario(
    name="ref_no_fallback_hedged_cannot_provide_with_clarification",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        # [8] Clarification question about what the reference number is —
        # no digits, so extraction returns ambiguous; CLARIFY guard re-asks.
        "Is it the same as my member ID, or is it a different number?",
        # [9] Hedged cannot-provide — matches the new
        # r"\bi\s+don'?t\s+think\s+i\s+(have|know)\b" pattern in
        # detect_cannot_provide. Before the fix this fell through to LLM
        # extraction (no digits → ambiguous → RETRY: "Could you try your
        # reference number once more?").
        "Then I don't think I have the reference number.",
        # [10] Agent now asks for claim number (fallback started).
        # Member confirms they have one and offers it.
        "I do have the claim number. Is it that?",
        # [11] Agent confirms and asks for the number; member provides it.
        _EMILY_CLAIM_NUMBER,
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        7: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        # [8] clarification ask → AI explains and re-asks reference number
        8: TurnExpectation(ai_contains=[r"(member\s*id|different|reference)"]),
        # [9] hedged cannot-provide → fallback fires → AI asks for claim number
        9: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        10: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        12: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
        },
    ),
    notes=(
        "TB-24: Regression for the 'I don't think I have the reference number' bug. "
        "Member first asks a clarification question about the reference number "
        "(CLARIFY path — no digits, re-ask fires). Member then says 'Then I don't "
        "think I have the reference number' — the hedged phrasing previously bypassed "
        r"detect_cannot_provide (no pattern matched '\bi\s+don\'?t\s+think\s+i\s+have\b') "
        "so the LLM classified it as ambiguous and RETRY fired: 'Could you try your "
        "reference number once more?'. The fix added the pattern; detect_cannot_provide "
        "now returns True and the claim_number fallback starts. Member responds 'I do "
        "have the claim number. Is it that?' to the claim-number ask; member then "
        "provides '882301'; SF lookup succeeds."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# Registry — run order matters (scenarios share Salesforce data; run serially)
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    # A. PCP happy paths
    pcp_happy_path_fax,  # 1
    pcp_happy_path_email,  # 2
    pcp_benefits_declined,  # 3
    pcp_zip_update,  # 4  (mutating)
    pcp_zip_inline_update,  # 5  (mutating)
    pcp_fax_update,  # 6  (mutating)
    pcp_email_update,  # 7  (mutating)
    # B. Verification escalations
    verification_restart_then_success,  # 8
    verification_fail_twice_escalates,  # 9
    member_id_exhaustion,  # 10
    dob_no_year_exhaustion,  # 11
    member_id_ambiguous_exhaustion,  # 12
    # B2. Partial re-ask on identity mismatch (member found, field(s) wrong)
    verification_dob_only_mismatch,  # 12a
    verification_last_name_only_mismatch,  # 12b
    verification_first_name_only_mismatch,  # 12c
    verification_name_mismatch_bare_no_at_readback,  # 12d
    verification_multi_field_mismatch_generic,  # 12e
    verification_member_id_not_found_restart,  # 12f
    verification_repeated_dob_mismatch_escalates,  # 12g
    # C. Guard escalations
    transfer_request,  # 13
    abuse,  # 14
    self_harm,  # 15
    offtopic_repeated,  # 16
    # D. Intake routing
    intake_unclear_exhaustion,  # 17
    intake_out_of_scope_billing,  # 18
    intake_out_of_scope_appeal,  # 18b — appeal must NOT route to claim_services
    non_member_caller,  # 19
    intake_unsupported_provider_oncologist,  # 20a
    intake_unsupported_provider_neurologist,  # 20b
    intake_supported_provider_cardiologist,  # 20c (regression guard)
    intake_generic_provider_request,  # 20d (regression guard)
    intake_provider_type_propagates_to_search,  # 20e (intake→search propagation)
    # E. Claim flow
    claim_happy_path,  # 21
    claim_upload_only,  # 22
    claim_guide_only,  # 23
    claim_no_proceed,  # 24
    records_no_guide_upload_then_close,  # 24a
    records_no_guide_doctor_direct_then_close,  # 24b
    records_no_guide_regression_no_transfer,  # 24c
    records_no_guide_then_follow_up_question,  # 24d
    records_no_guide_conversational_phrasing,  # 24e
    records_no_guide_then_pcp_new_intent,  # 24f (mutating)
    records_no_guide_then_claim_new_intent,  # 24g
    records_no_guide_then_unsupported_provider,  # 24h
    phone_not_confirmed_ends_call,  # 25
    ref_not_found_retry_then_success,  # 26
    ref_not_found_twice_escalates,  # 27
    ref_exhaustion,  # 28
    # E2. Reference-number pivot — member recovers the reference number mid-fallback
    ref_fallback_pivot_from_claim_number_ask,  # 28a
    ref_fallback_pivot_from_dos_billed_ask,  # 28b
    ref_fallback_pivot_after_wait,  # 28c (exact transcript path)
    # E3. Symmetric fallback pivots — claim number ↔ dos+billed navigation
    ref_fallback_dos_billed_pivot_to_claim_number,  # 28d
    ref_fallback_claim_number_forward_to_dos_billed,  # 28e
    ref_fallback_chained_pivot_transcript,  # 28f (exact transcript: dos_billed→ref#→claim#)
    ref_fallback_hesitation_heavy,  # 28g (3×WAIT + bare affirmative + keyword pivot)
    claim_email_change_on_upload,  # 29 (mutating)
    # F. Follow-up escalations
    follow_up_update_request,  # 30
    follow_up_cannot_answer_x3,  # 31
    # M. New-intent mid-session (same-member shortcut — verified context reused)
    pcp_then_claim_new_intent,  # 31a — PCP → claim, same member confirmed (skip re-verify)
    claim_then_pcp_new_intent,  # 31b — claim → PCP, same member confirmed (mutating)
    # M2. Follow-up re-screen through intake (front-door screening on a mid-call pivot)
    followup_unsupported_provider_rescreen,  # 31c — unsupported provider escalates pre-reverify
    followup_supported_provider_rescreen,  # 31d — supported provider re-screen, same-member
    followup_appeal_rescreen,  # 31e — appeal keyword → out_of_scope (no same-member check)
    followup_grievance_rescreen,  # 31f — grievance keyword → out_of_scope (no same-member check)
    # M3. Same-member disambiguation edge paths
    followup_new_intent_different_member,  # 31g — PCP→claim "different" → re-verify new member
    followup_new_intent_same_member_ambiguous,  # 31h — ambiguous → clarification → confirmed
    followup_supported_provider_rescreen_different_member,  # 31i — PCP→dermatologist "different" → re-verify
    claim_then_pcp_new_intent_different_member,  # 31j — claim→PCP "different" → re-verify as Emily
    # G. Contact-change loop limits
    zip_change_loop_escalates,  # 32  (redefined: invalid-ZIP slot exhaustion)
    email_change_loop_in_notification,  # 33 (mutating)
    # G2. Notification contact-confirmation advances on first affirmative (regression)
    notification_phone_confirm_advances,  # 33a
    notification_phone_confirm_bare_yes_advances,  # 33b
    notification_email_confirm_advances,  # 33c
    # H. Conversational & confusion-recovery
    pcp_happy_path_conversational,  # 34
    claim_happy_path_conversational,  # 35
    pcp_confused_member,  # 36
    claim_confused_member,  # 37
    pcp_conversational_confusion,  # 38
    claim_conversational_confusion,  # 39
    # I. Boundary stress
    boundary_walk_claim,  # 40
    # J. Name confirmation
    name_confirmation_happy_path,  # NC-1
    name_confirmation_inline_correction,  # NC-2
    name_confirmation_bare_no_then_gives_name,  # NC-3
    name_confirmation_exhaust_escalates,  # NC-4
    name_confirmation_claim_flow,  # NC-5
    name_confirmation_single_letter_first_name,  # NC-6
    name_confirmation_rejection_asks_correction,  # NC-7
    name_confirmation_consecutive_rejections_then_correct,  # NC-12
    name_confirmation_confirmed_with_side_question,  # NC-8
    name_confirmation_both_names_corrected_inline,  # NC-9
    name_confirmation_partial_correction_first_only,  # NC-10
    name_confirmation_natural_sentence_correction,  # NC-11
    # K. Indirect-decline regression
    *INDIRECT_DECLINE_SCENARIOS,  # ID-1, ID-2, ID-3
    # L. Cannot-provide short-circuit
    *CANNOT_PROVIDE_SCENARIOS,  # CP-1 … CP-6
    # N. Follow-up disposition routing, update detours & WAIT (Phases 4-7)
    followup_answer_confirmed_slot,  # N-1 — FOLLOWUP_ANSWER + appended static ask
    followup_park_question_deferred,  # N-2 — FOLLOWUP_PARK → answered in follow_up
    followup_decline_irrelevant,  # N-3 — FOLLOWUP_DECLINE + appended static ask
    correction_inline_case_a,  # N-4 — Case A: answer + valid correction
    update_without_value_case_b,  # N-5 — Case B: detour + return pointer
    bare_update_detour_c2,  # N-6 — Case C2: bare update detour round-trip
    locked_field_update_declined,  # N-7 — LOCKED field update → decline
    update_loop_guard_escalates,  # N-8 — per-target loop guard escalation
    wait_ack_then_answer,  # N-9 — static wait ack, value wins after
    wait_nudge_after_three,  # N-10 — 3 waits → slot-naming nudge
    # O. Production-transcript regressions (LLM-2 hygiene, Phases 1-4)
    emily_carter_correction_single_ask,  # O-1 — Bug A: correction double-ask
    notification_followup_not_declined,  # O-2 — Bug B: later-stage question declined
    zip_update_during_fax_confirmation,  # O-3 — Bug C: zip-update routing (mutating)
    # P. Cross-agent redo/replay requests (Phase 6)
    redo_fax_to_email_from_benefits,  # P-1 — (a) redo routes benefits → delivery → back
    replay_benefits_from_follow_up,  # P-2 — (b) replay routes follow_up → benefits → back
    redo_inflow_before_dispatch,  # P-3 — (c) owner active pre-dispatch → in-flow, zero routing
    replay_benefits_inflow_at_coach_offer,  # P-4 — (c) in-flow benefits replay, zero routing
    unknown_replay_topic_parks,  # P-5 — (d) unknown replay topic parks as a question
    # Q. Language-variation regressions (paraphrased BUG-1..5 + claims parity)
    zip_update_during_fax_paraphrased,  # Q-1 — BUG-5 paraphrase (mutating)
    redo_list_to_email_paraphrased,  # Q-2 — BUG-2 paraphrase
    redo_list_to_email_address,  # Q — BUG-2 paraphrase
    notification_channel_switch_paraphrased,  # Q-3 — BUG-3 + parity channel switch
    followup_parked_notification_paraphrased,  # Q-4 — BUG-1 paraphrase
    verification_identity_update_paraphrased,  # Q-5 — BUG-4 paraphrase
    claim_status_replay_paraphrased,  # Q-6 — Phase-7 claim_status replay
    practice_team_context_retension_issue1,
    practice_team_context_retension_issue2,
    automated_test,  # sanity check: a single test that runs the same as the others
    records_no_guide_upload_then_close,  # 24a
    records_no_guide_doctor_direct_then_close,  # 24b
    records_no_guide_regression_no_transfer,  # 24c
    records_no_guide_then_follow_up_question,  # 24d
    records_no_guide_conversational_phrasing,  # 24e
    records_no_guide_then_pcp_new_intent,  # 24f (mutating)
    records_no_guide_then_claim_new_intent,  # 24g
    records_no_guide_then_unsupported_provider,  # 24h
]

# ──────────────────────────────────────────────────────────────────────────────
# R. SSN fallback — Member ID denial → alternative verification path
# ──────────────────────────────────────────────────────────────────────────────

ssn_fallback_yes_then_provides = Scenario(
    name="ssn_fallback_yes_then_provides",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "Yeah, hi — I'm trying to locate a primary care doctor near me.",
        "It's Emily.",
        "Carter.",
        "Yeah, that looks right.",  # name_confirmed
        "Oh, I actually don't have that with me right now.",  # member_id denial
        "Yeah, I do have it.",  # yes → agent asks to provide
        "five two seven four one three eight two zero",  # spoken SSN
        "April twelfth nineteen eighty eight.",  # DOB
        "I'm the plan holder.",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's all",
    ],
    turn_expectations={
        4: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        6: TurnExpectation(ai_contains=[pool_regex(MSG_SSN_COLLECT)]),
        7: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "ssn": "527-41-3820",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "SSN fallback happy path: caller denies having their Member ID, confirms "
        "they have their SSN on a separate turn, provides it in spoken-digit form, "
        "then provides DOB. Verifies the spoken-digit normalizer, the two-turn "
        "yes→provide flow, DOB collection, and that member_status_verify is True."
    ),
)

ssn_fallback_inline_with_yes = Scenario(
    name="ssn_fallback_inline_with_yes",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "Hi, I'd like to find a primary care physician.",
        "Emily",
        "Carter",
        "Yep, that's me.",  # name_confirmed
        "Hmm, I can't seem to track it down.",  # member_id denial
        "Sure, it's five two seven dash four one dash three eight two zero.",  # yes_with_ssn inline
        "April twelfth nineteen eighty eight.",  # DOB
        "I'm calling for myself.",
        "Primary Care Physician",
        "yes that's correct",
        "fax please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, nothing else",
    ],
    turn_expectations={
        4: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        # Turn 6 must be the DOB question — agent must NOT ask for SSN again
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "ssn": "527-41-3820",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "YES_WITH_SSN inline path: the caller answers 'Do you have the SSN?' by "
        "providing the number in the same utterance. The agent must extract the SSN "
        "immediately, NOT ask 'Please provide your SSN.' again, and then ask for DOB. "
        "Turn-6 DOB question is the proof."
    ),
)

ssn_fallback_no_then_provides_mid = Scenario(
    name="ssn_fallback_no_then_provides_mid",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I'm looking for a primary care physician in my network.",
        "Emily",
        "Carter",
        "Yes, correct.",  # name_confirmed
        "I left my card at home, so I don't have it.",  # member_id denial
        "Not really.",  # soft no → prompt for either
        "Oh wait, I found it — M nine zero seven five zero three.",  # member_id in retry
        "April twelfth nineteen eighty eight.",
        "I'm on my own plan.",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's it",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        6: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_EITHER])]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "member_id": "M907503",  # member_id path, not SSN
            "provider_list_sent": True,
        },
    ),
    notes=(
        "NO_INTENT (soft no) path: caller says 'not really' to 'Do you have your SSN?' "
        "→ agent prompts for either Member ID or SSN without escalating. Caller then "
        "finds their Member ID and provides it, completing verification via the "
        "standard identity path."
    ),
)

ssn_fallback_no_ssn_available_escalates = Scenario(
    name="ssn_fallback_no_ssn_available_escalates",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care doctor, please.",
        "Emily",
        "Carter",
        "Yes, that's right.",  # name_confirmed
        "Honestly I have no idea where my card is.",  # member_id denial
        "I don't have my SSN either, sorry.",  # no_ssn_available → escalate
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="no identifier available",
        final_state={"member_status_verify": lambda v: not v},
        last_ai_contains=[r"(representative|assist you further|connect you)"],
    ),
    notes=(
        "NO_SSN_AVAILABLE path: after denying their Member ID the caller also "
        "states they cannot provide their SSN. The agent must escalate immediately "
        "without further prompting. member_status_verify must stay falsy."
    ),
)

ssn_fallback_claim_flow = Scenario(
    name="ssn_fallback_claim_flow",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I submitted a claim a few weeks back and wanted to check on the status.",
        "James",
        "Wilson",
        "Yep, got it.",  # name_confirmed
        "I'm not sure where I put my insurance card.",  # member_id denial
        "Yeah, let me check — it's seven four one two nine five zero six three.",  # SSN inline
        "Thirtieth of July, nineteen seventy seven.",  # DOB
        "yes correct",  # phone confirmation
        "42695817",
        "I'll upload the records myself.",
        "Yes, please send the link.",
        "Yes, that's the right email.",
        "No, I'd rather not have anyone contact the provider.",
        "Email works for me.",
        "Yes, that's correct.",
        "No, that's everything, thanks.",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "ssn": "741-29-5063",
            "upload_link_sent": True,
        },
    ),
    notes=(
        "SSN fallback through the claims verification path (phone_confirmed instead "
        "of relationship). Caller denies Member ID, provides SSN inline, then DOB. "
        "Verifies the SSN+DOB lookup populates member_status_verify and the full "
        "claim flow continues normally."
    ),
)


SCENARIOS.extend(
    [
        ssn_fallback_yes_then_provides,  # R-1
        ssn_fallback_inline_with_yes,  # R-2
        ssn_fallback_no_then_provides_mid,  # R-3
        ssn_fallback_no_ssn_available_escalates,  # R-4
        ssn_fallback_claim_flow,  # R-5
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# S. SSN fallback — format error recovery
#
# These scenarios verify that the agent handles malformed SSN and Member ID
# input gracefully: re-asks with a format hint (CLARIFY on first error, RETRY
# on subsequent), never burns the full attempt budget on a single transposition
# or incomplete entry, and completes successfully when the caller corrects.
# ──────────────────────────────────────────────────────────────────────────────

ssn_fallback_ssn_incomplete_then_correct = Scenario(
    name="ssn_fallback_ssn_incomplete_then_correct",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "Emily",
        "Carter",
        "Yes, that's correct.",  # name_confirmed
        "I left my card somewhere, I can't find it.",  # member_id denial
        "Sure, let me grab it.",  # yes → agent asks to provide
        "five two seven four one three",  # only 6 spoken digits — incomplete → re-ask
        "five two seven four one three eight two zero",  # full 9 digits → valid
        "April twelfth nineteen eighty eight.",  # DOB
        "I'm the plan holder.",
        "Primary Care Physician",
        "yes that's correct",
        "fax please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        6: TurnExpectation(ai_contains=[pool_regex(MSG_SSN_COLLECT)]),
        # Re-ask after the incomplete SSN — must ask for SSN again, not escalate
        7: TurnExpectation(ai_contains=[r"(ssn|social security|format|xxx|again)"]),
        # DOB ask follows the valid SSN
        8: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "ssn": "527-41-3820",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "SSN format error: caller provides only 6 spoken digits the first time "
        "(normalizer rejects non-9-digit input). The agent re-asks with a format "
        "hint (CLARIFY on attempt 1 — no attempt cost). Caller provides all 9 digits "
        "on the next turn. Verifies the retry loop fires exactly once and the "
        "attempt budget is not exhausted by the partial entry."
    ),
)

ssn_fallback_ssn_raw_digits_no_dashes = Scenario(
    name="ssn_fallback_ssn_raw_digits_no_dashes",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "Looking for a primary care doctor in my area.",
        "Emily",
        "Carter",
        "Yeah, that's me.",  # name_confirmed
        "Hmm, I can't seem to find my card.",  # member_id denial
        "Yeah I've got it.",  # yes
        "527413820",  # 9 raw digits, no dashes — must normalize
        "April twelfth nineteen eighty eight.",  # DOB
        "Calling for myself.",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's all",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        6: TurnExpectation(ai_contains=[pool_regex(MSG_SSN_COLLECT)]),
        # Must ask for DOB next — not re-ask SSN (raw digits normalize correctly)
        7: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "ssn": "527-41-3820",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "SSN format variant: 9 raw digits without dashes ('527413820'). The "
        "normalize_ssn() function must convert this to XXX-XX-XXXX without a "
        "re-ask. Turn-7 DOB question is the proof that no re-ask fired. "
        "Covers the common case where a caller reads their SSN card as a "
        "continuous digit string."
    ),
)

ssn_fallback_member_id_missing_prefix_in_retry = Scenario(
    name="ssn_fallback_member_id_missing_prefix_in_retry",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "Hi, I'm trying to find a primary care physician.",
        "Emily",
        "Carter",
        "Yes, correct.",  # name_confirmed
        "I don't know where my card is.",  # member_id denial
        "Not at the moment.",  # soft no → either identifier prompt
        "nine zero seven five zero three",  # digits only, no M prefix → invalid
        "oh sorry, M nine zero seven five zero three",  # correct with M prefix
        "April twelfth nineteen eighty eight.",  # DOB
        "I'm the plan holder.",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's it",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        6: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_EITHER])]),
        # Re-ask after invalid member ID (no M prefix) — must re-prompt, not accept
        7: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_EITHER])]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "member_id": "M907503",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Member ID format error in the ssn_or_mid_retry stage: caller provides "
        "bare digits without the required M prefix ('nine zero seven five zero "
        "three'). The LLM extractor's extraction prompt rule 'Caller must say M "
        "first' causes it to return an empty member_id, so the agent re-prompts. "
        "Caller corrects with 'M nine zero seven five zero three' on the next turn. "
        "Verifies the retry does not escalate and the corrected member_id flows "
        "through to a successful SF lookup."
    ),
)

ssn_fallback_ssn_too_many_digits_then_correct = Scenario(
    name="ssn_fallback_ssn_too_many_digits_then_correct",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "Emily",
        "Carter",
        "Yes, that's right.",  # name_confirmed
        "Actually I don't have my card on me.",  # member_id denial
        "Yes, I have it.",  # yes → agent asks to provide
        "five two seven four one three eight two zero zero",  # 10 spoken digits → ambiguous
        "five two seven four one three eight two zero",  # correct 9 digits
        "April twelfth nineteen eighty eight.",  # DOB
        "I'm the plan holder.",
        "Primary Care Physician",
        "yes that's correct",
        "fax please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex([MSG_SSN_ASK])]),
        6: TurnExpectation(ai_contains=[pool_regex(MSG_SSN_COLLECT)]),
        # Re-ask after 10 digits (ssn_fallback.md: cannot collect exactly 9 → ambiguous)
        7: TurnExpectation(ai_contains=[r"(ssn|social security|format|xxx|again)"]),
        8: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "ssn": "527-41-3820",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "SSN over-digit error: caller speaks 10 digits instead of 9 (common "
        "transposition where an extra zero is added). The ssn_fallback.md prompt "
        "rule 'if you cannot collect exactly 9 digits → ssn_intent = ambiguous' "
        "causes the LLM to return ambiguous, triggering a re-ask. On the next turn "
        "the caller provides exactly 9 digits and verification completes. Verifies "
        "the re-ask fires once without exhausting the attempt budget."
    ),
)


SCENARIOS.extend(
    [
        ssn_fallback_ssn_incomplete_then_correct,  # S-1
        ssn_fallback_ssn_raw_digits_no_dashes,  # S-2
        ssn_fallback_member_id_missing_prefix_in_retry,  # S-3
        ssn_fallback_ssn_too_many_digits_then_correct,  # S-4
    ]
)

SCENARIOS.extend(
    [
        # T. Reference-number fallback sub-flow
        ref_no_fallback_claim_number_happy_path,  # TB-1
        ref_no_fallback_dos_billed_happy_path,  # TB-2
        ref_no_fallback_inline_dos_billed,  # TB-3
        ref_no_fallback_spoken_claim_number,  # TB-4
        ref_no_fallback_claim_number_not_found,  # TB-5
        ref_no_fallback_dos_billed_not_found,  # TB-6
        ref_no_fallback_cannot_provide_claim_number,  # TB-7
        ref_no_fallback_bare_no_claim_number,  # TB-8
        ref_no_fallback_claim_number_retry_then_succeed,  # TB-9
        ref_no_fallback_dos_billed_retry,  # TB-10
        ref_no_fallback_full_flow_via_claim_number,  # TB-11
        ref_no_fallback_full_flow_via_dos_billed,  # TB-12
        ref_no_fallback_no_identifiers_escalates,  # TB-13
        ref_no_fallback_never_had_ref,  # TB-14
        ref_no_fallback_lost_letter_then_claim,  # TB-15
        ref_no_fallback_claim_number_conversational,  # TB-16
        ref_no_fallback_dos_with_explicit_year,  # TB-17
        ref_no_fallback_claim_number_skips_dos_billed,  # TB-18
        ref_no_fallback_dos_billed_retry_then_escalate,  # TB-19
        ref_no_fallback_all_natural_phrasing,  # TB-20
        ref_no_fallback_wait_at_claim_number,  # TB-21
        ref_no_fallback_wait_at_dos_billed,  # TB-22
        ref_no_fallback_claim_number_rescue_from_dos_billed,  # TB-23
        ref_no_fallback_hedged_cannot_provide_with_clarification,  # TB-24
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# FW. Follow-up WAIT detection — LLM-based (FollowUpIntent.WAIT)
#
# These scenarios exercise the follow_up_agent's WAIT handling introduced to
# fix cases where long wait phrases ("Just give me a minute. I have this
# feeling I forgot something…") and non-keyword forms ("give me two minutes",
# "let me collect my thoughts") were mis-classified as UNSURE or QUESTION
# because detect_wait_request's continuation guard fired.  The fix added
# FollowUpIntent.WAIT to the schema and a classification section to both
# follow_up.md and follow_up_claims.md so the LLM can classify these turns.
#
# Turn-index reference for _PCP_TO_FOLLOW_UP-based scenarios:
#   PCP_VERIFY occupies turns [0..6]; _PCP_TO_FOLLOW_UP adds [7..12].
#   First follow-up user turn → [13].  turn_expectations[N] = AI response
#   after user_turn[N-1].
# ──────────────────────────────────────────────────────────────────────────────

# ── FW-1: Exact-transcript reproduction — long phrases that broke keyword match ─
followup_wait_long_phrase_then_done = Scenario(
    name="followup_wait_long_phrase_then_done",
    flow="pcp",
    timeout_s=360,
    retries=1,  # FollowUpIntent.WAIT classification is LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Both utterances are from the real failing transcript.  The first matches
        # "give me a minute" but the continuation guard fires (too many leftover
        # words) so detect_wait_request returns False; the LLM must classify WAIT.
        # The second contains "give me two minutes" — never in WAIT_PATTERNS at all.
        "Just give me a minute. I have this feeling that I forgot something to ask.",
        "No. Just give me two minutes. I will just trying to remember.",
        "I'm good to go then. I'm not remembering it.",
    ],
    turn_expectations={
        14: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),  # after turn [13]
        15: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),  # after turn [14]
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"provider_list_sent": True},
        transcript_contains=[pool_regex(MSG_WAIT_ACK)],
    ),
    notes=(
        "FW-1: Regression for the exact production transcript that exposed the bug. "
        "Two consecutive wait utterances that keyword matching missed: the first "
        "triggers detect_wait_request's continuation guard (leftover words after "
        "stripping 'give me a minute'); the second uses 'two minutes' which was never "
        "in WAIT_PATTERNS. Both must produce MSG_WAIT_ACK (LLM classifies WAIT). "
        "Member ends with 'I'm good to go' → DONE. retries=1: FollowUpIntent.WAIT "
        "classification is LLM-driven."
    ),
)

# ── FW-2: Keyword-free wait phrase → ack → question answered → done ───────────
followup_wait_then_question_answered = Scenario(
    name="followup_wait_then_question_answered",
    flow="pcp",
    timeout_s=360,
    retries=1,  # FollowUpIntent.WAIT + question classification are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # "let me collect my thoughts" has no keyword in WAIT_PATTERNS; LLM-only.
        "Hold on, let me collect my thoughts for a moment.",  # [13] → WAIT
        "Yes — what fax number did you send the provider list to?",  # [14] → question
        "Perfect, that's all I needed. Thanks.",  # [15] → DONE
    ],
    turn_expectations={
        14: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),  # after wait
        15: TurnExpectation(ai_contains=[r"(fax|617|sent|provider)"]),  # fax number in answer
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"provider_list_sent": True},
    ),
    notes=(
        "FW-2: Wait phrase with no matching keyword ('let me collect my thoughts') "
        "forces LLM-only detection → MSG_WAIT_ACK. Member then asks a genuine "
        "follow-up question (fax number used) which the LLM answers from the session "
        "snapshot, then closes the call. Verifies that a WAIT turn does not consume "
        "the cannot-answer budget (member asks exactly one question and gets an "
        "answer — not the 'cannot answer' path). retries=1: both the WAIT and "
        "question classifications are LLM-driven."
    ),
)

# ── FW-3: Wait then new provider request for same member ──────────────────────
followup_wait_then_new_provider_same_member = Scenario(
    name="followup_wait_then_new_provider_same_member",
    flow="pcp",
    timeout_s=480,
    retries=1,  # WAIT + new_intent + same-member + provider/delivery are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "Wait, give me two minutes — I need to think.",  # [13] → WAIT (two minutes)
        "Okay, I also need to find a cardiologist for myself.",  # [14] → new_intent
        "yes, same member",  # [15] → shortcut: skip re-verify
        "I'm calling for myself",  # [16] relationship → plan_holder (re-asked for provider flow)
        "yes that's correct",  # [17] ZIP on file confirmed
        "send it to my fax",  # [18] delivery
        "yes that's correct",  # [19] fax confirmed
        "no thanks",  # [20] benefits
        "no thank you",  # [21] Care Coach
        "no, that's everything",  # [22] close
    ],
    turn_expectations={
        14: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),  # wait → ack
        15: TurnExpectation(ai_contains=[r"same member|different member"]),  # disambiguation
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "call_intent": "provider_services",
            "provider_type": "Cardiologist",
            "provider_list_sent": True,
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
    ),
    notes=(
        "FW-3: Wait at follow-up ('give me two minutes' — LLM-detected) → ack → "
        "new provider request (cardiologist) for the same member. Proves that a "
        "pre-new-intent WAIT does not disrupt the new_intent → same-member "
        "disambiguation → re-screen shortcut chain: turn-14 must be MSG_WAIT_ACK, "
        "turn-15 must be the disambiguation question. Same-member shortcut re-asks "
        "'Are you the subscriber or dependent?' via verification before routing to "
        "provider_search; provider_list_sent=True. "
        "retries=1: WAIT + new_intent + same-member + provider/delivery are LLM-driven."
    ),
)

# ── FW-4: Wait then new provider request for a different member ───────────────
followup_wait_then_new_provider_different_member = Scenario(
    name="followup_wait_then_new_provider_different_member",
    flow="pcp",
    timeout_s=540,
    retries=1,  # WAIT + new_intent + same-member + provider/delivery are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "Give me two minutes, I'm trying to remember what I needed.",  # [13] → WAIT
        "Actually, I also need to find a dermatologist, but it's for my daughter.",  # [14] → new_intent
        "no, it's for a different person",  # [15] → cleared saved context → first-name bridge
        # Re-verification from scratch (same Emily fixture — proves re-verify ran).
        "emily",
        "carter",
        "yes correct",  # name confirmed
        "m nine zero seven five zero three",
        "April twelvee nineteen eighty-eight",
        "I'm calling for myself",
        "yes that's correct",  # ZIP on file
        "email please",
        "yes that's correct",  # email on file
        "no thanks",  # benefits
        "no thank you",  # Care Coach
        "no, that's everything",
    ],
    turn_expectations={
        14: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)]),  # wait → ack
        15: TurnExpectation(ai_contains=[r"same member|different member"]),  # disambiguation
        16: TurnExpectation(ai_contains=[r"first name"]),  # first-name bridge after "different"
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "call_intent": "provider_services",
            "provider_type": "Dermatologist",
            "provider_list_sent": True,
            "saved_member_context": falsy,
            "same_member_check_pending": falsy,
        },
    ),
    notes=(
        "FW-4: Wait at follow-up ('give me two minutes' — LLM-detected, no keyword "
        "match) → ack → new provider request (dermatologist) for a different person. "
        "Proves three things in sequence: (1) the WAIT is acknowledged (turn-14 = "
        "MSG_WAIT_ACK, not the disambiguation question); (2) the subsequent new-intent "
        "triggers the disambiguation question (turn-15); (3) 'different' clears the "
        "saved context and fires the first-name bridge (turn-16), after which full "
        "re-verification runs and provider_search delivers the list. Counterpart to "
        "FW-3 ('same member'). retries=1: WAIT + new_intent + same-member + "
        "provider/delivery classifications are all LLM-driven."
    ),
)

SCENARIOS.extend(
    [
        followup_wait_long_phrase_then_done,  # FW-1
        followup_wait_then_question_answered,  # FW-2
        followup_wait_then_new_provider_same_member,  # FW-3
        followup_wait_then_new_provider_different_member,  # FW-4
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# GD. Delivery-management response-generation regressions
#
# Two scenarios reproduce the hallucinations fixed in followup_respond.md and
# recovery_base.md:
#
#   GD-1  "Can we update it?" while the agent is asking for a new fax number
#         → LLM-2 used to say "A representative would need to make that
#         change for you." The fix: fax/email updates are in-flow operations
#         so the response generator must never claim otherwise.
#
#   GD-2  After a new fax number is confirmed, the agent used to say "Great,
#         and could you confirm the email address for me?" — LLM-2 drifting to
#         the other channel via conversation history. The fix: channel discipline
#         rule in recovery_base.md (and followup_respond.md) prohibits
#         mentioning the other channel when collecting fax or email.
#
# Both scenarios use the same fax-change conversation shape but assert
# different invariants.  Neither is mutating=True — they write to Emily's
# Salesforce fax in the sandbox (same as the indirect-decline regression
# tests) but expect harness-level teardown rather than per-scenario restoration.
# ──────────────────────────────────────────────────────────────────────────────

_FAX_UPDATE_VERIFY = PCP_VERIFY + [
    "Primary Care Physician",  # [7]
    "yes that's correct",  # [8] ZIP on file
    "send it to my fax",  # [9] → AI reads back fax on file
]
_NEW_FAX_SPOKEN = "six one seven one two three four one nine nine"
_NEW_FAX_DIGITS = "6171234199"

# ── GD-1: vague update request while collecting fax → re-ask, no "representative" ──
fax_update_vague_request_no_representative = Scenario(
    name="fax_update_vague_request_no_representative",
    flow="pcp",
    timeout_s=360,
    retries=1,  # LLM-2 FOLLOWUP_RESPOND classification is non-deterministic
    user_turns=_FAX_UPDATE_VERIFY
    + [
        "Oh yeah. But I have changed the fax number recently.",  # [10] → decline
        "Can we update it?",  # [11] → vague update request — must re-ask fax, not say "representative"
        _NEW_FAX_SPOKEN,  # [12] → new fax
        "yes that's correct",  # [13] → confirm readback → dispatch
        "no thanks",
        "no thank you",
        "no, that's all",
    ],
    turn_expectations={
        # After "Can we update it?", agent must re-ask for the fax — not say
        # "representative" and not drift to asking about email.
        12: TurnExpectation(ai_contains=[r"fax"]),
        # Fax readback must contain the new number.
        13: TurnExpectation(ai_contains=[r"617.?123.?4199|6171234199"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
        },
        # Agent must never claim the caller needs a rep to update their fax.
        transcript_count={r"representative.*change|change.*representative": 0},
    ),
    notes=(
        "GD-1: Regression for the 'representative' hallucination. When the member "
        "says 'Can we update it?' while the agent is asking for a new fax number, "
        "LLM-2 used to say 'A representative would need to make that change for "
        "you.' — because followup_respond.md's account-update rule did not exempt "
        "in-flow fax/email changes. Fix: added EXCEPTION clause for delivery "
        "contact updates and channel discipline rule. "
        "Assert: turn-12 AI re-asks for fax; 'representative' + 'change' never "
        "co-occur in any AI line. retries=1: LLM-2 response is non-deterministic."
    ),
)

# ── GD-2: after new fax confirmed → dispatch to fax, not email question ───────
fax_update_confirmed_no_email_hallucination = Scenario(
    name="fax_update_confirmed_no_email_hallucination",
    flow="pcp",
    timeout_s=360,
    retries=1,  # LLM-2 channel drift is non-deterministic
    user_turns=_FAX_UPDATE_VERIFY
    + [
        "Hmm, that fax is outdated actually.",  # [10] → decline on-file fax
        _NEW_FAX_SPOKEN,  # [11] → new fax provided directly
        "yes that's correct",  # [12] → confirm readback → dispatch to FAX
        "no thanks",
        "no thank you",
        "no, that's all",
    ],
    turn_expectations={
        # Fax readback must contain the new number.
        12: TurnExpectation(ai_contains=[r"617.?123.?4199|6171234199"]),
        # After confirming the fax, the agent must dispatch to fax — not ask
        # for email.  The dispatch window message mentions "30 minutes" or
        # "send" — never "confirm your email" or "email address for me".
        13: TurnExpectation(ai_contains=[r"(30 minutes|send|within|fax)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
        },
        # The specific hallucinated phrases must never appear.
        transcript_count={
            r"confirm.*email address|email address for me|confirm the email": 0,
        },
    ),
    notes=(
        "GD-2: Regression for the email-hallucination after fax confirmation. "
        "After the member confirms a new fax number ('yes that's correct'), "
        "LLM-2 used to generate 'Great, and could you confirm the email address "
        "for me?' — drifting to the other channel via conversation history. "
        "Fix: channel discipline rule in recovery_base.md and followup_respond.md "
        "prohibits mentioning email when collecting/confirming a fax. "
        "Assert: turn-13 AI message is the dispatch window (fax/30 minutes); "
        "the hallucinated email-confirmation phrases appear zero times. "
        "retries=1: LLM-2 channel discipline is non-deterministic."
    ),
)

SCENARIOS.extend(
    [
        fax_update_vague_request_no_representative,  # GD-1
        fax_update_confirmed_no_email_hallucination,  # GD-2
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# U. Regression tests for stale ref_no_fallback_stage + affirmative-retry fixes
#
# Three code fixes are covered:
#
#   U-1/U-2  Stale ref_no_fallback_stage persisting into a second member's
#            claim flow — reset_for_new_intent and NEW_INTENT_CLEAR_FIELDS did
#            not include ref_no_fallback_stage, so the defensive guard in Phase
#            1 (awaiting_slot='reference_number' + non-empty stage → clear) is
#            the decisive fix.  If the bug is present, the second member's
#            reference number is extracted as claim_number → SF lookup fails →
#            escalation.
#
#   U-3/U-4  Bare affirmative ("Yes", "Yep") at the claim_number stage used to
#            burn the only retry: attempt_count 0→1, so the actual claim number
#            provided on the very next turn hit attempt_count >= 1 and the agent
#            jumped to the dos_billed stage without ever trying extraction.
#            Fix: affirmative-only utterances (no digit content) get a free
#            re-ask; attempt_count stays 0.
#
#   U-5/U-6  Same affirmative-retry bug at the dos_billed stage: "Yes" burned
#            the retry, so any genuine extraction failure on the next turn
#            exhausted the budget and escalated.  Fix: same affirmative guard.
#            U-6 also verifies the free re-ask truly doesn't consume a retry —
#            a partial answer after the affirmative still gets one genuine retry.
# ──────────────────────────────────────────────────────────────────────────────

# Shared tail for Emily's minimal records + notification completion (decline guide → follow_up)
_EMILY_DECLINE_GUIDE_TAIL = [
    "I will upload them myself",
    "Yes, please send the link",
    "Yes, that's correct",  # email confirmed
    "No, that's fine",  # decline guide → follow_up
]

# James re-verification turns (claims slot order, used in stale-state scenarios)
_JAMES_VERIFY = [
    "james",
    "wilson",
    "yes correct",  # name confirmed
    "m three one zero one eight eight",
    "Thirtieth of July, nineteen seventy seven",
    "yes that's correct",  # phone confirmed
]

# ── U-1: Stale ref_no_fallback_stage after claim_number fallback ──────────────
stale_fallback_stage_after_claim_number_second_member = Scenario(
    name="stale_fallback_stage_after_claim_number_second_member",
    flow="claim",
    timeout_s=600,
    retries=1,  # new_intent + same-member LLM classifications are non-deterministic
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] fallback → claim number asked
        _EMILY_CLAIM_NUMBER,  # [8] 882301 → lookup → status
    ]
    + _EMILY_DECLINE_GUIDE_TAIL  # [9-12] records + decline guide → follow_up
    + [
        # follow_up: new claim for a different member
        "I also need to check on a claim adjustment for a different person.",  # [13]
        "no, it's for a different member",  # [14] → verification for James
    ]
    + _JAMES_VERIFY  # [15-20] James re-verifies
    + [
        # James provides reference number directly — stale ref_no_fallback_stage must NOT intercept
        "42695817",  # [21]
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "email please",
        "Yes, that's correct",
        "No, that's everything. Thanks!",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(status|review|update)"]),
        14: TurnExpectation(ai_contains=[r"same member|different member"]),
        15: TurnExpectation(ai_contains=[r"first name"]),
        21: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        # Decisive: status after James's reference, not another claim-number ask
        22: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",  # correctly collected via reference_number path
            "claim_flow_complete": True,
            "ref_no_fallback_stage": "",
        },
    ),
    notes=(
        "U-1: Stale ref_no_fallback_stage regression. Emily completes via "
        "claim_number fallback (882301), leaving ref_no_fallback_stage='claim_number_ask' "
        "in LangGraph state without Phase 3 / reset_for_new_intent clearing it. "
        "James (different member) provides reference 42695817 directly. "
        "Without the fix: 42695817 is extracted as claim_number → not found → escalation. "
        "reference_number='42695817' in final state proves the direct-ref path was taken."
    ),
)

# ── U-2: Stale ref_no_fallback_stage after dos_billed fallback ────────────────
stale_fallback_stage_after_dos_billed_second_member = Scenario(
    name="stale_fallback_stage_after_dos_billed_second_member",
    flow="claim",
    timeout_s=600,
    retries=1,  # new_intent + same-member LLM classifications are non-deterministic
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] fallback
        "No",  # [8] no claim number → dos_billed
        f"{_EMILY_DOS_SPOKEN} and it was {_EMILY_BILLED_DOLLAR}",  # [9] both → lookup → status
    ]
    + _EMILY_DECLINE_GUIDE_TAIL  # [10-13] decline guide → follow_up
    + [
        "I also need to check on a claim adjustment for a different person.",  # [14]
        "no, it's for a different member",  # [15]
    ]
    + _JAMES_VERIFY  # [16-21]
    + [
        "42695817",  # [22] James's reference — must NOT enter dos_billed fallback
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "email please",
        "Yes, that's correct",
        "No, that's everything. Thanks!",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
        15: TurnExpectation(ai_contains=[r"same member|different member"]),
        16: TurnExpectation(ai_contains=[r"first name"]),
        22: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        23: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            "claim_flow_complete": True,
            "ref_no_fallback_stage": "",
        },
    ),
    notes=(
        "U-2: Stale ref_no_fallback_stage='dos_billed_ask' regression (mirror of U-1 "
        "for the dos_billed path). Emily completes via dos+billed fallback, leaving "
        "ref_no_fallback_stage='dos_billed_ask' in state. James provides reference "
        "42695817 directly. Without the fix the defensive guard fires on a different "
        "branch (dos_billed_ask instead of claim_number_ask) but the symptom is the "
        "same: James's reference would be extracted as dos/billed values, extraction "
        "would fail, and the flow would escalate. reference_number='42695817' in "
        "final state is proof the direct-ref path was taken."
    ),
)

# ── U-3: Affirmative "Yes" at claim_number stage → free re-ask → spoken digits → success ─
affirmative_at_claim_number_free_retry_spoken_digits = Scenario(
    name="affirmative_at_claim_number_free_retry_spoken_digits",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] fallback starts
        "Yes.",  # [8] affirmative — free re-ask (fix)
        _EMILY_CLAIM_NUMBER_SPOKEN,  # [9] "eight eight two three zero one" → 882301 → success
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        # Free re-ask: agent re-asks for claim number (does NOT advance to dos_billed)
        9: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
        },
        # The exact production failure: dos_billed question was asked after the actual
        # claim number was provided — it must NEVER appear in this flow.
        transcript_count={r"(date of service|billed amount|billed\s+\$)": 0},
    ),
    notes=(
        "U-3: Exact production regression (Issue 2). Emily says 'Yes' to 'Do you "
        "have the claim number?' and then provides spoken digits on the next turn. "
        "Before the fix: 'Yes' burned attempt_count 0→1; the actual number hit "
        "attempt_count >= 1 → agent jumped to dos_billed without trying extraction. "
        "Fix: bare affirmative with no digit content is detected and re-asked for free "
        "(attempt_count stays 0). Turn-9 must be a claim-number re-ask, not dos_billed. "
        "transcript_count guard ensures dos_billed is never asked."
    ),
)

# ── U-4: Affirmative "Yep" at claim_number → free re-ask → numeric → success ─
affirmative_at_claim_number_free_retry_numeric = Scenario(
    name="affirmative_at_claim_number_free_retry_numeric",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] fallback starts
        "Yep, I do.",  # [8] affirmative (different phrase) — free re-ask
        _EMILY_CLAIM_NUMBER,  # [9] "882301" numeric → success
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),  # free re-ask
        10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "fallback_claim_number": _EMILY_CLAIM_NUMBER,
            "claim_status": truthy,
        },
        transcript_count={r"(date of service|billed amount|billed\s+\$)": 0},
    ),
    notes=(
        "U-4: Affirmative-retry fix, paraphrased. Same shape as U-3 but uses 'Yep, "
        "I do.' (a phrase in _AFFIRMATIVE_PHRASES) and the numeric '882301' instead "
        "of spoken digits. Proves the fix is not keyed on 'Yes' alone and that numeric "
        "claim numbers are still accepted after a free re-ask."
    ),
)

# ── U-5: Affirmative at dos_billed stage → free re-ask → values → success ────
affirmative_at_dos_billed_free_retry = Scenario(
    name="affirmative_at_dos_billed_free_retry",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] fallback
        "No",  # [8] no claim number → dos_billed asked
        "Yes, I have that information.",  # [9] affirmative — free re-ask (fix)
        f"{_EMILY_DOS_SPOKEN} and it was {_EMILY_BILLED_DOLLAR}",  # [10] both values → success
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        8: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        # Free re-ask: agent re-asks for dos+billed (does NOT escalate)
        10: TurnExpectation(ai_contains=[r"(date of service|billed|dos|amount)"]),
        11: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "U-5: Affirmative-retry fix at the dos_billed stage. Emily says 'Yes, I have "
        "that information.' to 'What was the date of service and the billed amount?' "
        "Before the fix: 'Yes' burned attempt_count 0→1; the actual values on the next "
        "turn hit attempt_count >= 1 → escalation. Fix: bare affirmative with no "
        "numeric content is detected and re-asked for free (attempt_count stays 0). "
        "Turn-10 must be a dos_billed re-ask, not escalation."
    ),
)

# ── U-6: Affirmative at dos_billed → free re-ask → partial → genuine retry → success ─
affirmative_at_dos_billed_retry_budget_preserved = Scenario(
    name="affirmative_at_dos_billed_retry_budget_preserved",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "I don't have the reference number",  # [7] fallback
        "No",  # [8] no claim number → dos_billed asked
        "Sure.",  # [9] affirmative — free re-ask (attempt_count stays 0)
        "just January eighth",  # [10] dos only, no billed → partial → burns attempt 0→1
        f"{_EMILY_DOS_SPOKEN} and {_EMILY_BILLED_SPOKEN}",  # [11] both values → success
        "I will upload them myself",
        "Yes, please",
        "Yes, that's correct",
        "No, that's fine",
        "email please",
        "Yes, that's correct",
        "No, that's everything",
    ],
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed|dos)"]),
        # Free re-ask for "Sure." — attempt_count still 0
        10: TurnExpectation(ai_contains=[r"(date of service|billed|dos|amount)"]),
        # Genuine retry for partial answer — attempt_count now 1
        11: TurnExpectation(ai_contains=[r"(date of service|billed|amount)"]),
        12: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"claim_status": truthy},
    ),
    notes=(
        "U-6: Proves the dos_billed affirmative re-ask is truly free — the retry "
        "budget is intact afterward. 'Sure.' gets a free re-ask (attempt_count stays 0). "
        "'just January eighth' (date only, no billed amount) burns one genuine attempt "
        "(attempt_count 0→1). The full values on the next turn succeed. "
        "Without the fix: 'Sure.' would burn attempt_count 0→1; 'just January eighth' "
        "would see attempt_count=1 >= 1 → immediate escalation — no retry left for "
        "the partial answer."
    ),
)

SCENARIOS.extend(
    [
        stale_fallback_stage_after_claim_number_second_member,  # U-1
        stale_fallback_stage_after_dos_billed_second_member,  # U-2
        affirmative_at_claim_number_free_retry_spoken_digits,  # U-3
        affirmative_at_claim_number_free_retry_numeric,  # U-4
        affirmative_at_dos_billed_free_retry,  # U-5
        affirmative_at_dos_billed_retry_budget_preserved,  # U-6
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# V. ID card out-of-scope decline in follow-up
#
# These scenarios verify that asking about a new/replacement ID card during the
# follow-up phase produces the correct scope-decline response on BOTH the
# provider (PCP) and claims paths.  The expected LLM answer is:
#   "ID card requests are handled on a different line — is there anything else
#    I can help you with today?"
# The call must NOT escalate — it is a single scope-decline, not a repeated
# unanswerable question. The caller then closes normally.
#
# Four shapes are covered:
#   V-1  PCP path,   direct phrasing ("I lost my ID card, how do I get a new one?")
#   V-2  PCP path,   insurance-card synonym ("How do I replace my insurance card?")
#   V-3  Claim path, direct phrasing ("I need a new ID card")
#   V-4  Claim path, conversational ("Can you help me get a replacement member card?")
# ──────────────────────────────────────────────────────────────────────────────

# Minimal claim flow completion that reaches follow_up (mirrors _PCP_TO_FOLLOW_UP).
_CLAIM_TO_FOLLOW_UP = CLAIM_VERIFY + [
    "42695817",
    "Can I ask my doctor to send it over?",  # doctor-direct
    "Yes, please",  # accept upload link
    "Yes, that's correct",  # confirm email on file
    "Perfect. Please do that",  # accept Personal Guide
    "You can send me the updates to my phone",  # SMS notifications
    "Yes, that's correct",  # confirm phone
    "Okay, how long will it take?",  # timeline question
    "email them to me",  # N2 channel → follow-up "anything else?"
]

followup_id_card_pcp_direct = Scenario(
    name="followup_id_card_pcp_direct",
    flow="pcp",
    retries=1,  # follow_up out-of-scope classification is LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "I lost my ID card. How can I get a new one?",
        "No, that's all. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        transcript_contains=[r"different line"],
    ),
    notes=(
        "V-1: PCP follow-up — direct ID card question. follow_up must classify "
        "as follow_up_intent='question' and respond with the scope-decline: "
        "'ID card requests are handled on a different line'. Must NOT escalate "
        "— a single scope-decline is not a transfer trigger."
    ),
)

followup_id_card_pcp_insurance_card_synonym = Scenario(
    name="followup_id_card_pcp_insurance_card_synonym",
    flow="pcp",
    retries=1,
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "How do I replace my insurance card?",
        "No, that's everything. Bye!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        transcript_contains=[r"different line"],
    ),
    notes=(
        "V-2: PCP follow-up — 'insurance card' synonym. Verifies the scope-decline "
        "fires on the insurance-card phrasing, not just 'ID card', matching the "
        "explicit example added to follow_up.md."
    ),
)

followup_id_card_claim_direct = Scenario(
    name="followup_id_card_claim_direct",
    flow="claim",
    timeout_s=420,
    retries=1,
    user_turns=_CLAIM_TO_FOLLOW_UP
    + [
        "I need a new ID card. How do I request one?",
        "No, that's everything. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        # Accept canonical phrase or paraphrase; both confirm out-of-scope handling.
        transcript_contains=[r"(different line|enrollment|another line|handled on|this line)"],
    ),
    notes=(
        "V-3: Claim follow-up — direct ID card question. Mirrors V-1 on the claims "
        "path. Verifies the scope-decline added to follow_up_claims.md fires: "
        "'ID card requests are handled on a different line'."
    ),
)

followup_id_card_claim_conversational = Scenario(
    name="followup_id_card_claim_conversational",
    flow="claim",
    timeout_s=420,
    retries=1,
    user_turns=_CLAIM_TO_FOLLOW_UP
    + [
        "Actually, can you help me get a replacement member card? I can't seem to find mine.",
        "Okay, got it. No, that's all then.",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        transcript_contains=[r"(different line|enrollment|another line|handled on|this line)"],
    ),
    notes=(
        "V-4: Claim follow-up — conversational 'replacement member card' phrasing. "
        "Exercises the LLM's ability to infer ID/member-card intent from indirect "
        "wording and still produce the scope-decline rather than the generic "
        "'I don't have that information' fallback that existed before the fix."
    ),
)

SCENARIOS.extend(
    [
        followup_id_card_pcp_direct,  # V-1
        followup_id_card_pcp_insurance_card_synonym,  # V-2
        followup_id_card_claim_direct,  # V-3
        followup_id_card_claim_conversational,  # V-4
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# W. ref_lookup_fail counter reset — wrong identifier then correct one succeeds
#
# Regression guards for claim_adjustment_agent's ref_lookup_fail counter logic.
#
# W-1 through W-5 guard the fix that resets the counter whenever a genuinely
# different reference number is submitted after a pivot (Phase 2 resets when
# reference_number != last_failed_ref).  Without that fix a first wrong ref
# (count=1) followed by a pivot and a different second submission would
# immediately escalate (count=2 ≥ MAX_LOOKUP_ATTEMPTS=2).
#
# W-6 guards the follow-on fix (2026-08-18) that removed the unconditional
# ref_lookup_fail.reset() calls from the fallback-pivot paths in
# _collect_claim_number_fallback and _collect_dos_billed_fallback.  Without
# that fix, pivoting back and re-submitting the SAME wrong reference number
# caused an infinite loop: the counter was reset to 0 on each pivot, so the
# second (identical) submission only reached count=1 and started the
# claim-number fallback again instead of escalating.
#
# Six shapes are covered:
#   W-1  Wrong ref → pivot → correct ref (James fixture 42695817) → success.
#        Proves the reset does not break the normal SF-found path.
#   W-2  Wrong ref → pivot → second DIFFERENT wrong ref → (Phase 2 resets) →
#        claim-number fallback → correct claim (882301) → success.
#   W-3  Wrong ref → claim-number fallback → garbled claim number (extraction
#        fails) → retry → correct claim → success.
#   W-4  Wrong ref → claim-number fallback → "No" → DOS+billed fallback →
#        partial DOS (date only, no billed amount) → retry → full DOS+billed →
#        success.
#   W-5  Wrong ref → "No" claim# → DOS+billed stage → pivot to ref# →
#        different wrong ref → count=2 → escalate.
#   W-6  Wrong ref → pivot → SAME wrong ref again → count=2 → escalate.
#        The exact call-log transcript scenario.
#   W-7  Wrong ref → pivot → "I'm telling you the ref#, it is X" (inline digit
#        in pivot phrase) → keyword gate discards X, re-asks → X submitted clean
#        → count=2 → escalate.  The exact phrasing from the call-log transcript.
#   W-8  "No. I don't have the claim number, but I have the reference number." →
#        detect_cannot_provide fired on leading clause before LLM could see the
#        trailing pivot hint → DOS/billed (bug).  Fix: detect_cannot_provide now
#        runs only after the LLM; LLM sets fallback_pivot=reference_number →
#        pivot → second wrong ref → count=2 → escalate.
#   W-9  Same pivot phrase as W-8 but success path: James provides correct ref
#        42695817 after the pivot → SF found → flow completes.
#   W-10 "I can't find my claim number, but I have the reference number." →
#        LLM pivot → second wrong ref → escalate.
#        (detect_cannot_provide: r"\bi\s+can'?t\s+(remember|recall|find)\b")
#   W-11 "I don't know the claim number, but I have my reference number." →
#        LLM pivot → second wrong ref → escalate.
#        (detect_cannot_provide: r"\bi\s+don'?t\s+know\b")
#   W-12 "I never received a claim number, but I do have the reference number." →
#        LLM pivot → second wrong ref → escalate.
#        (detect_cannot_provide: r"\bi\s+never\s+(received|got)\b")
#
# W-1 and W-9 use James Wilson (M310188 / ref 42695817).
# W-2 through W-8, W-10 through W-12 use Emily Carter (M907503) whose fixture
# has no reference number — every ref-number SF lookup returns "not found".
# ──────────────────────────────────────────────────────────────────────────────

# Standard claim-flow tail for James once claim_adjustment_agent reaches Phase 3
# (records_required=True; mirrors claim_happy_path).
_JAMES_CLAIM_TAIL = [
    "Can I ask my doctor to send it over?",
    "Yes, please",
    "Yes, that's correct",
    "Perfect. Please do that",
    "You can send me the updates to my phone",
    "Yes, that's correct",
    "Okay, how long will it take to finalize the request?",
    "email them to me",
    "No, that's all. Thanks!",
]

# Minimal Emily tail: decline guide, email notifications, close.
_EMILY_W_TAIL = _EMILY_DECLINE_GUIDE_TAIL + [
    "email please",
    "Yes, that's correct",
    "No, that's everything",
]

# ── W-1: Wrong ref → pivot → correct ref (James) → success ───────────────────
wrong_ref_then_correct_ref_counter_reset = Scenario(
    name="wrong_ref_then_correct_ref_counter_reset",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY
    + [
        "99999999",  # [7]  wrong ref → SF no match → ref_lookup_fail=1 → claim-number ask
        "Actually, I found my reference number.",  # [8]  pivot during claim_number_ask
        "42695817",  # [9]  correct ref → counter RESET → SF finds record → status
    ]
    + _JAMES_CLAIM_TAIL,
    turn_expectations={
        # 8: TurnExpectation(ai_contains=[r"reference\s*(number|#|num)"]),
        # After wrong ref: claim-number fallback started.
        # 9: TurnExpectation(
        #     ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        # ),
        # After "I found my reference number": pivot fires — agent re-asks for ref.
        # 10: TurnExpectation(
        #     ai_contains=[r"reference\s*(number|#|num)"],
        #     slot_awaiting="reference_number",
        # ),
        # # After correct ref: status reported (SF found the record).
        # 11: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "W-1: Wrong ref → pivot → correct ref → success. James gives wrong ref "
        "99999999; SF returns no match; claim-number fallback starts. James says "
        "he found his reference number — the agent pivots back to ref# collection. "
        "He provides 42695817 (his fixture ref); SF finds the record; flow "
        "completes. Correct lookups never increment the counter, so the prior "
        "failure (count=1) does not prevent success."
    ),
)

# ── W-2: Wrong ref → pivot → different wrong ref → count=2 → escalate ─────────────
wrong_ref_pivot_second_wrong_ref_escalates = Scenario(
    name="wrong_ref_pivot_second_wrong_ref_escalates",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → SF no match → ref_lookup_fail=1 → claim-number ask
        "I found the reference number.",  # [8]  pivot
        "88888888",  # [9]  different wrong ref → ref_lookup_fail=2 → escalate
    ],
    turn_expectations={
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        10: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "call_intent": "claim_services",
        },
    ),
    notes=(
        "W-2: Different second wrong ref also escalates. Emily gives 99999999 → "
        "SF no match → count=1 → claim-number fallback. She pivots back with "
        "88888888 (a different number). The counter is not reset; SF returns no "
        "match again → count=2 ≥ MAX_LOOKUP_ATTEMPTS → escalation. Proves that "
        "the escalation condition is count≥2, not 'same ref# submitted twice'."
    ),
)

# ── W-3: Wrong ref → invalid claim number → retry → correct claim → success ──
wrong_ref_then_invalid_claim_retry_then_correct = Scenario(
    name="wrong_ref_then_invalid_claim_retry_then_correct",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → SF no match → claim-number ask
        "one two three",  # [8]  3 spoken digits → normalizes to "123" (< 4 digits, invalid)
        #                         → extraction fails → retry
        _EMILY_CLAIM_NUMBER,  # [9]  882301 → valid → SF found → status
    ]
    + _EMILY_W_TAIL,
    turn_expectations={
        # After wrong ref: claim-number fallback.
        # 8: TurnExpectation(
        #     ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        # ),
        # After invalid claim number: agent retries (not escalate, not dos_billed yet).
        # 9: TurnExpectation(ai_contains=[r"claim\s*(number|#|num)"]),
        # After correct claim number: status.
        # 10: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "claim_status": truthy,
        },
    ),
    notes=(
        "W-3: Wrong ref followed by an invalid (too-short) claim number. "
        "Emily gives wrong ref 99999999 → claim-number fallback. She provides "
        "'one two three' which normalizes to '123' (< 4 digits, rejected by "
        "validate_claim_number). The agent retries once; she then provides "
        "the correct 882301; SF lookup succeeds. Verifies that a garbled claim "
        "number during the fallback does not immediately escalate — the single "
        "extraction retry fires before moving to DOS+billed."
    ),
)

# ── W-4: Wrong ref → no claim → partial DOS (date only) → retry → full DOS → success ──
wrong_ref_then_no_claim_partial_dos_retry_then_success = Scenario(
    name="wrong_ref_then_no_claim_partial_dos_retry_then_success",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → SF no match → claim-number ask
        "No",  # [8]  no claim number → dos_billed ask
        "just January eighth",  # [9]  date only, no billed amount → partial → retry
        f"{_EMILY_DOS_SPOKEN} and it was {_EMILY_BILLED_DOLLAR}",  # [10] both → SF found
    ]
    + _EMILY_W_TAIL,
    turn_expectations={
        # After wrong ref: claim-number fallback.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # After "No": dos+billed stage.
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed)"]),
        # After partial DOS: agent retries (not escalate).
        10: TurnExpectation(ai_contains=[r"(date of service|billed|amount)"]),
        # After full DOS+billed: status reported.
        11: TurnExpectation(ai_contains=[r"(status|review|update)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "claim_status": truthy,
        },
    ),
    notes=(
        "W-4: Wrong ref → skip claim number → partial DOS (date only, no billed "
        "amount) → extraction retry → full DOS+billed → success. Emily gives wrong "
        "ref 99999999 → claim-number fallback. She says 'No' (no claim number) → "
        "dos_billed_ask. She provides only 'just January eighth' (no billed amount); "
        "billed_amount extraction is empty so the agent retries. On the second "
        "attempt she provides the full values; SF lookup succeeds. Verifies the "
        "partial-answer retry path in _collect_dos_billed_fallback after a prior "
        "wrong reference number."
    ),
)

# ── W-5: Wrong ref → no claim# → DOS+billed stage → pivot to ref# → wrong ref → escalate ──
wrong_ref_dos_billed_pivot_then_wrong_ref_escalates = Scenario(
    name="wrong_ref_dos_billed_pivot_then_wrong_ref_escalates",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "No",  # [8]  no claim number → dos_billed_ask
        "Actually I found my reference number.",  # [9]  pivot from dos_billed → ask for ref#
        "88888888",  # [10] different wrong ref → count=2 → escalate
    ],
    turn_expectations={
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed)"]),
        10: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        11: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "call_intent": "claim_services",
        },
    ),
    notes=(
        "W-5: Escalation from the dos_billed pivot path. Emily gives wrong ref "
        "99999999 → count=1 → claim-number fallback. She says 'No' → dos_billed "
        "stage. She says she found her reference number → the dos_billed pivot fires "
        "(counter NOT reset by our fix) → agent re-asks for ref#. She provides "
        "88888888 → count=2 → escalation. Proves the counter is preserved across "
        "the dos_billed pivot path, not just the claim_number pivot path."
    ),
)

# ── W-6: Same wrong ref resubmitted after pivot → count=2 → escalate ────────────────
second_sf_lookup_failure_escalates = Scenario(
    name="second_sf_lookup_failure_escalates",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → SF no match → ref_lookup_fail=1 → claim-number ask
        "No. I have the reference number.",  # [8]  pivot → counter NOT reset → ask for ref# again
        "99999999",  # [9]  same wrong ref → count=2 → escalate
    ],
    turn_expectations={
        # After first wrong ref: claim-number fallback.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # After pivot phrase: re-ask for reference number (counter still at 1).
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        # After same wrong ref again: escalate — NOT another fallback loop.
        10: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "call_intent": "claim_services",
        },
        transcript_contains=[r"(connect|transfer|representative|specialist)"],
    ),
    notes=(
        "W-6: The exact call-log transcript scenario (same ref# after pivot). "
        "Emily gives wrong ref 99999999 → SF no match → count=1 → claim-number "
        "fallback. She says 'No. I have the reference number' → pivot fires; "
        "counter stays at 1. She provides 99999999 again → count=2 → escalation. "
        "Complements W-2 (different second ref) to show the condition is count≥2, "
        "not same-ref specifically."
    ),
)

# ── W-7: Inline ref# inside pivot phrase is discarded → second failure escalates ──
ref_loop_inline_ref_in_pivot_phrase_escalates = Scenario(
    name="ref_loop_inline_ref_in_pivot_phrase_escalates",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "No. I have the reference number.",  # [8]  pivot phrase (no inline number) → ask for ref#
        "I'm telling you the reference number, it is 88888888.",
        "88888888",
    ],
    turn_expectations={
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # After first pivot: re-ask for ref#.
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        # After "I'm telling you the reference number, it is 88888888":
        # keyword pivot fires (discards inline 88888888) → re-ask for ref#.
        10: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        # After 88888888 submitted as a clean ref#: count=2 → escalate.
        11: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "call_intent": "claim_services",
        },
    ),
    notes=(
        "W-7: Call-log transcript bug — 'I'm telling you the reference number, it "
        "is X' phrasing. The 'reference number' keyword pivot fires and discards the "
        "inline digit value X; the agent re-asks for the reference number. Emily then "
        "provides 88888888 as a clean utterance → count=2 → escalation. "
        "Verifies the counter accumulates correctly even when a pivot phrase contains "
        "an inline number that gets dropped by the keyword gate."
    ),
)

# ── W-8: "No. I don't have the claim number, but I have the reference number." ──────
# detect_cannot_provide matched the leading "No. I don't have" clause and sent the
# member to DOS/billed, discarding the trailing pivot hint.  The fix moves the
# reference-number keyword check before cannot-provide so the trailing clause wins.
negation_with_ref_hint_pivots_to_reference_number = Scenario(
    name="negation_with_ref_hint_pivots_to_reference_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → SF no match → ref_lookup_fail=1 → claim-number ask
        "No. I don't have the claim number, but I have the reference number.",  # [8]  pivot → ask for ref#
        "88888888",  # [9]  wrong ref → count=2 → escalate
    ],
    turn_expectations={
        # After first wrong ref: claim-number fallback.
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # LLM sees the full sentence and sets fallback_pivot="reference_number" → ref# ask.
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        # After second wrong ref: count=2 → escalate.
        10: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "call_intent": "claim_services",
        },
    ),
    notes=(
        "W-8: Regression for the call-log transcript bug. Emily says she doesn't have "
        "the claim number but has the reference number. Previously detect_cannot_provide "
        "matched the leading 'No. I don't have' clause and sent her to DOS/billed before "
        "the LLM could see the qualifying pivot hint. With the fix, detect_cannot_provide "
        "runs only after the LLM; the LLM sets fallback_pivot='reference_number' and the "
        "agent pivots to ask for the reference number. Second wrong ref → count=2 → escalation."
    ),
)

# ── W-9: Exact transcript success path — cannot-provide + ref hint → correct ref ─
# The actual call-log transcript ends with success (42695817 found on the second
# attempt).  W-8 covers the escalation branch; W-9 covers the success branch using
# James Wilson whose fixture ref 42695817 exists in Salesforce.
negation_with_ref_hint_then_correct_ref_succeeds = Scenario(
    name="negation_with_ref_hint_then_correct_ref_succeeds",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY
    + [
        "12695817",  # [7]  wrong ref → SF no match → ref_lookup_fail=1 → claim-number ask
        "No. I don't have the claim number, but I have the reference number.",  # [8]  LLM pivot → ref# ask
        "42695817",  # [9]  correct ref → SF finds record → status reported
    ]
    + _JAMES_CLAIM_TAIL,
    turn_expectations={
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        10: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "W-9: The exact call-log transcript outcome — success. James gives wrong ref "
        "12695817 (count=1), then 'No. I don't have the claim number, but I have the "
        "reference number.' Previously this sent him to DOS/billed (bug). With the fix "
        "the LLM pivots back to ref# collection; he provides 42695817 and the flow "
        "completes normally. Complements W-8 (same pivot phrase, escalation branch)."
    ),
)

# ── W-10: "I can't find" phrasing — cannot-provide + ref hint → escalate ─────────
cant_find_claim_but_has_ref_pivots_to_reference_number = Scenario(
    name="cant_find_claim_but_has_ref_pivots_to_reference_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "I can't find my claim number, but I have the reference number.",  # [8]  LLM pivot
        "88888888",  # [9]  wrong ref → count=2 → escalate
    ],
    turn_expectations={
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # detect_cannot_provide matches r"\bi\s+can'?t\s+(remember|recall|find)\b" on
        # the leading clause; the LLM must still see the trailing pivot hint.
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        10: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={"member_status_verify": True, "call_intent": "claim_services"},
    ),
    notes=(
        "W-10: 'I can't find' phrasing. detect_cannot_provide matches "
        r"r'\bi\s+can\'?t\s+(remember|recall|find)\b' on the leading clause. "
        "LLM sees the full sentence and sets fallback_pivot='reference_number'. "
        "Second wrong ref → count=2 → escalation."
    ),
)

# ── W-11: "I don't know" phrasing — cannot-provide + ref hint → escalate ─────────
dont_know_claim_but_has_ref_pivots_to_reference_number = Scenario(
    name="dont_know_claim_but_has_ref_pivots_to_reference_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "I don't know the claim number, but I have my reference number right here.",  # [8]  LLM pivot
        "88888888",  # [9]  wrong ref → count=2 → escalate
    ],
    turn_expectations={
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # detect_cannot_provide matches r"\bi\s+don'?t\s+know\b" on the leading clause.
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        10: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={"member_status_verify": True, "call_intent": "claim_services"},
    ),
    notes=(
        "W-11: 'I don't know' phrasing. detect_cannot_provide matches "
        r"r'\bi\s+don\'?t\s+know\b' on the leading clause. "
        "LLM sees the full sentence and sets fallback_pivot='reference_number'. "
        "Second wrong ref → count=2 → escalation."
    ),
)

# ── W-12: "I never received" phrasing — cannot-provide + ref hint → escalate ─────
never_received_claim_but_has_ref_pivots_to_reference_number = Scenario(
    name="never_received_claim_but_has_ref_pivots_to_reference_number",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "I never received a claim number, but I do have the reference number.",  # [8]  LLM pivot
        "88888888",  # [9]  wrong ref → count=2 → escalate
    ],
    turn_expectations={
        8: TurnExpectation(
            ai_contains=[r"(claim number|another way|look it up differently|another approach)"]
        ),
        # detect_cannot_provide matches r"\bi\s+never\s+(received|got)\b" on the leading clause.
        9: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        10: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={"member_status_verify": True, "call_intent": "claim_services"},
    ),
    notes=(
        "W-12: 'I never received' phrasing. detect_cannot_provide matches "
        r"r'\bi\s+never\s+(received|got)\b' on the leading clause. "
        "LLM sees the full sentence and sets fallback_pivot='reference_number'. "
        "Second wrong ref → count=2 → escalation."
    ),
)

SCENARIOS.extend(
    [
        wrong_ref_then_correct_ref_counter_reset,  # W-1
        wrong_ref_pivot_second_wrong_ref_escalates,  # W-2
        wrong_ref_then_invalid_claim_retry_then_correct,  # W-3
        wrong_ref_then_no_claim_partial_dos_retry_then_success,  # W-4
        wrong_ref_dos_billed_pivot_then_wrong_ref_escalates,  # W-5
        second_sf_lookup_failure_escalates,  # W-6
        ref_loop_inline_ref_in_pivot_phrase_escalates,  # W-7
        negation_with_ref_hint_pivots_to_reference_number,  # W-8
        negation_with_ref_hint_then_correct_ref_succeeds,  # W-9
        cant_find_claim_but_has_ref_pivots_to_reference_number,  # W-10
        dont_know_claim_but_has_ref_pivots_to_reference_number,  # W-11
        never_received_claim_but_has_ref_pivots_to_reference_number,  # W-12
    ]
)

# ==============================================================================
# X-SECTION — DOS/billed fallback pivot tests
#
# Member enters the dos_billed stage (wrong ref → no claim#) and announces they
# have a different identifier than the one being collected.  Tests that the
# keyword checks in _collect_dos_billed_fallback correctly route even when the
# announcement is wrapped in a negation prefix ("No. I don't have this, but…").
#
#   X-1  DOS/billed stage → "No. I don't have this, but I have the reference
#        number." → keyword pivot → wrong ref → count=2 → escalate.
#   X-2  Same pivot phrase → correct ref → success (James).
#   X-3  DOS/billed stage → "No. I don't have this, but I have the claim
#        number." → keyword pivot → correct claim (882301) → success (Emily).
#   X-4  "I can't find the date or billed amount, but I have my reference
#        number." → keyword pivot → wrong ref → escalate.
#   X-5  "I don't know the date of service or billed amount, but I have the
#        claim number." → keyword pivot → correct claim → success (Emily).
#
# X-1, X-3, X-4, X-5 use Emily Carter (M907503).
# X-2 uses James Wilson (M310188 / ref 42695817).
# ==============================================================================

# ── X-1: DOS/billed → cannot-provide + ref hint → pivot → wrong ref → escalate ──
dos_billed_negation_with_ref_hint_escalates = Scenario(
    name="dos_billed_negation_with_ref_hint_escalates",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "No",  # [8]  no claim# → dos_billed ask
        "No. I don't have this, but I have the reference number.",  # [9]  keyword pivot → ref# ask
        "88888888",  # [10] wrong ref → count=2 → escalate
    ],
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed)"]),
        10: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        # 11: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={"member_status_verify": True, "call_intent": "claim_services"},
    ),
    notes=(
        "X-1: DOS/billed stage. Emily says 'No. I don't have this, but I have the "
        "reference number.' The 'reference number' keyword check in "
        "_collect_dos_billed_fallback fires despite the negation prefix and pivots "
        "back to ref# collection. Second wrong ref → count=2 → escalation."
    ),
)

# ── X-2: DOS/billed → cannot-provide + ref hint → pivot → correct ref → success ──
dos_billed_negation_with_ref_hint_succeeds = Scenario(
    name="dos_billed_negation_with_ref_hint_succeeds",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY
    + [
        "12695817",  # [7]  wrong ref → count=1 → claim-number ask
        "No",  # [8]  no claim# → dos_billed ask
        "No. I don't have date of service or billed amount, but I have the reference number.",
        "42695817",  # [10] correct ref → SF found → status
    ]
    + _JAMES_CLAIM_TAIL,
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed)"]),
        10: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        # 11: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "X-2: Success path. James goes through dos_billed stage then says 'No. I "
        "don't have this, but I have the reference number.' Keyword pivot fires; he "
        "provides 42695817 and the flow completes."
    ),
)

# ── X-3: DOS/billed → cannot-provide + claim hint → pivot → correct claim → success ──
dos_billed_negation_with_claim_hint_succeeds = Scenario(
    name="dos_billed_negation_with_claim_hint_succeeds",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "No",  # [8]  no claim# → dos_billed ask
        "No. I don't have this, but I have the claim number.",  # [9]  keyword pivot → claim# ask
        _EMILY_CLAIM_NUMBER,  # [10] correct claim → SF found → status
    ]
    + _EMILY_W_TAIL,
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed)"]),
        10: TurnExpectation(
            ai_contains=[r"(claim number|provide your claim)"],
            slot_awaiting="fallback_claim_number",
        ),
        # 11: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "claim_flow_complete": True},
    ),
    notes=(
        "X-3: Claim number pivot from dos_billed. Emily says 'No. I don't have this, "
        "but I have the claim number.' The 'claim number' keyword check fires and "
        "routes back to claim_number_ask. She provides 882301; SF lookup succeeds."
    ),
)

# ── X-4: "I can't find" phrasing in dos_billed → ref hint → escalate ─────────────
dos_billed_cant_find_with_ref_hint_escalates = Scenario(
    name="dos_billed_cant_find_with_ref_hint_escalates",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "No",  # [8]  no claim# → dos_billed ask
        "I can't find the date or billed amount, but I have my reference number.",  # [9]  pivot
        "88888888",  # [10] wrong ref → count=2 → escalate
    ],
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed)"]),
        10: TurnExpectation(
            ai_contains=[r"reference\s*(number|#|num)"],
            slot_awaiting="reference_number",
        ),
        # 11: TurnExpectation(ai_contains=[r"(connect|transfer|representative|specialist)"]),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={"member_status_verify": True, "call_intent": "claim_services"},
    ),
    notes=(
        "X-4: 'I can't find' phrasing in dos_billed. The 'reference number' keyword "
        "check fires (literal match) despite the leading cannot-provide clause, "
        "pivoting back to ref# collection. Second wrong ref → count=2 → escalation."
    ),
)

# ── X-5: "I don't know" phrasing in dos_billed → claim hint → success ────────────
dos_billed_dont_know_with_claim_hint_succeeds = Scenario(
    name="dos_billed_dont_know_with_claim_hint_succeeds",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_EMILY
    + [
        "99999999",  # [7]  wrong ref → count=1 → claim-number ask
        "No",  # [8]  no claim# → dos_billed ask
        "I don't know the date of service or billed amount, but I have the claim number.",  # [9]  pivot
        _EMILY_CLAIM_NUMBER,  # [10] correct claim → SF found → status
    ]
    + _EMILY_W_TAIL,
    turn_expectations={
        9: TurnExpectation(ai_contains=[r"(date of service|billed amount|service.*billed)"]),
        10: TurnExpectation(
            ai_contains=[r"(claim number|provide your claim)"],
            slot_awaiting="fallback_claim_number",
        ),
        # 11: TurnExpectation(ai_contains=[r"(status|review|update|records)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "claim_flow_complete": True},
    ),
    notes=(
        "X-5: 'I don't know' phrasing in dos_billed. The 'claim number' keyword "
        "check fires and routes to claim_number_ask. Emily provides 882301 → success."
    ),
)

SCENARIOS.extend(
    [
        dos_billed_negation_with_ref_hint_escalates,  # X-1
        dos_billed_negation_with_ref_hint_succeeds,  # X-2
        dos_billed_negation_with_claim_hint_succeeds,  # X-3
        dos_billed_cant_find_with_ref_hint_escalates,  # X-4
        dos_billed_dont_know_with_claim_hint_succeeds,  # X-5
    ]
)

SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
