"""
Deterministic slot classifier for PCP evaluation flow.
Keyword-based matching — clarification/correction checks run before slot checks
so that re-ask and ack messages are not misclassified as regular slot prompts.

FIXES (stability — these were sources of intermittent eval failures):
  1. The zip_update list contained "provide.*zip" used in a plain substring
     test — regex syntax in a `in` check can never match. Replaced with a
     real regex plus literal phrases, so rephrased ZIP re-asks ("could you
     provide your ZIP code?") classify as zip_update, not zip_confirm.
  2. benefits_offer is now checked BEFORE the zip checks. The dispatch
     confirmation can now include the updated ZIP ("...for your current ZIP
     code 02139 within 30 minutes. ...would you like to also get the
     benefits..."); the old order classified that message as zip_confirm and
     produced the wrong fallback ground truth ("Yes, that's correct").
  3. The bare "zip code" → zip_confirm fallback is guarded against dispatch
     messages ("within 30 minutes").
  4. coach_offer requires an actual offer question ("would you like" /
     "want" / "do you want"), so the Care-Coach *dispatch* message
     ("We will send the Care Coach details... Is there anything else?")
     classifies as follow_up instead of coach_offer.
  5. Claim flow: the N1 ask can be phrased "phone or email" — added to the
     channel phrase lists (it previously fell through to "other").
  6. Claim flow: N1 bridge vs N2 ask disambiguation. Both contain a channel
     phrase + "updates"; N1 mentions the provider outreach / Personal Guide,
     N2 mentions progress updates "in this request". The guide-scheduled /
     N1-bridge check now runs BEFORE the N2 check, and the N2 check excludes
     provider-outreach wording.
"""

import re

# Rephrased "give me a new ZIP" asks — checked with re.search, not substring.
_ZIP_UPDATE_RE = re.compile(r"(what is|what's|provide|give me|share)[^.?!]{0,40}zip")

# Spelled-out name pattern unique to name readback messages (e.g. "e-m-i-l-y c-a-r-t-e-r").
_NAME_SPELLED_RE = re.compile(r"[a-z]-[a-z]-[a-z]")

# Channel phrasings used by both notification asks (N1 and N2).
_CHANNEL_PHRASES = [
    "sms or email",
    "email or sms",
    "phone or email",
    "email or phone",
    "how do you want to be notified",
    "how would you like to be notified",
]

# Wording that marks the N1 provider-outreach notification context.
_N1_OUTREACH_PHRASES = [
    "provider outreach",
    "personal guide",
    "reached your doctor",
    "status of the provider",
    "contact the provider",
]


def classify_ai_slot(ai_message: str, flow: str = "pcp") -> str:  # noqa: C901
    """Return the slot label that best describes what the AI message is collecting."""
    if flow == "claim":
        return classify_ai_slot_claim(ai_message)

    if not ai_message:
        return "other"

    ai_lower = ai_message.lower()

    # --- Re-ask / clarification (must precede slot checks) ---
    if any(
        phrase in ai_lower
        for phrase in [
            "i'm sorry, i didn't quite catch",
            "i didn't quite catch",
            "could you repeat",
            "one more time",
            "i wasn't able to catch",
            "didn't catch that",
            "could you say that again",
            "could you please repeat",
        ]
    ):
        return "clarification"

    # --- Correction acknowledgement (must precede slot checks) ---
    if any(
        phrase in ai_lower
        for phrase in [
            "got it, i've updated",
            "got it i've updated",
            "i've updated",
            "i've corrected",
            "i have updated",
            "i have corrected",
            "thank you for the correction",
        ]
    ):
        return "correction_ack"

    # --- Greeting / intent ---
    if (
        any(
            phrase in ai_lower
            for phrase in [
                "how can i assist",
                "how may i assist",
                "how can i help",
                "how i can help",
                "please tell me how i can",
            ]
        )
        and "is there anything else" not in ai_lower
    ):
        return "intent_selection"

    # --- Name readback confirmation (before first/last name checks) ---
    if _NAME_SPELLED_RE.search(ai_lower):
        return "name_confirmed"

    # --- PCP collection slots ---
    if "first name" in ai_lower:
        return "first_name"

    if "last name" in ai_lower:
        return "last_name"

    if "member id" in ai_lower and "date of birth" not in ai_lower:
        return "member_id"

    if "date of birth" in ai_lower:
        return "dob"

    if "plan holder" in ai_lower or "subscriber" in ai_lower or "dependent" in ai_lower:
        return "subscriber_confirm"

    # provider_type — the agent draws from a pool; variants say "type of
    # care" / "kind of provider" rather than the scripted "type of provider".
    if any(
        phrase in ai_lower
        for phrase in [
            "type of provider",
            "what type of provider",
            "type of care",
            "kind of care",
            "kind of provider",
            "type of specialist",
            "type of doctor",
            "kind of doctor",
            "care can i help you find",
        ]
    ):
        return "provider_type"

    # --- Benefits offer (MUST precede zip checks) ---
    # The dispatch confirmation may now include the updated ZIP
    # ("...for your current ZIP code 02139 within 30 minutes. ...benefits...").
    # Without this ordering, that combined message matched the bare
    # "zip code" → zip_confirm fallback below.
    if "benefits" in ai_lower and any(
        phrase in ai_lower
        for phrase in [
            "would you like to also",
            "also get the benefits",
            "hear about",
            "would you also like",
            "would that be helpful",
        ]
    ):
        return "benefits_offer"

    # Explicit confirmation of an on-file zip (must precede zip_update to avoid
    # "correct zip" in "Is that the correct ZIP code?" matching zip_update)
    if "zip code" in ai_lower and any(
        phrase in ai_lower
        for phrase in [
            "is that correct",
            "is this correct",
            "is that right",
            "is this right",
            "is that the right",
            "is this the right",
            "on file",
            "to confirm",
            "for your search",
        ]
    ):
        return "zip_confirm"

    # ZIP re-ask: agent explicitly asks for a new/different zip code.
    # NOTE: "provide.*zip" was previously listed here as a SUBSTRING — regex
    # syntax inside an `in` check never matches. Replaced with real phrases
    # plus a compiled regex.
    if any(
        phrase in ai_lower
        for phrase in ["5-digit zip", "five-digit zip", "new zip", "current zip", "updated zip"]
    ) or _ZIP_UPDATE_RE.search(ai_lower):
        return "zip_update"

    # Bare zip mention → confirmation of the on-file ZIP, but never for
    # dispatch confirmations that merely name the ZIP used for the search.
    if "zip code" in ai_lower and "within 30 minutes" not in ai_lower:
        return "zip_confirm"

    if "via fax or email" in ai_lower or "fax or email" in ai_lower:
        return "fax_or_email"

    # Fax re-ask must precede fax_confirm
    if any(phrase in ai_lower for phrase in ["new fax number", "correct fax number"]):
        return "fax_update"

    if "fax number" in ai_lower and any(
        phrase in ai_lower
        for phrase in [
            "is that correct",
            "is this correct",
            "is that right",
            "is this right",
            "is that the right",
            "is this the right",
            "on file",
            "to confirm",
        ]
    ):
        return "fax_confirm"

    # --- Benefits / coaching ---
    if "individual deductible" in ai_lower or "family deductible" in ai_lower:
        return "benefits_explanation"

    # Care Coach OFFER requires an offer question. The Care-Coach DISPATCH
    # message ("We will send the Care Coach details... Is there anything
    # else?") must fall through to follow_up below.
    if ("care coach" in ai_lower or "wellness coach" in ai_lower) and any(
        phrase in ai_lower
        for phrase in ["would you like", "want me to send", "do you want", "want us to send"]
    ):
        return "coach_offer"

    # --- Closing ---
    if "thank you for calling" in ai_lower or "have a great day" in ai_lower:
        return "closing"

    if "is there anything else" in ai_lower or "anything else i can" in ai_lower:
        return "follow_up"

    return "other"


# ---------------------------------------------------------------------------
# Claim-adjustment flow dispatcher — called by classify_ai_slot(flow="claim")
# ---------------------------------------------------------------------------


def classify_ai_slot_claim(ai_message: str) -> str:  # noqa: C901
    """Return the slot label for claim-adjustment flow AI messages."""
    if not ai_message:
        return "other"

    ai_lower = ai_message.lower()

    # --- Re-ask / clarification (must precede slot checks) ---
    if any(
        phrase in ai_lower
        for phrase in [
            "i'm sorry, i didn't quite catch",
            "i didn't quite catch",
            "could you repeat",
            "one more time",
            "i wasn't able to catch",
            "didn't catch that",
            "could you say that again",
            "could you please repeat",
        ]
    ):
        return "clarification"

    # --- Correction acknowledgement (must precede slot checks) ---
    if any(
        phrase in ai_lower
        for phrase in [
            "got it, i've updated",
            "got it i've updated",
            "i've updated",
            "i've corrected",
            "i have updated",
            "i have corrected",
            "thank you for the correction",
        ]
    ):
        return "correction_ack"

    # --- Greeting / intent ---
    if (
        any(
            phrase in ai_lower
            for phrase in [
                "how can i assist",
                "how may i assist",
                "how can i help",
                "how i can help",
                "please tell me how i can",
            ]
        )
        and "is there anything else" not in ai_lower
    ):
        return "intent_selection"

    # --- Name readback confirmation (before first/last name checks) ---
    if _NAME_SPELLED_RE.search(ai_lower):
        return "name_confirmed"

    # --- Identity collection (shared with PCP flow) ---
    if "first name" in ai_lower:
        return "first_name"

    if "last name" in ai_lower:
        return "last_name"

    if "member id" in ai_lower and "date of birth" not in ai_lower:
        return "member_id"

    if "date of birth" in ai_lower:
        return "dob"

    if "phone number" in ai_lower and any(
        phrase in ai_lower for phrase in ["is your phone number", "is this correct", "on file"]
    ):
        return "phone_confirmed"

    # --- Claim-specific slots ---

    # reference_number must precede records_method (both can appear after auth)
    if any(phrase in ai_lower for phrase in ["reference number", "adjustment request"]):
        return "reference_number"

    # upload_link_sent: confirmation that the link was dispatched (contains delivery confirmation language)
    if any(
        phrase in ai_lower
        for phrase in [
            "expect to receive the link",
            "link has been sent",
            "upload link is on its way",
            "you should receive the upload link",
        ]
    ):
        return "upload_link_sent"

    # upload_consent: agent offering to send the upload link (precedes email_confirmed_upload)
    if any(
        phrase in ai_lower
        for phrase in ["send you a link", "upload the records directly", "link where you can upload"]
    ):
        return "upload_consent"

    # email_confirmed_upload: verifying on-file email before sending upload link.
    # Pool variants include "I have <addr> on file. Is that the correct email
    # to send the upload link to?" — the old check required the literal
    # "email address we have on file" and missed them (slot fell to "other").
    if "email" in ai_lower and (
        "correct or has it been changed" in ai_lower
        or "still the right address" in ai_lower
        or "correct email" in ai_lower
        or (
            "on file" in ai_lower
            and any(
                p in ai_lower
                for p in ["is that correct", "is this correct", "is that the right", "is this the right"]
            )
        )
    ):
        return "email_confirmed_upload"

    # --- Personal Guide scheduled / N1 notification bridge ----------------
    # MUST run BEFORE the N2 check: the combined guide-scheduled + bridge
    # message contains both a channel phrase and "updates"/"keep you posted",
    # which previously matched the N2 detector and returned the wrong slot.
    if any(
        phrase in ai_lower
        for phrase in [
            "personal guide will call",
            "personal guide will reach out",
            "next 24 hours",
            "team will contact the provider",
            "our team will contact",
            "status of the provider outreach",
        ]
    ):
        # The reference transcript delivers the guide-scheduled message and
        # the N1 channel ask as ONE combined turn ("...Our Personal Guide
        # will call the provider in the next 24 hours. ...How do you want to
        # be notified?"). When the bridge carries a channel question, the
        # caller is being asked to pick a channel — classify it as the N1
        # notification ask, not as the bare scheduled-acknowledgement.
        if any(p in ai_lower for p in _CHANNEL_PHRASES):
            return "notification_method"
        return "guide_scheduled"

    # personal_guide_consent: offering Personal Guide outreach
    if any(
        phrase in ai_lower
        for phrase in ["personal guide", "outreach to your doctor", "would you like us to proceed"]
    ):
        return "personal_guide_consent"

    # N2 preference confirmed — follow-up opening
    if any(
        phrase in ai_lower
        for phrase in [
            "i'll send progress updates",
            "i will send progress updates",
            "send progress updates",
            "sure, i will send sms",
            "sure, i will send email",
            "same email address on record",
            "perfect, i'll send progress",
            "got it, i'll send progress",
            "aside from this",
            "is there anything else i can help you with today",
            "is there anything else i can assist you with",
        ]
    ) and not any(p in ai_lower for p in _CHANNEL_PHRASES):
        return "follow_up"

    # n2_notification_method: second notification-channel ask (progress updates).
    # Requires a SPECIFIC progress-updates phrase AND a channel phrase, and
    # must NOT carry provider-outreach wording (that is the N1 bridge, handled
    # above — this exclusion is a second line of defence for rephrased
    # messages). The bare token "updates" is deliberately NOT in this list:
    # a fresh N1 ask ("How would you like to receive updates — phone or
    # email?") also contains it and must classify notification_method.
    if (
        any(p in ai_lower for p in _CHANNEL_PHRASES)
        and any(
            phrase in ai_lower
            for phrase in [
                "whenever there are updates",
                "updates in this request",
                "updates on this request",
                "progress updates",
                "keep you posted",
                "notifications",
            ]
        )
        and not any(p in ai_lower for p in _N1_OUTREACH_PHRASES)
    ):
        return "n2_notification_method"

    # notification_method: first notification-channel ask (incl. outreach bridge
    # variants that escaped the guide_scheduled check)
    if any(p in ai_lower for p in _CHANNEL_PHRASES):
        return "notification_method"

    # timeline_question: agent providing or offering timeline details
    if any(phrase in ai_lower for phrase in ["expected timeline", "how long will it take", "business days"]):
        return "timeline_question"

    # records_method: agent asking member how records will be delivered
    if any(
        phrase in ai_lower
        for phrase in [
            "send it over",
            "upload the records",
            "doctor to send",
            "complete copy of the medical records",
            "medical records for this request",
            "medical records for this adjustment",
            "copy of the medical records",
        ]
    ):
        return "records_method"

    # --- Closing ---
    if "thank you for calling" in ai_lower or "have a great day" in ai_lower:
        return "closing"

    if "is there anything else" in ai_lower or "anything else i can" in ai_lower:
        return "follow_up"

    return "other"
