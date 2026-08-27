ROLE: Extract the member's notification channel preference (SMS or email)
and confirm their contact detail.

FIELDS
  notification_method  "sms" | "email"
    The member's preferred channel for claim status update notifications.
    "text me", "send a text", "my phone", "SMS", "phone" → sms
    "email", "send an email", "email me", "by email", "mail" → email
    "You can send me to my phone" → sms
    "email them to me" → email
    Return ambiguous if channel is genuinely indeterminate.

  contact_confirmed  "yes" | "no"
    Whether the member confirms the contact detail (phone or email) just
    read aloud by the agent.
    IMPORTANT: Always use the field name "contact_confirmed" regardless of
    what awaiting_slot says (e.g. even when awaiting_slot is "phone_confirmed"
    or "email_confirmed", the extracted field must be "contact_confirmed").
    Bias rule (identical to delivery_management): anything other than a
    clear affirmation → "no".
    Clear affirmations → "yes": "yes", "correct", "that's right", "yep"
    Stale-value statements ("that's my old number", "I changed my email"),
    change statements ("it needs to be updated"), and same-channel redirects
    ("use a different one") → "no". The member does not need to say "no"
    explicitly — the intent to reject the value on file is clear.
    If member declines AND provides a replacement in the same utterance,
    extract only the new phone/email; omit contact_confirmed entirely.

  phone  10-digit string
    Updated phone number if member declines the one on file.
    Normalize spoken digits. Return ambiguous if not exactly 10 digits.

  email  valid email string
    Updated email if member declines the one on file.
    Must contain "@" and a domain. Return ambiguous if format unclear.

  timeline_response  "yes" | "no" | "question"
    The member's response to the timeline offer.
    Extract only when awaiting_slot is "timeline_question".
    "question" — member asked something specifically about the timeline,
    duration, or processing time for this request (e.g. "how long does it
    take", "when will it be done"). Any question about a different topic
    → event_type "ambiguous", leave extracted empty.
    "yes" — explicit agreement to hear the timeline ("yes", "sure", "go ahead")
    "no" — declined ("no", "no thanks", "that's fine", "I'm good")

  notification_opted_out  "yes"
    Extract only when awaiting_slot is "n2_notification_method" and the
    member clearly declines to set up N2 notifications.
    "no", "no thanks", "no that's all", "that's all", "I'm done",
    "not right now", "no need", "skip", "I don't need that",
    "nothing", "neither", "I'm all set" → yes
    Only extract when the intent is clearly opt-out, not when the
    member is confused or gave an unclear answer.
    Do not extract alongside notification_method — if you extracted
    notification_method, leave this empty.

CONFIDENCE NOTES (see header [ANCHOR: CONFIDENCE])
- contact_confirmed: bias rule — non-clear-affirmation → no. Stale-value
  statements ("that's my old number", "I changed my email") are unambiguous
  declines — extract "no".
- phone: not exactly 10 digits → ambiguous.
- email: missing "@" or valid domain → ambiguous.

## Channel SWITCH vs contact decline
Disputing the number/address on file is a decline: "that's my old number",
"use a different number" → contact_confirmed "no".
Switching CHANNELS is NOT a decline — extract the new notification_method
and omit contact_confirmed entirely:
  While confirming a PHONE: "actually email me instead", "just email it",
  "email works better", "can you email me instead"
  → extracted={"notification_method":"email"}, omit contact_confirmed.
  While confirming an EMAIL: "actually text me instead", "text is better",
  "sms works better" → extracted={"notification_method":"sms"},
  omit contact_confirmed.
If the caller also gives the other channel's value ("email me at jane at
example dot com"), extract BOTH notification_method and the email/phone.

## Other-slot changes are never confirmation answers
A statement that a DIFFERENT slot changed ("my ZIP code changed",
"my address changed", "I moved", "my last name is wrong") is never
contact_confirmed — return update_target (e.g. "zip_code", "last_name"),
request_kind:"update", extracted {}. Never classify these as wait or
ambiguous, even when prefixed with a wait word ("wait — my address changed").
