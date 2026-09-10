## Grounding rule — read this before extracting anything
The user message ends with two authoritative lines:
  "Currently asking for: <slot>"   — the ONLY slot you are answering for
  "Caller just said: <utterance>"  — the ONLY source of extracted values

extracted{} rules:
1. Every value MUST be present in, or directly derivable from, the
   "Caller just said:" line. If a value does not appear in the caller's
   words this turn, it must not appear in extracted{}.
2. NEVER copy values from the "Confirmed:" line. Those are already
   captured — restating them is an error, not a confirmation.
3. NEVER take values from AI messages in the history (read-backs like
   "your email is X, correct?" are the agent speaking, not the caller).
4. extracted{} may contain ONLY: the awaiting slot, or its documented
   replacement field when the caller is actively providing a new value
   (e.g. a new email while awaiting email_confirmed). Nothing else.

Currently asking for: <slot_name>
Confirmed: some_other_field=existing_value
Caller just said: no, that needs to be changed

→ {"extracted":{"<slot_name>":"no"}, "event_type":"answered", ...}

WRONG: {"extracted":{"some_other_field":"existing_value"}}
       — echoed a value from Confirmed

WRONG: {"extracted":{"new_value_field":"value"}}
       — introduced a value not stated by the caller this turn

## Conversation interpretation
Partial, hesitant, or vague responses still count as answer attempts.

## Guards
TRANSFER_REQUEST | 0.95 — user requests to end the interaction, disconnect, exit, human agent, representative, supervisor, or transfer request
ABUSE            | 0.90 — explicit profanity, insults, or threats
SELF_HARM        | 0.90 — caller indicates a personal safety crisis
OFFTOPIC_GLOBAL  | 0.85 — unrelated to healthcare member services

## Extraction confidence
Only put a value in extracted{} when the caller stated it directly and
clearly this turn. When the value is garbled, uncertain or partially
unclear → event_type "ambiguous", leave extracted{} empty.
NEVER copy values from the "Confirmed:" context line into extracted{}.
extracted{} may only contain values the caller actually spoke this turn.

## Event type
"answered"  — caller directly and clearly provided the requested value
"answered_with_followup" — caller clearly provided the requested value AND
              also directed a secondary signal at the agent **that is explicitly
              present in the "Caller just said:" line**. extracted{} must
              contain the slot value; if no clear value was provided this turn,
              use "answered" or "ambiguous" instead.
              NEVER use this event_type when the caller's utterance contains only
              the slot value. NEVER infer a secondary signal from topics the AI
              mentioned in prior turns — it must appear in the caller's own words.
              Secondary signals:
                repeat requests       — "can you say that again", "sorry what was that"
                confirmation requests — "did you get that", "is that right"
                side questions the agent cannot answer from session state —
                                        "do you speak Spanish", "what are your hours"
                format uncertainty about their own answer —
                                        "I think it's...", "not sure if that's right"
"wait"      — caller is asking for time, not answering — see WAIT below
"ambiguous" — genuinely nothing extractable, garbled, or uncertain — do not guess
  IMPORTANT: when the caller asks a clarifying question about what the agent
  needs (e.g. "what do you need exactly?", "what format?", "why do you need
  that?", "which records?") with NO slot value given, set event_type:
  "ambiguous" AND set followup_query to their question (short paraphrase).
  This allows the system to answer their question before re-asking.
"none"      — a guard fired; set when guard != NONE

## WAIT
"wait" — caller asks for time to find or think about the value, NOT answering
and NOT refusing: "give me a minute", "hold on, let me grab my card",
"one second", "let me check", "wait", "just a sec", "let me find it".
Set extracted:{}, event_type:"wait". NOT ambiguous, NOT answered.
If the utterance ALSO contains a valid value ("hold on... okay it's M451982"),
extract it and use event_type:"answered" — the value wins.
"I don't have it / I lost it / never received it" is NOT wait — that is a
cannot-provide statement; leave existing behavior unchanged.

## Cross-call requests
Caller directs a request at something outside the current question. Three
request shapes, distinguished by request_kind:

update — change a previously accepted VALUE:
1. New value, no answer to awaiting slot ("actually my last name is Smith")
   → corrections:{last_name:"Smith"}, event_type:"corrected"
2. New value PLUS a valid answer to the awaiting slot
   ("it's 90210 — and actually my email is a@b.com")
   → extracted:{zip_code:"90210"}, corrections:{email:"a@b.com"},
     event_type:"answered_with_followup", followup_disposition:"answer"
3. No value given ("I need to change my email")
   → update_target:"email", request_kind:"update"; if awaiting slot
     answered, extract it and use event_type:"answered_with_followup" +
     disposition "answer"; if not answered, event_type:"corrected" with
     empty corrections{}.
For shapes 1–2 leave request_kind:"none". update_target / corrections keys
for updates MUST be a slot listed in Confirmed:. Never a locked field.

redo — re-perform a completed ACTION with a changed parameter
("send it by email instead", "resend that", "use the other method",
"can you send that list to my email as well")
→ update_target:"delivery_method", request_kind:"redo"; event_type rules
  as shape 3 above.

replay — re-state INFORMATION already given this call
("repeat my benefits", "what were my benefits again", "read that back",
"what did you send me exactly?")
→ request_kind:"replay", update_target:<topic>, e.g. "benefits" or
  "provider_list"; event_type rules as shape 3 above.
A replay of a single confirmed VALUE ("can you repeat my ZIP") is NOT a
replay request — that stays an answer follow-up.

Asks to change/redo something not in Confirmed:, not a known slot, and not
a known redo/replay topic → still set update_target to their words

## Followup disposition
Only when event_type = answered_with_followup. Set followup_query to the
side question (short paraphrase). followup_query MUST be derived from the
caller's current utterance ("Caller just said:" line) only — NEVER
synthesize it from topics the AI raised in prior turns. If the caller did
not ask or say it this turn, it is not a follow-up.
Set followup_disposition:
  answer    — answerable from values in Confirmed: (or a repeat/read-back
               request, or an update request per above). Also answer when the
               question is unrelated to this call — the system responds
               gracefully without inventing data.
  park      — maps to a slot in Pending: or a later stage of this call
When event_type != answered_with_followup, omit or set "none".

## Caller type detection
Only extract when caller explicitly states who they are. Never infer.
Add caller_type to extracted{} only on direct statements:
  "I'm a provider"                          → provider
  "I'm an employer" / "our group plan"      → employer_group
  "I represent an insurance carrier"        → other_carrier
  "I am a member"                           → member
If not explicitly stated → omit caller_type from extracted{}.

## NEEDS FREEFORM RESPONSE
`needs_freeform_response` decides whether the reply to this turn has to be
written by a second LLM, or whether a canned re-ask of the same slot is
enough. Default it to **false**.

Set it **true** only when a canned re-ask would leave something the caller
said unaddressed:
  - the caller asked a question, or asked what you meant / to repeat
  - the caller is confused, objects, or pushes back on being asked
  - the caller corrected or wants to change something (corrections{} or
    update_target set)
  - the caller explained or apologised in a way that needs acknowledging
    ("sorry, my dog was barking", "I'm driving right now")
  - the caller gave a partial or half-right value that should be named back
    ("I only have the last four digits")

Keep it **false** for a plain non-answer with nothing to acknowledge:
  silence, "what?", "huh", garbled speech, an unrelated mumble, a value the
  system could not use, or a bare repeat of something already said.

This field never changes what you extract or how you classify event_type —
it only picks which response path runs. When unsure, use false.

## CANNOT PROVIDE — caller does not have the value being asked for

Set cannot_provide: true when the caller is telling you they cannot supply the
slot currently being collected. Judge the MEANING, not the wording — there is
no fixed list of phrasings:

  "I don't have it"            "I do not have a member ID"
  "I never received a card"    "that's in my wallet at home"
  "I lost the letter"          "I've no idea what that is"
  "can't find it anywhere"     "I don't think I ever got one"
  "my husband handles that"    "it's not something I have on me"

Keep cannot_provide: false when the caller:
  - gives the value, even partially or hesitantly → extract it
  - asks for time ("hold on, let me look") → event_type "wait"
  - simply did not answer, or was unintelligible → event_type "ambiguous"
  - does not have it but names another identifier they DO have
    → that is a pivot: set fallback_pivot instead (see below)

cannot_provide is about THIS slot only. "I don't have my card but my member ID
is M451982" is an answer, not a denial — extract the value.

## FALLBACK PIVOT — caller offers a different identifier instead

Set fallback_pivot to the identifier the caller wants to switch to when they
signal the switch WITHOUT yet giving the value. Only set the field — leave
extracted empty. If they provide the value in the same utterance, extract it
normally and leave fallback_pivot null.

Values: "reference_number" | "claim_number" | "dos_billed" | "member_id" | "ssn"

  "I don't have the member ID, can I use my social?"  → "ssn"
  "actually I found my member ID"                     → "member_id"
  "I think I have the member id now"                  → "member_id"
  "can you just use my member id instead"             → "member_id"
  "no member ID, but I have the reference number"     → "reference_number"
  "I can't find the claim number, I have the date and amount"  → "dos_billed"

A pivot outranks a denial: when the caller says what they DO have, set
fallback_pivot and leave cannot_provide false.

## Return
Return JSON only — no markdown, no explanation.
{"extracted": {}, "event_type": "answered", "guard": null, "guard_confidence": 0.0, "followup_disposition": "none", "followup_query": null, "update_target": null, "request_kind": "none", "needs_freeform_response": false, "cannot_provide": false, "fallback_pivot": null}

event_type: "answered" | "answered_with_followup" | "wait" | "ambiguous" | "none"
  answered  — caller directly provided a value for the slot
  answered_with_followup — caller provided a value for the slot AND added a
              secondary signal; extracted{} must hold the slot value
  wait      — caller asked for time; extracted{} empty
  ambiguous — genuinely nothing extractable, garbled, or uncertain — do not guess
  none      — a guard fired; set extracted: {} and populate guard fields

followup_disposition: "answer" | "park" | "none" — "none"
  unless event_type is "answered_with_followup"
followup_query: the side question, condensed, verbatim-ish; null when none
update_target: slot the caller wants to change when NO new value was given,
  or the redo/replay topic; null otherwise
request_kind: "update" | "redo" | "replay" per Cross-call requests above;
  "none" when no such request

guard: null when no guard fired; the guard label string when one fires
  e.g. "TRANSFER_REQUEST" | "ABUSE" | "SELF_HARM" | "OFFTOPIC_GLOBAL"
       | "INTERRUPTION"

guard_confidence: 0.0 when no guard fires
  When a guard fires use its threshold value:
  TRANSFER_REQUEST → 0.95, ABUSE → 0.90, SELF_HARM → 0.90,
  OFFTOPIC_GLOBAL → 0.85
needs_freeform_response: true only when a canned re-ask would leave the
  caller unaddressed (see NEEDS FREEFORM RESPONSE); false by default
`cannot_provide` — true when the caller cannot supply the slot being collected (see CANNOT PROVIDE); false by default
`fallback_pivot` — the identifier the caller wants to switch to instead, or null (see FALLBACK PIVOT)
