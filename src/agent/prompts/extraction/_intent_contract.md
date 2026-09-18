<!--
_intent_contract.md — what the caller's turn DID, and the return shape, in ONE
place.

This is a partial, not an agent prompt. Every extraction prompt gets it, and
gets the same copy: build_extraction_prompt, build_extraction_prompt_core and
build_extraction_prompt_extraction all compose it after their header and the
follow-up contract. The leading underscore marks it as a shared fragment.

It replaces around two hundred lines that used to sit inside the three headers:
EVENT_TYPE, WAIT, CROSS-CALL REQUESTS, LOCKED FIELDS, NEEDS FREEFORM RESPONSE,
CANNOT PROVIDE, FALLBACK PIVOT and three separate RETURN blocks. Eleven fields
came out of them, and six were the same fact written twice:

  - corrections{} asked the model to say which of the values it had just heard
    replaced one already on file. That is a fact about the pipeline's state,
    which the pipeline holds and the model was being handed on a context line
    so it could hand it back. It reads that line correctly most turns. The
    turns it does not are an accepted value silently overwritten.
  - cannot_provide, fallback_pivot and event_type "ambiguous" were three ways
    of saying "no usable value this turn", so the headers carried precedence
    prose ("a pivot outranks a denial", "never set both", "a value given in the
    same utterance always wins") and the pipelines carried code to enforce it.
  - event_type, request_kind and update_target were one intent and one target
    split across three fields that could contradict each other, and did.
  - followup_disposition was instructed to be "none" on every single turn.
  - needs_freeform_response asked a perception model which of two response
    paths was cheaper — something it cannot see, and which Python decided for
    itself on every turn that carried any content at all.

What is left is one classification and one target. Do not restate any of it in
a header or an agent file — a second copy is how the last drift started.
-->

## THE TURN — one classification, one target

`turn_intent` says what the caller's utterance DID this turn. Exactly one
value, always.

| `turn_intent` | The caller… |
|---|---|
| `answered` | responded to the question they were asked |
| `wait` | asked for time to find or think about it |
| `unusable` | said nothing usable — garbled, uncertain, or asked instead of answering |
| `cannot_provide` | cannot supply what was asked for, and named no alternative |
| `pivot` | wants to identify themselves a different way, and has not said the value yet |
| `update` | wants a stored value changed, and did not say the new one |
| `redo` | wants a completed action re-performed differently |
| `replay` | wants information already given re-stated |

`turn_target` names what the last four point at, and is null for the rest:

  update → the slot to change      ("email", "zip_code", "last_name")
  redo   → what changes            ("delivery_method")
  replay → the topic               ("benefits", "provider_list")
  pivot  → the identifier offered  ("ssn", "member_id", "reference_number",
                                    "claim_number", "dos_billed")

### A value spoken beats every other reading

If the caller said a usable value, put it in `extracted{}` and use `answered`
— whatever else the utterance contains.

  "hold on… okay it's M451982"                → answered, not wait
  "I don't have my card but my ID is M451982" → answered, not cannot_provide
  "I lost the claim number — oh wait, 882301" → answered, not pivot

### `extracted{}` — every value, no bookkeeping

`extracted{}` is the ONLY place a slot value goes. The agent section at the end
of this prompt names the slots it collects and their allowed values — under a
FIELDS heading, or inline. **Every name it lists is a key inside `extracted{}`,
never a field of its own.** A classification the agent asked you to make
(`intent`, `same_member`, `name_confirmed`, `care_coach_response`) is a value
like any other and goes in the same place:

  FIELDS
    intent: provider_services | claim_services | ...

  → {"extracted": {"intent": "claim_services"}, "turn_intent": "answered", ...}

Fill the key the `Currently asking for:` line points at whenever the caller's
words settle it — including when the answer is the "nothing specific yet" tag
the agent section lists. An empty `extracted{}` says they settled nothing, and
the agent believes you.

Put every value the caller spoke this turn in `extracted{}`, keyed by slot
name. Do not decide whether a value is new or replaces one already on file, and
do not sort it anywhere else: the system knows what it has already confirmed
and works that out itself.

  "actually my last name is Smith"          → extracted {last_name: "Smith"}
  "it's 90210 — and my email is a@b.com"    → extracted {zip_code: "90210",
                                                         email: "a@b.com"}

Both of those are `answered`. `update` is only for a change with NO new value
in it ("I need to change my email") — there is nothing to extract, so the
target carries the request instead.

### An answer and a request in the same turn

Extract the answer AND set the intent and target for the request. The system
handles both.

  "it's 90210, oh and I need to change my email"
    → extracted {zip_code: "90210"}, turn_intent "update", turn_target "email"
  "Fax is fine — actually, send it by email instead"
    → turn_intent "redo", turn_target "delivery_method"

A request aimed at something that is not a known slot or topic still gets
`turn_target` set, in the caller's own words — the system carries what it does
not recognise to a representative.

### `wait`

Asking for time, not answering and not refusing: "give me a minute", "hold on,
let me grab my card", "one second", "let me check", "just a sec". Leave
`extracted{}` empty. Not `unusable`.

A wait word in front of a change is not a wait — "wait, actually my ZIP
changed" is an update.

### `cannot_provide` vs `pivot`

Both mean there is no value this turn. The difference is whether the caller
named something else they DO have. Judge the meaning, not the wording — there
is no fixed list of phrasings.

  "I don't have it" / "I never got a card" / "that's in my wallet at home"
  "I lost the letter" / "my husband handles that" / "no idea what that is"
    → cannot_provide

  "I don't have the member ID, can I use my social?"        → pivot, "ssn"
  "actually I found my member ID"                           → pivot, "member_id"
  "no member ID, but I have the reference number"           → pivot, "reference_number"
  "can't find the claim number, I have the date and amount" → pivot, "dos_billed"

A caller who simply did not answer, or was unintelligible, is `unusable` —
not a denial.

## RETURN

Return JSON only — no markdown, no explanation.

{"extracted": {}, "turn_intent": "answered", "turn_target": null, "guard": null, "guard_confidence": 0.0, "followup_query": null}

`extracted` — slot name → what the caller's words settle for that slot; {} when none.
  Every field the agent section names lives in here. See `extracted{}` above.
`turn_intent` — "answered" | "wait" | "unusable" | "cannot_provide" | "pivot" | "update" | "redo" | "replay"; default "answered"
`turn_target` — what update / redo / replay / pivot points at; null otherwise
`guard` — the guard label when one fires, null when none
`guard_confidence` — the fired guard's threshold value; 0.0 when none fires
`followup_query` — the caller's side question in their own words, condensed; null otherwise. See THE FOLLOW-UP CONTRACT.

Nothing else. These six are the whole contract — the agent section adds keys
to `extracted`, never fields alongside it.
