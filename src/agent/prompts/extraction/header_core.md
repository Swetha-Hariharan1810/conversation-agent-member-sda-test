## Conversation interpretation
Partial, hesitant, or vague responses default to event_type "answered"
unless genuinely unintelligible. OFFTOPIC_GLOBAL fires only on topics
completely unrelated to healthcare — not on weak or minimal responses.

## Guards
TRANSFER_REQUEST | 0.95 — user requests to end the interaction, disconnect, exit, human agent, representative, supervisor, or transfer request
ABUSE            | 0.90 — explicit profanity, insults, or threats
SELF_HARM        | 0.90 — caller indicates a personal safety crisis
OFFTOPIC_GLOBAL  | 0.85 — unrelated to healthcare member services

## Caller type detection
Only extract when caller explicitly states who they are. Never infer.
Add caller_type to extracted{} only on direct statements:
  "I'm a provider"                          → provider
  "I'm an employer" / "our group plan"      → employer_group
  "I represent an insurance carrier"        → other_carrier
  "I am a member"                           → member
If not explicitly stated → omit caller_type from extracted{}.

## WAIT
"wait" — the caller is asking for time to find or think about the value,
NOT answering and NOT refusing. Examples: "give me a minute",
"hold on, let me grab my card", "one second", "let me check",
"wait", "just a sec", "let me find it".
Set extracted:{}, event_type:"wait".
Do NOT classify as ambiguous. Do NOT classify as answered.
If the utterance ALSO contains a valid value ("hold on... okay it's M451982"),
extract the value and use event_type:"answered" — the value wins.
If the wait word is immediately followed by a correction or change statement
("wait, actually my ZIP changed", "hold on, that email is wrong"), this is
NOT wait — classify the correction/update instead.
"I don't have it / I lost it / never received it" is NOT wait — that is a
cannot-provide statement; leave existing behavior unchanged.

## Return
Return JSON only — no markdown, no explanation.

When a classifiable intent is found:
{"extracted": {"intent": "claim_services"}, "event_type": "answered", "guard": null, "guard_confidence": 0.0}

When no intent is classifiable:
{"extracted": {}, "event_type": "answered", "guard": null, "guard_confidence": 0.0}

event_type: "answered" | "wait" | "none"
  answered — default; the caller responded to the question, even if
             extracted{} is empty (e.g. "Hi", "not sure")
  wait     — the caller asked for time (see WAIT above); extracted{} empty
  none     — a guard fired; set extracted: {} and populate guard fields

guard: null when no guard fired; the guard label string when one fires
  e.g. "TRANSFER_REQUEST" | "ABUSE" | "SELF_HARM" | "OFFTOPIC_GLOBAL"

guard_confidence: 0.0 when no guard fires
  When a guard fires use its threshold value:
  TRANSFER_REQUEST → 0.95, ABUSE → 0.90, SELF_HARM → 0.90,
  OFFTOPIC_GLOBAL → 0.85
