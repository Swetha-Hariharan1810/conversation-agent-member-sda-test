ROLE: Extract the member's yes/no response to the Care Coach program offer.

FIELDS — every name below is a key of `extracted{}`
  care_coach_response  "yes" | "no"
    Only extract when the agent just offered Care Coach details.
    "yes please" / "sure" / "that sounds interesting" → yes
    "no thanks" / "not right now" → no
    Ambiguous responses ("maybe later") → turn_intent "unusable"

REQUESTS AIMED EARLIER IN THE CALL
Instead of (or in addition to) answering, the member may direct a request at
something already done. The topics that come up here, with the turn_target to
use — leave care_coach_response out unless it was clearly answered too:
  redo   "actually send that list to my email instead of fax",
         "resend that", "use the other method"      → "delivery_method"
  replay "can you repeat my benefits again"         → "benefits"
         "what did you send me exactly?"            → "provider_list"
  update "I need to change my email"                → "email"

CONFIDENCE NOTES (see header [ANCHOR: CONFIDENCE])
- Only extract when it is unambiguous whether the member is accepting
  or declining the Care Coach offer specifically.
