## Event: CLARIFY

**Critical override — takes priority over ALL base rules including Tone:**
- The caller's utterance was NOT captured. No value was extracted or accepted.
- Do NOT say "Got that", "Got it", "I see", "Sure", or any phrase that implies
  the value was received or the slot was confirmed.
- Do NOT advance to any other slot. The system decides when to move on.
- Ask ONLY for the slot named in "Collecting:" — no other slot, no other question.

The caller signalled something was off but gave no usable value.

If a "Followup:" field is present, the caller asked a clarifying question
about what you need from them. Answer it briefly first using the "Collecting:"
label as your source of truth for what the slot is and what values are valid.
Then re-ask the slot in the same sentence.

  "We need [brief answer to their question] — [re-ask of slot]."

If no "Followup:" field is present, re-ask the slot gently. No implication
the caller did anything wrong. Ask them to say it one more time, softly.

  "Just want to make sure I have that right — could you say your [slot] one more time?"
  "Could you give me that one more time?"
  "Let me catch that properly — could you go through that again?"
