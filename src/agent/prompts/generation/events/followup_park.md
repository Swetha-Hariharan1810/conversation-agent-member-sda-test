## Event: FOLLOWUP_PARK

The value in "Extracted this turn" WAS captured successfully — "Collecting:"
shows "(nothing — …)" because no slot is being asked for this turn. The caller
also asked a side question ("Followup:") that WILL be handled later in this
same call — do not attempt to answer it now.

Acknowledge the captured value, then add one clause promising the question
will be covered later in this same call — for example "I'll get to that in
just a moment once we finish verifying you". Never attempt an answer, never
guess, and never say you cannot help — the question will be answered later.
Do NOT ask for any slot — the system appends the next question after your
sentence.

If "Extracted this turn" is absent, nothing was captured — "Collecting:"
names the real slot; promise the request will be handled later in this same
call AND re-ask that slot in the same sentence.

One spoken sentence. Thirty-five words maximum.

Your sentence must not end with a question mark unless these instructions
explicitly tell you to ask for a value.

Negative example — caller said "Emily Carter — and will I get text
updates about my claim?":
WRONG: "Thanks Emily Carter, I'll cover notifications in just a moment —
now could you give me your date of birth?" (asks for a slot — the system
appends the next question itself).
RIGHT: "Thanks Emily Carter — I'll get to text updates in just a moment
once we finish verifying you."
