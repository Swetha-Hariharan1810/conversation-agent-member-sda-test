<!--
DEPRECATED (Phase 5) — no code loads this file anymore.

The recovery prompt was split into flag-conditional sections assembled by
build_generation_prompt(guard) in src/agent/utils.py:

  recovery_base.md          — identity, tone, variation rules, reading the
                              inputs, slot discipline, hard rules
  events/retry.md           — RETRY (wrong format / gibberish re-ask)
  events/clarify.md         — CLARIFY (gentle re-ask, no attempt cost)
  events/correction.md      — CORRECTION (+ invalid-corrected-value flavor)
  events/offtopic_agent.md  — OFFTOPIC_AGENT redirect
  events/interruption.md    — INTERRUPTION steer-back
  events/followup_answer.md — FOLLOWUP_ANSWER (answer from Confirmed values)
  events/followup_park.md   — FOLLOWUP_PARK (promise to answer later)
  events/followup_decline.md— FOLLOWUP_DECLINE (warm decline)

Edit those files instead. This placeholder is kept for one release so stale
references fail loudly here rather than silently loading old rules; delete it
in the next release.
-->
