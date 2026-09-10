# Live End-to-End Conversation Tests

This suite drives the **real** LangGraph application graph (`agent.app_graph.build_graph`
with a `MemorySaver` checkpointer) turn-by-turn with scripted user utterances and asserts
hard, deterministic outcomes on graph state.

**Nothing is mocked.** Every run makes:

- real **Azure OpenAI** calls (extraction, generation, routing LLMs), and
- real **Salesforce** calls (`lookup_member`, `find_adjustment`, dispatch tools,
  contact updates).

> **⚠ Cost & data warning** — A full run is ~34 conversations × 8–17 LLM-driven turns
> each (several hundred LLM calls plus dozens of Salesforce queries). It costs real
> money, takes 15–45 minutes, and **5 scenarios write to Salesforce** (they are
> reverted by teardown, but run against a sandbox org only). Never point this at
> production data.

## How to run

From the repo root:

```bash
# canonical entry point
python -m tests.live_e2e.run_live_tests

# one or a few scenarios
python -m tests.live_e2e.run_live_tests --only pcp_happy_path_fax,claim_happy_path

# skip the Salesforce-writing scenarios
python -m tests.live_e2e.run_live_tests --skip-mutating

# custom results directory
python -m tests.live_e2e.run_live_tests --results-dir /tmp/live_results

# list scenario names
python -m tests.live_e2e.run_live_tests --list
```

Pytest wrapper (the `live` marker is excluded from default `pytest` runs via
`addopts = "-m 'not live'"` in `pyproject.toml`):

```bash
pytest -m live tests/live_e2e/test_live.py
```

Scenarios run **sequentially** — they share Salesforce fixture data; do not
parallelize (no `pytest-xdist`). Each scenario gets a fresh `thread_id` (uuid4)
and a fresh `MemorySaver`. Exit code is non-zero on any failure; preflight
failure exits 2 without running anything.

Per-scenario artifacts (full transcript + final state snapshot + metadata events)
are written to `tests/live_e2e/results/<scenario>_<timestamp>.json`.

## Required environment

Preflight fails fast (listing exactly what is missing) unless these are set
(directly or via `.env` at the repo root — the same loader the app uses):

| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_API_KEY` | worker / extraction / routing LLMs |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `SF_CLIENT_ID` | Salesforce OAuth |
| `SF_CLIENT_SECRET` | Salesforce OAuth |
| `SF_REFRESH_TOKEN` | Salesforce OAuth |
| `SF_INSTANCE_URL` | Salesforce instance |

Optional but used if present: `OPENAI_API_VERSION`, `WORKER_DEPLOYMENT`,
`SF_TOKEN_URL`, Gemini settings (`GCP_SA_BASE64`, …) for the generation LLM
(the app silently falls back to the Azure extraction LLM without them).

## Required Salesforce fixtures

Preflight live-verifies these via `agent.storage.queries` and **aborts with
instructions** if any are missing (it never skips silently):

- **Emily Carter** — `M_Member__c`: `Member_ID__c=M907503`, DOB `1988-04-12`,
  with non-empty zip, fax, and email on file. Original `zip_code`, `fax`,
  `email`, `phone_number` are snapshotted before any scenario runs.
- **James Wilson** — `M_Member__c`: `Member_ID__c=M310188`, DOB `1977-07-30`,
  phone `512-555-6101`, email `james.wilson@gmail.com`.
- **Adjustment request** — `M_Adjustment_Request__c`: `Reference_Number__c=42695817`
  linked to `M310188`. (`records_required` is not an SF field; the claim agent
  defaults it to `True`, which the records scenarios rely on.)
- **Benefit plan** — `M_Benefit_Plan__c` row for `M907503`.

### Teardown

Scenarios that mutate SF contact fields (`pcp_zip_update`, `pcp_fax_update`,
`pcp_email_update`, `claim_email_change_on_upload`, `email_change_loop_in_notification`)
are wrapped in a `finally` block that restores the snapshotted values via
`update_member_contact` — even when the scenario fails mid-way.

## Scenario matrix

34 scenarios across 8 groups (see `scenarios.py` for the exact scripts):

- **A. PCP happy paths** — fax delivery, email delivery, benefits declined,
  zip update, fax update, email update (`pcp_email_update` — mutating; the agent
  reads email addresses back with `@` replaced by ` at ` as an Azure
  content-filter workaround, so no assertions depend on a literal `@` in AI lines).
- **B. Verification escalations** — restart-then-success, two-round failure,
  member-id exhaustion, DOB-without-year exhaustion.
- **C. Guard escalations** — transfer request, abuse, self-harm, repeated off-topic.
  These depend on LLM guard classification, so they carry `retries=1`: a failed
  attempt is rerun once and reported `PASS*` / `flaky=True` if the retry passes.
- **D. Intake routing** — unclear-intent exhaustion, out-of-scope billing
  (hard END, routed phone number, no transfer event), non-member caller.
- **E. Claim flow** — happy path, upload-only, guide-only, decline-everything,
  phone-not-confirmed hard END, ref-not-found retry + escalation, ref exhaustion,
  email change during upload.
- **F. Follow-up escalations** — update request, 3× cannot-answer.
- **G. Contact-change loop limits** — zip change loop, email change loop in
  notification setup.
- **H. Conversational & confusion-recovery** —
  - `pcp_happy_path_conversational`: same Emily flow as A but with
    `PCP_VERIFY_CONVERSATIONAL` (natural phrasing, spelling-out of last name);
    email delivery, accept benefits + Care Coach. `retries=1`.
  - `claim_happy_path_conversational`: same James flow as E but with
    `CLAIM_VERIFY_CONVERSATIONAL` (natural phrasing, hesitant card-lookup);
    doctor-direct → upload link → Personal Guide → SMS → email N2. `retries=1`.
  - `pcp_confused_member`: injects a ZIP read-back clarify turn, a hedged
    delivery-method answer, and a benign side-question mid-benefits-offer;
    asserts no escalation. `retries=1`, `timeout_s=360`.
  - `claim_confused_member`: injects a hesitant ref-number non-answer (retry
    without escalation), a confused records question (upload_method re-ask),
    and an ambiguous email-confirmation answer (gentle re-ask); asserts
    no escalation. `retries=1`, `timeout_s=360`.

## Assertion philosophy

LLM phrasing varies between runs, so assertions never compare exact sentences:

- **state keys** (`provider_list_sent`, `delivery_method`, `claim_flow_complete`, …),
- **escalation reasons** as substrings/regex, checked across every source the app
  uses (`state["escalation_reason"]`, `last_agent_signal.escalation_reason`, and
  the `AgentCallTransfer` event `detail`),
- **metadata events** — `AgentCallTransfer` presence + `transferInitiator`,
  accumulated across the whole run (the `metadata_events` state key has no
  reducer and is overwritten each node),
- **END / interrupt flags** — including hard-END paths (phone-not-confirmed,
  out-of-scope) where `is_interrupt=False` and `next_node=END` with **no**
  transfer event,
- tolerant case-insensitive regexes over the transcript; where wording comes from
  a static pool the pool constant is imported and matched via `pool_regex()`.

If a scenario's user script is exhausted before END, the scenario fails with the
full transcript so off-script agent behavior is debuggable. Every turn is logged
at INFO as `[scenario] AI: … / USER: …`. Each scenario is bounded by a wall-clock
timeout (default 240 s; long claim flows 360 s) and a `max_turns` guard.

## Known issues (found while building the suite — assertions were NOT weakened)

These were established by code reading; a live run will confirm them. Scenario
expectations follow the documented/spec behavior, so the affected scenarios are
expected to FAIL until the agent code is fixed:

1. **`claim_upload_only` cannot complete.** In
   `src/agent/agents/records_coordination/agent.py`, `personal_guide_consent == "no"`
   escalates unconditionally with `member_declined_personal_guide` — even when the
   upload link was already sent (`upload_link_sent=True`). The static transcript
   `claim_adjustment_upload_only.txt` expects the call to continue to notification
   setup after declining the Personal Guide. Related: `records_branch_taken` is
   never set to `"member_upload"` anywhere in the code, only to `"personal_guide"`.

2. **`claim_email_change_on_upload`: new email is never persisted to Salesforce.**
   The records-coordination email-change path keeps the new email in graph state
   and uses it for the upload link, but never calls `update_member_contact`
   (unlike the zip/fax update paths in provider search / delivery management).
   The SF post-check on this scenario documents the gap. Additionally
   `send_upload_link` / `trigger_personal_guide_outreach_for_claim` /
   `set_claim_*notification` are stubbed no-ops in this org
   (`M_Claim_Upload_Link__c` etc. do not exist), so "link sent" is state-only.

3. **`state["escalation_reason"]` is only set by the intake out-of-scope path**
   (`src/agent/agents/intake/handlers.py`). Agent escalations carry the reason in
   `last_agent_signal.escalation_reason` and in the `AgentCallTransfer` event
   `detail`; `escalation_agent` then overwrites `last_agent_signal` with a
   COMPLETE signal. The harness therefore collects reasons from all three
   sources at every graph pause.

4. **Self-harm message pool drift.** The second member of
   `MSG_SELF_HARM_ESCALATION` ("…better placed to help. Please hold — I'm
   transferring you now.") does not match the supportive-phrasing regex
   `(support|help right now|stay on the line)`. The scenario matches the spec
   regex OR any pool member, so it documents rather than masks the drift.

5. **`records_required` is not a real Salesforce field** on
   `M_Adjustment_Request__c` — `claim_adjustment_agent` defaults it to `True`.
   The preflight check for "adjustment 42695817 with records_required=True"
   verifies the record's existence; the flag itself is a code default.

Update this section with live findings after each full run.
