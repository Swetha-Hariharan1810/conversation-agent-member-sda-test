# Conversation Agent — Project Walkthrough, Extraction & Generation Contracts

**Repository:** `conversation-agent-member-sda-test`
**Scope of this document:** what the project is, in plain terms; the exact input/output
shapes of the two LLM tiers (extraction and generation); and the precise rules that
decide when the generation model is called and what happens to what it produces.

This is a *descriptive* document — it records how the code behaves today. Every claim
below is anchored to a file and, where useful, a line number.

---

## Table of contents

1. [The project in simple terms](#1-the-project-in-simple-terms)
2. [The moving parts](#2-the-moving-parts)
3. [The life of one conversational turn](#3-the-life-of-one-conversational-turn)
4. [Extraction (LLM 1) — input and output structure](#4-extraction-llm-1--input-and-output-structure)
5. [Generation (LLM 2) — input and output structure](#5-generation-llm-2--input-and-output-structure)
6. [When generation gets triggered, and how it is handled](#6-when-generation-gets-triggered-and-how-it-is-handled)
7. [The third path: the follow-up agent's single call](#7-the-third-path-the-follow-up-agents-single-call)
8. [Limits, guards and configuration knobs](#8-limits-guards-and-configuration-knobs)
9. [Why the code is shaped this way — the failure it was built around](#9-why-the-code-is-shaped-this-way--the-failure-it-was-built-around)

---

## 1. The project in simple terms

This is an **automated member-services representative** for a health insurer. A member
calls (or chats), and the system does what a human rep would do:

1. Greets them and works out what they want (a provider list? a claim adjustment?).
2. Verifies who they are — first name, last name, member ID, date of birth — against
   Salesforce.
3. Runs the actual errand: finds providers and faxes/emails the list, explains benefits,
   collects medical-records preferences for a claim, sets up notifications.
4. Answers any last questions and says goodbye — or hands the call to a human when it
   cannot help.

### The one idea that explains the whole codebase

A phone call is a sequence of **turns**. On each turn the system has exactly one job:
*get one piece of information (a "slot") from the caller.* Everything else — the graph,
the agents, the prompts, the guards — exists to make that one job survive contact with
real people, who interrupt, correct themselves, mumble, ask side questions, and say
"hold on, let me find my card".

So each turn splits cleanly into two questions:

| Question | Who answers it | Tier |
|---|---|---|
| **"What did the caller just say?"** | An LLM that returns structured JSON, never prose | **Extraction — LLM 1** |
| **"What do we say back?"** | Either a canned template (free, instant, cannot go wrong) or an LLM that writes one sentence | **Generation — LLM 2** |

That split is the spine of the project. Section 6 is entirely about the boundary
between those two answers to the second question.

### Technology

- **LangGraph** — a state machine where each node is an agent. Between every agent turn
  a `human_node` *interrupts* the graph and waits for the caller's next utterance
  (`src/agent/app_graph.py:112`).
- **Azure OpenAI** — extraction, follow-up, routing.
- **Google Gemini** — generation (recovery sentences).
- **Salesforce** — the system of record for members, benefits, claims, notifications.
- **Prompts are Markdown files**, not Python strings (`src/agent/prompts/`), assembled at
  runtime and cached.

---

## 2. The moving parts

```
src/agent/
├── app_graph.py            LangGraph wiring: nodes, edges, human_node, warm-up
├── state.py                The State TypedDict — every field the call carries
│
├── agents/<name>/          One package per agent (12 of them)
│   ├── agent.py            The agent class + its async node function
│   ├── llm.py              The ONE extraction call this agent makes
│   ├── handlers.py         Salesforce calls (pure async functions)
│   ├── pipelines.py        SlotPipeline configuration (which slots, in what order)
│   └── constants.py        Static message pools, labels, limits
│
├── core/
│   ├── agent.py            BaseAgent — execute() wraps every agent's run()
│   ├── slot_manager.py     ★ _collect_slot: the per-turn collector. 2,498 lines.
│   ├── guards.py           Safety/routing guards, run before slot logic
│   ├── signals.py          ask_member / signal_complete / signal_escalate
│   ├── constants.py        MAX_SLOT_ATTEMPTS=3, MAX_DEFLECTED_TURNS=4, …
│   └── followup_grounding.py  Recovers/vetoes side questions the extractor got wrong
│
├── llm/
│   ├── config.py           Model tiers, lru_cache'd singletons, feature flags
│   ├── schema.py           ★ WorkerResult / FollowUpResult / SsnFallbackResult
│   ├── extractor.py        ★ build_worker_input — the LLM-1 input contract
│   ├── response_generator.py ★ LLM-2 payload, trigger test, output sanitizer
│   └── redaction.py        mask_confirmed — member_id/dob never reach LLM 2 raw
│
├── prompts/
│   ├── system/             global_extraction.md, global_generation.md
│   ├── extraction/         header*.md + _followup_contract.md + per-agent files
│   └── generation/         recovery_base.md + events/<guard>.md
│
├── slots/                  Normalizers, validators, SlotType enum, SlotPipeline
├── responses/              Static message pools + build_initial/transition/retry_prompt
├── orchestration/          Orchestrator, deterministic fast-path, safeguards
└── storage/                Salesforce async client + query modules
```

**The twelve agents:** `intake`, `verification`, `provider_search`,
`delivery_management`, `benefits`, `care_wellness`, `follow_up`, `closure`,
`escalation`, `claim_adjustment`, `records_coordination`, `notification_setup`.
Plus the `orchestrator`, which is a router, not an agent.

---

## 3. The life of one conversational turn

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ human_node  — graph is paused, waiting for the caller                │
 │   interrupt() → caller's text → clean_asr_input() strips "um", "uh"  │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Agent node  (e.g. verification_agent)                                │
 │                                                                      │
 │  BaseAgent.execute()                                                 │
 │    └─ run()                        ← the agent's own logic           │
 │        1. Deterministic pre-checks  (SSN fallback stage, bridges…)   │
 │        2. Build system prompt       build_extraction_prompt(file)    │
 │        3. ══ EXTRACTION CALL (LLM 1) ══  → WorkerResult              │
 │        4. reconcile_worker_result() ← regex fallback + veto layer     │
 │        5. run_conversation_guards() ← may take the whole turn        │
 │        6. apply_corrections()                                        │
 │        7. SlotPipeline.collect() → _collect_slot() per slot          │
 │             ├─ value good      → confirm, move to next slot          │
 │             └─ anything else   → decide the reply:                   │
 │                   ┌ static template  (no LLM)                        │
 │                   └ ══ GENERATION CALL (LLM 2) ══ → one sentence     │
 │                          → sanitize_generated()                      │
 │    └─ _answer_unanswered_side_question()  ← safety net               │
 │    └─ stamp_metadata_events()             ← CallAgentField events    │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
            ask_member() → is_interrupt=True, next_node=<this agent>
                                ▼
                          back to human_node
```

When an agent finishes its whole job it calls `signal_complete()` instead, which routes
to the `orchestrator` — which first tries a **deterministic fast path**
(`orchestration/fast_path.py`) and only calls the routing LLM when the fast path has no
opinion.

**Key invariant:** exactly one thing is spoken per turn, and exactly one question is
asked in it. Much of `response_generator.py` exists to enforce the second half.

---

## 4. Extraction (LLM 1) — input and output structure

### 4.1 The model

`src/agent/llm/config.py:118` — `get_extraction_llm()`

| Setting | Value |
|---|---|
| Provider | Azure OpenAI (`AzureChatOpenAI`) |
| Deployment | `WORKER_DEPLOYMENT` env var (config default `gpt-5.4-nano`; README documents GPT-4.1-mini) |
| Temperature | `WORKER_TEMPERATURE`, default `0` |
| `max_tokens` | 300 |
| `timeout` | 30.0 s hard ceiling |
| `max_retries` | 1 — "don't let the SDK silently backoff to 7 s" |
| `streaming` | False |
| Caching | `@lru_cache(maxsize=1)` — one instance per process |

The call is always made through LangChain structured output:

```python
result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
```

So the model is *schema-constrained*: it cannot return prose here.

### 4.2 INPUT — the message list

Built by `build_worker_input()` in `src/agent/llm/extractor.py:45`. It returns exactly
two messages.

#### Message 1 — `system`

Assembled and `lru_cache`d by one of three builders in `src/agent/utils.py`, picked by
how much machinery the agent needs:

| Builder | Header used | Used by |
|---|---|---|
| `build_extraction_prompt` | `extraction/header.md` (full: spelling, corrections, LOCKED fields) | verification, claim_adjustment, records_coordination, notification_setup |
| `build_extraction_prompt_extraction` | `extraction/header_extraction.md` (mid-tier) | provider_search, delivery_management, follow_up |
| `build_extraction_prompt_core` | `extraction/header_core.md` (minimal) | intake, benefits, care_wellness |

Each concatenates, joined by `\n\n---\n\n`:

```
system/global_extraction.md          ← global behavioural rules
extraction/header*.md                ← the tier header (guards, event types, RETURN shape)
extraction/_followup_contract.md     ← THE shared side-question rules (one copy, all agents)
extraction/_confirmation_contract.md ← THE shared read-back/confirmation rules
extraction/<agent>.md                ← agent-specific slot rules
```

The two `_`-prefixed fragments exist because those rules were previously written three
times in three headers and drifted — the same caller utterance was classified two
different ways depending on which slot they were on. See the comment block at the top of
`_followup_contract.md`.

#### Message 2 — `user`

A plain-text block, built in this order:

```
<history block>                        ← "AI: …" / "Caller: …" lines
                                          last 4 turns, or last 2 when attempt >= 2
Currently asking for: <awaiting_slot>
Confirmed: first_name=Monique, last_name=Customer, member_id=M907503
Pending: dob, relationship
Caller just said: <the caller's latest utterance, verbatim>
```

- `Confirmed:` — only non-empty string values; omitted entirely when there are none.
- `Pending:` — slots still to come later in the call. This is what lets the model
  classify a side question about a future step as *parkable*.
- `Caller just said:` — deliberately duplicated out of the history block so the model
  does not have to re-parse it. Without it, elevated attempt counts made the model
  over-conservative on spoken-digit values.

**Prompt-injection sanitising** runs first (`extractor.py:25`): phrases like
"ignore previous instructions", "you are now", "system prompt", `[INST]` are replaced
with `[*]` in *caller* turns only, and a warning is logged. These patterns never appear
in legitimate caller speech and they trip Azure's jailbreak content filter.

#### A concrete example

```
AI: Monique Customer — and your Member ID?
Caller: My member ID is m nine zero seven five zero three.
AI: Got that — and your date of birth?
Caller: April twelfth nineteen eighty eight.

Currently asking for: dob
Confirmed: first_name=Monique, last_name=Customer, member_id=M907503
Pending: dob, relationship
Caller just said: April twelfth nineteen eighty eight.
```

### 4.3 OUTPUT — `WorkerResult`

`src/agent/llm/schema.py:47`. Pydantic, `extra="forbid"`.

| Field | Type | Meaning |
|---|---|---|
| `extracted` | `dict[str,str] \| None` | New slot values found this turn. **All** identity fields mentioned, not just the awaited one. |
| `corrections` | `dict[str,str] \| None` | Values the caller is *changing* ("actually my name is James"). |
| `event_type` | enum | What the utterance *did*: `answered`, `answered_with_followup`, `corrected`, `ambiguous`, `wait`, `none`. |
| `guard` | enum | `TRANSFER_REQUEST`, `ABUSE`, `SELF_HARM`, `INTERRUPTION`, `OFFTOPIC_GLOBAL`, `OFFTOPIC_AGENT`, `NONE`. |
| `guard_confidence` | float 0–1 | Guards only act at **≥ 0.7**; below that, regex fallbacks take over. |
| `followup_disposition` | enum | `answer` / `park` / `none` (+ legacy `answer_now`, `decline`). The prompt tells the model to always emit `none` — Python decides. |
| `followup_query` | `str \| None` | The caller's side question, condensed, quotable from "Caller just said:". |
| `update_target` | `str \| None` | Slot the caller wants changed when they gave **no** new value; or the redo/replay topic. |
| `request_kind` | enum | `update` (change a stored value) / `redo` (re-do an action with a new parameter) / `replay` (re-state info already given) / `none`. |
| `cannot_provide` | bool | Caller cannot supply this slot at all ("I never received a card"). Semantic, not a phrase list. |
| `fallback_pivot` | `str \| None` | Caller offers a different identifier instead: `member_id`, `ssn`, `reference_number`, `claim_number`, `dos_billed`. |
| **`needs_freeform_response`** | bool | **The generation trigger flag — see §6.** Default `false`. |

Defaults on every field mean a failed call degrades to a well-formed empty result rather
than an exception at the call site.

The prompt's own `RETURN` contract (`extraction/header.md`, end of file):

```json
{ "extracted": {}, "corrections": {}, "event_type": "answered", "guard": null,
  "guard_confidence": 0.0, "followup_disposition": "none", "followup_query": null,
  "update_target": null, "request_kind": "none", "needs_freeform_response": false,
  "cannot_provide": false, "fallback_pivot": null }
```

#### Worked outputs

| Caller said | Output |
|---|---|
| "April twelfth nineteen eighty eight." | `extracted={"dob":"04/12/1988"}, event_type=answered, needs_freeform_response=false` |
| "90210, and when will I get the list?" | `extracted={"zip_code":"90210"}, event_type=answered_with_followup, followup_query="when will the list arrive", needs_freeform_response=true` |
| "Actually my last name is Smith." | `corrections={"last_name":"Smith"}, event_type=corrected, needs_freeform_response=true` |
| "Hold on, let me grab my card." | `extracted={}, event_type=wait` |
| "I never received a card." | `cannot_provide=true` |
| "I don't have the member ID, can I use my social?" | `fallback_pivot="ssn"` |
| "what?" | `extracted={}, event_type=ambiguous, needs_freeform_response=false` |
| *(garbled)* | `event_type=ambiguous` |

### 4.4 Two sibling schemas

- **`FollowUpResult`** (`schema.py:97`) — used only by `follow_up_agent`. Adds
  `follow_up_intent` (`done`/`question`/`unsure`/`update_request`/`new_intent`/`wait`)
  and, crucially, **`answer: str | None`** — a generated sentence returned by the *same*
  call. See §7.
- **`SsnFallbackResult`** (`schema.py:131`) — `ssn_intent` +`ssn`, for the
  member-ID-unavailable branch. Has a deterministic keyword fallback
  (`agents/verification/llm.py`) if the call throws.

### 4.5 After the call — reconcile, then guards

`reconcile_worker_result(result, last_user_message)`
(`core/request_detection.py`) is a regex fallback **and** veto layer applied to every
extraction result:

- Fills an `update_target` / `request_kind` the model dropped ("I need to change my
  email" plainly says so).
- Clears a `wait` label on a turn that is actually a correction.
- Recovers a `followup_query` the extractor missed, and vetoes one it invented
  (`core/followup_grounding.py`).

It runs **before** the guards, because `note_side_question()` inside the guard layer is
what records the turn's side question for the safety net in `BaseAgent.execute`.

### 4.6 Failure handling

| Failure | Behaviour |
|---|---|
| Any exception | `logger.exception`, return empty `WorkerResult()` — the slot loop treats it as "nothing extracted" and re-asks |
| Azure `content_filter` code | Logged as a warning (jailbreak pattern), same empty result |
| SSN extractor exception | Falls back to `_ssn_keyword_fallback()` — spoken-digit parsing + yes/no keywords |

---

## 5. Generation (LLM 2) — input and output structure

### 5.1 The model

`src/agent/llm/config.py:160` — `get_generation_llm()` → `get_gemini_llm()`.

| Setting | Value |
|---|---|
| Provider | Google Gemini via `langchain_google_genai` |
| Model | `LLM_MODEL`, default `gemini-2.5-flash-lite` |
| Auth | GCP service-account JSON, base64 in `GCP_SA_BASE64` |
| Temperature | 0.3 |
| Thinking | `thinking_budget` (gemini-2.x) or `thinking_level` (gemini-3.x) |
| Structured output | **None** — returns free text |
| Fallback | If Gemini is unavailable for any reason → `get_extraction_llm()` |
| Caching | `@lru_cache(maxsize=1)` |

### 5.2 INPUT — two messages

`generate_recovery_message()` in `src/agent/llm/response_generator.py:641`.

#### Message 1 — `system`, assembled **per guard**

`build_generation_prompt(guard)` (`utils.py:266`):

```
system/global_generation.md
---
generation/recovery_base.md           ← identity, tone, variation, slot discipline, hard rules
---
generation/events/<guard>.md          ← what happened this turn; falls back to retry.md
```

Available event files: `retry`, `clarify`, `correction`, `correction_ack`,
`interruption`, `offtopic_agent`, `followup_respond`, `followup_answer`,
`followup_park`, `followup_decline`.

The base prompt's hard rules are worth knowing, because they explain the sanitizer:

- **One spoken sentence**, 25–35 words.
- **You are collecting exactly one slot** — the one named in `Collecting:`. Never name
  or imply another.
- **Never re-ask anything listed in `Confirmed:`.**
- When `Collecting:` reads `(nothing — …)`, ask for nothing at all; Python appends the
  next question.
- Never open two consecutive turns with the same word.
- Never speak from the caller's perspective, never complete their sentence.
- Delivery-channel discipline: on a fax slot never mention email, and vice versa.

#### Message 2 — `user`, rendered by `_render_payload()` (`response_generator.py:558`)

A labelled block. Lines appear only when they have content:

```
Conversation:
AI: Monique Customer — and your Member ID?
Caller: I'm sorry, could you say that again?

Caller just said: I'm sorry, could you say that again?
Collecting: Member ID — Must begin with m followed by 6 digit (…)
Tone:       gentle retry
Confirmed:  first_name=Monique, last_name=Customer, member_id=on file
Extracted this turn: 90210
Followup:   when will the list arrive
Coming up:  choosing fax or email, benefits summary
Ask for new value: yes
Event:      RETRY
```

| Line | Source | Notes |
|---|---|---|
| `Conversation:` | last **4** messages | `build_history()` |
| `Caller just said:` | latest user message | |
| `Collecting:` | `_SLOT_LABELS[slot]`, or a runtime override, or a `(nothing — …)` sentinel | The label is written as *the question being asked*, not the field name — see below |
| `Tone:` | `_tone_hint(attempt)` → `first ask` / `gentle retry` / `patient retry` | **The raw attempt count never reaches the LLM** |
| `Confirmed:` | `mask_confirmed()` | `member_id` and `dob` render as `on file`. Masking is centralised so no call site can leak |
| `Extracted this turn:` | the captured value | Only rendered when non-empty — an empty line read as "nothing captured" and caused re-asks |
| `Followup:` | the caller's side question | Its presence is what makes this turn an answering turn |
| `Coming up:` | remaining call stages, in words | Lets the model answer "when do I choose fax?" instead of deferring |
| `Ask for new value: yes` | update detours | The sentence must end by asking for the new value |
| `Event:` | the guard label | Only added for `CORRECTION`, `CORRECTION_ACK`, `CLARIFY`, `OFFTOPIC_AGENT`, `FOLLOWUP_*` |

**On `_SLOT_LABELS`** (`response_generator.py:20`): these are phrased as things a person
can be asked for. `benefits_response` is not `"benefits response"` but *"whether they
want to hear the benefits for office visits with the provider type they asked about —
yes or no"*. The code comment explains why: a model told to redirect to "benefits
response" reaches for the nearest askable thing it knows instead — which is how a
benefits turn started asking for the notification channel.

### 5.3 OUTPUT

**One spoken sentence of plain text.** No JSON, no schema, no labels.

It is then put through `sanitize_generated()` (`response_generator.py:421`), which is
where the real contract lives. See §6.4.

---

## 6. When generation gets triggered, and how it is handled

This is the heart of the design. The short version:

> **Generation is the exception, not the default.** A turn whose entire content is
> "ask the same question again" is answered from a template. A turn that carries
> something a template cannot say — a question, a correction, a partial value, an
> apology, a topic switch — goes to the LLM.

### 6.1 The three possible replies to a turn

| Reply source | Cost | Can it ask the wrong question? |
|---|---|---|
| **Static template** (`responses/builder.py`) — first asks, transitions, retries, wait acks, escalations, goodbyes | 0 ms, 0 tokens | No. Deterministic, per-slot pools. |
| **Generated sentence** (LLM 2) | ~1 network round trip | Yes — which is why the sanitizer exists. |
| **Generated answer inside the extraction call** (`follow_up_agent` only) | free (same call) | Bounded by the session snapshot. |

### 6.2 The trigger test

Every generated recovery sentence in the slot layer goes through
`_generate_slot_retry_response()` (`core/slot_manager.py:421`). The first thing it does
is ask whether it can avoid the LLM entirely:

```python
if (
    Config.STATIC_RETRY_FAST_PATH                     # feature flag, default ON
    and has_static_retry(slot_type, slot_name)        # a purpose-written template exists
    and not needs_freeform_response(...)              # nothing a template can't carry
):
    return build_retry_prompt(...)                    # ← no LLM call at all
```

with one override: **if the static line is identical to what was just said**, generate
anyway — "a caller who is already struggling hears a machine looping rather than a
person re-asking."

`needs_freeform_response()` (`response_generator.py:180`) returns **True** (→ generate)
if *any* of:

| # | Condition | Rationale |
|---|---|---|
| 1 | `guard` is not `RETRY` or `CLARIFY` | Every other guard has real content to convey |
| 2 | A `followup_query` is present | A template cannot answer a question |
| 3 | A value was extracted this turn | It must be named back |
| 4 | No extraction result was passed (`decision is None`) | Un-wired call sites keep the old always-generate behaviour |
| 5 | `decision.corrections` has any value | |
| 6 | `decision.update_target` is set | |
| 7 | `decision.followup_query` is set | |
| 8 | `carries_freeform_content(utterance)` | **Safety net**: the utterance contains a *request cue* and enough words for it to mean something |
| 9 | `decision.needs_freeform_response is None` | Missing flag ⇒ generate |
| 10 | Otherwise: **the model's own flag** | |

Conditions 5–8 are deliberately a net *under* the model's flag, not a replacement:
the flag is set by a model that can be wrong about its own output, and it was — on
*"Please check my claim status today"*, which got "Sorry, I didn't catch that" twice.

`carries_freeform_content` (`core/followup_grounding.py:153`) used to be a word count
(≥ 4 words ⇒ generate). That was the wrong proxy: on a voice call nearly every non-answer
clears four words, so the static path almost never ran. It now requires a *request cue*
plus a minimum length — so `"what?"` stays canned, where the canned line is both true and
exactly right.

`has_static_retry()` (`responses/builder.py:449`) keeps slots with no purpose-written
template on the LLM path rather than reading a field name at the caller — "upload
consent", "timeline question".

### 6.3 The guard catalogue — what each label means and who raises it

The `guard` argument to `generate_recovery_message()` is a **Python-internal routing
label**, distinct from `WorkerResult.guard`. Full table (the comment block above `_FALLBACKS`, `response_generator.py:120`):

| Guard | Raised by | Generation? | What the sentence must do |
|---|---|---|---|
| `RETRY` | `_collect_slot` — a genuine failed answer, attempt counted | **Conditional** (fast path eligible) | Re-ask the same slot, never another |
| `CLARIFY` | `_collect_slot` — first `AMBIGUOUS` turn, no attempt cost | **Conditional** (fast path eligible) | Re-ask gently; no implication of fault |
| `CORRECTION` | `_generate_correction_ack` | **Always** | Acknowledge the corrected value, re-ask the awaited slot |
| `CORRECTION_ACK` | `_handle_answered_followup` | **Always** | Acknowledge the answer *and* the correction; ask nothing |
| `INTERRUPTION` | `guards.py` | **Always** | Acknowledge the topic switch, return to the slot |
| `OFFTOPIC_AGENT` / `OFFTOPIC` | `guards.py` | **Always** | Decline honestly, steer back |
| `FOLLOWUP_RESPOND` | `_collect_slot` | **Always** | Answer the side question and re-ask, in one sentence |
| `FOLLOWUP_ANSWER` | `_collect_slot` (detour) | **Always** | Slot confirmed + question answerable from `Confirmed:` |
| `FOLLOWUP_PARK` | `_collect_slot` | **Always** | Slot confirmed + an update another flow owns; acknowledge only |
| `FOLLOWUP_DECLINE` | `_collect_slot` | **Always** | Slot confirmed + a question we cannot answer |

Disposition → guard mapping (`slot_manager.py:254`):
`answer → FOLLOWUP_RESPOND`, `park → FOLLOWUP_PARK`, legacy `answer_now`/`decline →
FOLLOWUP_RESPOND`. The extraction model is told to *always* emit `none`; Python decides,
so the mapping is mostly a compatibility shim for cached results.

### 6.4 Every path through `_collect_slot`, and what each speaks

`_collect_slot()` (`slot_manager.py:1833`) is the one function every agent's slots flow
through. In evaluation order:

| # | Branch | Condition | Reply | LLM 2? |
|---|---|---|---|---|
| 1 | **Already valid in state** | value present and validates | — (proceed) | No |
| 2 | **Clean answer** | `pre_extracted` normalises + validates, plain `ANSWERED` | — (proceed) | No |
| 3 | **Answer + side question** | `event_type = answered_with_followup`, or an answer plus a bare update request | `_handle_answered_followup` → `FOLLOWUP_*` | **Yes** |
| 4 | **Extraction rejected** | normaliser/validator said no | `cannot_provide`? escalate : `RETRY` | **Conditional** |
| 5 | **WAIT** | `event_type = wait` or `detect_wait_request()` | `MSG_WAIT_ACK` / `MSG_WAIT_NUDGE` pool | **No — never** |
| 6 | **CORRECTED + values (C1)** | `corrections` non-empty | `CORRECTION` ack, or a detour if the new value is invalid | **Yes** |
| 7 | **CORRECTED, no values (C2)** | bare `update_target` | `allow`→detour, `route`→hand off, else `FOLLOWUP_PARK` / `FOLLOWUP_RESPOND` | **Yes** (except `route`) |
| 8 | **Salvage** | the value is in the caller's words but the extractor missed it — the slot's own normaliser can read it | — (proceed) | **No** |
| 9 | **AMBIGUOUS, 1st time** | | `CLARIFY`, no attempt cost | **Conditional** |
| 10 | **AMBIGUOUS, 2nd time** | | `slot_fail` then `RETRY` | **Conditional** |
| 11 | **cannot_provide** | `detect_cannot_provide()` | Empathetic escalation message, transfer | **No** |
| 12 | **Question instead of an answer** | `followup_query` set, no value, within `MAX_FREE_FOLLOWUP_TURNS=2` | `FOLLOWUP_RESPOND` | **Yes** |
| 13 | **Plain non-answer** | everything else | `slot_fail` then `RETRY` | **Conditional** |
| 14 | **Exhausted** | `attempt_count >= 3` at any of the above | `build_slot_exhausted_message` + escalate | **No** |
| 15 | **First ask** | this slot has not been asked yet | `build_initial_prompt` / `build_transition_prompt` | **No** |

Note branch 8: **salvage** runs *after* WAIT and CORRECTED but *before* AMBIGUOUS and the
default retry, precisely so that "April twelvee nineteen eighty-eight" — which
`normalize_dob` understands perfectly — never costs the caller an attempt just because
the extractor returned nothing.

### 6.5 Generation is also triggered outside the slot loop

| Call site | File | Guard |
|---|---|---|
| `_generate_guard_response` | `core/guards.py:78` | `INTERRUPTION`, `OFFTOPIC_AGENT` |
| `_generate_correction_ack` | `core/slot_manager.py:576` | `CORRECTION` |
| `answer_side_question` | `core/slot_manager.py:1101` | `FOLLOWUP_RESPOND` |
| `_open_update_detour` | `core/slot_manager.py:1404` | `FOLLOWUP_ANSWER` |
| `BaseAgent._answer_unanswered_side_question` | `core/agent.py:145` | safety net — any question a hand-written handler forgot to answer |

The guards that are **always static, never generated**: `TRANSFER_REQUEST`, `ABUSE`,
`SELF_HARM`, `OFFTOPIC_GLOBAL`, non-member-caller routing. These take the whole turn and
the turn's side question is explicitly *discarded* with them — "a transfer or an abuse
escalation must not carry an aside about ID cards" (`guards.py:190`).

### 6.6 What happens to the generated sentence — the single-ask invariant

`sanitize_generated()` (`response_generator.py:421`) is applied to essentially every
generated sentence. It splits the text into sentences and drops or trims each one:

1. **Confirmed-slot re-ask** — a sentence with `?` that fuzzy-matches a slot in
   `Confirmed:` is stripped. The model must never re-ask something already given.
2. **Foreign-slot ask** — a sentence with `?` matching *any other known slot* while
   `collecting_slot` is set is stripped, and logged at **WARNING**. This is the
   cross-slot hallucination guard. `collecting_slot=""` means *nothing* is being
   collected, so every slot is foreign; `None` turns the check off.
3. **Next-slot mention / trailing question** — when Python is about to append its own
   static ask (`will_append_ask=True`), any competing question is removed so the
   appended one is the only one.
4. **Clause-level trimming** — if the model packed an answer *and* an ask into one
   sentence ("I can only ask for your information, not look it up, so could you tell me
   your first name?"), `_declarative_lead()` cuts at the clause boundary and keeps the
   answer. Dropping the whole sentence would lose the answer too and leave a canned
   fallback in its place.
5. **Empty result** → `fallback_text` if the caller supplied one, else the guard's entry
   in `_FALLBACKS` formatted with the slot label. `allow_empty=True` returns `""` for
   text that is only ever *prefixed* to a turn that already speaks.

Fuzzy matching uses `_slot_match_terms()`: the slot name with `_`→space (label
qualifiers after an em-dash dropped) plus `SLOT_ASK_SYNONYMS` entries ("date of birth",
"sms or email", "fax or email"). Terms that overlap the allowed slot's wording are
dropped in both directions, so a legitimate "phone number" ask is never stripped because
`phone_confirmed` also says "phone number".

Every strip is logged with the guard and the dropped sentence, for eval visibility.

**On exception**, `generate_recovery_message` logs and returns the `_FALLBACKS` template
for that guard, e.g. `RETRY → "Could you try your {slot_label} once more?"`. A caller
never hears silence because Gemini was down.

### 6.7 The decision, end to end

```
                       caller utterance
                              │
                    ┌─────────▼─────────┐
                    │  EXTRACTION LLM   │  → WorkerResult
                    └─────────┬─────────┘
                              │ reconcile_worker_result()
                    ┌─────────▼─────────┐
                    │      GUARDS       │  ABUSE/TRANSFER/SELF_HARM/OFFTOPIC_GLOBAL
                    └─────────┬─────────┘  → STATIC, turn over
                              │  INTERRUPTION / OFFTOPIC_AGENT → GENERATE
                    ┌─────────▼─────────┐
                    │   _collect_slot   │
                    └─────────┬─────────┘
            ┌─────────────────┼──────────────────┐
     value good          WAIT / salvage      needs a reply
            │            /  exhausted             │
       proceed ✓          STATIC ✓        ┌───────▼────────┐
                                          │ guard == RETRY │
                                          │  or CLARIFY ?  │
                                          └───┬────────┬───┘
                                           no │        │ yes
                                              │        ▼
                                              │  needs_freeform_response()?
                                              │        │
                                              │   no ──┴── yes
                                              │    │        │
                                              │  STATIC     │
                                              └────┬────────┘
                                                   ▼
                                            GENERATION LLM
                                                   │
                                          sanitize_generated()
                                                   │
                                        (empty?) → _FALLBACKS
                                                   ▼
                                              ask_member()
```

---

## 7. The third path: the follow-up agent's single call

`follow_up_agent` is the only agent that generates its answer **inside the extraction
call**. `agents/follow_up/llm.py` builds the normal `build_worker_input()` messages and
then prepends a **session snapshot** to the user content:

```
OUR CONVERSATION:
Member name: Monique Customer
Member email on file: monique dot customer at example dot com

Benefits:
  Individual deductible: $750 per calendar year
  Coinsurance: 20% after deductible is met
  …

Provider search:
  Provider type requested: Primary Care Physicians
  ZIP code used: 12139
  In-network provider list was sent to the member this call.

Claim adjustment:
  Reference number: 25357301
  Status: Under review
  Resolution timeline: 5 to 10 business days …
```

The model returns a `FollowUpResult` carrying both the classification
(`follow_up_intent`) and the prose (`answer`) in one forward pass — one call instead of
two, which matters against a 2.5 s p50 latency SLA.

Every email and URL in the snapshot is rendered in **spoken form** (`speak_email`,
`speak_url`) so any answer generated from it reads them out correctly on a voice call.

---

## 8. Limits, guards and configuration knobs

### Limits (`core/constants.py`)

| Constant | Value | Effect |
|---|---|---|
| `MAX_SLOT_ATTEMPTS` | 3 | Three failures on one slot → escalate to a human |
| `MAX_DEFLECTED_TURNS` | 4 | Shared budget across interruption/off-topic/redirect turns for the whole call |
| `MAX_WAIT_TURNS` | 3 | Consecutive "hold on"s before a gentle nudge |
| `MAX_FREE_FOLLOWUP_TURNS` | 2 | Question-only turns per slot before they start costing attempts |
| `MAX_ROUTER_LOOPS` | 25 | Orchestrator cycle cap |
| `_MAX_UPDATE_DETOURS` | 2 | Detours per target per call |
| `HISTORY_WINDOW_SIZE` | 6 | Default history window |

### Feature flags (`llm/config.py`)

| Env var | Default | Effect |
|---|---|---|
| `STATIC_RETRY_FAST_PATH` | `1` (on) | Set to `0`/`false`/`no`/`off` to send **every** recovery turn back through the generation LLM |
| `WORKER_TEMPERATURE` | `0` | Extraction determinism |
| `GEMINI_THINKING_BUDGET` | `0` | No thinking tokens on generation |

### Safety ordering (`core/guards.py`)

Guards act only at `guard_confidence >= 0.7`; below that, regex fallbacks run. Order:
non-member caller → `TRANSFER_REQUEST` → `ABUSE` → `SELF_HARM` → `INTERRUPTION` →
`OFFTOPIC_GLOBAL` → `OFFTOPIC_AGENT`.

Two important suppressions:

- A caller asking for a moment suppresses the *soft* guards (`INTERRUPTION`,
  `OFFTOPIC_*`) so the agent's own WAIT branch owns the turn. `TRANSFER`, `ABUSE`,
  `SELF_HARM` always win.
- An off-topic question asked *alongside* a valid slot value suppresses the off-topic
  guard, so the value is confirmed and the side question is declined by the slot
  pipeline — otherwise the guard fires first and the slot is never captured.

### Privacy

`llm/redaction.py::mask_confirmed` is applied centrally in `_render_payload`, so
`member_id` and `dob` reach the generation model as `on file` no matter what a call site
passes. Raw attempt counts never reach LLM 2 either — only the coarse `Tone:` label.

---

## 9. Why the code is shaped this way — the failure it was built around

A stray note sits at the top of `pyproject.toml` recording the bug that produced most of
§6. The transcript:

```
AI      Got that — and your date of birth?
Caller  Yeah. It's April twelfth nineteen eighty eight.
AI      I wasn't able to capture your last name after a few tries.
        Let me connect you with a representative…
```

The generation model, asked to re-ask for `last_name`, asked for the Member ID instead.
The caller answered *that*, `last_name` never advanced, and the call escalated on a slot
the caller had already given.

Four mechanisms in the current code exist because of that turn:

1. **`WorkerResult.needs_freeform_response`** — the extraction model, which has already
   read the utterance, says whether prose is even needed. No extra call.
2. **The static fast path** — a plain re-ask is rendered from a per-slot template, which
   is instant and *structurally incapable* of drifting onto another slot.
3. **`sanitize_generated`'s foreign-slot check** — when the model does run and asks for
   the wrong slot anyway, that sentence is stripped and logged at WARNING.
4. **`salvage_slot_value`** — before any of that, the slot's own normaliser gets a look
   at the raw utterance, so a value the extractor missed but the normaliser can read
   costs the caller nothing.

The same defensive pattern repeats throughout: **an LLM decides, Python verifies, and a
deterministic path is always available.** Extraction is schema-constrained and backed by
regex reconciliation; guards are LLM-first with keyword fallbacks; generation is optional,
bounded, sanitized, and has a canned line behind it. No single model failure can take the
call with it.

---

*Generated by [Claude Code](https://claude.ai/code)*
