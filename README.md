# Sagility Health — Conversational Member Services Agent

A multi-agent voice/chat AI system for health insurance member services. Built on **LangGraph**, it routes calls through a graph of specialised agents that collect identity, look up records in Salesforce, and perform actions — all while staying within strict latency, quality, and safety constraints.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Agent Graph](#agent-graph)
  - [Agent Descriptions](#agent-descriptions)
  - [LLM Tiers](#llm-tiers)
  - [Salesforce Integration](#salesforce-integration)
- [Supported Workflows](#supported-workflows)
  - [Provider Search (PCP)](#provider-search-pcp)
  - [Claim Adjustment Follow-up](#claim-adjustment-follow-up)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Testing](#testing)
  - [CI Pipeline](#ci-pipeline)
  - [Conversational Evaluation Benchmark](#conversational-evaluation-benchmark)
  - [Latency Benchmark](#latency-benchmark)
  - [Live Integration Tests](#live-integration-tests)
- [Evaluation Architecture](#evaluation-architecture)
- [Safety & Guards](#safety--guards)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## Overview

The system acts as an automated Member Services Representative. A caller is greeted, their intent is classified, their identity is verified against Salesforce, and then one or more specialised agents handle the request end-to-end — including delivering provider lists, processing claim adjustment follow-ups, explaining benefits, enrolling in Care Coach programmes, and setting up notifications.

Key design goals:

- **Latency-first**: LLM-only latency SLA of 2.5 s (p50), tracked per graph node
- **Resilient slot collection**: each data point retried up to 3 times with escalation fallback
- **Independent evaluation**: simulator and ground-truth pipelines are strictly separated
- **Safety by default**: every turn passes through an ordered guard stack before proceeding

---

## Architecture

### Agent Graph

```
intake_agent
    │
    └─► verification_agent
              │
              ├─► provider_search_agent
              │         └─► delivery_management_agent
              │                   └─► benefits_agent
              │                             └─► care_wellness_agent
              │                                       └─► follow_up_agent
              │                                                 └─► closure_agent
              │
              └─► claim_adjustment_agent
                        │
                        ├─► records_coordination_agent
                        │         └─► notification_setup_agent
                        │                   └─► follow_up_agent
                        │                             └─► closure_agent
                        │
                        └─► notification_setup_agent (records not required)

All agents → escalation_agent (on guard trigger, slot exhaustion, or tool failure)
```

Each node communicates via `AgentSignal` status codes (`COMPLETE`, `ESCALATE`, `BLOCKED`) written into LangGraph state. The **orchestrator** reads those signals and applies a deterministic fast-path before invoking the LLM routing layer. A human interrupt node (`human_node`) pauses the graph between every agent turn and waits for the member's next utterance.

### Agent Descriptions

| Agent | Responsibility |
|---|---|
| `intake_agent` | Greeting; intent classification (`provider_services`, `claim_services`, `out_of_scope`) |
| `verification_agent` | Collects first name, last name, member ID, DOB; SF lookup; relationship/phone confirmation |
| `provider_search_agent` | Collects provider type and confirms/updates ZIP code |
| `delivery_management_agent` | Confirms delivery method (fax/email), dispatches provider list, makes benefits offer |
| `benefits_agent` | Fetches benefit plan from SF; explains deductible/coinsurance/OOP; offers Care Coach |
| `care_wellness_agent` | Dispatches Care Coach details to confirmed delivery contact |
| `follow_up_agent` | Handles post-call questions from session context; classifies DONE/QUESTION/UPDATE_REQUEST |
| `closure_agent` | Delivers goodbye message; sets `next_node=END` |
| `escalation_agent` | Warm transfer; writes `AgentCallTransfer` metadata event; sets `next_node=END` |
| `claim_adjustment_agent` | Collects reference number; SF lookup for claim status; routes to records or notification |
| `records_coordination_agent` | Manages medical records upload link, doctor-direct, and Personal Guide branches |
| `notification_setup_agent` | Collects N1 (outreach status) and N2 (timeline progress) notification preferences |
| `orchestrator` | LLM-based routing with deterministic fast-path; safeguard overrides; loop cap |

### LLM Tiers

| Tier | Model | Used for |
|---|---|---|
| Extraction (LLM 1) | Azure OpenAI GPT-4.1-mini (`WORKER_DEPLOYMENT`) | Slot extraction, guard classification, structured output |
| Follow-up | Same model, higher `max_tokens` | Combined intent classification + answer generation |
| Generation (LLM 2) | Gemini 2.5 Flash Lite via GCP service account | Recovery/retry utterance generation |
| Routing | Azure OpenAI (same deployment) | Orchestrator next-agent decision |

LLM instances are cached via `lru_cache` — one instance per process, per tier.

### Salesforce Integration

All member data lives in a custom Salesforce org. The async `SalesforceClient` uses OAuth2 refresh tokens, HTTP keep-alive via `httpx.AsyncClient`, and background token pre-refresh to minimise latency. Key objects:

| SF Object | Usage |
|---|---|
| `M_Member__c` | Identity verification, contact fields |
| `M_Benefit_Plan__c` | Deductible, coinsurance, OOP max |
| `M_Adjustment_Request__c` | Claim status and last-update date |
| `M_Provider_Update__c` | Provider list dispatch records |
| `M_Claim_Upload_Link__c` | Secure upload link tracking |
| `M_Claim_Outreach__c` | Personal Guide outreach records |
| `M_Notification_Preference__c` | Member notification channel/contact |

Benefits are **prefetched concurrently** with the member lookup using `asyncio.gather()` during verification, so `benefits_agent` can skip a second SF call.

---

## Supported Workflows

### Provider Search (PCP)

```
Greeting → Intent → Verify identity → Provider type + ZIP
       → Delivery method (fax/email) → Dispatch list
       → Benefits offer → Care Coach offer
       → Follow-up Q&A → Goodbye
```

Scenario tags: `pcp_happy_path`, `pcp_clarification_zip`, `pcp_correction_first_name`, `pcp_correction_member_id`, `pcp_clarification_fax`

### Claim Adjustment Follow-up

```
Greeting → Intent → Verify identity (with phone confirmation)
       → Reference number → SF claim lookup → Status report
       → Records coordination branch:
           A. Member uploads (link sent to email)
           B. Doctor sends directly (acknowledge + offer link)
           C. Personal Guide contacts provider
           D. Member declines all → escalate
       → Notification setup (N1: outreach status, N2: timeline updates)
       → Follow-up Q&A → Goodbye
```

Scenario tags: `claim_adjustment_happy_path`, `claim_adjustment_no_proceed`, `claim_adjustment_upload_only`, `claim_adjustment_guide_only`

---

## Project Structure

```
.
├── src/
│   └── agent/
│       ├── agents/                  # One sub-package per agent
│       │   ├── intake/
│       │   ├── verification/
│       │   ├── provider_search/
│       │   ├── delivery_management/
│       │   ├── benefits/
│       │   ├── care_wellness/
│       │   ├── follow_up/
│       │   ├── closure/
│       │   ├── escalation/
│       │   ├── claim_adjustment/
│       │   ├── records_coordination/
│       │   └── notification_setup/
│       │
│       ├── core/                    # BaseAgent, guards, signals, slot manager
│       ├── conversation/            # ConversationContext (session state)
│       ├── llm/                     # LLM config, extractor, response generator, schema
│       ├── orchestration/           # Orchestrator, fast-path, safeguards
│       ├── prompts/                 # Markdown prompt files (extraction/, generation/, system/)
│       ├── responses/               # Static messages, response builder
│       ├── slots/                   # Normalizers, validators, pipeline, types
│       ├── storage/                 # SF client, db layer, query modules, tools
│       ├── tests/live/              # Live integration test suite
│       ├── app_graph.py             # LangGraph build_graph() entry point
│       ├── state.py                 # LangGraph State TypedDict
│       └── utils.py                 # Shared helpers
│
├── scripts/
│   ├── conversational_workload/     # Eval harness (runner, simulator, judge, ground truth)
│   └── latency_workload/            # Per-node latency benchmark
│
└── .github/workflows/
    ├── ci.yml                       # Pre-commit + LangGraph Docker build + Tenable scan
    ├── conversation-eval-bench.yml  # Conversational quality gate on every PR
    └── langgraph-latency-bench.yml  # Per-node latency gate on every PR
```

Each agent package follows a consistent structure:

```
agents/<name>/
├── agent.py      # Agent class + async function entry point
├── constants.py  # Message pools, log labels, limits
├── handlers.py   # SF tool calls (pure async functions)
├── llm.py        # Single LLM extraction call, returns WorkerResult
└── pipelines.py  # SlotPipeline configurations (where applicable)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- Docker (for LangGraph deployment)
- Access to:
  - Azure OpenAI with a deployed GPT-4.1-mini endpoint
  - Google Cloud project with Gemini access (for generation LLM)
  - Salesforce org with the custom objects listed above

### Installation

```bash
git clone <repo-url>
cd <repo>
uv sync
```

### Environment Variables

Create a `.env` file in the project root (or export variables in your shell):

```dotenv
# Azure OpenAI — extraction, follow-up, routing LLMs
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
OPENAI_API_VERSION=2025-01-01-preview
WORKER_DEPLOYMENT=gpt-5.4-nano

# Gemini — generation LLM (recovery messages)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-lite
GCP_PROJECT_ID=
GCP_LOCATION=global
GEMINI_THINKING_LEVEL=low
GEMINI_THINKING_BUDGET=0
GCP_SA_BASE64=         # base64-encoded GCP service account JSON

# Salesforce
SF_CLIENT_ID=
SF_CLIENT_SECRET=
SF_REFRESH_TOKEN=
SF_TOKEN_URL=https://login.salesforce.com/services/oauth2/token
SF_API_VERSION=v60.0
SF_INSTANCE_URL=

# LangSmith (optional tracing)
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=conversation-agent-member-sda
```

---

## Running Locally

```bash
# Start the LangGraph dev server
langgraph dev

# Or run a single evaluation pass
python scripts/conversational_workload/run_eval.py

# Run the latency benchmark
python -m scripts.latency_workload.main \
  --iterations 5 \
  --threshold-sec 2.5 \
  --sla-metric llm_sec \
  --scenarios-path scripts/latency_workload/scenarios
```

---

## Testing

### CI Pipeline

On every pull request to `develop`, `main`, `stage`, or `test`, three workflows run in parallel:

| Workflow | What it does |
|---|---|
| `ci.yml` | Pre-commit hooks (`ruff`, `mypy`, etc.) + LangGraph Docker build + Tenable container scan |
| `conversation-eval-bench.yml` | Runs all 9 eval scenarios, comments score table on the PR, fails if any scenario drops below `SCORE_THRESHOLD=0.75` |
| `langgraph-latency-bench.yml` | Runs 5 iterations per scenario, comments per-node latency table on the PR, fails if LLM-only avg exceeds `THRESHOLD_SEC=2.5` |

### Conversational Evaluation Benchmark

The eval harness tests agent quality end-to-end across 9 scenarios (5 PCP + 4 claim) without using recorded transcripts for the simulator.

**Three independent components:**

```
static transcript  →  ground_truth_builder  →  expected user response
                                                        │
live agent turn    →  user_simulator (LLM)  →  actual user response
                                                        │
                        judge (LLM)  ←──────────────────┘
                           │
                    score (intent, constraint, completeness, naturalness)
```

- **Ground truth** (`transcript_cursor.py`): walks a static transcript in lockstep with the live conversation using a position-anchored cursor with look-ahead/look-back tolerance
- **Simulator** (`user_simulator.py`): an LLM given only the live agent message, member entity data, and a scenario persona — never sees the transcript
- **Judge** (`judge.py`): scores the simulator response against ground truth on 4 dimensions; fast-path skips the LLM call on exact/substring matches

Run locally:

```bash
ITERATIONS=3 python scripts/conversational_workload/run_eval.py
```

Output: `scripts/conversational_workload/results/conversation_eval_report.json`

Key metrics per scenario:

| Metric | Description |
|---|---|
| `average_final_score` | Mean per-turn score (0–1). Gate threshold: 0.75 |
| `average_exact_match_rate` | Fraction of turns where fast-path matched without LLM judge |
| `total_llm_judge_calls` | LLM judge invocations (high = agent deviating from expected flow) |
| `average_pass_rate` | Fraction of turns scoring ≥ 0.8 |

### Latency Benchmark

Measures **per-graph-node** latency using LangGraph's `astream_events` API. Separates controllable LLM-call time (`llm_sec`) from framework overhead (`total_sec`).

```bash
python -m scripts.latency_workload.main \
  --iterations 10 \
  --threshold-sec 2.5 \
  --sla-metric llm_sec
```

Output: `scripts/latency_workload/results/bench-metrics.json`

Reports: per-turn p50/p95/p99/avg/std_dev for both `total_sec` and `llm_sec`, plus a per-node breakdown sorted by average latency.

### Live Integration Tests

A comprehensive pytest suite in `src/agent/tests/live/` runs against a real LLM and Salesforce sandbox. Tests are tagged `@pytest.mark.live` and auto-skip when `AZURE_OPENAI_API_KEY` is absent.

```bash
# Run all live tests
pytest -m live -v

# Run a specific agent group
pytest -m live src/agent/tests/live/test_verification_agent_live.py -k "happy_path" -v

# Run with latency summary output
pytest -m live src/agent/tests/live/test_delivery_management_agent_live.py -k "latency" -v
```

Live test coverage by agent:

| Test file | Scenarios covered |
|---|---|
| `test_intake_agent_live.py` | Happy path, unclear → clarify, all guards, caller type detection, latency |
| `test_verification_agent_live.py` | Happy path, format errors, exhaustion, SF lookup failures, corrections, AMBIGUOUS events, guards, relationship/phone variants, bonus extraction, re-entry, latency |
| `test_provider_search_agent_live.py` | Provider type normalisation, unsupported types, ZIP confirmation/rejection/update |
| `test_delivery_management_agent_live.py` | Delivery method variants, fax/email confirmation bias rule, inline update, benefits offer, guards, exhaustion, latency |
| `test_benefits_care_followup_live.py` | Benefits offer variations, Care Coach offer variations, follow-up intent classification, cannot-answer escalation, guards, latency |
| `test_claim_adjustment_agent_live.py` | Reference number collection, phone confirmation, records branches, email confirmed slot, notification setup N1/N2 combinations, follow-up claims, retry paths, latency |

---

## Evaluation Architecture

The eval system is designed around a strict separation of concerns:

```
ground_truth_builder.py   ← reads static_transcripts/
user_simulator.py         ← reads only: live AI message + entity data + scenario persona
judge.py                  ← scores: simulator output vs ground truth
runner.py                 ← orchestrates: these three must NEVER share data
```

Transcript files in `scripts/conversational_workload/static_transcripts/` define the canonical human responses for each scenario. The `TranscriptCursor` walks these files positionally with configurable look-ahead (`LOOKAHEAD=4`) and look-back (`LOOKBACK=1`) tolerances to handle agent rephrasing gracefully.

Scenario personas in `user_simulator.py` encode behavioural instructions for the simulator LLM — e.g. "when asked for your Member ID, give a slightly wrong one: 'm nine oh seven five oh two'" — so the simulator can reproduce clarification and correction scenarios independently of the ground-truth transcript.

---

## Safety & Guards

Every agent turn passes through `run_conversation_guards()` before slot extraction results are applied. Guards fire in priority order:

| Guard | Trigger | Response |
|---|---|---|
| `NON_MEMBER_CALLER` | Caller identifies as provider/employer | Route to dedicated line, `next_node=END` |
| `TRANSFER_REQUEST` | Caller requests human agent | Warm transfer message, escalate |
| `ABUSE` | Explicit profanity/threats (LLM ≥0.9 or regex fallback) | Escalate with `MSG_ABUSE_ESCALATION` |
| `SELF_HARM` | Self-harm ideation | Escalate with compassionate `MSG_SELF_HARM_ESCALATION` |
| `INTERRUPTION` | Mid-flow topic switch | LLM-generated acknowledgement, return to slot |
| `OFFTOPIC_GLOBAL` | Non-healthcare topic | Static redirect; escalate after 3 occurrences |

Additional safeguards:

- **Slot exhaustion**: every slot has `MAX_SLOT_ATTEMPTS=3`; three failures trigger `signal_escalate`
- **Router loop cap**: `MAX_ROUTER_LOOPS=25` prevents infinite orchestrator cycles
- **Locked slots**: `CALLER_LOCKED_SLOTS` (e.g. `member_status_verify`, `zip_code`) cannot be overwritten by caller corrections
- **Cascade clears**: correcting `first_name` clears `last_name`; correcting `member_id` clears `dob`

---

## Deployment

The system is packaged as a LangGraph application:

```bash
# Build Docker image
langgraph build -t <image-name>:latest

# Container scan (Tenable Cloud Security)
# Runs automatically in CI via .github/workflows/ci.yml
```

The LangGraph deployment provides its own checkpointer. When running locally via `langgraph dev`, a `MemorySaver` is used.

The `warm_llm_connections()` coroutine in `app_graph.py` should be called from your ASGI lifespan handler to pre-warm Azure OpenAI and Salesforce HTTP connections before the first call:

```python
from agent.app_graph import warm_llm_connections

@asynccontextmanager
async def lifespan(app):
    await warm_llm_connections()
    yield
```

---

## Contributing

1. Run pre-commit before pushing: `pre-commit run --all-files`
2. All CI checks must pass: pre-commit, Docker build, Tenable scan, eval bench, latency bench
3. When adding a new agent:
   - Follow the `agents/<name>/` package structure
   - Add scenario(s) to `scripts/conversational_workload/static_transcripts/`
   - Register the scenario in `run_eval.py` and add an entry to `_SCENARIO_FILE_MAP` in `transcript_cursor.py`
   - Add live integration tests under `src/agent/tests/live/`
4. When editing a transcript or simulator persona, run the consistency checker:
   ```bash
   python scripts/conversational_workload/validate_transcripts.py
   ```
5. Prompt changes should be validated by running the eval bench locally before pushing:
   ```bash
   ITERATIONS=3 python scripts/conversational_workload/run_eval.py
   ```
