# Graph Report - .  (2026-08-27)

## Corpus Check
- Large corpus: 202 files · ~233,240 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 2475 nodes · 7860 edges · 53 communities detected
- Extraction: 60% EXTRACTED · 40% INFERRED · 0% AMBIGUOUS · INFERRED: 3137 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Claim Adjustment Live Tests|Claim Adjustment Live Tests]]
- [[_COMMUNITY_Benefits & Care Wellness Agents|Benefits & Care Wellness Agents]]
- [[_COMMUNITY_Base Agent Classes|Base Agent Classes]]
- [[_COMMUNITY_Conversation Logger & Records|Conversation Logger & Records]]
- [[_COMMUNITY_Delivery Management Live Tests|Delivery Management Live Tests]]
- [[_COMMUNITY_BenefitsCare Followup Live Tests|Benefits/Care Followup Live Tests]]
- [[_COMMUNITY_LLM Config & Model Tiers|LLM Config & Model Tiers]]
- [[_COMMUNITY_Provider Search Live Tests|Provider Search Live Tests]]
- [[_COMMUNITY_Claim Adjustment SF Queries|Claim Adjustment SF Queries]]
- [[_COMMUNITY_Intake Agent Live Tests|Intake Agent Live Tests]]
- [[_COMMUNITY_SSN Fallback & Name Readback Helpers|SSN Fallback & Name Readback Helpers]]
- [[_COMMUNITY_LangGraph Build & CLI Entry|LangGraph Build & CLI Entry]]
- [[_COMMUNITY_Extraction Prompt Library|Extraction Prompt Library]]
- [[_COMMUNITY_Ground Truth Builder|Ground Truth Builder]]
- [[_COMMUNITY_Logging Utilities|Logging Utilities]]
- [[_COMMUNITY_Shared Constants & Detection Patterns|Shared Constants & Detection Patterns]]
- [[_COMMUNITY_Generation Event Prompts|Generation Event Prompts]]
- [[_COMMUNITY_LangGraph State & Call Events|LangGraph State & Call Events]]
- [[_COMMUNITY_Slot Ground Truth Mapping|Slot Ground Truth Mapping]]
- [[_COMMUNITY_CI GitHub Actions Output|CI GitHub Actions Output]]
- [[_COMMUNITY_Cache Helpers|Cache Helpers]]
- [[_COMMUNITY_Latency Report Formatting|Latency Report Formatting]]
- [[_COMMUNITY_App Lifespan & Main Entry|App Lifespan & Main Entry]]
- [[_COMMUNITY_Live E2E Test Package|Live E2E Test Package]]
- [[_COMMUNITY_Package Init 24|Package Init 24]]
- [[_COMMUNITY_Package Init 25|Package Init 25]]
- [[_COMMUNITY_Package Init 26|Package Init 26]]
- [[_COMMUNITY_Package Init 27|Package Init 27]]
- [[_COMMUNITY_Package Init 28|Package Init 28]]
- [[_COMMUNITY_Package Init 29|Package Init 29]]
- [[_COMMUNITY_Package Init 30|Package Init 30]]
- [[_COMMUNITY_Package Init 31|Package Init 31]]
- [[_COMMUNITY_Package Init 32|Package Init 32]]
- [[_COMMUNITY_Package Init 33|Package Init 33]]
- [[_COMMUNITY_Package Init 34|Package Init 34]]
- [[_COMMUNITY_Package Init 35|Package Init 35]]
- [[_COMMUNITY_Package Init 36|Package Init 36]]
- [[_COMMUNITY_Package Init 37|Package Init 37]]
- [[_COMMUNITY_Package Init 38|Package Init 38]]
- [[_COMMUNITY_Package Init 39|Package Init 39]]
- [[_COMMUNITY_Package Init 40|Package Init 40]]
- [[_COMMUNITY_Package Init 41|Package Init 41]]
- [[_COMMUNITY_Package Init 42|Package Init 42]]
- [[_COMMUNITY_Package Init 43|Package Init 43]]
- [[_COMMUNITY_Package Init 44|Package Init 44]]
- [[_COMMUNITY_Package Init 45|Package Init 45]]
- [[_COMMUNITY_Package Init 46|Package Init 46]]
- [[_COMMUNITY_Package Init 47|Package Init 47]]
- [[_COMMUNITY_Package Init 48|Package Init 48]]
- [[_COMMUNITY_Package Init 49|Package Init 49]]
- [[_COMMUNITY_Package Init 50|Package Init 50]]
- [[_COMMUNITY_Latency Metric Aggregation|Latency Metric Aggregation]]
- [[_COMMUNITY_Package Init 52|Package Init 52]]

## God Nodes (most connected - your core abstractions)
1. `ConversationRecord` - 600 edges
2. `assert_and_record()` - 494 edges
3. `run_conversation()` - 176 edges
4. `State` - 155 edges
5. `assert_not_escalated()` - 149 edges
6. `run_conversation()` - 100 edges
7. `ConversationContext` - 96 edges
8. `assert_provider_list_sent()` - 93 edges
9. `assert_not_escalated()` - 90 edges
10. `run_conversation()` - 80 edges

## Surprising Connections (you probably didn't know these)
- `lifespan()` --calls--> `warm_llm_connections()`  [INFERRED]
  main.py → src/agent/app_graph.py
- `_drive()` --calls--> `build_graph()`  [INFERRED]
  tests/live_e2e/harness.py → src/agent/app_graph.py
- `_new_graph()` --calls--> `build_graph()`  [INFERRED]
  tests/live_e2e/test_intent_change.py → src/agent/app_graph.py
- `run_preflight()` --calls--> `warm_llm_connections()`  [INFERRED]
  tests/live_e2e/preflight.py → src/agent/app_graph.py
- `verify_fixtures()` --calls--> `find_adjustment()`  [INFERRED]
  tests/live_e2e/preflight.py → src/agent/storage/queries/adjustments.py

## Hyperedges (group relationships)
- **Generation Event-Conditional Prompt Fragments** — recovery_base_prompt, event_retry, event_correction, event_followup_answer, event_followup_park, event_offtopic_agent, event_interruption, event_correction_ack, event_clarify, event_followup_decline, event_followup_respond [EXTRACTED 0.90]
- **Shared Extraction Header Variants** — extraction_header, extraction_header_extraction, extraction_header_core [INFERRED 0.85]
- **PCP Provider Search Flow Prompt Chain** — extraction_intake, extraction_verification_provider, extraction_provider_search, extraction_delivery_management, extraction_benefits, extraction_care_wellness, extraction_follow_up [INFERRED 0.80]
- **Claim Adjustment Flow Prompt Chain** — extraction_intake, extraction_verification_claims, extraction_claim_adjustment, extraction_records_coordination, extraction_notification_setup, extraction_follow_up_claims [INFERRED 0.80]
- **Eval Harness Strictly Separated Components** — readme_eval_architecture, transcript_pcp_happy_path, transcript_claim_happy_path [INFERRED 0.70]

## Communities

### Community 0 - "Claim Adjustment Live Tests"
Cohesion: 0.01
Nodes (399): assert_any_agent_message_contains(), assert_attempt_count_below_max(), assert_claim_flow_complete(), assert_claim_status_reported(), assert_escalated(), assert_member_verified(), assert_n2_notification_channel(), assert_not_escalated() (+391 more)

### Community 1 - "Benefits & Care Wellness Agents"
Cohesion: 0.02
Nodes (257): benefits_agent(), BenefitsAgent, care_wellness_agent(), claim_adjustment_agent(), _clean_amount(), closure_agent(), _completion_context(), delivery_management_agent() (+249 more)

### Community 2 - "Base Agent Classes"
Cohesion: 0.02
Nodes (205): ABC, BaseAgent, CareWellnessAgent, ClaimAdjustmentAgent, ClosureAgent, _detect_wait_follow_up(), EscalationAgent, FollowUpAgent (+197 more)

### Community 3 - "Conversation Logger & Records"
Cohesion: 0.02
Nodes (219): _append_summary_csv(), AssertionRecord, ConversationRecord, _now(), _percentile(), conversation_logger.py — Captures every turn of a live test run.  Writes per-run, Complete record of one test scenario execution., Return a formatted latency summary string (stdlib only). (+211 more)

### Community 4 - "Delivery Management Live Tests"
Cohesion: 0.03
Nodes (228): assert_and_record(), assert_any_agent_message_contains(), assert_benefits_offer_made(), assert_delivery_management_was_active(), assert_delivery_method(), assert_email_used(), assert_escalated(), assert_fax_used() (+220 more)

### Community 5 - "Benefits/Care Followup Live Tests"
Cohesion: 0.03
Nodes (165): assert_any_agent_message_contains(), assert_benefits_explained(), assert_call_closed(), _assert_caller_type_is(), assert_care_coach_accepted(), assert_care_coach_declined(), assert_care_coach_details_sent(), assert_care_coach_no_offer_sent() (+157 more)

### Community 6 - "LLM Config & Model Tiers"
Cohesion: 0.03
Nodes (104): Config, get_gemini_llm(), get_routing_llm(), get_worker_llm(), llm_config.py  Two model tiers:    get_worker_llm()     → GPT-4.1-mini on Azure, Orchestrator LLM — Gemini Flash 2.5 Lite via GCP service account.     Fast and c, Build ChatGoogleGenerativeAI using GCP service account credentials.      Uses la, Worker agent LLM — GPT-4.1-mini on Azure.     streaming=True required for stream (+96 more)

### Community 7 - "Provider Search Live Tests"
Cohesion: 0.05
Nodes (103): assert_any_agent_message_contains(), assert_escalated(), assert_not_escalated(), assert_provider_search_was_active(), assert_provider_type(), assert_routed_to(), assert_routed_to_delivery(), assert_zip_code_used() (+95 more)

### Community 8 - "Claim Adjustment SF Queries"
Cohesion: 0.03
Nodes (82): fetch_claim_request_delivery(), find_adjustment(), find_adjustment_by_claim_number(), find_adjustment_by_dos_and_billed(), _now(), _adjustments.py — async claim adjustment queries., Write the member's notification channel preference to Salesforce.     Reuses the, Look up an adjustment request by claim_number + member_id. (+74 more)

### Community 9 - "Intake Agent Live Tests"
Cohesion: 0.05
Nodes (100): Async factory fixture.  Call it inside your test with:          record = await r, run_intake_conversation(), _assert_active_agent_was_intake(), assert_agent_message_contains(), assert_any_agent_message_contains(), assert_call_ended(), assert_caller_type(), _assert_caller_type_handled() (+92 more)

### Community 10 - "SSN Fallback & Name Readback Helpers"
Cohesion: 0.05
Nodes (83): _build_name_readback_message(), _extract_ssn_from_text(), _is_member_id_denial(), _is_no_ssn_available(), _spell_name(), VerificationAgent, extract_ssn_decision(), _get_ssn_fallback_prompt() (+75 more)

### Community 11 - "LangGraph Build & CLI Entry"
Cohesion: 0.04
Nodes (59): run(), build_graph(), BaseModel, _load_env(), pytest_sessionfinish(), conftest.py — Pytest fixtures for live IntakeAgent tests.  .env loading happens, Print a summary table of all live test outcomes at the end of the session., Drive the LangGraph through a scripted conversation.      Pattern:       1. ainv (+51 more)

### Community 12 - "Extraction Prompt Library"
Cohesion: 0.04
Nodes (58): Benefits/Care Coach Extraction Rules, Care & Wellness Extraction Rules, Claim Adjustment Extraction, Delivery Management Extraction Rules, Follow-up Extraction (Provider path), Follow-up Extraction (Claims path), Extraction Header (shared guard/event rules), Extraction Header - Core Variant (intake) (+50 more)

### Community 13 - "Ground Truth Builder"
Cohesion: 0.07
Nodes (35): build_dynamic_ground_truth(), build_ground_truth(), _curated_override(), _entity_get(), _flow_for(), _internal_slot_fallback(), Ground-truth builder — decides what the simulated user should say in reply to a, Slot-fallback corrections for asks where the canonical slot table is     context (+27 more)

### Community 14 - "Logging Utilities"
Cohesion: 0.21
Nodes (8): CorrelationIdFallbackFilter, _ensure_log_dir(), get_logger(), _load_config(), Fallback JSON formatter that doesn't require python-json-logger., Create directory for any RotatingFileHandler with a filename., SafeJSONFormatter, SanitizingAdapter

### Community 15 - "Shared Constants & Detection Patterns"
Cohesion: 0.15
Nodes (2): constants.py — Shared detection patterns and limits for all agents.  Single sour, # NOTE: last_agent_signal is deliberately NOT cleared here. _build() applies

### Community 16 - "Generation Event Prompts"
Cohesion: 0.17
Nodes (12): Event: CLARIFY, Event: CORRECTION, Event: CORRECTION_ACK, Event: FOLLOWUP_ANSWER, Event: FOLLOWUP_DECLINE, Event: FOLLOWUP_PARK, Event: FOLLOWUP_RESPOND, Event: INTERRUPTION (+4 more)

### Community 17 - "LangGraph State & Call Events"
Cohesion: 0.48
Nodes (6): AgentCallEventData, AgentCallLifecycleEvent, CallAgentFieldData, CallAgentFieldEvent, SlotState, TypedDict

### Community 18 - "Slot Ground Truth Mapping"
Cohesion: 0.6
Nodes (4): _claim_map(), ground_truth_for_slot(), _pcp_map(), Maps slot labels to expected user responses for the PCP and claim-adjustment eva

### Community 19 - "CI GitHub Actions Output"
Cohesion: 0.5
Nodes (4): Write GitHub Actions workflow outputs for CI integration., Write per-step latency outputs to GITHUB_OUTPUT (no-op if not set)., report_to_github(), set_github_outputs()

### Community 20 - "Cache Helpers"
Cohesion: 0.67
Nodes (3): _clear(), clear_caches(), cache.py — Cache helpers (core agents only).

### Community 21 - "Latency Report Formatting"
Cohesion: 0.5
Nodes (1): Output formatting: Markdown table printed to stdout and JSON file writer.

### Community 22 - "App Lifespan & Main Entry"
Cohesion: 0.67
Nodes (1): lifespan()

### Community 23 - "Live E2E Test Package"
Cohesion: 1.0
Nodes (1): Live end-to-end conversation tests.  These tests drive the REAL LangGraph applic

### Community 24 - "Package Init 24"
Cohesion: 1.0
Nodes (0):

### Community 25 - "Package Init 25"
Cohesion: 1.0
Nodes (0):

### Community 26 - "Package Init 26"
Cohesion: 1.0
Nodes (0):

### Community 27 - "Package Init 27"
Cohesion: 1.0
Nodes (0):

### Community 28 - "Package Init 28"
Cohesion: 1.0
Nodes (0):

### Community 29 - "Package Init 29"
Cohesion: 1.0
Nodes (0):

### Community 30 - "Package Init 30"
Cohesion: 1.0
Nodes (0):

### Community 31 - "Package Init 31"
Cohesion: 1.0
Nodes (0):

### Community 32 - "Package Init 32"
Cohesion: 1.0
Nodes (0):

### Community 33 - "Package Init 33"
Cohesion: 1.0
Nodes (0):

### Community 34 - "Package Init 34"
Cohesion: 1.0
Nodes (0):

### Community 35 - "Package Init 35"
Cohesion: 1.0
Nodes (0):

### Community 36 - "Package Init 36"
Cohesion: 1.0
Nodes (0):

### Community 37 - "Package Init 37"
Cohesion: 1.0
Nodes (0):

### Community 38 - "Package Init 38"
Cohesion: 1.0
Nodes (0):

### Community 39 - "Package Init 39"
Cohesion: 1.0
Nodes (0):

### Community 40 - "Package Init 40"
Cohesion: 1.0
Nodes (0):

### Community 41 - "Package Init 41"
Cohesion: 1.0
Nodes (0):

### Community 42 - "Package Init 42"
Cohesion: 1.0
Nodes (0):

### Community 43 - "Package Init 43"
Cohesion: 1.0
Nodes (0):

### Community 44 - "Package Init 44"
Cohesion: 1.0
Nodes (0):

### Community 45 - "Package Init 45"
Cohesion: 1.0
Nodes (0):

### Community 46 - "Package Init 46"
Cohesion: 1.0
Nodes (0):

### Community 47 - "Package Init 47"
Cohesion: 1.0
Nodes (0):

### Community 48 - "Package Init 48"
Cohesion: 1.0
Nodes (0):

### Community 49 - "Package Init 49"
Cohesion: 1.0
Nodes (0):

### Community 50 - "Package Init 50"
Cohesion: 1.0
Nodes (0):

### Community 51 - "Latency Metric Aggregation"
Cohesion: 1.0
Nodes (1): Sum of time spent in nodes that made LLM calls.

### Community 52 - "Package Init 52"
Cohesion: 1.0
Nodes (0):

## Knowledge Gaps
- **212 isolated node(s):** `Build a compact turn-by-turn history for LLM context.`, `System prompt for LLM 1 (get_extraction_llm).     Combines extraction header + a`, `Minimal extraction prompt for agents that only need guard detection     and simp`, `Mid-tier extraction prompt for agents that collect structured slot     values an`, `System prompt for LLM 2 (response generator) and orchestrator.     Used by respo` (+207 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Live E2E Test Package`** (2 nodes): `Live end-to-end conversation tests.  These tests drive the REAL LangGraph applic`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 24`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 25`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 26`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 27`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 28`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 29`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 30`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 31`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 32`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 33`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 34`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 35`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 36`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 37`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 38`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 39`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 40`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 41`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 42`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 43`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 44`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 45`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 46`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 47`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 48`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 49`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 50`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Latency Metric Aggregation`** (1 nodes): `Sum of time spent in nodes that made LLM calls.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init 52`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConversationRecord` connect `Conversation Logger & Records` to `Claim Adjustment Live Tests`, `Benefits & Care Wellness Agents`, `Delivery Management Live Tests`, `Benefits/Care Followup Live Tests`, `Provider Search Live Tests`, `Intake Agent Live Tests`, `LangGraph Build & CLI Entry`?**
  _High betweenness centrality (0.191) - this node is a cross-community bridge._
- **Why does `assert_and_record()` connect `Delivery Management Live Tests` to `Claim Adjustment Live Tests`, `Conversation Logger & Records`, `Benefits/Care Followup Live Tests`, `Provider Search Live Tests`, `Intake Agent Live Tests`, `LangGraph Build & CLI Entry`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `assert_not_escalated()` connect `Claim Adjustment Live Tests` to `Benefits & Care Wellness Agents`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 589 inferred relationships involving `ConversationRecord` (e.g. with `test_verification_agent_live.py — Live integration tests for VerificationAgent.` and `Alias so verification tests read naturally. Same graph runner underneath.`) actually correct?**
  _`ConversationRecord` has 589 INFERRED edges - model-reasoned connections that need verification._
- **Are the 492 inferred relationships involving `assert_and_record()` (e.g. with `test_verification_provider_happy_path()` and `test_verification_claim_happy_path()`) actually correct?**
  _`assert_and_record()` has 492 INFERRED edges - model-reasoned connections that need verification._
- **Are the 153 inferred relationships involving `State` (e.g. with `app_graph.py — Trimmed LangGraph workflow (core agents only). Agents: intake · v` and `Conditional edge out of the verification node.      On a mid-call intent switch,`) actually correct?**
  _`State` has 153 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Build a compact turn-by-turn history for LLM context.`, `System prompt for LLM 1 (get_extraction_llm).     Combines extraction header + a`, `Minimal extraction prompt for agents that only need guard detection     and simp` to the rest of the system?**
  _212 weakly-connected nodes found - possible documentation gaps or missing edges._
