# Live Tests — IntakeAgent

End-to-end integration tests that drive a real LangGraph instance with a real
LLM.  Each test runs the full `build_graph()` pipeline, sends scripted user
inputs, and asserts on the resulting LangGraph state.

---

## Running the tests

### Prerequisites

Set at least one of these environment variables before running:

```bash
# Azure OpenAI (primary)
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_ENDPOINT=<your-endpoint>

# Google / Gemini (fallback)
export GOOGLE_API_KEY=<your-key>
```

Copy `.env.example` → `.env` and fill in values, then:

```bash
source .env   # or use python-dotenv in your shell
```

### Run all live tests

```bash
pytest -m live
```

### Run a single test

```bash
pytest -m live -k "test_intake_provider_services_happy_path" -v
```

### Run a test group

```bash
pytest -m live -k "guard" -v          # Group C
pytest -m live -k "caller_type" -v    # Group D
```

### Skip saving conversation transcripts

```bash
pytest -m live --no-save-conversations
```

### Custom transcript directory

```bash
pytest -m live --conversations-dir /tmp/my-test-runs
```

### Skip in CI (no credentials)

Tests auto-skip when `AZURE_OPENAI_API_KEY` and `GOOGLE_API_KEY` are both
absent — no extra pytest flag needed.

---

## Where conversation logs are saved

```
src/agent/tests/live/conversations/intake/
  20260602T143012_test_intake_provider_services_happy_path.json
  20260602T143045_test_intake_guard_abuse.json
  ...
  summary.csv
```

The `summary.csv` columns are:

| Column    | Description                              |
|-----------|------------------------------------------|
| test_name | pytest test function name                |
| outcome   | PASS / FAIL / ERROR                      |
| turns     | total conversation turns (incl. greeting)|
| intent    | `call_intent` in final state             |
| escalated | True if `escalation_reason` was set      |
| timestamp | ISO-8601 UTC timestamp                   |

---

## JSON transcript format

```json
{
  "test_name": "test_intake_guard_transfer_request",
  "scenario": "User immediately asks for human agent",
  "conversation_id": "abc12345",
  "outcome": "PASS",
  "failure_reason": "",
  "started_at": "2026-06-02T14:30:12Z",
  "ended_at": "2026-06-02T14:30:18Z",
  "total_turns": 2,
  "final_state": {
    "call_intent": null,
    "next_node": "escalation_agent",
    "escalation_reason": "Transfer requested during intake_agent",
    ...
  },
  "turns": [
    {
      "turn": 0,
      "user": "[SYSTEM START]",
      "agent": "Thank you for calling Sagility Health...",
      "state": {
        "call_intent": null,
        "next_node": "intake_agent",
        "is_interrupt": true
      }
    },
    {
      "turn": 1,
      "user": "I want to speak to a real person please",
      "agent": "No problem. Let me connect you to one of our live representatives...",
      "state": {
        "call_intent": null,
        "next_node": "escalation_agent",
        "escalation_reason": "Transfer requested during intake_agent"
      }
    }
  ],
  "assertions": [
    {"check": "escalation_triggered",       "result": "PASS", "detail": ""},
    {"check": "routes_to_escalation_agent", "result": "PASS", "detail": ""},
    {"check": "warm_transfer_message_contains_hold", "result": "PASS", "detail": ""}
  ]
}
```

---

## Test groups

| Group | Description                        | Tests |
|-------|------------------------------------|-------|
| A     | Happy-path intent classification   | 4     |
| B     | Unclear intent + retry flow        | 3     |
| C     | Guard triggers                     | 6     |
| D     | Caller type detection              | 3     |
| E     | Edge cases & combinations          | 4     |
| F     | Conversation continuity            | 2     |

---

## Adding new test scenarios

1. Open `test_intake_agent_live.py`.
2. Add a new `async def test_intake_<name>` function.
3. Decorate it with `@pytest.mark.live`.
4. Call `run_intake_conversation(user_inputs=[...], test_name=..., scenario=...)`.
5. Use `assert_and_record(record, [(lambda: ..., "label"), ...])` to register assertions.

Example skeleton:

```python
@pytest.mark.live
async def test_intake_my_new_scenario(run_intake_conversation, assert_and_record):
    """What this test verifies and why it matters."""
    record = await run_intake_conversation(
        user_inputs=["My scripted user input"],
        test_name="test_intake_my_new_scenario",
        scenario="Short description for the summary table",
    )
    assert_and_record(record, [
        (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
        (lambda: assert_not_escalated(record), "no_escalation"),
    ])
```

---

## Interpreting failures

**FAIL** — at least one assertion returned False.  Check the `assertions` list
in the JSON transcript for `"result": "FAIL"` entries.

**ERROR** — the graph itself threw an exception (LLM timeout, bad credentials,
unhandled code path).  The `failure_reason` field contains the exception message.

**Flaky tests** — live tests call a real LLM so occasional non-determinism is
expected.  Re-run with `-v --count=3` (requires `pytest-repeat`) or increase
assertion tolerances (e.g. use `assert_any_agent_message_contains` instead of
strict equality checks).

---

## Architecture notes

```
conftest.py
  └── run_intake_conversation fixture
        └── _run_graph_conversation()
              ├── build_graph(checkpointer=MemorySaver())
              ├── ainvoke({})                    ← greeting turn
              └── ainvoke(Command(resume=...))   ← each user turn

conversation_logger.py
  ├── ConversationRecord   (one per test run)
  ├── TurnRecord           (one per graph invocation)
  └── ConversationLogger   (manages output directory + CSV)
```

The graph pauses at `human_node` (via LangGraph `interrupt()`) whenever
`is_interrupt=True`.  Each `Command(resume=user_input)` resumes the graph,
which routes back through the appropriate agent.
