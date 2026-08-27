"""
conftest.py — Pytest fixtures for live IntakeAgent tests.

.env loading happens at import time so credential checks in test files
see the correct values regardless of how pytest is invoked.

Fixtures
--------
run_intake_conversation
    Build a fresh graph, drive it through a conversation script, and return
    a ConversationRecord.  Reusable across all live agent test files.

assert_and_record
    Thin wrapper that runs assertion callables, records each result into the
    ConversationRecord, and defers failures to the end of the test so every
    check is always reported (not just the first failure).

CLI flags added
---------------
--save-conversations     Save JSON transcripts (default: True)
--conversations-dir      Directory for transcripts (default: src/agent/tests/live/conversations/intake)

Usage
-----
    pytest -m live
    pytest -m live --conversations-dir /tmp/my-runs
    pytest -m live --no-save-conversations
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Callable, List

import pytest

from agent.tests.live.conversation_logger import ConversationLogger, ConversationRecord

# ---------------------------------------------------------------------------
# .env loading — runs at import time, before any credential checks
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Walk up from this file to find a .env and load it into os.environ."""
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        env_file = candidate / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # Never overwrite values already set in the shell environment
                if k and k not in os.environ:
                    os.environ[k] = v
            break


_load_env()


# ---------------------------------------------------------------------------
# pytest CLI option registration
# ---------------------------------------------------------------------------

_DEFAULT_CONV_DIR = str(Path(__file__).parent / "conversations" / "intake")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--save-conversations",
        action="store_true",
        default=True,
        help="Save JSON conversation transcripts after each live test (default: True).",
    )
    parser.addoption(
        "--no-save-conversations",
        action="store_true",
        default=False,
        help="Disable saving JSON conversation transcripts.",
    )
    parser.addoption(
        "--conversations-dir",
        default=_DEFAULT_CONV_DIR,
        help=f"Directory for conversation transcripts (default: {_DEFAULT_CONV_DIR}).",
    )


# ---------------------------------------------------------------------------
# Session-scoped summary table
# ---------------------------------------------------------------------------

_session_results: List[dict] = []


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print a summary table of all live test outcomes at the end of the session."""
    live_results = [r for r in _session_results if r]
    if not live_results:
        return

    print("\n\n" + "=" * 80)
    print("  LIVE TEST SUMMARY")
    print("=" * 80)
    print(f"  {'TEST NAME':<55} {'OUTCOME':<8} {'TURNS':<6} {'INTENT'}")
    print("-" * 80)
    for r in live_results:
        icon = "✓" if r["outcome"] == "PASS" else "✗"
        print(f"  {icon} {r['test_name']:<53} {r['outcome']:<8} {r['turns']:<6} {r['intent'] or '—'}")
    passed = sum(1 for r in live_results if r["outcome"] == "PASS")
    print("-" * 80)
    print(f"  {passed}/{len(live_results)} passed")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Core graph runner
# ---------------------------------------------------------------------------


async def _run_graph_conversation(
    user_inputs: list[str],
    record: ConversationRecord,
) -> ConversationRecord:
    """
    Drive the LangGraph through a scripted conversation.

    Pattern:
      1. ainvoke({}) to get the greeting (graph pauses at first interrupt).
      2. For each user turn: ainvoke(Command(resume=user_input)).
      3. Stop when next_node == END or all inputs consumed.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END
    from langgraph.types import Command

    from agent.app_graph import build_graph

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # ── Turn 0: initial invocation (receives greeting) ────────────────────
    try:
        t0 = time.perf_counter()
        state = await graph.ainvoke({}, config=config)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        record.finalize("ERROR", f"Initial ainvoke failed: {exc}")
        return record

    record.add_turn("[SYSTEM START]", state)
    record.turns[-1].duration_sec = elapsed

    # ── Subsequent turns ──────────────────────────────────────────────────
    for user_input in user_inputs:
        current_state = record.final_state

        # Stop if the graph already ended
        if current_state.get("next_node") == END or current_state.get("next_node") == "__end__":
            break

        try:
            t0 = time.perf_counter()
            state = await graph.ainvoke(Command(resume=user_input), config=config)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            record.finalize("ERROR", f"ainvoke failed on turn '{user_input}': {exc}")
            return record

        record.add_turn(user_input, state)
        record.turns[-1].duration_sec = elapsed

        if state.get("next_node") in (END, "__end__"):
            break

    return record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conversations_dir(request: pytest.FixtureRequest) -> str:
    no_save = request.config.getoption("--no-save-conversations", default=False)
    if no_save:
        return ""
    return request.config.getoption("--conversations-dir", default=_DEFAULT_CONV_DIR)


@pytest.fixture
def run_intake_conversation(request: pytest.FixtureRequest, conversations_dir: str):
    """
    Async factory fixture.  Call it inside your test with:

        record = await run_intake_conversation(
            user_inputs=["I need a doctor"],
            test_name="test_something",
            scenario="Short description of what this test verifies",
        )

    Returns a ConversationRecord.  The transcript is saved automatically
    unless --no-save-conversations is passed.
    """
    # logger = ConversationLogger(conversations_dir) if conversations_dir else None

    async def _factory(
        user_inputs: list[str],
        test_name: str,
        scenario: str,
    ) -> ConversationRecord:
        conv_id = str(uuid.uuid4())[:8]
        record = ConversationRecord(
            test_name=test_name,
            scenario_description=scenario,
            conversation_id=conv_id,
            started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        )
        record = await _run_graph_conversation(user_inputs, record)
        return record

    yield _factory

    # Post-test: save and register for summary table
    # (actual save happens inside assert_and_record or the test itself)


@pytest.fixture
def assert_and_record(conversations_dir: str):
    """
    Returns a function that:
      - Runs every assertion callable
      - Records PASS/FAIL per check into the ConversationRecord
      - Collects all failures instead of stopping at the first one
      - Saves the record then raises a combined AssertionError if any failed

    Usage:
        assert_and_record(record, [
            (lambda: assert_intent(record, "provider_services"), "intent==provider_services"),
            (lambda: assert_routed_to(record, "verification_agent"), "routes_to_verification"),
        ])
    """
    logger = ConversationLogger(conversations_dir) if conversations_dir else None

    def _run(record: ConversationRecord, checks: list[tuple[Callable, str]]) -> None:
        failures = []
        for fn, label in checks:
            try:
                fn()
                record.record_assertion(label, passed=True)
            except AssertionError as exc:
                record.record_assertion(label, passed=False, detail=str(exc))
                failures.append(f"[{label}] {exc}")

        outcome = "PASS" if not failures else "FAIL"
        failure_reason = "; ".join(failures) if failures else ""
        record.finalize(outcome, failure_reason)

        # Register for session summary
        _session_results.append(
            {
                "test_name": record.test_name,
                "outcome": outcome,
                "turns": record.total_turns,
                "intent": record.final_state.get("call_intent", ""),
            }
        )

        # Always print the conversation and latency summary for visibility
        record.print_conversation()
        print(record.latency_summary())

        # Save transcript
        if logger:
            try:
                path = logger.save(record)
                print(f"  Transcript saved: {path}")
            except Exception as exc:
                print(f"  Warning: could not save transcript: {exc}")

        if failures:
            raise AssertionError("Live test failures:\n" + "\n".join(failures))

    return _run
