"""
harness.py — Reusable async driver for live end-to-end conversation scenarios.

Mirrors the mechanics of scripts/conversational_workload/runner.py:

    state = await graph.ainvoke({}, config=config)
    while turns remain:
        if state["next_node"] == END: break
        if state["is_interrupt"]:
            ai_msg = extract_last_ai_message(state["messages"])
            user = next scripted turn          # script exhausted → fail w/ transcript
            state = await graph.ainvoke(Command(resume=user), config=config)
        else:
            state = await graph.ainvoke(Command(resume=""), config=config)

NOTHING is mocked: the graph is built via agent.app_graph.build_graph with a
MemorySaver checkpointer and every LLM / Salesforce call is live.

Because agent question phrasing is LLM-generated, assertions never compare
exact sentences. They check:
  - state keys (final_state)
  - escalation reasons (substring / regex over every reason source)
  - AgentCallTransfer metadata events (accumulated across the whole run)
  - END / interrupt flags
  - tolerant case-insensitive regexes over the transcript
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END

_TERMINAL = (END, "END")

logger = logging.getLogger("live_e2e")

DEFAULT_SCENARIO_TIMEOUT_S = 240.0


# ──────────────────────────────────────────────────────────────────────────────
# Scenario definition dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TurnExpectation:
    """Optional per-turn assertion checked when the AI asks before the Nth user turn."""

    ai_contains: list[str] | None = None  # case-insensitive regex alternatives
    slot_awaiting: str | None = None  # expected state["awaiting_slot"] after the AI asks


@dataclass
class Expected:
    completed: bool  # graph reached END
    escalated: bool = False  # transfer event OR escalation_reference_number
    escalation_reason_contains: str | None = None  # substring over all reason sources
    escalation_reason_regex: str | None = None  # regex over all reason sources
    transfer_event: bool = False  # AgentCallTransfer in metadata_events
    transfer_initiator: str | None = None  # "Agent" | "Caller"
    final_state: dict = field(default_factory=dict)  # key → expected value or callable predicate
    last_ai_contains: list[str] = field(default_factory=list)  # regexes on the final AI message
    transcript_contains: list[str] = field(default_factory=list)  # regexes over all AI lines
    # regex -> exact number of AI lines that must match (e.g. a spelled-name
    # read-back that must appear exactly once across the whole transcript)
    transcript_count: dict[str, int] = field(default_factory=dict)
    final_is_interrupt: bool | None = None  # assert state["is_interrupt"] at END
    max_turns: int = 35


@dataclass
class Scenario:
    name: str
    flow: str  # "pcp" | "claim"
    user_turns: list[str]  # scripted utterances, consumed in order
    expect: Expected
    # keyed by 0-based user-turn index: expectation on the AI prompt that PRECEDES that turn
    turn_expectations: dict[int, TurnExpectation] = field(default_factory=dict)
    retries: int = 0  # rerun budget for LLM-nondeterministic scenarios (guards)
    mutating: bool = False  # writes Salesforce data — excluded by --skip-mutating
    timeout_s: float = DEFAULT_SCENARIO_TIMEOUT_S
    # async callables (final_state) -> failure string | None, run after the conversation
    post_checks: list = field(default_factory=list)
    notes: str = ""


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    turns: int = 0
    duration_s: float = 0.0
    escalation_reason: str = ""
    flaky: bool = False  # passed only on retry
    attempts: int = 1
    result_file: str = ""


class ScenarioFailure(Exception):
    """Hard driver failure (script exhausted, timeout, max turns)."""


# ──────────────────────────────────────────────────────────────────────────────
# Message / state helpers
# ──────────────────────────────────────────────────────────────────────────────


def extract_last_ai_message(messages: list) -> str:
    """Return content of the most recent assistant message (dict or LangChain Message)."""
    for m in reversed(messages or []):
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        else:
            role = getattr(m, "type", "") or getattr(m, "role", "")
            content = getattr(m, "content", "")
        if role in ("assistant", "ai"):
            return content.strip() if isinstance(content, str) else str(content).strip()
    return ""


def _simplify_messages(messages: list) -> list[dict]:
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            role, content = m.get("role", ""), m.get("content", "")
        else:
            role = getattr(m, "type", "") or getattr(m, "role", "")
            content = getattr(m, "content", "")
        out.append({"role": str(role), "content": str(content)})
    return out


def _transfer_events(events: list[dict]) -> list[dict]:
    return [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("eventType") == "AgentCallEvent"
        and (e.get("data") or {}).get("eventName") == "AgentCallTransfer"
    ]


def pool_regex(pool) -> str:
    """
    Build a tolerant case-insensitive regex matching ANY member of a static
    message pool (constants from src/agent). Use this where wording comes from
    a constant pool, so a re-pick never breaks the assertion.
    Matches on the first ~50 characters of each pool member (escalation_agent
    appends a reference-number suffix and may strip trailing punctuation).
    """
    if isinstance(pool, str):
        pool = [pool]
    prefixes = []
    for msg in pool:
        prefix = msg.strip().rstrip(". ")[:50]
        prefixes.append(re.escape(prefix))
    return "(" + "|".join(prefixes) + ")"


def _matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text or "", re.IGNORECASE) is not None


# ──────────────────────────────────────────────────────────────────────────────
# Run recorder — survives timeouts so partial transcripts are always dumped
# ──────────────────────────────────────────────────────────────────────────────


class RunRecorder:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.transcript: list[dict] = []  # [{"role": "ai"|"user", "content": str}]
        self.events: list[dict] = []  # cumulative metadata_events seen at every pause
        self.reasons: list[str] = []  # every escalation-reason string observed
        self.ai_turns = 0
        self.final_state: dict = {}

    def record_ai(self, content: str) -> None:
        if content:
            self.transcript.append({"role": "ai", "content": content})
            self.ai_turns += 1
            logger.info("[%s] AI: %s", self.scenario_name, content)

    def record_user(self, content: str) -> None:
        self.transcript.append({"role": "user", "content": content})
        logger.info("[%s] USER: %s", self.scenario_name, content)

    def harvest(self, state: dict) -> None:
        """Accumulate metadata events + escalation reasons at each graph pause.

        metadata_events has no reducer in State (each node overwrites it), so
        events must be collected at every pause or they are lost.
        """
        self.final_state = state
        for e in state.get("metadata_events") or []:
            if e not in self.events:
                self.events.append(e)
        candidates = [
            state.get("escalation_reason") or "",
            (state.get("last_agent_signal") or {}).get("escalation_reason") or "",
        ]
        for e in _transfer_events(state.get("metadata_events") or []):
            candidates.append((e.get("data") or {}).get("detail") or "")
        for c in candidates:
            if c and c not in self.reasons:
                self.reasons.append(c)

    def ai_text(self) -> str:
        return "\n".join(t["content"] for t in self.transcript if t["role"] == "ai")

    def last_ai(self) -> str:
        for t in reversed(self.transcript):
            if t["role"] == "ai":
                return t["content"]
        return ""

    def dump(self) -> str:
        lines = [f"  {t['role']:>4}: {t['content']}" for t in self.transcript]
        return "\n".join(lines) or "  <empty transcript>"


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────


async def _drive(scenario: Scenario, recorder: RunRecorder) -> dict:
    """Drive one scenario against the live graph. Returns the final state."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from agent.app_graph import build_graph

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["live_e2e", scenario.flow, scenario.name],
        "metadata": {"flow": scenario.flow, "scenario_tag": scenario.name},
    }

    state = await graph.ainvoke({}, config=config)
    turn_idx = 0  # index of the NEXT scripted user turn

    for _ in range(scenario.expect.max_turns):
        recorder.harvest(state)

        if state.get("next_node") in _TERMINAL and not state.get("is_interrupt"):
            break

        if state.get("is_interrupt"):
            ai_msg = extract_last_ai_message(state.get("messages", []))
            if not ai_msg:
                state = await graph.ainvoke(Command(resume=""), config=config)
                continue
            recorder.record_ai(ai_msg)

            expectation = scenario.turn_expectations.get(turn_idx)
            if expectation:
                if expectation.ai_contains and not any(_matches(p, ai_msg) for p in expectation.ai_contains):
                    raise ScenarioFailure(
                        f"Turn {turn_idx}: AI prompt matched none of "
                        f"{expectation.ai_contains!r}.\nAI said: {ai_msg!r}\n"
                        f"Transcript:\n{recorder.dump()}"
                    )
                if expectation.slot_awaiting is not None and (
                    state.get("awaiting_slot") != expectation.slot_awaiting
                ):
                    raise ScenarioFailure(
                        f"Turn {turn_idx}: awaiting_slot="
                        f"{state.get('awaiting_slot')!r}, expected "
                        f"{expectation.slot_awaiting!r}.\nTranscript:\n{recorder.dump()}"
                    )

            if turn_idx >= len(scenario.user_turns):
                raise ScenarioFailure(
                    f"User script exhausted after {turn_idx} turns but the graph "
                    f"is still asking questions — off-script agent behavior.\n"
                    f"Last AI message: {ai_msg!r}\nTranscript:\n{recorder.dump()}"
                )
            user = scenario.user_turns[turn_idx]
            turn_idx += 1
            recorder.record_user(user)
            state = await graph.ainvoke(Command(resume=user), config=config)
        else:
            state = await graph.ainvoke(Command(resume=""), config=config)
    else:
        recorder.harvest(state)
        raise ScenarioFailure(
            f"max_turns={scenario.expect.max_turns} exceeded without reaching END.\n"
            f"Transcript:\n{recorder.dump()}"
        )

    recorder.harvest(state)
    # Hard-END paths (escalation, out-of-scope, phone-not-confirmed) deliver
    # their final AI message without an interrupt — record it now.
    final_ai = extract_last_ai_message(state.get("messages", []))
    if final_ai and final_ai != recorder.last_ai():
        recorder.record_ai(final_ai)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Assertions
# ──────────────────────────────────────────────────────────────────────────────


def _check_value(key: str, expected: Any, actual: Any) -> Optional[str]:
    if callable(expected):
        try:
            ok = expected(actual)
        except Exception as exc:  # predicate itself blew up
            return f"final_state[{key!r}]: predicate raised {exc!r} on value {actual!r}"
        if not ok:
            name = getattr(expected, "__name__", repr(expected))
            return f"final_state[{key!r}]={actual!r} failed predicate {name}"
        return None
    if actual != expected:
        return f"final_state[{key!r}]={actual!r}, expected {expected!r}"
    return None


def _check_transcript_count(exp: Expected, recorder: RunRecorder) -> list[str]:
    """Assert each transcript_count regex matches exactly N AI lines."""
    ai_lines = [t["content"] for t in recorder.transcript if t["role"] == "ai"]
    out: list[str] = []
    for pattern, expected_n in exp.transcript_count.items():
        actual_n = sum(1 for line in ai_lines if _matches(pattern, line))
        if actual_n != expected_n:
            out.append(f"transcript: {actual_n} AI line(s) match {pattern!r}, expected {expected_n}")
    return out


def evaluate(scenario: Scenario, recorder: RunRecorder, reached_end: bool) -> list[str]:
    """Evaluate Expected against the recorded run. Returns failure strings."""

    exp = scenario.expect
    state = recorder.final_state
    failures: list[str] = []

    if reached_end != exp.completed:
        failures.append(
            f"completed={reached_end} (next_node={state.get('next_node')!r}), "
            f"expected completed={exp.completed}"
        )

    transfers = _transfer_events(recorder.events)
    escalated = bool(transfers) or bool(state.get("escalation_reference_number"))
    if escalated != exp.escalated:
        failures.append(
            f"escalated={escalated} (transfer_events={len(transfers)}, "
            f"escalation_reference_number={state.get('escalation_reference_number')!r}), "
            f"expected escalated={exp.escalated}"
        )

    if exp.transfer_event and not transfers:
        failures.append("expected an AgentCallTransfer metadata event — none observed")
    if not exp.transfer_event and not exp.escalated and transfers:
        failures.append(f"unexpected AgentCallTransfer event(s): {transfers!r}")
    if exp.transfer_initiator and transfers:
        initiators = {(t.get("data") or {}).get("transferInitiator") for t in transfers}
        if exp.transfer_initiator not in initiators:
            failures.append(f"transferInitiator(s)={initiators!r}, expected {exp.transfer_initiator!r}")

    all_reasons = " | ".join(recorder.reasons)
    if exp.escalation_reason_contains:
        if exp.escalation_reason_contains.lower() not in all_reasons.lower():
            failures.append(
                f"escalation reason(s) {recorder.reasons!r} do not contain {exp.escalation_reason_contains!r}"
            )
    if exp.escalation_reason_regex:
        if not _matches(exp.escalation_reason_regex, all_reasons):
            failures.append(
                f"escalation reason(s) {recorder.reasons!r} do not match regex "
                f"{exp.escalation_reason_regex!r}"
            )

    for key, expected in exp.final_state.items():
        if err := _check_value(key, expected, state.get(key)):
            failures.append(err)

    if exp.final_is_interrupt is not None:
        actual = bool(state.get("is_interrupt"))
        if actual != exp.final_is_interrupt:
            failures.append(f"is_interrupt={actual}, expected {exp.final_is_interrupt}")

    last_ai = recorder.last_ai()
    for pattern in exp.last_ai_contains:
        if not _matches(pattern, last_ai):
            failures.append(f"last AI message {last_ai!r} does not match {pattern!r}")

    ai_text = recorder.ai_text()
    for pattern in exp.transcript_contains:
        if not _matches(pattern, ai_text):
            failures.append(f"no AI transcript line matches {pattern!r}")

    failures.extend(_check_transcript_count(exp, recorder))

    # next_node END double-check for completed scenarios
    if exp.completed and state.get("next_node") not in _TERMINAL:
        failures.append(f"final next_node={state.get('next_node')!r}, expected END")

    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Result persistence
# ──────────────────────────────────────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def save_result(
    scenario: Scenario,
    recorder: RunRecorder,
    failures: list[str],
    duration_s: float,
    results_dir: Path,
    attempt: int,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{scenario.name}_{ts}.json"
    state_snapshot = {
        k: (_simplify_messages(v) if k == "messages" else _json_safe(v))
        for k, v in (recorder.final_state or {}).items()
    }
    payload = {
        "scenario": scenario.name,
        "flow": scenario.flow,
        "attempt": attempt,
        "passed": not failures,
        "failures": failures,
        "duration_s": round(duration_s, 2),
        "ai_turns": recorder.ai_turns,
        "escalation_reasons": recorder.reasons,
        "metadata_events": recorder.events,
        "transcript": recorder.transcript,
        "final_state": state_snapshot,
        "notes": scenario.notes,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


async def _run_once(
    scenario: Scenario, results_dir: Path, attempt: int
) -> tuple[list[str], RunRecorder, float]:
    recorder = RunRecorder(scenario.name)
    start = time.monotonic()
    failures: list[str] = []
    try:
        state = await asyncio.wait_for(_drive(scenario, recorder), timeout=scenario.timeout_s)
        reached_end = state.get("next_node") in _TERMINAL
        failures = evaluate(scenario, recorder, reached_end)
    except ScenarioFailure as exc:
        failures = [str(exc)]
    except asyncio.TimeoutError:
        failures = [
            f"scenario wall-clock timeout after {scenario.timeout_s}s.\nTranscript so far:\n{recorder.dump()}"
        ]
    except Exception as exc:  # surface unexpected harness/graph errors verbatim
        logger.exception("[%s] unexpected error", scenario.name)
        failures = [f"unexpected error: {exc!r}\nTranscript:\n{recorder.dump()}"]

    if not failures:
        for check in scenario.post_checks:
            try:
                err = await check(recorder.final_state)
            except Exception as exc:
                err = f"post-check {getattr(check, '__name__', check)!r} raised {exc!r}"
            if err:
                failures.append(err)

    duration = time.monotonic() - start
    save_result(scenario, recorder, failures, duration, results_dir, attempt)
    return failures, recorder, duration


async def run_scenario(scenario: Scenario, results_dir: Path) -> ScenarioResult:
    """Run a scenario with its retry budget. Each attempt gets a fresh graph,
    MemorySaver and thread_id."""
    total_duration = 0.0
    attempt = 0
    last_failures: list[str] = []
    last_recorder: RunRecorder | None = None

    while attempt <= scenario.retries:
        attempt += 1
        logger.info("[%s] starting attempt %d/%d", scenario.name, attempt, scenario.retries + 1)
        failures, recorder, duration = await _run_once(scenario, results_dir, attempt)
        total_duration += duration
        last_failures, last_recorder = failures, recorder
        if not failures:
            return ScenarioResult(
                name=scenario.name,
                passed=True,
                turns=recorder.ai_turns,
                duration_s=total_duration,
                escalation_reason=" | ".join(recorder.reasons),
                flaky=attempt > 1,
                attempts=attempt,
            )
        logger.warning("[%s] attempt %d failed:\n%s", scenario.name, attempt, "\n".join(failures))

    return ScenarioResult(
        name=scenario.name,
        passed=False,
        failures=last_failures,
        turns=last_recorder.ai_turns if last_recorder else 0,
        duration_s=total_duration,
        escalation_reason=" | ".join(last_recorder.reasons) if last_recorder else "",
        attempts=attempt,
    )
