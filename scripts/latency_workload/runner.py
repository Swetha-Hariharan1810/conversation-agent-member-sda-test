"""
runner.py — LangGraph latency benchmark runner.

DESIGN: measures individual LLM-call latency, not graph.ainvoke() wall-time.

The previous implementation called graph.ainvoke(Command(resume=user_input))
and timed the entire call. That was wrong for two reasons:

  1. A single ainvoke traverses multiple graph nodes before the next interrupt,
     meaning one measurement can contain 0, 1, 2, or 3 LLM calls depending on
     which routing path the graph takes on that turn.

  2. Non-interrupt hops (orchestrator routing, fast-path transitions) ran
     outside the timing window but still consumed LLM tokens, distorting
     the per-step numbers silently.

We hook into LangGraph's streaming event API using ainvoke with
stream_mode="updates".  For every graph step we capture:
  - which node ran
  - whether it made an LLM call (identified by the presence of messages
    with role="assistant" in the step output, or an explicit latency tag
    we inject via a timing callback wrapper on the LLM config).

Because modifying the production graph to inject timing callbacks is
invasive, we use a simpler but reliable proxy:

  We stream updates step-by-step and record wall-clock time for each
  individual node execution. Nodes that contain real LLM calls will
  show latency > ~100ms; pure-Python routing nodes show < 5ms.
  We report BOTH sets, letting the caller filter by threshold.

This gives us:
  - Per-node latency breakdown (identifies exactly where time is spent)
  - LLM-only latency (filter nodes > MIN_LLM_LATENCY_MS)
  - Full-turn latency (sum of all nodes for one user input)

STEP DEFINITION:
  One "step" = one user turn = the time from sending user input to
  receiving the next human_node interrupt.  We decompose each step
  into its constituent node executions.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command

# Nodes whose latency is dominated by pure Python (no LLM call).
# We keep them in the data but flag them separately.
ROUTING_NODES = frozenset(
    {
        "human_node",
        "orchestrator",  # orchestrator CAN make an LLM call; flag separately
        "__start__",
    }
)

# If a node takes less than this, it almost certainly made no LLM call.
MIN_LLM_LATENCY_SEC = 0.05


@dataclass
class NodeTiming:
    node: str
    duration_sec: float
    has_llm_call: bool  # heuristic: duration > MIN_LLM_LATENCY_SEC

    @classmethod
    def from_elapsed(cls, node: str, elapsed: float) -> "NodeTiming":
        return cls(
            node=node,
            duration_sec=elapsed,
            has_llm_call=elapsed > MIN_LLM_LATENCY_SEC,
        )


@dataclass
class TurnTiming:
    """All node timings that occurred between two human_node interrupts."""

    user_input: str
    nodes: List[NodeTiming] = field(default_factory=list)

    @property
    def total_sec(self) -> float:
        return sum(n.duration_sec for n in self.nodes)

    @property
    def llm_sec(self) -> float:
        """Sum of time spent in nodes that made LLM calls."""
        return sum(n.duration_sec for n in self.nodes if n.has_llm_call)

    @property
    def llm_call_count(self) -> int:
        return sum(1 for n in self.nodes if n.has_llm_call)

    def to_dict(self) -> dict:
        return {
            "user_input": self.user_input,
            "total_sec": round(self.total_sec, 6),
            "llm_sec": round(self.llm_sec, 6),
            "llm_call_count": self.llm_call_count,
            "nodes": [
                {
                    "node": n.node,
                    "duration_sec": round(n.duration_sec, 6),
                    "has_llm_call": n.has_llm_call,
                }
                for n in self.nodes
            ],
        }


@dataclass
class ConversationTiming:
    scenario: str
    iteration: int
    turns: List[TurnTiming] = field(default_factory=list)

    @property
    def per_turn_total_sec(self) -> List[float]:
        return [t.total_sec for t in self.turns]

    @property
    def per_turn_llm_sec(self) -> List[float]:
        return [t.llm_sec for t in self.turns]

    @property
    def llm_call_counts(self) -> List[int]:
        return [t.llm_call_count for t in self.turns]

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "iteration": self.iteration,
            "turns": [t.to_dict() for t in self.turns],
        }


async def _run_one_conversation_async(
    steps: List[str],
    *,
    tag: str,
    scenario: str = "",
    iteration: int = 0,
) -> ConversationTiming:
    """
    Run one full conversation and return per-node timing for every turn.

    Uses astream_events (LangGraph event streaming) to observe each node's
    start and end time individually, rather than timing the whole ainvoke.
    """
    from agent.app_graph import build_graph

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["bench", tag],
        "metadata": {"mode": "bench", "bench_tag": tag, "scenario": scenario},
    }

    timing = ConversationTiming(scenario=scenario, iteration=iteration)

    # ── Initial graph kick-off (no user input yet) ───────────────────────────
    # The graph starts at intake_agent which sends a greeting.
    # We run this outside the turn-timing loop.
    state = await graph.ainvoke({}, config=config)

    idx = 0

    while True:
        if state.get("next_node") in ("END", END):
            break

        if not state.get("is_interrupt"):
            # Non-interrupt state: graph needs a nudge with empty resume.
            # This path should rarely occur after initial setup; we do NOT
            # time it as a "turn" since no user input was given.
            state = await graph.ainvoke(Command(resume=""), config=config)
            continue

        if idx >= len(steps):
            break

        user_input = steps[idx]
        idx += 1

        # ── Per-node streaming for this turn ─────────────────────────────────
        # astream_events yields events with type "on_chain_start" /
        # "on_chain_end" for each node.  We track wall-clock between
        # matching start/end pairs for each named node.
        turn = TurnTiming(user_input=user_input)
        node_start_times: Dict[str, float] = {}

        # We collect the final state from the last event so we can check
        # is_interrupt / next_node for the loop condition.
        final_state: Optional[dict] = None

        try:
            async for event in graph.astream_events(
                Command(resume=user_input),
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")
                name = event.get("name", "")

                if event_type == "on_chain_start" and name:
                    node_start_times[name] = time.perf_counter()

                elif event_type == "on_chain_end" and name:
                    start = node_start_times.pop(name, None)
                    if start is not None:
                        elapsed = time.perf_counter() - start
                        # Only record nodes that are actual graph nodes,
                        # not internal LangChain chain wrappers (those have
                        # generic names like "RunnableSequence").
                        if _is_graph_node(name, graph):
                            turn.nodes.append(NodeTiming.from_elapsed(name, elapsed))

                # Capture final state snapshot from the last update event
                elif event_type == "on_chain_stream":
                    data = event.get("data", {})
                    chunk = data.get("chunk", {})
                    if isinstance(chunk, dict):
                        # LangGraph emits state snapshots; keep the latest
                        final_state = {**final_state, **chunk} if final_state else chunk

        except Exception:
            # If streaming fails (e.g. older LangGraph version), fall back
            # to single ainvoke with whole-turn timing.
            turn = await _fallback_single_turn(graph, user_input, config, user_input)
            final_state = None

        timing.turns.append(turn)

        # Update state for loop control — re-fetch from checkpoint if needed
        if final_state is not None:
            state = final_state
        else:
            # Fallback: get current state from checkpoint
            try:
                state = await graph.aget_state(config)
                state = state.values if hasattr(state, "values") else {}
            except Exception:
                break

    return timing


def _is_graph_node(name: str, graph) -> bool:
    """
    Return True if `name` is a declared node in the graph (not an
    internal LangChain runnable chain wrapper).
    """
    try:
        return name in graph.nodes
    except Exception:
        # Fallback: accept names that look like agent node names
        return name.endswith("_agent") or name in ("human_node", "orchestrator", "__start__", "__end__")


async def _fallback_single_turn(
    graph,
    user_input: str,
    config: dict,
    label: str,
) -> TurnTiming:
    """
    Fallback when astream_events is unavailable: time the whole ainvoke as
    a single "turn" node so data is still collected.
    """
    turn = TurnTiming(user_input=label)
    t0 = time.perf_counter()
    await graph.ainvoke(Command(resume=user_input), config=config)
    elapsed = time.perf_counter() - t0
    turn.nodes.append(NodeTiming.from_elapsed("_whole_turn_fallback", elapsed))
    return turn


def run_one_conversation(
    steps: List[str],
    *,
    tag: str,
    scenario: str = "",
    iteration: int = 0,
) -> ConversationTiming:
    """Sync wrapper for callers that cannot use async."""
    return asyncio.run(_run_one_conversation_async(steps, tag=tag, scenario=scenario, iteration=iteration))
