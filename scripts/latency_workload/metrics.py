"""
metrics.py — Latency benchmark metrics.

Computes statistics over three distinct latency series:

  1. total_sec   — full wall-clock time per turn (all graph nodes)
  2. llm_sec     — only the time spent in nodes that made LLM calls
  3. llm_calls   — how many LLM calls occurred per turn

All values are in seconds unless noted.  Uses only the standard library.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List


def _percentile(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def compute_metrics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "avg": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    s = sorted(values)
    return {
        "count": len(values),
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
        "avg": statistics.mean(values),
        "std_dev": statistics.pstdev(values),
        "min": s[0],
        "max": s[-1],
    }


def compute_full_report(
    conversation_timings,  # List[ConversationTiming]
    threshold_sec: float = 2.5,
) -> Dict:
    """
    Build a full metrics report from a list of ConversationTiming objects.

    Report structure:
      overall:
        total_sec:  metrics over all per-turn total latencies
        llm_sec:    metrics over all per-turn LLM-only latencies
        llm_calls:  distribution of LLM calls per turn
      per_scenario:
        <scenario_name>:
          total_sec, llm_sec, llm_calls
          per_node: { <node_name>: metrics over all timings for that node }
      violations:
        total_sec_over_threshold: count of turns where total > threshold
        llm_sec_over_threshold:   count of turns where llm_sec > threshold
      meta:
        turn_count, scenario_count, threshold_sec
    """
    from collections import defaultdict

    all_total: List[float] = []
    all_llm: List[float] = []
    all_llm_calls: List[float] = []  # treat as float for percentile calc

    per_scenario_total: Dict[str, List[float]] = defaultdict(list)
    per_scenario_llm: Dict[str, List[float]] = defaultdict(list)
    per_scenario_llm_calls: Dict[str, List[float]] = defaultdict(list)
    per_scenario_per_node: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # Per-node global
    global_per_node: Dict[str, List[float]] = defaultdict(list)

    for ct in conversation_timings:
        scenario = ct.scenario
        for turn in ct.turns:
            all_total.append(turn.total_sec)
            all_llm.append(turn.llm_sec)
            all_llm_calls.append(float(turn.llm_call_count))

            per_scenario_total[scenario].append(turn.total_sec)
            per_scenario_llm[scenario].append(turn.llm_sec)
            per_scenario_llm_calls[scenario].append(float(turn.llm_call_count))

            for node_t in turn.nodes:
                per_scenario_per_node[scenario][node_t.node].append(node_t.duration_sec)
                global_per_node[node_t.node].append(node_t.duration_sec)

    violations_total = sum(1 for v in all_total if v > threshold_sec)
    violations_llm = sum(1 for v in all_llm if v > threshold_sec)

    per_scenario_report = {}
    for scenario in per_scenario_total:
        per_scenario_report[scenario] = {
            "total_sec": compute_metrics(per_scenario_total[scenario]),
            "llm_sec": compute_metrics(per_scenario_llm[scenario]),
            "llm_calls_per_turn": compute_metrics(per_scenario_llm_calls[scenario]),
            "per_node": {
                node: compute_metrics(vals) for node, vals in per_scenario_per_node[scenario].items()
            },
        }

    return {
        "overall": {
            "total_sec": compute_metrics(all_total),
            "llm_sec": compute_metrics(all_llm),
            "llm_calls_per_turn": compute_metrics(all_llm_calls),
            "per_node": {node: compute_metrics(vals) for node, vals in global_per_node.items()},
        },
        "per_scenario": per_scenario_report,
        "violations": {
            "total_sec_over_threshold": violations_total,
            "llm_sec_over_threshold": violations_llm,
            "threshold_sec": threshold_sec,
        },
        "meta": {
            "turn_count": len(all_total),
            "scenario_count": len(per_scenario_total),
            "threshold_sec": threshold_sec,
        },
    }
