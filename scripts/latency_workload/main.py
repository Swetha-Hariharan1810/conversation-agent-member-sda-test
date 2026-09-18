"""
main.py — LangGraph latency benchmark CLI.

Key differences from the original:
  - Times individual graph NODES, not whole ainvoke() calls.
  - Separates LLM-call latency from pure-Python routing latency.
  - Reports per-node breakdown so you can see exactly which agent / which
    LLM call is slow.
  - SLA threshold applies to LLM-only latency (the controllable cost),
    not total wall-clock which includes un-attributable framework overhead.
  - Provides per-scenario AND per-node tables in the markdown output.

Usage:
    python -m scripts.latency_workload.main \\
        --iterations 5 \\
        --threshold-sec 2.5 \\
        --scenarios-path scripts/latency_workload/scenarios
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List

from agent.logger import get_logger
from scripts.latency_workload.data_loader import load_scenarios
from scripts.latency_workload.metrics import compute_full_report
from scripts.latency_workload.runner import (
    ConversationTiming,
    _run_one_conversation_async,
)

logger = get_logger(__name__)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_config(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph latency benchmark")
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.getenv("ITERATIONS", "1")),
    )
    parser.add_argument(
        "--threshold-sec",
        type=float,
        default=float(os.getenv("THRESHOLD_SEC", "2.5")),
    )
    parser.add_argument("--output", default="bench-metrics.json")
    parser.add_argument(
        "--scenarios-path",
        default=str(Path(__file__).parent / "scenarios"),
    )
    # Whether to fail on LLM-only latency or total latency.
    # Default: llm_sec (controllable) rather than total (includes framework).
    parser.add_argument(
        "--sla-metric",
        choices=["llm_sec", "total_sec"],
        default="llm_sec",
        help="Which latency series to apply the SLA threshold to.",
    )
    return parser.parse_args(argv)


def _write_github_outputs(report: dict, json_path: str) -> None:
    """Write step outputs for GitHub Actions CI summary."""
    gh_output = os.getenv("GITHUB_OUTPUT")
    if not gh_output:
        return
    overall = report["overall"]
    total = overall["total_sec"]
    llm = overall["llm_sec"]
    try:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"total_p50={total['p50']:.6f}\n")
            fh.write(f"total_p95={total['p95']:.6f}\n")
            fh.write(f"total_avg={total['avg']:.6f}\n")
            fh.write(f"total_std_dev={total['std_dev']:.6f}\n")
            fh.write(f"llm_p50={llm['p50']:.6f}\n")
            fh.write(f"llm_p95={llm['p95']:.6f}\n")
            fh.write(f"llm_avg={llm['avg']:.6f}\n")
            fh.write(f"llm_std_dev={llm['std_dev']:.6f}\n")
            fh.write(f"turn_count={total['count']}\n")
            fh.write(f"json_path={json_path}\n")
    except Exception as exc:
        logger.warning("Could not write GitHub outputs: %s", exc)


def _print_markdown(report: dict, args: argparse.Namespace) -> None:
    overall = report["overall"]
    total = overall["total_sec"]
    llm = overall["llm_sec"]
    calls = overall["llm_calls_per_turn"]
    viols = report["violations"]
    meta = report["meta"]

    print("\n## LangGraph Latency Bench (PCP per-step)")
    print(
        f"Iterations: {args.iterations} | "
        f"Turns: {meta['turn_count']} | "
        f"Threshold: {args.threshold_sec:.3f}s ({args.sla_metric})"
    )

    print("\n### Overall — Total Turn Latency (all graph nodes)")
    print("| Metric | Value (s) |\n|---|---:|")
    for k in ("p50", "p95", "p99", "avg", "std_dev", "min", "max"):
        print(f"| {k} | {total[k]:.3f} |")

    print("\n### Overall — LLM-Only Latency (nodes that called an LLM)")
    print("| Metric | Value (s) |\n|---|---:|")
    for k in ("p50", "p95", "p99", "avg", "std_dev", "min", "max"):
        print(f"| {k} | {llm[k]:.3f} |")

    print("\n### LLM Calls Per Turn")
    print("| Metric | Count |\n|---|---:|")
    for k in ("p50", "p95", "avg", "min", "max"):
        print(f"| {k} | {calls[k]:.1f} |")

    print("\n### Per-Node Breakdown (global avg ± std)")
    print("| Node | avg (s) | p95 (s) | count |\n|---|---:|---:|---:|")
    per_node = overall.get("per_node", {})
    for node, m in sorted(per_node.items(), key=lambda x: -x[1]["avg"]):
        print(f"| {node} | {m['avg']:.3f} | {m['p95']:.3f} | {m['count']} |")

    print("\n### SLA Violations")
    print("| Metric | Violations | Total Turns |")
    print("|---|---:|---:|")
    print(
        f"| total_sec > {args.threshold_sec:.1f}s | "
        f"{viols['total_sec_over_threshold']} | {meta['turn_count']} |"
    )
    print(
        f"| llm_sec > {args.threshold_sec:.1f}s | {viols['llm_sec_over_threshold']} | {meta['turn_count']} |"
    )

    print("\n### Per-Scenario Summary")
    print("| Scenario | total_p50 | llm_p50 | llm_calls_avg |\n|---|---:|---:|---:|")
    for scenario, sr in report["per_scenario"].items():
        total_metrics = sr["total_sec"]
        llm_metrics = sr["llm_sec"]
        call_metrics = sr["llm_calls_per_turn"]

        print(
            f"| {scenario} | "
            f"{total_metrics['p50']:.3f} | "
            f"{llm_metrics['p50']:.3f} | "
            f"{call_metrics['avg']:.1f} |"
        )


async def main_async(argv=None) -> int:
    args = _get_config(argv)

    output_dir_env = os.getenv("BENCH_OUTPUT_DIR")
    if output_dir_env:
        _ensure_dir(output_dir_env)
        args.output = str(Path(output_dir_env) / Path(args.output).name)

    scenarios = load_scenarios(args.scenarios_path)
    if not scenarios:
        logger.error("No scenario files found in %s", args.scenarios_path)
        return 1

    logger.info(
        "Loaded %d scenario(s): %s",
        len(scenarios),
        ", ".join(scenarios.keys()),
    )
    logger.info("Running %d iteration(s) per scenario", args.iterations)

    t0 = time.perf_counter()
    all_timings: List[ConversationTiming] = []

    for i in range(args.iterations):
        for scenario_name, steps in scenarios.items():
            tag = f"{scenario_name}-iter-{i + 1}"
            logger.info("Running %s iteration %d/%d", scenario_name, i + 1, args.iterations)
            ct = await _run_one_conversation_async(
                steps,
                tag=tag,
                scenario=scenario_name,
                iteration=i + 1,
            )
            all_timings.append(ct)

    wall_time = time.perf_counter() - t0
    report = compute_full_report(all_timings, threshold_sec=args.threshold_sec)
    report["meta"]["wall_time_sec"] = round(wall_time, 3)
    report["meta"]["iterations"] = args.iterations
    report["meta"]["scenarios"] = list(scenarios.keys())

    # Save raw per-conversation timing alongside the summary
    report["raw"] = [ct.to_dict() for ct in all_timings]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", output_path)

    _print_markdown(report, args)
    _write_github_outputs(report, str(output_path))

    # SLA check: apply to whichever metric was requested
    sla_metric = args.sla_metric  # "llm_sec" or "total_sec"
    sla_value = report["overall"][sla_metric]["avg"]
    if sla_value > args.threshold_sec:
        logger.warning(
            "SLA violated: %s avg %.3fs > threshold %.3fs",
            sla_metric,
            sla_value,
            args.threshold_sec,
        )
        return 1

    logger.info(
        "All scenarios passed. %s avg=%.3fs <= threshold=%.3fs. Wall time: %.2fs.",
        sla_metric,
        sla_value,
        args.threshold_sec,
        wall_time,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
