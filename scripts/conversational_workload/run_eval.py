#!/usr/bin/env python3
"""
run_eval.py — Entry point for the conversational evaluation benchmark.

What the new diagnostics mean
──────────────────────────────
exact_match_rate:
  Fraction of turns where the simulator output and transcript ground truth
  agreed closely enough to skip the LLM judge.  For a correct happy-path
  scenario this should be HIGH (> 0.8).  If it is low on the happy path,
  the agent is rephrasing its questions significantly — the simulator is
  still responding correctly but the transcript keywords are not matching.

llm_judge_calls:
  Number of turns where the LLM judge fired.  For clarification/correction
  scenarios some judge calls are EXPECTED and CORRECT — those are the
  turns where the simulator deliberately gives a different (scenario-driven)
  response.  A high count on pcp_happy_path is a warning that the agent
  is deviating from the expected flow.

pass_rate:
  Fraction of turns with score >= 0.8.  This is the primary quality signal.
  For a working agent with a correct simulator, this should be >= 0.85 on
  all scenarios.

average_final_score:
  Mean per-turn score across all turns.  This is what the CI threshold
  gate checks.  Should be >= SCORE_THRESHOLD (default 0.75) to pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import aiofiles

from scripts.conversational_workload.runner import run_evaluation_async

ENTITY_DATA = {
    "first_name": "Emily",
    "last_name": "Carter",
    "member_id": "M907503",
    "date_of_birth": "1988-04-12",
    "subscriber_type": "myself",
    "provider_type": "Primary Care Physician",
    "zip_code": "12139",
    "fax_number": "6175554199",
}

SCENARIOS = [
    "pcp_happy_path",
    "pcp_clarification_zip",
    # "pcp_correction_first_name",
    "pcp_correction_member_id",
    "pcp_clarification_fax",
]

CLAIM_ENTITY_DATA = {
    "first_name": "James",
    "last_name": "Wilson",
    "member_id": "M310188",
    "date_of_birth": "1977-07-30",
    "phone_number": "5125556101",
    "reference_number": "42695817",
    "email": "james.wilson@gmail.com",
}

CLAIM_SCENARIOS = [
    "claim_adjustment_happy_path",
    "claim_adjustment_no_proceed",
    "claim_adjustment_upload_only",
    "claim_adjustment_guide_only",
]

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.75"))


def _aggregate_run(run_data: dict) -> dict:
    turns = run_data.get("turns", [])
    if not turns:
        return {
            "exact_match_rate": 0.0,
            "llm_judge_calls": 0,
            "pass_rate": 0.0,
        }

    exact = sum(
        1
        for t in turns
        if t.get("scores", {}).get("judge_method") in ("exact_match", "substring_match", "no_ground_truth")
    )
    llm_calls = sum(1 for t in turns if t.get("scores", {}).get("judge_method") == "llm")
    passed = sum(1 for t in turns if t.get("scores", {}).get("overall", 0.0) >= 0.8)
    return {
        "exact_match_rate": round(exact / len(turns), 3),
        "llm_judge_calls": llm_calls,
        "pass_rate": round(passed / len(turns), 3),
    }


async def main_async() -> int:
    num_runs = int(os.getenv("ITERATIONS", "1"))

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = Path(
        os.getenv(
            "RESULTS_DIR",
            str(repo_root / "scripts/conversational_workload/results"),
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = os.getenv("REPORT_FILE", "conversation_eval_report.json")

    all_scenario_tags = SCENARIOS + CLAIM_SCENARIOS
    print("\n" + "=" * 55)
    print("Conversational Evaluation Benchmark")
    print(f"Scenarios  : {', '.join(all_scenario_tags)}")
    print(f"Runs/scen  : {num_runs}")
    print(f"Threshold  : {SCORE_THRESHOLD}")
    print("=" * 55 + "\n")

    overall_summary: dict = {}

    scenario_batches = [
        (SCENARIOS, "pcp", ENTITY_DATA),
        (CLAIM_SCENARIOS, "claim", CLAIM_ENTITY_DATA),
    ]

    for scenarios, flow, entity_data in scenario_batches:
        for scenario_tag in scenarios:
            all_reports = []
            for run_idx in range(num_runs):
                print(f"  [{scenario_tag}] run {run_idx + 1}/{num_runs}")
                report = await run_evaluation_async(
                    entity_data=entity_data,
                    flow=flow,
                    scenario_tag=scenario_tag,
                )
                run_dict = report.model_dump()
                diag = _aggregate_run(run_dict)
                run_dict.update(diag)
                print(
                    f"    score={report.final_score:.2f}  "
                    f"turns={len(report.turns)}  "
                    f"exact_match={diag['exact_match_rate']:.0%}  "
                    f"llm_judge={diag['llm_judge_calls']}  "
                    f"pass_rate={diag['pass_rate']:.0%}  "
                    f"completed={report.completed}"
                )
                all_reports.append(run_dict)

            avg_score = round(sum(r["final_score"] for r in all_reports) / num_runs, 2)
            avg_exact = round(sum(r["exact_match_rate"] for r in all_reports) / num_runs, 3)
            total_llm = sum(r["llm_judge_calls"] for r in all_reports)
            avg_pass = round(sum(r["pass_rate"] for r in all_reports) / num_runs, 3)

            overall_summary[scenario_tag] = {
                "total_runs": num_runs,
                "average_final_score": avg_score,
                "average_exact_match_rate": avg_exact,
                "total_llm_judge_calls": total_llm,
                "average_pass_rate": avg_pass,
                "runs": all_reports,
            }
            print(
                f"  → {scenario_tag}: score={avg_score:.2f}  exact={avg_exact:.0%}  llm_judge={total_llm}\n"
            )

    summary_path = results_dir / report_file
    async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(overall_summary, indent=2))

    print(f"Summary written to {summary_path}")

    # Diagnostic warning: happy path with low exact match means agent drift
    # happy = overall_summary.get("pcp_happy_path", {})
    # if happy.get("average_exact_match_rate", 1.0) < 0.6:
    #     print(
    #         "\n⚠️  WARNING: pcp_happy_path exact_match_rate is "
    #         f"{happy['average_exact_match_rate']:.0%}.  "
    #         "The agent is significantly rephrasing its questions relative "
    #         "to the reference transcript.  Review agent prompt changes."
    #     )

    # Scores must be above threshold
    failures = [
        (tag, data["average_final_score"])
        for tag, data in overall_summary.items()
        if data["average_final_score"] < SCORE_THRESHOLD
    ]
    if failures:
        print(f"\nFAILED (below threshold {SCORE_THRESHOLD:.2f}):")
        for tag, score in failures:
            print(f"  {tag}: {score:.2f}")
        return 1

    print(f"\nAll scenarios passed (threshold={SCORE_THRESHOLD}).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
