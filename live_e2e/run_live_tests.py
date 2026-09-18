"""
run_live_tests.py — Canonical CLI entry point for the live E2E suite.

Usage (from the repo root):
    python -m tests.live_e2e.run_live_tests
    python -m tests.live_e2e.run_live_tests --only pcp_happy_path_fax,claim_happy_path
    python -m tests.live_e2e.run_live_tests --only verification_dob_only_mismatch --repeat 10
    python -m tests.live_e2e.run_live_tests --skip-mutating
    python -m tests.live_e2e.run_live_tests --results-dir /tmp/live_results
    python -m tests.live_e2e.run_live_tests --list

Scenarios run SEQUENTIALLY — they share Salesforce fixture data and must not
be parallelized. Exits non-zero on any failure (and 2 on preflight failure).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import Scenario, ScenarioResult, run_scenario
from tests.live_e2e.preflight import PreflightError, restore_contacts, run_preflight
from tests.live_e2e.scenarios import SCENARIOS, SCENARIOS_BY_NAME

logger = logging.getLogger("live_e2e")

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten chatty third-party loggers; keep agent + harness at INFO.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _select(only: str | None, skip_mutating: bool) -> list[Scenario]:
    if only:
        names = [n.strip() for n in only.split(",") if n.strip()]
        unknown = [n for n in names if n not in SCENARIOS_BY_NAME]
        if unknown:
            raise SystemExit(
                f"Unknown scenario name(s): {', '.join(unknown)}\nAvailable: {', '.join(SCENARIOS_BY_NAME)}"
            )
        selected = [SCENARIOS_BY_NAME[n] for n in names]
    else:
        selected = list(SCENARIOS)
    if skip_mutating:
        skipped = [s.name for s in selected if s.mutating]
        if skipped:
            logger.info("--skip-mutating: excluding %s", ", ".join(skipped))
        selected = [s for s in selected if not s.mutating]
    return selected


def _print_summary(results: list[ScenarioResult]) -> None:
    name_w = max((len(r.name) for r in results), default=8) + 2
    header = f"{'scenario':<{name_w}}{'result':<10}{'turns':<7}{'duration':<10}escalation_reason"
    print("\n" + "=" * len(header))
    print("LIVE E2E SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed and r.flaky:
            status = "PASS*"  # passed on retry — flaky
        print(f"{r.name:<{name_w}}{status:<10}{r.turns:<7}{r.duration_s:<10.1f}{r.escalation_reason or '-'}")
    print("-" * len(header))
    passed = sum(1 for r in results if r.passed)
    flaky = sum(1 for r in results if r.flaky)
    print(f"{passed}/{len(results)} passed" + (f" ({flaky} flaky — passed on retry)" if flaky else ""))
    for r in results:
        if not r.passed:
            print(f"\n--- FAILURES: {r.name} ---")
            for f in r.failures:
                print(f"  * {f}")


async def _run_all(scenarios: list[Scenario], results_dir: Path, snapshot) -> list[ScenarioResult]:
    """Run the selected scenarios once, sequentially."""
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        logger.info("=" * 70)
        logger.info(
            "SCENARIO: %s (%s)%s", scenario.name, scenario.flow, " [mutating]" if scenario.mutating else ""
        )
        logger.info("=" * 70)
        try:
            result = await run_scenario(scenario, results_dir)
        finally:
            if scenario.mutating:
                # Restore snapshotted SF contact fields even if the scenario failed.
                await restore_contacts(snapshot)
        results.append(result)
    return results


def _print_stress_aggregate(per_iter: list[list[ScenarioResult]]) -> None:
    """Per-scenario pass count across N iterations (stress / flakiness view)."""
    n = len(per_iter)
    names = [r.name for r in per_iter[0]]
    name_w = max((len(x) for x in names), default=8) + 2
    header = f"{'scenario':<{name_w}}{'passed':<10}{'pass rate'}"
    print("\n" + "=" * len(header))
    print(f"STRESS AGGREGATE — {n} iterations")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    all_green = True
    for name in names:
        passes = sum(1 for it in per_iter for r in it if r.name == name and r.passed)
        rate = passes / n
        if passes != n:
            all_green = False
        flag = "" if passes == n else "  <-- FLAKY/FAIL"
        print(f"{name:<{name_w}}{f'{passes}/{n}':<10}{rate:>6.0%}{flag}")
    print("-" * len(header))
    print("ALL GREEN across every iteration" if all_green else "NOT stable — see flagged scenarios above")


async def _amain(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    scenarios = _select(args.only, args.skip_mutating)
    if not scenarios:
        print("Nothing to run.")
        return 0

    repeat = max(1, args.repeat)
    print(f"Running {len(scenarios)} scenario(s) sequentially against LIVE services.")
    if repeat > 1:
        print(f"Stress mode: {repeat} iterations.")
    print("This makes real Azure OpenAI and Salesforce calls — expect cost and latency.\n")

    try:
        snapshot = await run_preflight(warm=True)
    except PreflightError as exc:
        print(f"\nPREFLIGHT FAILED — no scenarios were run.\n\n{exc}", file=sys.stderr)
        return 2

    per_iter: list[list[ScenarioResult]] = []
    overall_ok = True
    for i in range(1, repeat + 1):
        if repeat > 1:
            print(f"\n########################  ITERATION {i}/{repeat}  ########################")
        results = await _run_all(scenarios, results_dir, snapshot)
        _print_summary(results)
        per_iter.append(results)
        if not all(r.passed for r in results):
            overall_ok = False

    if repeat > 1:
        _print_stress_aggregate(per_iter)

    return 0 if overall_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tests.live_e2e.run_live_tests",
        description="Live end-to-end conversation tests (real LLM + Salesforce).",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated scenario names to run (default: all).",
    )
    parser.add_argument(
        "--skip-mutating",
        action="store_true",
        help="Exclude scenarios that write to Salesforce "
        "(pcp_zip_update, pcp_zip_inline_update, pcp_fax_update, "
        "pcp_email_update, claim_email_change_on_upload, "
        "email_change_loop_in_notification).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run the selected scenarios N times back-to-back (stress / flakiness "
        "check). Prints a per-iteration summary plus an aggregate pass rate. Default 1.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory for per-scenario JSON results (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List scenario names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for s in SCENARIOS:
            flags = []
            if s.mutating:
                flags.append("mutating")
            if s.retries:
                flags.append(f"retries={s.retries}")
            print(f"{s.name:<40} flow={s.flow:<6} {' '.join(flags)}")
        return

    _configure_logging()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
