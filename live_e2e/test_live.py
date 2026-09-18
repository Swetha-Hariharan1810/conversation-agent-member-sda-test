"""
test_live.py — Thin pytest wrapper over the live E2E scenarios.

    pytest -m live tests/live_e2e/test_live.py

The `live` marker is excluded by default (see pyproject addopts), so normal
`pytest` runs never hit live services. The canonical entry point remains
`python -m tests.live_e2e.run_live_tests` — use that for the summary table
and CLI filters. Scenarios share Salesforce data: never run with xdist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import run_scenario
from tests.live_e2e.preflight import PreflightError, restore_contacts, run_preflight
from tests.live_e2e.scenarios import SCENARIOS

pytestmark = pytest.mark.live

RESULTS_DIR = Path(__file__).parent / "results"


@pytest.fixture(scope="session")
async def fixture_snapshot():
    """Preflight once per session; yields the Salesforce contact snapshot."""
    try:
        snapshot = await run_preflight(warm=True)
    except PreflightError as exc:
        pytest.fail(f"Preflight failed — aborting live run:\n{exc}", pytrace=False)
    yield snapshot
    # Final safety net: restore fixture contacts at end of session.
    await restore_contacts(snapshot)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
async def test_live_scenario(scenario, fixture_snapshot):
    try:
        result = await run_scenario(scenario, RESULTS_DIR)
    finally:
        if scenario.mutating:
            await restore_contacts(fixture_snapshot)
    assert result.passed, f"{scenario.name} failed after {result.attempts} attempt(s):\n" + "\n".join(
        f"  * {f}" for f in result.failures
    )
