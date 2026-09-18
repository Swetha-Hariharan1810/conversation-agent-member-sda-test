"""
check_transcripts.py — run the turn-level checks over the seed transcripts.

No network, no LLM, no graph. Runs in milliseconds.

    python -m scripts.turn_contract.check_transcripts
    python -m scripts.turn_contract.check_transcripts --verbose

Exit code 0 when every transcript matches its EXPECTED_VIOLATIONS entry, 1
otherwise — so it can gate CI. A fix that resolves a known bug will fail here
until EXPECTED_VIOLATIONS is updated, which is the point: the expectations file
is the record of what is still broken.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.turn_contract.checks import Transcript, Violation, run_all_checks  # noqa: E402
from scripts.turn_contract.seed_transcripts import (  # noqa: E402
    ALL_TRANSCRIPTS,
    EXPECTED_VIOLATIONS,
)

_RESET, _BOLD, _RED, _GREEN, _YELLOW, _DIM = (
    "\033[0m",
    "\033[1m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[2m",
)


def _fmt(v: Violation, transcript: Transcript, verbose: bool) -> str:
    colour = _RED if v.severity == "error" else _YELLOW
    turn = transcript.turns[v.turn_index]
    head = f"  {colour}{v.severity:7}{_RESET} {_BOLD}{v.check}{_RESET}  turn {v.turn_index} ({turn.role})"
    lines = [head, f"          {v.detail}"]
    if verbose and v.excerpt:
        lines.append(f"          {_DIM}{v.excerpt}{_RESET}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn-level contract checks over seed transcripts.")
    ap.add_argument("--verbose", "-v", action="store_true", help="show the offending text")
    ap.add_argument("--check", action="append", help="run only these checks (repeatable)")
    args = ap.parse_args()

    total_violations = 0
    mismatches: list[str] = []

    for transcript in ALL_TRANSCRIPTS:
        violations = run_all_checks(transcript, only=args.check)
        total_violations += len(violations)
        found = {v.check for v in violations}
        expected = EXPECTED_VIOLATIONS.get(transcript.name, set())
        if args.check:
            expected &= set(args.check)

        ok = found == expected
        mark = f"{_GREEN}as expected{_RESET}" if ok else f"{_RED}MISMATCH{_RESET}"
        print(f"\n{_BOLD}{transcript.name}{_RESET}  [{mark}]")
        if transcript.note:
            print(f"  {_DIM}{transcript.note}{_RESET}")

        for v in violations:
            print(_fmt(v, transcript, args.verbose))
        if not violations:
            print(f"  {_DIM}(no violations){_RESET}")

        if not ok:
            missing = expected - found
            unexpected = found - expected
            if missing:
                mismatches.append(f"{transcript.name}: expected but NOT detected — {sorted(missing)}")
            if unexpected:
                mismatches.append(f"{transcript.name}: detected but NOT expected — {sorted(unexpected)}")

    print(f"\n{_BOLD}{'─' * 68}{_RESET}")
    print(f"{len(ALL_TRANSCRIPTS)} transcripts, {total_violations} violations")

    if mismatches:
        print(f"\n{_RED}{_BOLD}Expectation mismatches:{_RESET}")
        for m in mismatches:
            print(f"  {_RED}✗{_RESET} {m}")
        return 1

    print(f"{_GREEN}All transcripts match their expectations.{_RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
