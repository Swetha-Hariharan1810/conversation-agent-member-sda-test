"""
run_extraction.py — run the extraction contract corpus against a live model.

One LLM call per case, concurrent, no graph and no Salesforce. A full pass is
seconds, so this can run on every prompt change.

    python -m scripts.turn_contract.run_extraction
    python -m scripts.turn_contract.run_extraction --model gemini
    python -m scripts.turn_contract.run_extraction --case dob_clean --verbose
    python -m scripts.turn_contract.run_extraction --json out.json

--model azure   the production extraction tier (get_extraction_llm)
--model gemini  the same structured-output contract against Gemini. Useful when
                the Azure host is unreachable, and as a second opinion on
                whether a failure is the prompt or the model. Numbers from the
                two are NOT comparable — a baseline must name its model.

Exit code 0 when every "contract" case passes. "gap" cases are reported but
never fail the run: they are the spec for work not yet done.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.turn_contract.extraction_cases import CASES, ExtractionCase  # noqa: E402

_RESET, _BOLD, _RED, _GREEN, _YELLOW, _DIM, _CYAN = (
    "\033[0m",
    "\033[1m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[2m",
    "\033[36m",
)


@dataclass
class CaseResult:
    id: str
    status: str
    passed: bool
    seconds: float
    mismatches: list[str]
    raw: dict[str, Any]
    error: str = ""
    fixed_by: str = ""
    note: str = ""


def _get_llm(which: str):
    from agent.llm.config import Config, get_extraction_llm

    if which == "azure":
        return get_extraction_llm(), Config.WORKER_DEPLOYMENT
    if which == "gemini":
        from agent.llm.config import get_gemini_llm

        return get_gemini_llm(), Config.LLM_MODEL
    raise SystemExit(f"unknown model {which!r} — use azure or gemini")


def _normalizers() -> dict[str, Any]:
    """slot name → the normalizer the pipeline would apply to it.

    The extraction LLM returns what the caller said ("m nine zero seven five
    zero three"); the slot pipeline normalizes it before validating. A contract
    case asserting 'M907503' is asserting the END state, so the comparison has
    to run the same normalizer — otherwise the corpus fails on turns the real
    system handles correctly, and every one of those false alarms costs trust.
    """
    from agent.slots import normalizers as n

    return {
        "first_name": n.normalize_name,
        "last_name": n.normalize_name,
        "member_id": n.normalize_member_id,
        "dob": n.normalize_dob,
        "ssn": n.normalize_ssn,
        "zip_code": n.normalize_zip_code,
        "phone_number": n.normalize_phone_number,
        "email": n.normalize_email,
        "fax": n.normalize_fax_number,
        "reference_number": n.normalize_reference_number,
        "claim_number": n.normalize_claim_number,
        "provider_type": n.normalize_provider_type,
        "delivery_method": n.normalize_delivery_method,
        "notification_method": n.normalize_notification_method,
    }


def _apply_normalizers(payload: dict[str, Any]) -> dict[str, Any]:
    """A copy of the payload with extracted/corrections values normalized."""
    table = _normalizers()
    out = dict(payload)
    for bucket in ("extracted", "corrections"):
        values = payload.get(bucket)
        if not isinstance(values, dict):
            continue
        normalized = {}
        for slot, raw in values.items():
            fn = table.get(slot)
            if fn and isinstance(raw, str) and raw:
                try:
                    normalized[slot] = fn(raw) or raw
                except Exception:
                    normalized[slot] = raw
            else:
                normalized[slot] = raw
        out[bucket] = normalized
    return out


def _resolve(payload: dict[str, Any], path: str) -> Any:
    """Read a dotted path, so a case can assert 'extracted.dob' directly."""
    node: Any = payload
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


def _compare(expected: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Subset match. Strings compare case-insensitively after stripping."""
    out: list[str] = []
    for path, want in expected.items():
        got = _resolve(payload, path)
        if isinstance(want, str) and isinstance(got, str):
            ok = want.strip().lower() == got.strip().lower()
        else:
            ok = want == got
        if not ok:
            out.append(f"{path}: expected {want!r}, got {got!r}")
    return out


async def _run_case(case: ExtractionCase, llm, sem: asyncio.Semaphore) -> CaseResult:
    from agent.llm.extractor import build_worker_input
    from agent.llm.schema import WorkerResult
    from agent.utils import build_extraction_prompt

    async with sem:
        started = time.time()
        try:
            messages = build_worker_input(
                build_extraction_prompt(case.prompt_file),
                awaiting_slot=case.awaiting_slot,
                last_agent_message=case.last_agent_message,
                last_user_message=case.utterance,
                confirmed_slots=dict(case.confirmed) or None,
                pending_slots=list(case.pending) or None,
                attempt=case.attempt,
                recent_messages=[dict(m) for m in case.history] or None,
            )
            result = None
            last_exc: Optional[Exception] = None
            for attempt in range(3):  # transport flakes are not findings
                try:
                    result = await llm.with_structured_output(WorkerResult).ainvoke(messages)
                    break
                except Exception as exc:  # noqa: PERF203
                    last_exc = exc
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
            if result is None:
                raise last_exc  # type: ignore[misc]
            payload = _apply_normalizers(result.model_dump(mode="json"))
        except Exception as exc:
            return CaseResult(
                case.id,
                case.status,
                False,
                time.time() - started,
                [],
                {},
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
                fixed_by=case.fixed_by,
                note=case.note,
            )

        mismatches = _compare(case.expect, payload)
        return CaseResult(
            case.id,
            case.status,
            not mismatches,
            time.time() - started,
            mismatches,
            payload,
            fixed_by=case.fixed_by,
            note=case.note,
        )


async def _main(args: argparse.Namespace) -> int:
    cases = [c for c in CASES if not args.case or c.id in args.case]
    if not cases:
        raise SystemExit(f"no cases matched {args.case}")

    llm, model_name = _get_llm(args.model)
    print(
        f"{_BOLD}Extraction contract{_RESET}  model={_CYAN}{model_name}{_RESET} "
        f"({args.model})  cases={len(cases)}\n"
    )

    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()
    results = await asyncio.gather(*(_run_case(c, llm, sem) for c in cases))
    elapsed = time.time() - started

    contract = [r for r in results if r.status == "contract"]
    gaps = [r for r in results if r.status == "gap"]

    for r in results:
        if r.passed:
            mark = f"{_GREEN}PASS{_RESET}"
        elif r.status == "gap":
            mark = f"{_YELLOW}GAP {_RESET}"
        else:
            mark = f"{_RED}FAIL{_RESET}"
        print(f"  {mark}  {r.id:32} {_DIM}{r.seconds:5.2f}s{_RESET}")
        if r.error:
            print(f"        {_RED}{r.error}{_RESET}")
        for m in r.mismatches:
            colour = _YELLOW if r.status == "gap" else _RED
            print(f"        {colour}{m}{_RESET}")
        if r.mismatches and r.status == "gap" and r.fixed_by:
            print(f"        {_DIM}→ {r.fixed_by}{_RESET}")
        if args.verbose and r.raw:
            compact = {k: v for k, v in r.raw.items() if v not in (None, {}, [], "", 0.0, False)}
            print(f"        {_DIM}{json.dumps(compact)}{_RESET}")

    passed_contract = sum(1 for r in contract if r.passed)
    closed_gaps = sum(1 for r in gaps if r.passed)

    print(f"\n{_BOLD}{'─' * 68}{_RESET}")
    print(
        f"contract: {passed_contract}/{len(contract)} passing    "
        f"gaps: {closed_gaps}/{len(gaps)} already closed    "
        f"{elapsed:.1f}s wall"
    )

    if args.json:
        # to_thread: this is an async function, and a blocking write here would
        # stall the loop (ruff ASYNC240).
        await asyncio.to_thread(
            Path(args.json).write_text,
            json.dumps(
                {
                    "model": model_name,
                    "tier": args.model,
                    "elapsed_sec": round(elapsed, 2),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            ),
        )
        print(f"{_DIM}wrote {args.json}{_RESET}")

    failed = [r for r in contract if not r.passed]
    if failed:
        print(f"\n{_RED}{_BOLD}Contract failures: {', '.join(r.id for r in failed)}{_RESET}")
        return 1
    print(f"{_GREEN}All contract cases pass.{_RESET}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn-level extraction contract against a live model.")
    ap.add_argument("--model", default="azure", choices=("azure", "gemini"))
    ap.add_argument("--case", action="append", help="run only these case ids (repeatable)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--verbose", "-v", action="store_true", help="print the full extraction result")
    ap.add_argument("--json", help="write results to this path")
    return asyncio.run(_main(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
