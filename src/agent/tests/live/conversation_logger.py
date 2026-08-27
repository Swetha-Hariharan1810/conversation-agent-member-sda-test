"""
conversation_logger.py — Captures every turn of a live test run.

Writes per-run JSON transcripts and a running summary CSV so test failures
can be diagnosed without re-running the full suite.

Output locations (relative to the repo root):
  src/agent/tests/live/conversations/intake/<timestamp>_<test_name>.json
  src/agent/tests/live/conversations/intake/summary.csv

Changes from previous version
──────────────────────────────
_safe_state_snapshot: added 13 state keys that claim-adjustment, records-
  coordination, and notification-setup tests read from record.final_state
  and from turn.state_snapshot:

    reference_number                  – assert_reference_collected (Group A)
    claim_status                      – assert_claim_status_reported (Group A)
    records_required                  – assert_records_required_set (Group A)
    upload_link_sent                  – assert_upload_link_sent (Groups B/B2/R)
    personal_guide_outreach_requested – assert_personal_guide_triggered (Groups B/R)
    records_branch_taken              – assert_records_branch (Phase 4)
    notification_channel              – assert_notification_channel (Groups C/R)
    claim_notification_contact        – Phase 5 D-combo asserts
    claim_timeline_notification_channel – assert_n2_notification_channel (Phase 5)
    claim_timeline_notification_contact – Phase 5 D-combo asserts
    claim_flow_complete               – assert_claim_flow_complete (Phase 5)
    phone_update_requested            – Phase 6 A2 no-variant asserts
    closure_requested                 – assert_call_closed any-turn scan

_percentile: extracted as a module-level function (was duplicated inline
  inside both to_dict() and latency_summary(), risking silent drift).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level percentile helper — single source of truth
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float:
    """
    Return the p-th percentile (0–100) of a pre-sorted list of floats.

    Returns 0.0 for an empty list, the single value for a one-element list.
    Uses linear interpolation between adjacent ranks.
    """
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """All observable state for a single conversation turn."""

    turn_number: int
    user_input: str
    agent_message: str
    state_snapshot: dict  # full LangGraph state dict at this turn
    slot_attempts: dict
    awaiting_slot: str
    next_node: str
    is_interrupt: bool
    active_agent: str
    guard_fired: str  # from WorkerResult.guard if available
    intent_classified: str  # call_intent from state
    metadata_events: list
    timestamp: str
    duration_sec: float = 0.0  # wall-clock seconds for this turn's ainvoke call

    def to_dict(self) -> dict:
        return {
            "turn": self.turn_number,
            "user": self.user_input,
            "agent": self.agent_message,
            "duration_sec": self.duration_sec,
            "state": {
                "call_intent": self.state_snapshot.get("call_intent"),
                "next_node": self.next_node,
                "active_agent": self.active_agent,
                "is_interrupt": self.is_interrupt,
                "awaiting_slot": self.awaiting_slot,
                "escalation_reason": self.state_snapshot.get("escalation_reason"),
                "caller_type": self.state_snapshot.get("caller_type"),
                "caller_type_handled": self.state_snapshot.get("caller_type_handled"),
                "slot_attempts": self.slot_attempts,
                "offtopic_global_count": self.state_snapshot.get("offtopic_global_count"),
            },
            "guard_fired": self.guard_fired,
            "intent_classified": self.intent_classified,
            "metadata_events": self.metadata_events,
            "timestamp": self.timestamp,
        }


@dataclass
class AssertionRecord:
    check: str
    result: str  # "PASS" | "FAIL"
    detail: str = ""

    def to_dict(self) -> dict:
        return {"check": self.check, "result": self.result, "detail": self.detail}


@dataclass
class ConversationRecord:
    """Complete record of one test scenario execution."""

    test_name: str
    scenario_description: str
    conversation_id: str
    started_at: str
    ended_at: str = ""
    total_turns: int = 0
    final_state: dict = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    test_outcome: str = "PENDING"  # PASS / FAIL / ERROR
    failure_reason: str = ""
    assertions_checked: list[AssertionRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_turn(self, user_input: str, state: dict) -> None:
        """Append a turn from the current LangGraph state dict."""
        messages = state.get("messages", [])
        agent_message = ""
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                if last.get("role") == "assistant":
                    agent_message = last.get("content", "")
            else:
                # LangChain message object
                role = getattr(last, "type", None) or getattr(last, "role", "")
                if role in ("assistant", "ai"):
                    agent_message = getattr(last, "content", "")

        turn = TurnRecord(
            turn_number=len(self.turns),
            user_input=user_input,
            agent_message=agent_message,
            state_snapshot=_safe_state_snapshot(state),
            slot_attempts=dict(state.get("slot_attempts") or {}),
            awaiting_slot=state.get("awaiting_slot", ""),
            next_node=state.get("next_node", ""),
            is_interrupt=bool(state.get("is_interrupt", False)),
            active_agent=state.get("active_agent", ""),
            guard_fired=_extract_guard(state),
            intent_classified=state.get("call_intent", ""),
            metadata_events=list(state.get("metadata_events") or []),
            timestamp=_now(),
        )
        self.turns.append(turn)
        self.final_state = _safe_state_snapshot(state)
        self.total_turns = len(self.turns)

    def record_assertion(self, check: str, passed: bool, detail: str = "") -> None:
        self.assertions_checked.append(
            AssertionRecord(
                check=check,
                result="PASS" if passed else "FAIL",
                detail=detail,
            )
        )

    def finalize(self, outcome: str, failure_reason: str = "") -> None:
        self.ended_at = _now()
        self.test_outcome = outcome
        self.failure_reason = failure_reason

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        durations = [t.duration_sec for t in self.turns if t.duration_sec > 0]
        s = sorted(durations)

        latency_block = {
            "turn_durations_sec": [t.duration_sec for t in self.turns],
            "p50": round(_percentile(s, 50), 4),
            "p95": round(_percentile(s, 95), 4),
            "avg": round(statistics.mean(durations), 4) if durations else 0.0,
        }

        return {
            "test_name": self.test_name,
            "scenario": self.scenario_description,
            "conversation_id": self.conversation_id,
            "outcome": self.test_outcome,
            "failure_reason": self.failure_reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_turns": self.total_turns,
            "final_state": self.final_state,
            "turns": [t.to_dict() for t in self.turns],
            "assertions": [a.to_dict() for a in self.assertions_checked],
            "latency_summary": latency_block,
        }

    def latency_summary(self) -> str:
        """Return a formatted latency summary string (stdlib only)."""
        durations = [t.duration_sec for t in self.turns if t.duration_sec > 0]
        if not durations:
            return "  No latency data recorded."

        s = sorted(durations)
        lines = [
            f"  Latency — turns={len(durations)}  "
            f"avg={statistics.mean(durations):.3f}s  "
            f"p50={_percentile(s, 50):.3f}s  "
            f"p95={_percentile(s, 95):.3f}s"
        ]
        for t in self.turns:
            if t.duration_sec > 0:
                label = t.user_input[:40]
                lines.append(f"    turn {t.turn_number:>2}  {label!r:<44}  {t.duration_sec:.3f}s")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, directory: str) -> Path:
        """Write the JSON transcript and append to summary.csv. Returns the path."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_name = self.test_name.replace(" ", "_")[:80]
        json_path = out_dir / f"{ts}_{safe_name}.json"
        json_path.write_text(self.to_json(), encoding="utf-8")

        _append_summary_csv(out_dir / "summary.csv", self)
        return json_path

    def print_conversation(self) -> None:
        """Pretty-print the conversation to stdout for terminal inspection."""
        width = 78
        print("\n" + "=" * width)
        print(f"  TEST : {self.test_name}")
        print(f"  DESC : {self.scenario_description}")
        print(f"  ID   : {self.conversation_id}")
        print(f"  OUT  : {self.test_outcome}  ({self.failure_reason or 'ok'})")
        print("=" * width)

        for t in self.turns:
            print(f"\n[Turn {t.turn_number}]  {t.timestamp}")
            if t.user_input and t.user_input != "[SYSTEM START]":
                print(f"  USER  : {t.user_input}")
            if t.agent_message:
                print(f"  AGENT : {t.agent_message}")
            print(
                f"  STATE : intent={t.intent_classified!r}  "
                f"next={t.next_node!r}  "
                f"interrupt={t.is_interrupt}  "
                f"guard={t.guard_fired!r}"
            )

        if self.assertions_checked:
            print("\n" + "-" * width)
            print("  ASSERTIONS")
            for a in self.assertions_checked:
                icon = "✓" if a.result == "PASS" else "✗"
                detail = f"  [{a.detail}]" if a.detail else ""
                print(f"    {icon}  {a.check}{detail}")

        print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Logger class (thin wrapper used by conftest fixture)
# ---------------------------------------------------------------------------


class ConversationLogger:
    """Manages the conversations directory and accumulates summary rows."""

    def __init__(self, conversations_dir: str) -> None:
        self.conversations_dir = conversations_dir
        Path(conversations_dir).mkdir(parents=True, exist_ok=True)

    def new_record(self, test_name: str, scenario: str, conversation_id: str) -> ConversationRecord:
        return ConversationRecord(
            test_name=test_name,
            scenario_description=scenario,
            conversation_id=conversation_id,
            started_at=_now(),
        )

    def save(self, record: ConversationRecord) -> Path:
        return record.save(self.conversations_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_state_snapshot(state: dict) -> dict:
    """
    Return a JSON-serialisable subset of the LangGraph state dict.

    The `keep` set must contain every state key that any test helper reads
    from record.final_state or from turn.state_snapshot.  Add new keys here
    whenever a new assertion helper is introduced in any test file.

    Grouped by which agent/flow populates them:
      Core routing      — call_intent, next_node, is_interrupt, active_agent,
                          awaiting_slot, app_run_id, resolved_intents,
                          previous_agents, metadata_events, slot_attempts
      Guards / safety   — escalation_reason, caller_type, caller_type_handled,
                          offtopic_global_count, closure_requested
      Verification      — member_status_verify, first_name, last_name, member_id,
                          dob, phone_confirmed, phone_update_requested,
                          relationship, verification_restart_index
      Contact / delivery — phone_number, zip_code, fax, email
      Benefits          — individual_deductible, family_deductible,
                          coinsurance_percent, individual_oop_max, family_oop_max
      Provider search   — provider_type
      Delivery mgmt     — delivery_method, provider_list_sent, benefits_offer_made,
                          proactive_offer_available, delivery_timestamp
      Care & wellness   — benefits_explained, care_coach_offered,
                          care_coach_nooffer_sent, care_coach_details_sent
      Claim adjustment  — reference_number, claim_status, records_required,
                          last_update_date
      Records coord     — upload_link_sent, personal_guide_outreach_requested,
                          records_branch_taken
      Notification      — notification_channel, claim_notification_contact,
                          claim_timeline_notification_channel,
                          claim_timeline_notification_contact, claim_flow_complete
    """
    keep = {
        # ── Core routing ────────────────────────────────────────────────────
        "call_intent",
        "next_node",
        "is_interrupt",
        "active_agent",
        "awaiting_slot",
        "app_run_id",
        "resolved_intents",
        "previous_agents",
        "metadata_events",
        "slot_attempts",
        # ── Guards / safety ──────────────────────────────────────────────────
        "escalation_reason",
        "caller_type",
        "caller_type_handled",
        "offtopic_global_count",
        # closure_requested is set by signal_complete(closure_requested=True)
        # assert_call_closed scans any(t.state_snapshot.get("closure_requested"))
        "closure_requested",
        # ── Verification ─────────────────────────────────────────────────────
        "member_status_verify",
        "first_name",
        "last_name",
        "member_id",
        "dob",
        "phone_confirmed",
        # phone_update_requested is set when caller declines verification phone
        # Phase 6 A2_no tests assert phone_update_requested=True
        "phone_update_requested",
        "relationship",
        "verification_restart_index",
        # ── Contact fields (from Salesforce via context_updates) ──────────────
        "phone_number",
        "zip_code",
        "fax",
        "email",
        # ── Benefits (prefetched during verification) ─────────────────────────
        "individual_deductible",
        "family_deductible",
        "coinsurance_percent",
        "individual_oop_max",
        "family_oop_max",
        # ── Provider search ───────────────────────────────────────────────────
        "provider_type",
        # ── Delivery management ───────────────────────────────────────────────
        "delivery_method",
        "provider_list_sent",
        "benefits_offer_made",
        "proactive_offer_available",
        "delivery_timestamp",
        # ── Care & wellness ───────────────────────────────────────────────────
        "benefits_explained",
        "care_coach_offered",
        "care_coach_nooffer_sent",
        "care_coach_details_sent",
        "rewards_portal_shared",
        # ── Claim adjustment ──────────────────────────────────────────────────
        # reference_number: assert_reference_collected (Group A)
        "reference_number",
        # claim_status: assert_claim_status_reported (Group A)
        "claim_status",
        # records_required: assert_records_required_set (Group A)
        "records_required",
        # last_update_date: surfaced in claim status message checks
        "last_update_date",
        # ── Records coordination ──────────────────────────────────────────────
        # upload_link_sent: assert_upload_link_sent (Groups B/B2/R)
        # Also scanned across turns: any(t.state_snapshot.get("upload_link_sent"))
        "upload_link_sent",
        # personal_guide_outreach_requested: assert_personal_guide_triggered
        "personal_guide_triggered",
        # Also scanned across turns
        "personal_guide_outreach_requested",
        # records_branch_taken: assert_records_branch (Phase 4)
        "records_branch_taken",
        # ── Notification setup ────────────────────────────────────────────────
        # notification_channel: assert_notification_channel (Groups C/R/C2)
        # Also scanned across turns for C10 bridge test
        "notification_channel",
        # claim_notification_contact: Phase 5 D-combo asserts
        "claim_notification_contact",
        # claim_timeline_notification_channel: assert_n2_notification_channel (Phase 5)
        # Also scanned across turns
        "claim_timeline_notification_channel",
        # claim_timeline_notification_contact: Phase 5 D-combo asserts
        "claim_timeline_notification_contact",
        # claim_flow_complete: assert_claim_flow_complete (Phase 5)
        "claim_flow_complete",
    }

    snapshot = {}
    for k in keep:
        val = state.get(k)
        if val is not None:
            try:
                json.dumps(val, default=str)
                snapshot[k] = val
            except Exception:
                snapshot[k] = str(val)

    # Include messages as simple role/content pairs
    raw_messages = state.get("messages", [])
    simple_msgs = []
    for m in raw_messages:
        if isinstance(m, dict):
            simple_msgs.append({"role": m.get("role", ""), "content": m.get("content", "")})
        else:
            role = getattr(m, "type", None) or getattr(m, "role", "unknown")
            content = getattr(m, "content", "")
            simple_msgs.append({"role": role, "content": content})
    snapshot["messages"] = simple_msgs
    return snapshot


def _extract_guard(state: dict) -> str:
    """Best-effort extraction of the guard that fired this turn."""
    for event in state.get("metadata_events") or []:
        if isinstance(event, dict) and event.get("eventType") == "AgentCallEvent":
            data = event.get("data", {})
            if data.get("eventName") == "AgentCallTransfer":
                return f"TRANSFER:{data.get('detail', '')}"
    return ""


def _append_summary_csv(csv_path: Path, record: ConversationRecord) -> None:
    """Append one row to the summary CSV, creating headers if needed."""
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_name", "outcome", "turns", "intent", "escalated", "timestamp"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "test_name": record.test_name,
                "outcome": record.test_outcome,
                "turns": record.total_turns,
                "intent": record.final_state.get("call_intent", ""),
                "escalated": bool(record.final_state.get("escalation_reason")),
                "timestamp": record.ended_at or record.started_at,
            }
        )
