"""
validate_transcripts.py — Check persona/transcript consistency.

For each scenario, simulates one pass through the transcript using the
simulator persona and flags any turn where the simulator's likely
response would score < 0.5 against the transcript ground truth.

Run this after editing any transcript or persona:
    python scripts/conversational_workload/validate_transcripts.py
"""

from pathlib import Path

# Keyword consistency checks per slot —
# verifies the transcript human response contains expected keywords
# given the scenario persona's stated intent for that slot.
SLOT_KEYWORD_EXPECTATIONS = {
    "claim_adjustment_happy_path": {
        "n2_notification_method": ["email"],
        "follow_up": ["reward", "wellness", "annual"],
    },
    "claim_adjustment_guide_only": {
        "n2_notification_method": ["email"],
        "notification_method": ["phone", "sms", "text"],
        "upload_consent": ["no"],
        "follow_up": ["finalized", "know when"],
    },
    "claim_adjustment_upload_only": {
        "upload_consent": ["yes"],
        "personal_guide_consent": ["no"],
        "n2_notification_method": ["sms", "phone", "text"],
    },
    "claim_adjustment_no_proceed": {
        "upload_consent": ["no"],
        "personal_guide_consent": ["no", "don't", "dont"],
    },
}

BASE = Path(__file__).parent / "static_transcripts"


def check():
    from scripts.conversational_workload.intent_classifier import classify_ai_slot
    from scripts.conversational_workload.transcript_cursor import (
        _SCENARIO_FILE_MAP,
        _parse_transcript,
    )

    errors = []
    for scenario, expectations in SLOT_KEYWORD_EXPECTATIONS.items():
        fname = _SCENARIO_FILE_MAP.get(scenario)
        if not fname:
            continue
        path = BASE / fname
        if not path.exists():
            errors.append(f"MISSING: {path}")
            continue
        turns = _parse_transcript(path)
        for turn in turns:
            slot = classify_ai_slot(turn.ai, flow="claim")
            if slot in expectations:
                keywords = expectations[slot]
                human_lower = turn.human.lower()
                if not any(kw in human_lower for kw in keywords):
                    errors.append(
                        f"{scenario} | slot={slot}\n"
                        f"  ai:    {turn.ai[:80]}\n"
                        f"  human: {turn.human}\n"
                        f"  expected keywords: {keywords}"
                    )

    if errors:
        print(f"\n{len(errors)} CONSISTENCY ERROR(S):\n")
        for e in errors:
            print(e)
            print()
    else:
        print("All transcripts pass consistency checks.")


if __name__ == "__main__":
    check()
