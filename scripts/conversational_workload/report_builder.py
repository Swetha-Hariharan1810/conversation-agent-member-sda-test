"""
Saves per-conversation evaluation reports to the results directory as JSON.
"""

import json
import os
from pathlib import Path

from scripts.conversational_workload.models import ConversationReport

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(REPO_ROOT / "scripts/conversational_workload/results")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_report(report: ConversationReport) -> Path:
    filename = f"{report.scenario_tag}_{report.conversation_id}.json"
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    return path
