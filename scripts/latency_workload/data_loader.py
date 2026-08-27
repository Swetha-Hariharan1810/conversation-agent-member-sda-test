"""
Parses scenario text files into lists of user inputs for benchmarking.

File format (each line is either ai: or human:):
    ai: <agent message>
    human: <user turn>

Only the human: lines are extracted and returned in order.
"""

import re
from pathlib import Path
from typing import Dict, List


def load_scenarios(scenarios_path: str) -> Dict[str, List[str]]:
    scenarios = {}
    base = Path(scenarios_path)

    if not base.exists():
        raise FileNotFoundError(f"Scenario path not found: {base}")

    pattern = re.compile(r"(?i)^(user|human)\s*:+\s*")

    for file in sorted(base.glob("*.txt")):
        steps: List[str] = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if pattern.match(stripped):
                    text = pattern.sub("", stripped)
                    steps.append(text)
        if steps:
            scenarios[file.stem] = steps

    return scenarios
