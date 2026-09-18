"""
Output formatting: Markdown table printed to stdout and JSON file writer.
"""

import json
from pathlib import Path
from typing import Dict


def print_markdown_table(
    title: str,
    metrics: Dict[str, float],
    step_count: int,
    iterations: int,
    threshold: float,
) -> None:
    print(f"\n## {title}")
    print(f"Iterations: {iterations} | Steps: {step_count} | Threshold: {threshold:.3f}s")
    print("\n| Metric | Value (s) |\n|---|---:|")
    for k, v in metrics.items():
        print(f"| {k} | {v:.3f} |")
    print()


def save_json(output_path: str, data: Dict) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
