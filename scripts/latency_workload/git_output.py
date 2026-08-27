"""
Write GitHub Actions workflow outputs for CI integration.
"""

import os

from agent.logger import get_logger

logger = get_logger(__name__)


def set_github_outputs(values: dict) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for k, v in values.items():
                fh.write(f"{k}={v}\n")
    except Exception as exc:
        logger.info("Warning: Could not write GitHub outputs: %s", exc)


def report_to_github(overall: dict, json_path: str, step_count: int) -> None:
    """Write per-step latency outputs to GITHUB_OUTPUT (no-op if not set)."""
    set_github_outputs(
        {
            "avg": f"{overall['avg']:.6f}",
            "p50": f"{overall['p50']:.6f}",
            "p95": f"{overall['p95']:.6f}",
            "std_dev": f"{overall['std_dev']:.6f}",
            "json_path": str(json_path),
            "step_count": str(step_count),
        }
    )
