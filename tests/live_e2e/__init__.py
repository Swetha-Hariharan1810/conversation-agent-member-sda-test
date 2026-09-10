"""
Live end-to-end conversation tests.

These tests drive the REAL LangGraph application graph with REAL Azure OpenAI
LLM calls and REAL Salesforce queries — nothing is mocked or patched.

Entry points:
    python -m tests.live_e2e.run_live_tests          # canonical CLI
    pytest -m live tests/live_e2e/test_live.py       # pytest wrapper

See tests/live_e2e/README.md for required environment and fixture data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `agent.*` importable when the project package is not pip-installed
# (e.g. running from a bare checkout with PYTHONPATH unset).
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
