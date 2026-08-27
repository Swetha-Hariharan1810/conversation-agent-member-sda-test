"""
LLM-as-judge for evaluating simulated user responses against ground truth.
Uses the project's extraction LLM (Azure OpenAI).
"""

import json
import re

from agent.llm.config import get_extraction_llm
from scripts.conversational_workload.models import JudgeResult


def _safe_float(value, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _extract_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
    content = content.strip()
    if content.startswith("json"):
        content = content[4:].strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON found in LLM response")


def judge_turn(user_text: str, ground_truth: str) -> JudgeResult:
    """
    Evaluate a simulated user response against the expected ground truth.
    Returns a JudgeResult with scores 0.0–1.0 and a PASS/FAIL verdict.
    """
    if not ground_truth or not ground_truth.strip():
        return JudgeResult(
            intent_score=1.0,
            constraint_score=1.0,
            completeness_score=1.0,
            naturalness_score=1.0,
            overall=1.0,
        ).finalize()

    if user_text.strip().lower() == ground_truth.strip().lower():
        return JudgeResult(
            intent_score=1.0,
            constraint_score=1.0,
            completeness_score=1.0,
            naturalness_score=1.0,
            overall=1.0,
        ).finalize()

    prompt = f"""You are evaluating a simulated healthcare member call.

Ground Truth (expected user reply):
{ground_truth}

Actual User Response:
{user_text}

Score each category from 0.0 to 1.0:
1. intent_score — Does the response express the same intent?
2. constraint_score — Does it satisfy required confirmations, numbers, or yes/no constraints?
3. completeness_score — Does it fully answer what was requested?
4. naturalness_score — Does it sound like a real human caller?

Return ONLY valid JSON:
{{
  "intent_score": <float>,
  "constraint_score": <float>,
  "completeness_score": <float>,
  "naturalness_score": <float>
}}"""

    try:
        llm = get_extraction_llm()
        result = llm.invoke(prompt)
        raw = result.content.strip() if hasattr(result, "content") else str(result).strip()
        data = _extract_json(raw)
    except Exception:
        data = {
            "intent_score": 0.5,
            "constraint_score": 0.5,
            "completeness_score": 0.5,
            "naturalness_score": 0.5,
        }

    i = _safe_float(data.get("intent_score"))
    c = _safe_float(data.get("constraint_score"))
    co = _safe_float(data.get("completeness_score"))
    n = _safe_float(data.get("naturalness_score"))
    overall = round((i + c + co + n) / 4, 2)

    return JudgeResult(
        intent_score=i,
        constraint_score=c,
        completeness_score=co,
        naturalness_score=n,
        overall=overall,
    ).finalize()
