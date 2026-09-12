"""follow_up asks "anything else?" once per turn, not twice.

    AI  Your deductible is $750 for the year. Is there anything else I can help
        you with? Is there anything else from our call today I can help with?

The classifier's answer usually ends by handing the call back, and
MSG_CONTINUATION says the same thing. Whether to append it was decided by
testing the answer against the exact strings in the pool — three entries
against the thousands of ways a model phrases it — so any phrasing one word off
the pool went unrecognised and the caller heard the question twice running.

The test is now on the shape of the closing sentence rather than its wording.
"""

from __future__ import annotations

import pytest

from agent.agents.follow_up.agent import _ends_with_continuation
from agent.agents.follow_up.constants import MSG_CONTINUATION


@pytest.mark.parametrize("phrasing", MSG_CONTINUATION)
def test_the_pool_itself_is_recognised(phrasing):
    assert _ends_with_continuation(f"Your deductible is $750 for the year. {phrasing}") is True


@pytest.mark.parametrize(
    "closing",
    [
        "Is there anything else I can help you with?",
        "Is there anything else I can help you with today?",
        "Anything else I can do for you?",
        "Do you have any other questions?",
        "Was there something else you needed?",
        "What else can I help you with?",
        "How else can I help today?",
        "Are you all set for today?",
    ],
)
def test_a_phrasing_off_the_pool_is_recognised_too(closing):
    """This is the bug: each of these already hands the call back, and each one
    used to get MSG_CONTINUATION appended behind it."""
    assert _ends_with_continuation(f"Your deductible is $750 for the year. {closing}") is True


@pytest.mark.parametrize(
    "answer",
    [
        "Your deductible is $750 for the year.",
        "The list went out to your fax a moment ago.",
        # Mentions other questions and then stops: a declarative leaves the
        # caller nothing to respond to, so the continuation is still owed.
        "I covered your other questions about the deductible earlier in the call.",
        "",
    ],
)
def test_an_answer_that_does_not_hand_back_still_gets_the_continuation(answer):
    assert _ends_with_continuation(answer) is False


def test_a_hand_back_without_a_question_mark_counts():
    """ "Let me know if…" puts something to the member just as a question does."""
    assert (
        _ends_with_continuation(
            "Your deductible is $750 for the year. Let me know if you have any other questions."
        )
        is True
    )
