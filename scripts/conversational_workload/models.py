from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PCPInquiryEntity(BaseModel):
    first_name: str
    last_name: str
    member_id: str
    date_of_birth: str
    subscriber_type: Optional[str] = Field(default="myself")
    provider_type: Optional[str] = Field(default="Primary Care Physician")
    zip_code: Optional[str] = Field(default=None)
    fax_number: Optional[str] = Field(default=None)


class ClaimAdjustmentEntity(BaseModel):
    first_name: str
    last_name: str
    member_id: str
    date_of_birth: str
    phone_number: str
    reference_number: str
    email: str


class JudgeResult(BaseModel):
    intent_score: float
    constraint_score: float
    completeness_score: float
    naturalness_score: float
    overall: float
    verdict: Optional[str] = None

    def finalize(self) -> "JudgeResult":
        if not self.verdict:
            self.verdict = "PASS" if self.overall >= 0.8 else "FAIL"
        return self


class TurnEvaluation(BaseModel):
    ai_prompt: str
    user_response: str
    ground_truth: str
    slot: str
    scenario: str = ""
    scores: Dict


class ConversationReport(BaseModel):
    conversation_id: str
    flow: str
    scenario_tag: str
    completed: bool
    turns: List[TurnEvaluation]
    final_score: float
