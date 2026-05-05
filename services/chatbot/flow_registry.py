from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.medical import medical_flow_service
from services.motor import motor_flow_service
from services.claim.flow import CLAIM_ROUTER_FLOW, MEDICAL_CLAIM_FLOW


@dataclass
class ChatbotFlowRegistry:
    initial_questions: list[Any]
    medical_questions: list[Any]
    individual_questions: list[Any]
    motor_insurance_questions: list[Any]
    car_questions: list[Any]
    bike_questions: list[Any]
    existing_policy_questions: list[Any]
    motor_claim_questions: list[Any]
    medical_claim_questions: list[Any]
    claim_router_questions: list[Any]
    general_insurance_questions: list[Any]

    def get_questions_for_flow(self, current_flow: str) -> list[Any]:
        if current_flow == "initial":
            return self.initial_questions
        if medical_flow_service.is_flow(current_flow):
            return self.medical_questions
        if current_flow == "individual":
            return self.individual_questions
        if motor_flow_service.is_flow(current_flow):
            return self.motor_insurance_questions
        if current_flow == "car_questions":
            return self.car_questions
        if current_flow == "bike_questions":
            return self.bike_questions
        if current_flow == "existing_policy":
            return self.existing_policy_questions
        if current_flow == CLAIM_ROUTER_FLOW:
            return self.claim_router_questions
        if motor_flow_service.is_claim_flow(current_flow):
            return self.motor_claim_questions
        if current_flow == MEDICAL_CLAIM_FLOW:
            return self.medical_claim_questions
        if current_flow == "general_insurance":
            return self.general_insurance_questions
        return []

    def is_medical_start_question(self, question: str) -> bool:
        return medical_flow_service.is_start_question(question)

    def is_motor_start_question(self, question: str) -> bool:
        return motor_flow_service.is_start_question(question)

