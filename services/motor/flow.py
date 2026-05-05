from __future__ import annotations

from typing import Any

from services.claim.flow import CLAIM_ROUTER_FLOW, MOTOR_CLAIM_FLOW

MOTOR_FLOWS = frozenset({"motor", "motor_insurance"})

MOTOR_START_OPTIONS = (
    "Tell me your Emirate",
    "Tell me your Emirate sponsor located in?",
    "In which emirate would you prefer your vehicle to be repaired?",
    "Let's start with your motor insurance details. Select the city of registration",
)


class MotorFlowService:
    def __init__(self) -> None:
        self._flows = MOTOR_FLOWS
        self._claim_flow = MOTOR_CLAIM_FLOW
        self._claim_router_flow = CLAIM_ROUTER_FLOW
        self._start_options = set(MOTOR_START_OPTIONS)

    def is_flow(self, flow_name: str) -> bool:
        return flow_name in self._flows

    def is_claim_flow(self, flow_name: str) -> bool:
        return flow_name == self._claim_flow

    def is_claim_router_flow(self, flow_name: str) -> bool:
        return flow_name == self._claim_router_flow

    def is_start_question(self, question: str) -> bool:
        return question in self._start_options

    def get_entry_response(
        self, motor_questions: list[dict[str, Any]]
    ) -> tuple[str, list[str]]:
        first_question = motor_questions[0]
        body = first_question["question"]
        opts = first_question.get("options", [])
        # Motor submenu JSON already includes full greeting + prompt (no extra prefix).
        if first_question.get("step_id") == "motor_menu":
            return body, opts
        return f"Great choice! {body}", opts

    def get_claim_entry_response(
        self, motor_claim_questions: list[Any]
    ) -> tuple[str, list[str]]:
        first_question = motor_claim_questions[0]
        if isinstance(first_question, dict):
            return f"Great choice! {first_question['question']}", first_question.get(
                "options", []
            )
        return f"Great choice! {first_question}", []


motor_flow_service = MotorFlowService()


def is_motor_flow(flow_name: str) -> bool:
    return motor_flow_service.is_flow(flow_name)


def is_motor_claim_flow(flow_name: str) -> bool:
    return motor_flow_service.is_claim_flow(flow_name)


def get_motor_entry_response(motor_questions: list[dict[str, Any]]) -> tuple[str, list[str]]:
    return motor_flow_service.get_entry_response(motor_questions)


def get_motor_claim_entry_response(motor_claim_questions: list[Any]) -> tuple[str, list[str]]:
    return motor_flow_service.get_claim_entry_response(motor_claim_questions)
