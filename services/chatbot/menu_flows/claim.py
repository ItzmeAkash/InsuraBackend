"""Motor (and related) claims flow entry."""

from __future__ import annotations

from typing import Any

from services.motor.flow import get_motor_claim_entry_response


def get_claim_entry_response(motor_claim_questions: list[Any]) -> tuple[str, list[str]]:
    return get_motor_claim_entry_response(motor_claim_questions)
