"""Motor purchase insurance flow entry."""

from __future__ import annotations

from typing import Any

from services.motor.flow import get_motor_entry_response


def get_motor_menu_entry_response(motor_questions: list[dict[str, Any]]) -> tuple[str, list[str]]:
    return get_motor_entry_response(motor_questions)
