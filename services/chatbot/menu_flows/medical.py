"""Medical insurance flow entry (passkey → individual visa emirate questions)."""

from __future__ import annotations

from typing import Any

from services.medical.flow import get_medical_entry_response


def get_medical_menu_entry_response(medical_questions: list[Any]) -> tuple[str, list[str]]:
    return get_medical_entry_response(medical_questions)
