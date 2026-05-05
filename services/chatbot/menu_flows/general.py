"""General insurance submenu (car, bike, renew) — questions from ``questions/general/questions.json``."""

from __future__ import annotations

from typing import Any

from services.general.flow import GENERAL_INSURANCE_INTRO, get_general_options_page


def get_general_insurance_entry_response(
    _general_questions: list[Any],
) -> tuple[str, list[str]]:
    return GENERAL_INSURANCE_INTRO, get_general_options_page(0)
