"""Helpers for question objects from JSON (stable routing without brittle display-string equality)."""

from __future__ import annotations

from typing import Any

from services.chatbot.question_steps import LEGACY_QUESTION_TO_STEP


def _display_question_text(question_data: Any) -> str | None:
    if isinstance(question_data, str):
        return question_data
    if isinstance(question_data, dict):
        q = question_data.get("question")
        if isinstance(q, str):
            return q
    return None


def resolve_step_id(question_data: Any) -> str | None:
    """
    Resolve a stable step id for routing:

    1. Dict with ``step_id`` / ``question_key``.
    2. Else optional legacy lookup (currently unused — empty map).

    Plain-string question rows are not supported; use dicts with ``step_id``.
    """
    if isinstance(question_data, dict):
        raw = question_data.get("step_id")
        if raw is None:
            raw = question_data.get("question_key")
        if raw is not None and str(raw).strip():
            return str(raw).strip().lower()
        qt = question_data.get("question")
        if isinstance(qt, str):
            mapped = LEGACY_QUESTION_TO_STEP.get(qt)
            if mapped:
                return mapped
        return None
    if isinstance(question_data, str):
        return LEGACY_QUESTION_TO_STEP.get(question_data)
    return None


def display_question_matches_current_index(
    questions: list[Any],
    conversation_state: dict[str, Any],
    question_text: str,
) -> bool:
    """True if ``question_text`` matches the question at the current index (dict or str row)."""
    idx = conversation_state.get("current_question_index")
    if not isinstance(idx, int) or idx < 0 or idx >= len(questions):
        return False
    q = questions[idx]
    cur = q["question"] if isinstance(q, dict) else q
    return cur == question_text
