"""Shared helpers for resetting chatbot conversation state (cancel / restart / reset)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

RESET_COMMANDS = frozenset({"cancel", "restart", "reset", "start over"})

DEFAULT_RESET_PREFIX = "Your conversation has been reset. Let's start fresh! "


def is_reset_command(user_message: str) -> bool:
    return user_message.strip().lower() in RESET_COMMANDS


def build_reset_conversation_state(conversation_state: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh initial flow state, preserving language preferences."""
    saved_language = conversation_state.get("preferred_language", "English")
    saved_language_code = conversation_state.get("language_code", "en")
    saved_language_explicitly_set = conversation_state.get("language_explicitly_set", False)

    return {
        "current_question_index": 0,
        "responses": {},
        "current_flow": "initial",
        "welcome_shown": True,
        "awaiting_document_name": False,
        "document_name": "",
        "preferred_language": saved_language,
        "language_code": saved_language_code,
        "language_explicitly_set": saved_language_explicitly_set,
    }


def reset_to_initial_and_format_first_question(
    *,
    user_states: dict[str, Any],
    user_id: str,
    conversation_state: dict[str, Any],
    initial_questions: list,
    format_response_in_language: Callable[..., dict[str, Any]],
    reset_prefix: str = DEFAULT_RESET_PREFIX,
) -> dict[str, Any]:
    """
    Replace the user's state with a reset snapshot and return the first initial question,
    prefixed with a reset acknowledgment, in the user's preferred language.
    """
    saved_language = conversation_state.get("preferred_language", "English")
    user_states[user_id] = build_reset_conversation_state(conversation_state)

    first_question = initial_questions[0]
    if isinstance(first_question, dict):
        question_text = first_question["question"]
        options = first_question.get("options", [])
        reset_message = reset_prefix + question_text
        return format_response_in_language(reset_message, options, saved_language)

    reset_message = reset_prefix + str(first_question)
    return format_response_in_language(reset_message, [], saved_language)
