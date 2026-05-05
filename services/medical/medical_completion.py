"""Medical individual flow: quotation API call + repurchase question with options in the same reply."""

from __future__ import annotations

import json
from typing import Any

from services.chatbot.constants import (
    MEDICAL_INDIVIDUAL_COMPLETION_RESPONSE,
    WEHBE_GOOGLE_REVIEW_URL,
    WEHBE_REVIEW_INVITE_MESSAGE,
)
from services.medical.flow import MEDICAL_REPURCHASE_QUESTION


def respond_medical_quotation_complete(
    *,
    responses: dict[str, Any],
    conversation_state: dict[str, Any],
    questions: list[Any],
    user_language: str,
    fetching_medical_detail: Any,
    translate_text: Any,
    get_language_code: Any,
    insurance_lab_base_url: str,
) -> dict[str, Any]:
    """Submit medical payload; reply includes plan link, review fields, and repurchase Yes/No."""
    try:
        medical_detail_response = fetching_medical_detail(responses)
        if isinstance(medical_detail_response, int):
            conversation_state["current_question_index"] = len(questions)
            conversation_state["awaiting_medical_repurchase"] = True
            conversation_state.pop("pending_medical_repurchase_prompt", None)

            customer_plan_link = (
                f"{insurance_lab_base_url}/customer_plan/{medical_detail_response}"
            )
            responses["customer_plan_link"] = customer_plan_link

            translated_success = translate_text(
                MEDICAL_INDIVIDUAL_COMPLETION_RESPONSE, user_language
            )
            translated_review = translate_text(
                WEHBE_REVIEW_INVITE_MESSAGE, user_language
            )
            translated_repurchase = translate_text(
                MEDICAL_REPURCHASE_QUESTION, user_language
            )
            yes_no = [
                translate_text("Yes", user_language),
                translate_text("No", user_language),
            ]

            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
            except OSError:
                pass
            return {
                "response": translated_success,
                "link": customer_plan_link,
                "review_message": translated_review,
                "review_link": WEHBE_GOOGLE_REVIEW_URL,
                "language": user_language,
                "language_code": get_language_code(user_language),
                "restart_conversation": True,
                "question": translated_repurchase,
                "options": ", ".join(yes_no),
            }

        fallback = translate_text(str(medical_detail_response), user_language)
        return {
            "response": fallback,
            "language": user_language,
            "language_code": get_language_code(user_language),
        }
    except Exception as exc:
        err = translate_text(
            f"An error occurred while submitting your details: {exc}",
            user_language,
        )
        return {
            "response": err,
            "language": user_language,
            "language_code": get_language_code(user_language),
        }
