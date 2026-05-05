from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from services.chatbot.language_service import (
    detect_document_type_from_question,
    format_response_in_language,
    llm,
    translate_text,
    validate_response_multilingual,
)
from services.chatbot.question_store import user_states
from services.chatbot.question_steps import STEP_MEDICAL_PLAN_TYPE, STEP_MONTHLY_SALARY
from services.medical.flow import (
    MEDICAL_MONTHLY_SALARY_QUESTION,
    MEDICAL_PLAN_TYPE_QUESTION,
    patch_medical_marital_status_question,
)


def _transition_intro_for_next_question(next_question: dict | str) -> str:
    """Different intro lines when introducing certain next steps."""
    if isinstance(next_question, dict):
        q = next_question.get("question")
        sid = next_question.get("step_id")
        if sid == STEP_MEDICAL_PLAN_TYPE or q == MEDICAL_PLAN_TYPE_QUESTION:
            return "Wonderful! Now, let's move on to:"
        if sid == STEP_MONTHLY_SALARY or q == MEDICAL_MONTHLY_SALARY_QUESTION:
            return "Now, let's move on to:"
    elif next_question == MEDICAL_PLAN_TYPE_QUESTION:
        return "Wonderful! Now, let's move on to:"
    elif next_question == MEDICAL_MONTHLY_SALARY_QUESTION:
        return "Now, let's move on to:"
    return "Thank you! Now, let's move on to:"


def handle_option_validation_multilingual(
    user_message: str,
    valid_options: list,
    question: str,
    user_language: str,
    conversation_state: dict,
    questions: list,
    responses: dict,
    user_id: str,
) -> dict:
    validation_result = validate_response_multilingual(user_message, valid_options, user_language)

    if validation_result["is_valid"]:
        english_value = validation_result["matched_value"]
        if english_value == "Abudhabi":
            english_value = "Abu Dhabi"
        responses[question] = english_value
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            _nq_i = conversation_state["current_question_index"]
            questions[_nq_i] = patch_medical_marital_status_question(
                questions[_nq_i], responses, conversation_state
            )
            next_question = questions[_nq_i]

            if isinstance(next_question, dict):
                next_question_text = next_question["question"]
                next_options = next_question.get("options", [])
                intro = _transition_intro_for_next_question(next_question)
                response_message = f"{intro} {next_question_text}"
                msg_type, doc_type = detect_document_type_from_question(next_question_text)
                return format_response_in_language(
                    response_message, next_options, user_language, msg_type, doc_type
                )

            intro = _transition_intro_for_next_question(next_question)
            response_message = f"{intro} {next_question}"
            msg_type, doc_type = detect_document_type_from_question(next_question)
            return format_response_in_language(response_message, [], user_language, msg_type, doc_type)

        with open("user_responses.json", "w") as file:
            json.dump(responses, file, indent=4)
        if user_id in user_states:
            del user_states[user_id]

        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
        result = format_response_in_language(final_message, [], user_language)
        result["final_responses"] = responses
        return result

    error_prompt = (
        f"The user said '{user_message}' but needs to choose from: {', '.join(valid_options)}. "
        "Provide a brief, helpful message explaining they need to select a valid option."
    )
    error_response = llm.invoke(
        [
            SystemMessage(
                content=f"You are Insura, a friendly insurance assistant. Respond in {user_language}. Be brief and helpful."
            ),
            HumanMessage(content=error_prompt),
        ]
    )

    retry_message = f"Let's try again: {question}"
    retry_translated = translate_text(retry_message, user_language)
    translated_options = [translate_text(opt, user_language) for opt in valid_options]

    return {
        "response": error_response.content.strip(),
        "question": retry_translated,
        "options": ", ".join(translated_options),
    }

