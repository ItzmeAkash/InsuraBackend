from __future__ import annotations

import json
import re
from typing import Any

from services.chatbot.question_utils import display_question_matches_current_index
from services.medical.flow import (
    MEDICAL_SPONSOR_EMAIL_QUESTION,
    patch_medical_marital_status_question,
)


class MedicalIndividualHandler:
    _EMAIL_QUESTIONS = {
        MEDICAL_SPONSOR_EMAIL_QUESTION,
        "May i know your Email address",
    }
    _FULL_NAME_QUESTION = "Could you please provide your full name"

    def handle_email_questions(
        self,
        *,
        question: str,
        user_message: str,
        user_language: str,
        current_flow: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        translate_text: Any,
        get_language_code: Any,
        llm: Any,
        system_message_cls: Any,
        human_message_cls: Any,
    ) -> dict[str, Any] | None:
        if question not in self._EMAIL_QUESTIONS:
            return None

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        is_valid_input = re.match(email_pattern, user_message) or user_message.strip().isdigit()
        if is_valid_input:
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                _nq_i = conversation_state["current_question_index"]
                questions[_nq_i] = patch_medical_marital_status_question(
                    questions[_nq_i], responses, conversation_state
                )
                next_question = questions[_nq_i]
                thank_you_message = "Thank you for sharing your email! 📧"

                next_question_text = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else next_question
                )
                response_msg = translate_text(
                    f"{thank_you_message} Now, let's move on to: {next_question_text}",
                    user_language,
                )

                if isinstance(next_question, dict) and "options" in next_question:
                    translated_options = [
                        translate_text(opt, user_language)
                        for opt in next_question["options"]
                    ]
                    return {
                        "response": response_msg,
                        "options": ", ".join(translated_options),
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }

                return {
                    "response": response_msg,
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }

            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)

            completion_msg = translate_text(
                "Thank you for sharing the details! 🎉 We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae",
                user_language,
            )
            return {
                "response": completion_msg,
                "final_responses": responses,
                "language": user_language,
                "language_code": get_language_code(user_language),
            }

        general_assistant_prompt = (
            f"The user entered '{user_message}'. Please assist them in {user_language}."
        )
        general_assistant_response = llm.invoke(
            [
                system_message_cls(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                ),
                human_message_cls(content=general_assistant_prompt),
            ]
        )
        return {
            "response": general_assistant_response.content.strip(),
            "question": f"Let's move back to: {question}",
            "example": "The email address should be in this format: example@gmail.com",
        }

    def handle_identity_questions(
        self,
        *,
        question: str,
        user_message: str,
        user_language: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        translate_text: Any,
        translate_to_english_for_storage: Any,
        format_response_in_language: Any,
        llm: Any,
        system_message_cls: Any,
        human_message_cls: Any,
    ) -> dict[str, Any] | None:
        if question != self._FULL_NAME_QUESTION:
            return None
        if not display_question_matches_current_index(
            questions, conversation_state, question
        ):
            return None

        check_prompt = f"The user has responded with: '{user_message}'. Is this a valid person's  name? Respond with 'Yes' or 'No'."
        llm_response = llm.invoke(
            [
                system_message_cls(
                    content="You are Insura, an AI assistant specialized in insurance-related tasks. Your task is to determine if the input provided by the user is a valid person's  name. Make sure it is a valid  name for a person."
                ),
                human_message_cls(content=check_prompt),
            ]
        )
        is_person_name = llm_response.content.strip().lower() == "yes"
        if is_person_name:
            responses[question] = translate_to_english_for_storage(user_message, user_language)
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if isinstance(next_question, dict):
                    return format_response_in_language(
                        f"Thank you for providing the name. Now, let's move on to: {next_question['question']}",
                        next_question.get("options", []),
                        user_language,
                    )
                return format_response_in_language(
                    f"Thank you for providing the name. Now, let's move on to: {next_question}",
                    [],
                    user_language,
                )
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            result = format_response_in_language(
                "Thank you for using Insura. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                [],
                user_language,
            )
            result["final_responses"] = responses
            return result

        general_assistant_response = llm.invoke(
            [
                system_message_cls(
                    content=f"You are Insura, an AI assistant created by CloudSubset. Respond in {user_language}. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately."
                ),
                human_message_cls(
                    content=f"The user entered '{user_message}', which does not appear to be a person's name. Please assist them in {user_language}."
                ),
            ]
        )
        return {
            "response": general_assistant_response.content.strip(),
            "question": translate_text(f"Let's move back to: {question}", user_language),
        }

