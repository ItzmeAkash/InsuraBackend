from __future__ import annotations

import json
from typing import Any

from services.chatbot.constants import (
    MEDICAL_CUSTOMER_PLAN_SUCCESS_RESPONSE,
    WEHBE_GOOGLE_REVIEW_URL,
    WEHBE_REVIEW_INVITE_MESSAGE,
)
from services.chatbot.option_handlers import handle_option_validation_multilingual
from services.chatbot.question_utils import display_question_matches_current_index
from services.medical import medical_flow_service


class MedicalConversationHandler:
    _ADVISOR_CODE_QUESTION = (
        "Please enter your Insurance Advisor code for assigning your enquiry for further assistance"
    )
    _EMIRATE_OPTIONS = [
        "Abu Dhabi",
        "Ajman",
        "Dubai",
        "Fujairah",
        "Ras Al Khaimah",
        "Sharjah",
        "Umm Al Quwain",
    ]

    def can_handle_start_question(self, question: str) -> bool:
        return medical_flow_service.is_start_question(question)

    def handle_start_question(
        self,
        *,
        user_message: str,
        question: str,
        user_language: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        user_id: str,
    ) -> dict[str, Any]:
        return handle_option_validation_multilingual(
            user_message,
            self._EMIRATE_OPTIONS,
            question,
            user_language,
            conversation_state,
            questions,
            responses,
            user_id,
        )

    def handle_medical_question_set(
        self,
        *,
        question: str,
        user_message: str,
        user_id: str,
        user_language: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        user_states: dict[str, Any],
        valid_adivisor_code: Any,
        fetching_medical_detail: Any,
        translate_text: Any,
        get_language_code: Any,
        format_response_in_language: Any,
        llm: Any,
        SystemMessage: Any,
        HumanMessage: Any,
        insurance_lab_base_url: str,
    ) -> dict[str, Any] | None:
        if question != self._ADVISOR_CODE_QUESTION:
            return None

        if valid_adivisor_code(user_message):
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                response_message = f"Thank you! Now, let's move on to: {next_question}"
                return format_response_in_language(response_message, [], user_language)

            try:
                medical_detail_response = fetching_medical_detail(responses)
                if user_id in user_states:
                    del user_states[user_id]

                if isinstance(medical_detail_response, int):
                    success_message = MEDICAL_CUSTOMER_PLAN_SUCCESS_RESPONSE
                    review_message = WEHBE_REVIEW_INVITE_MESSAGE
                    translated_success = translate_text(success_message, user_language)
                    translated_review = translate_text(review_message, user_language)
                    customer_plan_link = (
                        f"{insurance_lab_base_url}/customer_plan/{medical_detail_response}"
                    )

                    saved_language = conversation_state.get("preferred_language", "English")
                    saved_language_code = conversation_state.get("language_code", "en")
                    saved_language_explicitly_set = conversation_state.get(
                        "language_explicitly_set", False
                    )

                    user_states[user_id] = {
                        "current_question_index": 0,
                        "responses": {},
                        "current_flow": "initial",
                        "welcome_shown": False,
                        "awaiting_document_name": False,
                        "document_name": "",
                        "preferred_language": saved_language,
                        "language_code": saved_language_code,
                        "language_explicitly_set": saved_language_explicitly_set,
                    }
                    return {
                        "response": translated_success,
                        "link": customer_plan_link,
                        "review_message": translated_review,
                        "review_link": WEHBE_GOOGLE_REVIEW_URL,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                        "restart_conversation": True,
                    }

                fallback_message = "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae."
                return {
                    "response": translate_text(fallback_message, user_language),
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }
            except Exception as exc:
                error_message = f"An error occurred while fetching medical details: {exc}"
                return {
                    "response": translate_text(error_message, user_language),
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }

            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                no_agent_message = "Since you don't have an agent code, we will arrange a callback from the next available agent to assist you further. Thank you!"
                return {
                    "response": translate_text(no_agent_message, user_language),
                    "final_responses": responses,
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }
            except Exception as exc:
                save_error_message = f"An error occurred while saving your responses: {exc}"
                return {
                    "response": translate_text(save_error_message, user_language),
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }

        general_assistant_prompt = (
            f"user response: {user_message}. Please assist in {user_language}."
        )
        general_assistant_response = llm.invoke(
            [
                SystemMessage(
                    content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                ),
                HumanMessage(content=general_assistant_prompt),
            ]
        )
        example_message = "The Advisor code should be a 4-digit numeric value. Please enter a valid code"
        retry_question = translate_text(f"Let's try again: {question}", user_language)
        return {
            "response": general_assistant_response.content.strip(),
            "example": translate_text(example_message, user_language),
            "question": retry_question,
            "language": user_language,
            "language_code": get_language_code(user_language),
        }

    def handle_medical_dynamic_questions(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        llm: Any,
        system_message_cls: Any,
        human_message_cls: Any,
    ) -> dict[str, Any] | None:
        if question == "Could you kindly share your contact details with me? To start, may I know your name, please?":
            if not display_question_matches_current_index(
                questions, conversation_state, question
            ):
                return None
            check_prompt = f"The user has responded with: '{user_message}'. Is this a valid person's name? Respond with 'Yes' or 'No'."
            llm_response = llm.invoke(
                [
                    system_message_cls(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. Your task is to determine if the input provided by the user is a valid person's name.Make sure it a valide name for a person"
                    ),
                    human_message_cls(content=check_prompt),
                ]
            )
            if llm_response.content.strip().lower() == "yes":
                responses[question] = user_message
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    next_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )
                    return {
                        "response": f"Thank you for providing your name. Now, let's move on to: {next_text}"
                    }
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "Thank you for using Insura. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                    "final_responses": responses,
                }
            general_assistant_response = llm.invoke(
                [
                    system_message_cls(
                        content="You are Insura, an AI assistant created by CloudSubset. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately."
                    ),
                    human_message_cls(
                        content=f"The user entered '{user_message}', which does not appear to be a person's name. Please assist."
                    ),
                ]
            )
            return {
                "response": general_assistant_response.content.strip(),
                "question": f"Let's move back to: {question}",
            }

        if question == "Could you kindly provide me with the sponsor's Source of Income":
            valid_options = ["Business", "Salary"]
            if user_message in valid_options:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    if isinstance(next_question, dict) and "options" in next_question:
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                            "options": ", ".join(next_question["options"]),
                        }
                    return {"response": f"Thank you. Now, let's move on to: {next_question}"}
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
            general_assistant_response = llm.invoke(
                [human_message_cls(content=f"user response: {user_message}. Please assist.")]
            )
            return {
                "response": general_assistant_response.content.strip(),
                "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
            }

        if question == "Are you suffering from any pre-existing or chronic conditions?":
            if user_message not in ["Yes", "No"]:
                return None
            responses[question] = user_message
            if user_message == "No":
                if "Please provide us with the details of your Chronic Conditions Medical Report" in questions:
                    questions.remove("Please provide us with the details of your Chronic Conditions Medical Report")
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                        "options": ", ".join(next_question["options"]),
                    }
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                    "final_responses": responses,
                }
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                return {
                    "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                    "options": ", ".join(next_question["options"]),
                }
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                "final_responses": responses,
            }

        return None

    def handle_vaccination_questions(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        valid_date_format: Any,
    ) -> dict[str, Any] | None:
        if question == "Have you been vaccinated for Covid-19?":
            if user_message not in ["Yes", "No"]:
                return None
            responses[question] = user_message
            if user_message == "Yes":
                if "Can you please tell me the date of your first dose?" not in questions:
                    responses["Can you please tell me the date of your first dose?"] = None
                    questions.insert(conversation_state["current_question_index"] + 1, "Can you please tell me the date of your first dose?")
                if "Can you please tell me the date of your second dose?" not in questions:
                    responses["Can you please tell me the date of your second dose?"] = None
                    questions.insert(conversation_state["current_question_index"] + 2, "Can you please tell me the date of your second dose?")
                conversation_state["current_question_index"] += 1
                next_question = questions[conversation_state["current_question_index"]]
                return {"response": f"Thank you! Now, let's move on to: {next_question}"}

            if "Can you please tell me the date of your first dose?" in questions:
                questions.remove("Can you please tell me the date of your first dose?")
            if "Can you please tell me the date of your second dose?" in questions:
                questions.remove("Can you please tell me the date of your second dose?")
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                return {
                    "response": f"Thank you for your response. Now, let's move on to: {next_question['question']}",
                    "options": ", ".join(next_question["options"]),
                }
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                "final_responses": responses,
            }

        if question in {"Can you please tell me the date of your first dose?", "Can you please tell me the date of your second dose?"}:
            if not valid_date_format(user_message):
                return {
                    "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY or MM-DD-YYYY."
                }
            responses[question] = user_message
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if question == "Can you please tell me the date of your second dose?":
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                        "options": ", ".join(next_question["options"]),
                    }
                return {"response": f"Thank you! Now, let's move on to: {next_question}"}

            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                "final_responses": responses,
            }

        return None

    def handle_current_policy_company_question(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        llm: Any,
        system_message_cls: Any,
        human_message_cls: Any,
    ) -> dict[str, Any] | None:
        if question != "Which insurance company is your current policy with?":
            return None
        if not display_question_matches_current_index(
            questions, conversation_state, question
        ):
            return None

        check_prompt = (
            f"This is the company name: '{user_message}'. Please check if that name could be a company name and respond with 'Yes' or 'No'"
        )
        llm_response = llm.invoke(
            [
                system_message_cls(
                    content="You are a friendly assistant working in Isuran's company department. Your primary task is to verify the user provided input could be a company name. The input might include examples such as 'Fallout Private Limited' or 'Fallout Technologies'. Your role is to validate and identify whether the given input is a valid company name "
                ),
                human_message_cls(content=check_prompt),
            ]
        )
        if llm_response.content.strip().lower() == "yes":
            responses[question] = user_message
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                next_text = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else next_question
                )
                return {
                    "response": f"Thank you for providing the company name. Now, let's move on to: {next_text}"
                }
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                "final_responses": responses,
            }

        general_assistant_response = llm.invoke(
            [
                system_message_cls(
                    content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                ),
                human_message_cls(content=f"User response: {user_message}. Please assist."),
            ]
        )
        return {
            "response": general_assistant_response.content.strip(),
            "question": f"Let's move back to: {question}",
        }

