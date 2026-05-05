from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from services.chatbot.question_utils import display_question_matches_current_index

from langchain_core.messages import HumanMessage, SystemMessage

from services.chatbot.language_service import llm
from services.motor import motor_flow_service
from utils.question_helper import handle_emirate_question


class MotorConversationHandler:
    _USER_RESPONSES_FILE = "user_responses.json"
    _FINAL_MESSAGE = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
    _INSURA_COMPLETE_MESSAGE = "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!"
    _INSURA_SHORT_COMPLETE_MESSAGE = "Thank you for using Insura. Your request has been processed. Have a great day!"

    def _advance_or_finish(self, conversation_state: dict[str, Any], questions: list[Any], responses: dict[str, Any], success_message: str) -> dict[str, Any]:
        conversation_state["current_question_index"] += 1
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            next_text = (
                next_question["question"]
                if isinstance(next_question, dict)
                else next_question
            )
            return {"response": f"{success_message}{next_text}"}
        with open(self._USER_RESPONSES_FILE, "w") as file:
            json.dump(responses, file, indent=4)
        return {"response": self._FINAL_MESSAGE, "final_responses": responses}

    def _handle_fixed_options(
        self,
        *,
        question: str,
        user_message: str,
        valid_options: list[str],
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
    ) -> dict[str, Any]:
        if user_message in valid_options:
            responses[question] = user_message
            return self._advance_or_finish(
                conversation_state,
                questions,
                responses,
                "Thank you for your response. Now, let's move on to: ",
            )
        general_assistant_response = llm.invoke([HumanMessage(content=f"user response: {user_message}. Please assist.")])
        return {
            "response": general_assistant_response.content.strip(),
            "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
        }

    def _handle_year_question(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
    ) -> dict[str, Any]:
        if not display_question_matches_current_index(
            questions, conversation_state, question
        ):
            return {}
        try:
            year = int(user_message)
            current_year = datetime.now().year
            if 1886 <= year <= current_year:
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
                        "response": f"Thank you for providing the year. Now, let's move on to: {next_text}"
                    }
                with open(self._USER_RESPONSES_FILE, "w") as file:
                    json.dump(responses, file, indent=4)
                return {"response": self._INSURA_COMPLETE_MESSAGE, "final_responses": responses}
        except ValueError:
            pass
        general_assistant_response = llm.invoke(
            [
                SystemMessage(
                    content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                ),
                HumanMessage(content=f"The user entered '{user_message}' when asked for the year their car was made. Please assist."),
            ]
        )
        return {
            "response": general_assistant_response.content.strip(),
            "question": f"Let's revisit: {question}",
        }

    def can_handle_start_question(self, question: str) -> bool:
        return motor_flow_service.is_start_question(question)

    def handle_start_question(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        user_language: str,
    ) -> dict[str, Any]:
        return handle_emirate_question(
            question,
            user_message,
            conversation_state,
            questions,
            responses,
            user_language,
        )

    def handle_area_preference_question(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
    ) -> dict[str, Any] | None:
        target_question = "Which area you prefer for the vehicle repair? Please type the name of the area"
        if question != target_question:
            return None
        if not display_question_matches_current_index(
            questions, conversation_state, question
        ):
            return None

        emirate = (
            responses.get("In which emirate would you prefer your vehicle to be repaired?", "")
            .strip()
            .lower()
        )
        check_prompt = (
            f"The user has responded with: '{user_message}'. Determine if this is a valid area within the emirate '{emirate}'. "
            "Respond only with 'Yes' or 'No'."
        )
        llm_response = llm.invoke(
            [
                SystemMessage(
                    content=f"You are Insura, an AI assistant specialized in identifying the area based on the {emirate}. "
                    "Your task is to verify if the provided input is a valid area within the specified emirate."
                ),
                HumanMessage(content=check_prompt),
            ]
        )
        is_valid_area = llm_response.content.strip().lower() == "yes"
        if is_valid_area:
            responses[question] = user_message
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                        "options": ", ".join(next_question["options"]),
                    }
                return {
                    "response": f"Thank you for providing the area. Now, let's move on to: {next_question['question']}"
                }

            with open(self._USER_RESPONSES_FILE, "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                "final_responses": responses,
            }

        general_assistant_prompt = (
            f"The user entered '{user_message}', which was not validated as a valid area within the emirate '{emirate}' by Insura. "
            "Please assist them in correcting their input."
        )
        general_assistant_response = llm.invoke(
            [
                SystemMessage(
                    content="You are Insura, an AI assistant created by CloudSubset. "
                    "Your role is to assist users with their inquiries and guide them appropriately."
                ),
                HumanMessage(content=general_assistant_prompt),
            ]
        )
        return {
            "response": general_assistant_response.content.strip(),
            "question": f"Let's try again: {question}\n",
        }

    def handle_motor_question_set(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
    ) -> dict[str, Any] | None:
        if question == "How many years of driving experience do you have in the UAE?":
            return self._handle_fixed_options(
                question=question,
                user_message=user_message,
                valid_options=["0-1 year", "1-2 years", "2+ years"],
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
            )
        if question in {
            "Could you please let me know the year your car was made?",
            "Could you please provide the registration details? When was your car first registered?",
            "Could you please tell me the year your bike was made?",
            "Could you please provide the registration details? When was your bike first registered?",
        }:
            return self._handle_year_question(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
            )
        if question == "Do you have a No Claim certificate?":
            return self._handle_fixed_options(
                question=question,
                user_message=user_message,
                valid_options=["No", "1 Year", "2 Years", "3+ Years"],
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
            )
        if question == "Does your policy include agency repair?":
            result = self._handle_fixed_options(
                question=question,
                user_message=user_message,
                valid_options=["Yes", "No"],
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
            )
            if "final_responses" in result or "Please choose" in result.get("question", ""):
                return result
            if conversation_state["current_question_index"] <= len(questions) - 1:
                next_question = questions[conversation_state["current_question_index"]]
                if isinstance(next_question, dict) and "options" in next_question:
                    return {
                        "response": f"Thank you for your response. Now, let's move on to: {next_question['question']}",
                        "options": ", ".join(next_question["options"]),
                    }
            return result
        return None

    def handle_vehicle_identity_questions(
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
        config: dict[str, tuple[str, str, str, str]] = {
            "Let me know the make of the car": (
                "car make name",
                "car maker agency",
                "car make",
                "car make",
            ),
            "Now, let's gather some details about your bike. Let me know the make of the bike.": (
                "bike make name",
                "bike maker agency",
                "bike make",
                "bike make",
            ),
            "May I know the model number of your car, please?": (
                "car model number",
                "car maker agency",
                "car model number",
                "car model number",
            ),
            "Could you please tell me the model number of your bike": (
                "bike model number",
                "bike maker agency",
                "bike model number",
                "bike model number",
            ),
            "May I know the variant of your car, please?": (
                "car variant",
                "car maker agency",
                "car variant",
                "car variant",
            ),
            "Could you please tell me the Variant of your bike": (
                "bike variant",
                "bike maker agency",
                "bike variant",
                "bike variant",
            ),
        }
        if question not in config:
            return None
        if not display_question_matches_current_index(
            questions, conversation_state, question
        ):
            return None

        descriptor, agency, success_label, fail_label = config[question]
        check_prompt = (
            f"The user has responded with: '{user_message}'. Determine if this is a valid {descriptor}. "
            "Respond only with 'Yes' or 'No'."
        )
        llm_response = llm.invoke(
            [
                system_message_cls(
                    content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                    f"Your task is to act as a {agency}. Check if the given input is a valid {descriptor}."
                ),
                human_message_cls(content=check_prompt),
            ]
        )
        is_valid = llm_response.content.strip().lower() == "yes"
        if is_valid:
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
                    "response": f"Thank you for providing the {success_label}. Now, let's move on to: {next_text}"
                }
            with open(self._USER_RESPONSES_FILE, "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": self._INSURA_SHORT_COMPLETE_MESSAGE,
                "final_responses": responses,
            }

        general_assistant_response = llm.invoke(
            [
                system_message_cls(
                    content="You are Insura, an AI assistant created by CloudSubset. "
                    "Your role is to assist users with their inquiries and guide them appropriately."
                ),
                human_message_cls(
                    content=f"The user entered '{user_message}', which was not validated as a {fail_label} by Insura. Please assist them in correcting their input."
                ),
            ]
        )
        return {
            "response": general_assistant_response.content.strip(),
            "question": f"Let's try again: {question}",
        }

