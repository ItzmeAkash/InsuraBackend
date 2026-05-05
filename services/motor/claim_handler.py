from __future__ import annotations

import json
import re
from typing import Any

from services.chatbot.question_utils import display_question_matches_current_index


class MotorClaimHandler:
    _ALLOWED_EXT = re.compile(r"\.(pdf|docx|jpg|jpeg|png)$", re.IGNORECASE)
    _INVALID_FORMAT_MESSAGE = "The file format seems incorrect. Please upload a valid document."
    _COMPLETE_MESSAGE = "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!"

    def _normalized_upload_reference(self, user_message: str) -> str:
        s = user_message.strip().replace("\\", "/")
        while "//" in s:
            s = s.replace("//", "/")
        if s.startswith("./"):
            s = s[2:]
        return s

    def _is_acceptable_upload_reference(self, user_message: str) -> bool:
        """
        Accept stored paths like ``uploads/general/user_id/hash_file.name.pdf``.
        Older regex only allowed simple basename characters and rejected dots in
        filenames and Windows-style separators.
        """
        s = self._normalized_upload_reference(user_message)
        if ".." in s.split("/"):
            return False
        low = s.lower()
        if not low.startswith("uploads/"):
            return False
        return bool(self._ALLOWED_EXT.search(s))

    def _is_upload_success_message(self, user_message: str) -> bool:
        low = user_message.strip().lower()
        if not low:
            return False
        return (
            "upload successfully" in low
            or "uploaded successfully" in low
            or "file uploaded" in low
            or "document uploaded" in low
            or ("document upload" in low and "success" in low)
        )

    def _is_structured_upload_payload(self, user_message: str) -> bool:
        """
        Accept JSON OCR payloads returned by document extraction endpoints.
        Example Mulkiya payload contains keys like owner/traffic_plate_no/chassis_no.
        """
        raw = user_message.strip()
        if not raw.startswith("{"):
            return False
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict) or not payload:
            return False
        expected_keys = {
            "owner",
            "traffic_plate_no",
            "chassis_no",
            "engine_no",
            "policy_no",
            "expiry_date",
        }
        return bool(set(payload.keys()) & expected_keys)

    def _handle_upload_question(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        success_prefix: str,
        include_options: bool = False,
    ) -> dict[str, Any] | None:
        if not display_question_matches_current_index(
            questions, conversation_state, question
        ):
            return None
        is_ref = self._is_acceptable_upload_reference(user_message)
        is_success_text = self._is_upload_success_message(user_message)
        is_structured_payload = self._is_structured_upload_payload(user_message)
        if not is_ref and not is_success_text and not is_structured_payload:
            return {"response": self._INVALID_FORMAT_MESSAGE}

        responses[question] = (
            self._normalized_upload_reference(user_message)
            if is_ref
            else user_message.strip()
        )
        conversation_state["current_question_index"] += 1
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            next_text = (
                next_question["question"]
                if isinstance(next_question, dict)
                else next_question
            )
            if include_options and isinstance(next_question, dict):
                return {
                    "response": f"{success_prefix}{next_text}",
                    "options": ", ".join(next_question.get("options", [])),
                }
            return {"response": f"{success_prefix}{next_text}"}

        with open("user_responses.json", "w") as file:
            json.dump(responses, file, indent=4)
        return {"response": self._COMPLETE_MESSAGE, "final_responses": responses}

    def handle_claim_upload_questions(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
    ) -> dict[str, Any] | None:
        mapping: dict[str, tuple[str, bool]] = {
            "Upload Required Documents\n\nPlease upload your Vehicle Registration Card (Mulkiya).": (
                "Thank you for uploading your Vehicle Registration Card. Now, let's move on to: ",
                False,
            ),
            "Please upload your valid driving license.": (
                "Thank you for uploading your driving license. Now, let's move on to: ",
                False,
            ),
            "Please upload your Emirates ID.": (
                "Thank you for uploading your Emirates ID. Now, let's move on to: ",
                False,
            ),
            "Please upload police verification documents related to the incident.": (
                "Thank you for uploading the police verification documents. Now, let's move on to: ",
                False,
            ),
        }
        if question not in mapping:
            return None
        success_prefix, include_options = mapping[question]
        return self._handle_upload_question(
            question=question,
            user_message=user_message,
            conversation_state=conversation_state,
            questions=questions,
            responses=responses,
            success_prefix=success_prefix,
            include_options=include_options,
        )

