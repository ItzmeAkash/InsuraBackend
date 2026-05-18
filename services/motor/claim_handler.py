from __future__ import annotations

import json
import re
from typing import Any

from services.chatbot.language_service import format_response_in_language, translate_text
from services.chatbot.question_steps import STEP_MOTOR_CLAIM_REPAIR_WORKSHOP
from services.chatbot.question_utils import (
    display_question_matches_current_index,
    resolve_step_id,
)
from services.claim.api_submission import claim_flow_try_upload_from_message
from services.claim.flow import repair_workshop_paged_options


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

    def _is_claim_multipart_upload_payload(self, user_message: str) -> bool:
        raw = user_message.strip()
        if not raw.startswith("{"):
            return False
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and bool(payload.get("claim_document_upload"))

    def _format_next_claim_step_response(
        self,
        *,
        success_prefix: str,
        next_question: Any,
        conversation_state: dict[str, Any],
        user_language: str,
    ) -> dict[str, Any]:
        next_text = (
            next_question["question"]
            if isinstance(next_question, dict)
            else str(next_question)
        )
        next_opts: list[str] = (
            list(next_question.get("options", []))
            if isinstance(next_question, dict)
            else []
        )
        step_id = (
            resolve_step_id(next_question) if isinstance(next_question, dict) else ""
        )
        if step_id == STEP_MOTOR_CLAIM_REPAIR_WORKSHOP:
            conversation_state["motor_claim_repair_page"] = 0
            next_opts = repair_workshop_paged_options(next_opts, 0)

        thank = translate_text(success_prefix.strip(), user_language)
        question_tr = translate_text(next_text, user_language)
        result = format_response_in_language(thank, next_opts, user_language)
        result["question"] = question_tr
        if thank and question_tr:
            result["response"] = f"{thank}\n\n{question_tr}"
        return result

    def _handle_upload_question(
        self,
        *,
        question: str,
        user_message: str,
        conversation_state: dict[str, Any],
        questions: list[Any],
        responses: dict[str, Any],
        success_prefix: str,
        user_language: str = "English",
        file_path: str = "",
        user_id: str = "",
    ) -> dict[str, Any] | None:
        if not display_question_matches_current_index(
            questions, conversation_state, question
        ):
            return None
        is_ref = self._is_acceptable_upload_reference(user_message)
        is_success_text = self._is_upload_success_message(user_message)
        is_structured_payload = self._is_structured_upload_payload(user_message)
        is_claim_payload = self._is_claim_multipart_upload_payload(user_message)
        if (
            not is_ref
            and not is_success_text
            and not is_structured_payload
            and not is_claim_payload
        ):
            return {"response": self._INVALID_FORMAT_MESSAGE}

        responses[question] = (
            self._normalized_upload_reference(user_message)
            if is_ref
            else user_message.strip()
        )
        step_idx = conversation_state["current_question_index"]
        current_q = questions[step_idx] if step_idx < len(questions) else None
        step_id = resolve_step_id(current_q) if isinstance(current_q, dict) else ""
        claim_flow_try_upload_from_message(
            current_flow=str(conversation_state.get("current_flow", "")),
            responses=responses,
            user_message=user_message,
            step_id=step_id,
            file_path=file_path,
            user_id=user_id or str(conversation_state.get("user_id", "")),
        )
        conversation_state["current_question_index"] += 1
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            return self._format_next_claim_step_response(
                success_prefix=success_prefix,
                next_question=next_question,
                conversation_state=conversation_state,
                user_language=user_language,
            )

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
        user_language: str = "English",
        file_path: str = "",
        user_id: str = "",
    ) -> dict[str, Any] | None:
        mapping: dict[str, str] = {
            "Upload Required Documents\n\nPlease upload your Vehicle Registration Card (Mulkiya).": (
                "Thank you for uploading your Vehicle Registration Card."
            ),
            "Please upload your valid driving license.": (
                "Thank you for uploading your driving license."
            ),
            "Please upload your Emirates ID.": (
                "Thank you for uploading your Emirates ID."
            ),
            "Please upload police verification documents related to the incident.": (
                "Thank you for uploading the police verification documents."
            ),
        }
        if question not in mapping:
            return None
        return self._handle_upload_question(
            question=question,
            user_message=user_message,
            conversation_state=conversation_state,
            questions=questions,
            responses=responses,
            success_prefix=mapping[question],
            user_language=user_language,
            file_path=file_path,
            user_id=user_id,
        )
