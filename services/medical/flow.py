from __future__ import annotations

from typing import Any

from services.chatbot.question_steps import (
    STEP_ADDITIONAL_MEMBER_RELATIONSHIP,
    STEP_ADDITIONAL_MEMBER_RESIDENCY,
    STEP_EMIRATE_CHOICE_ADDITIONAL_MEMBER,
    STEP_MARITAL_STATUS_MEMBER,
    STEP_MEDICAL_ADD_ANOTHER_MEMBER,
)

MEDICAL_FLOWS = frozenset({"medical", "medical_insurance"})

# Individual medical: visa-issuing emirate (displayed after passkey). Must match questions JSON.
MEDICAL_VISA_ISSUED_QUESTION = (
    "Great! Now let me get the best medical insurance quote for you!\n\n"
    "Choose your Visa issued by Emirates."
)

# Plan-type step — must match ``questions/medical/questions.json`` (options + question text).
MEDICAL_PLAN_TYPE_QUESTION = "What type of plan are you looking for?"
MEDICAL_PLAN_TYPE_OPTIONS: tuple[str, ...] = (
    "Basic Plan",
    "Enhanced Plan",
    "Enhanced Plan Standalone",
    "Flexi Plan",
    "Group Medical",
)

MEDICAL_MONTHLY_SALARY_QUESTION = "Please tell me your monthly salary?"
MEDICAL_MONTHLY_SALARY_OPTIONS: tuple[str, ...] = (
    "Below 4000",
    "Above 4000",
)

MEDICAL_SPONSOR_MOBILE_QUESTION = "May I have your mobile number?"

MEDICAL_SPONSOR_EMAIL_QUESTION = "Please share your Email Address"

# Member name from Emirates ID (matches keys used in ``processor`` upload handlers).
MEDICAL_MEMBER_NAME_RESPONSE_KEY = (
    "Next, we need the details of the member for whom the policy is being purchased. "
    "Please provide Name"
)

MEDICAL_ADD_ANOTHER_QUESTION = (
    "Would you like to add another member to your policy?"
)

ADDITIONAL_MEMBER_RELATIONSHIP_QUESTION = (
    "Could you kindly share the relationship of the member you'd like to add?"
)

MEDICAL_EMIRATE_CHOICE_ADDITIONAL_QUESTION = (
    "Great! Now, Next Step. We need the investor details. Please upload their Emirates "
    "ID or enter the information manually?"
)

MEDICAL_ADDITIONAL_MEMBER_NAME_RESPONSE_KEY = (
    "Next, we need the details of this member. Please provide Name"
)

MEDICAL_ADDITIONAL_DOB_QUESTION = "Additional member: Date of Birth (DOB)"

MEDICAL_ADDITIONAL_GENDER_QUESTION = "Additional member: Please confirm gender (Male or Female)"

MEDICAL_PRIMARY_RESIDENCY_QUESTION = "Kindly confirm your current Residency Status?"

MEDICAL_ADDITIONAL_RESIDENCY_QUESTION = (
    "Kindly confirm this member's current Residency Status?"
)

MEDICAL_REPURCHASE_QUESTION = "Would you like to purchase our insurance again?"

MEDICAL_RELATIONSHIP_OPTION_KEYS = (
    "Investor",
    "Employee",
    "Spouse",
    "Child",
    "4th Child",
    "Parent",
    "Domestic",
)

# Template for marital status — ``{name}`` filled from document / manual capture.
MEDICAL_MEMBER_MARITAL_STATUS_TEMPLATE = "Please confirm the marital status of {name}"

CONV_STATE_MEMBER_NAME_KEY = "medical_member_name_response_key"


def medical_member_marital_question_text(
    responses: dict[str, Any],
    *,
    conversation_state: dict[str, Any] | None = None,
) -> str:
    name_key = MEDICAL_MEMBER_NAME_RESPONSE_KEY
    if conversation_state:
        name_key = conversation_state.get(CONV_STATE_MEMBER_NAME_KEY) or MEDICAL_MEMBER_NAME_RESPONSE_KEY
    name = str(responses.get(name_key) or "").strip()

    if not name:
        # Fallback: derive name from stored upload payloads.
        upload_payloads: list[dict[str, Any]] = []
        for value in responses.values():
            if isinstance(value, dict):
                upload_payloads.append(value)
        first_upload = responses.get("first_document_upload")
        if isinstance(first_upload, dict):
            upload_payloads.append(first_upload)

        for payload in upload_payloads:
            for key in ("full_name", "fullName", "name"):
                v = payload.get(key)
                if isinstance(v, str) and v.strip():
                    name = v.strip()
                    break
            if name:
                break
            first_name = payload.get("first_name")
            last_name = payload.get("last_name")
            if isinstance(first_name, str) and first_name.strip():
                if isinstance(last_name, str) and last_name.strip():
                    name = f"{first_name.strip()} {last_name.strip()}"
                else:
                    name = first_name.strip()
                break

    if not name:
        name = "the member"
    return MEDICAL_MEMBER_MARITAL_STATUS_TEMPLATE.format(name=name)


def patch_medical_marital_status_question(
    question_data: Any,
    responses: dict[str, Any],
    conversation_state: dict[str, Any] | None = None,
) -> Any:
    """Replace ``{name}`` in the marital-status prompt when advancing in medical flow."""
    if isinstance(question_data, dict):
        q = question_data.get("question", "")
        if "{name}" not in q:
            return question_data
        formatted = medical_member_marital_question_text(
            responses, conversation_state=conversation_state
        )
        return {**question_data, "question": formatted}
    if isinstance(question_data, str) and "{name}" in question_data:
        return medical_member_marital_question_text(
            responses, conversation_state=conversation_state
        )
    return question_data


def medical_member_identity_keys(
    conversation_state: dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Response dict keys for the active primary/additional member identity capture."""
    name_key = MEDICAL_MEMBER_NAME_RESPONSE_KEY
    if conversation_state:
        name_key = (
            conversation_state.get(CONV_STATE_MEMBER_NAME_KEY) or MEDICAL_MEMBER_NAME_RESPONSE_KEY
        )
    if name_key == MEDICAL_ADDITIONAL_MEMBER_NAME_RESPONSE_KEY:
        return (
            MEDICAL_ADDITIONAL_MEMBER_NAME_RESPONSE_KEY,
            MEDICAL_ADDITIONAL_DOB_QUESTION,
            MEDICAL_ADDITIONAL_GENDER_QUESTION,
        )
    return (
        MEDICAL_MEMBER_NAME_RESPONSE_KEY,
        "Date of Birth (DOB)",
        "Please confirm this gender of",
    )


def medical_additional_member_cycle_questions() -> list[dict[str, Any]]:
    """Inserted after user chooses to add another member (single iteration)."""
    opts = list(MEDICAL_RELATIONSHIP_OPTION_KEYS)
    return [
        {
            "step_id": STEP_ADDITIONAL_MEMBER_RELATIONSHIP,
            "question": ADDITIONAL_MEMBER_RELATIONSHIP_QUESTION,
            "options": opts,
        },
        {
            "step_id": STEP_EMIRATE_CHOICE_ADDITIONAL_MEMBER,
            "question": MEDICAL_EMIRATE_CHOICE_ADDITIONAL_QUESTION,
            "options": ["Yes", "No"],
        },
        {
            "step_id": STEP_MARITAL_STATUS_MEMBER,
            "question": MEDICAL_MEMBER_MARITAL_STATUS_TEMPLATE,
            "options": ["Single", "Married"],
        },
        {
            "step_id": STEP_ADDITIONAL_MEMBER_RESIDENCY,
            "question": MEDICAL_ADDITIONAL_RESIDENCY_QUESTION,
            "options": opts,
        },
        {
            "step_id": STEP_MEDICAL_ADD_ANOTHER_MEMBER,
            "question": MEDICAL_ADD_ANOTHER_QUESTION,
            "options": ["Yes", "No"],
        },
    ]


def append_additional_medical_member_row(responses: dict[str, Any]) -> None:
    """Snapshot scratch keys for one additional member into ``medical_additional_members``."""
    name = str(responses.get(MEDICAL_ADDITIONAL_MEMBER_NAME_RESPONSE_KEY) or "").strip()
    marital_q = MEDICAL_MEMBER_MARITAL_STATUS_TEMPLATE.format(name=name)
    row = {
        "name": name,
        "dob": responses.get(MEDICAL_ADDITIONAL_DOB_QUESTION, ""),
        "gender": responses.get(MEDICAL_ADDITIONAL_GENDER_QUESTION, ""),
        "marital_status": responses.get(marital_q, ""),
        "relation": responses.get(MEDICAL_ADDITIONAL_RESIDENCY_QUESTION, ""),
        "relationship_to_primary": responses.get(
            ADDITIONAL_MEMBER_RELATIONSHIP_QUESTION, ""
        ),
    }
    responses.setdefault("medical_additional_members", []).append(row)
    for k in (
        MEDICAL_ADDITIONAL_MEMBER_NAME_RESPONSE_KEY,
        MEDICAL_ADDITIONAL_DOB_QUESTION,
        MEDICAL_ADDITIONAL_GENDER_QUESTION,
        MEDICAL_ADDITIONAL_RESIDENCY_QUESTION,
        ADDITIONAL_MEMBER_RELATIONSHIP_QUESTION,
        marital_q,
    ):
        responses.pop(k, None)


def medical_marital_status_answer_from_responses(
    responses_dict: dict[str, Any],
) -> str:
    """Read marital answer from ``responses`` for API payloads."""
    prefix = MEDICAL_MEMBER_MARITAL_STATUS_TEMPLATE.split("{name}", 1)[0]
    for k, v in responses_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            return str(v)
    return ""


# Medical flow questions that require emirate option validation.
MEDICAL_START_OPTIONS = (
    MEDICAL_VISA_ISSUED_QUESTION,
    "Tell me your Emirate sponsor located in?",
)


class MedicalFlowService:
    def __init__(self) -> None:
        self._flows = MEDICAL_FLOWS
        self._start_options = set(MEDICAL_START_OPTIONS)

    def is_flow(self, flow_name: str) -> bool:
        return flow_name in self._flows

    def is_start_question(self, question: str) -> bool:
        return question in self._start_options

    def get_entry_response(self, medical_questions: list[Any]) -> tuple[str, list[str]]:
        first_question = medical_questions[0]
        if isinstance(first_question, dict):
            return f"Great choice! {first_question['question']}", first_question.get(
                "options", []
            )
        return f"Great choice! {first_question}", []


medical_flow_service = MedicalFlowService()


def is_medical_flow(flow_name: str) -> bool:
    return medical_flow_service.is_flow(flow_name)


def get_medical_entry_response(medical_questions: list[Any]) -> tuple[str, list[str]]:
    return medical_flow_service.get_entry_response(medical_questions)
