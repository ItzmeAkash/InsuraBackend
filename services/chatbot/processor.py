import asyncio
from datetime import datetime
from typing import Any
from utils.helper import (
    emaf_document,
    fetching_medical_detail,
    is_valid_mobile_number,
    valid_adivisor_code,
)
from utils.helper import (
    get_user_name,
    valid_date_format,
    valid_emirates_id,
    is_valid_name,
)
from utils.question_helper import (
    handle_adiviosr_code,
    handle_client_name_question,
    handle_date_question,
    handle_emaf_document,
    handle_emirate_upload_document,
    handle_emirate_upload_document_car_insurance,
    handle_validate_name,
    handle_vehicle_registration_car_question,
    handle_what_would_you_do_today_question,
    handle_yes_or_no,
)
from langchain_core.messages import HumanMessage, SystemMessage
from models.model import UserInput
from random import choice
import json
import re

from services.chatbot.constants import (
    MOTOR_DRIVING_LICENSE_COMBINED_UPLOAD_QUESTION,
    MOTOR_EMIRATES_ID_COMBINED_UPLOAD_QUESTION,
    PASSKEY_VALID_CODE,
    WEHBE_GOOGLE_REVIEW_URL,
    WEHBE_REVIEW_INVITE_MESSAGE,
)
from services.chatbot.emaf_company_options import (
    EMAF_COMPANY_NUMBER_BY_OPTION,
    EMAF_INSURANCE_VALID_OPTIONS,
)
from services.chatbot.question_steps import (
    STEP_ADDITIONAL_MEMBER_RELATIONSHIP,
    STEP_ADDITIONAL_MEMBER_RESIDENCY,
    STEP_ADVISOR_CODE_ENTRY,
    STEP_ADVISOR_YES_NO,
    STEP_CLAIM_TYPE_CHOICE,
    STEP_CLIENT_MOBILE,
    STEP_CLIENT_NAME,
    STEP_COMPREHENSIVE_COVER,
    STEP_CURRENCY_CHOICE,
    STEP_EMAF_COMPANY,
    STEP_EMAF_NAME,
    STEP_EMAF_PHONE,
    STEP_EMIRATE_CHOICE_ADDITIONAL_MEMBER,
    STEP_EMIRATE_CHOICE_CAR,
    STEP_EMIRATE_CHOICE_MEDICAL,
    STEP_EXCEL_MEDICAL_UPLOAD,
    STEP_GENDER_CONFIRM,
    STEP_IDS_VALIDATE_NAME,
    STEP_INSURANCE_EXPIRY_YEAR,
    STEP_MARITAL_STATUS_MEMBER,
    STEP_MEDICAL_ADD_ANOTHER_MEMBER,
    STEP_MEDICAL_CLAIM_ASSISTANCE_TYPE,
    STEP_MEDICAL_CLAIM_CONTACT_MOBILE,
    STEP_MEDICAL_CLAIM_EMIRATES_ID_UPLOAD,
    STEP_MEDICAL_CLAIM_INSURANCE_CARD_UPLOAD,
    STEP_MEDICAL_CLAIM_REPURCHASE_OFFER,
    STEP_MEDICAL_PLAN_TYPE,
    STEP_MEMBER_DOB,
    STEP_MEMBER_PURCHASE_NAME,
    STEP_MONTHLY_SALARY,
    STEP_MOTOR_CLAIM_CONTACT_MOBILE,
    STEP_MOTOR_CLAIM_RECOVERY_LOCATION,
    STEP_MOTOR_CLAIM_REPURCHASE_OFFER,
    STEP_MOTOR_CLAIM_REPAIR_WORKSHOP,
    STEP_MOTOR_CLAIM_ROAD_RECOVERY,
    STEP_MOTOR_COVER_TYPE,
    STEP_MOTOR_ENQUIRY_PHONE,
    STEP_IDS_MAIN_MENU,
    STEP_PASSKEY,
    STEP_POLICY_RENEWAL_CHOICE,
    STEP_SPONSOR_COMPANY,
    STEP_SPONSOR_EMIRATES_ID,
    STEP_SPONSOR_EMAIL,
    STEP_SPONSOR_MARITAL_STATUS,
    STEP_SPONSOR_MOBILE,
    STEP_SPONSOR_RELATIONSHIP,
    STEP_SPONSOR_TYPE,
    STEP_UPLOAD_DRIVING_LICENSE,
    STEP_UPLOAD_DRIVING_LICENSE_BACK,
    STEP_UPLOAD_DRIVING_LICENSE_FRONT,
    STEP_UPLOAD_EMIRATES_DOC,
    STEP_UPLOAD_TRADE_LICENSE,
    STEP_UPLOAD_EID_BACK,
    STEP_UPLOAD_EID_FRONT,
    STEP_UPLOAD_MULKIYA,
    STEP_UPLOAD_MULKIYA_BACK,
    STEP_UPLOAD_MULKIYA_FRONT,
    STEP_UPLOAD_VAT_CERTIFICATE,
    STEP_VEHICLE_REGISTRATION_TYPE,
    STEP_VEHICLE_TEST_CERT,
    STEP_VAT_CERTIFICATE_CHOICE,
)
from services.chatbot.question_utils import (
    display_question_matches_current_index,
    resolve_step_id,
)
from services.chatbot.conversation_reset import (
    is_reset_command,
    reset_to_initial_and_format_first_question,
)
from services.chatbot.initial_flow_routes import (
    InitialMenuContext,
    resolve_initial_menu_choice,
)
from services.chatbot.menu_flows.medical import get_medical_menu_entry_response
from services.chatbot.language_service import (
    detect_document_type_from_question,
    detect_language,
    format_response_in_language,
    get_language_code,
    llm,
    translate_text,
    translate_to_english_for_storage,
    validate_response_multilingual,
)
from services.chatbot.option_handlers import handle_option_validation_multilingual
from services.chatbot.conversation_handlers import (
    medical_conversation_handler,
    medical_individual_handler,
    motor_claim_handler,
    motor_conversation_handler,
)
from services.chatbot.flow_registry import ChatbotFlowRegistry
from services.claim.api_submission import (
    claim_flow_try_enquiry,
    claim_flow_try_first_step,
    claim_flow_try_upload_from_message,
    store_claim_mobile,
)
from services.claim.roadside_hotlines import format_motor_claim_roadside_hotline_message
from services.claim.flow import (
    CLAIM_ROUTER_FLOW,
    MEDICAL_CLAIM_FLOW,
    MOTOR_CLAIM_FLOW,
    repair_workshop_paged_options,
)
from services.general.api_submission import (
    general_flow_try_enquiry_after_type,
    general_flow_try_first_step,
    general_flow_try_upload_from_payload,
)
from services.motor.api_submission import (
    motor_flow_try_enquiry_after_phone,
    motor_flow_try_second_step_after_cover,
    motor_flow_try_third_step_after_contact,
    motor_flow_try_upload_document,
)
from services.upload_cleanup import wipe_flow_session_upload_files
from services.medical.flow import (
    CONV_STATE_MEMBER_NAME_KEY,
    MEDICAL_ADDITIONAL_MEMBER_NAME_RESPONSE_KEY,
    MEDICAL_MEMBER_NAME_RESPONSE_KEY,
    MEDICAL_MONTHLY_SALARY_OPTIONS,
    MEDICAL_PLAN_TYPE_OPTIONS,
    MEDICAL_RELATIONSHIP_OPTION_KEYS,
    MEDICAL_REPURCHASE_QUESTION,
    MEDICAL_VISA_ISSUED_QUESTION,
    append_additional_medical_member_row,
    medical_additional_member_cycle_questions,
    medical_member_identity_keys,
    patch_medical_marital_status_question,
)
from services.medical.medical_completion import respond_medical_quotation_complete
from services.general.flow import (
    GENERAL_DOC_FIELD_STRICT_WORKBOOK_VALIDATION,
    GENERAL_DOC_KIND_CREDIT_INSURANCE_MENU,
    GENERAL_DOC_KIND_GENERAL_TEMPLATE,
    GENERAL_DOC_KIND_GROUP_LIFE_MENU,
    GENERAL_DOC_KIND_PROPERTY_ALL_RISKS_MENU,
    GENERAL_DOC_MSG_ANYTHING_ELSE_DECLINED_SIGNOFF_EN,
    GENERAL_DOC_MSG_ASSISTANCE_ANYTHING_ELSE,
    GENERAL_DOC_MSG_COMPANY_EMAIL_INTRO,
    GENERAL_DOC_MSG_COMPANY_EMAIL_PROMPT,
    GENERAL_DOC_MSG_COMPANY_NAME,
    GENERAL_DOC_MSG_DESIGNATION,
    GENERAL_DOC_MSG_DETAILS_CLOSING,
    GENERAL_DOC_MSG_EMAIL_THANK,
    GENERAL_DOC_MSG_EXISTING_POLICY,
    GENERAL_DOC_MSG_FULL_NAME,
    GENERAL_DOC_MSG_PHONE_FOR_SPECIALIST,
    GENERAL_DOC_MSG_POLICY_SCHEDULE,
    GENERAL_DOC_MSG_POLICY_SCHEDULE_THANK,
    GENERAL_DOC_MSG_SPECIALIST_FORWARDED,
    GENERAL_DOC_MSG_TRADE_LICENCE_THANK,
    GENERAL_DOC_MSG_TRADE_LICENCE_UPLOAD,
    GENERAL_DOC_MSG_RETRY_FILLED_FORM_EN,
    GENERAL_DOC_MSG_TRAVEL_DOWNLOAD,
    GENERAL_DOC_MSG_TRAVEL_UPLOAD_RECEIVED,
    GENERAL_DOC_MSG_VAT_AVAILABLE,
    GENERAL_DOC_MSG_VAT_THANK,
    GENERAL_DOC_MSG_VAT_UPLOAD,
    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN,
    GENERAL_DOC_MSG_WRONG_UPLOAD_USE_FORM_NOT_TRADE_VAT_EN,
    GENERAL_FLOW_INSURANCE_CLUB_CLOSING,
    GENERAL_UPLOAD_DOCUMENT_CATEGORY,
    GENERAL_UPLOAD_TYPE_TRADE_LICENCE,
    GENERAL_UPLOAD_TYPE_TRAVEL_FORM,
    GENERAL_UPLOAD_TYPE_VAT_CERTIFICATE,
    GENERAL_DOC_OPT_CONNECT_SPECIALIST,
    GENERAL_DOC_OPT_WAIT,
    GENERAL_DOC_PHASE_AWAITING_COMPANY_EMAIL,
    GENERAL_DOC_PHASE_AWAITING_COMPANY_NAME,
    GENERAL_DOC_PHASE_AWAITING_DESIGNATION,
    GENERAL_DOC_PHASE_AWAITING_EXISTING_POLICY,
    GENERAL_DOC_PHASE_AWAITING_FULL_NAME,
    GENERAL_DOC_PHASE_AWAITING_POLICY_SCHEDULE_DATE,
    GENERAL_DOC_PHASE_AWAITING_SPECIALIST_CHOICE,
    GENERAL_DOC_PHASE_AWAITING_SPECIALIST_PHONE,
    GENERAL_DOC_PHASE_AFTER_WAIT_CLOSING,
    GENERAL_DOC_PHASE_AWAITING_ANYTHING_ELSE,
    GENERAL_DOC_PHASE_CREDIT_AWAITING_CHOICE,
    GENERAL_DOC_PHASE_PAR_AWAITING_CHOICE,
    GENERAL_DOC_PHASE_GLI_LEVEL1,
    GENERAL_DOC_PHASE_GLI_MALPRACTICE,
    GENERAL_DOC_PHASE_AWAITING_TRADE_LICENCE,
    GENERAL_DOC_PHASE_AWAITING_UPLOAD,
    GENERAL_DOC_PHASE_AWAITING_VAT_AVAILABLE,
    GENERAL_DOC_PHASE_AWAITING_VAT_CERTIFICATE,
    GENERAL_DOC_PHASE_DOWNLOAD_SHOWN,
    GENERAL_DOC_SPECIALIST_OPTIONS,
    GENERAL_DOC_YES_NO_OPTIONS,
    GENERAL_INSURANCE_MORE_OPTION,
    GENERAL_INSURANCE_OPTIONS,
    GENERAL_INSURANCE_PICK_PROMPT,
    PAR_FILE_PROPOSAL_FORM_DOCX,
    PAR_FILE_PROPERTY_INSURANCE_XLSX,
    PAR_FILE_SCOPE_DETAILS_XLSX,
    PAR_MSG_PICK_ONE,
    PAR_MSG_THANK_SELECTION,
    PAR_SUBMENU_OPTIONS,
    PAR_SUBMENU_PROPERTY_INSURANCES,
    PAR_SUBMENU_PROPOSAL_FORM,
    PAR_SUBMENU_SCOPE_DETAILS,
    STEP_GENERAL_INSURANCE_TYPE,
    CAR_PROPOSAL_FORM_LABEL,
    CAR_PROPOSAL_PDF,
    CONTRACT_ALL_RISKS_CAR_LABEL,
    CREDIT_FILE_CREDIT_INSURANCES_XLSX,
    CREDIT_FILE_SINGLE_RISK_DOC,
    CREDIT_INSURANCE_LABEL,
    CREDIT_SUBMENU_CREDIT_INSURANCES,
    CREDIT_SUBMENU_OPTIONS,
    CREDIT_SUBMENU_SINGLE_CREDIT,
    GLI_FILE_GLPA_CENSUS,
    GLI_FILE_GROUP_HEALTH,
    GLI_FILE_GROUP_TRAVEL,
    GLI_FILE_MALP_ESTABLISHMENTS,
    GLI_FILE_MMP_STAFF_LIST,
    GLI_LEVEL1_OPTIONS,
    GLI_MALP_PROPOSAL,
    GLI_MALP_STAFF_LIST,
    GLI_MALP_SUBMENU_OPTIONS,
    GLI_OPTION_ACCIDENT,
    GLI_OPTION_HEALTH,
    GLI_OPTION_MALPRACTICING,
    GLI_OPTION_TRAVEL,
    GROUP_LIFE_INSURANCE_LABEL,
    INDIVIDUAL_TRAVEL_LABEL,
    MARINE_CARGO_INSURANCE_LABEL,
    MARINE_CARGO_PROPOSAL_PDF,
    HAULIERS_LIABILITY_INSURANCE_LABEL,
    HAULIERS_LIABILITY_POLICY_DOC,
    BOND_PROPOSAL_LABEL,
    BOND_REQUIREMENTS_AND_FORMS_DOCX,
    DRONE_PROPOSAL_FORM_LABEL,
    DRONE_INSURANCE_XLSX,
    FIDELITY_PROPOSAL_LABEL,
    FIDELITY_PROPOSAL_DOC,
    FIRE_FIGHTING_FORM_LABEL,
    FIRE_FIGHTING_FACILITIES_DOCX,
    JEWELLER_PROPOSAL_LABEL,
    JEWELLERS_BLOCK_PROPOSAL_FORM_DOC,
    MONEY_INSURANCES_LABEL,
    MONEY_INSURANCE_XLSX,
    VEHICLE_DETAIL_FORM_LABEL,
    VEHICLE_DETAIL_FLEET_FORMAT_XLS,
    PI_MISC_ANNUAL_DOCX,
    PROFESSIONAL_INDEMNITY_LABEL,
    PROPERTY_ALL_RISKS_LABEL,
    THIRD_PARTY_LABEL,
    THIRD_PARTY_PROPOSAL_PDF,
    TRAVEL_INSURANCE_LABEL,
    TRAVEL_INSURANCE_XLSX,
    WORKMEN_COMPENSATION_LABEL,
    WORKMEN_COMPENSATION_XLSX,
    general_document_download_url,
    get_general_options_page,
    match_yes_no_english,
    parse_general_upload_payload,
    normalized_upload_payload_file_type,
    should_accept_general_template_form_upload,
    should_accept_trade_licence_upload,
    should_accept_travel_form_upload,
    should_accept_vat_certificate_upload,
    validate_any_nonempty_upload_payload,
    validate_travel_insurance_completed_payload,
)
from services.chatbot.question_store import (
    bike_questions,
    car_questions,
    claim_router_questions,
    existing_policy_questions,
    general_insurance_questions,
    greeting_templates,
    initial_questions,
    individual_questions,
    medical_questions,
    medical_claim,
    motor_claim,
    motor_insurance_questions,
    peek_last_upload_relative_path,
    refresh_medical_questions_if_changed,
    user_states,
)

# API Configuration - Base URLs for InsuranceLab
INSURANCE_LAB_BASE_URL = "https://insurancelab.ae"
INSURANCE_LAB_API_BASE_URL = f"{INSURANCE_LAB_BASE_URL}/Api"
INSURANCE_LAB_SME_ADD_API = f"{INSURANCE_LAB_API_BASE_URL}/sme_add/"
INSURANCE_LAB_SME_PLAN_BASE = f"{INSURANCE_LAB_BASE_URL}/sme_plan"

# Older builds stored these kinds before unifying on GENERAL_DOC_KIND_GENERAL_TEMPLATE.
_LEGACY_GENERAL_DOC_WORKBOOK_KINDS = frozenset(
    {"travel_insurance", "workmen_compensation"}
)

FLOW_REGISTRY = ChatbotFlowRegistry(
    initial_questions=initial_questions,
    medical_questions=medical_questions,
    individual_questions=individual_questions,
    motor_insurance_questions=motor_insurance_questions,
    car_questions=car_questions,
    bike_questions=bike_questions,
    existing_policy_questions=existing_policy_questions,
    motor_claim_questions=motor_claim,
    medical_claim_questions=medical_claim,
    claim_router_questions=claim_router_questions,
    general_insurance_questions=general_insurance_questions,
)


def _parse_vehicle_insurance_expiry_date(raw_date):
    """Parse mulkiya insurance expiry from common OCR date formats."""
    if not raw_date:
        return None
    if not isinstance(raw_date, str):
        raw_date = str(raw_date)

    normalized = raw_date.strip()
    if not normalized:
        return None

    date_formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d-%b-%y",
        "%d-%b-%Y",
        "%d-%B-%y",
        "%d-%B-%Y",
        "%d/%b/%y",
        "%d/%b/%Y",
        "%d/%B/%y",
        "%d/%B/%Y",
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    return None


def _is_motor_renewal_company_flow(responses):
    has_renewal = any(
        isinstance(v, str) and v.strip().lower() == "renewal"
        for v in responses.values()
    )
    has_company = any(
        isinstance(v, str) and v.strip().lower() == "company (business)"
        for v in responses.values()
    )
    return has_renewal and has_company


def _looks_like_pdf_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value.strip().replace("\\", "/")
    if not raw:
        return False
    lowered = raw.lower().split("?", 1)[0].split("#", 1)[0]
    return lowered.endswith(".pdf")


def _mulkiya_upload_contains_both_sides(
    document_data: dict[str, Any],
    file_path: str = "",
    raw_message: str = "",
    user_id: str = "",
) -> bool:
    """Best-effort detection for combined Mulkiya uploads.

    We skip the extra back-side prompt when the upload is clearly a PDF / multi-page
    document or when the payload explicitly says it contains multiple pages.
    """
    stored_path = peek_last_upload_relative_path(user_id, "mulkiya") if user_id else ""
    candidates: list[Any] = [file_path, raw_message, stored_path]
    for key in (
        "document_stored_path",
        "stored_relative_path",
        "upload_relative_path",
        "file_path",
        "document_path",
        "local_path",
        "document_url",
        "pdf_url",
        "file_url",
        "mulkiya_pdf_url",
        "_file_reference",
    ):
        candidates.append(document_data.get(key))

    for sub_key in ("document", "file", "pdf", "source", "mulkiya_file", "attachment"):
        sub = document_data.get(sub_key)
        if isinstance(sub, dict):
            for nested_key in (
                "url",
                "href",
                "src",
                "path",
                "local_path",
                "stored_path",
                "document_url",
                "pdf_url",
                "file_path",
            ):
                candidates.append(sub.get(nested_key))
        else:
            candidates.append(sub)

    if any(_looks_like_pdf_reference(candidate) for candidate in candidates):
        return True

    for count_key in ("page_count", "pages_count", "total_pages"):
        count_value = document_data.get(count_key)
        if isinstance(count_value, (int, float)) and count_value > 1:
            return True
        if isinstance(count_value, str) and count_value.strip().isdigit():
            if int(count_value.strip()) > 1:
                return True

    pages = document_data.get("pages")
    if isinstance(pages, list) and len(pages) > 1:
        return True

    side = document_data.get("side")
    if isinstance(side, str) and side.strip().lower() in {"both", "combined", "full"}:
        return True

    return False


def _emirates_id_member_name(document_data: dict[str, Any]) -> str:
    for key in ("full_name", "fullName", "name"):
        value = document_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    first_name = document_data.get("first_name")
    last_name = document_data.get("last_name")
    if isinstance(first_name, str) and first_name.strip():
        if isinstance(last_name, str) and last_name.strip():
            return f"{first_name.strip()} {last_name.strip()}"
        return first_name.strip()
    return ""


def _emirates_front_detected(document_data: dict[str, Any]) -> bool:
    return bool(document_data.get("date_of_birth") and _emirates_id_member_name(document_data))


def _emirates_back_detected(document_data: dict[str, Any]) -> bool:
    return bool(document_data.get("card_number"))


def _emirates_upload_contains_both_sides(
    document_data: dict[str, Any],
    file_path: str = "",
    raw_message: str = "",
    user_id: str = "",
) -> bool:
    if _emirates_front_detected(document_data) and _emirates_back_detected(document_data):
        return True

    candidates: list[Any] = [
        file_path,
        raw_message,
        peek_last_upload_relative_path(user_id, "emirates_id") if user_id else "",
    ]
    for key in (
        "stored_relative_path",
        "upload_relative_path",
        "file_path",
        "document_path",
        "local_path",
        "document_url",
        "pdf_url",
        "file_url",
        "_file_reference",
    ):
        candidates.append(document_data.get(key))
    if any(_looks_like_pdf_reference(candidate) for candidate in candidates):
        return True
    for count_key in ("page_count", "pages_count", "total_pages"):
        count_value = document_data.get(count_key)
        if isinstance(count_value, (int, float)) and count_value > 1:
            return True
        if isinstance(count_value, str) and count_value.strip().isdigit():
            if int(count_value.strip()) > 1:
                return True
    pages = document_data.get("pages")
    return isinstance(pages, list) and len(pages) > 1


def _remember_emirates_payload(
    responses: dict[str, Any],
    document_data: dict[str, Any],
    conversation_state: dict[str, Any] | None = None,
) -> tuple[bool, bool]:
    has_back = _emirates_back_detected(document_data)
    has_front = _emirates_front_detected(document_data)

    if has_back:
        responses["back_page_received"] = True
        responses["Card Number"] = document_data.get("card_number")
        responses["_motor_eid_back_payload"] = document_data

    if has_front:
        responses["front_page_received"] = True
        responses["_motor_eid_front_payload"] = document_data
        _name_k, _dob_k, _gender_k = medical_member_identity_keys(
            conversation_state
        )
        member_name = _emirates_id_member_name(document_data)
        responses[_name_k] = member_name or document_data.get("name")
        responses[_dob_k] = document_data.get("date_of_birth")
        if "gender" in document_data and document_data.get("gender"):
            responses[_gender_k] = document_data.get("gender")

    return has_front, has_back


def _merged_emirates_payload(
    responses: dict[str, Any], current_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in (
        responses.get("_motor_eid_front_payload"),
        responses.get("_motor_eid_back_payload"),
        current_payload,
    ):
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if value not in (None, "", [], {}, "null"):
                merged[key] = value
            elif key not in merged:
                merged[key] = value
    return merged


def _apply_driving_license_fields(
    responses: dict[str, Any], document_data: dict[str, Any]
) -> None:
    responses["driving license Name in the License"] = document_data.get("name")
    responses["Date of Birth (DOB) in the License"] = document_data.get("date_of_birth")
    responses["License No in the License"] = document_data.get("license_no")
    responses["Nationality in the License"] = document_data.get("nationality")
    responses["Issue Date in the License"] = document_data.get("issue_date")
    responses["Expiry Date in the License"] = document_data.get("expiry_date")
    responses["Place Of Issue in the License"] = document_data.get("place_of_issue")


def _remember_driving_license_payload(
    responses: dict[str, Any], document_data: dict[str, Any], *, is_back: bool = False
) -> None:
    _apply_driving_license_fields(responses, document_data)
    key = (
        "_motor_driving_license_back_payload"
        if is_back
        else "_motor_driving_license_front_payload"
    )
    responses[key] = document_data


def _merged_driving_license_payload(
    responses: dict[str, Any], current_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in (
        responses.get("_motor_driving_license_front_payload"),
        responses.get("_motor_driving_license_back_payload"),
        current_payload,
    ):
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if value not in (None, "", [], {}, "null"):
                merged[key] = value
            elif key not in merged:
                merged[key] = value
    return merged


def _driving_license_upload_contains_both_sides(
    document_data: dict[str, Any],
    file_path: str = "",
    raw_message: str = "",
    user_id: str = "",
) -> bool:
    candidates: list[Any] = [
        file_path,
        raw_message,
        peek_last_upload_relative_path(user_id, "driving_license") if user_id else "",
    ]
    for key in (
        "stored_relative_path",
        "upload_relative_path",
        "file_path",
        "document_path",
        "local_path",
        "document_url",
        "pdf_url",
        "file_url",
        "_file_reference",
    ):
        candidates.append(document_data.get(key))
    if any(_looks_like_pdf_reference(candidate) for candidate in candidates):
        return True
    for count_key in ("page_count", "pages_count", "total_pages"):
        count_value = document_data.get(count_key)
        if isinstance(count_value, (int, float)) and count_value > 1:
            return True
        if isinstance(count_value, str) and count_value.strip().isdigit():
            if int(count_value.strip()) > 1:
                return True
    pages = document_data.get("pages")
    if isinstance(pages, list) and len(pages) > 1:
        return True
    side = document_data.get("side")
    return isinstance(side, str) and side.strip().lower() in {"both", "combined", "full"}


def _has_both_driving_license_payloads(responses: dict[str, Any]) -> bool:
    return (
        isinstance(responses.get("_motor_driving_license_front_payload"), dict)
        and isinstance(responses.get("_motor_driving_license_back_payload"), dict)
    )


def _driving_license_upload_complete(
    responses: dict[str, Any],
    document_data: dict[str, Any],
    *,
    file_path: str = "",
    raw_message: str = "",
    user_id: str = "",
) -> bool:
    return _driving_license_upload_contains_both_sides(
        document_data,
        file_path=file_path,
        raw_message=raw_message,
        user_id=user_id,
    ) or _has_both_driving_license_payloads(responses)


def _driving_license_incomplete_prompt(_responses: dict[str, Any]) -> str:
    """Always re-prompt with front+back wording until both sides are collected."""
    return MOTOR_DRIVING_LICENSE_COMBINED_UPLOAD_QUESTION


def _rewind_to_driving_license_step(
    conversation_state: dict[str, Any], questions: list[Any]
) -> None:
    idx = conversation_state["current_question_index"]
    for i in range(idx, -1, -1):
        if i >= len(questions):
            continue
        q = questions[i]
        if isinstance(q, dict) and q.get("step_id") == STEP_UPLOAD_DRIVING_LICENSE:
            conversation_state["current_question_index"] = i
            return
    for i in range(idx):
        q = questions[i]
        if isinstance(q, dict) and q.get("step_id") == STEP_UPLOAD_DRIVING_LICENSE:
            conversation_state["current_question_index"] = i
            return


def _skip_past_driving_license_substeps(
    conversation_state: dict[str, Any], questions: list[Any]
) -> None:
    _dl_steps = {
        STEP_UPLOAD_DRIVING_LICENSE,
        STEP_UPLOAD_DRIVING_LICENSE_FRONT,
        STEP_UPLOAD_DRIVING_LICENSE_BACK,
    }
    idx = conversation_state["current_question_index"]
    while idx < len(questions):
        q = questions[idx]
        if isinstance(q, dict) and q.get("step_id") in _dl_steps:
            idx += 1
            continue
        break
    conversation_state["current_question_index"] = idx


def _ensure_motor_contact_steps_before_cover(
    questions: list[Any], current_index: int
) -> None:
    """Insert email only before cover — mobile is collected at car flow start (motor_enquiry)."""
    if current_index >= len(questions):
        return
    current = questions[current_index]
    if not isinstance(current, dict) or current.get("step_id") != STEP_MOTOR_COVER_TYPE:
        return
    if current_index > 0:
        prev = questions[current_index - 1]
        if isinstance(prev, dict) and prev.get("step_id") == STEP_SPONSOR_EMAIL:
            return
    questions.insert(
        current_index,
        {
            "step_id": STEP_SPONSOR_EMAIL,
            "question": "May I have the Email Address",
        },
    )


def _claim_upload_file_type(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    raw = payload.get("file_type")
    return raw.strip() if isinstance(raw, str) else ""


def _expected_claim_upload_types_for_step(step: str) -> set[str]:
    mapping = {
        STEP_UPLOAD_DRIVING_LICENSE: {"driving_license"},
        STEP_UPLOAD_DRIVING_LICENSE_FRONT: {"driving_license"},
        STEP_UPLOAD_DRIVING_LICENSE_BACK: {"driving_license"},
        STEP_UPLOAD_EMIRATES_DOC: {"emirates_id"},
        STEP_UPLOAD_EID_FRONT: {"emirates_id"},
        STEP_UPLOAD_EID_BACK: {"emirates_id"},
        STEP_UPLOAD_MULKIYA: {"mulkiya", "vehicle_registration"},
        STEP_UPLOAD_MULKIYA_FRONT: {"mulkiya", "vehicle_registration"},
        STEP_UPLOAD_MULKIYA_BACK: {"mulkiya", "vehicle_registration"},
        STEP_VEHICLE_TEST_CERT: {"passing_paper"},
    }
    return mapping.get(step, set())


def _claim_upload_payload_mismatch(payload: Any, step: str) -> bool:
    if not isinstance(payload, dict) or not payload.get("claim_document_upload"):
        return False
    file_type = _claim_upload_file_type(payload)
    expected = _expected_claim_upload_types_for_step(step)
    return bool(file_type and expected and file_type not in expected)


def _claim_upload_mismatch_response(
    *,
    payload: dict[str, Any],
    question: str,
    user_language: str,
) -> dict[str, Any]:
    file_type = _claim_upload_file_type(payload).replace("_", " ").strip() or "document"
    msg_type, doc_type = detect_document_type_from_question(question)
    result = format_response_in_language(
        f"We stored your {file_type}, but this step is asking for a different document. "
        f"Please upload the requested file.",
        [],
        user_language,
        msg_type,
        doc_type,
    )
    result["question"] = translate_text(question, user_language)
    return result


def _motor_quote_try_upload(
    *,
    user_id: str,
    file_path: str,
    current_flow: str,
    responses: dict[str, Any],
    step: str,
    user_message: str,
) -> None:
    motor_flow_try_upload_document(
        user_id=user_id,
        file_path=file_path,
        current_flow=current_flow,
        responses=responses,
        step_id=step,
        raw_message=user_message,
    )


def _merge_upload_file_path_into_message(user_input: UserInput, message: str) -> str:
    """Clients often POST upload success text in ``message`` and the path in ``file_path``."""
    fp_from_body = (user_input.file_path or "").strip()
    fp_raw = fp_from_body
    low = message.lower()
    boilerplate = (
        not message
        or "upload successfully" in low
        or "uploaded successfully" in low
        or "file uploaded" in low
        or "document uploaded" in low
        or "document upload successfully" in low
        or ("document upload" in low and "success" in low)
    )
    if not fp_raw:
        return message
    fp_norm = fp_raw.replace("\\", "/")
    # Keep structured JSON in ``message``; motor InsuranceLab upload reads ``file_path`` separately.
    msg_stripped = message.strip()
    if msg_stripped.startswith("{"):
        try:
            parsed_json = json.loads(msg_stripped)
        except json.JSONDecodeError:
            parsed_json = None
        if isinstance(parsed_json, (dict, list)):
            return message
    if boilerplate or fp_from_body:
        return fp_norm
    return message


def _motor_document_transition_message(
    next_question_text: str, user_language: str
) -> str:
    """Thank-you line plus next upload prompt (matches Emirates ID hand-off style)."""
    intro = translate_text(
        "Thank you for uploading the document. Now, let's move on to:",
        user_language,
    )
    return f"{intro} {translate_text(next_question_text, user_language)}"


def _motor_upload_question_display_text(question: Any) -> str:
    if isinstance(question, dict):
        if question.get("step_id") == STEP_UPLOAD_DRIVING_LICENSE:
            return MOTOR_DRIVING_LICENSE_COMBINED_UPLOAD_QUESTION
        return str(question.get("question") or "")
    return str(question)


def _format_info_then_question(
    info_text: str, question_text: str, options: list[str], user_language: str
) -> dict:
    info_tr = translate_text(info_text, user_language)
    question_tr = translate_text(question_text, user_language)
    translated_options = [translate_text(opt, user_language) for opt in options]
    return {
        "response": info_tr,
        "question": question_tr,
        "options": ", ".join(translated_options),
        "language": user_language,
        "language_code": get_language_code(user_language),
    }


def _medical_claim_registered_message() -> str:
    return (
        "Thank you! Your claim request has been successfully registered.\n\n"
        "Our team is currently reviewing your details and will forward them to the "
        "relevant insurance provider for prompt assistance. We'll keep you updated "
        "at every step until it's resolved 🤝"
    )


def _is_claim_upload_input(user_message: str) -> bool:
    raw = (user_message or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if (
        "upload successfully" in low
        or "uploaded successfully" in low
        or "file uploaded" in low
        or "document uploaded" in low
        or ("document upload" in low and "success" in low)
    ):
        return True
    normalized = raw.replace("\\", "/")
    if normalized.lower().startswith("uploads/") and re.search(
        r"\.(pdf|docx|jpg|jpeg|png)$", normalized, re.IGNORECASE
    ):
        return True
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
            return isinstance(payload, dict) and bool(payload)
        except json.JSONDecodeError:
            return False
    return False


def _motor_document_message_to_json(user_message: str) -> dict[str, Any] | None:
    """Parse OCR JSON, or accept upload path / success text as a minimal dict for motor flow."""
    raw = (user_message or "").strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
    if _is_claim_upload_input(raw):
        return {"_file_reference": raw}
    return None


def _restart_to_initial_menu(
    *, user_id: str, conversation_state: dict[str, Any]
) -> dict[str, Any]:
    return reset_to_initial_and_format_first_question(
        user_states=user_states,
        user_id=user_id,
        conversation_state=conversation_state,
        initial_questions=initial_questions,
        format_response_in_language=format_response_in_language,
        reset_prefix="Great! Let's start over. ",
    )


def process_user_input(user_input: UserInput):
    refresh_medical_questions_if_changed()
    user_id = user_input.user_id.strip()
    user_message = _merge_upload_file_path_into_message(user_input, user_input.message.strip())
    # Initialize user state if not already presents
    if user_id not in user_states:
        user_states[user_id] = {
            "current_question_index": 0,
            "responses": {},
            "current_flow": "initial",
            "welcome_shown": False,
            "awaiting_document_name": False,
            "document_name": "",
            "preferred_language": "English",  # Default language
            "language_code": "en",
            "language_explicitly_set": False,  # Track if user explicitly set language
        }

    conversation_state = user_states[user_id]
    conversation_state["user_id"] = user_id
    user_name = get_user_name(user_id)

    # ==================== LANGUAGE DETECTION ====================
    # Check for explicit language requests first
    language_requests = {
        "say in arabic": {"language": "Arabic", "code": "ar"},
        "speak in arabic": {"language": "Arabic", "code": "ar"},
        "change language to arabic": {"language": "Arabic", "code": "ar"},
        "arabic": {"language": "Arabic", "code": "ar"},
        "عربي": {"language": "Arabic", "code": "ar"},
        "بالعربية": {"language": "Arabic", "code": "ar"},
        "say in hindi": {"language": "Hindi", "code": "hi"},
        "hindi": {"language": "Hindi", "code": "hi"},
        "हिंदी": {"language": "Hindi", "code": "hi"},
        "say in urdu": {"language": "Urdu", "code": "ur"},
        "urdu": {"language": "Urdu", "code": "ur"},
        "اردو": {"language": "Urdu", "code": "ur"},
        "say in english": {"language": "English", "code": "en"},
        "english": {"language": "English", "code": "en"},
    }

    # Check if this is a document upload success message (define this before the if/else block)
    is_document_upload_success = (
        "document upload successfully" in user_message.lower()
        or "upload successfully" in user_message.lower()
        or "file uploaded" in user_message.lower()
        or "document uploaded" in user_message.lower()
    )

    # Check if user is making an explicit language request
    user_message_lower = user_message.lower().strip()
    if user_message_lower in language_requests:
        requested_lang = language_requests[user_message_lower]
        conversation_state["preferred_language"] = requested_lang["language"]
        conversation_state["language_code"] = requested_lang["code"]
        conversation_state["language_explicitly_set"] = True  # Mark as explicitly set
        print(
            f"[Language Request] User {user_id} explicitly requested {requested_lang['language']}"
        )
        user_language = requested_lang["language"]
    else:
        # Only detect language for the first few messages or when user explicitly changes language
        # Don't change language for numeric inputs, short responses, or when already in a flow
        current_flow = conversation_state.get("current_flow", "initial")
        current_question_index = conversation_state.get("current_question_index", 0)

        # Check if this is a numeric input, very short response, or document upload success message
        is_numeric_or_short = (
            user_message.strip().isdigit()
            or len(user_message.strip()) <= 3
            or user_message.strip().lower() in ["yes", "no", "y", "n", "ok", "okay"]
        )

        # Only detect language if:
        # 1. Language was not explicitly set by user
        # 2. We're still in initial flow (first few questions)
        # 3. User hasn't established a language preference yet
        # 4. The input is not numeric/short
        # 5. The input is not a document upload success message (regardless of current language)
        should_detect_language = (
            not conversation_state.get("language_explicitly_set", False)
            and current_flow == "initial"
            and current_question_index < 3
            and not is_numeric_or_short
            and not is_document_upload_success  # This prevents language detection for document upload messages in ANY language
            and conversation_state.get("preferred_language", "English") == "English"
        )

        if should_detect_language:
            detected_lang = detect_language(user_message)
            print(f"[DEBUG] Detected language: {detected_lang}")

            # Update the user's preferred language if it's different
            if detected_lang["language"] != conversation_state.get(
                "preferred_language", "English"
            ):
                conversation_state["preferred_language"] = detected_lang["language"]
                conversation_state["language_code"] = detected_lang["code"]
                print(
                    f"[Language Detection] User {user_id} switched to {detected_lang['language']}"
                )
        else:
            reason = "unknown"
            if conversation_state.get("language_explicitly_set", False):
                reason = "language explicitly set by user"
            elif current_flow != "initial":
                reason = "not in initial flow"
            elif current_question_index >= 3:
                reason = "beyond initial questions"
            elif is_numeric_or_short:
                reason = "numeric or short input"
            elif is_document_upload_success:
                reason = "document upload success message"
            elif conversation_state.get("preferred_language", "English") != "English":
                reason = "language already established"

            print(
                f"[DEBUG] Skipping language detection ({reason}) - preserving current language: {conversation_state.get('preferred_language', 'English')}"
            )

        user_language = conversation_state.get("preferred_language", "English")

    print(f"[DEBUG] Final user language: {user_language}")
    # ==================== END LANGUAGE DETECTION ====================

    # Handle document upload success messages - maintain current language flow
    if is_document_upload_success:
        print(
            f"[DEBUG] Document upload success detected - maintaining current language: {user_language}"
        )
        # Continue with the current flow without changing language
        # The system will proceed to the next question in the same language

    # Handle language requests - present current question in new language
    if user_message_lower in language_requests:
        # Get current question
        current_question_index = conversation_state.get("current_question_index", 0)
        current_flow = conversation_state.get("current_flow", "initial")

        # Get the appropriate questions list based on current flow
        if current_flow == "car":
            questions_list = car_questions
        elif current_flow == "bike":
            questions_list = bike_questions
        else:
            questions_list = FLOW_REGISTRY.get_questions_for_flow(current_flow) or initial_questions

        # Present current question in new language
        if current_question_index < len(questions_list):
            current_question = questions_list[current_question_index]
            if isinstance(current_question, dict):
                question_text = current_question["question"]
                options = current_question.get("options", [])
                return format_response_in_language(
                    question_text, options, user_language
                )
            else:
                return format_response_in_language(current_question, [], user_language)
        else:
            # No current question, show welcome
            first_question = initial_questions[0]
            if isinstance(first_question, dict):
                question_text = first_question["question"]
                options = first_question.get("options", [])
                return format_response_in_language(
                    question_text, options, user_language
                )
            else:
                return format_response_in_language(first_question, [], user_language)

    # Handle cancel / restart / reset — shared helper in conversation_reset
    if is_reset_command(user_message):
        return reset_to_initial_and_format_first_question(
            user_states=user_states,
            user_id=user_id,
            conversation_state=conversation_state,
            initial_questions=initial_questions,
            format_response_in_language=format_response_in_language,
        )

    # Show welcome message with the first question if not already shown
    if not conversation_state["welcome_shown"]:
        conversation_state["welcome_shown"] = True
        first_question = initial_questions[0]
        next_options = first_question.get("options", [])
        greeting = choice(greeting_templates).format(
            user_name=user_name, first_question=first_question["question"]
        )

        # Translate greeting and options to user's language
        return format_response_in_language(greeting, next_options, user_language)

    # Determine the current flow and questions
    current_flow = conversation_state["current_flow"]
    questions = FLOW_REGISTRY.get_questions_for_flow(current_flow)

    # Get current question index
    current_index = conversation_state["current_question_index"]
    responses = conversation_state["responses"]

    if conversation_state.get("awaiting_medical_repurchase"):
        repurchase_vr = validate_response_multilingual(
            user_message, ["Yes", "No"], user_language
        )
        if repurchase_vr["is_valid"]:
            conversation_state["awaiting_medical_repurchase"] = False
            if repurchase_vr["matched_value"] == "Yes":
                conversation_state.pop(CONV_STATE_MEMBER_NAME_KEY, None)
                return _restart_to_initial_menu(
                    user_id=user_id, conversation_state=conversation_state
                )
            gb = translate_text(
                "Thank you for using Insura. We hope to serve you again soon!",
                user_language,
            )
            return {
                "response": gb,
                "language": user_language,
                "language_code": get_language_code(user_language),
            }
        rp_q = translate_text(MEDICAL_REPURCHASE_QUESTION, user_language)
        rp_opts = [
            translate_text("Yes", user_language),
            translate_text("No", user_language),
        ]
        hint = translate_text(
            "Please choose Yes or No to continue.", user_language
        )
        return {
            "response": hint,
            "question": rp_q,
            "options": ", ".join(rp_opts),
            "language": user_language,
            "language_code": get_language_code(user_language),
        }

    gdf = conversation_state.get("general_document_followup")
    if (
        gdf
        and current_flow == "general_insurance"
        and isinstance(gdf, dict)
        and (
            gdf.get("kind") == GENERAL_DOC_KIND_GENERAL_TEMPLATE
            or gdf.get("kind") in _LEGACY_GENERAL_DOC_WORKBOOK_KINDS
        )
    ):
        phase = gdf.get("phase")
        expected_template = (gdf.get("expected_template") or "").strip()
        _k_legacy = gdf.get("kind")
        if not expected_template and _k_legacy == "workmen_compensation":
            expected_template = WORKMEN_COMPENSATION_XLSX
        elif not expected_template and _k_legacy == "travel_insurance":
            expected_template = TRAVEL_INSURANCE_XLSX
        product_name = (
            responses.get("General Insurance Type")
            or gdf.get("product_display")
            or "General Insurance"
        )
        strict_form_validation = bool(
            gdf.get(GENERAL_DOC_FIELD_STRICT_WORKBOOK_VALIDATION)
        ) or (_k_legacy in _LEGACY_GENERAL_DOC_WORKBOOK_KINDS)
        _hint_dt_raw = (gdf.get("upload_hint_document_type") or "excel").strip().lower()
        upload_hint_dt = (
            "excel"
            if _hint_dt_raw in ("excel", "xlsx", "xls")
            else _hint_dt_raw
        )
        yes_no_opts = list(GENERAL_DOC_YES_NO_OPTIONS)
        specialist_opts = list(GENERAL_DOC_SPECIALIST_OPTIONS)

        def _accept_primary_form_upload(pl):
            if strict_form_validation:
                return should_accept_travel_form_upload(pl)
            return should_accept_general_template_form_upload(pl)

        def _persist_user_responses_json():
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
            except OSError:
                pass

        def _travel_corporate_complete(thank_message_en: str):
            conversation_state.pop("general_document_followup", None)
            conversation_state["current_flow"] = "initial"
            conversation_state["current_question_index"] = 0
            _persist_user_responses_json()
            wipe_flow_session_upload_files(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
            )
            snapshot = responses
            conversation_state["responses"] = {}
            result = format_response_in_language(
                thank_message_en, [], user_language
            )
            result["final_responses"] = snapshot
            return result

        def _schedule_travel_anything_else_thank(thank_message_en: str):
            general_flow_try_first_step(
                current_flow=current_flow, responses=responses
            )
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_ANYTHING_ELSE
            gdf["pending_travel_thank_en"] = thank_message_en
            _persist_user_responses_json()
            return format_response_in_language(
                GENERAL_DOC_MSG_ASSISTANCE_ANYTHING_ELSE, yes_no_opts, user_language
            )

        def _specialist_choice_invalid():
            hint = translate_text(
                "Please choose one of the options below to continue.", user_language
            )
            closing = translate_text(GENERAL_DOC_MSG_DETAILS_CLOSING, user_language)
            return format_response_in_language(
                f"{hint}\n\n{closing}", specialist_opts, user_language
            )

        def _resolve_yes_no_choice(message: str):
            direct = match_yes_no_english(message)
            if direct is not None:
                return direct
            vr = validate_response_multilingual(
                message, yes_no_opts, user_language
            )
            if vr.get("is_valid") and vr.get("matched_value") in ("Yes", "No"):
                return vr["matched_value"]
            return None

        def _attach_general_document_upload_hints(res: dict, upload_type: str) -> dict:
            """Mirrors POST ``/upload-document/`` ``type`` + general-insurance scope for UIs."""
            res["upload_category"] = GENERAL_UPLOAD_DOCUMENT_CATEGORY
            res["upload_type"] = upload_type
            return res

        upload_ok, payload = parse_general_upload_payload(
            user_message, is_document_upload_success=is_document_upload_success
        )

        def _after_travel_form_upload_ok(form_payload):
            responses[f"{product_name} completed upload"] = form_payload
            responses[f"{product_name} expected template"] = expected_template
            general_flow_try_upload_from_payload(
                current_flow=current_flow,
                responses=responses,
                payload=form_payload,
                product_name=product_name,
            )
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_TRADE_LICENCE
            _persist_user_responses_json()
            thank = translate_text(
                GENERAL_DOC_MSG_TRAVEL_UPLOAD_RECEIVED, user_language
            )
            trade_prompt = translate_text(
                GENERAL_DOC_MSG_TRADE_LICENCE_UPLOAD, user_language
            )
            body = f"{thank}\n\n{trade_prompt}"
            mt, dt = detect_document_type_from_question(
                GENERAL_DOC_MSG_TRADE_LICENCE_UPLOAD
            )
            out = format_response_in_language(
                body, [], user_language, mt, dt
            )
            return _attach_general_document_upload_hints(
                out, GENERAL_UPLOAD_TYPE_TRADE_LICENCE
            )

        if phase == GENERAL_DOC_PHASE_DOWNLOAD_SHOWN:
            if upload_ok and _accept_primary_form_upload(payload):
                return _after_travel_form_upload_ok(payload)
            if upload_ok and normalized_upload_payload_file_type(payload) in (
                "trade_license",
                "vat_certificate",
            ):
                wrong_en = GENERAL_DOC_MSG_WRONG_UPLOAD_USE_FORM_NOT_TRADE_VAT_EN
                return _attach_general_document_upload_hints(
                    format_response_in_language(
                        wrong_en,
                        [],
                        user_language,
                        "document_upload_request",
                        upload_hint_dt,
                    ),
                    GENERAL_UPLOAD_TYPE_TRAVEL_FORM,
                )
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_UPLOAD
            fill_prompt_en = GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN
            return _attach_general_document_upload_hints(
                format_response_in_language(
                    fill_prompt_en,
                    [],
                    user_language,
                    "document_upload_request",
                    upload_hint_dt,
                ),
                GENERAL_UPLOAD_TYPE_TRAVEL_FORM,
            )

        if phase == GENERAL_DOC_PHASE_AWAITING_UPLOAD:
            if upload_ok and _accept_primary_form_upload(payload):
                return _after_travel_form_upload_ok(payload)
            if upload_ok and normalized_upload_payload_file_type(payload) in (
                "trade_license",
                "vat_certificate",
            ):
                wrong_en = GENERAL_DOC_MSG_WRONG_UPLOAD_USE_FORM_NOT_TRADE_VAT_EN
                return _attach_general_document_upload_hints(
                    format_response_in_language(
                        wrong_en,
                        [],
                        user_language,
                        "document_upload_request",
                        upload_hint_dt,
                    ),
                    GENERAL_UPLOAD_TYPE_TRAVEL_FORM,
                )
            retry_msg_en = GENERAL_DOC_MSG_RETRY_FILLED_FORM_EN
            return _attach_general_document_upload_hints(
                format_response_in_language(
                    retry_msg_en,
                    [],
                    user_language,
                    "document_upload_request",
                    upload_hint_dt,
                ),
                GENERAL_UPLOAD_TYPE_TRAVEL_FORM,
            )

        if phase == GENERAL_DOC_PHASE_AWAITING_TRADE_LICENCE:
            if upload_ok and should_accept_trade_licence_upload(payload):
                responses["Trade licence upload"] = payload
                general_flow_try_upload_from_payload(
                    current_flow=current_flow,
                    responses=responses,
                    payload=payload,
                    product_name=product_name,
                )
                gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_VAT_AVAILABLE
                _persist_user_responses_json()
                thank = translate_text(
                    GENERAL_DOC_MSG_TRADE_LICENCE_THANK, user_language
                )
                q = translate_text(GENERAL_DOC_MSG_VAT_AVAILABLE, user_language)
                return format_response_in_language(
                    f"{thank}\n\n{q}", yes_no_opts, user_language
                )
            mt, dt = detect_document_type_from_question(
                GENERAL_DOC_MSG_TRADE_LICENCE_UPLOAD
            )
            return _attach_general_document_upload_hints(
                format_response_in_language(
                    GENERAL_DOC_MSG_TRADE_LICENCE_UPLOAD,
                    [],
                    user_language,
                    mt,
                    dt,
                ),
                GENERAL_UPLOAD_TYPE_TRADE_LICENCE,
            )

        if phase == GENERAL_DOC_PHASE_AWAITING_VAT_AVAILABLE:
            yn = _resolve_yes_no_choice(user_message)
            if yn is None:
                return format_response_in_language(
                    GENERAL_DOC_MSG_VAT_AVAILABLE, yes_no_opts, user_language
                )
            if yn == "Yes":
                responses["VAT certificate available"] = "Yes"
                gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_VAT_CERTIFICATE
                _persist_user_responses_json()
                msg = translate_text(GENERAL_DOC_MSG_VAT_UPLOAD, user_language)
                mt, dt = detect_document_type_from_question(
                    GENERAL_DOC_MSG_VAT_UPLOAD
                )
                return _attach_general_document_upload_hints(
                    format_response_in_language(
                        msg, [], user_language, mt, dt
                    ),
                    GENERAL_UPLOAD_TYPE_VAT_CERTIFICATE,
                )
            responses["VAT certificate available"] = "No"
            responses.pop("VAT certificate upload", None)
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_EXISTING_POLICY
            _persist_user_responses_json()
            q = translate_text(GENERAL_DOC_MSG_EXISTING_POLICY, user_language)
            return format_response_in_language(q, yes_no_opts, user_language)

        if phase == GENERAL_DOC_PHASE_AWAITING_VAT_CERTIFICATE:
            if _resolve_yes_no_choice(user_message) == "No":
                responses["VAT certificate available"] = "No"
                responses.pop("VAT certificate upload", None)
                gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_EXISTING_POLICY
                _persist_user_responses_json()
                q = translate_text(GENERAL_DOC_MSG_EXISTING_POLICY, user_language)
                return format_response_in_language(q, yes_no_opts, user_language)
            if upload_ok and should_accept_vat_certificate_upload(payload):
                responses["VAT certificate upload"] = payload
                general_flow_try_upload_from_payload(
                    current_flow=current_flow,
                    responses=responses,
                    payload=payload,
                    product_name=product_name,
                )
                gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_EXISTING_POLICY
                _persist_user_responses_json()
                thank = translate_text(GENERAL_DOC_MSG_VAT_THANK, user_language)
                q = translate_text(GENERAL_DOC_MSG_EXISTING_POLICY, user_language)
                return format_response_in_language(
                    f"{thank}\n\n{q}", yes_no_opts, user_language
                )
            mt, dt = detect_document_type_from_question(
                GENERAL_DOC_MSG_VAT_UPLOAD
            )
            return _attach_general_document_upload_hints(
                format_response_in_language(
                    GENERAL_DOC_MSG_VAT_UPLOAD,
                    [],
                    user_language,
                    mt,
                    dt,
                ),
                GENERAL_UPLOAD_TYPE_VAT_CERTIFICATE,
            )

        if phase == GENERAL_DOC_PHASE_AWAITING_EXISTING_POLICY:
            yn = _resolve_yes_no_choice(user_message)
            if yn is None:
                return format_response_in_language(
                    GENERAL_DOC_MSG_EXISTING_POLICY, yes_no_opts, user_language
                )
            if yn == "Yes":
                responses["Existing policy"] = "Yes"
                gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_POLICY_SCHEDULE_DATE
                _persist_user_responses_json()
                msg = translate_text(GENERAL_DOC_MSG_POLICY_SCHEDULE, user_language)
                return format_response_in_language(msg, [], user_language)
            responses["Existing policy"] = "No"
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_COMPANY_EMAIL
            _persist_user_responses_json()
            intro = translate_text(
                GENERAL_DOC_MSG_COMPANY_EMAIL_INTRO, user_language
            )
            prompt = translate_text(
                GENERAL_DOC_MSG_COMPANY_EMAIL_PROMPT, user_language
            )
            return format_response_in_language(f"{intro}\n\n{prompt}", [], user_language)

        if phase == GENERAL_DOC_PHASE_AWAITING_POLICY_SCHEDULE_DATE:
            raw = user_message.strip()
            if len(raw) < 2:
                retry = translate_text(
                    "Please share your latest policy schedule date.", user_language
                )
                return format_response_in_language(retry, [], user_language)
            responses["Policy schedule date"] = raw
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_COMPANY_EMAIL
            _persist_user_responses_json()
            thank = translate_text(
                GENERAL_DOC_MSG_POLICY_SCHEDULE_THANK, user_language
            )
            intro = translate_text(
                GENERAL_DOC_MSG_COMPANY_EMAIL_INTRO, user_language
            )
            prompt = translate_text(
                GENERAL_DOC_MSG_COMPANY_EMAIL_PROMPT, user_language
            )
            body = f"{thank}\n\n{intro}\n\n{prompt}"
            return format_response_in_language(body, [], user_language)

        if phase == GENERAL_DOC_PHASE_AWAITING_COMPANY_EMAIL:
            email_value = user_message.strip()
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email_value):
                retry = translate_text(
                    "Please enter a valid email address.", user_language
                )
                intro = translate_text(
                    GENERAL_DOC_MSG_COMPANY_EMAIL_INTRO, user_language
                )
                prompt = translate_text(
                    GENERAL_DOC_MSG_COMPANY_EMAIL_PROMPT, user_language
                )
                return format_response_in_language(
                    f"{retry}\n\n{intro}\n\n{prompt}", [], user_language
                )
            responses["Official company email"] = email_value
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_COMPANY_NAME
            _persist_user_responses_json()
            ty = translate_text(GENERAL_DOC_MSG_EMAIL_THANK, user_language)
            cq = translate_text(GENERAL_DOC_MSG_COMPANY_NAME, user_language)
            return format_response_in_language(f"{ty}\n\n{cq}", [], user_language)

        if phase == GENERAL_DOC_PHASE_AWAITING_COMPANY_NAME:
            name_val = user_message.strip()
            if len(name_val) < 2:
                retry = translate_text(
                    "Please share your company name.", user_language
                )
                return format_response_in_language(retry, [], user_language)
            responses["Company name"] = name_val
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_FULL_NAME
            _persist_user_responses_json()
            fq = translate_text(GENERAL_DOC_MSG_FULL_NAME, user_language)
            return format_response_in_language(fq, [], user_language)

        if phase == GENERAL_DOC_PHASE_AWAITING_FULL_NAME:
            name_val = user_message.strip()
            if len(name_val) < 2:
                retry = translate_text(
                    "Please share your full name.", user_language
                )
                return format_response_in_language(retry, [], user_language)
            responses["Full name"] = name_val
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_DESIGNATION
            _persist_user_responses_json()
            dq = translate_text(GENERAL_DOC_MSG_DESIGNATION, user_language)
            return format_response_in_language(dq, [], user_language)

        if phase == GENERAL_DOC_PHASE_AWAITING_DESIGNATION:
            desig = user_message.strip()
            if len(desig) < 2:
                retry = translate_text(
                    "Please share your designation or position.", user_language
                )
                return format_response_in_language(retry, [], user_language)
            responses["Designation"] = desig
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_SPECIALIST_CHOICE
            _persist_user_responses_json()
            closing = translate_text(GENERAL_DOC_MSG_DETAILS_CLOSING, user_language)
            return format_response_in_language(closing, specialist_opts, user_language)

        if phase == GENERAL_DOC_PHASE_AWAITING_SPECIALIST_CHOICE:
            vr = validate_response_multilingual(
                user_message, specialist_opts, user_language
            )
            if not vr["is_valid"]:
                return _specialist_choice_invalid()
            if vr["matched_value"] == GENERAL_DOC_OPT_CONNECT_SPECIALIST:
                responses["Corporate specialist preference"] = (
                    GENERAL_DOC_OPT_CONNECT_SPECIALIST
                )
                gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_SPECIALIST_PHONE
                _persist_user_responses_json()
                msg = translate_text(
                    GENERAL_DOC_MSG_PHONE_FOR_SPECIALIST, user_language
                )
                return format_response_in_language(msg, [], user_language)
            responses["Corporate specialist preference"] = GENERAL_DOC_OPT_WAIT
            general_flow_try_first_step(
                current_flow=current_flow, responses=responses
            )
            gdf["phase"] = GENERAL_DOC_PHASE_AFTER_WAIT_CLOSING
            gdf["pending_travel_thank_en"] = (
                GENERAL_DOC_MSG_ANYTHING_ELSE_DECLINED_SIGNOFF_EN
            )
            _persist_user_responses_json()
            return format_response_in_language(
                GENERAL_FLOW_INSURANCE_CLUB_CLOSING, [], user_language
            )

        if phase == GENERAL_DOC_PHASE_AFTER_WAIT_CLOSING:
            gdf["phase"] = GENERAL_DOC_PHASE_AWAITING_ANYTHING_ELSE
            _persist_user_responses_json()
            return format_response_in_language(
                GENERAL_DOC_MSG_ASSISTANCE_ANYTHING_ELSE,
                yes_no_opts,
                user_language,
            )

        if phase == GENERAL_DOC_PHASE_AWAITING_SPECIALIST_PHONE:
            phone_raw = re.sub(r"[\s\-]", "", user_message.strip())
            if not is_valid_mobile_number(phone_raw):
                retry = translate_text(
                    "Please share a valid phone number (digits only, 10–15 digits).",
                    user_language,
                )
                msg = translate_text(
                    GENERAL_DOC_MSG_PHONE_FOR_SPECIALIST, user_language
                )
                return format_response_in_language(f"{retry}\n\n{msg}", [], user_language)
            responses["Specialist contact phone"] = phone_raw
            return _schedule_travel_anything_else_thank(
                GENERAL_DOC_MSG_SPECIALIST_FORWARDED
            )

        if phase == GENERAL_DOC_PHASE_AWAITING_ANYTHING_ELSE:
            thank_pending = gdf.get("pending_travel_thank_en")
            if not isinstance(thank_pending, str) or not thank_pending.strip():
                thank_pending = GENERAL_FLOW_INSURANCE_CLUB_CLOSING
            yn2 = _resolve_yes_no_choice(user_message)
            if yn2 is None:
                return format_response_in_language(
                    GENERAL_DOC_MSG_ASSISTANCE_ANYTHING_ELSE,
                    yes_no_opts,
                    user_language,
                )
            if yn2 == "Yes":
                snapshot_more = dict(responses)
                _persist_user_responses_json()
                wipe_flow_session_upload_files(
                    user_id=user_id,
                    current_flow=current_flow,
                    responses=responses,
                )
                conversation_state.pop("general_document_followup", None)
                conversation_state["current_flow"] = "initial"
                conversation_state["current_question_index"] = 0
                conversation_state["responses"] = {}
                menu_first = initial_questions[0]
                if isinstance(menu_first, dict):
                    menu_result = format_response_in_language(
                        menu_first["question"],
                        menu_first.get("options", []),
                        user_language,
                    )
                else:
                    menu_result = format_response_in_language(
                        str(menu_first), [], user_language
                    )
                menu_result["final_responses"] = snapshot_more
                return menu_result
            return _travel_corporate_complete(thank_pending)

        conversation_state.pop("general_document_followup", None)

    if (
        gdf
        and current_flow == "general_insurance"
        and isinstance(gdf, dict)
        and gdf.get("kind") == GENERAL_DOC_KIND_PROPERTY_ALL_RISKS_MENU
    ):
        if gdf.get("phase") == GENERAL_DOC_PHASE_PAR_AWAITING_CHOICE:
            par_opts = list(PAR_SUBMENU_OPTIONS)
            par_vr = validate_response_multilingual(
                user_message, par_opts, user_language
            )
            if not par_vr.get("is_valid"):
                thank_m = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                pick_m = translate_text(PAR_MSG_PICK_ONE, user_language)
                return format_response_in_language(
                    f"{thank_m}\n\n{pick_m}", par_opts, user_language
                )
            par_choice = par_vr["matched_value"]
            if par_choice == PAR_SUBMENU_PROPERTY_INSURANCES:
                _par_fn, _par_doc_type = (
                    PAR_FILE_PROPERTY_INSURANCE_XLSX,
                    "excel",
                )
            elif par_choice == PAR_SUBMENU_SCOPE_DETAILS:
                _par_fn, _par_doc_type = (
                    PAR_FILE_SCOPE_DETAILS_XLSX,
                    "excel",
                )
            else:
                _par_fn, _par_doc_type = (
                    PAR_FILE_PROPOSAL_FORM_DOCX,
                    "docx",
                )
            responses["Property All Risks document"] = par_choice
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
            except OSError:
                pass
            par_snapshot = dict(responses)
            _par_link = general_document_download_url(_par_fn)
            thank_out = translate_text(PAR_MSG_THANK_SELECTION, user_language)
            conversation_state["current_flow"] = "general_insurance"
            conversation_state["current_question_index"] = len(questions)
            conversation_state["responses"] = dict(par_snapshot)
            _par_nm = f"{PROPERTY_ALL_RISKS_LABEL} — {par_choice}"
            conversation_state["general_document_followup"] = {
                "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                "expected_template": _par_fn,
                "product_display": _par_nm,
                "upload_hint_document_type": _par_doc_type,
            }
            _par_result = format_response_in_language(
                thank_out, [], user_language
            )
            _par_out = dict(_par_result)
            _par_out["general_link"] = _par_link
            _par_out["message_type"] = "document_download_request"
            _par_out["document_type"] = _par_doc_type
            _par_out["question"] = translate_text(
                GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
            )
            _par_out["final_responses"] = par_snapshot
            return _par_out
        conversation_state.pop("general_document_followup", None)

    if (
        gdf
        and current_flow == "general_insurance"
        and isinstance(gdf, dict)
        and gdf.get("kind") == GENERAL_DOC_KIND_CREDIT_INSURANCE_MENU
    ):
        if gdf.get("phase") == GENERAL_DOC_PHASE_CREDIT_AWAITING_CHOICE:
            _credit_opts = list(CREDIT_SUBMENU_OPTIONS)
            _credit_vr = validate_response_multilingual(
                user_message, _credit_opts, user_language
            )
            if not _credit_vr.get("is_valid"):
                _credit_ty = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                _credit_pk = translate_text(PAR_MSG_PICK_ONE, user_language)
                return format_response_in_language(
                    f"{_credit_ty}\n\n{_credit_pk}", _credit_opts, user_language
                )
            _credit_choice = _credit_vr["matched_value"]
            if _credit_choice == CREDIT_SUBMENU_CREDIT_INSURANCES:
                _credit_fn, _credit_dt = CREDIT_FILE_CREDIT_INSURANCES_XLSX, "excel"
            else:
                _credit_fn, _credit_dt = CREDIT_FILE_SINGLE_RISK_DOC, "doc"
            responses["Credit Insurance document"] = _credit_choice
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
            except OSError:
                pass
            _credit_snap = dict(responses)
            _credit_link = general_document_download_url(_credit_fn)
            _credit_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
            conversation_state["current_flow"] = "general_insurance"
            conversation_state["current_question_index"] = len(questions)
            conversation_state["responses"] = dict(_credit_snap)
            _credit_nm = f"{CREDIT_INSURANCE_LABEL} — {_credit_choice}"
            conversation_state["general_document_followup"] = {
                "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                "expected_template": _credit_fn,
                "product_display": _credit_nm,
                "upload_hint_document_type": _credit_dt,
            }
            _credit_base = format_response_in_language(_credit_th, [], user_language)
            _credit_out = dict(_credit_base)
            _credit_out["general_link"] = _credit_link
            _credit_out["message_type"] = "document_download_request"
            _credit_out["document_type"] = _credit_dt
            _credit_out["question"] = translate_text(
                GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
            )
            _credit_out["final_responses"] = _credit_snap
            return _credit_out
        conversation_state.pop("general_document_followup", None)

    if (
        gdf
        and current_flow == "general_insurance"
        and isinstance(gdf, dict)
        and gdf.get("kind") == GENERAL_DOC_KIND_GROUP_LIFE_MENU
    ):
        _gli_phase = gdf.get("phase")
        if _gli_phase == GENERAL_DOC_PHASE_GLI_LEVEL1:
            _gli_l1 = list(GLI_LEVEL1_OPTIONS)
            _gli_v1 = validate_response_multilingual(
                user_message, _gli_l1, user_language
            )
            if not _gli_v1.get("is_valid"):
                _gli_t1 = translate_text(
                    PAR_MSG_THANK_SELECTION, user_language
                )
                _gli_p1 = translate_text(PAR_MSG_PICK_ONE, user_language)
                return format_response_in_language(
                    f"{_gli_t1}\n\n{_gli_p1}", _gli_l1, user_language
                )
            _gli_c1 = _gli_v1["matched_value"]
            if _gli_c1 == GLI_OPTION_MALPRACTICING:
                gdf["phase"] = GENERAL_DOC_PHASE_GLI_MALPRACTICE
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _gli_t2 = translate_text(
                    PAR_MSG_THANK_SELECTION, user_language
                )
                _gli_p2 = translate_text(PAR_MSG_PICK_ONE, user_language)
                _gli_l2 = list(GLI_MALP_SUBMENU_OPTIONS)
                return format_response_in_language(
                    f"{_gli_t2}\n\n{_gli_p2}", _gli_l2, user_language
                )
            _gli_files = {
                GLI_OPTION_HEALTH: (GLI_FILE_GROUP_HEALTH, "excel"),
                GLI_OPTION_TRAVEL: (GLI_FILE_GROUP_TRAVEL, "excel"),
                GLI_OPTION_ACCIDENT: (GLI_FILE_GLPA_CENSUS, "xls"),
            }
            _gli_fn, _gli_dt = _gli_files[_gli_c1]
            responses["Group Life Insurance selection"] = _gli_c1
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
            except OSError:
                pass
            _gli_snap = dict(responses)
            _gli_link = general_document_download_url(_gli_fn)
            _gli_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
            conversation_state["current_flow"] = "general_insurance"
            conversation_state["current_question_index"] = len(questions)
            conversation_state["responses"] = dict(_gli_snap)
            _gli_nm = f"{GROUP_LIFE_INSURANCE_LABEL} — {_gli_c1}"
            conversation_state["general_document_followup"] = {
                "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                "expected_template": _gli_fn,
                "product_display": _gli_nm,
                "upload_hint_document_type": _gli_dt,
            }
            _gli_base = format_response_in_language(_gli_th, [], user_language)
            _gli_out = dict(_gli_base)
            _gli_out["general_link"] = _gli_link
            _gli_out["message_type"] = "document_download_request"
            _gli_out["document_type"] = _gli_dt
            _gli_out["question"] = translate_text(
                GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
            )
            _gli_out["final_responses"] = _gli_snap
            return _gli_out

        if _gli_phase == GENERAL_DOC_PHASE_GLI_MALPRACTICE:
            _gli_sl = list(GLI_MALP_SUBMENU_OPTIONS)
            _gli_v2 = validate_response_multilingual(
                user_message, _gli_sl, user_language
            )
            if not _gli_v2.get("is_valid"):
                _gli_tr = translate_text(
                    PAR_MSG_THANK_SELECTION, user_language
                )
                _gli_pr = translate_text(PAR_MSG_PICK_ONE, user_language)
                return format_response_in_language(
                    f"{_gli_tr}\n\n{_gli_pr}", _gli_sl, user_language
                )
            _gli_c2 = _gli_v2["matched_value"]
            if _gli_c2 == GLI_MALP_STAFF_LIST:
                _gli_f2, _gli_d2 = GLI_FILE_MMP_STAFF_LIST, "excel"
            else:
                _gli_f2, _gli_d2 = GLI_FILE_MALP_ESTABLISHMENTS, "doc"
            responses["Group Life Insurance selection"] = (
                f"{GLI_OPTION_MALPRACTICING} — {_gli_c2}"
            )
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
            except OSError:
                pass
            _gli_s2 = dict(responses)
            _gli_lk2 = general_document_download_url(_gli_f2)
            _gli_th2 = translate_text(PAR_MSG_THANK_SELECTION, user_language)
            conversation_state["current_flow"] = "general_insurance"
            conversation_state["current_question_index"] = len(questions)
            conversation_state["responses"] = dict(_gli_s2)
            _gli_nm2 = (
                f"{GROUP_LIFE_INSURANCE_LABEL} — "
                f"{GLI_OPTION_MALPRACTICING} — {_gli_c2}"
            )
            conversation_state["general_document_followup"] = {
                "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                "expected_template": _gli_f2,
                "product_display": _gli_nm2,
                "upload_hint_document_type": _gli_d2,
            }
            _gli_b2 = format_response_in_language(_gli_th2, [], user_language)
            _gli_o2 = dict(_gli_b2)
            _gli_o2["general_link"] = _gli_lk2
            _gli_o2["message_type"] = "document_download_request"
            _gli_o2["document_type"] = _gli_d2
            _gli_o2["question"] = translate_text(
                GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
            )
            _gli_o2["final_responses"] = _gli_s2
            return _gli_o2
        conversation_state.pop("general_document_followup", None)

    if current_index < len(questions):
        # Current question
        question_data = questions[current_index]
        question_data = patch_medical_marital_status_question(
            question_data, responses, conversation_state
        )
        questions[current_index] = question_data
        if isinstance(question_data, dict):
            _dl_steps = {
                STEP_UPLOAD_DRIVING_LICENSE,
                STEP_UPLOAD_DRIVING_LICENSE_FRONT,
                STEP_UPLOAD_DRIVING_LICENSE_BACK,
            }
            if (
                question_data.get("step_id") in _dl_steps
                and not _has_both_driving_license_payloads(responses)
            ):
                question_data = dict(question_data)
                question_data["step_id"] = STEP_UPLOAD_DRIVING_LICENSE
                question_data["question"] = MOTOR_DRIVING_LICENSE_COMBINED_UPLOAD_QUESTION
                questions[current_index] = question_data
            question = question_data["question"]
            options = question_data.get("options", [])
        else:
            question = question_data
            options = []

        step = resolve_step_id(question_data)

        # Handle options
        if options:
            if step == STEP_MOTOR_CLAIM_REPAIR_WORKSHOP:
                page = int(conversation_state.get("motor_claim_repair_page", 0) or 0)
                options = repair_workshop_paged_options(options, page)
            elif step == STEP_MOTOR_CLAIM_RECOVERY_LOCATION:
                page = int(
                    conversation_state.get("motor_claim_recovery_page", 0) or 0
                )
                options = repair_workshop_paged_options(options, page)

            # Duplicate upload POST after /upload-document/ must not count as an answer.
            if step in (
                STEP_MOTOR_CLAIM_REPAIR_WORKSHOP,
                STEP_MOTOR_CLAIM_RECOVERY_LOCATION,
            ) and _is_claim_upload_input(user_message):
                prompt = translate_text(question, user_language)
                return format_response_in_language(
                    prompt, options, user_language
                )

            # Validate user response against options using multilingual validation
            validation_result = validate_response_multilingual(
                user_message, options, user_language
            )

            if validation_result["is_valid"]:
                # Store the English version
                matched_option = validation_result["matched_value"]
                responses[question] = matched_option

                menu_ctx = InitialMenuContext(
                    medical_questions=medical_questions,
                    motor_insurance_questions=motor_insurance_questions,
                    car_questions=car_questions,
                    bike_questions=bike_questions,
                    existing_policy_questions=existing_policy_questions,
                    motor_claim=motor_claim,
                    claim_router_questions=claim_router_questions,
                    general_insurance_questions=general_insurance_questions,
                )
                initial_reply = resolve_initial_menu_choice(
                    matched_option,
                    conversation_state=conversation_state,
                    ctx=menu_ctx,
                )
                if initial_reply is not None:
                    response_message, next_options = initial_reply
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )

                if step == STEP_CLAIM_TYPE_CHOICE:
                    if matched_option == "Motor insurance":
                        claim_flow_try_enquiry(
                            user_id=user_id,
                            current_flow=current_flow or CLAIM_ROUTER_FLOW,
                            responses=responses,
                            claim_type="Motor insurance",
                            claim_question_type="Motor insurance",
                        )
                        conversation_state["current_flow"] = MOTOR_CLAIM_FLOW
                        conversation_state["current_question_index"] = 0
                        intro_en = (
                            "Alright 👍 Let's get started on your motor insurance claim. "
                            "I'll guide you step by step so it's quick and easy."
                        )
                        first_q_en = motor_claim[0]["question"]
                        msg_type, doc_type = detect_document_type_from_question(
                            first_q_en
                        )
                        body_en = f"{intro_en}\n\n{first_q_en}"
                        return format_response_in_language(
                            body_en,
                            [],
                            user_language,
                            msg_type,
                            doc_type,
                        )
                    if matched_option == "Medical insurance":
                        responses["claim_type"] = "Medical insurance"
                        conversation_state["current_flow"] = MEDICAL_CLAIM_FLOW
                        conversation_state["current_question_index"] = 0
                        first_med_claim = medical_claim[0]
                        med_body = (
                            first_med_claim["question"]
                            if isinstance(first_med_claim, dict)
                            else str(first_med_claim)
                        )
                        med_opts = (
                            first_med_claim.get("options", [])
                            if isinstance(first_med_claim, dict)
                            else []
                        )
                        return format_response_in_language(
                            med_body, med_opts, user_language
                        )

                if step == STEP_MEDICAL_CLAIM_ASSISTANCE_TYPE:
                    claim_flow_try_enquiry(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                        claim_type=responses.get("claim_type", "Medical insurance"),
                        claim_question_type=matched_option,
                    )
                    conversation_state["current_question_index"] += 1
                    nq = questions[conversation_state["current_question_index"]]
                    next_q = nq["question"] if isinstance(nq, dict) else str(nq)
                    next_opts = nq.get("options", []) if isinstance(nq, dict) else []
                    msg_type, doc_type = detect_document_type_from_question(next_q)
                    return format_response_in_language(
                        next_q, next_opts, user_language, msg_type, doc_type
                    )

                if step == STEP_MOTOR_CLAIM_REPAIR_WORKSHOP:
                    if matched_option == "More":
                        current_page = int(
                            conversation_state.get("motor_claim_repair_page", 0) or 0
                        )
                        conversation_state["motor_claim_repair_page"] = current_page + 1
                        full_options = (
                            question_data.get("options", [])
                            if isinstance(question_data, dict)
                            else []
                        )
                        next_page_options = repair_workshop_paged_options(
                            full_options, conversation_state["motor_claim_repair_page"]
                        )
                        return format_response_in_language(
                            question,
                            next_page_options,
                            user_language,
                        )
                    conversation_state.pop("motor_claim_repair_page", None)
                    responses["motor_claim_insurance_provider"] = matched_option
                    conversation_state["current_question_index"] += 1
                    nq = questions[conversation_state["current_question_index"]]
                    next_body = nq["question"]
                    next_opts = nq.get("options", [])
                    return format_response_in_language(
                        next_body, next_opts, user_language
                    )

                if step == STEP_MOTOR_CLAIM_ROAD_RECOVERY:
                    responses["motor_claim_recover_from_road"] = matched_option
                    if matched_option == "No":
                        # Skip roadside recovery location; go straight to repurchase offer.
                        conversation_state["current_question_index"] += 2
                        nq = questions[conversation_state["current_question_index"]]
                        return format_response_in_language(
                            nq["question"], nq.get("options", []), user_language
                        )
                    conversation_state["current_question_index"] += 1
                    nq = questions[conversation_state["current_question_index"]]
                    return format_response_in_language(
                        nq["question"], nq.get("options", []), user_language
                    )

                if step == STEP_MOTOR_CLAIM_RECOVERY_LOCATION:
                    if matched_option == "More":
                        current_page = int(
                            conversation_state.get("motor_claim_recovery_page", 0)
                            or 0
                        )
                        conversation_state["motor_claim_recovery_page"] = (
                            current_page + 1
                        )
                        full_options = (
                            question_data.get("options", [])
                            if isinstance(question_data, dict)
                            else []
                        )
                        next_page_options = repair_workshop_paged_options(
                            full_options,
                            conversation_state["motor_claim_recovery_page"],
                        )
                        return format_response_in_language(
                            question,
                            next_page_options,
                            user_language,
                        )
                    conversation_state.pop("motor_claim_recovery_page", None)
                    responses["motor_claim_repair_location"] = matched_option
                    responses["motor_claim_roadside_provider"] = matched_option
                    conversation_state["current_question_index"] += 1
                    nq = questions[conversation_state["current_question_index"]]
                    hotline_msg = format_motor_claim_roadside_hotline_message(
                        matched_option
                    )
                    if hotline_msg:
                        return _format_info_then_question(
                            hotline_msg,
                            nq["question"],
                            nq.get("options", []),
                            user_language,
                        )
                    return format_response_in_language(
                        nq["question"], nq.get("options", []), user_language
                    )

                if step == STEP_MOTOR_CLAIM_REPURCHASE_OFFER:
                    if matched_option == "Yes":
                        return _restart_to_initial_menu(
                            user_id=user_id, conversation_state=conversation_state
                        )
                    claim_flow_try_first_step(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                        force=True,
                    )
                    conversation_state["current_question_index"] += 1
                    try:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                    except OSError:
                        pass
                    wipe_flow_session_upload_files(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                    )
                    completion_msg = translate_text(
                        motor_claim_handler._COMPLETE_MESSAGE,
                        user_language,
                    )
                    return {
                        "response": completion_msg,
                        "final_responses": responses,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }

                if step == STEP_MEDICAL_CLAIM_REPURCHASE_OFFER:
                    if matched_option == "Yes":
                        return _restart_to_initial_menu(
                            user_id=user_id, conversation_state=conversation_state
                        )
                    claim_flow_try_first_step(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                        force=True,
                    )
                    conversation_state["current_question_index"] += 1
                    try:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                    except OSError:
                        pass
                    wipe_flow_session_upload_files(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                    )
                    done_msg = translate_text(
                        "Thank you for using Insura. If you need anything else, we are here to help.",
                        user_language,
                    )
                    return {
                        "response": done_msg,
                        "final_responses": responses,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }
            else:
                # Invalid option - provide helpful error in user's language
                error_prompt = f"The user said '{user_message}' but needs to choose from: {', '.join(options)}. Provide a brief, helpful message."
                error_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly insurance assistant. Respond in {user_language}. Be brief and helpful."
                    ),
                    HumanMessage(content=error_prompt),
                ])

                retry_message = f"Let's try again: {question}"
                retry_translated = translate_text(retry_message, user_language)
                translated_options = [
                    translate_text(opt, user_language) for opt in options
                ]

                return {
                    "response": error_response.content.strip(),
                    "question": retry_translated,
                    "options": ", ".join(translated_options),
                }

        if medical_conversation_handler.can_handle_start_question(question):
            return medical_conversation_handler.handle_start_question(
                user_message=user_message,
                question=question,
                user_language=user_language,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                user_id=user_id,
            )

        elif "emaf" in user_message.lower() or "from" in user_message.lower():
            return handle_emaf_document(
                question,
                user_message,
                responses,
                conversation_state,
                questions,
            )
        elif "post a review" in user_message.lower():
            user_message = user_message.lower()  # Convert user_message to lowercase
            if "post a review" in user_message:
                return {
                    "review_message": WEHBE_REVIEW_INVITE_MESSAGE,
                    "review_link": WEHBE_GOOGLE_REVIEW_URL,
                }
            else:
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                # Safely access the next question
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if isinstance(next_question, dict) and "options" in next_question:
                        options = ", ".join(next_question["options"])
                        return {
                            "response": f"{general_assistant_response.content.strip()}",
                            "question": f"Let's try again: {next_question['question']}",
                            "options": options,
                        }
                    else:
                        question_text = (
                            next_question["question"]
                            if isinstance(next_question, dict)
                            else next_question
                        )
                        return {
                            "response": f"{general_assistant_response.content.strip()}",
                            "question": f"Let's try again: {question_text}",
                        }
                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": "It seems we have reached the end of the questions.",
                    }
        elif step == STEP_MOTOR_ENQUIRY_PHONE:
            phone_raw = re.sub(r"[\s\-]", "", user_message.strip())
            if not is_valid_mobile_number(phone_raw):
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                return {
                    "response": translate_text(
                        "Please provide a valid mobile number (10–15 digits).",
                        user_language,
                    ),
                    "question": retry_question,
                }

            responses[question] = phone_raw
            motor_flow_try_enquiry_after_phone(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
                phone=phone_raw,
                display_name=user_name,
            )
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                next_question_text = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else next_question
                )
                next_options = (
                    next_question.get("options", [])
                    if isinstance(next_question, dict)
                    else []
                )
                response_message = translate_text(
                    f"Thank you! Now, let's move on to: {next_question_text}",
                    user_language,
                )
                return format_response_in_language(
                    response_message, next_options, user_language
                )

            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            wipe_flow_session_upload_files(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
            )
            del user_states[user_id]
            result = format_response_in_language(
                "You're all set! Thank you for providing your details. "
                "If you need further assistance, feel free to ask.",
                [],
                user_language,
            )
            result["final_responses"] = responses
            return result

        elif step == STEP_PASSKEY:
            responses[question] = user_message

            if user_message == PASSKEY_VALID_CODE:
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    if "options" in next_question:
                        options = next_question["options"]
                        next_question_text = next_question["question"]
                        response_message = f"Great choice! {next_question_text}"
                        return format_response_in_language(
                            response_message, options, user_language
                        )
                    else:
                        next_question_text = (
                            next_question["question"]
                            if isinstance(next_question, dict)
                            else next_question
                        )
                        response_message = f"Great choice! {next_question_text}"
                        return format_response_in_language(
                            response_message, [], user_language
                        )
                else:
                    # Medical: passkey is the only step in medical_questions → go to individual flow
                    if conversation_state.get("current_flow") == "medical_insurance":
                        conversation_state["current_flow"] = "individual"
                        conversation_state["current_question_index"] = 0
                        first = individual_questions[0]
                        if isinstance(first, dict):
                            return format_response_in_language(
                                first["question"],
                                first.get("options", []),
                                user_language,
                            )
                        return format_response_in_language(
                            str(first), [], user_language
                        )
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                    result = format_response_in_language(
                        final_message, [], user_language
                    )
                    result["final_responses"] = responses
                    return result
            else:
                error_message = f"Incorrect passkey. Please try again. {question}"
                return format_response_in_language(error_message, [], user_language)

        elif step == STEP_EMAF_NAME:
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                next_questions = next_question["question"]
                return {
                    "response": f"Thanks a lot for providing your name! Alright, moving on {next_questions}"
                }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        elif step == STEP_EMAF_PHONE:
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    next_questions = next_question["question"]
                    return {
                        "response": f"Thank you so much! I'd really appreciate it {next_questions}",
                        "dropdown": options,
                    }
                else:
                    return {
                        "response": f"Thank you so much! I'd really appreciate it {next_question}"
                    }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        elif step == STEP_EMAF_COMPANY:
            if user_message in EMAF_INSURANCE_VALID_OPTIONS:
                # Update the Response
                responses[question] = user_message
                selected_company_number = EMAF_COMPANY_NUMBER_BY_OPTION.get(user_message)
                responses["emaf_company_id"] = selected_company_number
                conversation_state["current_question_index"] += 1
                emaf_id = emaf_document(responses)

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    del user_states[user_id]
                    return {
                        "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}"
                    }
                else:
                    if isinstance(emaf_id, int):
                        del user_states[user_id]
                        return {
                            "response": f"Thank you for sharing the details. Please find the link below to view your emaf document:",
                            "link": f"https://www.insuranceclub.ae/medical_form/view/{emaf_id}",
                        }
                    else:
                        return {
                            "response": "Thank you for sharing the details. If you have any questions, please contact support@insuranceclub.ae."
                        }
            else:
                general_assistant_prompt = (
                    f"The user entered '{user_message}', . Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        "options": options,
                    }

                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                    }

        elif step == STEP_MEDICAL_PLAN_TYPE:
            # Use multilingual validation for plan type selection
            valid_options = list(MEDICAL_PLAN_TYPE_OPTIONS)
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif step == STEP_UPLOAD_EMIRATES_DOC:
            try:
                # Try to parse as JSON first, or accept upload path / success message (merged to path).
                document_data = _motor_document_message_to_json(user_message)
                if document_data is None:
                    raise json.JSONDecodeError("Invalid document payload", user_message, 0)
                if _claim_upload_payload_mismatch(document_data, step):
                    return _claim_upload_mismatch_response(
                        payload=document_data,
                        question=question,
                        user_language=user_language,
                    )
                print(f"Parsed document data: {document_data}")

                # Initialize flags if they don't exist
                if "back_page_received" not in responses:
                    responses["back_page_received"] = False
                if "front_page_received" not in responses:
                    responses["front_page_received"] = False

                back_page_question = {
                    "step_id": STEP_UPLOAD_EID_BACK,
                    "question": "Please upload your Emirates ID — Back side",
                }
                front_page_question = {
                    "step_id": STEP_UPLOAD_EID_FRONT,
                    "question": "Please upload your Emirates ID — Front side",
                }
                file_path_hint = (user_input.file_path or "").strip() or (
                    peek_last_upload_relative_path(user_id, "emirates_id") if user_id else ""
                )
                combined_upload = _emirates_upload_contains_both_sides(
                    document_data,
                    file_path=file_path_hint,
                    raw_message=user_message,
                    user_id=user_id,
                )

                if isinstance(document_data, dict):
                    responses[question] = document_data
                    _remember_emirates_payload(
                        responses, document_data, conversation_state
                    )
                    if combined_upload:
                        responses["front_page_received"] = True
                        responses["back_page_received"] = True

                # Determine if we need to ask for additional pages
                if not responses["back_page_received"]:
                    # Need to ask for back page
                    if back_page_question not in questions:
                        questions.insert(
                            conversation_state["current_question_index"] + 1,
                            back_page_question,
                        )
                        responses[back_page_question["question"]] = None

                    response_message = _motor_document_transition_message(
                        back_page_question["question"], user_language
                    )
                    result = format_response_in_language(
                        response_message,
                        [],
                        user_language,
                        message_type="document_upload_request",
                        document_type="emirates_id_back",
                    )
                    return result
                elif not responses["front_page_received"]:
                    # Need to ask for front page
                    if front_page_question not in questions:
                        questions.insert(
                            conversation_state["current_question_index"] + 1,
                            front_page_question,
                        )
                        responses[front_page_question["question"]] = None

                    response_message = _motor_document_transition_message(
                        front_page_question["question"], user_language
                    )
                    result = format_response_in_language(
                        response_message,
                        [],
                        user_language,
                        message_type="document_upload_request",
                        document_type="emirates_id_front",
                    )
                    return result

                merged_payload = _merged_emirates_payload(responses, document_data)
                _motor_quote_try_upload(
                    user_id=user_id,
                    file_path=file_path_hint,
                    current_flow=current_flow,
                    responses=responses,
                    step=step,
                    user_message=json.dumps(merged_payload, ensure_ascii=False),
                )

                # If both pages have been received, continue with normal flow
                conversation_state["current_question_index"] += 1

                # Remove the page questions if they exist in the question list
                for q in [back_page_question, front_page_question]:
                    if q in questions:
                        questions.remove(q)

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    _nq_idx = conversation_state["current_question_index"]
                    _ensure_motor_contact_steps_before_cover(questions, _nq_idx)
                    questions[_nq_idx] = patch_medical_marital_status_question(
                        questions[_nq_idx], responses, conversation_state
                    )
                    next_question = questions[_nq_idx]
                    if isinstance(next_question, dict) and "options" in next_question:
                        options = next_question["options"]
                        next_question_text = next_question["question"]
                        response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                        # Detect document type from next question
                        msg_type, doc_type = detect_document_type_from_question(
                            next_question_text
                        )
                        return format_response_in_language(
                            response_message, options, user_language, msg_type, doc_type
                        )
                    else:
                        next_question_text = (
                            next_question["question"]
                            if isinstance(next_question, dict)
                            else next_question
                        )
                        response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                        # Detect document type from next question
                        msg_type, doc_type = detect_document_type_from_question(
                            next_question_text
                        )
                        return format_response_in_language(
                            response_message, [], user_language, msg_type, doc_type
                        )
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                    result = format_response_in_language(
                        final_message, [], user_language
                    )
                    result["final_responses"] = responses
                    return result

            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )
                return {
                    "response": f"{general_assistant_response.content.strip()} \n\n",
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )
                return {
                    "response": f"{general_assistant_response.content.strip()} \n\n",
                    "example": example_message,
                    "question": retry_question,
                }
        elif step == STEP_UPLOAD_EID_BACK:
            try:
                document_data = _motor_document_message_to_json(user_message)
                if document_data is None:
                    raise json.JSONDecodeError("Invalid document payload", user_message, 0)
                if _claim_upload_payload_mismatch(document_data, step):
                    return _claim_upload_mismatch_response(
                        payload=document_data,
                        question=question,
                        user_language=user_language,
                    )
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    _remember_emirates_payload(
                        responses, document_data, conversation_state
                    )
                    merged_payload = _merged_emirates_payload(responses, document_data)
                    file_path_hint = (user_input.file_path or "").strip() or (
                        peek_last_upload_relative_path(user_id, "emirates_id_back")
                        if user_id
                        else ""
                    )
                    _motor_quote_try_upload(
                        user_id=user_id,
                        file_path=file_path_hint,
                        current_flow=current_flow,
                        responses=responses,
                        step=step,
                        user_message=json.dumps(merged_payload, ensure_ascii=False),
                    )
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        _nq_idx = conversation_state["current_question_index"]
                        _ensure_motor_contact_steps_before_cover(questions, _nq_idx)
                        questions[_nq_idx] = patch_medical_marital_status_question(
                            questions[_nq_idx], responses, conversation_state
                        )
                        next_question = questions[_nq_idx]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = next_question["question"]
                            if next_question.get("step_id") == STEP_MOTOR_COVER_TYPE:
                                response_message = next_question_text
                            else:
                                response_message = (
                                    "Thank you for uploading the document. "
                                    f"Now, let's move on to: {next_question_text}"
                                )
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = (
                                next_question["question"]
                                if isinstance(next_question, dict)
                                else next_question
                            )
                            if (
                                isinstance(next_question, dict)
                                and next_question.get("step_id")
                                == STEP_MOTOR_COVER_TYPE
                            ):
                                response_message = next_question_text
                            else:
                                response_message = (
                                    "Thank you for uploading the document. "
                                    f"Now, let's move on to: {next_question_text}"
                                )
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
        elif step == STEP_UPLOAD_EID_FRONT:
            try:
                document_data = _motor_document_message_to_json(user_message)
                if document_data is None:
                    raise json.JSONDecodeError("Invalid document payload", user_message, 0)
                if _claim_upload_payload_mismatch(document_data, step):
                    return _claim_upload_mismatch_response(
                        payload=document_data,
                        question=question,
                        user_language=user_language,
                    )
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    _remember_emirates_payload(
                        responses, document_data, conversation_state
                    )
                    merged_payload = _merged_emirates_payload(responses, document_data)
                    file_path_hint = (user_input.file_path or "").strip() or (
                        peek_last_upload_relative_path(user_id, "emirates_id_front")
                        if user_id
                        else ""
                    )
                    _motor_quote_try_upload(
                        user_id=user_id,
                        file_path=file_path_hint,
                        current_flow=current_flow,
                        responses=responses,
                        step=step,
                        user_message=json.dumps(merged_payload, ensure_ascii=False),
                    )
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        _nq_idx = conversation_state["current_question_index"]
                        _ensure_motor_contact_steps_before_cover(questions, _nq_idx)
                        questions[_nq_idx] = patch_medical_marital_status_question(
                            questions[_nq_idx], responses, conversation_state
                        )
                        next_question = questions[_nq_idx]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = next_question["question"]
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = (
                                next_question["question"]
                                if isinstance(next_question, dict)
                                else next_question
                            )
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
        elif step in (
            STEP_UPLOAD_DRIVING_LICENSE,
            STEP_UPLOAD_DRIVING_LICENSE_FRONT,
            STEP_UPLOAD_DRIVING_LICENSE_BACK,
        ):
            try:
                document_data = _motor_document_message_to_json(user_message)
                if document_data is None:
                    raise json.JSONDecodeError("Invalid document payload", user_message, 0)
                if _claim_upload_payload_mismatch(document_data, step):
                    return _claim_upload_mismatch_response(
                        payload=document_data,
                        question=question,
                        user_language=user_language,
                    )
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    file_path_hint = (user_input.file_path or "").strip() or (
                        peek_last_upload_relative_path(user_id, "driving_license")
                        if user_id
                        else ""
                    )

                    has_front_before = isinstance(
                        responses.get("_motor_driving_license_front_payload"), dict
                    )
                    side = ""
                    if isinstance(document_data, dict):
                        side = str(document_data.get("side") or "").strip().lower()
                    is_back_upload = step == STEP_UPLOAD_DRIVING_LICENSE_BACK or side in (
                        "back",
                        "rear",
                    )
                    if side == "front":
                        is_back_upload = False
                    elif (
                        not is_back_upload
                        and has_front_before
                        and step
                        in (
                            STEP_UPLOAD_DRIVING_LICENSE,
                            STEP_UPLOAD_DRIVING_LICENSE_FRONT,
                        )
                    ):
                        is_back_upload = True

                    _remember_driving_license_payload(
                        responses, document_data, is_back=is_back_upload
                    )

                    if not _driving_license_upload_complete(
                        responses,
                        document_data,
                        file_path=file_path_hint,
                        raw_message=user_message,
                        user_id=user_id,
                    ):
                        if step in (
                            STEP_UPLOAD_DRIVING_LICENSE_BACK,
                            STEP_UPLOAD_DRIVING_LICENSE_FRONT,
                        ):
                            _rewind_to_driving_license_step(conversation_state, questions)
                        response_message = _motor_document_transition_message(
                            _driving_license_incomplete_prompt(responses),
                            user_language,
                        )
                        return format_response_in_language(
                            response_message,
                            [],
                            user_language,
                            message_type="document_upload_request",
                            document_type="driving_license",
                        )

                    merged_payload = _merged_driving_license_payload(
                        responses, document_data
                    )
                    _motor_quote_try_upload(
                        user_id=user_id,
                        file_path=file_path_hint,
                        current_flow=current_flow,
                        responses=responses,
                        step=step,
                        user_message=json.dumps(merged_payload, ensure_ascii=False),
                    )
                    conversation_state["current_question_index"] += 1
                    _skip_past_driving_license_substeps(conversation_state, questions)

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        _nq_idx = conversation_state["current_question_index"]
                        questions[_nq_idx] = patch_medical_marital_status_question(
                            questions[_nq_idx], responses, conversation_state
                        )
                        next_question = questions[_nq_idx]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = _motor_upload_question_display_text(
                                next_question
                            )
                            response_message = _motor_document_transition_message(
                                next_question_text, user_language
                            )
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = _motor_upload_question_display_text(
                                next_question
                            )
                            response_message = _motor_document_transition_message(
                                next_question_text, user_language
                            )
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        wipe_flow_session_upload_files(
                            user_id=user_id,
                            current_flow=current_flow,
                            responses=responses,
                        )
                        del user_states[user_id]
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }

        elif step in (
            STEP_UPLOAD_MULKIYA,
            STEP_UPLOAD_MULKIYA_FRONT,
            STEP_UPLOAD_MULKIYA_BACK,
        ):
            try:
                document_data = _motor_document_message_to_json(user_message)
                if document_data is None:
                    raise json.JSONDecodeError("Invalid document payload", user_message, 0)
                if _claim_upload_payload_mismatch(document_data, step):
                    return _claim_upload_mismatch_response(
                        payload=document_data,
                        question=question,
                        user_language=user_language,
                    )
                file_path_hint = (user_input.file_path or "").strip()
                responses["Owner in the Vehicle Mulkiya"] = document_data.get("owner")
                responses["Place of Issues in the Vehicle License"] = document_data.get(
                    "place_of_issue"
                )
                responses["Traffic Plate No in the Vehicle License"] = (
                    document_data.get("traffic_plate_no")
                )
                responses["T.C.NO in the Vehicle Mulkiya"] = document_data.get(
                    "nationality"
                )
                responses["Nationality in the Vehicle Mulkiya"] = document_data.get(
                    "nationality"
                )
                responses["Expiry Date in the Vehicle Mulkiya"] = document_data.get(
                    "expiry_date"
                )
                responses["Registertion Date in the Vehicle Mulkiya"] = (
                    document_data.get("reg_date")
                )
                responses["Issues Date in the Vehicle Mulkiya"] = document_data.get(
                    "ins_exp"
                )
                responses["Policy No in the Vehicle Mulkiya"] = document_data.get(
                    "policy_no"
                )
                responses["Model in the Vehicle Mulkiya"] = document_data.get(
                    "model_no"
                )
                responses["Origin in the Vehicle Mulkiya"] = document_data.get("origin")
                responses["Vehicle Type in the Vehicle Mulkiya"] = document_data.get(
                    "vehicle_type"
                )
                responses["Num of pass in the Vehicle Mulkiya"] = document_data.get(
                    "number_of_pass"
                )
                responses["G V M in the Vehicle Mulkiya"] = document_data.get("gvw")
                responses["Empty Weight in the Vehicle Mulkiya"] = document_data.get(
                    "empty_weight"
                )
                responses["Engine Number in the Vehicle Mulkiya"] = document_data.get(
                    "engine_no"
                )
                responses["Chassis Number in the Vehicle Mulkiya"] = document_data.get(
                    "chassis_no"
                )

                print(user_message)
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    print(document_data)
                    _motor_quote_try_upload(
                        user_id=user_id,
                        file_path=file_path_hint,
                        current_flow=current_flow,
                        responses=responses,
                        step=step,
                        user_message=user_message,
                    )

                    if step == STEP_UPLOAD_MULKIYA_FRONT and not _mulkiya_upload_contains_both_sides(
                        document_data,
                        file_path=file_path_hint,
                        raw_message=user_message,
                        user_id=user_id,
                    ):
                        back_question = {
                            "step_id": STEP_UPLOAD_MULKIYA_BACK,
                            "question": (
                                "Please upload your vehicle registration (Mulkiya) — "
                                "Back side"
                            ),
                        }
                        next_index = conversation_state["current_question_index"] + 1
                        next_is_back_question = (
                            next_index < len(questions)
                            and isinstance(questions[next_index], dict)
                            and questions[next_index].get("step_id") == STEP_UPLOAD_MULKIYA_BACK
                        )
                        if not next_is_back_question:
                            questions.insert(next_index, back_question)
                        conversation_state["current_question_index"] += 1
                        result = format_response_in_language(
                            back_question["question"],
                            [],
                            user_language,
                            message_type="document_upload_request",
                            document_type="mulkiya",
                        )
                        return result

                    conversation_state["current_question_index"] += 1

                    insurance_expiry = _parse_vehicle_insurance_expiry_date(
                        document_data.get("ins_exp") or document_data.get("expiry_date")
                    )
                    if insurance_expiry and insurance_expiry < datetime.today().date():
                        _nq_idx = conversation_state["current_question_index"]
                        expired_question = {
                            "step_id": STEP_VEHICLE_TEST_CERT,
                            "question": (
                                "Your insurance has expired, so we need passing paper. "
                                "Please upload it."
                            ),
                        }
                        questions.insert(_nq_idx, expired_question)
                        msg_type, doc_type = detect_document_type_from_question(
                            expired_question["question"]
                        )
                        return format_response_in_language(
                            expired_question["question"],
                            [],
                            user_language,
                            msg_type,
                            doc_type,
                        )

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        _nq_idx = conversation_state["current_question_index"]
                        questions[_nq_idx] = patch_medical_marital_status_question(
                            questions[_nq_idx], responses, conversation_state
                        )
                        next_question = questions[_nq_idx]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = next_question["question"]
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = (
                                next_question["question"]
                                if isinstance(next_question, dict)
                                else next_question
                            )
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        wipe_flow_session_upload_files(
                            user_id=user_id,
                            current_flow=current_flow,
                            responses=responses,
                        )
                        del user_states[user_id]
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }

        elif step == STEP_GENDER_CONFIRM:
            # Use multilingual validation for gender selection
            valid_options = ["Male", "Female"]
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif step == STEP_EMIRATE_CHOICE_MEDICAL:
            conversation_state[CONV_STATE_MEMBER_NAME_KEY] = (
                MEDICAL_MEMBER_NAME_RESPONSE_KEY
            )
            return handle_emirate_upload_document(
                user_message,
                conversation_state,
                questions,
                responses,
                question_data,
                user_language,
            )
        elif step == STEP_EMIRATE_CHOICE_ADDITIONAL_MEMBER:
            conversation_state[CONV_STATE_MEMBER_NAME_KEY] = (
                MEDICAL_ADDITIONAL_MEMBER_NAME_RESPONSE_KEY
            )
            return handle_emirate_upload_document(
                user_message,
                conversation_state,
                questions,
                responses,
                question_data,
                user_language,
            )
        elif step == STEP_VEHICLE_REGISTRATION_TYPE:
            return handle_vehicle_registration_car_question(
                user_message=user_message,
                question=question,
                user_language=user_language,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                user_id=user_id,
                current_flow=current_flow,
            )

        elif step == STEP_EMIRATE_CHOICE_CAR:
            # Use multilingual validation for Yes/No questions
            valid_options = ["Yes", "No"]
            return handle_emirate_upload_document_car_insurance(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif step == STEP_MOTOR_COVER_TYPE:
            valid_options = [
                "Comprehensive",
                "ThirdParty Liability",
                "Know More",
                "Comprehensive (Full Cover)",
                "Third Party",
            ]

            # Use multilingual validation
            validation_result = validate_response_multilingual(
                user_message, valid_options, user_language
            )

            if validation_result["is_valid"]:
                # Store the English version of the response
                english_value = validation_result["matched_value"]
                responses[question] = english_value

                motor_flow_try_second_step_after_cover(
                    user_id=user_id,
                    current_flow=current_flow,
                    responses=responses,
                    cover_choice=english_value,
                )

                if english_value in {"ThirdParty Liability", "Third Party"}:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    wipe_flow_session_upload_files(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                    )
                    del user_states[user_id]
                    final_message = (
                        "Thank you for your interest in ThirdParty Liability insurance! 🙌\n\n"
                        "Our team will contact you shortly to discuss your options and provide personalized quotes.\n\n"
                        "For immediate assistance, please contact us at enquiry@insuranceclub.ae or Call 800 3239"
                    )
                    result = format_response_in_language(
                        final_message, [], user_language
                    )
                    result["final_responses"] = responses
                    return result

                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if "options" in next_question:
                        next_questions_text = next_question["question"]
                        next_options = next_question["options"]
                        response_message = (
                            f"Thank you! Now, let's move on to: {next_questions_text}"
                        )

                        # Format in user's language
                        return format_response_in_language(
                            response_message, next_options, user_language
                        )
                    else:
                        next_question_text = (
                            next_question
                            if isinstance(next_question, str)
                            else next_question.get("question", "")
                        )
                        response_message = (
                            f"Thank you. Now, let's move on to: {next_question_text}"
                        )
                        return format_response_in_language(
                            response_message, [], user_language
                        )
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    wipe_flow_session_upload_files(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                    )
                    del user_states[user_id]
                    if _is_motor_renewal_company_flow(responses):
                        final_message = (
                            f"Thank you, {user_name}💚! 🙌 I'm now checking multiple insurance "
                            "providers to get you the best motor insurance options 🚗\n\n"
                            "⏱️ You'll receive your personalized quotes shortly."
                        )
                        translated_followup = translate_text(
                            "Would you like assistance with anything else?",
                            user_language,
                        )
                        yes_no_options = [
                            translate_text("Yes", user_language),
                            translate_text("No", user_language),
                        ]
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        result["restart_conversation"] = True
                        result["question"] = translated_followup
                        result["options"] = ", ".join(yes_no_options)
                        return result

                    final_message = (
                        "Thank you for sharing the details. We will inform Insura to assist "
                        "you further with your enquiry. Please wait for further assistance. "
                        "If you have any questions, please contact support@insuranceclub.ae."
                    )

                    result = format_response_in_language(
                        final_message, [], user_language
                    )
                    result["final_responses"] = responses
                    return result
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist in a helpful manner. "
                    f"Explain that they need to choose from: {', '.join(valid_options)}"
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly insurance assistant. Respond in {user_language}. "
                        "Help the user understand they need to select a valid option."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                error_message = general_assistant_response.content.strip()
                retry_question = translate_text(
                    f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                    user_language,
                )

                return {
                    "response": error_message,
                    "question": retry_question,
                }

        elif step == STEP_GENERAL_INSURANCE_TYPE:
            page = conversation_state.get("general_insurance_page", 0)
            page_options = get_general_options_page(page)
            validation_result = validate_response_multilingual(
                user_message, page_options, user_language
            )

            if not validation_result["is_valid"]:
                retry_message = translate_text(
                    f"Let's try again: {GENERAL_INSURANCE_PICK_PROMPT}",
                    user_language,
                )
                return format_response_in_language(
                    retry_message, page_options, user_language
                )

            selected_option = validation_result["matched_value"]
            if selected_option == GENERAL_INSURANCE_MORE_OPTION:
                next_page = page + 1
                next_options = get_general_options_page(next_page)
                if not next_options:
                    return format_response_in_language(
                        GENERAL_INSURANCE_PICK_PROMPT, page_options, user_language
                    )

                conversation_state["general_insurance_page"] = next_page
                if (
                    conversation_state["current_question_index"] < len(questions)
                    and isinstance(
                        questions[conversation_state["current_question_index"]], dict
                    )
                ):
                    questions[conversation_state["current_question_index"]][
                        "options"
                    ] = next_options
                return format_response_in_language(
                    GENERAL_INSURANCE_PICK_PROMPT, next_options, user_language
                )

            if selected_option not in GENERAL_INSURANCE_OPTIONS:
                return format_response_in_language(
                    GENERAL_INSURANCE_PICK_PROMPT, page_options, user_language
                )

            responses[question] = selected_option
            responses["General Insurance Type"] = selected_option
            conversation_state.pop("general_insurance_page", None)
            general_flow_try_enquiry_after_type(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
                general_insurance_type=selected_option,
            )

            if selected_option in (
                TRAVEL_INSURANCE_LABEL,
                WORKMEN_COMPENSATION_LABEL,
            ):
                template_xlsx = (
                    TRAVEL_INSURANCE_XLSX
                    if selected_option == TRAVEL_INSURANCE_LABEL
                    else WORKMEN_COMPENSATION_XLSX
                )
                upload_question_en = GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": template_xlsx,
                    "product_display": selected_option,
                    "upload_hint_document_type": "excel",
                    GENERAL_DOC_FIELD_STRICT_WORKBOOK_VALIDATION: True,
                }
                conversation_state["current_question_index"] = len(questions)
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                doc_link = general_document_download_url(template_xlsx)
                base = format_response_in_language(
                    GENERAL_DOC_MSG_TRAVEL_DOWNLOAD,
                    [],
                    user_language,
                )
                q_upload = translate_text(upload_question_en, user_language)
                result = dict(base)
                result["general_link"] = doc_link
                result["question"] = q_upload
                result["message_type"] = "document_upload_request"
                result["document_type"] = "excel"
                result["upload_category"] = GENERAL_UPLOAD_DOCUMENT_CATEGORY
                result["upload_type"] = GENERAL_UPLOAD_TYPE_TRAVEL_FORM
                result["final_responses"] = responses
                return result

            if selected_option == PROPERTY_ALL_RISKS_LABEL:
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_PROPERTY_ALL_RISKS_MENU,
                    "phase": GENERAL_DOC_PHASE_PAR_AWAITING_CHOICE,
                }
                conversation_state["current_question_index"] = len(questions)
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                thank_par = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                pick_par = translate_text(PAR_MSG_PICK_ONE, user_language)
                return format_response_in_language(
                    f"{thank_par}\n\n{pick_par}",
                    list(PAR_SUBMENU_OPTIONS),
                    user_language,
                )

            if selected_option == THIRD_PARTY_LABEL:
                responses["Third Party document"] = THIRD_PARTY_PROPOSAL_PDF
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                tpl_snap = dict(responses)
                tpl_link = general_document_download_url(THIRD_PARTY_PROPOSAL_PDF)
                thank_tpl = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(tpl_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": THIRD_PARTY_PROPOSAL_PDF,
                    "product_display": THIRD_PARTY_LABEL,
                    "upload_hint_document_type": "pdf",
                }
                tpl_base = format_response_in_language(
                    thank_tpl, [], user_language
                )
                tpl_out = dict(tpl_base)
                tpl_out["general_link"] = tpl_link
                tpl_out["message_type"] = "document_download_request"
                tpl_out["document_type"] = "pdf"
                tpl_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                tpl_out["final_responses"] = tpl_snap
                return tpl_out

            if selected_option == CONTRACT_ALL_RISKS_CAR_LABEL:
                responses["Contract All Risks (CAR) document"] = CAR_PROPOSAL_PDF
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                car_snap = dict(responses)
                car_link = general_document_download_url(CAR_PROPOSAL_PDF)
                thank_car = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(car_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": CAR_PROPOSAL_PDF,
                    "product_display": CONTRACT_ALL_RISKS_CAR_LABEL,
                    "upload_hint_document_type": "pdf",
                }
                car_base = format_response_in_language(
                    thank_car, [], user_language
                )
                car_out = dict(car_base)
                car_out["general_link"] = car_link
                car_out["message_type"] = "document_download_request"
                car_out["document_type"] = "pdf"
                car_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                car_out["final_responses"] = car_snap
                return car_out

            if selected_option == PROFESSIONAL_INDEMNITY_LABEL:
                responses["Professional Indemnity document"] = PI_MISC_ANNUAL_DOCX
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                pi_snap = dict(responses)
                pi_link = general_document_download_url(PI_MISC_ANNUAL_DOCX)
                thank_pi = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(pi_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": PI_MISC_ANNUAL_DOCX,
                    "product_display": PROFESSIONAL_INDEMNITY_LABEL,
                    "upload_hint_document_type": "docx",
                }
                pi_base = format_response_in_language(
                    thank_pi, [], user_language
                )
                pi_out = dict(pi_base)
                pi_out["general_link"] = pi_link
                pi_out["message_type"] = "document_download_request"
                pi_out["document_type"] = "docx"
                pi_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                pi_out["final_responses"] = pi_snap
                return pi_out

            if selected_option == INDIVIDUAL_TRAVEL_LABEL:
                responses["Individual Travel document"] = TRAVEL_INSURANCE_XLSX
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                ind_snap = dict(responses)
                ind_link = general_document_download_url(TRAVEL_INSURANCE_XLSX)
                thank_ind = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(ind_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": TRAVEL_INSURANCE_XLSX,
                    "product_display": INDIVIDUAL_TRAVEL_LABEL,
                    "upload_hint_document_type": "excel",
                }
                ind_base = format_response_in_language(
                    thank_ind, [], user_language
                )
                ind_out = dict(ind_base)
                ind_out["general_link"] = ind_link
                ind_out["message_type"] = "document_download_request"
                ind_out["document_type"] = "excel"
                ind_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                ind_out["final_responses"] = ind_snap
                return ind_out

            if selected_option == MARINE_CARGO_INSURANCE_LABEL:
                responses["Marine & Cargo Insurance document"] = (
                    MARINE_CARGO_PROPOSAL_PDF
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                mc_snap = dict(responses)
                mc_link = general_document_download_url(MARINE_CARGO_PROPOSAL_PDF)
                thank_mc = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(mc_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": MARINE_CARGO_PROPOSAL_PDF,
                    "product_display": MARINE_CARGO_INSURANCE_LABEL,
                    "upload_hint_document_type": "pdf",
                }
                mc_base = format_response_in_language(
                    thank_mc, [], user_language
                )
                mc_out = dict(mc_base)
                mc_out["general_link"] = mc_link
                mc_out["message_type"] = "document_download_request"
                mc_out["document_type"] = "pdf"
                mc_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                mc_out["final_responses"] = mc_snap
                return mc_out

            if selected_option == HAULIERS_LIABILITY_INSURANCE_LABEL:
                responses[f"{HAULIERS_LIABILITY_INSURANCE_LABEL} document"] = (
                    HAULIERS_LIABILITY_POLICY_DOC
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _haul_snap = dict(responses)
                _haul_link = general_document_download_url(
                    HAULIERS_LIABILITY_POLICY_DOC
                )
                _haul_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_haul_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": HAULIERS_LIABILITY_POLICY_DOC,
                    "product_display": HAULIERS_LIABILITY_INSURANCE_LABEL,
                    "upload_hint_document_type": "doc",
                }
                _haul_base = format_response_in_language(_haul_th, [], user_language)
                _haul_out = dict(_haul_base)
                _haul_out["general_link"] = _haul_link
                _haul_out["message_type"] = "document_download_request"
                _haul_out["document_type"] = "doc"
                _haul_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _haul_out["final_responses"] = _haul_snap
                return _haul_out

            if selected_option == BOND_PROPOSAL_LABEL:
                responses[f"{BOND_PROPOSAL_LABEL} document"] = (
                    BOND_REQUIREMENTS_AND_FORMS_DOCX
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _bond_snap = dict(responses)
                _bond_link = general_document_download_url(
                    BOND_REQUIREMENTS_AND_FORMS_DOCX
                )
                _bond_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_bond_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": BOND_REQUIREMENTS_AND_FORMS_DOCX,
                    "product_display": BOND_PROPOSAL_LABEL,
                    "upload_hint_document_type": "docx",
                }
                _bond_base = format_response_in_language(_bond_th, [], user_language)
                _bond_out = dict(_bond_base)
                _bond_out["general_link"] = _bond_link
                _bond_out["message_type"] = "document_download_request"
                _bond_out["document_type"] = "docx"
                _bond_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _bond_out["final_responses"] = _bond_snap
                return _bond_out

            if selected_option == DRONE_PROPOSAL_FORM_LABEL:
                responses[f"{DRONE_PROPOSAL_FORM_LABEL} document"] = (
                    DRONE_INSURANCE_XLSX
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _drone_snap = dict(responses)
                _drone_link = general_document_download_url(DRONE_INSURANCE_XLSX)
                _drone_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_drone_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": DRONE_INSURANCE_XLSX,
                    "product_display": DRONE_PROPOSAL_FORM_LABEL,
                    "upload_hint_document_type": "excel",
                }
                _drone_base = format_response_in_language(_drone_th, [], user_language)
                _drone_out = dict(_drone_base)
                _drone_out["general_link"] = _drone_link
                _drone_out["message_type"] = "document_download_request"
                _drone_out["document_type"] = "excel"
                _drone_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _drone_out["final_responses"] = _drone_snap
                return _drone_out

            if selected_option == FIDELITY_PROPOSAL_LABEL:
                responses[f"{FIDELITY_PROPOSAL_LABEL} document"] = (
                    FIDELITY_PROPOSAL_DOC
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _fid_snap = dict(responses)
                _fid_link = general_document_download_url(FIDELITY_PROPOSAL_DOC)
                _fid_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_fid_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": FIDELITY_PROPOSAL_DOC,
                    "product_display": FIDELITY_PROPOSAL_LABEL,
                    "upload_hint_document_type": "doc",
                }
                _fid_base = format_response_in_language(_fid_th, [], user_language)
                _fid_out = dict(_fid_base)
                _fid_out["general_link"] = _fid_link
                _fid_out["message_type"] = "document_download_request"
                _fid_out["document_type"] = "doc"
                _fid_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _fid_out["final_responses"] = _fid_snap
                return _fid_out

            if selected_option == FIRE_FIGHTING_FORM_LABEL:
                responses[f"{FIRE_FIGHTING_FORM_LABEL} document"] = (
                    FIRE_FIGHTING_FACILITIES_DOCX
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _fire_snap = dict(responses)
                _fire_link = general_document_download_url(
                    FIRE_FIGHTING_FACILITIES_DOCX
                )
                _fire_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_fire_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": FIRE_FIGHTING_FACILITIES_DOCX,
                    "product_display": FIRE_FIGHTING_FORM_LABEL,
                    "upload_hint_document_type": "docx",
                }
                _fire_base = format_response_in_language(_fire_th, [], user_language)
                _fire_out = dict(_fire_base)
                _fire_out["general_link"] = _fire_link
                _fire_out["message_type"] = "document_download_request"
                _fire_out["document_type"] = "docx"
                _fire_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _fire_out["final_responses"] = _fire_snap
                return _fire_out

            if selected_option == JEWELLER_PROPOSAL_LABEL:
                responses[f"{JEWELLER_PROPOSAL_LABEL} document"] = (
                    JEWELLERS_BLOCK_PROPOSAL_FORM_DOC
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _jew_snap = dict(responses)
                _jew_link = general_document_download_url(
                    JEWELLERS_BLOCK_PROPOSAL_FORM_DOC
                )
                _jew_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_jew_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": JEWELLERS_BLOCK_PROPOSAL_FORM_DOC,
                    "product_display": JEWELLER_PROPOSAL_LABEL,
                    "upload_hint_document_type": "doc",
                }
                _jew_base = format_response_in_language(_jew_th, [], user_language)
                _jew_out = dict(_jew_base)
                _jew_out["general_link"] = _jew_link
                _jew_out["message_type"] = "document_download_request"
                _jew_out["document_type"] = "doc"
                _jew_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _jew_out["final_responses"] = _jew_snap
                return _jew_out

            if selected_option == MONEY_INSURANCES_LABEL:
                responses[f"{MONEY_INSURANCES_LABEL} document"] = MONEY_INSURANCE_XLSX
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _money_snap = dict(responses)
                _money_link = general_document_download_url(MONEY_INSURANCE_XLSX)
                _money_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_money_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": MONEY_INSURANCE_XLSX,
                    "product_display": MONEY_INSURANCES_LABEL,
                    "upload_hint_document_type": "excel",
                }
                _money_base = format_response_in_language(_money_th, [], user_language)
                _money_out = dict(_money_base)
                _money_out["general_link"] = _money_link
                _money_out["message_type"] = "document_download_request"
                _money_out["document_type"] = "excel"
                _money_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _money_out["final_responses"] = _money_snap
                return _money_out

            if selected_option == VEHICLE_DETAIL_FORM_LABEL:
                responses[f"{VEHICLE_DETAIL_FORM_LABEL} document"] = (
                    VEHICLE_DETAIL_FLEET_FORMAT_XLS
                )
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _veh_snap = dict(responses)
                _veh_link = general_document_download_url(
                    VEHICLE_DETAIL_FLEET_FORMAT_XLS
                )
                _veh_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_veh_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": VEHICLE_DETAIL_FLEET_FORMAT_XLS,
                    "product_display": VEHICLE_DETAIL_FORM_LABEL,
                    "upload_hint_document_type": "xls",
                }
                _veh_base = format_response_in_language(_veh_th, [], user_language)
                _veh_out = dict(_veh_base)
                _veh_out["general_link"] = _veh_link
                _veh_out["message_type"] = "document_download_request"
                _veh_out["document_type"] = "xls"
                _veh_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _veh_out["final_responses"] = _veh_snap
                return _veh_out

            if selected_option == CAR_PROPOSAL_FORM_LABEL:
                responses[f"{CAR_PROPOSAL_FORM_LABEL} document"] = CAR_PROPOSAL_PDF
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _car_form_snap = dict(responses)
                _car_form_link = general_document_download_url(CAR_PROPOSAL_PDF)
                _car_form_th = translate_text(PAR_MSG_THANK_SELECTION, user_language)
                conversation_state["current_flow"] = "general_insurance"
                conversation_state["current_question_index"] = len(questions)
                conversation_state["responses"] = dict(_car_form_snap)
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GENERAL_TEMPLATE,
                    "phase": GENERAL_DOC_PHASE_AWAITING_UPLOAD,
                    "expected_template": CAR_PROPOSAL_PDF,
                    "product_display": CAR_PROPOSAL_FORM_LABEL,
                    "upload_hint_document_type": "pdf",
                }
                _car_form_base = format_response_in_language(
                    _car_form_th, [], user_language
                )
                _car_form_out = dict(_car_form_base)
                _car_form_out["general_link"] = _car_form_link
                _car_form_out["message_type"] = "document_download_request"
                _car_form_out["document_type"] = "pdf"
                _car_form_out["question"] = translate_text(
                    GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN, user_language
                )
                _car_form_out["final_responses"] = _car_form_snap
                return _car_form_out

            if selected_option == CREDIT_INSURANCE_LABEL:
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_CREDIT_INSURANCE_MENU,
                    "phase": GENERAL_DOC_PHASE_CREDIT_AWAITING_CHOICE,
                }
                conversation_state["current_question_index"] = len(questions)
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _credit_intro_th = translate_text(
                    PAR_MSG_THANK_SELECTION, user_language
                )
                _credit_intro_pk = translate_text(PAR_MSG_PICK_ONE, user_language)
                return format_response_in_language(
                    f"{_credit_intro_th}\n\n{_credit_intro_pk}",
                    list(CREDIT_SUBMENU_OPTIONS),
                    user_language,
                )

            if selected_option == GROUP_LIFE_INSURANCE_LABEL:
                conversation_state["general_document_followup"] = {
                    "kind": GENERAL_DOC_KIND_GROUP_LIFE_MENU,
                    "phase": GENERAL_DOC_PHASE_GLI_LEVEL1,
                }
                conversation_state["current_question_index"] = len(questions)
                try:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                except OSError:
                    pass
                _gli_t0 = translate_text(
                    PAR_MSG_THANK_SELECTION, user_language
                )
                _gli_p0 = translate_text(PAR_MSG_PICK_ONE, user_language)
                return format_response_in_language(
                    f"{_gli_t0}\n\n{_gli_p0}",
                    list(GLI_LEVEL1_OPTIONS),
                    user_language,
                )

            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            wipe_flow_session_upload_files(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
            )
            del user_states[user_id]

            result = format_response_in_language(
                GENERAL_FLOW_INSURANCE_CLUB_CLOSING, [], user_language
            )
            result["final_responses"] = responses
            return result

        elif step == STEP_SPONSOR_EMAIL:
            email_value = user_message.strip()
            if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email_value):
                responses["motor_email"] = email_value
                responses[question] = email_value
                motor_flow_try_third_step_after_contact(
                    user_id=user_id,
                    current_flow=current_flow,
                    responses=responses,
                )
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    next_question_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )
                    next_options = (
                        next_question.get("options", [])
                        if isinstance(next_question, dict)
                        else []
                    )
                    response_message = (
                        f"Thank you! Now, let's move on to: {next_question_text}"
                    )
                    msg_type, doc_type = detect_document_type_from_question(
                        next_question_text
                    )
                    return format_response_in_language(
                        response_message,
                        next_options,
                        user_language,
                        msg_type,
                        doc_type,
                    )
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                wipe_flow_session_upload_files(
                    user_id=user_id,
                    current_flow=current_flow,
                    responses=responses,
                )
                del user_states[user_id]
                result = format_response_in_language(
                    "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    [],
                    user_language,
                )
                result["final_responses"] = responses
                return result

            retry_question = translate_text(
                f"Let's try again: {question}", user_language
            )
            return {
                "response": translate_text(
                    "Please enter a valid email address (example: name@example.com).",
                    user_language,
                ),
                "question": retry_question,
            }

        elif step == STEP_UPLOAD_TRADE_LICENSE:
            try:
                document_data = json.loads(user_message)
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    responses["Trade License"] = document_data
                    conversation_state["current_question_index"] += 1
                    next_question = questions[conversation_state["current_question_index"]]
                    next_question_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )
                    next_options = (
                        next_question.get("options", [])
                        if isinstance(next_question, dict)
                        else []
                    )
                    return format_response_in_language(
                        next_question_text, next_options, user_language
                    )
                raise ValueError("Please Upload Again")
            except ValueError:
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                return {
                    "response": translate_text(
                        "Please upload the Trade License document in valid format.",
                        user_language,
                    ),
                    "question": retry_question,
                }

        elif step == STEP_VAT_CERTIFICATE_CHOICE:
            valid_options = ["Yes", "No"]
            validation_result = validate_response_multilingual(
                user_message, valid_options, user_language
            )
            if validation_result["is_valid"]:
                selected = validation_result["matched_value"]
                responses[question] = selected
                conversation_state["current_question_index"] += 1
                if selected == "Yes":
                    _nq_idx = conversation_state["current_question_index"]
                    questions.insert(
                        _nq_idx,
                        {
                            "step_id": STEP_UPLOAD_VAT_CERTIFICATE,
                            "question": "Please upload your VAT Certificate",
                        },
                    )
                next_question = questions[conversation_state["current_question_index"]]
                next_question_text = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else next_question
                )
                next_options = (
                    next_question.get("options", [])
                    if isinstance(next_question, dict)
                    else []
                )
                return format_response_in_language(
                    next_question_text, next_options, user_language
                )
            return {
                "response": translate_text(
                    "Please choose Yes or No.", user_language
                ),
                "question": translate_text(f"Let's try again: {question}", user_language),
            }

        elif step == STEP_UPLOAD_VAT_CERTIFICATE:
            try:
                document_data = json.loads(user_message)
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    responses["VAT Certificate"] = document_data
                    conversation_state["current_question_index"] += 1
                    next_question = questions[conversation_state["current_question_index"]]
                    next_question_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )
                    next_options = (
                        next_question.get("options", [])
                        if isinstance(next_question, dict)
                        else []
                    )
                    return format_response_in_language(
                        next_question_text, next_options, user_language
                    )
                raise ValueError("Please Upload Again")
            except ValueError:
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                return {
                    "response": translate_text(
                        "Please upload the VAT Certificate document in valid format.",
                        user_language,
                    ),
                    "question": retry_question,
                }

        elif (
            motor_identity_response := motor_conversation_handler.handle_vehicle_identity_questions(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                llm=llm,
                system_message_cls=SystemMessage,
                human_message_cls=HumanMessage,
            )
        ) is not None:
            return motor_identity_response

        elif step == STEP_SPONSOR_MOBILE:
            is_mobile_number = is_valid_mobile_number(user_message)

            if is_mobile_number:
                # Store the mobile number
                responses["motor_mobile"] = user_message.strip()
                responses[question] = user_message
                motor_flow_try_third_step_after_contact(
                    user_id=user_id,
                    current_flow=current_flow,
                    responses=responses,
                )
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    # Get the next question text
                    next_question_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )

                    # Translate response to user's language
                    response_msg = translate_text(
                        f"Thank you for providing the mobile number! 📱 Now, let's move on to: {next_question_text}",
                        user_language,
                    )

                    # Handle options if they exist
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
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    wipe_flow_session_upload_files(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                    )
                    if _is_motor_renewal_company_flow(responses):
                        final_message = (
                            f"Thank you, {user_name}💚! 🙌 I'm now checking multiple insurance "
                            "providers to get you the best motor insurance options 🚗\n\n"
                            "⏱️ You'll receive your personalized quotes shortly."
                        )
                        translated_followup = translate_text(
                            "Would you like assistance with anything else?",
                            user_language,
                        )
                        yes_no_options = [
                            translate_text("Yes", user_language),
                            translate_text("No", user_language),
                        ]
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        result["restart_conversation"] = True
                        result["question"] = translated_followup
                        result["options"] = ", ".join(yes_no_options)
                        return result

                    completion_msg = translate_text(
                        "You're all set! 🎉 Thank you for providing your details. If you need further assistance, feel free to ask.",
                        user_language,
                    )
                    return {
                        "response": completion_msg,
                        "final_responses": responses,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }
            else:
                general_assistant_prompt = f"The user entered '{user_message}'. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                next_question = questions[conversation_state["current_question_index"]]
                if isinstance(next_question, dict) and "options" in next_question:
                    next_question_text = next_question["question"]
                    translated_options = [
                        translate_text(opt, user_language)
                        for opt in next_question["options"]
                    ]
                    retry_msg = translate_text(
                        f"Let's Move Back: {next_question_text}", user_language
                    )
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": retry_msg,
                        "options": options,
                    }

                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                    }

        elif step == STEP_CLIENT_NAME:
            return handle_client_name_question(
                question,
                user_message,
                conversation_state,
                questions,
                responses,
                is_valid_name,
            )

        elif step == STEP_CLIENT_MOBILE:
            is_mobile_number = is_valid_mobile_number(user_message)

            if is_mobile_number:
                # Store the mobile number
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    return {
                        "response": f"Thank you for providing the mobile number. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                general_assistant_prompt = (
                    f"The user entered '{user_message}', . Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    next_question = next_question["question"]
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {next_question}",
                        "options": options,
                    }

                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                    }

        elif step in STEP_IDS_MAIN_MENU:
            return handle_what_would_you_do_today_question(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif step == STEP_MARITAL_STATUS_MEMBER:
            # Use multilingual validation for marital status
            valid_options = ["Single", "Married"]
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif step == STEP_SPONSOR_MARITAL_STATUS:
            valid_options = ["Single", "Married"]
            if user_message in valid_options:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you for your response. Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    return {
                        "response": f"Thank you for your response. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif step == STEP_INSURANCE_EXPIRY_YEAR:
            # Store the user-provided year
            if (
                user_message.isdigit() and len(user_message) == 4
            ):  # Ensure valid year format
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions to ask
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                        "options": options,
                    }
                else:
                    # All questions have been answered
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!.",
                        "final_responses": responses,
                    }
            else:
                # Redirect to general assistant for help
                general_assistant_prompt = (
                    f"User response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurances assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's Move back to {question}",
                }

        elif step == STEP_SPONSOR_COMPANY:
            if display_question_matches_current_index(
                questions, conversation_state, question
            ):
                # Check if the input is a company name using LLM
                check_prompt = f"This is the company name: '{user_message}'. Please check if that name could be a company name and respond with 'Yes' or 'No'"
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are a friendly assistant working in Isuran's company department. Your primary task is to verify the user provided input could be a company name. The input might include examples such as 'Fallout Private Limited' or 'Fallout Technologies'. Your role is to validate and identify whether the given input is a valid company name "
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_company_name = llm_response.content.strip().lower() == "yes"

                if is_company_name:
                    # Store the company name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question}"
                        }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = (
                        f"User response: {user_message}. Please assist."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's move back to: {question}",
                    }
        elif step == STEP_POLICY_RENEWAL_CHOICE:
            valid_options = ["Yes", "No"]
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "Yes":
                    # Proceed to the next predefined question
                    conversation_state["current_question_index"] += 1
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_question}"
                        }
                    else:
                        # All predefined questions have been answered
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }

                elif user_message == "No":
                    # Update the responses and return the final response
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for your response. Your request has been updated accordingly. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                return {
                    "response": "Invalid response. Please answer with 'Yes' or 'No'."
                }

        elif step in STEP_IDS_VALIDATE_NAME:
            return handle_validate_name(
                question,
                user_message,
                conversation_state,
                questions,
                responses,
                is_valid_name,
            )

        elif step == STEP_MEMBER_PURCHASE_NAME:
            responses[question] = user_message
            conversation_state["current_question_index"] += 1
            member_name = user_message

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    next_questions = next_question["question"]
                    return {
                        "response": f"Thank you,May I know the {next_questions} of {member_name}.Please ensure it is in the format DD/MM/YYYY.",
                        "options": options,
                    }
                next_questions = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else next_question
                )
                return {
                    "response": f"Thank you,May I know the {next_questions} of {member_name}.Please ensure it is in the format DD/MM/YYYY."
                }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        elif (
            motor_flow_response := motor_conversation_handler.handle_motor_question_set(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
            )
        ) is not None:
            return motor_flow_response

        elif step == STEP_SPONSOR_EMIRATES_ID:
            # Validate sponsor Emirates ID
            if valid_emirates_id(user_message):
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Move to the next question or finalize responses
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
                else:
                    # All questions answered
                    try:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your responses have been recorded. "
                            "Feel free to ask any other questions. Have a great day!",
                            "final_responses": responses,
                        }
                    except Exception as e:
                        return {
                            "response": f"An error occurred while saving your responses: {str(e)}"
                        }
            else:
                # Handle invalid Emirates ID or unrelated query
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])

                # Example of a valid Emirates ID
                emirates_id_example = "784-1990-1234567-0"

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": f"Here's an example of a valid Emirates ID for your reference: {emirates_id_example}.",
                    "question": f"Let's try again: {question}",
                }

        elif step == STEP_VEHICLE_TEST_CERT:
            try:
                document_data = _motor_document_message_to_json(user_message)
                if document_data is None:
                    raise json.JSONDecodeError("Invalid document payload", user_message, 0)
                if _claim_upload_payload_mismatch(document_data, step):
                    return _claim_upload_mismatch_response(
                        payload=document_data,
                        question=question,
                        user_language=user_language,
                    )
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    responses["Vehicle Passing Paper"] = document_data
                    _motor_quote_try_upload(
                        user_id=user_id,
                        file_path=(user_input.file_path or "").strip(),
                        current_flow=current_flow,
                        responses=responses,
                        step=step,
                        user_message=user_message,
                    )
                    conversation_state["current_question_index"] += 1

                    _nq_idx = conversation_state["current_question_index"]
                    is_renewal_company = _is_motor_renewal_company_flow(responses)
                    if is_renewal_company:
                        questions.insert(
                            _nq_idx,
                            {
                                "step_id": STEP_UPLOAD_TRADE_LICENSE,
                                "question": "Passing paper received. Next: Trade License",
                            },
                        )
                        questions.insert(
                            _nq_idx + 1,
                            {
                                "step_id": STEP_VAT_CERTIFICATE_CHOICE,
                                "question": "VAT Certificate (if available)",
                                "options": ["Yes", "No"],
                            },
                        )
                        questions.insert(
                            _nq_idx + 2,
                            {
                                "step_id": STEP_SPONSOR_EMAIL,
                                "question": "May I have the Email Address",
                            },
                        )
                        questions.insert(
                            _nq_idx + 3,
                            {
                                "step_id": STEP_MOTOR_COVER_TYPE,
                                "question": "What type of motor insurance are you looking for?",
                                "options": [
                                    "Comprehensive",
                                    "ThirdPartyLiability",
                                    "Know More",
                                    "Comprehensive (Full Cover)",
                                    "Third Party",
                                ],
                            },
                        )
                    else:
                        questions.insert(
                            _nq_idx,
                            {
                                "step_id": STEP_UPLOAD_DRIVING_LICENSE,
                                "question": MOTOR_DRIVING_LICENSE_COMBINED_UPLOAD_QUESTION,
                            },
                        )
                        questions.insert(
                            _nq_idx + 1,
                            {
                                "step_id": STEP_UPLOAD_EMIRATES_DOC,
                                "question": MOTOR_EMIRATES_ID_COMBINED_UPLOAD_QUESTION,
                            },
                        )
                        questions.insert(
                            _nq_idx + 2,
                            {
                                "step_id": STEP_SPONSOR_EMAIL,
                                "question": "May I have the Email Address",
                            },
                        )

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        if isinstance(next_question, dict):
                            next_question_text = next_question["question"]
                            options = next_question.get("options", [])
                        else:
                            next_question_text = next_question
                            options = []

                        if is_renewal_company:
                            response_message = next_question_text
                        else:
                            response_message = (
                                "Thank you for uploading the passing paper. "
                                f"Now, let's move on to: {next_question_text}"
                            )
                        msg_type, doc_type = detect_document_type_from_question(
                            next_question_text
                        )
                        return format_response_in_language(
                            response_message,
                            options,
                            user_language,
                            msg_type,
                            doc_type,
                        )

                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    wipe_flow_session_upload_files(
                        user_id=user_id,
                        current_flow=current_flow,
                        responses=responses,
                    )
                    del user_states[user_id]
                    final_message = (
                        "You're all set! Thank you for providing your details. "
                        "If you need further assistance, feel free to ask."
                    )
                    result = format_response_in_language(final_message, [], user_language)
                    result["final_responses"] = responses
                    return result
                raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the passing paper is in JPEG format.",
                    user_language,
                )
                return {
                    "response": translate_text(
                        "I could not read the passing paper details. Please upload again.",
                        user_language,
                    ),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError:
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the passing paper is in the correct format and try uploading again.",
                    user_language,
                )
                return {
                    "response": translate_text(
                        "The uploaded passing paper format is invalid. Please try again.",
                        user_language,
                    ),
                    "example": example_message,
                    "question": retry_question,
                }

        elif step == STEP_COMPREHENSIVE_COVER:
            return handle_yes_or_no(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif step == STEP_ADVISOR_CODE_ENTRY:
            medical_response = medical_conversation_handler.handle_medical_question_set(
                question=question,
                user_message=user_message,
                user_id=user_id,
                user_language=user_language,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                user_states=user_states,
                valid_adivisor_code=valid_adivisor_code,
                fetching_medical_detail=fetching_medical_detail,
                translate_text=translate_text,
                get_language_code=get_language_code,
                format_response_in_language=format_response_in_language,
                llm=llm,
                SystemMessage=SystemMessage,
                HumanMessage=HumanMessage,
                insurance_lab_base_url=INSURANCE_LAB_BASE_URL,
            )
            if medical_response is not None:
                return medical_response
        elif (
            medical_dynamic_response := medical_conversation_handler.handle_medical_dynamic_questions(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                llm=llm,
                system_message_cls=SystemMessage,
                human_message_cls=HumanMessage,
            )
        ) is not None:
            return medical_dynamic_response
        elif (
            medical_vaccination_response := medical_conversation_handler.handle_vaccination_questions(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                valid_date_format=valid_date_format,
            )
        ) is not None:
            return medical_vaccination_response
        elif (
            medical_policy_company_response := medical_conversation_handler.handle_current_policy_company_question(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                llm=llm,
                system_message_cls=SystemMessage,
                human_message_cls=HumanMessage,
            )
        ) is not None:
            return medical_policy_company_response
        elif (
            medical_email_response := medical_individual_handler.handle_email_questions(
                question=question,
                user_message=user_message,
                user_language=user_language,
                current_flow=current_flow,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                translate_text=translate_text,
                get_language_code=get_language_code,
                llm=llm,
                system_message_cls=SystemMessage,
                human_message_cls=HumanMessage,
            )
        ) is not None:
            return medical_email_response
        elif (
            medical_identity_response := medical_individual_handler.handle_identity_questions(
                question=question,
                user_message=user_message,
                user_language=user_language,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                translate_text=translate_text,
                translate_to_english_for_storage=translate_to_english_for_storage,
                format_response_in_language=format_response_in_language,
                llm=llm,
                system_message_cls=SystemMessage,
                human_message_cls=HumanMessage,
            )
        ) is not None:
            return medical_identity_response
        elif step == STEP_EXCEL_MEDICAL_UPLOAD:
            if display_question_matches_current_index(
                questions, conversation_state, question
            ):
                # Import user_states from excel_upload module
                from routes.excel_upload import user_states as excel_user_states
                import requests

                # Check if user has uploaded Excel file via the upload-excel endpoint
                excel_data_exists = (
                    user_id in excel_user_states
                    and "responses" in excel_user_states[user_id]
                    and "excel_employee_data" in excel_user_states[user_id]["responses"]
                )

                # Also check for file path patterns (backwards compatibility)
                upload_pattern = re.compile(
                    r"^(?:uploads\/)?(?:[\w\s\-\.\/]+\/)*[\w\s\-\.]+\.(xlsx|xls)$",
                    re.IGNORECASE,
                )

                is_valid_file_path = (
                    upload_pattern.match(user_message)
                    or user_message.lower().endswith((".xlsx", ".xls"))
                    or "uploaded" in user_message.lower()
                    or "excel" in user_message.lower()
                )

                if excel_data_exists or is_valid_file_path:
                    # Valid Excel file format or data exists
                    responses[question] = user_message

                    # Store the Excel file path/confirmation for processing
                    responses["excel_file_path"] = user_message

                    # If Excel data exists, store it in responses
                    if excel_data_exists:
                        responses["excel_employee_data"] = excel_user_states[user_id][
                            "responses"
                        ]["excel_employee_data"]

                        # Get the Excel employee data
                        excel_data = excel_user_states[user_id]["responses"][
                            "excel_employee_data"
                        ]
                        employees_list = excel_data.get("employees", [])
                        print(
                            f"Excel Employees List: {json.dumps(employees_list, indent=2)}"
                        )

                        # Build the members array from Excel data
                        members = []
                        for emp in employees_list:
                            member = {
                                "mem_name": emp.get("first_name", ""),
                                "mem_dob": emp.get("date_of_birth", ""),
                                "mem_gender": emp.get("gender", ""),
                                "mem_marital_status": emp.get("marital_status", ""),
                                "mem_relation": emp.get("relation", ""),
                                "mem_nationality": emp.get("nationality", ""),
                                "mem_emirate": emp.get("visa_issued_location", ""),
                            }
                            members.append(member)
                        print(f"Built Members Array: {json.dumps(members, indent=2)}")

                        # Get the other responses from the conversation
                        visa_issued_emirates = ""
                        plan = ""
                        client_name = ""
                        client_mobile = ""
                        client_email = ""

                        print(f"All Responses: {json.dumps(responses, indent=2)}")

                        # Find these from the responses dictionary
                        for key, value in responses.items():
                            if "Visa issued Emirate" in key or key == MEDICAL_VISA_ISSUED_QUESTION:
                                visa_issued_emirates = value
                            elif (
                                "type of plan" in key.lower()
                                or key == "What type of plan are you looking for?"
                            ):
                                plan = value
                            elif "Client Name" in key:
                                client_name = value
                            elif "Client mobile" in key:
                                client_mobile = value
                            elif "Client Email" in key:
                                client_email = value

                        print(f"Extracted Values:")
                        print(f"  visa_issued_emirates: {visa_issued_emirates}")
                        print(f"  plan: {plan}")
                        print(f"  client_name: {client_name}")
                        print(f"  client_mobile: {client_mobile}")
                        print(f"  client_email: {client_email}")

                        # Validate required fields
                        if (
                            not visa_issued_emirates
                            or not plan
                            or not client_name
                            or not client_mobile
                            or not client_email
                        ):
                            print("ERROR: Missing required fields!")
                            return {
                                "response": "Thank you for uploading the Excel file. However, some required information is missing. Please provide all client details.",
                                "missing_fields": {
                                    "visa_issued_emirates": not visa_issued_emirates,
                                    "plan": not plan,
                                    "client_name": not client_name,
                                    "client_mobile": not client_mobile,
                                    "client_email": not client_email,
                                },
                            }

                        # Prepare the JSON payload
                        payload = {
                            "visa_issued_emirates": visa_issued_emirates,
                            "plan": plan,
                            "client_name": client_name,
                            "client_mobile": client_mobile,
                            "client_email": client_email,
                            "currency": "",
                            "census_sheet": "",
                            "members": members,
                        }

                        # Submit to the API
                        try:
                            # Print the payload for debugging
                            print(f"API Payload: {json.dumps(payload, indent=2)}")
                            print(f"API URL: {INSURANCE_LAB_SME_ADD_API}")

                            # Set proper headers to avoid Mod_Security issues
                            headers = {
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                                "User-Agent": "InsuraBot/1.0",
                            }

                            # Try to send as JSON first
                            try:
                                api_response = requests.post(
                                    INSURANCE_LAB_SME_ADD_API,
                                    json=payload,
                                    headers=headers,
                                    timeout=30,
                                )
                            except:
                                # If JSON fails, try as form data
                                print("JSON request failed, trying form data...")
                                api_response = requests.post(
                                    INSURANCE_LAB_SME_ADD_API,
                                    data=payload,
                                    headers=headers,
                                    timeout=30,
                                )

                            print(f"API Response Status: {api_response.status_code}")
                            print(f"API Response Text: {api_response.text}")

                            if api_response.status_code == 200:
                                response_data = api_response.json()
                                print(
                                    f"API Response Data: {json.dumps(response_data, indent=2)}"
                                )

                                # Store the ID from response if it exists
                                response_id = response_data.get("id", "")
                                print(f"Extracted ID: {response_id}")

                                # Build the customer plan link using the ID
                                customer_plan_link = (
                                    f"{INSURANCE_LAB_SME_PLAN_BASE}/{response_id}"
                                )

                                # Store API response in user state
                                responses["api_response_id"] = response_id
                                responses["api_submission_status"] = "success"
                                responses["customer_plan_link"] = customer_plan_link

                                # Check if there are more questions after this
                                conversation_state["current_question_index"] += 1

                                if conversation_state["current_question_index"] < len(
                                    questions
                                ):
                                    next_question = questions[
                                        conversation_state["current_question_index"]
                                    ]
                                    if isinstance(next_question, dict):
                                        if "options" in next_question:
                                            options = ", ".join(
                                                next_question["options"]
                                            )
                                            next_questions = next_question["question"]
                                            return {
                                                "response": f"Thank you for uploading the Excel file. Your data has been processed successfully (ID: {response_id}). Now, let's move on to: {next_questions}",
                                                "options": options,
                                                "submission_id": response_id,
                                                "customer_plan_link": customer_plan_link,
                                            }
                                        else:
                                            next_questions = next_question["question"]
                                            return {
                                                "response": f"Thank you for uploading the Excel file. Your data has been processed successfully (ID: {response_id}). Now, let's move on to: {next_questions}",
                                                "submission_id": response_id,
                                                "customer_plan_link": customer_plan_link,
                                            }
                                    else:
                                        return {
                                            "response": f"Thank you for uploading the Excel file. Your data has been processed successfully (ID: {response_id}). Now, let's move on to: {next_question}",
                                            "submission_id": response_id,
                                            "customer_plan_link": customer_plan_link,
                                        }
                                else:
                                    # Save responses and end the conversation - SMA flow completion
                                    with open("user_responses.json", "w") as file:
                                        json.dump(responses, file, indent=4)

                                    # Format the response similar to individual flow
                                    success_message = "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please find the link below to view your quotation:"
                                    review_message = WEHBE_REVIEW_INVITE_MESSAGE

                                    # Translate messages if needed
                                    translated_success = (
                                        translate_text(success_message, user_language)
                                        if user_language != "English"
                                        else success_message
                                    )
                                    translated_review = (
                                        translate_text(review_message, user_language)
                                        if user_language != "English"
                                        else review_message
                                    )

                                    # Reset conversation state to allow starting a new inquiry
                                    # Save the language preference before resetting
                                    saved_language = conversation_state.get(
                                        "preferred_language", "English"
                                    )
                                    saved_language_code = conversation_state.get(
                                        "language_code", "en"
                                    )
                                    saved_language_explicitly_set = (
                                        conversation_state.get(
                                            "language_explicitly_set", False
                                        )
                                    )

                                    user_states[user_id] = {
                                        "current_question_index": 0,
                                        "responses": {},
                                        "current_flow": "initial",
                                        "welcome_shown": False,  # Set to False to allow new greeting on restart
                                        "awaiting_document_name": False,
                                        "document_name": "",
                                        "preferred_language": saved_language,  # Preserve language
                                        "language_code": saved_language_code,
                                        "language_explicitly_set": saved_language_explicitly_set,  # Preserve explicit setting
                                    }

                                    return {
                                        "response": translated_success,
                                        "link": customer_plan_link,
                                        "review_message": translated_review,
                                        "review_link": WEHBE_GOOGLE_REVIEW_URL,
                                        "language": user_language,
                                        "language_code": get_language_code(
                                            user_language
                                        ),
                                        "restart_conversation": True,  # Signal to frontend to restart
                                    }
                            else:
                                # API call failed
                                print(
                                    f"API Error - Status: {api_response.status_code}, Response: {api_response.text}"
                                )
                                responses["api_submission_status"] = "error"
                                responses["api_error_message"] = api_response.text
                                return {
                                    "response": f"Thank you for uploading the Excel file. However, there was an issue processing your data (Error: {api_response.status_code}). Please try again or contact support@insuranceclub.ae",
                                    "error_details": api_response.text,
                                }
                        except requests.exceptions.RequestException as e:
                            # Handle request exceptions
                            responses["api_submission_status"] = "error"
                            responses["api_error_message"] = str(e)
                            print(f"API request error: {e}")
                            # Continue with normal flow even if API fails
                            conversation_state["current_question_index"] += 1

                            if conversation_state["current_question_index"] < len(
                                questions
                            ):
                                next_question = questions[
                                    conversation_state["current_question_index"]
                                ]
                                if isinstance(next_question, dict):
                                    options = ", ".join(
                                        next_question.get("options", [])
                                    )
                                    next_questions = next_question.get("question", "")
                                    return {
                                        "response": f"Thank you for uploading the Excel file. There was a temporary issue, but we've saved your data. Now, let's move on to: {next_questions}",
                                        "options": options,
                                    }
                                else:
                                    return {
                                        "response": f"Thank you for uploading the Excel file. Now, let's move on to: {next_question}"
                                    }
                            else:
                                return {
                                    "response": "Thank you for uploading the Excel file. We will inform  Insura to assist you further with your enquiry.",
                                    "final_responses": responses,
                                }
                    else:
                        # Excel data doesn't exist yet - wait for it
                        conversation_state["current_question_index"] += 1

                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]
                            if isinstance(next_question, dict):
                                options = ", ".join(next_question.get("options", []))
                                next_questions = next_question.get("question", "")
                                return {
                                    "response": f"Thank you for sharing the details. We will inform Insura to assist you further with your enquiry. Now, let's move on to: {next_questions}",
                                    "options": options,
                                }
                            else:
                                return {
                                    "response": f"Thank you for sharing the details.  We will inform Insura to  to assist you further with your enquiry. Now, let's move on to: {next_question}"
                                }
                        else:
                            # Save responses and end the conversation - SMA flow completion
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)

                            # Reset conversation state to allow starting a new inquiry
                            user_states[user_id] = {
                                "current_question_index": 0,
                                "responses": {},
                                "current_flow": "initial",
                                "welcome_shown": True,
                                "awaiting_document_name": False,
                                "document_name": "",
                            }

                            return {
                                "response": "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae",
                                "final_responses": responses,
                            }
                else:
                    # Invalid file format
                    return {
                        "response": "The file format seems incorrect. Please upload a valid Excel file (xlsx or xls format) using the upload button."
                    }
        elif step == STEP_ADVISOR_YES_NO:
            # Use the advisor code handler with multilingual support
            return handle_adiviosr_code(
                question,
                user_message,
                responses,
                conversation_state,
                questions,
                user_language,
            )
        elif step == STEP_ADDITIONAL_MEMBER_RESIDENCY:
            residency_vr = validate_response_multilingual(
                user_message,
                list(MEDICAL_RELATIONSHIP_OPTION_KEYS),
                user_language,
            )
            if residency_vr["is_valid"]:
                responses[question] = residency_vr["matched_value"]
                append_additional_medical_member_row(responses)
                conversation_state.pop(CONV_STATE_MEMBER_NAME_KEY, None)
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    _nq_i = conversation_state["current_question_index"]
                    questions[_nq_i] = patch_medical_marital_status_question(
                        questions[_nq_i], responses, conversation_state
                    )
                    next_question = questions[_nq_i]
                    next_question_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )
                    next_options = (
                        next_question.get("options", [])
                        if isinstance(next_question, dict)
                        else []
                    )
                    intro = translate_text(
                        "Thank you! Now, let's move on to:", user_language
                    )
                    response_message = f"{intro} {next_question_text}"
                    msg_type, doc_type = detect_document_type_from_question(
                        next_question_text
                    )
                    return format_response_in_language(
                        response_message,
                        next_options,
                        user_language,
                        msg_type,
                        doc_type,
                    )
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                final_message = translate_text(
                    "You're all set! Thank you for providing your details.",
                    user_language,
                )
                result = format_response_in_language(
                    final_message, [], user_language
                )
                result["final_responses"] = responses
                return result
            err_prompt = (
                f"The user said '{user_message}' but must choose from: "
                f"{', '.join(MEDICAL_RELATIONSHIP_OPTION_KEYS)}. Brief helpful hint."
            )
            err_llm = llm.invoke([
                SystemMessage(
                    content=f"You are Insura. Respond in {user_language}. Be brief."
                ),
                HumanMessage(content=err_prompt),
            ])
            retry_q = translate_text(f"Let's try again: {question}", user_language)
            ropts = [
                translate_text(o, user_language)
                for o in MEDICAL_RELATIONSHIP_OPTION_KEYS
            ]
            return {
                "response": err_llm.content.strip(),
                "question": retry_q,
                "options": ", ".join(ropts),
                "language": user_language,
                "language_code": get_language_code(user_language),
            }

        elif step == STEP_MEDICAL_ADD_ANOTHER_MEMBER:
            add_vr = validate_response_multilingual(
                user_message, ["Yes", "No"], user_language
            )
            if add_vr["is_valid"]:
                responses[question] = add_vr["matched_value"]
                if add_vr["matched_value"] == "Yes":
                    idx = conversation_state["current_question_index"]
                    for j, qdict in enumerate(medical_additional_member_cycle_questions()):
                        questions.insert(idx + 1 + j, qdict)
                    conversation_state["current_question_index"] += 1
                    _ni = conversation_state["current_question_index"]
                    questions[_ni] = patch_medical_marital_status_question(
                        questions[_ni], responses, conversation_state
                    )
                    next_question = questions[_ni]
                    next_question_text = next_question["question"]
                    next_options = next_question.get("options", [])
                    intro = translate_text(
                        "Thank you! Now, let's move on to:", user_language
                    )
                    response_message = f"{intro} {next_question_text}"
                    msg_type, doc_type = detect_document_type_from_question(
                        next_question_text
                    )
                    return format_response_in_language(
                        response_message,
                        next_options,
                        user_language,
                        msg_type,
                        doc_type,
                    )
                return respond_medical_quotation_complete(
                    responses=responses,
                    conversation_state=conversation_state,
                    questions=questions,
                    user_language=user_language,
                    fetching_medical_detail=fetching_medical_detail,
                    translate_text=translate_text,
                    get_language_code=get_language_code,
                    insurance_lab_base_url=INSURANCE_LAB_BASE_URL,
                )
            err_prompt = (
                f"The user said '{user_message}' but must answer Yes or No. Brief hint."
            )
            err_llm = llm.invoke([
                SystemMessage(
                    content=f"You are Insura. Respond in {user_language}. Be brief."
                ),
                HumanMessage(content=err_prompt),
            ])
            retry_q = translate_text(f"Let's try again: {question}", user_language)
            yn = [
                translate_text("Yes", user_language),
                translate_text("No", user_language),
            ]
            return {
                "response": err_llm.content.strip(),
                "question": retry_q,
                "options": ", ".join(yn),
                "language": user_language,
                "language_code": get_language_code(user_language),
            }

        elif step in (STEP_SPONSOR_RELATIONSHIP, STEP_ADDITIONAL_MEMBER_RELATIONSHIP):
            valid_options = list(MEDICAL_RELATIONSHIP_OPTION_KEYS)
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )
        elif step in (
            STEP_MEDICAL_CLAIM_EMIRATES_ID_UPLOAD,
            STEP_MEDICAL_CLAIM_INSURANCE_CARD_UPLOAD,
        ):
            if not _is_claim_upload_input(user_message):
                return {
                    "response": "The file format seems incorrect. Please upload a valid document."
                }
            responses[question] = user_message.strip()
            claim_flow_try_upload_from_message(
                current_flow=current_flow,
                responses=responses,
                user_message=user_message,
                step_id=step,
                file_path=(user_input.file_path or "").strip(),
                user_id=user_id,
            )
            conversation_state["current_question_index"] += 1
            next_question = questions[conversation_state["current_question_index"]]
            next_text = (
                next_question["question"]
                if isinstance(next_question, dict)
                else str(next_question)
            )
            next_opts = (
                next_question.get("options", [])
                if isinstance(next_question, dict)
                else []
            )
            msg_type, doc_type = detect_document_type_from_question(next_text)
            return format_response_in_language(
                next_text, next_opts, user_language, msg_type, doc_type
            )
        elif step == STEP_MEDICAL_CLAIM_CONTACT_MOBILE:
            if not is_valid_mobile_number(user_message):
                gen_prompt = (
                    f"The user entered '{user_message}'. Briefly ask for a valid mobile number."
                )
                gen_llm = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura. Respond in {user_language}. Be brief."
                    ),
                    HumanMessage(content=gen_prompt),
                ])
                retry_q = translate_text(
                    f"Please provide a valid mobile number: {question}",
                    user_language,
                )
                return {
                    "response": gen_llm.content.strip(),
                    "question": retry_q,
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }
            responses[question] = user_message
            store_claim_mobile(responses, user_message)
            claim_flow_try_first_step(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
            )
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[
                    conversation_state["current_question_index"]
                ]
                next_text = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else str(next_question)
                )
                next_opts = (
                    next_question.get("options", [])
                    if isinstance(next_question, dict)
                    else []
                )
                return _format_info_then_question(
                    _medical_claim_registered_message(),
                    next_text,
                    next_opts,
                    user_language,
                )
            claim_flow_try_first_step(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
                force=True,
            )
            done_msg = translate_text(
                "Thank you for using Insura. If you need anything else, we are here to help.",
                user_language,
            )
            return {
                "response": done_msg,
                "final_responses": responses,
                "language": user_language,
                "language_code": get_language_code(user_language),
            }
        elif step == STEP_MOTOR_CLAIM_CONTACT_MOBILE:
            if not is_valid_mobile_number(user_message):
                gen_prompt = (
                    f"The user entered '{user_message}'. Briefly ask for a valid mobile number."
                )
                gen_llm = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura. Respond in {user_language}. Be brief."
                    ),
                    HumanMessage(content=gen_prompt),
                ])
                retry_q = translate_text(
                    f"Please provide a valid mobile number: {question}",
                    user_language,
                )
                return {
                    "response": gen_llm.content.strip(),
                    "question": retry_q,
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }
            responses[question] = user_message
            store_claim_mobile(responses, user_message)
            claim_flow_try_first_step(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
            )
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[
                    conversation_state["current_question_index"]
                ]
                next_text = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else next_question
                )
                next_opts = (
                    next_question.get("options", [])
                    if isinstance(next_question, dict)
                    else []
                )
                if (
                    isinstance(next_question, dict)
                    and resolve_step_id(next_question) == STEP_MOTOR_CLAIM_REPAIR_WORKSHOP
                ):
                    conversation_state["motor_claim_repair_page"] = 0
                    next_opts = repair_workshop_paged_options(next_opts, 0)
                msg_type, doc_type = detect_document_type_from_question(next_text)
                return format_response_in_language(
                    next_text, next_opts, user_language, msg_type, doc_type
                )
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
            except OSError:
                pass
            claim_flow_try_first_step(
                user_id=user_id,
                current_flow=current_flow,
                responses=responses,
                force=True,
            )
            completion_msg = translate_text(
                motor_claim_handler._COMPLETE_MESSAGE,
                user_language,
            )
            return {
                "response": completion_msg,
                "final_responses": responses,
                "language": user_language,
                "language_code": get_language_code(user_language),
            }

        elif (
            claim_upload_response := motor_claim_handler.handle_claim_upload_questions(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                user_language=user_language,
                file_path=(user_input.file_path or "").strip(),
                user_id=user_id,
            )
        ) is not None:
            return claim_upload_response

        elif step == STEP_SPONSOR_TYPE:
            valid_options = ["Employee", "Investors"]
            # Use generic multilingual handler
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif step == STEP_CURRENCY_CHOICE:
            valid_options = ["AED", "USD"]
            if user_message in valid_options:
                responses["question"] = user_message
                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        return {
                            "response": f"Thank you. Now, let's move on to: {next_question}"
                        }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif step == STEP_MONTHLY_SALARY:
            return handle_option_validation_multilingual(
                user_message,
                list(MEDICAL_MONTHLY_SALARY_OPTIONS),
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif motor_conversation_handler.can_handle_start_question(question):
            return motor_conversation_handler.handle_start_question(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
                user_language=user_language,
            )

        elif (
            area_response := motor_conversation_handler.handle_area_preference_question(
                question=question,
                user_message=user_message,
                conversation_state=conversation_state,
                questions=questions,
                responses=responses,
            )
        ) is not None:
            return area_response
        elif step == STEP_MEMBER_DOB:
            return handle_date_question(
                question,
                user_message,
                responses,
                conversation_state,
                questions,
                user_language,
            )
        # For other free-text questions - Use multilingual evaluation

        evaluation_prompt = f"Is the user's response '{user_message}' correct for the question '{question}'? The user is responding in {user_language}. Answer 'yes' or 'no'."
        evaluation_response = llm.invoke([
            SystemMessage(
                content=f"You are evaluating user responses in {user_language}. Consider language variations and cultural context."
            ),
            HumanMessage(content=evaluation_prompt),
        ])
        evaluation = evaluation_response.content.strip().lower()

        if evaluation == "yes":
            # Translate to English for storage if needed
            english_response = translate_to_english_for_storage(
                user_message, user_language
            )
            responses[question] = english_response
            conversation_state["current_question_index"] += 1

            # Check if there are more questions
            if conversation_state["current_question_index"] < len(questions):
                next_question_data = questions[
                    conversation_state["current_question_index"]
                ]
                if isinstance(next_question_data, dict):
                    next_question = next_question_data["question"]
                    next_options = next_question_data.get("options", [])
                    response_message = f"Thank you! That was helpful. Now, let's move on to: {next_question}"

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )
                else:
                    next_question = next_question_data
                    response_message = f"Thank you! That was helpful. Now, let's move on to: {next_question}"

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, [], user_language
                    )
            else:
                # All questions answered
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)

                final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                result = format_response_in_language(final_message, [], user_language)
                result["final_responses"] = responses
                return result
        else:
            # Redirect to general assistant for help in user's language
            general_assistant_prompt = (
                f"User response: {user_message}. Please assist them in {user_language}."
            )
            general_assistant_response = llm.invoke([
                SystemMessage(
                    content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                ),
                HumanMessage(content=general_assistant_prompt),
            ])

            retry_question = translate_text(
                f"Let's Move back to {question}", user_language
            )
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": retry_question,
                "language": user_language,
                "language_code": get_language_code(user_language),
            }
    else:
        # Get user language for general queries
        user_language = conversation_state.get("preferred_language", "English")

        general_assistant_prompt = f"General query: {user_message}."
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])

        return {
            "response": f"{general_assistant_response.content.strip()}",
            "language": user_language,
            "language_code": get_language_code(user_language),
        }


async def clear_user_states_task():
    while True:
        await asyncio.sleep(86400)  # Sleep for 24 hours
        user_states.clear()
        print(f"User states cleared at {datetime.utcnow()}")


def start_clear_user_states_task():
    loop = asyncio.get_event_loop()
    loop.create_task(clear_user_states_task())


# Ensure the task starts when the module is imported
start_clear_user_states_task()
