from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote

STEP_GENERAL_INSURANCE_TYPE = "general_insurance_type"

TRAVEL_INSURANCE_LABEL = "Travel Insurance"
TRAVEL_INSURANCE_XLSX = "Travel Insurance.XLSX"
INDIVIDUAL_TRAVEL_LABEL = "Individual Travel"

WORKMEN_COMPENSATION_LABEL = "Workmen's Compensation"
WORKMEN_COMPENSATION_XLSX = "WORKMEN COMPENSATION.xlsx"

PROPERTY_ALL_RISKS_LABEL = "Property All Risks"
PAR_SUBMENU_PROPERTY_INSURANCES = "Property Insurances"
PAR_SUBMENU_SCOPE_DETAILS = "Scope Details"
PAR_SUBMENU_PROPOSAL_FORM = "Proposal Form"
PAR_SUBMENU_OPTIONS: tuple[str, ...] = (
    PAR_SUBMENU_PROPERTY_INSURANCES,
    PAR_SUBMENU_SCOPE_DETAILS,
    PAR_SUBMENU_PROPOSAL_FORM,
)
# Filenames under ``services/general/documents`` (served at ``/general-documents/``).
PAR_FILE_PROPERTY_INSURANCE_XLSX = "Property Insurance.xlsx"
PAR_FILE_SCOPE_DETAILS_XLSX = "PAR - C O P E DETAILS.xlsx"
PAR_FILE_PROPOSAL_FORM_DOCX = "PAR - PROPOSAL FORM.docx"
PAR_MSG_THANK_SELECTION = "Thank you for your selection 👍"
PAR_MSG_PICK_ONE = "Please select one of the following:"

THIRD_PARTY_LABEL = "Third Party"
THIRD_PARTY_PROPOSAL_PDF = "TPL Proposal form.pdf"

CONTRACT_ALL_RISKS_CAR_LABEL = "Contract All Risks (CAR)"
CAR_PROPOSAL_PDF = "CAR Proposal From.pdf"
CAR_PROPOSAL_FORM_LABEL = "Car proposal form"

PROFESSIONAL_INDEMNITY_LABEL = "Professional Indemnity"
PI_MISC_ANNUAL_DOCX = "PI - MISC annual.docx"

MARINE_CARGO_INSURANCE_LABEL = "Marine & Cargo Insurance"
MARINE_CARGO_PROPOSAL_PDF = "Marine Cargo Proposal Form .pdf"

HAULIERS_LIABILITY_INSURANCE_LABEL = "Hauliers' Liability Insurance"
HAULIERS_LIABILITY_POLICY_DOC = "HAULIERS LIABILITY INSURANCE POLICY.doc"

BOND_PROPOSAL_LABEL = "BOND PROPOSAL"
BOND_REQUIREMENTS_AND_FORMS_DOCX = "BOND REQUIREMENTS AND FORMS.docx"

DRONE_PROPOSAL_FORM_LABEL = "Drone Proposal Form"
DRONE_INSURANCE_XLSX = "DRONE -INSURANCE.xlsx"

FIDELITY_PROPOSAL_LABEL = "Fidelity Proposal"
FIDELITY_PROPOSAL_DOC = "FIDELITY PROPOSAL.DOC"

FIRE_FIGHTING_FORM_LABEL = "Fire Fighting Form"
FIRE_FIGHTING_FACILITIES_DOCX = "Fire Fighting Facilities.docx"

JEWELLER_PROPOSAL_LABEL = "Jeweller proposal"
JEWELLERS_BLOCK_PROPOSAL_FORM_DOC = "Jewellers Block Proposal Form.doc"

MONEY_INSURANCES_LABEL = "Money insurances"
MONEY_INSURANCE_XLSX = "Money Insurance.xlsx"

VEHICLE_DETAIL_FORM_LABEL = "Vehicle detail form"
VEHICLE_DETAIL_FLEET_FORMAT_XLS = "Vehicle Detail- Fleet Format.xls"

GROUP_LIFE_INSURANCE_LABEL = "Group Life Insurance"
GLI_OPTION_HEALTH = "Health insurance"
GLI_OPTION_TRAVEL = "Travel insurances"
GLI_OPTION_ACCIDENT = "Accident insurances"
GLI_OPTION_MALPRACTICING = "Malpracticing insurances"
GLI_LEVEL1_OPTIONS: tuple[str, ...] = (
    GLI_OPTION_HEALTH,
    GLI_OPTION_TRAVEL,
    GLI_OPTION_ACCIDENT,
    GLI_OPTION_MALPRACTICING,
)
GLI_FILE_GROUP_HEALTH = "Group Health Insurance Data.xlsx"
GLI_FILE_GROUP_TRAVEL = "Group travel Insurance.XLSX"
GLI_FILE_GLPA_CENSUS = "GLPA Census.xls"
GLI_MALP_STAFF_LIST = "Staff list"
GLI_MALP_PROPOSAL = "Malpractices proposal"
GLI_MALP_SUBMENU_OPTIONS: tuple[str, ...] = (
    GLI_MALP_STAFF_LIST,
    GLI_MALP_PROPOSAL,
)
GLI_FILE_MMP_STAFF_LIST = "MMP Staff list.xlsx"
GLI_FILE_MALP_ESTABLISHMENTS = "Medical Malpractice - Establishments.doc"

CREDIT_INSURANCE_LABEL = "Credit Insurance"
CREDIT_SUBMENU_CREDIT_INSURANCES = "Credit insurances"
CREDIT_SUBMENU_SINGLE_CREDIT = "Single credit"
CREDIT_SUBMENU_OPTIONS: tuple[str, ...] = (
    CREDIT_SUBMENU_CREDIT_INSURANCES,
    CREDIT_SUBMENU_SINGLE_CREDIT,
)
CREDIT_FILE_CREDIT_INSURANCES_XLSX = "Credit Insurance 1.xlsx"
CREDIT_FILE_SINGLE_RISK_DOC = "Credit Insurance Single Risk Application Form.doc"


GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN = "Please fill and upload the form."

GENERAL_DOC_MSG_WRONG_UPLOAD_USE_FORM_NOT_TRADE_VAT_EN = (
    "For this step, please upload your filled form "
    "(not your trade licence or VAT certificate yet)."
)
GENERAL_DOC_MSG_RETRY_FILLED_FORM_EN = (
    "We could not confirm your upload. Please upload your filled form again."
)


def general_upload_form_question_en(product_display_name: str, document_type: str) -> str:
    """
    Same prompt for every general-insurance template step (download → upload).
    Arguments kept for backward-compatible call sites.
    """
    return GENERAL_DOC_MSG_UPLOAD_GENERIC_FORM_EN


# Single follow-up kind for “download template → upload → trade licence → …”.
# Use ``strict_workbook_validation`` on the state dict for Travel / Workmen Excel checks only.
GENERAL_DOC_KIND_GENERAL_TEMPLATE = "general_template"
GENERAL_DOC_FIELD_STRICT_WORKBOOK_VALIDATION = "strict_workbook_validation"
GENERAL_DOC_KIND_PROPERTY_ALL_RISKS_MENU = "property_all_risks_menu"
GENERAL_DOC_KIND_GROUP_LIFE_MENU = "group_life_menu"
GENERAL_DOC_KIND_CREDIT_INSURANCE_MENU = "credit_insurance_menu"
GENERAL_DOC_PHASE_PAR_AWAITING_CHOICE = "awaiting_par_choice"
GENERAL_DOC_PHASE_CREDIT_AWAITING_CHOICE = "awaiting_credit_choice"
GENERAL_DOC_PHASE_GLI_LEVEL1 = "awaiting_gli_level1"
GENERAL_DOC_PHASE_GLI_MALPRACTICE = "awaiting_gli_malpractice_sub"
GENERAL_DOC_PHASE_DOWNLOAD_SHOWN = "download_shown"
GENERAL_DOC_PHASE_AWAITING_UPLOAD = "awaiting_upload"
GENERAL_DOC_PHASE_AWAITING_TRADE_LICENCE = "awaiting_trade_licence"
GENERAL_DOC_PHASE_AWAITING_VAT_AVAILABLE = "awaiting_vat_available"
GENERAL_DOC_PHASE_AWAITING_VAT_CERTIFICATE = "awaiting_vat_certificate"
GENERAL_DOC_PHASE_AWAITING_EXISTING_POLICY = "awaiting_existing_policy"
GENERAL_DOC_PHASE_AWAITING_POLICY_SCHEDULE_DATE = "awaiting_policy_schedule_date"
GENERAL_DOC_PHASE_AWAITING_COMPANY_EMAIL = "awaiting_company_email"
GENERAL_DOC_PHASE_AWAITING_COMPANY_NAME = "awaiting_company_name"
GENERAL_DOC_PHASE_AWAITING_FULL_NAME = "awaiting_full_name"
GENERAL_DOC_PHASE_AWAITING_DESIGNATION = "awaiting_designation"
GENERAL_DOC_PHASE_AWAITING_SPECIALIST_CHOICE = "awaiting_specialist_choice"
GENERAL_DOC_PHASE_AWAITING_SPECIALIST_PHONE = "awaiting_specialist_phone"
GENERAL_DOC_PHASE_AFTER_WAIT_CLOSING = "after_wait_closing"
GENERAL_DOC_PHASE_AWAITING_ANYTHING_ELSE = "awaiting_anything_else"

GENERAL_DOC_MSG_TRAVEL_DOWNLOAD = (
    "Thank you for your selection 👍 Please download the form using the link below."
)
GENERAL_DOC_MSG_FILL_AND_UPLOAD = (
    "Please fill the document below and upload it back once completed."
)
# Travel Excel upload step — surfaced via ``question`` (same pattern as repurchase Yes/No: ``response`` + ``question``).
GENERAL_DOC_MSG_TRAVEL_FORM_UPLOAD_QUESTION_EN = general_upload_form_question_en(
    TRAVEL_INSURANCE_LABEL, "excel"
)
GENERAL_DOC_MSG_WORKMEN_FORM_UPLOAD_QUESTION_EN = general_upload_form_question_en(
    WORKMEN_COMPENSATION_LABEL, "excel"
)

# Reference roadmap (documentation / future use); primary UX uses ``question`` per step.
TRAVEL_INSURANCE_FLOW_NEXT_QUESTIONS_EN = (
    GENERAL_DOC_MSG_TRAVEL_FORM_UPLOAD_QUESTION_EN,
    "Please upload your trade licence.",
    "VAT certificate: answer Yes or No; upload the certificate if you answer Yes.",
    "Existing policy: answer Yes or No; share the latest policy schedule date if you answer Yes.",
    "Provide your official company email address.",
    "Provide company name, your full name, and your designation.",
    "Review the summary, then choose Connect me or Wait.",
    "If you chose Connect me, share your phone number.",
)
GENERAL_DOC_MSG_TRAVEL_UPLOAD_RECEIVED = (
    "Thank you for uploading the filled document. We've received it."
)
GENERAL_DOC_MSG_TRADE_LICENCE_UPLOAD = "Please upload your trade licence."
GENERAL_DOC_MSG_TRADE_LICENCE_THANK = "Thank you! Trade licence uploaded."
GENERAL_DOC_MSG_VAT_AVAILABLE = "Is VAT certificate available?"
GENERAL_DOC_MSG_VAT_UPLOAD = "Please upload your VAT certificate."
GENERAL_DOC_MSG_VAT_THANK = "Thank you! VAT certificate uploaded."
GENERAL_DOC_MSG_EXISTING_POLICY = "Do you have an existing policy?"
GENERAL_DOC_MSG_POLICY_SCHEDULE = "Please share the latest policy schedule date."
GENERAL_DOC_MSG_POLICY_SCHEDULE_THANK = "Thank you! Policy schedule date received."
GENERAL_DOC_MSG_COMPANY_EMAIL_INTRO = (
    "To proceed, may I kindly request that you share your official company email address?"
)
GENERAL_DOC_MSG_COMPANY_EMAIL_PROMPT = (
    "Please share your official email address.\n(Example: name@company.com)"
)
GENERAL_DOC_MSG_EMAIL_THANK = "Thank you."
GENERAL_DOC_MSG_COMPANY_NAME = "May I have your Company Name, please?"
GENERAL_DOC_MSG_FULL_NAME = "Your Full Name, please?"
GENERAL_DOC_MSG_DESIGNATION = "Your Designation / Position in the company?"
GENERAL_DOC_MSG_DETAILS_CLOSING = (
    "Thank you for the details.\n"
    "You will shortly receive an official email from Insura requesting the documents "
    "and information required to prepare your insurance proposal.\n"
    "Kindly share the requested details via email to ensure proper documentation and compliance.\n\n"
    "Once our team reviews your information and works on the quotation, 🔔 You will be notified "
    "here on WhatsApp regarding the next steps. If you require immediate assistance, I can connect "
    "you with one of our corporate insurance specialists."
)
GENERAL_DOC_OPT_CONNECT_SPECIALIST = "Connect me"
GENERAL_DOC_OPT_WAIT = "Wait"
GENERAL_DOC_MSG_PHONE_FOR_SPECIALIST = (
    "Please share your phone number so we can connect you with our specialist."
)
GENERAL_DOC_MSG_SPECIALIST_FORWARDED = (
    "Thank you! We have forwarded your details to our staff. "
    "One of our corporate insurance specialists will connect with you soon."
)
# Shown with ``Would you like assistance…`` right after the user taps Wait (specialist step).
GENERAL_FLOW_INSURANCE_CLUB_CLOSING = (
    "Thank you for contacting InsuranceClub.\n"
    "We appreciate the opportunity to support your corporate insurance requirements.\n"
    "Should you need any further assistance, please feel free to reach out at any time."
)
GENERAL_DOC_MSG_ASSISTANCE_ANYTHING_ELSE = "Would you like assistance with anything else?"
# After Wait path: user already saw ``GENERAL_FLOW_INSURANCE_CLUB_CLOSING``; use this on Anything else → No.
GENERAL_DOC_MSG_ANYTHING_ELSE_DECLINED_SIGNOFF_EN = (
    "Thank you for contacting InsuranceClub. We hope to serve you again soon."
)
GENERAL_DOC_MSG_WAIT_COMPLETE = GENERAL_FLOW_INSURANCE_CLUB_CLOSING
GENERAL_DOC_YES_NO_OPTIONS = ("Yes", "No")
GENERAL_DOC_SPECIALIST_OPTIONS = (
    GENERAL_DOC_OPT_CONNECT_SPECIALIST,
    GENERAL_DOC_OPT_WAIT,
)

# Client hint for POST ``/upload-document/``: same ``type`` values (general-insurance travel flow).
GENERAL_UPLOAD_DOCUMENT_CATEGORY = "general_insurance"
GENERAL_UPLOAD_TYPE_TRAVEL_FORM = "general_insurance_form"
GENERAL_UPLOAD_TYPE_TRADE_LICENCE = "trade_license"
GENERAL_UPLOAD_TYPE_VAT_CERTIFICATE = "vat_certificate"

GENERAL_DOCUMENTS_MOUNT_PREFIX = "/general-documents"


def parse_general_upload_payload(
    user_message: str, *, is_document_upload_success: bool
) -> tuple[bool, Any]:
    """Parse client upload: explicit success string or JSON extraction payload."""
    if is_document_upload_success:
        return True, {"_upload": "client_success_signal", "detail": user_message.strip()}
    try:
        data = json.loads(user_message.strip())
        if isinstance(data, dict) and data:
            return True, data
    except json.JSONDecodeError:
        pass
    return False, None


def validate_travel_insurance_completed_payload(payload: Any) -> bool:
    """Light validation that upload looks like the completed Travel Insurance template."""
    if not isinstance(payload, dict) or not payload:
        return False
    if payload.get("_upload") == "client_success_signal":
        return True
    blob = json.dumps(payload).lower()
    hints = (
        "travel",
        "trip",
        "destination",
        "passport",
        "insured",
        "visa",
        "departure",
        "arrival",
        "policy",
        "premium",
        "nominee",
        "duration",
        "workmen",
        "compensation",
        "employer",
        "employee",
        "wages",
        "injury",
    )
    if any(h in blob for h in hints):
        return True
    filled = sum(
        1 for v in payload.values() if v not in (None, "", [], {}, "null")
    )
    return filled >= 4


def validate_any_nonempty_upload_payload(payload: Any) -> bool:
    """Accept trade licence / VAT uploads: client success or non-empty extracted JSON."""
    if not isinstance(payload, dict) or not payload:
        return False
    if payload.get("_upload") == "client_success_signal":
        return True
    return (
        sum(1 for v in payload.values() if v not in (None, "", [], {}, "null")) >= 1
    )


def normalized_upload_payload_file_type(payload: Any) -> str | None:
    """``file_type`` from multipart/json upload payloads (see ``upload-document``)."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("file_type")
    if raw is None or str(raw).strip() == "":
        return None
    from services.general.document_upload_service import (
        normalize_general_upload_file_type,
    )

    return normalize_general_upload_file_type(str(raw))


def should_accept_travel_form_upload(payload: Any) -> bool:
    """Travel Excel step: reject uploads explicitly tagged as trade or VAT."""
    if not validate_travel_insurance_completed_payload(payload):
        return False
    nft = normalized_upload_payload_file_type(payload)
    if nft in ("trade_license", "vat_certificate"):
        return False
    return True


def should_accept_general_template_form_upload(payload: Any) -> bool:
    """
    Non–travel/workmen templates (PDF/Word/other): accept any substantive upload
    except trade licence / VAT when explicitly tagged.
    """
    if not validate_any_nonempty_upload_payload(payload):
        return False
    nft = normalized_upload_payload_file_type(payload)
    if nft in ("trade_license", "vat_certificate"):
        return False
    return True


def should_accept_trade_licence_upload(payload: Any) -> bool:
    """
    Trade-licence step: require ``file_type=trade_license`` when present.
    Rejects another Travel form upload (common bug: repeating ``general_insurance_form``).
    """
    if not validate_any_nonempty_upload_payload(payload):
        return False
    nft = normalized_upload_payload_file_type(payload)
    if nft == "travel_insurance_form":
        return False
    if nft == "vat_certificate":
        return False
    if nft == "trade_license":
        return True
    if isinstance(payload, dict) and payload.get("_upload") == "client_success_signal":
        return True
    return False


def should_accept_vat_certificate_upload(payload: Any) -> bool:
    """VAT upload step: require ``file_type=vat_certificate`` when typed."""
    if not validate_any_nonempty_upload_payload(payload):
        return False
    nft = normalized_upload_payload_file_type(payload)
    if nft in ("travel_insurance_form", "trade_license"):
        return False
    if nft == "vat_certificate":
        return True
    if isinstance(payload, dict) and payload.get("_upload") == "client_success_signal":
        return True
    return False


def match_yes_no_english(user_message: str) -> str | None:
    """
    Fast path when the client sends plain Yes/No (buttons often mirror English labels).
    Returns "Yes", "No", or None when ambiguous; caller may fall back to LLM validation.
    """
    raw = (user_message or "").strip().lower()
    if raw in {"yes", "y", "yeah", "yep"}:
        return "Yes"
    if raw in {"no", "n", "nope", "nah"}:
        return "No"
    return None


def general_document_download_url(filename: str) -> str:
    """Public URL path for a file under ``services/general/documents``."""
    path = f"{GENERAL_DOCUMENTS_MOUNT_PREFIX}/{quote(filename)}"
    base = os.getenv("INSURA_PUBLIC_BASE_URL", "").strip().rstrip("/")
    return f"{base}{path}" if base else path


GENERAL_INSURANCE_INTRO = (
    "Thank you for choosing General Insurance with InsuranceClub.\n\n"
    "Please select the type of General Insurance you are looking for from the options below.\n\n"
    "Kindly select one of the following:"
)
GENERAL_INSURANCE_PICK_PROMPT = "Kindly select one of the following:"
GENERAL_INSURANCE_PAGE_SIZE = 8
GENERAL_INSURANCE_MORE_OPTION = "More"

GENERAL_INSURANCE_OPTIONS = [
    "Travel Insurance",
    "Workmen's Compensation",
    "Property All Risks",
    "Third Party",
    "Contract All Risks (CAR)",
    "Professional Indemnity",
    "Individual Travel",
    "Marine & Cargo Insurance",
    "Group Life Insurance",
    "Credit Insurance",
    "Hauliers' Liability Insurance",
    "BOND PROPOSAL",
    "Drone Proposal Form",
    "Fidelity Proposal",
    "Fire Fighting Form",
    "Jeweller proposal",
    "Money insurances",
    "Vehicle detail form",
    "Car proposal form",
]


def get_general_options_page(page: int) -> list[str]:
    start = page * GENERAL_INSURANCE_PAGE_SIZE
    end = start + GENERAL_INSURANCE_PAGE_SIZE
    options = GENERAL_INSURANCE_OPTIONS[start:end]
    if end < len(GENERAL_INSURANCE_OPTIONS):
        options.append(GENERAL_INSURANCE_MORE_OPTION)
    return options
