"""InsuranceLab API submission for general insurance."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import requests

from services.general.flow import normalized_upload_payload_file_type

CONTENT_TYPE_JSON = "application/json"

INSURANCE_LAB_BASE_URL = (
    os.getenv("GENERAL_INSURANCELAB_BASE_URL", "https://insurancelab.ae")
    .strip()
    .rstrip("/")
)
GENERAL_ENQUIRY_URL = f"{INSURANCE_LAB_BASE_URL}/Api/general_enquiry"
GENERAL_UPLOAD_DOCUMENT_URL = f"{INSURANCE_LAB_BASE_URL}/Api/general_upload_document"
GENERAL_FIRST_STEP_URL = f"{INSURANCE_LAB_BASE_URL}/Api/general_first_step"

GENERAL_SERVICE_TYPE = "General Insurance"
GENERAL_QUOTE_FLOWS = frozenset({"general_insurance"})

_DOC_TITLE_BY_FILE_TYPE: dict[str, str] = {
    "travel_insurance_form": "proposal_form",
    "trade_license": "trade_license",
    "vat_certificate": "vat_certificate",
}


class GeneralAPISubmission:
    """Handle general insurance API submission."""

    def submit_general_enquiry(
        self,
        user_id: str,
        responses: dict[str, Any],
        service_type: str,
        general_insurance_type: str,
    ) -> bool:
        """Submit selected general insurance details to API."""
        payload = {
            "user_id": user_id,
            "service_type": service_type,
            "general_insurance_type": general_insurance_type,
        }
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }

        try:
            res = requests.post(
                GENERAL_ENQUIRY_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(
                f"General enquiry API response: {json.dumps(response_data, indent=2)}"
            )
            enquiry_id = response_data.get("id") or response_data.get("result", {}).get(
                "id"
            )
            if enquiry_id:
                responses["general_enquiry_id"] = str(enquiry_id)
            responses["general_enquiry_stored"] = True
            responses["general_enquiry_api_response"] = response_data
            return True
        except requests.RequestException as e:
            print(f"Error calling general_enquiry API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            responses["general_enquiry_stored"] = False
            responses["general_enquiry_api_error"] = str(e)
            return False

    def upload_general_document(
        self,
        responses: dict[str, Any],
        doc_title: str,
        document_data: bytes,
        filename: str,
        parsed_text: str = "",
    ) -> bool:
        """Upload general flow document to API."""
        enquiry_id = responses.get("general_enquiry_id")
        if not enquiry_id:
            print("Error: general_enquiry_id not found. Cannot upload document.")
            return False

        files = {"document": (filename, document_data)}
        data = {
            "id": enquiry_id,
            "doc_title": doc_title,
            "file_name": filename,
            "parsed_text": parsed_text,
        }
        try:
            res = requests.post(
                GENERAL_UPLOAD_DOCUMENT_URL, files=files, data=data, timeout=30
            )
            res.raise_for_status()
            print(f"General upload document API response: {res.text}")
            return True
        except requests.RequestException as e:
            print(f"Error calling general_upload_document API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            return False

    def submit_general_first_step(self, responses: dict[str, Any]) -> bool:
        """Submit final general form details to API."""
        enquiry_id = responses.get("general_enquiry_id")
        if not enquiry_id:
            print("Error: general_enquiry_id not found. Cannot call general_first_step.")
            return False

        payload = {
            "id": str(enquiry_id),
            "general_existing_policy": _field(
                responses, "general_existing_policy", "Existing policy"
            ),
            "general_policy_schedule_date": _field(
                responses, "general_policy_schedule_date", "Policy schedule date"
            ),
            "general_company_email": _field(
                responses, "general_company_email", "Official company email"
            ),
            "general_company_name": _field(
                responses, "general_company_name", "Company name"
            ),
            "general_full_name": _field(
                responses, "general_full_name", "Full name"
            ),
            "general_designation": _field(
                responses, "general_designation", "Designation"
            ),
            "general_specialist_phone": _field(
                responses,
                "general_specialist_phone",
                "Specialist contact phone",
            ),
        }
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            res = requests.post(
                GENERAL_FIRST_STEP_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(
                f"General first step API response: {json.dumps(response_data, indent=2)}"
            )
            responses["general_first_step_stored"] = True
            responses["general_first_step_api_response"] = response_data
            return True
        except requests.RequestException as e:
            print(f"Error calling general_first_step API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            responses["general_first_step_stored"] = False
            responses["general_first_step_api_error"] = str(e)
            return False


def _field(responses: dict[str, Any], api_key: str, display_key: str) -> str:
    val = responses.get(api_key)
    if val is None or (isinstance(val, str) and not val.strip()):
        val = responses.get(display_key)
    if val is None:
        return ""
    return str(val).strip()


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_upload_file_path(relative: str) -> Optional[str]:
    norm = (relative or "").strip().replace("\\", "/")
    if not norm or ".." in norm.split("/"):
        return None
    if os.path.isabs(norm) and os.path.isfile(norm):
        return norm
    if os.path.isfile(norm):
        return norm
    candidate = os.path.join(_repo_root(), norm)
    if os.path.isfile(candidate):
        return candidate
    return None


def _load_bytes_from_payload(payload: dict[str, Any]) -> Optional[tuple[bytes, str]]:
    path = (
        payload.get("stored_path")
        or payload.get("stored_relative_path")
        or ""
    )
    path = str(path).strip()
    if not path:
        return None
    resolved = _resolve_upload_file_path(path)
    if not resolved:
        return None
    try:
        with open(resolved, "rb") as f:
            blob = f.read()
    except OSError:
        return None
    name = str(payload.get("original_filename") or "").strip()
    if not name:
        name = os.path.basename(resolved) or "document"
    return blob, name


def _doc_title_for_payload(payload: dict[str, Any], product_name: str) -> str:
    ft = normalized_upload_payload_file_type(payload)
    if ft and ft in _DOC_TITLE_BY_FILE_TYPE:
        return _DOC_TITLE_BY_FILE_TYPE[ft]
    slug = (product_name or "general").lower().replace(" ", "_").replace("'", "")
    return slug[:64] or "document"


def general_flow_try_enquiry_after_type(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
    general_insurance_type: str,
) -> None:
    """Create InsuranceLab general enquiry when the user picks a product type."""
    if current_flow not in GENERAL_QUOTE_FLOWS:
        return
    if responses.get("general_enquiry_id"):
        return
    if not (general_insurance_type or "").strip():
        return

    api = GeneralAPISubmission()
    api.submit_general_enquiry(
        user_id,
        responses,
        service_type=GENERAL_SERVICE_TYPE,
        general_insurance_type=general_insurance_type.strip(),
    )


def general_flow_try_upload_from_payload(
    *,
    current_flow: str,
    responses: dict[str, Any],
    payload: dict[str, Any],
    product_name: str = "",
) -> None:
    """Upload a general document to InsuranceLab when binary is available on disk."""
    if current_flow not in GENERAL_QUOTE_FLOWS:
        return
    if not responses.get("general_enquiry_id"):
        return
    if not isinstance(payload, dict):
        return

    pair = _load_bytes_from_payload(payload)
    if not pair:
        return

    blob, filename = pair
    parsed = json.dumps(payload, ensure_ascii=False)
    doc_title = _doc_title_for_payload(payload, product_name)

    api = GeneralAPISubmission()
    ok = api.upload_general_document(
        responses,
        doc_title=doc_title,
        document_data=blob,
        filename=filename,
        parsed_text=parsed,
    )
    responses.setdefault("general_upload_log", []).append(
        {"doc_title": doc_title, "filename": filename, "ok": ok}
    )


def general_flow_try_first_step(
    *,
    current_flow: str,
    responses: dict[str, Any],
) -> None:
    """Submit corporate contact details after the general follow-up questionnaire."""
    if current_flow not in GENERAL_QUOTE_FLOWS:
        return
    if not responses.get("general_enquiry_id"):
        return
    if responses.get("general_first_step_stored"):
        return

    api = GeneralAPISubmission()
    api.submit_general_first_step(responses)
