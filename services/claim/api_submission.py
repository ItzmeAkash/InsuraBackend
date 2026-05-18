"""InsuranceLab API submission for insurance claims."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import requests

from services.claim.flow import (
    CLAIM_ROUTER_FLOW,
    MEDICAL_CLAIM_FLOW,
    MOTOR_CLAIM_FLOW,
)

CONTENT_TYPE_JSON = "application/json"

_CLAIM_MOBILE_QUESTION = "Please provide your mobile number so we can reach you."
_MOTOR_REPAIR_QUESTION = "Please tell us where you would like to repair your vehicle:"
_MOTOR_RECOVERY_LOCATION_QUESTION = (
    "Please select where you would like roadside recovery:"
)


def _claim_insurancelab_base_url() -> str:
    raw = (
        os.getenv("CLAIM_INSURANCELAB_BASE_URL")
        or os.getenv("MOTOR_INSURANCELAB_BASE_URL")
        or os.getenv("GENERAL_INSURANCELAB_BASE_URL")
        or "https://insurancelab.ae"
    ).strip()
    return raw.rstrip("/")


_INSURANCE_LAB_BASE_URL = _claim_insurancelab_base_url()
CLAIM_ENQUIRY_URL = f"{_INSURANCE_LAB_BASE_URL}/Api/claim_enquiry"
CLAIM_UPLOAD_DOCUMENT_URL = f"{_INSURANCE_LAB_BASE_URL}/Api/claim_upload_document"
CLAIM_FIRST_STEP_URL = f"{_INSURANCE_LAB_BASE_URL}/Api/claim_first_step"

CLAIM_SERVICE_TYPE = "Claim Insurance"
CLAIM_FLOWS = frozenset({MOTOR_CLAIM_FLOW, MEDICAL_CLAIM_FLOW})
CLAIM_ENQUIRY_FLOWS = CLAIM_FLOWS | frozenset({CLAIM_ROUTER_FLOW})

_DOC_TITLE_BY_STEP: dict[str, str] = {
    "motor_claim_vehicle_registration": "vehicle_registration",
    "motor_claim_driving_license": "driving_license",
    "motor_claim_emirates_id": "emirates_id",
    "motor_claim_police_verification": "police_verification",
    "medical_claim_emirates_id_upload": "emirates_id",
    "medical_claim_insurance_card_upload": "insurance_card",
}

_FILE_TYPE_TO_DOC_TITLE: dict[str, str] = {
    "vehicle_registration": "vehicle_registration",
    "mulkiya": "mulkiya",
    "driving_license": "driving_license",
    "emirates_id": "emirates_id",
    "insurance_card": "insurance_card",
    "police_verification": "police_verification",
    "passing_paper": "passing_paper",
}

_STEP_TO_UPLOAD_TYPES: dict[str, list[str]] = {
    "motor_claim_vehicle_registration": ["vehicle_registration", "mulkiya"],
    "motor_claim_driving_license": ["driving_license"],
    "motor_claim_emirates_id": ["emirates_id"],
    "motor_claim_police_verification": ["police_verification"],
    "medical_claim_emirates_id_upload": ["emirates_id"],
    "medical_claim_insurance_card_upload": ["insurance_card"],
}

_PHONE_RE = re.compile(r"^\+?\d{10,15}$")


class ClaimAPISubmission:
    """Handle claim enquiry and claim document API submission."""

    def submit_claim_enquiry(
        self,
        user_id: str,
        responses: dict[str, Any],
        service_type: str,
        claim_question_type: str,
        claim_type: str,
    ) -> bool:
        """Create claim enquiry and store returned ID."""
        payload = {
            "user_id": user_id,
            "service_type": service_type,
            "claim_question_type": claim_question_type,
            "claim_type": claim_type,
        }
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            res = requests.post(
                CLAIM_ENQUIRY_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(f"Claim enquiry API response: {json.dumps(response_data, indent=2)}")
            enquiry_id = response_data.get("id") or response_data.get("result", {}).get(
                "id"
            )
            if enquiry_id:
                responses["claim_enquiry_id"] = str(enquiry_id)
            responses["claim_enquiry_stored"] = True
            responses["claim_enquiry_api_response"] = response_data
            return True
        except requests.RequestException as e:
            print(f"Error calling claim_enquiry API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            responses["claim_enquiry_stored"] = False
            responses["claim_enquiry_api_error"] = str(e)
            return False

    def upload_claim_document(
        self,
        responses: dict[str, Any],
        doc_title: str,
        document_data: bytes,
        filename: str,
        parsed_text: str = "",
    ) -> bool:
        """Upload a claim document using claim enquiry ID."""
        enquiry_id = responses.get("claim_enquiry_id")
        if not enquiry_id:
            print("Error: claim_enquiry_id not found. Cannot upload claim document.")
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
                CLAIM_UPLOAD_DOCUMENT_URL, files=files, data=data, timeout=30
            )
            res.raise_for_status()
            print(f"Claim upload document API response: {res.text}")
            return True
        except requests.RequestException as e:
            print(f"Error calling claim_upload_document API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            return False

    def submit_claim_first_step(
        self, user_id: str, responses: dict[str, Any]
    ) -> bool:
        """Submit final motor/medical claim step fields to API."""
        enquiry_id = responses.get("claim_enquiry_id")
        if not enquiry_id:
            print("Error: claim_enquiry_id not found. Cannot call claim_first_step.")
            return False

        _sync_claim_api_fields_from_responses(responses)

        is_medical_claim = "medical" in str(responses.get("claim_type", "")).lower()
        mobile = _resolve_claim_mobile(responses, user_id)
        if not mobile:
            print("claim_first_step skipped: valid claim_mobile not found in responses.")
            return False

        payload = {
            "id": str(enquiry_id),
            "claim_mobile": mobile,
            "motor_claim_repair_location": (
                ""
                if is_medical_claim
                else str(responses.get("motor_claim_repair_location", "") or "")
            ),
            "motor_claim_recover_from_road": (
                ""
                if is_medical_claim
                else str(responses.get("motor_claim_recover_from_road", "") or "")
            ),
            "motor_claim_insurance_provider": (
                ""
                if is_medical_claim
                else str(responses.get("motor_claim_insurance_provider", "") or "")
            ),
        }
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        print(f"Claim first step payload: {json.dumps(payload, indent=2)}")
        try:
            res = requests.post(
                CLAIM_FIRST_STEP_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(
                f"Claim first step API response: {json.dumps(response_data, indent=2)}"
            )
            responses["claim_first_step_stored"] = True
            responses["claim_first_step_mobile_sent"] = mobile
            responses["claim_first_step_api_response"] = response_data
            if payload.get("motor_claim_insurance_provider") or payload.get(
                "motor_claim_repair_location"
            ):
                responses["claim_first_step_motor_synced"] = True
            return True
        except requests.RequestException as e:
            print(f"Error calling claim_first_step API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            responses["claim_first_step_stored"] = False
            responses["claim_first_step_api_error"] = str(e)
            return False


def _field(responses: dict[str, Any], api_key: str, display_key: str) -> str:
    val = responses.get(api_key)
    if val is None or (isinstance(val, str) and not val.strip()):
        val = responses.get(display_key)
    if val is None:
        return ""
    return str(val).strip()


def _normalize_mobile_digits(value: str) -> str:
    s = (value or "").strip().replace(" ", "")
    if s.startswith("+"):
        return "+" + re.sub(r"\D", "", s[1:])
    return re.sub(r"\D", "", s)


def _is_valid_claim_mobile(value: str) -> bool:
    normalized = _normalize_mobile_digits(value)
    return bool(normalized and _PHONE_RE.match(normalized))


def _resolve_claim_mobile(responses: dict[str, Any], user_id: str) -> str:
    """Find a full mobile number from API keys or any chat answer (incl. translated keys)."""
    stored = responses.get("claim_mobile")
    if isinstance(stored, str) and _is_valid_claim_mobile(stored):
        return _normalize_mobile_digits(stored)

    from_question = _field(responses, "claim_mobile", _CLAIM_MOBILE_QUESTION)
    if _is_valid_claim_mobile(from_question):
        return _normalize_mobile_digits(from_question)

    for key, val in responses.items():
        if not isinstance(key, str) or not isinstance(val, str):
            continue
        lk = key.lower()
        if ("mobile" in lk or "phone" in lk) and "plate" not in lk:
            if _is_valid_claim_mobile(val):
                return _normalize_mobile_digits(val)

    for val in responses.values():
        if isinstance(val, str) and _is_valid_claim_mobile(val):
            return _normalize_mobile_digits(val)

    if _is_valid_claim_mobile(user_id):
        return _normalize_mobile_digits(user_id)

    return ""


def _sync_claim_api_fields_from_responses(responses: dict[str, Any]) -> None:
    """Map chat question answers to InsuranceLab claim_first_step keys."""
    repair = _field(responses, "motor_claim_insurance_provider", _MOTOR_REPAIR_QUESTION)
    if repair:
        responses["motor_claim_insurance_provider"] = repair

    location = _field(
        responses, "motor_claim_repair_location", _MOTOR_RECOVERY_LOCATION_QUESTION
    )
    if location:
        responses["motor_claim_repair_location"] = location

    if not responses.get("motor_claim_recover_from_road"):
        for key, val in responses.items():
            if not isinstance(key, str) or not isinstance(val, str):
                continue
            if "recover from road" in key.lower():
                responses["motor_claim_recover_from_road"] = val.strip()
                break

    mobile = _resolve_claim_mobile(responses, "")
    if mobile:
        responses["claim_mobile"] = mobile


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


def _load_bytes_from_path(path: str) -> Optional[tuple[bytes, str]]:
    resolved = _resolve_upload_file_path(path)
    if not resolved:
        return None
    try:
        with open(resolved, "rb") as f:
            blob = f.read()
    except OSError:
        return None
    name = os.path.basename(resolved) or "document"
    return blob, name


def _document_local_relative_from_payload(data: dict[str, Any]) -> Optional[str]:
    keys = (
        "stored_path",
        "stored_relative_path",
        "document_stored_path",
        "upload_relative_path",
        "file_location",
        "document_path",
        "local_path",
        "_file_reference",
        "file_path",
    )
    for key in keys:
        val = data.get(key)
        if not isinstance(val, str):
            continue
        s = val.strip().replace("\\", "/")
        if not s or s.startswith(("http://", "https://")):
            continue
        if _resolve_upload_file_path(s):
            return s
    return None


def _load_bytes_from_payload(payload: dict[str, Any]) -> Optional[tuple[bytes, str]]:
    rel = _document_local_relative_from_payload(payload)
    if rel:
        return _load_bytes_from_path(rel)
    path = str(payload.get("_file_reference") or "").strip()
    if path:
        return _load_bytes_from_path(path)
    return None


def _parse_upload_message(
    user_message: str, file_path: str = ""
) -> Optional[dict[str, Any]]:
    raw = (user_message or "").strip()
    if not raw and not (file_path or "").strip():
        return None
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
    norm = raw.replace("\\", "/")
    if norm.lower().startswith("uploads/"):
        return {"_file_reference": norm}
    fp = (file_path or "").strip()
    if fp:
        return {"stored_path": fp, "_file_reference": fp.replace("\\", "/")}
    low = raw.lower()
    if raw and (
        "upload successfully" in low
        or "uploaded successfully" in low
        or "file uploaded" in low
        or "document uploaded" in low
    ):
        return {"_upload": "client_success_signal", "detail": raw}
    return None


def _gather_upload_path_hints(
    *,
    user_message: str,
    file_path: str,
    user_id: str,
    step_id: str,
    payload: Optional[dict[str, Any]],
) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        n = (path or "").strip().replace("\\", "/")
        if n and n not in seen:
            seen.add(n)
            hints.append(n)

    fp = (file_path or "").strip()
    if fp:
        _add(fp)

    if isinstance(payload, dict):
        rel = _document_local_relative_from_payload(payload)
        if rel:
            _add(rel)

    if user_id:
        from services.chatbot.question_store import peek_last_upload_relative_path

        for upload_type in _STEP_TO_UPLOAD_TYPES.get(step_id, []):
            _add(peek_last_upload_relative_path(user_id, upload_type))
        _add(peek_last_upload_relative_path(user_id, ""))

    raw = (user_message or "").strip().replace("\\", "/")
    if raw.lower().startswith("uploads/"):
        _add(raw)

    return hints


def _resolve_claim_upload_bytes(
    *,
    user_message: str,
    file_path: str,
    user_id: str,
    step_id: str,
) -> tuple[Optional[tuple[bytes, str]], Optional[dict[str, Any]]]:
    payload = _parse_upload_message(user_message, file_path)
    if isinstance(payload, dict):
        pair = _load_bytes_from_payload(payload)
        if pair:
            return pair, payload

    for hint in _gather_upload_path_hints(
        user_message=user_message,
        file_path=file_path,
        user_id=user_id,
        step_id=step_id,
        payload=payload if isinstance(payload, dict) else None,
    ):
        pair = _load_bytes_from_path(hint)
        if pair:
            meta = payload if isinstance(payload, dict) else {"_file_reference": hint}
            return pair, meta

    return None, payload if isinstance(payload, dict) else None


def _doc_title_for_upload(step_id: str, payload: dict[str, Any]) -> str:
    if step_id and step_id in _DOC_TITLE_BY_STEP:
        return _DOC_TITLE_BY_STEP[step_id]
    ft = payload.get("file_type")
    if isinstance(ft, str):
        mapped = _FILE_TYPE_TO_DOC_TITLE.get(ft.strip().lower().replace("-", "_"))
        if mapped:
            return mapped
    return step_id or "claim_document"


def claim_flow_try_enquiry(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
    claim_type: str,
    claim_question_type: str,
) -> None:
    """Create InsuranceLab claim enquiry once per session."""
    if current_flow not in CLAIM_ENQUIRY_FLOWS:
        return
    if responses.get("claim_enquiry_id"):
        return
    if not (claim_type or "").strip():
        return

    responses["claim_type"] = claim_type.strip()
    responses["claim_question_type"] = (claim_question_type or claim_type).strip()

    api = ClaimAPISubmission()
    api.submit_claim_enquiry(
        user_id,
        responses,
        service_type=CLAIM_SERVICE_TYPE,
        claim_question_type=responses["claim_question_type"],
        claim_type=responses["claim_type"],
    )


def claim_flow_try_upload_from_message(
    *,
    current_flow: str,
    responses: dict[str, Any],
    user_message: str,
    step_id: str = "",
    file_path: str = "",
    user_id: str = "",
) -> None:
    """Upload claim document bytes to InsuranceLab when available on disk."""
    if current_flow not in CLAIM_FLOWS:
        return
    if not responses.get("claim_enquiry_id"):
        print(
            f"claim upload skipped ({step_id or 'unknown'}): no claim_enquiry_id yet."
        )
        return

    pair, payload = _resolve_claim_upload_bytes(
        user_message=user_message,
        file_path=file_path,
        user_id=user_id,
        step_id=step_id,
    )
    if not pair:
        print(
            f"claim upload skipped ({step_id or 'unknown'}): "
            "could not resolve document bytes from message/file_path."
        )
        return

    blob, filename = pair
    meta = payload if isinstance(payload, dict) else {}
    parsed = json.dumps(meta, ensure_ascii=False) if meta else ""
    doc_title = _doc_title_for_upload(step_id, meta)

    api = ClaimAPISubmission()
    ok = api.upload_claim_document(
        responses,
        doc_title=doc_title,
        document_data=blob,
        filename=filename,
        parsed_text=parsed,
    )
    responses.setdefault("claim_upload_log", []).append(
        {"step_id": step_id, "doc_title": doc_title, "filename": filename, "ok": ok}
    )


def store_claim_mobile(responses: dict[str, Any], raw_mobile: str) -> str:
    """Normalize and persist mobile for InsuranceLab ``claim_first_step``."""
    mobile = _normalize_mobile_digits(raw_mobile)
    if _is_valid_claim_mobile(mobile):
        responses["claim_mobile"] = mobile
    return mobile


def claim_flow_try_first_step(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
    force: bool = False,
) -> None:
    """Submit claim contact / motor recovery details (mobile required)."""
    if current_flow not in CLAIM_FLOWS:
        return
    if not responses.get("claim_enquiry_id"):
        return

    mobile = _resolve_claim_mobile(responses, user_id)
    if not mobile:
        print("claim_first_step deferred: waiting for valid mobile in responses.")
        return

    has_motor_fields = bool(
        str(responses.get("motor_claim_insurance_provider", "") or "").strip()
        or str(responses.get("motor_claim_repair_location", "") or "").strip()
        or str(responses.get("motor_claim_recover_from_road", "") or "").strip()
    )
    if (
        not force
        and responses.get("claim_first_step_stored")
        and responses.get("claim_first_step_mobile_sent") == mobile
        and (
            responses.get("claim_first_step_motor_synced")
            or not has_motor_fields
        )
    ):
        return

    api = ClaimAPISubmission()
    api.submit_claim_first_step(user_id, responses)
