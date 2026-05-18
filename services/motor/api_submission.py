from __future__ import annotations

import json
import os
import random
import re
from io import BytesIO
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import requests

from services.chatbot.question_store import peek_last_upload_relative_path

CONTENT_TYPE_JSON = "application/json"


def _motor_insurancelab_base_url() -> str:
    """Origin for InsuranceLab motor APIs (scheme + host, no trailing slash).

    Set ``MOTOR_INSURANCELAB_BASE_URL`` in the environment (e.g. ``.env``).
    Default matches the production host used for motor enquiry / steps / upload.
    """
    raw = (os.getenv("MOTOR_INSURANCELAB_BASE_URL") or "https://Insurancelab.ae").strip()
    return raw.rstrip("/")


_MOTOR_IL_BASE = _motor_insurancelab_base_url()
MOTOR_ENQUIRY_URL = f"{_MOTOR_IL_BASE}/api/motor_enquiry"
MOTOR_FIRST_STEP_URL = f"{_MOTOR_IL_BASE}/api/motor_first_step"
MOTOR_SECOND_STEP_URL = f"{_MOTOR_IL_BASE}/api/motor_second_step"
MOTOR_THIRD_STEP_URL = f"{_MOTOR_IL_BASE}/api/motor_third_step"
UPLOAD_DOCUMENT_URL = f"{_MOTOR_IL_BASE}/api/upload_document"

API_MOTOR_TYPE_NEW = "New Insurance"
API_MOTOR_TYPE_RENEWAL = "Renewal"
API_REGISTRATION_PRIVATE = "Private (Individual)"
API_REGISTRATION_COMPANY = "Company (Business)"
API_COVER_COMPREHENSIVE = "Comprehensive"
API_COVER_THIRD_PARTY = "ThirdParty Liability"

# Must match ``car_questions`` JSON ``question`` for motor cover step (used to backfill IL second step).
MOTOR_COVER_QUESTION_KEY = "What type of motor insurance are you looking for?"

# Optional: full-form motor payload endpoint (set in env if different from enquiry).
MOTOR_API_URL = os.getenv("MOTOR_API_URL", "").strip()


class MotorAPISubmission:
    """InsuranceLab motor APIs — enquiry, steps, document upload."""

    def _generate_user_id(self) -> str:
        random_digits = random.randint(10000, 99999)
        return f"INS{random_digits}"

    def _extract_info_to_text(self, extracted_info: dict) -> str:
        if not extracted_info:
            return ""
        return json.dumps(extracted_info, ensure_ascii=False)

    def call_motor_enquiry(
        self,
        responses: dict[str, Any],
        name: str,
        phone_number: str,
    ) -> Optional[str]:
        user_id = self._generate_user_id()
        payload = {"user_id": user_id, "name": name, "phone_number": phone_number}
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            res = requests.post(
                MOTOR_ENQUIRY_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(f"Motor enquiry API response: {json.dumps(response_data, indent=2)}")

            enquiry_id = response_data.get("id") or response_data.get("result", {}).get(
                "id"
            )
            if enquiry_id:
                responses["motor_enquiry_id"] = str(enquiry_id)
                responses["motor_user_id"] = user_id
                print(f"Motor enquiry ID stored: {enquiry_id}")
                return str(enquiry_id)
            print(f"Warning: No ID found in motor_enquiry response: {response_data}")
            return None
        except requests.RequestException as e:
            print(f"Error calling motor_enquiry API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            return None

    def call_motor_first_step(
        self,
        responses: dict[str, Any],
        motor_type: str,
        registration_type: str,
    ) -> bool:
        enquiry_id = responses.get("motor_enquiry_id")
        if not enquiry_id:
            print("Error: motor_enquiry_id not found. Cannot call motor_first_step.")
            return False

        type_mapping = {
            API_MOTOR_TYPE_NEW: API_MOTOR_TYPE_NEW,
            API_MOTOR_TYPE_RENEWAL: API_MOTOR_TYPE_RENEWAL,
        }
        api_type = type_mapping.get(motor_type, motor_type)

        registration_mapping = {
            API_REGISTRATION_PRIVATE: API_REGISTRATION_PRIVATE,
            API_REGISTRATION_COMPANY: API_REGISTRATION_COMPANY,
        }
        api_registration = registration_mapping.get(
            registration_type, registration_type
        )

        payload = {
            "id": enquiry_id,
            "type": api_type,
            "registration_type": api_registration,
        }
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            res = requests.post(
                MOTOR_FIRST_STEP_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(
                f"Motor first step API response: {json.dumps(response_data, indent=2)}"
            )
            responses["motor_first_step_ok"] = True
            return True
        except requests.RequestException as e:
            print(f"Error calling motor_first_step API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            responses["motor_first_step_ok"] = False
            return False

    def upload_document(
        self,
        responses: dict[str, Any],
        doc_title: str,
        document_data: bytes,
        filename: str,
        parsed_text: str = "",
    ) -> bool:
        enquiry_id = responses.get("motor_enquiry_id")
        if not enquiry_id:
            print("Error: motor_enquiry_id not found. Cannot upload document.")
            return False

        files = {
            "document": (
                filename,
                BytesIO(document_data),
                _mime_for_upload_filename(filename),
            )
        }
        data = {"id": enquiry_id, "doc_title": doc_title, "parsed_text": parsed_text}
        try:
            res = requests.post(UPLOAD_DOCUMENT_URL, files=files, data=data, timeout=30)
            res.raise_for_status()
            response_data = res.json()
            print(
                f"Upload document API response for {doc_title}: "
                f"{json.dumps(response_data, indent=2)}"
            )
            return True
        except requests.RequestException as e:
            print(f"Error calling upload_document API for {doc_title}: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            return False

    def call_motor_second_step(
        self,
        responses: dict[str, Any],
        insurance_type: str,
        vehicle_value: str = "",
        email: str = "",
        mobile: str = "",
    ) -> bool:
        enquiry_id = responses.get("motor_enquiry_id")
        if not enquiry_id:
            print("Error: motor_enquiry_id not found. Cannot call motor_second_step.")
            return False

        insurance_mapping = {
            API_COVER_COMPREHENSIVE: API_COVER_COMPREHENSIVE,
            API_COVER_THIRD_PARTY: API_COVER_THIRD_PARTY,
            "Third Party": API_COVER_THIRD_PARTY,
        }
        api_insurance_type = insurance_mapping.get(insurance_type, insurance_type)

        payload = {
            "id": enquiry_id,
            "insurance_type": api_insurance_type,
            "vehicle_value": vehicle_value,
            "email": email,
            "mobile": mobile,
        }
        responses["motor_second_step_payload"] = dict(payload)
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            print(f"Motor second step payload: {json.dumps(payload, indent=2)}")
            res = requests.post(
                MOTOR_SECOND_STEP_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(
                f"Motor second step API response: {json.dumps(response_data, indent=2)}"
            )
            responses["motor_second_step_ok"] = True
            return True
        except requests.RequestException as e:
            print(f"Error calling motor_second_step API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            responses["motor_second_step_ok"] = False
            return False

    def call_motor_third_step(
        self,
        responses: dict[str, Any],
        email: str = "",
        phone: str = "",
    ) -> bool:
        enquiry_id = responses.get("motor_enquiry_id")
        if not enquiry_id:
            print("Error: motor_enquiry_id not found. Cannot call motor_third_step.")
            return False

        email_val = (email or "").strip()
        phone_val = _normalize_phone_digits(phone or "")
        if not email_val or not _EMAIL_RE.match(email_val):
            print("motor_third_step skipped: valid email required.")
            return False
        if not phone_val or not _PHONE_RE.match(phone_val):
            print("motor_third_step skipped: valid phone required.")
            return False

        payload = {
            "id": enquiry_id,
            "email": email_val,
            "phone": phone_val,
        }
        responses["motor_third_step_payload"] = dict(payload)
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            print(f"Motor third step payload: {json.dumps(payload, indent=2)}")
            res = requests.post(
                MOTOR_THIRD_STEP_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            response_data = res.json()
            print(
                f"Motor third step API response: {json.dumps(response_data, indent=2)}"
            )
            responses["motor_third_step_ok"] = True
            return True
        except requests.RequestException as e:
            print(f"Error calling motor_third_step API: {e}")
            if getattr(e, "response", None) is not None:
                print(f"Response: {e.response.text}")
            responses["motor_third_step_ok"] = False
            return False

    def submit_motor_enquiry(self, responses: dict[str, Any]) -> bool:
        if not MOTOR_API_URL:
            print("MOTOR_API_URL not set; skip submit_motor_enquiry.")
            return False

        motor_name = responses.get("motor_name", "") or responses.get("name", "")
        motor_email = _email_from_responses(responses)
        motor_mobile = _mobile_from_responses(responses, "")
        payload = {
            "name": motor_name,
            "email": motor_email,
            "mobile": motor_mobile,
            "motor_type": responses.get("motor_type", ""),
            "registration": responses.get("motor_registration", ""),
            "insurance_type": responses.get("motor_insurance_type", ""),
            "vehicle_value": responses.get("motor_vehicle_value", ""),
            "mulkiya_extracted": responses.get("motor_mulkiya_extracted", {}),
        }
        responses["motor_enquiry_payload"] = dict(payload)
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            print(f"Motor enquiry payload sent: {json.dumps(payload, indent=2)}")
            res = requests.post(
                MOTOR_API_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            print(f"API response status: {res.status_code}")
            print(f"API response: {res.text}")
            responses["motor_enquiry_stored"] = True
            responses["motor_enquiry_api_response"] = res.text
            return True
        except requests.RequestException as e:
            print(f"Error calling motor_enquiry API: {e}")
            responses["motor_enquiry_stored"] = False
            responses["motor_enquiry_api_error"] = str(e)
            return False

    def submit_motor_enquiry_basic(self, responses: dict[str, Any], name: str, phone: str) -> bool:
        if not MOTOR_API_URL:
            print("MOTOR_API_URL not set; skip submit_motor_enquiry_basic.")
            return False

        payload = {"name": name, "phone": phone}
        headers = {
            "Content-Type": CONTENT_TYPE_JSON,
            "Accept": CONTENT_TYPE_JSON,
        }
        try:
            res = requests.post(
                MOTOR_API_URL, json=payload, headers=headers, timeout=10
            )
            res.raise_for_status()
            print(f"Motor enquiry payload sent: {json.dumps(payload, indent=2)}")
            print(f"API response status: {res.status_code}")
            print(f"API response: {res.text}")
            responses["motor_enquiry_stored"] = True
            responses["motor_enquiry_api_response"] = res.text
            return True
        except requests.RequestException as e:
            print(f"Error calling motor_enquiry API: {e}")
            responses["motor_enquiry_stored"] = False
            responses["motor_enquiry_api_error"] = str(e)
            return False


_MOTOR_MENU_QUESTION_PREFIX = "Great! Now let me get the best motor insurance quote"
_NEW_OR_RENEWAL = frozenset({API_MOTOR_TYPE_NEW, API_MOTOR_TYPE_RENEWAL})
_PHONE_RE = re.compile(r"^\+?\d{10,15}$")


def _motor_type_from_responses(responses: dict[str, Any]) -> str:
    for key, val in responses.items():
        if isinstance(key, str) and _MOTOR_MENU_QUESTION_PREFIX in key:
            if isinstance(val, str) and val in _NEW_OR_RENEWAL:
                return val
    for val in responses.values():
        if isinstance(val, str) and val.strip() in _NEW_OR_RENEWAL:
            return val.strip()
    return API_MOTOR_TYPE_NEW


def _display_name_from_responses(responses: dict[str, Any], fallback_user_id: str) -> str:
    for key in ("motor_name", "name", "customer_name"):
        v = responses.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k, v in responses.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        lk = k.lower()
        if "name" in lk and v.strip() and "insurance" not in lk:
            return v.strip()
    return fallback_user_id[:80] if fallback_user_id else "Insura motor customer"


def _normalize_phone_digits(value: str) -> str:
    s = (value or "").strip().replace(" ", "")
    if s.startswith("+"):
        s = "+" + re.sub(r"\D", "", s[1:])
    else:
        s = re.sub(r"\D", "", s)
    return s


_KNOWN_MOBILE_QUESTION_KEYS: tuple[str, ...] = (
    "May I have your mobile number?",
    "Please provide your mobile number so we can reach you.",
    "May i know your  mobile number, please?",
    "May i know your mobile number, please?",
    "May I have the sponsor's mobile number, please?",
)


def _phone_from_responses(responses: dict[str, Any], user_id: str) -> str:
    for q in _KNOWN_MOBILE_QUESTION_KEYS:
        v = responses.get(q)
        if isinstance(v, str):
            n = _normalize_phone_digits(v)
            if _PHONE_RE.match(n):
                return n
    for key in (
        "motor_enquiry_phone",
        "phone",
        "mobile",
        "phone_number",
        "motor_phone",
    ):
        v = responses.get(key)
        if isinstance(v, str):
            n = _normalize_phone_digits(v)
            if _PHONE_RE.match(n):
                return n
    for k, v in responses.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        lk = k.lower()
        if ("mobile" in lk or "phone" in lk) and "plate" not in lk:
            n = _normalize_phone_digits(v)
            if _PHONE_RE.match(n):
                return n
    uid = _normalize_phone_digits(user_id or "")
    if _PHONE_RE.match(uid):
        return uid
    return ""


class MotorInsuranceLabClient:
    """Facade used by ``question_helper`` for the car motor flow."""

    def __init__(self) -> None:
        self._api = MotorAPISubmission()

    def ensure_enquiry_and_first_step(
        self,
        *,
        user_id: str,
        responses: dict[str, Any],
        registration_type: str,
    ) -> None:
        """Send motor_first_step when enquiry id exists (enquiry is created at flow start via phone)."""
        if responses.get("motor_first_step_ok"):
            return
        if not responses.get("motor_enquiry_id"):
            print(
                "motor_first_step skipped: no motor_enquiry_id (collect phone at flow start first)."
            )
            return

        motor_type = _motor_type_from_responses(responses)
        self._api.call_motor_first_step(
            responses, motor_type=motor_type, registration_type=registration_type
        )


motor_insurancelab_client = MotorInsuranceLabClient()


def motor_flow_try_enquiry_after_phone(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
    phone: str,
    display_name: str = "",
) -> Optional[str]:
    """Create InsuranceLab motor enquiry at flow start using the customer's real phone."""
    if current_flow not in MOTOR_QUOTE_FLOWS:
        return None
    if responses.get("motor_enquiry_id"):
        return str(responses["motor_enquiry_id"])

    phone_digits = _normalize_phone_digits(phone)
    if not phone_digits or not _PHONE_RE.match(phone_digits):
        print("motor_enquiry skipped: invalid phone at flow start.")
        return None

    responses["motor_enquiry_phone"] = phone_digits
    responses["motor_mobile"] = phone_digits
    name = (display_name or "").strip() or _display_name_from_responses(
        responses, user_id
    )

    api = MotorAPISubmission()
    return api.call_motor_enquiry(responses, name=name, phone_number=phone_digits)


# Flows that sync to InsuranceLab motor quote APIs
MOTOR_QUOTE_FLOWS = frozenset({"car_questions"})

_DOC_TITLE_BY_STEP: dict[str, str] = {
    "upload_driving_license": "driving_license",
    "upload_driving_license_front": "driving_license",
    "upload_driving_license_back": "driving_license",
    "upload_mulkiya": "mulkiya",
    "upload_mulkiya_front": "mulkiya",
    "upload_mulkiya_back": "mulkiya",
    "upload_emirates_doc": "emirates_id",
    "upload_eid_front": "emirates_id",
    "upload_eid_back": "emirates_id",
    "vehicle_test_cert": "passing_paper",
}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_ALLOWED_INSURANCELAB_UPLOAD_SUFFIXES: tuple[str, ...] = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".doc",
    ".docx",
)

_MAX_REMOTE_DOCUMENT_BYTES = 30 * 1024 * 1024

_CONTENT_DISPOSITION_FILENAME_RE = re.compile(
    r"filename\*=UTF-8''([^;\s]+)|filename=\"([^\"]+)\"",
    re.IGNORECASE,
)

_DOCUMENT_BINARY_URL_KEYS: tuple[str, ...] = (
    "document_url",
    "pdf_url",
    "file_url",
    "file_path",
    "source_pdf_url",
    "mulkiya_pdf_url",
    "signed_document_url",
    "signed_url",
    "documentUrl",
    "pdfUrl",
    "fileUrl",
    "document_file_url",
    "original_pdf_url",
)


def _upload_suffix_allowed(filename: str) -> bool:
    low = (filename or "").lower()
    return any(low.endswith(suf) for suf in _ALLOWED_INSURANCELAB_UPLOAD_SUFFIXES)


def _mime_for_upload_filename(filename: str) -> str:
    low = filename.lower()
    if low.endswith(".pdf"):
        return "application/pdf"
    if low.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if low.endswith(".png"):
        return "image/png"
    if low.endswith(".doc"):
        return "application/msword"
    if low.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def _document_url_from_ocr_dict(data: dict[str, Any]) -> Optional[str]:
    """Pick an HTTPS URL to the binary document from common OCR / extraction payloads."""
    for key in _DOCUMENT_BINARY_URL_KEYS:
        val = data.get(key)
        if isinstance(val, str):
            s = val.strip()
            if s.startswith(("https://", "http://")):
                return s
    for sub_key in ("document", "file", "pdf", "source", "mulkiya_file", "attachment"):
        sub = data.get(sub_key)
        if isinstance(sub, dict):
            for sk in ("url", "href", "src", "signed_url", "document_url", "pdf_url"):
                u = sub.get(sk)
                if isinstance(u, str) and u.strip().startswith(("https://", "http://")):
                    return u.strip()
        if isinstance(sub, str) and sub.strip().startswith(("https://", "http://")):
            return sub.strip()
    u = data.get("url")
    if isinstance(u, str) and u.strip().startswith(("https://", "http://")):
        path = (urlparse(u).path or "").lower()
        if any(path.endswith(ext) for ext in _ALLOWED_INSURANCELAB_UPLOAD_SUFFIXES):
            return u.strip()
    return None


def _typed_upload_hint_for_doc_title(user_id: str, doc_title: str) -> str:
    if not user_id or not doc_title.strip():
        return ""
    candidate_types = [doc_title.strip()]
    if doc_title.strip() == "mulkiya":
        candidate_types.append("vehicle_registration")
    if doc_title.strip() == "emirates_id":
        candidate_types.extend(["emirates_id_front", "emirates_id_back"])
    for candidate_type in candidate_types:
        rel = peek_last_upload_relative_path(user_id, candidate_type)
        if rel:
            return rel
    return ""


def _filename_from_url_and_headers(url: str, content_disposition: str) -> str:
    m = _CONTENT_DISPOSITION_FILENAME_RE.search(content_disposition or "")
    if m:
        name = (m.group(1) or m.group(2) or "").strip()
        if name:
            return unquote(name)
    path_last = unquote((urlparse(url).path or "").rsplit("/", 1)[-1] or "")
    return path_last if path_last else "document.pdf"


def _try_binary_document_from_https_url(url: str) -> Optional[tuple[bytes, str]]:
    """Download PDF/image from a public URL for InsuranceLab multipart upload."""
    u = (url or "").strip()
    if not u.startswith(("https://", "http://")):
        return None
    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    try:
        with requests.get(
            u,
            timeout=90,
            stream=True,
            headers={"User-Agent": "InsuraBot/1.0 (motor-document)"},
        ) as resp:
            resp.raise_for_status()
            total = 0
            chunks: list[bytes] = []
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    total += len(chunk)
                    if total > _MAX_REMOTE_DOCUMENT_BYTES:
                        print(
                            "motor Insurancelab upload: remote document exceeded "
                            f"{_MAX_REMOTE_DOCUMENT_BYTES} bytes cap"
                        )
                        return None
                    chunks.append(chunk)
            blob = b"".join(chunks)
            cd = resp.headers.get("Content-Disposition") or ""
            ct = (resp.headers.get("Content-Type") or "").lower()
    except requests.RequestException as exc:
        print(f"motor Insurancelab upload: failed to fetch document URL: {exc}")
        return None

    fname = _filename_from_url_and_headers(u, cd)
    if not _upload_suffix_allowed(fname):
        if "pdf" in ct:
            fname = "document.pdf"
        elif "jpeg" in ct or "jpg" in ct:
            fname = "document.jpg"
        elif "png" in ct:
            fname = "document.png"
        elif "wordprocessingml" in ct or "msword" in ct:
            fname = "document.docx" if "openxml" in ct else "document.doc"
        else:
            return None
    return blob, fname


def _doc_title_for_step(step_id: str) -> Optional[str]:
    return _DOC_TITLE_BY_STEP.get(step_id)


def _repo_root() -> str:
    """InsuraBackend project root (parent of ``services``)."""
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


def _try_binary_from_filesystem_hint(hint: str) -> Optional[tuple[bytes, str]]:
    """Load PDF/image bytes from ``uploads/...``, repo-relative path, or absolute path on disk."""
    norm = (hint or "").strip().replace("\\", "/")
    if not norm or norm.lower().startswith(("http://", "https://")):
        return None
    resolved = _resolve_upload_file_path(norm)
    if not resolved:
        return None
    if not _upload_suffix_allowed(resolved):
        print(
            f"motor Insurancelab upload: skipped unsupported extension for path {resolved!r}"
        )
        return None
    try:
        with open(resolved, "rb") as f:
            blob = f.read()
    except OSError:
        return None
    base = os.path.basename(resolved) or "document.bin"
    return blob, base


def _document_local_relative_from_ocr_dict(data: dict[str, Any]) -> Optional[str]:
    """Pick a non-URL filesystem hint (e.g. ``uploads/mulkiya_....pdf``) from extraction payloads."""
    keys = (
        "document_stored_path",
        "stored_relative_path",
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
    for sub_key in ("document", "file", "pdf", "attachment"):
        sub = data.get(sub_key)
        if isinstance(sub, dict):
            for sk in ("path", "local_path", "stored_path", "document_stored_path"):
                p = sub.get(sk)
                if not isinstance(p, str):
                    continue
                s = p.strip().replace("\\", "/")
                if s and not s.startswith(("http://", "https://")) and _resolve_upload_file_path(
                    s
                ):
                    return s
    return None


def _upload_payload_from_raw_message(
    raw_message: str, file_path: str = "", user_id: str = "", doc_title: str = ""
) -> Optional[tuple[bytes, str, str]]:
    """Build (file_bytes, filename, parsed_text) for InsuranceLab upload, or None.

    For OCR JSON we never send ``ocr_extract.json`` (rejected). We attach bytes from, in order:
    a URL inside the JSON, a stored path inside the JSON (e.g. ``stored_relative_path``),
    ``file_path`` as HTTP(S) URL or ``uploads/...`` path, or a remote fetch of ``raw_message``
    when it is a bare URL.
    """
    raw = (raw_message or "").strip()
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    if not raw:
        return None

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        parsed = json.dumps(data, ensure_ascii=False)

        doc_url = _document_url_from_ocr_dict(data)
        if doc_url:
            pair = _try_binary_document_from_https_url(doc_url)
            if pair:
                blob, fname = pair
                return blob, fname, parsed

        local_from_json = _document_local_relative_from_ocr_dict(data)
        if local_from_json:
            pair = _try_binary_from_filesystem_hint(local_from_json)
            if pair:
                blob, fname = pair
                return blob, fname, parsed

        fp = (file_path or "").strip()
        if not fp and user_id and doc_title:
            fp = _typed_upload_hint_for_doc_title(user_id, doc_title)
        if fp.startswith(("https://", "http://")):
            pair = _try_binary_document_from_https_url(fp)
            if pair:
                blob, fname = pair
                return blob, fname, parsed

        pair = _try_binary_from_filesystem_hint(fp)
        if pair:
            blob, fname = pair
            return blob, fname, parsed

        print(
            "motor Insurancelab upload: OCR JSON but no usable binary source (URL, "
            "stored path in JSON, resolvable file_path, or typed OCR upload) — skipped "
            "(cannot send ocr_extract.json)."
        )
        return None

    if raw.startswith(("https://", "http://")):
        pair = _try_binary_document_from_https_url(raw)
        if pair:
            blob, fname = pair
            return blob, fname, ""

    norm = raw.replace("\\", "/")
    pair = _try_binary_from_filesystem_hint(norm)
    if not pair:
        return None
    blob, base = pair
    return blob, base, ""


def motor_flow_try_upload_document(
    *,
    user_id: str,
    file_path: str,
    current_flow: str,
    responses: dict[str, Any],
    step_id: str,
    raw_message: str,
) -> None:
    """After a motor OCR JSON or upload path, push document to InsuranceLab if enquiry exists."""
    if current_flow not in MOTOR_QUOTE_FLOWS:
        return
    if not responses.get("motor_enquiry_id"):
        return
    doc_title = _doc_title_for_step(step_id)
    if not doc_title:
        return
    payload = _upload_payload_from_raw_message(
        raw_message, file_path, user_id=user_id, doc_title=doc_title
    )
    if not payload:
        return
    for hint in _gather_local_hints_for_motor_upload(
        raw_message, file_path, user_id, doc_title
    ):
        _register_motor_local_upload_paths(responses, hint)
    blob, filename, parsed_text = payload
    api = MotorAPISubmission()
    ok = api.upload_document(
        responses, doc_title, blob, filename, parsed_text=parsed_text
    )
    responses.setdefault("motor_upload_log", []).append(
        {"step_id": step_id, "doc_title": doc_title, "ok": ok}
    )


def _vehicle_value_from_responses(responses: dict[str, Any]) -> str:
    for k, v in responses.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        lk = k.lower()
        if "vehicle" in lk and "value" in lk and v.strip():
            return v.strip()
    for v in responses.values():
        if not isinstance(v, dict):
            continue
        for key in ("vehicle_value", "estimated_value", "car_value", "value"):
            x = v.get(key)
            if isinstance(x, str) and x.strip():
                return x.strip()
            if isinstance(x, (int, float)) and str(x).strip():
                return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    return ""


def _email_from_responses(responses: dict[str, Any]) -> str:
    sponsor_q = "May I have the Email Address"
    me = responses.get("motor_email")
    if isinstance(me, str) and _EMAIL_RE.match(me.strip()):
        return me.strip()
    v = responses.get(sponsor_q)
    if isinstance(v, str) and _EMAIL_RE.match(v.strip()):
        return v.strip()
    for key, val in responses.items():
        if not isinstance(key, str) or not isinstance(val, str):
            continue
        if "email" in key.lower() and _EMAIL_RE.match(val.strip()):
            return val.strip()
    for val in responses.values():
        if isinstance(val, str) and _EMAIL_RE.match(val.strip()):
            return val.strip()
    return ""


def _cover_choice_to_api_insurance(choice: str) -> Optional[str]:
    cover_map = {
        "Comprehensive": API_COVER_COMPREHENSIVE,
        "Comprehensive (Full Cover)": API_COVER_COMPREHENSIVE,
        "ThirdPartyLiability": API_COVER_THIRD_PARTY,
        "ThirdParty Liability": API_COVER_THIRD_PARTY,
        "Third Party": API_COVER_THIRD_PARTY,
    }
    return cover_map.get((choice or "").strip())


def _resolve_motor_api_insurance_type_for_second_step(
    responses: dict[str, Any],
) -> Optional[str]:
    """Resolve InsuranceLab ``insurance_type`` from cache or stored cover answer."""
    stored = responses.get("_motor_api_insurance_type")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    raw = responses.get(MOTOR_COVER_QUESTION_KEY)
    if isinstance(raw, str) and raw.strip():
        api = _cover_choice_to_api_insurance(raw)
        if api:
            responses["_motor_api_insurance_type"] = api
            return api
    return None


def _mobile_from_responses(responses: dict[str, Any], user_id: str) -> str:
    mv = responses.get("motor_mobile")
    if isinstance(mv, str):
        n = _normalize_phone_digits(mv)
        if _PHONE_RE.match(n):
            return n
    mobile_q = "Please provide your mobile number so we can reach you."
    v = responses.get(mobile_q)
    if isinstance(v, str):
        n = _normalize_phone_digits(v)
        if _PHONE_RE.match(n):
            return n
    return _phone_from_responses(responses, user_id)


def motor_flow_try_second_step_after_cover(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
    cover_choice: str,
) -> None:
    """Send motor_second_step when user picks a concrete cover type (not Know More)."""
    if current_flow not in MOTOR_QUOTE_FLOWS:
        return
    if not responses.get("motor_enquiry_id"):
        return
    if cover_choice.strip() == "Know More":
        return

    api_insurance = _cover_choice_to_api_insurance(cover_choice)
    if not api_insurance:
        return

    responses["_motor_api_insurance_type"] = api_insurance

    vehicle_value = _vehicle_value_from_responses(responses)
    email = _email_from_responses(responses)
    mobile = _mobile_from_responses(responses, user_id)

    api = MotorAPISubmission()
    api.call_motor_second_step(
        responses,
        api_insurance,
        vehicle_value=vehicle_value,
        email=email,
        mobile=mobile,
    )


def motor_flow_try_third_step_after_contact(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
) -> None:
    """Send motor_third_step with enquiry id, email, and phone after contact details are collected."""
    if current_flow not in MOTOR_QUOTE_FLOWS:
        return
    if not responses.get("motor_enquiry_id"):
        return

    email = _email_from_responses(responses)
    phone = _mobile_from_responses(responses, user_id)
    if not email or not phone:
        return

    api = MotorAPISubmission()
    api.call_motor_third_step(responses, email=email, phone=phone)


def motor_flow_try_second_step_refresh(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
) -> None:
    """Re-send motor_second_step with latest email/mobile/value after user completes later steps."""
    if current_flow not in MOTOR_QUOTE_FLOWS:
        return
    if not responses.get("motor_enquiry_id"):
        return
    api_insurance = _resolve_motor_api_insurance_type_for_second_step(responses)
    if not api_insurance:
        return

    vehicle_value = _vehicle_value_from_responses(responses)
    email = _email_from_responses(responses)
    mobile = _mobile_from_responses(responses, user_id)

    api = MotorAPISubmission()
    api.call_motor_second_step(
        responses,
        api_insurance,
        vehicle_value=vehicle_value,
        email=email,
        mobile=mobile,
    )


def _register_motor_local_upload_paths(responses: dict[str, Any], *paths: str) -> None:
    from services.upload_cleanup import _normalize_stored_rel_path

    if not isinstance(responses, dict):
        return
    acc = responses.setdefault("_motor_local_upload_paths", [])
    if not isinstance(acc, list):
        responses["_motor_local_upload_paths"] = []
        acc = responses["_motor_local_upload_paths"]
    for raw in paths:
        n = _normalize_stored_rel_path(raw)
        if n and n not in acc:
            acc.append(n)


def _gather_local_hints_for_motor_upload(
    raw_message: str, file_path: str, user_id: str, doc_title: str
) -> list[str]:
    out: list[str] = []
    fp = (file_path or "").strip()
    if fp and not fp.startswith(("http://", "https://")):
        out.append(fp)
    raw = (raw_message or "").strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            rel = _document_local_relative_from_ocr_dict(data)
            if rel:
                out.append(rel)
    if user_id and doc_title:
        hint = _typed_upload_hint_for_doc_title(user_id, doc_title)
        if hint:
            out.append(hint)
    return out


def wipe_motor_car_quote_session_files(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
) -> None:
    """Backward-compatible wrapper; prefer ``wipe_flow_session_upload_files``."""
    from services.upload_cleanup import wipe_flow_session_upload_files

    wipe_flow_session_upload_files(
        user_id=user_id, current_flow=current_flow, responses=responses
    )
