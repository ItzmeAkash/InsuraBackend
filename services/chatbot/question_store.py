from __future__ import annotations

import json
import os

from cachetools import TTLCache

MEDICAL_QUESTIONS_FILE = "questions/medical/questions.json"


def list_pdfs(directory: str = "pdf") -> list[str]:
    return [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".pdf")]


user_states = TTLCache(maxsize=1000, ttl=3600)


def load_questions(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{file_path} not found. Please ensure the file exists.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Error decoding JSON: {exc}") from exc


common_questions = load_questions("questions/common.json")
medical_questions_data = load_questions(MEDICAL_QUESTIONS_FILE)
motor_questions_data = load_questions("questions/motor/questions.json")
general_questions_data = load_questions("questions/general/questions.json")
claim_questions_data = load_questions("questions/claim/questions.json")
claim_motor_questions_data = load_questions("questions/claim/motor/questions.json")
claim_medical_questions_data = load_questions("questions/claim/medical/questions.json")

initial_questions = common_questions["initial_questions"]
greeting_templates = common_questions["greeting_templates"]

medical_questions = medical_questions_data["medical_questions"]
individual_questions = medical_questions_data["individual_questions"]
existing_policy_questions = medical_questions_data["existing_policy_questions"]

try:
    _medical_questions_mtime = os.path.getmtime(MEDICAL_QUESTIONS_FILE)
except OSError:
    _medical_questions_mtime = None


def refresh_medical_questions_if_changed() -> None:
    """Reload medical JSON when the file changes (import-time lists stay stale otherwise)."""
    global medical_questions_data, _medical_questions_mtime
    try:
        mtime = os.path.getmtime(MEDICAL_QUESTIONS_FILE)
    except OSError:
        return
    if _medical_questions_mtime == mtime:
        return
    data = load_questions(MEDICAL_QUESTIONS_FILE)
    medical_questions_data = data
    medical_questions[:] = data["medical_questions"]
    individual_questions[:] = data["individual_questions"]
    existing_policy_questions[:] = data["existing_policy_questions"]
    _medical_questions_mtime = mtime

motor_insurance_questions = motor_questions_data["motor_insurance_questions"]
car_questions = motor_questions_data["car_questions"]
bike_questions = motor_questions_data["bike_questions"]
motor_claim = claim_motor_questions_data.get(
    "motor_claim", motor_questions_data.get("motor_claim", [])
)
claim_router_questions = claim_questions_data.get(
    "claim_router_questions",
    motor_questions_data.get("claim_router_questions", []),
)
medical_claim = claim_medical_questions_data.get("medical_claim", [])
general_insurance_questions = general_questions_data["general_insurance_questions"]

# Latest binary upload path per user (see routes/upload.py). Consumed by chat when the client
# sends a success banner in ``message`` but omits ``UserInput.file_path``.
_LAST_UPLOAD_RELATIVE_PATH_BY_USER: dict[str, str] = {}
_LAST_UPLOAD_RELATIVE_PATH_BY_USER_AND_TYPE: dict[str, dict[str, str]] = {}


def remember_last_upload_relative_path(
    user_id: str, relative_path: str, upload_type: str = ""
) -> None:
    uid = (user_id or "").strip()
    if not uid or not (relative_path or "").strip():
        return
    normalized_path = relative_path.strip().replace("\\", "/").replace("//", "/")
    _LAST_UPLOAD_RELATIVE_PATH_BY_USER[uid] = normalized_path
    normalized_type = (upload_type or "").strip().lower().replace("-", "_")
    if normalized_type:
        typed = _LAST_UPLOAD_RELATIVE_PATH_BY_USER_AND_TYPE.setdefault(uid, {})
        typed[normalized_type] = normalized_path


def pop_last_upload_relative_path(user_id: str) -> str:
    uid = (user_id or "").strip()
    if not uid:
        return ""
    return _LAST_UPLOAD_RELATIVE_PATH_BY_USER.pop(uid, "") or ""


def peek_last_upload_relative_path(user_id: str, upload_type: str = "") -> str:
    """Latest stored upload path for ``user_id`` without removing it (non-destructive)."""
    uid = (user_id or "").strip()
    if not uid:
        return ""
    normalized_type = (upload_type or "").strip().lower().replace("-", "_")
    if normalized_type:
        return _LAST_UPLOAD_RELATIVE_PATH_BY_USER_AND_TYPE.get(uid, {}).get(
            normalized_type, ""
        ) or ""
    return _LAST_UPLOAD_RELATIVE_PATH_BY_USER.get(uid, "") or ""


def snapshot_user_upload_paths(user_id: str) -> list[str]:
    """All remembered relative paths for ``user_id`` (generic + per-type), for cleanup."""
    uid = (user_id or "").strip()
    if not uid:
        return []
    out: list[str] = []
    generic = (_LAST_UPLOAD_RELATIVE_PATH_BY_USER.get(uid) or "").strip()
    if generic:
        out.append(generic.replace("\\", "/"))
    typed = _LAST_UPLOAD_RELATIVE_PATH_BY_USER_AND_TYPE.get(uid) or {}
    for rel in typed.values():
        s = (rel or "").strip().replace("\\", "/")
        if s and s not in out:
            out.append(s)
    return out


def clear_user_upload_registry(user_id: str) -> None:
    """Forget last-upload hints for ``user_id`` (e.g. after motor quote session ends)."""
    uid = (user_id or "").strip()
    if not uid:
        return
    _LAST_UPLOAD_RELATIVE_PATH_BY_USER.pop(uid, None)
    _LAST_UPLOAD_RELATIVE_PATH_BY_USER_AND_TYPE.pop(uid, None)

