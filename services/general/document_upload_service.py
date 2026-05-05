"""Persist general-insurance uploads and build chat payloads for ``process_user_input``."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

GENERAL_UPLOAD_SUBDIR = "general"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024

ALLOWED_GENERAL_UPLOAD_FILE_TYPES = frozenset(
    {
        "travel_insurance_form",
        "trade_license",
        "vat_certificate",
    }
)


def normalize_general_upload_file_type(raw: str) -> str:
    t = (raw or "").strip().lower().replace("-", "_")
    if t in ("trade_licence",):
        return "trade_license"
    if t in (
        "travel",
        "travel_insurance",
        "travel_form",
        "travel_insurance_form",
        "general_insurance_form",
        "general_insurance",
    ):
        return "travel_insurance_form"
    return t


def sanitize_upload_filename(name: str) -> str:
    base = (name or "").strip() or "document"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return (cleaned[:200] if cleaned else "document")


def general_upload_storage_dir(user_id: str) -> Path:
    return Path("uploads") / GENERAL_UPLOAD_SUBDIR / user_id.strip()


def save_general_upload_file(
    *,
    user_id: str,
    original_filename: str,
    content: bytes,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    """
    Write bytes under ``uploads/general/{user_id}/``.
    Returns a workspace-relative path using forward slashes.
    """
    if len(content) > max_bytes:
        raise ValueError("file_too_large")
    safe = sanitize_upload_filename(original_filename)
    disk_name = f"{uuid.uuid4().hex[:10]}_{safe}"
    dest_dir = general_upload_storage_dir(user_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    full_path = dest_dir / disk_name
    full_path.write_bytes(content)
    return str(full_path).replace("\\", "/")


def build_general_upload_chat_message(
    *,
    file_type: str,
    stored_relative_path: str,
    original_filename: str,
) -> str:
    """
    JSON message consumed by ``parse_general_upload_payload`` / travel follow-up phases.
    """
    payload: dict = {
        "general_document_upload": True,
        "file_type": file_type,
        "stored_path": stored_relative_path,
        "original_filename": original_filename,
    }
    if file_type == "travel_insurance_form":
        payload["travel_insurance_form"] = True
        payload["destination"] = "general_upload_endpoint"
    elif file_type == "trade_license":
        payload["trade_license"] = True
    elif file_type == "vat_certificate":
        payload["vat_certificate"] = True
    return json.dumps(payload)
