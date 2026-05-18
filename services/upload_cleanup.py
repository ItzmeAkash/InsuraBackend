"""Delete local ``uploads/`` artifacts when a chat session finishes."""

from __future__ import annotations

import os
import shutil
from typing import Any

from services.chatbot.question_store import (
    clear_user_upload_registry,
    snapshot_user_upload_paths,
)
from services.claim.flow import MEDICAL_CLAIM_FLOW, MOTOR_CLAIM_FLOW
from services.general.document_upload_service import general_upload_storage_dir

MOTOR_QUOTE_FLOWS = frozenset({"car_questions"})
GENERAL_QUOTE_FLOWS = frozenset({"general_insurance"})
CLAIM_UPLOAD_FLOWS = frozenset({MOTOR_CLAIM_FLOW, MEDICAL_CLAIM_FLOW})

_PATH_KEYS_IN_NESTED_DICTS: frozenset[str] = frozenset(
    {
        "stored_relative_path",
        "stored_path",
        "document_stored_path",
        "upload_relative_path",
        "document_path",
        "local_path",
        "file_path",
        "pre_existing_conditions_file",
    }
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_upload_file_path(relative: str) -> str | None:
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


def _normalize_stored_rel_path(p: str) -> str:
    s = (p or "").strip().replace("\\", "/")
    if not s or s.startswith(("http://", "https://")):
        return ""
    if ".." in s.split("/"):
        return ""
    return s


def _uploads_root_abs() -> str:
    return os.path.abspath(os.path.join(_repo_root(), "uploads"))


def _is_under_uploads_root(candidate_abs: str) -> bool:
    try:
        c = os.path.realpath(candidate_abs)
        u = os.path.realpath(_uploads_root_abs())
    except OSError:
        return False
    return c == u or c.startswith(u + os.sep)


def safe_unlink_uploads_path(rel_or_abs: str) -> None:
    norm = _normalize_stored_rel_path(rel_or_abs)
    if not norm:
        return
    abs_path = _resolve_upload_file_path(norm)
    if not abs_path or not _is_under_uploads_root(abs_path):
        return
    try:
        os.remove(abs_path)
    except OSError:
        pass


def _collect_upload_paths_from_object(obj: Any, bucket: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _PATH_KEYS_IN_NESTED_DICTS and isinstance(v, str):
                n = _normalize_stored_rel_path(v)
                if n:
                    bucket.add(n)
            _collect_upload_paths_from_object(v, bucket)
    elif isinstance(obj, list):
        for item in obj:
            _collect_upload_paths_from_object(item, bucket)
    elif isinstance(obj, str):
        s = obj.strip().replace("\\", "/")
        if (
            len(s) < 512
            and s.startswith("uploads/")
            and not s.startswith(("http://", "https://"))
            and ".." not in s
        ):
            bucket.add(s)


def _try_remove_general_user_upload_dir(user_id: str) -> None:
    uid = (user_id or "").strip()
    if not uid:
        return
    try:
        user_dir = general_upload_storage_dir(uid)
        if user_dir.is_dir():
            shutil.rmtree(user_dir, ignore_errors=True)
    except OSError:
        pass


def wipe_session_local_uploads(
    *,
    user_id: str,
    responses: dict[str, Any],
    extra_relative_paths: list[str] | None = None,
    remove_general_user_dir: bool = False,
) -> None:
    """Delete tracked files under ``uploads/`` and clear in-memory upload hints."""
    paths: set[str] = set()
    for p in snapshot_user_upload_paths(user_id):
        n = _normalize_stored_rel_path(p)
        if n:
            paths.add(n)
    _collect_upload_paths_from_object(responses, paths)
    if extra_relative_paths:
        for raw in extra_relative_paths:
            n = _normalize_stored_rel_path(str(raw))
            if n:
                paths.add(n)
    for rel in paths:
        safe_unlink_uploads_path(rel)
    clear_user_upload_registry(user_id)
    if remove_general_user_dir:
        _try_remove_general_user_upload_dir(user_id)


def wipe_flow_session_upload_files(
    *,
    user_id: str,
    current_flow: str,
    responses: dict[str, Any],
) -> None:
    """Remove local uploads when motor quote, general insurance, or claim flow ends."""
    flow = (current_flow or "").strip()
    if not flow or not isinstance(responses, dict):
        return

    extra: list[str] = []
    if flow in MOTOR_QUOTE_FLOWS:
        motor_extra = responses.get("_motor_local_upload_paths")
        if isinstance(motor_extra, list):
            extra.extend(str(p) for p in motor_extra)

    if flow not in MOTOR_QUOTE_FLOWS | GENERAL_QUOTE_FLOWS | CLAIM_UPLOAD_FLOWS:
        return

    wipe_session_local_uploads(
        user_id=user_id,
        responses=responses,
        extra_relative_paths=extra or None,
        remove_general_user_dir=flow in GENERAL_QUOTE_FLOWS | CLAIM_UPLOAD_FLOWS,
    )
    if flow in MOTOR_QUOTE_FLOWS:
        responses.pop("_motor_local_upload_paths", None)
