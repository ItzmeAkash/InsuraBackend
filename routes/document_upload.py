"""Multipart document uploads that tie into chat state via ``process_user_input``."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.model import UserInput
from services.chatbot.flow_labels import attach_flow_type_to_chat_response
from services.chatbot.question_store import user_states
from services.general.document_upload_service import (
    ALLOWED_GENERAL_UPLOAD_FILE_TYPES,
    DEFAULT_MAX_BYTES,
    build_general_upload_chat_message,
    normalize_general_upload_file_type,
    save_general_upload_file,
)
from services.llm_services import process_user_input

router = APIRouter(tags=["Document upload"])

_CLAIM_UPLOAD_FILE_TYPES = frozenset(
    {
        "vehicle_registration",
        "mulkiya",
        "driving_license",
        "emirates_id",
        "insurance_card",
        "police_verification",
    }
)


def _normalize_upload_type(raw: str) -> str:
    base = normalize_general_upload_file_type(raw.strip())
    if base in ALLOWED_GENERAL_UPLOAD_FILE_TYPES:
        return base
    t = (raw or "").strip().lower().replace("-", "_")
    claim_aliases = {
        "vehicle_registration_card": "vehicle_registration",
        "vehicle_registration_document": "vehicle_registration",
        "vehicle_registration": "vehicle_registration",
        "mulkiya": "mulkiya",
        "driving_licence": "driving_license",
        "driving_license": "driving_license",
        "emirates_id": "emirates_id",
        "insurance_card": "insurance_card",
        "police_verification": "police_verification",
        "police_report": "police_verification",
    }
    return claim_aliases.get(t, t)


@router.post("/upload-document/")
@router.post("/uploaddocument/")
async def upload_document(
    user_id: str = Form(..., description="Same user_id used with POST /chat/"),
    type: str = Form(  # noqa: A002
        ...,
        description=(
            "Upload kind. Examples: general_insurance_form (filled travel/general Excel), "
            "travel_insurance_form, trade_license, vat_certificate."
        ),
    ),
    file_name: str = Form("", description="Optional display name; defaults to uploaded filename"),
    file: UploadFile = File(..., description="Document binary"),
):
    """
    Save the file and forward a structured JSON message into the chat processor.
    Routing is inferred from ``type`` (no separate ``general`` / ``category`` field).
    """
    ft = _normalize_upload_type(type.strip())
    if ft not in ALLOWED_GENERAL_UPLOAD_FILE_TYPES and ft not in _CLAIM_UPLOAD_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid type `{type}`. "
                "Allowed (after normalization): "
                f"{', '.join(sorted([*ALLOWED_GENERAL_UPLOAD_FILE_TYPES, *_CLAIM_UPLOAD_FILE_TYPES]))}. "
                "Examples: general_insurance_form, travel_insurance_form, trade_license, "
                "vat_certificate, police_verification, mulkiya."
            ),
        )

    original_name = (file_name or "").strip() or (file.filename or "document")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        stored_path = save_general_upload_file(
            user_id=user_id,
            original_filename=original_name,
            content=content,
            max_bytes=DEFAULT_MAX_BYTES,
        )
    except ValueError as exc:
        if str(exc) == "file_too_large":
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size ({DEFAULT_MAX_BYTES // (1024 * 1024)} MB)",
            ) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if ft in _CLAIM_UPLOAD_FILE_TYPES:
        # Claim upload steps accept a stored path message.
        message = stored_path
    else:
        message = build_general_upload_chat_message(
            file_type=ft,
            stored_relative_path=stored_path,
            original_filename=original_name,
        )

    chat_result = process_user_input(UserInput(user_id=user_id, message=message))

    if isinstance(chat_result, dict):
        chat_result = dict(chat_result)
        chat_result["upload_meta"] = {
            "type": type.strip(),
            "file_type": ft,
            "stored_path": stored_path,
            "original_filename": original_name,
        }
        return attach_flow_type_to_chat_response(
            chat_result, user_id=user_id, user_states=user_states
        )

    wrapped = {
        "upload_meta": {
            "type": type.strip(),
            "file_type": ft,
            "stored_path": stored_path,
            "original_filename": original_name,
        },
        "chat": chat_result,
    }
    return attach_flow_type_to_chat_response(
        wrapped, user_id=user_id, user_states=user_states
    )
