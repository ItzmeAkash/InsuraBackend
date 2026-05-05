"""Coarse ``flow_type`` labels for API responses (client routing)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from services.claim.flow import MEDICAL_CLAIM_FLOW
from services.medical.flow import medical_flow_service
from services.motor.flow import motor_flow_service

_MOTOR_SUBFLOWS = frozenset({"car_questions", "bike_questions", "existing_policy"})


def resolve_api_flow_type(current_flow: str | None) -> str:
    """Map internal ``conversation_state[''current_flow'']`` to a stable API enum."""
    cf = (current_flow or "").strip()
    if not cf:
        return "initial"
    if cf == "general_insurance":
        return "general_insurance"
    if motor_flow_service.is_claim_router_flow(cf):
        return "claim"
    if motor_flow_service.is_claim_flow(cf):
        return "claim"
    if cf == MEDICAL_CLAIM_FLOW:
        return "claim"
    if motor_flow_service.is_flow(cf) or cf in _MOTOR_SUBFLOWS:
        return "motor"
    if medical_flow_service.is_flow(cf) or cf == "individual":
        return "medical"
    if cf == "initial":
        return "initial"
    return "initial"


def attach_flow_type_to_chat_response(
    result: Any,
    *,
    user_id: str,
    user_states: Mapping[str, Any],
) -> Any:
    """Shallow-copy dict responses and set ``flow_type`` from persisted chat state."""
    if not isinstance(result, dict):
        return result
    uid = (user_id or "").strip()
    state = user_states.get(uid) if uid else None
    current_flow = None
    if isinstance(state, dict):
        current_flow = state.get("current_flow")
    out = dict(result)
    out["flow_type"] = resolve_api_flow_type(
        current_flow if isinstance(current_flow, str) else None
    )
    return out
