"""Registry for the first main-menu option → flow transition (standardized, data-driven)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from services.chatbot.menu_flows import (
    get_claim_entry_response,
    get_general_insurance_entry_response,
    get_medical_menu_entry_response,
    get_motor_menu_entry_response,
)
from services.claim.flow import CLAIM_ROUTER_FLOW, MOTOR_CLAIM_FLOW


def _first_question_great_choice(questions: list[Any]) -> tuple[str, list[str]]:
    """First step message for flows that start with a plain question list."""
    first = questions[0]
    if isinstance(first, dict):
        return (
            f"Great choice! {first['question']}",
            first.get("options", []),
        )
    return f"Great choice! {first}", []


EntryKind = Literal[
    "medical",
    "motor",
    "motor_claim",
    "claim_router_entry",
    "first_question",
    "general_submenu",
]


@dataclass(frozen=True)
class InitialMenuRoute:
    """Maps one label from ``initial_questions`` options to a flow and how to build the first reply."""

    flow_id: str
    entry: EntryKind
    # For entry == "first_question", which list to use (resolved via InitialMenuContext).
    questions_attr: str | None = None


@dataclass
class InitialMenuContext:
    medical_questions: list[Any]
    motor_insurance_questions: list[Any]
    car_questions: list[Any]
    bike_questions: list[Any]
    existing_policy_questions: list[Any]
    motor_claim: list[Any]
    claim_router_questions: list[Any]
    general_insurance_questions: list[Any]


_QUESTIONS_FOR_FIRST = {
    "car_questions": lambda c: c.car_questions,
    "bike_questions": lambda c: c.bike_questions,
    "existing_policy_questions": lambda c: c.existing_policy_questions,
    "general_insurance_questions": lambda c: c.general_insurance_questions,
}

# Map legacy / old UI labels to current main-menu option names (for stored sessions & copy changes).
_MAIN_MENU_OPTION_ALIASES: dict[str, str] = {
    "Purchase a Medical Insurance": "Medical Insurance",
    "Purchase a Motor Insurance": "Motor Insurance",
    "Claim a Motor Insurance": "Claim Insurance",
}


def _canonical_main_menu_option(label: str) -> str:
    return _MAIN_MENU_OPTION_ALIASES.get(label, label)


# Single source of truth: option label (English, as stored in responses / JSON) → route.
INITIAL_MENU_ROUTES: dict[str, InitialMenuRoute] = {
    "Medical Insurance": InitialMenuRoute(
        flow_id="medical_insurance",
        entry="medical",
    ),
    "Motor Insurance": InitialMenuRoute(
        flow_id="motor_insurance",
        entry="motor",
    ),
    "General Insurance": InitialMenuRoute(
        flow_id="general_insurance",
        entry="general_submenu",
    ),
    "Claim Insurance": InitialMenuRoute(
        flow_id=CLAIM_ROUTER_FLOW,
        entry="claim_router_entry",
    ),
    # Motor submenu (after choosing Motor Insurance)
    "New Insurance": InitialMenuRoute(
        flow_id="car_questions",
        entry="first_question",
        questions_attr="car_questions",
    ),
    "Renewal": InitialMenuRoute(
        flow_id="car_questions",
        entry="first_question",
        questions_attr="car_questions",
    ),
}


def resolve_initial_menu_choice(
    matched_option: str,
    *,
    conversation_state: dict[str, Any],
    ctx: InitialMenuContext,
) -> tuple[str, list[str]] | None:
    """
    If ``matched_option`` is a registered main-menu choice, updates ``conversation_state``
    (flow + question index) and returns ``(response_message, next_options)`` for
    ``format_response_in_language``. Otherwise returns ``None``.
    """
    canonical = _canonical_main_menu_option(matched_option)
    route = INITIAL_MENU_ROUTES.get(canonical) or INITIAL_MENU_ROUTES.get(
        matched_option
    )
    if route is None:
        return None

    conversation_state["current_flow"] = route.flow_id
    conversation_state["current_question_index"] = 0

    if route.entry == "medical":
        return get_medical_menu_entry_response(ctx.medical_questions)
    if route.entry == "motor":
        return get_motor_menu_entry_response(ctx.motor_insurance_questions)
    if route.entry == "claim_router_entry":
        first = ctx.claim_router_questions[0]
        if isinstance(first, dict):
            return first["question"], first.get("options", [])
        return str(first), []

    if route.entry == "motor_claim":
        return get_claim_entry_response(ctx.motor_claim)
    if route.entry == "general_submenu":
        return get_general_insurance_entry_response(ctx.general_insurance_questions)
    if route.entry == "first_question":
        assert route.questions_attr is not None
        getter = _QUESTIONS_FOR_FIRST.get(route.questions_attr)
        if getter is None:
            raise ValueError(
                f"Unknown questions_attr for initial route: {route.questions_attr}"
            )
        questions = getter(ctx)
        return _first_question_great_choice(questions)

    raise RuntimeError(f"Unhandled entry kind: {route.entry}")
