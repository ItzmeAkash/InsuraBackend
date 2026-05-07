"""Shared chatbot copy and URLs. Change here (or via env) instead of duplicating in handlers."""

from __future__ import annotations

import os

# Full Google review URL for Wehbe Insurance Services (same link everywhere).
_DEFAULT_WEHBE_GOOGLE_REVIEW_URL = "https://insuranceclub.ae/"

WEHBE_GOOGLE_REVIEW_URL: str = os.environ.get(
    "WEHBE_GOOGLE_REVIEW_URL",
    _DEFAULT_WEHBE_GOOGLE_REVIEW_URL,
)

# Advisor-code completion flow (Wehbe wording).
MEDICAL_CUSTOMER_PLAN_SUCCESS_RESPONSE = (
    "Thank you for sharing the details. We will inform Insura "
    "to assist you further with your enquiry. Please find the link below to view your quotation:"
)

# Medical individual flow (add-member / no advisor): main message before ``link`` (customer_plan).
MEDICAL_INDIVIDUAL_COMPLETION_RESPONSE = (
    "Thank you for providing all the required details.\n\n"
    "Please use the link below to review and compare the available proposal options. "
    'Please select your preferred choice and click "Buy" to proceed.'
)

_DEFAULT_WEHBE_REVIEW_INVITE_MESSAGE = (
    "If you are satisfied with Insura services, please leave a review to share your "
    "happiness with others!!😊"
)

WEHBE_REVIEW_INVITE_MESSAGE: str = os.environ.get(
    "WEHBE_REVIEW_INVITE_MESSAGE",
    _DEFAULT_WEHBE_REVIEW_INVITE_MESSAGE,
)

_DEFAULT_INSURA_REVIEW_INVITE_MESSAGE = (
    "If you are satisfied with Insura services, please leave a review for sharing "
    "happiness to others!!😊"
)

INSURA_REVIEW_INVITE_MESSAGE: str = os.environ.get(
    "INSURA_REVIEW_INVITE_MESSAGE",
    _DEFAULT_INSURA_REVIEW_INVITE_MESSAGE,
)

INSURA_REVIEW_URL: str = os.environ.get(
    "INSURA_REVIEW_URL",
    WEHBE_GOOGLE_REVIEW_URL,
)

# Expected user entry for the passkey step. Override in env for different environments.
PASSKEY_VALID_CODE: str = os.environ.get("PASSKEY_VALID_CODE", "6754")
