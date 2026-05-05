"""Shared chatbot copy and URLs. Change here (or via env) instead of duplicating in handlers."""

from __future__ import annotations

import os

# Full Google review URL for Wehbe Insurance Services (same link everywhere).
_DEFAULT_WEHBE_GOOGLE_REVIEW_URL = "https://www.google.com/search?client=ms-android-samsung-ss&sca_esv=4eb717e6f42bf628&sxsrf=AHTn8zprabdPVFL3C2gXo4guY8besI3jqQ:1744004771562&q=wehbe+insurance+services+llc+reviews&uds=ABqPDvy-z0dcsfm2PY76_gjn-YWou9-AAVQ4iWjuLR6vmDV0vf3KpBMNjU5ZkaHGmSY0wBrWI3xO9O55WuDmXbDq6a3SqlwKf2NJ5xQAjebIw44UNEU3t4CpFvpLt9qFPlVh2F8Gfv8sMuXXSo2Qq0M_ZzbXbg2c323G_bE4tVi7Ue7d_sW0CrnycpJ1CvV-OyrWryZw_TeQ3gLGDgzUuHD04MpSHquYZaSQ0_mIHLWjnu7fu8c7nb6_aGDb_H1Q-86fD2VmWluYA5jxRkC9U2NsSwSSXV4FPW9w1Q2T_Wjt6koJvLgtikd66MqwYiJPX2x9MwLhoGYlpTbKtkJuHwE9eM6wQgieChskow6tJCVjQ75I315dT8n3tUtasGdBkprOlUK9ibPrYr9HqRz4AwzEQaxAq9_EDcsSG_XW0CHuqi2lRKHw592MlGlhjyQibXKSZJh-v3KW4wIVqa-2x0k1wfbZdpaO3BZaKYCacLOxwUKTnXPbQqDPLQDeYgDBwaTLvaCN221H&si=APYL9bvoDGWmsM6h2lfKzIb8LfQg_oNQyUOQgna9TyfQHAoqUvvaXjJhb-NHEJtDKiWdK3OqRhtZNP2EtNq6veOxTLUq88TEa2J8JiXE33-xY1b8ohiuDLBeOOGhuI1U6V4mDc9jmZkDoxLC9b6s6V8MAjPhY-EC_g%3D%3D&sa=X&sqi=2&ved=2ahUKEwi05JSHnMWMAxUw8bsIHRRCDd0Qk8gLegQIHxAB&ictx=1&stq=1&cs=0&lei=o2bzZ_SGIrDi7_UPlIS16A0#ebo=1"

WEHBE_GOOGLE_REVIEW_URL: str = os.environ.get(
    "WEHBE_GOOGLE_REVIEW_URL",
    _DEFAULT_WEHBE_GOOGLE_REVIEW_URL,
)

# Advisor-code completion flow (Wehbe wording).
MEDICAL_CUSTOMER_PLAN_SUCCESS_RESPONSE = (
    "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance "
    "to assist you further with your enquiry. Please find the link below to view your quotation:"
)

# Medical individual flow (add-member / no advisor): main message before ``link`` (customer_plan).
MEDICAL_INDIVIDUAL_COMPLETION_RESPONSE = (
    "Thank you for providing all the required details.\n\n"
    "Please use the link below to review and compare the available proposal options. "
    'Please select your preferred choice and click "Buy" to proceed.'
)

_DEFAULT_WEHBE_REVIEW_INVITE_MESSAGE = (
    "If you are satisfied with Wehbe (Broker) services, please leave a review to share your "
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
PASSKEY_VALID_CODE: str = os.environ.get("PASSKEY_VALID_CODE", "5514")
