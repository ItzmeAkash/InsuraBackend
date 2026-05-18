from __future__ import annotations

# Shared flow ids for the claim domain.
CLAIM_ROUTER_FLOW = "claim_router"
MOTOR_CLAIM_FLOW = "motor_claim"
MEDICAL_CLAIM_FLOW = "medical_claim"

# Claim policy choices used by the claim router.
CLAIM_POLICY_OPTIONS = ("Motor insurance", "Medical insurance")


def repair_workshop_paged_options(
    full_options: list[str], page: int, page_size: int = 10
) -> list[str]:
    """First page of workshop / recovery providers plus ``More`` when needed."""
    safe_page = max(0, page)
    start = safe_page * page_size
    end = start + page_size
    page_options = full_options[start:end]
    if end < len(full_options):
        return [*page_options, "More"]
    return page_options
