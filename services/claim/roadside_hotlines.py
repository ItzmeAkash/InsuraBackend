"""Roadside assistance hotlines shown after motor claim recovery provider selection."""

from __future__ import annotations

# Insurer / provider name → official roadside or support number(s).
MOTOR_CLAIM_ROADSIDE_HOTLINES: dict[str, str] = {
    "Adamjee": "8007326",
    "Al Sagr": "8007541",
    "Al Wathba": "600575751",
    "AXA": "800292",
    "DNI": "600575751",
    "Ihouse": "600575751",
    "Oman": "8006565",
    "Orient": "600575751",
    "QIC": "Toll Free Number 8004742 or +971 4 2224045 (Dubai) or +971 2 6769466 (Abu Dhabi)",
    "New India": "800247772",
    "Tokio Marine": "600 50 8181",
    "Union Insurance": "600575751",
    "Watania": "600575751",
    "Alliance": "600 50 81 81",
    "Arabia Insurance": "80009104001",
    "Arabian": "8007326 - Dubai, 600 575751 - Abu Dhabi",
    "Fidelity United": "600575751",
    "Orient Takaful": "600508181",
    "Sukoon": "800 6565 (Within the UAE)\n+971 4 387 6649 (Outside the UAE)",
    "LIVA": "600544060",
    "RAK": "IMC on Toll Free Number: 600575751",
    "Salama": "800725262",
    "Al Buhairah": "8008181",
    "GIG": "800292",
    "National General": "8004900",
    "Oriental": "600575751",
    "Noor": "8004101",
    "RSA": "800462372",
    "Emirates": "Roadside Assistance 24/7 Emergency Hotline: 80073",
    "Methaq": "600565695",
    "Dubai": "800382467",
}

# Same order as repair-workshop list in ``questions/claim/motor/questions.json``.
MOTOR_CLAIM_RECOVERY_PROVIDER_OPTIONS: tuple[str, ...] = (
    "Adamjee",
    "Al Sagr",
    "Al Wathba",
    "AXA",
    "DNI",
    "Ihouse",
    "Oman",
    "Orient",
    "QIC",
    "Sukoon",
    "LIVA",
    "RAK",
    "Salama",
    "Al Buhairah",
    "GIG",
    "National General",
    "Oriental",
    "Noor",
    "RSA",
    "Emirates",
    "Methaq",
    "Dubai",
    "New India",
    "Tokio Marine",
    "Union Insurance",
    "Watania",
    "Alliance",
    "Arabia Insurance",
    "Arabian",
    "Fidelity United",
    "Orient Takaful",
)


def format_motor_claim_roadside_hotline_message(provider: str) -> str:
    """Build user-facing hotline text for the selected insurer / provider."""
    name = (provider or "").strip()
    if not name:
        return ""

    hotline = MOTOR_CLAIM_ROADSIDE_HOTLINES.get(name, "").strip()
    if not hotline:
        return ""

    if name == "Emirates":
        return (
            "Emirates\n\n"
            "EMIRATES INSURANCE\n\n"
            "This is the number you can reach them anytime: "
            "Roadside Assistance 24/7 Emergency Hotline: 80073"
        )

    return f"{name}\n\nThis is the number you can reach them anytime: {hotline}"
