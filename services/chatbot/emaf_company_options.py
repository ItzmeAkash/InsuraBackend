"""EMAF insurance company dropdown options and numeric mapping (single source)."""

from __future__ import annotations

EMAF_INSURANCE_VALID_OPTIONS: list[str] = [
    "Takaful Emarat (Ecare)",
    "National Life & General Insurance (Innayah)",
    "Takaful Emarat (Aafiya)",
    "National Life & General Insurance (NAS)",
    "Orient UNB Takaful (Nextcare)",
    "Orient Mednet (Mednet)",
    "Al Sagr Insurance (Nextcare)",
    "RAK Insurance (Mednet)",
    "Dubai Insurance (Dubai Care)",
    "Fidelity United (Nextcare)",
    "Salama April International (Salama)",
    "Sukoon (Sukoon)",
    "Orient basic",
    "Daman",
    "Dubai insurance(Mednet)",
    "Takaful Emarat(NAS)",
    "Takaful emarat(Nextcare)",
]

EMAF_COMPANY_NUMBER_BY_OPTION: dict[str, int] = {
    "Takaful Emarat (Ecare)": 1,
    "National Life & General Insurance (Innayah)": 2,
    "Takaful Emarat (Aafiya)": 3,
    "National Life & General Insurance (NAS)": 4,
    "Orient UNB Takaful (Nextcare)": 6,
    "Orient Mednet (Mednet)": 7,
    "Al Sagr Insurance (Nextcare)": 8,
    "RAK Insurance (Mednet)": 9,
    "Dubai Insurance (Dubai Care)": 10,
    "Fidelity United (Nextcare)": 11,
    "Salama April International (Salama)": 12,
    "Sukoon (Sukoon)": 13,
    "Orient basic": 14,
    "Daman": 15,
    "Dubai insurance(Mednet)": 16,
    "Takaful Emarat(NAS)": 17,
    "Takaful emarat(Nextcare)": 18,
}
