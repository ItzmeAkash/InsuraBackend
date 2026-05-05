"""Main-menu flow entry points: medical, motor, general (car/bike/renew), and claims.

Routing labels live in ``INITIAL_MENU_ROUTES`` in ``initial_flow_routes.py``.
Questions for each area load from ``questions/medical/``, ``questions/motor/``, and
``questions/general/``.
"""

from services.chatbot.menu_flows.claim import get_claim_entry_response
from services.chatbot.menu_flows.general import get_general_insurance_entry_response
from services.chatbot.menu_flows.medical import get_medical_menu_entry_response
from services.chatbot.menu_flows.motor import get_motor_menu_entry_response

__all__ = [
    "get_claim_entry_response",
    "get_general_insurance_entry_response",
    "get_medical_menu_entry_response",
    "get_motor_menu_entry_response",
]
