from __future__ import annotations

from services.medical.conversation_handler import MedicalConversationHandler
from services.medical.individual_handler import MedicalIndividualHandler
from services.motor.claim_handler import MotorClaimHandler
from services.motor.conversation_handler import MotorConversationHandler

medical_conversation_handler = MedicalConversationHandler()
medical_individual_handler = MedicalIndividualHandler()
motor_conversation_handler = MotorConversationHandler()
motor_claim_handler = MotorClaimHandler()

