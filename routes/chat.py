from fastapi import APIRouter
from models.model import UserInput
from services.chatbot.flow_labels import attach_flow_type_to_chat_response
from services.chatbot.question_store import user_states
from services.llm_services import process_user_input

router = APIRouter()
#Chat
@router.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    result = process_user_input(user_input)
    return attach_flow_type_to_chat_response(
        result, user_id=user_input.user_id, user_states=user_states
    )
