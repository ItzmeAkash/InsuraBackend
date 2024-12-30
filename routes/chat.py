from fastapi import APIRouter
from models.model import UserInput
from services.llm_services import process_user_input

router = APIRouter()

@router.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    return process_user_input(user_input)
