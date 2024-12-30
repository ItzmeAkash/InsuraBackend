from pydantic import BaseModel

class UserInput(BaseModel):
    user_id: str
    message: str

