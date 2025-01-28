from pydantic import BaseModel
from typing import Optional

class UserInput(BaseModel):
    user_id: str
    message: str
    file_path: Optional[str] = None

