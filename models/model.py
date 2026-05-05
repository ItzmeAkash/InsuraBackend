from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class UserInput(BaseModel):
    """Incoming chat payload. Mobile apps often send ``filePath`` (camelCase)."""

    model_config = ConfigDict(populate_by_name=True)

    user_id: str
    message: str
    file_path: Optional[str] = Field(default=None, alias="filePath")

