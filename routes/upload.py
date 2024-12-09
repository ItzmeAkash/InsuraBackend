from fastapi import APIRouter, File, UploadFile
import os
import json

# Initialize Router
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Track user states globally if needed
user_states = {}

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = ""):
    # Save the uploaded file
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Update user state with file information
    if user_id in user_states:
        user_states[user_id]["responses"]["pre_existing_conditions_file"] = file_location

    return {"message": "File uploaded successfully", "file_path": file_location}
