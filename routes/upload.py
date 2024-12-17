from fastapi import APIRouter, File, UploadFile, HTTPException
import os

# Initialize Router
router = APIRouter()

UPLOAD_DIR = "uploads"

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print(f"Created folder: {UPLOAD_DIR}")

# Track user states globally if needed
user_states = {}

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = ""):
    try:
        # Ensure the uploads folder exists at runtime
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
            print(f"Created folder during upload: {UPLOAD_DIR}")

        # Save the uploaded file
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Update user state with file information
        if user_id:
            if user_id not in user_states:
                user_states[user_id] = {"responses": {}}
            user_states[user_id]["responses"]["pre_existing_conditions_file"] = file_location

        return {
            "message": "File uploaded successfully",
            "file_path": file_location,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
