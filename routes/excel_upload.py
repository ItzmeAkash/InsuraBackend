from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import os
import tempfile
from cachetools import TTLCache
from routes.utils import extract_excel_sme_census

# Initialize Router
router = APIRouter()

# Track user states globally if needed
user_states = TTLCache(maxsize=1000, ttl=3600)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload-excel/", tags=["Excel Processing"])
async def upload_excel_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """
    Upload an Excel file and extract employee data from SME Census Sheet format.
    Supports .xlsx and .xls files.
    """
    user_id = user_id.strip()

    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {"responses": {}}

    # Get file extension and convert to lowercase
    file_extension = os.path.splitext(file.filename)[1].lower()

    # Validate file type
    valid_extensions = [".xlsx", ".xls"]

    if file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Only Excel files are allowed. Supported formats: {', '.join(valid_extensions)}",
        )

    # Create a temporary file with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")

            temp_file.write(content)
            temp_file.flush()

            # Process the Excel file
            result = await extract_excel_sme_census(temp_file.name)

            # Update user state with extracted data
            user_states[user_id]["responses"]["excel_employee_data"] = result

            return {
                "message": "Excel file processed successfully",
                "data": result,
                "user_id": user_id,
                "file_path": f"uploads/{file.filename}",  # Return a file path for the chatbot
                "file_name": file.filename,
            }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Excel processing failed: {str(e)}"
            )

        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
