from re import search
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import (
    chat,
    searchInternet,
    upload,
    pdf2text,
    excel_upload,
    language_detection,
)
from routes.pdf2text import get_pdf
from fastapi import FastAPI, File, UploadFile, HTTPException
from utils.helper import transcribe_audio
import aiofiles

# Initialize FastAPI app
app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    print(
        f"Received file: {file.filename}, content_type: {file.content_type}, size: {file.size}"
    )
    if not file:
        raise HTTPException(status_code=422, detail="No file provided")
    try:
        async with aiofiles.tempfile.NamedTemporaryFile(
            "wb", delete=False
        ) as temp_file:
            content = await file.read()
            print(f"File content size: {len(content)} bytes")
            await temp_file.write(content)
            temp_file_path = temp_file.name

        async with aiofiles.open(temp_file_path, "rb") as f:
            audio_data = await f.read()
            print(f"Audio data size: {len(audio_data)} bytes")

        transcript = await transcribe_audio(audio_data)
        if transcript is None:
            print("Transcription returned None")
            raise HTTPException(status_code=500, detail="Transcription failed")

        return {"transcript": transcript}
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"Error details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        import os

        if "temp_file_path" in locals():
            os.unlink(temp_file_path)


# Routes
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(searchInternet.router)
app.include_router(pdf2text.router)
app.include_router(upload.router)
app.include_router(excel_upload.router)
app.include_router(language_detection.router)
# app.include_router(livekitToken.router)
