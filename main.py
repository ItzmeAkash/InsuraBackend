from re import search
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat, searchInternet,upload,pdf2text

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

# Routes
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(searchInternet.router)
app.include_router(pdf2text.router)
app.include_router(upload.router)

