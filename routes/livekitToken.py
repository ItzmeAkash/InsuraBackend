import os
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


# Request models for POST requests
class TokenRequest(BaseModel):
    identity: Optional[str] = None
    name: Optional[str] = None
    room: Optional[str] = None


def _generate_token(identity: str, name: str, room: str):
    """Helper function to generate LiveKit token"""
    # Get API credentials from environment
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500, detail="LiveKit API credentials not configured"
        )

    # Import livekit api
    try:
        from livekit import api
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LiveKit package not installed. Please install livekit-api package.",
        )

    # Generate token
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )

    return {
        "token": token.to_jwt(),
        "url": os.getenv("LIVEKIT_URL", ""),
        "room": room,
        "identity": identity,
        "name": name,
    }


@router.get("/getToken")
def getToken_get(
    identity: Optional[str] = Query(default="user", description="User identity"),
    name: Optional[str] = Query(default="User", description="User name"),
    room: Optional[str] = Query(default="my-room", description="Room name"),
):
    try:
        return _generate_token(identity, name, room)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/getToken")
def getToken_post(request_body: Optional[TokenRequest] = None):
    try:
        # Get parameters from JSON body or use defaults
        # This matches the Flask behavior: data.get("key", "default")
        identity = (
            request_body.identity
            if request_body and request_body.identity is not None
            else "user"
        )
        name = (
            request_body.name
            if request_body and request_body.name is not None
            else "User"
        )
        room = (
            request_body.room
            if request_body and request_body.room is not None
            else "my-room"
        )

        return _generate_token(identity, name, room)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "livekit_configured": bool(
            os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET")
        ),
    }
