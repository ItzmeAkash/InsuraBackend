"""
Language Detection API Endpoint
Detects the language of text and optionally translates it
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.llm_services import translate_text, detect_language, get_language_code

router = APIRouter()


class LanguageDetectionRequest(BaseModel):
    text: str
    translate_to: str = None  # Optional: if provided, also translates the text


class LanguageDetectionResponse(BaseModel):
    detected_language: str
    language_code: str
    text: str
    translated_text: str = None
    confidence: str = "high"  # Can be enhanced with actual confidence scores


@router.post("/detect-language/", response_model=LanguageDetectionResponse)
async def detect_text_language(request: LanguageDetectionRequest):
    """
    Detect the language of provided text and optionally translate it.

    Args:
        text: The text to analyze
        translate_to: (Optional) Target language for translation

    Returns:
        LanguageDetectionResponse with detected language and optional translation
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Detect language (returns dict with 'language' and 'code' keys)
        detected_lang = detect_language(request.text)

        response_data = {
            "detected_language": detected_lang.get("language", "English"),
            "language_code": detected_lang.get("code", "en"),
            "text": request.text,
        }

        # Translate if requested
        if request.translate_to:
            translated = translate_text(request.text, request.translate_to)
            response_data["translated_text"] = translated

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Language detection failed: {str(e)}"
        )


class TranslationRequest(BaseModel):
    text: str
    source_language: str = None  # Auto-detect if not provided
    target_language: str = "English"


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    source_language_code: str
    target_language: str
    target_language_code: str


@router.post("/translate/", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    """
    Translate text from one language to another.

    Args:
        text: Text to translate
        source_language: Source language (auto-detected if not provided)
        target_language: Target language (default: English)

    Returns:
        TranslationResponse with translation details
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Detect source language if not provided
        source_lang = request.source_language
        if not source_lang:
            detected = detect_language(request.text)
            source_lang = detected.get("language", "English")

        # Get language codes
        source_code = get_language_code(source_lang)
        target_code = get_language_code(request.target_language)

        # Translate
        translated = translate_text(request.text, request.target_language, source_lang)

        return {
            "original_text": request.text,
            "translated_text": translated,
            "source_language": source_lang,
            "source_language_code": source_code,
            "target_language": request.target_language,
            "target_language_code": target_code,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


class NormalizeTextRequest(BaseModel):
    texts: list[str]  # Multiple texts to normalize
    normalize_to: str = "English"  # Language to normalize to


class NormalizeTextResponse(BaseModel):
    normalized_texts: list[str]
    target_language: str


@router.post("/normalize-text/", response_model=NormalizeTextResponse)
async def normalize_texts(request: NormalizeTextRequest):
    """
    Normalize multiple texts to a common language for comparison.
    Useful for comparing user input against expected values in different languages.

    Args:
        texts: List of texts to normalize
        normalize_to: Target language (default: English)

    Returns:
        List of normalized texts
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")

        normalized = []
        for text in request.texts:
            if not text:
                normalized.append("")
                continue

            # Detect language and translate if needed
            detected = detect_language(text)
            detected_lang = detected.get("language", "English")

            if (
                detected_lang.lower() in [request.normalize_to.lower(), "english"]
                and request.normalize_to.lower() == "english"
            ):
                normalized.append(text)
            else:
                translated = translate_text(text, request.normalize_to, detected_lang)
                normalized.append(translated)

        return {
            "normalized_texts": normalized,
            "target_language": request.normalize_to,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Text normalization failed: {str(e)}"
        )
