from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq.chat_models import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    groq_proxy=None,
)


def get_language_code(language_name: str) -> str:
    language_map = {
        "english": "en",
        "arabic": "ar",
        "hindi": "hi",
        "urdu": "ur",
        "french": "fr",
        "spanish": "es",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
    }
    return language_map.get(language_name.lower(), "en")


def detect_language(text: str) -> dict:
    text_clean = text.strip()
    if (
        text_clean.isdigit()
        or len(text_clean) <= 2
        or text_clean.lower() in ["yes", "no", "y", "n", "ok", "okay", "1", "2", "3", "4", "5"]
    ):
        return {"language": "English", "code": "en"}

    detection_prompt = f"""Detect the language of this text: "{text}"

Respond ONLY in this exact JSON format:
{{
    "language": "Language Name",
    "code": "language_code"
}}

Examples:
- For English: {{"language": "English", "code": "en"}}
- For Arabic: {{"language": "Arabic", "code": "ar"}}
- For Hindi: {{"language": "Hindi", "code": "hi"}}
- For Urdu: {{"language": "Urdu", "code": "ur"}}
- For French: {{"language": "French", "code": "fr"}}

If mixed languages, identify the dominant one.
If the text is mostly numbers or very short, default to English."""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a language detection expert. Respond ONLY with valid JSON as instructed. For numeric inputs or very short text, default to English."
                ),
                HumanMessage(content=detection_prompt),
            ]
        )
        return json.loads(response.content.strip())
    except Exception:
        return {"language": "English", "code": "en"}


def translate_text(text: str, target_language: str, source_language: str = "auto") -> str:
    if source_language == "auto":
        translation_prompt = f"""Translate this text to {target_language}. Maintain the same tone and meaning.

Text to translate: "{text}"

Respond ONLY with the translated text, nothing else."""
    else:
        translation_prompt = f"""Translate this text from {source_language} to {target_language}. Maintain the same tone and meaning.

Text to translate: "{text}"

Respond ONLY with the translated text, nothing else."""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=f"You are a professional translator. Translate accurately while maintaining the original meaning and tone. For insurance and technical terms, use appropriate terminology in {target_language}."
                ),
                HumanMessage(content=translation_prompt),
            ]
        )
        return response.content.strip()
    except Exception:
        return text


def validate_response_multilingual(user_response: str, expected_values: list, user_language: str) -> dict:
    validation_prompt = f"""You are validating a user's response for an insurance chatbot.

User's response: "{user_response}"
User's language: {user_language}

Expected valid values (in English): {", ".join(expected_values)}

Determine if the user's response matches any of the expected values. Consider:
1. The response might be in {user_language}, so check for translations
2. Consider synonyms and variations
3. Be flexible but ensure accuracy

Respond ONLY in this exact JSON format:
{{
    "is_valid": true/false,
    "matched_value": "English version of the matched option" or null,
    "explanation": "Brief explanation"
}}"""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a validation expert for a multilingual insurance chatbot. Be accurate and consider language variations."
                ),
                HumanMessage(content=validation_prompt),
            ]
        )
        return json.loads(response.content.strip())
    except Exception as exc:
        return {"is_valid": False, "matched_value": None, "explanation": f"Validation error: {exc}"}


def format_response_in_language(
    response_text: str,
    options: list,
    user_language: str,
    message_type: str = None,
    document_type: str = None,
) -> dict:
    if user_language.lower() in ["english", "en"]:
        result = {"response": response_text}
        if options:
            result["options"] = ", ".join(options)
        if message_type:
            result["message_type"] = message_type
        if document_type:
            result["document_type"] = document_type
        result["language"] = "English"
        result["language_code"] = "en"
        return result

    translated_response = translate_text(response_text, user_language)
    result = {"response": translated_response}
    if options:
        translated_options = [translate_text(opt, user_language) for opt in options]
        result["options"] = ", ".join(translated_options)
    if message_type:
        result["message_type"] = message_type
    if document_type:
        result["document_type"] = document_type
    result["language"] = user_language
    result["language_code"] = get_language_code(user_language)
    return result


def translate_to_english_for_storage(text: str, detected_language: str) -> str:
    if detected_language.lower() in ["english", "en"]:
        return text
    return translate_text(text, "English", detected_language)


def detect_document_type_from_question(question_text: str) -> tuple:
    question_lower = question_text.lower()
    if "front page" in question_lower and ("document" in question_lower or "emirates" in question_lower):
        return ("document_upload_request", "emirates_id_front")
    elif "back page" in question_lower and ("document" in question_lower or "emirates" in question_lower):
        return ("document_upload_request", "emirates_id_back")
    elif "driving license" in question_lower or "driving licence" in question_lower:
        return ("document_upload_request", "driving_license")
    elif "emirates id" in question_lower:
        return ("document_upload_request", "emirates_id")
    elif "mulkiya" in question_lower:
        return ("document_upload_request", "mulkiya")
    elif "excel" in question_lower and "upload" in question_lower:
        return ("document_upload_request", "excel")
    elif "upload" in question_lower and "document" in question_lower:
        return ("document_upload_request", "emirates_id")
    elif "trade licence" in question_lower or "trade license" in question_lower:
        return ("document_upload_request", "trade_license")
    elif "vat certificate" in question_lower:
        return ("document_upload_request", "vat_certificate")
    return (None, None)

