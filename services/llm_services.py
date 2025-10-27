import asyncio
from datetime import datetime
from utils.helper import (
    emaf_document,
    fetching_medical_detail,
    is_valid_mobile_number,
    valid_adivisor_code,
)
from utils.helper import (
    get_user_name,
    valid_date_format,
    valid_emirates_id,
    is_valid_name,
)
from utils.question_helper import (
    handle_adiviosr_code,
    handle_client_name_question,
    handle_date_question,
    handle_emaf_document,
    handle_emirate_question,
    handle_emirate_upload_document,
    handle_emirate_upload_document_car_insurance,
    handle_gender,
    handle_individual_sma_choice,
    handle_job_title_question,
    handle_marital_status,
    handle_type_plan_question,
    handle_validate_name,
    handle_visa_issued_emirate_question,
    handle_what_would_you_do_today_question,
    handle_yes_or_no,
)
from langchain_groq.chat_models import ChatGroq
from fastapi import FastAPI, File, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage
from models.model import UserInput
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import initialize_agent
from random import choice
import requests
import json
import re
import os
from dotenv import load_dotenv
from cachetools import TTLCache

load_dotenv()

# API Configuration - Base URLs for InsuranceLab
INSURANCE_LAB_BASE_URL = "https://insurancelab.ae"
INSURANCE_LAB_API_BASE_URL = f"{INSURANCE_LAB_BASE_URL}/Api"
INSURANCE_LAB_SME_ADD_API = f"{INSURANCE_LAB_API_BASE_URL}/sme_add/"
INSURANCE_LAB_SME_PLAN_BASE = f"{INSURANCE_LAB_BASE_URL}/sme_plan"

# Updated
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    groq_proxy=None,
)


# ==================== MULTI-LANGUAGE SUPPORT FUNCTIONS ====================


def get_language_code(language_name: str) -> str:
    """
    Map language name to ISO 639-1 language code.

    Args:
        language_name: Language name (e.g., 'Arabic', 'English', 'Hindi')

    Returns:
        Language code (e.g., 'ar', 'en', 'hi')
    """
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

    # Normalize to lowercase and get code, default to 'en' if not found
    return language_map.get(language_name.lower(), "en")


def detect_language(text: str) -> dict:
    """
    Detect the language of the user's input using LLM.
    Returns a dict with 'language' (e.g., 'Arabic', 'English') and 'code' (e.g., 'ar', 'en')
    """
    # Skip language detection for numeric inputs, very short text, or common responses
    text_clean = text.strip()
    if (
        text_clean.isdigit()
        or len(text_clean) <= 2
        or text_clean.lower()
        in ["yes", "no", "y", "n", "ok", "okay", "1", "2", "3", "4", "5"]
    ):
        print(
            f"[Language Detection] Skipping detection for numeric/short input: '{text_clean}'"
        )
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
        response = llm.invoke([
            SystemMessage(
                content="You are a language detection expert. Respond ONLY with valid JSON as instructed. For numeric inputs or very short text, default to English."
            ),
            HumanMessage(content=detection_prompt),
        ])

        # Parse the response
        result = json.loads(response.content.strip())
        return result
    except (json.JSONDecodeError, Exception) as e:
        # Default to English if detection fails
        print(f"[Language Detection] Error: {e}. Defaulting to English.")
        return {"language": "English", "code": "en"}


def translate_text(
    text: str, target_language: str, source_language: str = "auto"
) -> str:
    """
    Translate text to the target language using LLM.

    Args:
        text: The text to translate
        target_language: Target language (e.g., 'Arabic', 'English', 'Hindi')
        source_language: Source language (default 'auto' for auto-detection)

    Returns:
        Translated text
    """
    if source_language == "auto":
        translation_prompt = f"""Translate this text to {target_language}. Maintain the same tone and meaning.

Text to translate: "{text}"

Respond ONLY with the translated text, nothing else."""
    else:
        translation_prompt = f"""Translate this text from {source_language} to {target_language}. Maintain the same tone and meaning.

Text to translate: "{text}"

Respond ONLY with the translated text, nothing else."""

    try:
        response = llm.invoke([
            SystemMessage(
                content=f"You are a professional translator. Translate accurately while maintaining the original meaning and tone. For insurance and technical terms, use appropriate terminology in {target_language}."
            ),
            HumanMessage(content=translation_prompt),
        ])

        return response.content.strip()
    except Exception as e:
        # Return original text if translation fails
        print(f"[Translation] Error: {e}. Returning original text.")
        return text


def validate_response_multilingual(
    user_response: str, expected_values: list, user_language: str
) -> dict:
    """
    Validate user response against expected values, supporting multiple languages.

    Args:
        user_response: The user's response in any language
        expected_values: List of expected values in English
        user_language: The language the user is speaking

    Returns:
        dict with 'is_valid' (bool), 'matched_value' (English version), 'explanation' (str)
    """
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
        response = llm.invoke([
            SystemMessage(
                content="You are a validation expert for a multilingual insurance chatbot. Be accurate and consider language variations."
            ),
            HumanMessage(content=validation_prompt),
        ])

        result = json.loads(response.content.strip())
        return result
    except Exception as e:
        return {
            "is_valid": False,
            "matched_value": None,
            "explanation": f"Validation error: {str(e)}",
        }


def format_response_in_language(
    response_text: str,
    options: list,
    user_language: str,
    message_type: str = None,
    document_type: str = None,
) -> dict:
    """
    Format the bot's response in the user's preferred language.

    Args:
        response_text: The response text in English
        options: List of options in English (if any)
        user_language: The target language
        message_type: Optional metadata indicating message type (e.g., 'document_upload_request', 'confirmation', 'question')
        document_type: Optional metadata indicating document type (e.g., 'emirates_id_front', 'driving_license', 'mulkiya', 'emirates_id_back', 'excel')

    Returns:
        dict with 'response' and optionally 'options', 'message_type', 'document_type'
    """
    print(f"[DEBUG format_response_in_language] User language: {user_language}")
    print(f"[DEBUG format_response_in_language] Response text: {response_text}")
    print(f"[DEBUG format_response_in_language] Options: {options}")
    print(f"[DEBUG format_response_in_language] Message type: {message_type}")
    print(f"[DEBUG format_response_in_language] Document type: {document_type}")

    if user_language.lower() in ["english", "en"]:
        result = {"response": response_text}
        if options:
            result["options"] = ", ".join(options)
        if message_type:
            result["message_type"] = message_type
        if document_type:
            result["document_type"] = document_type
        # Always include language information for frontend
        result["language"] = "English"
        result["language_code"] = "en"
        print(f"[DEBUG format_response_in_language] English result: {result}")
        return result

    # Translate response
    translated_response = translate_text(response_text, user_language)
    print(
        f"[DEBUG format_response_in_language] Translated response: {translated_response}"
    )
    result = {"response": translated_response}

    # Translate options if present
    if options:
        translated_options = [translate_text(opt, user_language) for opt in options]
        result["options"] = ", ".join(translated_options)
        print(
            f"[DEBUG format_response_in_language] Translated options: {translated_options}"
        )

    # Add metadata (not translated - used for routing/logic)
    if message_type:
        result["message_type"] = message_type
    if document_type:
        result["document_type"] = document_type

    # Always include language information for frontend
    result["language"] = user_language
    result["language_code"] = get_language_code(user_language)

    print(f"[DEBUG format_response_in_language] Final result: {result}")
    return result


def translate_to_english_for_storage(text: str, detected_language: str) -> str:
    """
    Translate user's response to English for storage if not already in English.

    Args:
        text: The user's response
        detected_language: The detected language of the text

    Returns:
        Text in English
    """
    if detected_language.lower() in ["english", "en"]:
        return text

    # Translate to English
    return translate_text(text, "English", detected_language)


def detect_document_type_from_question(question_text: str) -> tuple:
    """
    Detect message type and document type from question text.

    Args:
        question_text: The question text (English)

    Returns:
        tuple: (message_type, document_type) or (None, None) if not a document question
    """
    question_lower = question_text.lower()

    # Document upload requests
    if "front page" in question_lower and (
        "document" in question_lower or "emirates" in question_lower
    ):
        return ("document_upload_request", "emirates_id_front")
    elif "back page" in question_lower and (
        "document" in question_lower or "emirates" in question_lower
    ):
        return ("document_upload_request", "emirates_id_back")
    elif "driving license" in question_lower or "driving licence" in question_lower:
        return ("document_upload_request", "driving_license")
    elif "mulkiya" in question_lower:
        return ("document_upload_request", "mulkiya")
    elif "excel" in question_lower and "upload" in question_lower:
        return ("document_upload_request", "excel")
    elif "upload" in question_lower and "document" in question_lower:
        return ("document_upload_request", "emirates_id")

    # Not a document upload question
    return (None, None)


# ==================== END MULTI-LANGUAGE SUPPORT ====================


def handle_option_validation_multilingual(
    user_message: str,
    valid_options: list,
    question: str,
    user_language: str,
    conversation_state: dict,
    questions: list,
    responses: dict,
    user_id: str,
) -> dict:
    """
    Generic handler for option-based questions with multilingual support.

    Args:
        user_message: User's response
        valid_options: List of valid options in English
        question: The current question being asked
        user_language: User's preferred language
        conversation_state: Current conversation state
        questions: List of all questions
        responses: User's responses so far
        user_id: User identifier

    Returns:
        Response dictionary
    """
    # Validate using multilingual validation
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if validation_result["is_valid"]:
        # Store the English version
        english_value = validation_result["matched_value"]
        responses[question] = english_value
        conversation_state["current_question_index"] += 1

        # Check if there are more questions
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]

            if isinstance(next_question, dict):
                next_question_text = next_question["question"]
                next_options = next_question.get("options", [])
                response_message = (
                    f"Thank you! Now, let's move on to: {next_question_text}"
                )
                # Detect document type from question
                msg_type, doc_type = detect_document_type_from_question(
                    next_question_text
                )
                return format_response_in_language(
                    response_message, next_options, user_language, msg_type, doc_type
                )
            else:
                response_message = f"Thank you! Now, let's move on to: {next_question}"
                # Detect document type from question
                msg_type, doc_type = detect_document_type_from_question(next_question)
                return format_response_in_language(
                    response_message, [], user_language, msg_type, doc_type
                )
        else:
            # Save responses
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            if user_id in user_states:
                del user_states[user_id]

            final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
            result = format_response_in_language(final_message, [], user_language)
            result["final_responses"] = responses
            return result
    else:
        # Invalid response - provide helpful error
        error_prompt = (
            f"The user said '{user_message}' but needs to choose from: {', '.join(valid_options)}. "
            "Provide a brief, helpful message explaining they need to select a valid option."
        )
        error_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly insurance assistant. Respond in {user_language}. "
                "Be brief and helpful."
            ),
            HumanMessage(content=error_prompt),
        ])

        retry_message = f"Let's try again: {question}"
        retry_translated = translate_text(retry_message, user_language)

        # Translate options for display
        translated_options = [
            translate_text(opt, user_language) for opt in valid_options
        ]

        return {
            "response": error_response.content.strip(),
            "question": retry_translated,
            "options": ", ".join(translated_options),
        }


def list_pdfs(directory="pdf"):
    """List all PDF file names (without extension) in the specified directory."""
    return [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".pdf")]


user_states = TTLCache(maxsize=1000, ttl=3600)


def load_questions(file_path="questions/questions.json"):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{file_path} not found. Please ensure the file exists."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")


# Load questions from the questions folder
questions_data = load_questions()

# Access questions in your logic
initial_questions = questions_data["initial_questions"]
medical_questions = questions_data["medical_questions"]
individual_questions = questions_data["individual_questions"]
sma_questions = questions_data["sma_questions"]
# new_policy_questions = questions_data["new_policy_questions"]
existing_policy_questions = questions_data["existing_policy_questions"]
motor_insurance_questions = questions_data["motor_insurance_questions"]
car_questions = questions_data["car_questions"]
bike_questions = questions_data["bike_questions"]
motor_claim = questions_data["motor_claim"]
greeting_templates = questions_data["greeting_templates"]


def process_user_input(user_input: UserInput):
    user_id = user_input.user_id.strip()
    user_message = user_input.message.strip()
    # Initialize user state if not already presents
    if user_id not in user_states:
        user_states[user_id] = {
            "current_question_index": 0,
            "responses": {},
            "current_flow": "initial",
            "welcome_shown": False,
            "awaiting_document_name": False,
            "document_name": "",
            "last_takaful_query_time": None,
            "awaiting_takaful_followup": False,
            "last_chronic_conditions_time": None,
            "awaiting_chronic_conditions_followup": False,
            "takaful_emarat_asked": False,
            "preferred_language": "English",  # Default language
            "language_code": "en",
            "language_explicitly_set": False,  # Track if user explicitly set language
        }

    conversation_state = user_states[user_id]
    user_name = get_user_name(user_id)

    # ==================== LANGUAGE DETECTION ====================
    # Check for explicit language requests first
    language_requests = {
        "say in arabic": {"language": "Arabic", "code": "ar"},
        "speak in arabic": {"language": "Arabic", "code": "ar"},
        "change language to arabic": {"language": "Arabic", "code": "ar"},
        "arabic": {"language": "Arabic", "code": "ar"},
        "عربي": {"language": "Arabic", "code": "ar"},
        "بالعربية": {"language": "Arabic", "code": "ar"},
        "say in hindi": {"language": "Hindi", "code": "hi"},
        "hindi": {"language": "Hindi", "code": "hi"},
        "हिंदी": {"language": "Hindi", "code": "hi"},
        "say in urdu": {"language": "Urdu", "code": "ur"},
        "urdu": {"language": "Urdu", "code": "ur"},
        "اردو": {"language": "Urdu", "code": "ur"},
        "say in english": {"language": "English", "code": "en"},
        "english": {"language": "English", "code": "en"},
    }

    # Check if this is a document upload success message (define this before the if/else block)
    is_document_upload_success = (
        "document upload successfully" in user_message.lower()
        or "upload successfully" in user_message.lower()
        or "file uploaded" in user_message.lower()
        or "document uploaded" in user_message.lower()
    )

    # Check if user is making an explicit language request
    user_message_lower = user_message.lower().strip()
    if user_message_lower in language_requests:
        requested_lang = language_requests[user_message_lower]
        conversation_state["preferred_language"] = requested_lang["language"]
        conversation_state["language_code"] = requested_lang["code"]
        conversation_state["language_explicitly_set"] = True  # Mark as explicitly set
        print(
            f"[Language Request] User {user_id} explicitly requested {requested_lang['language']}"
        )
        user_language = requested_lang["language"]
    else:
        # Only detect language for the first few messages or when user explicitly changes language
        # Don't change language for numeric inputs, short responses, or when already in a flow
        current_flow = conversation_state.get("current_flow", "initial")
        current_question_index = conversation_state.get("current_question_index", 0)

        # Check if this is a numeric input, very short response, or document upload success message
        is_numeric_or_short = (
            user_message.strip().isdigit()
            or len(user_message.strip()) <= 3
            or user_message.strip().lower() in ["yes", "no", "y", "n", "ok", "okay"]
        )

        # Only detect language if:
        # 1. Language was not explicitly set by user
        # 2. We're still in initial flow (first few questions)
        # 3. User hasn't established a language preference yet
        # 4. The input is not numeric/short
        # 5. The input is not a document upload success message (regardless of current language)
        should_detect_language = (
            not conversation_state.get("language_explicitly_set", False)
            and current_flow == "initial"
            and current_question_index < 3
            and not is_numeric_or_short
            and not is_document_upload_success  # This prevents language detection for document upload messages in ANY language
            and conversation_state.get("preferred_language", "English") == "English"
        )

        if should_detect_language:
            detected_lang = detect_language(user_message)
            print(f"[DEBUG] Detected language: {detected_lang}")

            # Update the user's preferred language if it's different
            if detected_lang["language"] != conversation_state.get(
                "preferred_language", "English"
            ):
                conversation_state["preferred_language"] = detected_lang["language"]
                conversation_state["language_code"] = detected_lang["code"]
                print(
                    f"[Language Detection] User {user_id} switched to {detected_lang['language']}"
                )
        else:
            reason = "unknown"
            if conversation_state.get("language_explicitly_set", False):
                reason = "language explicitly set by user"
            elif current_flow != "initial":
                reason = "not in initial flow"
            elif current_question_index >= 3:
                reason = "beyond initial questions"
            elif is_numeric_or_short:
                reason = "numeric or short input"
            elif is_document_upload_success:
                reason = "document upload success message"
            elif conversation_state.get("preferred_language", "English") != "English":
                reason = "language already established"

            print(
                f"[DEBUG] Skipping language detection ({reason}) - preserving current language: {conversation_state.get('preferred_language', 'English')}"
            )

        user_language = conversation_state.get("preferred_language", "English")

    print(f"[DEBUG] Final user language: {user_language}")
    # ==================== END LANGUAGE DETECTION ====================

    # Handle document upload success messages - maintain current language flow
    if is_document_upload_success:
        print(
            f"[DEBUG] Document upload success detected - maintaining current language: {user_language}"
        )
        # Continue with the current flow without changing language
        # The system will proceed to the next question in the same language

    # Handle language requests - present current question in new language
    if user_message_lower in language_requests:
        # Get current question
        current_question_index = conversation_state.get("current_question_index", 0)
        current_flow = conversation_state.get("current_flow", "initial")

        # Get the appropriate questions list based on current flow
        if current_flow == "initial":
            questions_list = initial_questions
        elif current_flow == "medical" or current_flow == "medical_insurance":
            questions_list = medical_questions
        elif current_flow == "individual":
            questions_list = individual_questions
        elif current_flow == "sma":
            questions_list = sma_questions
        elif current_flow == "motor" or current_flow == "motor_insurance":
            questions_list = motor_insurance_questions
        elif current_flow == "car" or current_flow == "car_questions":
            questions_list = car_questions
        elif current_flow == "bike" or current_flow == "bike_questions":
            questions_list = bike_questions
        elif current_flow == "existing_policy":
            questions_list = existing_policy_questions
        elif current_flow == "motor_claim":
            questions_list = motor_claim
        else:
            questions_list = initial_questions

        # Present current question in new language
        if current_question_index < len(questions_list):
            current_question = questions_list[current_question_index]
            if isinstance(current_question, dict):
                question_text = current_question["question"]
                options = current_question.get("options", [])
                return format_response_in_language(
                    question_text, options, user_language
                )
            else:
                return format_response_in_language(current_question, [], user_language)
        else:
            # No current question, show welcome
            first_question = initial_questions[0]
            if isinstance(first_question, dict):
                question_text = first_question["question"]
                options = first_question.get("options", [])
                return format_response_in_language(
                    question_text, options, user_language
                )
            else:
                return format_response_in_language(first_question, [], user_language)

    # Handle cancel command - reset everything and start fresh
    if user_message.lower() in ["cancel", "restart", "reset", "start over"]:
        # Save the language preference before resetting
        saved_language = conversation_state.get("preferred_language", "English")
        saved_language_code = conversation_state.get("language_code", "en")
        saved_language_explicitly_set = conversation_state.get(
            "language_explicitly_set", False
        )

        # Reset the entire conversation state
        user_states[user_id] = {
            "current_question_index": 0,
            "responses": {},
            "current_flow": "initial",
            "welcome_shown": True,  # Set to True to prevent duplicate greeting
            "awaiting_document_name": False,
            "document_name": "",
            "last_takaful_query_time": None,
            "awaiting_takaful_followup": False,
            "last_chronic_conditions_time": None,
            "awaiting_chronic_conditions_followup": False,
            "takaful_emarat_asked": False,
            "preferred_language": saved_language,  # Preserve language
            "language_code": saved_language_code,
            "language_explicitly_set": saved_language_explicitly_set,  # Preserve explicit setting
        }

        # Return the first initial question
        first_question = initial_questions[0]
        if isinstance(first_question, dict):
            question_text = first_question["question"]
            options = first_question.get("options", [])
            reset_message = (
                "Your conversation has been reset. Let's start fresh! " + question_text
            )

            # Format response in user's language
            return format_response_in_language(reset_message, options, saved_language)
        else:
            reset_message = (
                "Your conversation has been reset. Let's start fresh! " + first_question
            )
            return format_response_in_language(reset_message, [], saved_language)

    # Handle Takaful Emarat Silver query
    if "takaful emarat silver" in user_message.lower():
        conversation_state["last_takaful_query_time"] = datetime.now()
        conversation_state["awaiting_takaful_followup"] = True
        conversation_state["takaful_emarat_asked"] = (
            True  # Set flag to indicate Takaful Emarat was asked
        )
        # Generate a more natural welcome response using LLM
        welcome_prompt = "Rewrite this welcome message in a friendly, conversational way as if a real insurance agent is greeting a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: 'Welcome to the Takaful Emarat Silver plan! What do you need to know about the Takaful Emarat Silver plan? Please let me know, I am here to help you!'"
        welcome_response = llm.invoke([
            SystemMessage(
                content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to greet customers warmly and make them feel welcome and comfortable. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
            ),
            HumanMessage(content=welcome_prompt),
        ])
        return {
            "response": f"{welcome_response.content.strip()}",
        }

    # Handle Pre-existing & Chronic conditions query (only if Takaful Emarat Silver was asked first)
    if "pre existing & chronic conditions" in user_message.lower():
        # Check if user has already asked about Takaful Emarat Silver
        if conversation_state.get("takaful_emarat_asked", False):
            conversation_state["last_chronic_conditions_time"] = datetime.now()
            conversation_state["awaiting_chronic_conditions_followup"] = True
            conversation_state["chronic_conditions_shown"] = (
                True  # Flag to track that response was shown
            )
            # Generate a more natural response using LLM
            chronic_conditions_prompt = "Rewrite this exact information about pre-existing and chronic conditions coverage in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: This is the content(answer)'Covered only if declared in the Application Form and the terms, and additional premium to be agreed.'"
            chronic_conditions_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to explain insurance terms in a warm, conversational manner. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
                ),
                HumanMessage(content=chronic_conditions_prompt),
            ])
            return {
                "response": f"{chronic_conditions_response.content.strip()}",
                "pdf_link": "/pdf-view/",
            }
        else:
            # If Takaful Emarat Silver wasn't asked first, provide a general response
            return {
                "response": "I'd be happy to help you with information about pre-existing and chronic conditions. However, to provide you with the most accurate and specific information, could you please first ask about the Takaful Emarat Silver plan? This will help me give you the most relevant details for your situation."
            }

    # Handle Area of Coverage query (only if Takaful Emarat Silver was asked first)
    if "area of coverage" in user_message.lower():
        # Check if user has already asked about Takaful Emarat Silver
        if conversation_state.get("takaful_emarat_asked", False):
            conversation_state["awaiting_takaful_followup"] = True
            # Generate a more natural response using LLM
            coverage_prompt = "Rewrite this exact information about area of coverage in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: This is the content(answer)'Worldwide'"
            coverage_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to explain insurance terms in a warm, conversational manner. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
                ),
                HumanMessage(content=coverage_prompt),
            ])
            return {
                "response": f"{coverage_response.content.strip()}.",
                "pdf_link": "/pdf-view/",
            }
        else:
            # If Takaful Emarat Silver wasn't asked first, provide a general response
            return {
                "response": "I'd be happy to help you with information about area of coverage. However, to provide you with the most accurate and specific information, could you please first ask about the Takaful Emarat Silver plan? This will help me give you the most relevant details for your situation."
            }

    # Handle Annual Medicine Limit query (only if Takaful Emarat Silver was asked first)
    if (
        "annual medicine limit" in user_message.lower()
        or "medicine limit" in user_message.lower()
    ):
        # Check if user has already asked about Takaful Emarat Silver
        if conversation_state.get("takaful_emarat_asked", False):
            conversation_state["awaiting_takaful_followup"] = True
            # Generate a more natural response using LLMs
            medicine_prompt = "Rewrite this exact information about annual medicine limit in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: This is the content(answer)'AED 5,000'"
            medicine_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to explain insurance terms in a warm, conversational manner. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
                ),
                HumanMessage(content=medicine_prompt),
            ])
            return {
                "response": f"{medicine_response.content.strip()}.",
                "pdf_link": "/pdf-view/",
            }
        else:
            # If Takaful Emarat Silver wasn't asked first, provide a general response
            return {
                "response": "I'd be happy to help you with information about annual medicine limit. However, to provide you with the most accurate and specific information, could you please first ask about the Takaful Emarat Silver plan? This will help me give you the most relevant details for your situation."
            }

    # Handle Consultation Fee query (only if Takaful Emarat Silver was asked first)
    if "consultation fee" in user_message.lower() or "fee" in user_message.lower():
        # Check if user has already asked about Takaful Emarat Silver
        if conversation_state.get("takaful_emarat_asked", False):
            conversation_state["awaiting_takaful_followup"] = True
            # Generate a more natural response using LLM
            consultation_prompt = "Rewrite this exact information about consultation fee in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: This is the content(answer)'AED 50'"
            consultation_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to explain insurance terms in a warm, conversational manner. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
                ),
                HumanMessage(content=consultation_prompt),
            ])
            return {
                "response": f"{consultation_response.content.strip()}.",
                "pdf_link": "/pdf-view/",
            }
        else:
            # If Takaful Emarat Silver wasn't asked first, provide a general response
            return {
                "response": "I'd be happy to help you with information about consultation fee. However, to provide you with the most accurate and specific information, could you please first ask about the Takaful Emarat Silver plan? This will help me give you the most relevant details for your situation."
            }

    # Handle Network query (only if Takaful Emarat Silver was asked first)
    if "network" in user_message.lower():
        # Check if user has already asked about Takaful Emarat Silver
        if conversation_state.get("takaful_emarat_asked", False):
            conversation_state["awaiting_takaful_followup"] = True
            # Generate a more natural response using LLM
            network_prompt = "Rewrite this exact information about network in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: This is the content(answer)'Nextcare'"
            network_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to explain insurance terms in a warm, conversational manner. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
                ),
                HumanMessage(content=network_prompt),
            ])
            return {
                "response": f"{network_response.content.strip()}.",
                "pdf_link": "/pdf-view/",
            }
        else:
            # If Takaful Emarat Silver wasn't asked first, provide a general response
            return {
                "response": "I'd be happy to help you with information about the network. However, to provide you with the most accurate and specific information, could you please first ask about the Takaful Emarat Silver plan? This will help me give you the most relevant details for your situation."
            }

    # Handle Dental Treatment query (only if Takaful Emarat Silver was asked first)
    if "dental treatment" in user_message.lower() or "dental" in user_message.lower():
        # Check if user has already asked about Takaful Emarat Silver
        if conversation_state.get("takaful_emarat_asked", False):
            conversation_state["awaiting_takaful_followup"] = True
            # Generate a more natural response using LLM
            dental_prompt = "Rewrite this exact information about dental treatment coverage in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: This is the content(answer)'Routine Dental is not covered. Cover only Emergency, injury cases & surgeries.'"
            dental_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to explain insurance terms in a warm, conversational manner. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
                ),
                HumanMessage(content=dental_prompt),
            ])
            return {
                "response": f"{dental_response.content.strip()}.",
                "pdf_link": "/pdf-view/",
            }
        else:
            # If Takaful Emarat Silver wasn't asked first, provide a general response
            return {
                "response": "I'd be happy to help you with information about dental treatment coverage. However, to provide you with the most accurate and specific information, could you please first ask about the Takaful Emarat Silver plan? This will help me give you the most relevant details for your situation."
            }

    # Handle Direct Access to Hospital query (only if Takaful Emarat Silver was asked first)
    if (
        "direct access" in user_message.lower()
        or "hospital access" in user_message.lower()
    ):
        # Check if user has already asked about Takaful Emarat Silver
        if conversation_state.get("takaful_emarat_asked", False):
            conversation_state["awaiting_takaful_followup"] = True
            # Generate a more natural response using LLM
            access_prompt = "Rewrite this exact information about direct access to hospital in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: This is the content(answer)'Yes'"
            access_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to explain insurance terms in a warm, conversational manner. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
                ),
                HumanMessage(content=access_prompt),
            ])
            return {
                "response": f"{access_response.content.strip()}.",
                "pdf_link": "/pdf-view/",
            }
        else:
            # If Takaful Emarat Silver wasn't asked first, provide a general response
            return {
                "response": "I'd be happy to help you with information about direct access to hospital. However, to provide you with the most accurate and specific information, could you please first ask about the Takaful Emarat Silver plan? This will help me give you the most relevant details for your situation."
            }

    # Check for Takaful questions follow-up (show options after response)
    if (
        conversation_state.get("awaiting_takaful_followup")
        and not user_message.lower() in ["yes", "no"]
        and any(
            keyword in user_message.lower()
            for keyword in [
                "area of coverage",
                "annual medicine limit",
                "medicine limit",
                "consultation fee",
                "fee",
                "network",
                "dental treatment",
                "dental",
                "direct access",
                "hospital access",
            ]
        )
    ):
        # This means user just got a Takaful question response, show follow-up options
        return {
            "response": "Is there anything else you want me to help you with related to Takaful Emarat Silver?",
            "options": "Yes, No",
        }

    # Handle Yes/No response for Chronic conditions follow-up
    if conversation_state.get(
        "awaiting_chronic_conditions_followup"
    ) and user_message.lower() in ["yes", "no"]:
        if user_message.lower() == "yes":
            conversation_state["awaiting_chronic_conditions_followup"] = False
            conversation_state["awaiting_takaful_followup"] = True
            conversation_state["chronic_conditions_shown"] = False  # Reset the flag
            return {
                "response": "Great! Please ask your question about the Takaful Emarat Silver plan, and I'll assist you."
            }
        else:
            conversation_state["awaiting_chronic_conditions_followup"] = False
            conversation_state["current_flow"] = "initial"
            conversation_state["current_question_index"] = 0
            conversation_state["chronic_conditions_shown"] = False  # Reset the flag
            first_question = initial_questions[0]
            next_options = first_question.get("options", [])
            response_message = (
                f"Alright, let's go back to the main menu. {first_question['question']}"
            )

            # Translate to user's language
            return format_response_in_language(
                response_message, next_options, user_language
            )

    # Check for Chronic conditions follow-up (show options after response)
    if (
        conversation_state.get("awaiting_chronic_conditions_followup")
        and conversation_state.get("chronic_conditions_shown", False)
        and not user_message.lower() in ["yes", "no"]
    ):
        # This means user just got the chronic conditions response, show follow-up options
        conversation_state["chronic_conditions_shown"] = False  # Reset the flag
        return {
            "response": "Is there anything else you want me to help you with?",
            "options": "Yes, No",
        }

    # Handle Yes/No response for Takaful follow-up
    if conversation_state.get("awaiting_takaful_followup") and user_message.lower() in [
        "yes",
        "no",
    ]:
        if user_message.lower() == "yes":
            conversation_state["awaiting_takaful_followup"] = True
            return {
                "response": "Is there anything else Please ask, I am here to help you. realted to Takaful Emarat Silver",
            }
        else:
            conversation_state["awaiting_takaful_followup"] = False
            conversation_state["current_flow"] = "initial"
            conversation_state["current_question_index"] = 0
            first_question = initial_questions[0]
            next_options = first_question.get("options", [])
            response_message = f"Alright, let's move on. {first_question['question']}"

            # Translate to user's language
            return format_response_in_language(
                response_message, next_options, user_language
            )

    # Check for Takaful follow-up (handle questions after initial response)
    if conversation_state.get(
        "awaiting_takaful_followup"
    ) and not user_message.lower() in ["yes", "no"]:
        # User is asking a question after the initial Takaful response
        # Generate a more natural response using LLM
        takaful_prompt = "Rewrite this response in a friendly, conversational way as if a real insurance agent is speaking to a customer. Keep the same content but make it sound natural and warm. Use only 1-3 lines maximum: 'I'm here to help with any questions about the Takaful Emarat Silver plan. Please ask your specific question and I'll provide you with the most accurate information.'"
        takaful_response = llm.invoke([
            SystemMessage(
                content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to communicate in a warm, conversational manner that makes customers feel comfortable and well-cared for. Keep responses short (1-3 lines) and maintain the exact same information while making it sound natural."
            ),
            HumanMessage(content=takaful_prompt),
        ])
        return {
            "response": f"{takaful_response.content.strip()} Do you need to know anything else related Takaful Emarat Silver plan?",
            "options": "Yes, No",
        }

    # Show welcome message with the first question if not already shown
    if not conversation_state["welcome_shown"]:
        conversation_state["welcome_shown"] = True
        first_question = initial_questions[0]
        next_options = first_question.get("options", [])
        greeting = choice(greeting_templates).format(
            user_name=user_name, first_question=first_question["question"]
        )

        # Translate greeting and options to user's language
        return format_response_in_language(greeting, next_options, user_language)

    # Determine the current flow and questions
    current_flow = conversation_state["current_flow"]
    if current_flow == "initial":
        questions = initial_questions
    elif current_flow == "medical_insurance":
        questions = medical_questions
    elif current_flow == "individual":
        questions = individual_questions
    elif current_flow == "sma":
        questions = sma_questions
    elif current_flow == "motor_insurance":
        questions = motor_insurance_questions
    elif current_flow == "car_questions":
        questions = car_questions
    elif current_flow == "bike_questions":
        questions = bike_questions
    # elif current_flow == "new_policy":
    #     questions = new_policy_questions
    elif current_flow == "existing_policy":
        questions = existing_policy_questions
    elif current_flow == "motor_claim":
        questions = motor_claim
    else:
        questions = []

    # Get current question index
    current_index = conversation_state["current_question_index"]
    responses = conversation_state["responses"]

    if current_index < len(questions):
        # Current question
        question_data = questions[current_index]
        if isinstance(question_data, dict):
            question = question_data["question"]
            options = question_data.get("options", [])
        else:
            question = question_data
            options = []

        # Handle options
        if options:
            # Validate user response against options using multilingual validation
            validation_result = validate_response_multilingual(
                user_message, options, user_language
            )

            if validation_result["is_valid"]:
                # Store the English version
                matched_option = validation_result["matched_value"]
                responses[question] = matched_option

                if matched_option == "Purchase a Medical Insurance":
                    conversation_state["current_flow"] = "medical_insurance"
                    conversation_state["current_question_index"] = 0

                    # Check if medical_questions is a list of dictionaries
                    if isinstance(medical_questions[0], dict):
                        next_options = medical_questions[0].get("options", [])
                        response_message = (
                            f"Great choice! {medical_questions[0]['question']}"
                        )
                        # Translate to user's language
                        return format_response_in_language(
                            response_message, next_options, user_language
                        )
                    # If medical_questions is a list of strings
                    else:
                        response_message = f"Great choice! {medical_questions[0]}"
                        return format_response_in_language(
                            response_message, [], user_language
                        )
                elif matched_option == "Purchase a Motor Insurance":
                    conversation_state["current_flow"] = "motor_insurance"
                    conversation_state["current_question_index"] = 0
                    next_options = motor_insurance_questions[0].get("options", [])
                    response_message = (
                        f"Great choice! {motor_insurance_questions[0]['question']}"
                    )

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )
                elif matched_option == "Purchase a Car Insurance":
                    conversation_state["current_flow"] = "car_questions"
                    conversation_state["current_question_index"] = 0
                    # Check if car_questions is a list of dictionaries
                    if isinstance(car_questions[0], dict):
                        next_options = car_questions[0].get("options", [])
                        response_message = (
                            f"Great choice! {car_questions[0]['question']}"
                        )
                    else:
                        next_options = []
                        response_message = f"Great choice! {car_questions[0]}"

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )

                elif matched_option == "Purchase a Bike Insurance":
                    conversation_state["current_flow"] = "bike_questions"
                    conversation_state["current_question_index"] = 0
                    # Check if bike_questions is a list of dictionaries
                    if isinstance(bike_questions[0], dict):
                        next_options = bike_questions[0].get("options", [])
                        response_message = (
                            f"Great choice! {bike_questions[0]['question']}"
                        )
                    else:
                        next_options = []
                        response_message = f"Great choice! {bike_questions[0]}"

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )
                # elif matched_option == "Purchase a new policy":
                #     conversation_state["current_flow"] = "new_policy"
                #     conversation_state["current_question_index"] = 0
                #     return {
                #         "response": f"Great choice! {new_policy_questions[0]}"
                #     }
                elif matched_option == "Renew my existing policy":
                    conversation_state["current_flow"] = "existing_policy"
                    conversation_state["current_question_index"] = 0
                    # Check if existing_policy_questions is a list of dictionaries
                    if isinstance(existing_policy_questions[0], dict):
                        next_options = existing_policy_questions[0].get("options", [])
                        response_message = (
                            f"Great choice! {existing_policy_questions[0]['question']}"
                        )
                    else:
                        next_options = []
                        response_message = (
                            f"Great choice! {existing_policy_questions[0]}"
                        )

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )
                elif matched_option == "Claim a Motor Insurance":
                    conversation_state["current_flow"] = "motor_claim"
                    conversation_state["current_question_index"] = 0
                    # Get the first motor_claim question
                    if isinstance(motor_claim[0], dict):
                        next_options = motor_claim[0].get("options", [])
                        response_message = f"Great choice! {motor_claim[0]['question']}"
                    else:
                        next_options = []
                        response_message = f"Great choice! {motor_claim[0]}"

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )
            else:
                # Invalid option - provide helpful error in user's language
                error_prompt = f"The user said '{user_message}' but needs to choose from: {', '.join(options)}. Provide a brief, helpful message."
                error_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly insurance assistant. Respond in {user_language}. Be brief and helpful."
                    ),
                    HumanMessage(content=error_prompt),
                ])

                retry_message = f"Let's try again: {question}"
                retry_translated = translate_text(retry_message, user_language)
                translated_options = [
                    translate_text(opt, user_language) for opt in options
                ]

                return {
                    "response": error_response.content.strip(),
                    "question": retry_translated,
                    "options": ", ".join(translated_options),
                }

        if question in [
            "Let's start with your Medical insurance details. Choose your Visa issued Emirate?",
            "Tell me your Emirate sponsor located in?",
        ]:
            # Use multilingual validation for emirate selection
            valid_options = [
                "Abudhabi",
                "Ajman",
                "Dubai",
                "Fujairah",
                "Ras Al Khaimah",
                "Sharjah",
                "Umm Al Quwain",
            ]
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif "emaf" in user_message.lower() or "from" in user_message.lower():
            return handle_emaf_document(
                question, user_message, responses, conversation_state, questions
            )
        elif "post a review" in user_message.lower():
            user_message = user_message.lower()  # Convert user_message to lowercase
            if "post a review" in user_message:
                return {
                    "review_message": "If you are satisfied with Wehbe(Broker) services, please leave a review for sharing happiness to others!!😊",
                    "review_link": "https://www.google.com/search?client=ms-android-samsung-ss&sca_esv=4eb717e6f42bf628&sxsrf=AHTn8zprabdPVFL3C2gXo4guY8besI3jqQ:1744004771562&q=wehbe+insurance+services+llc+reviews&uds=ABqPDvy-z0dcsfm2PY76_gjn-YWou9-AAVQ4iWjuLR6vmDV0vf3KpBMNjU5ZkaHGmSY0wBrWI3xO9O55WuDmXbDq6a3SqlwKf2NJ5xQAjebIw44UNEU3t4CpFvpLt9qFPlVh2F8Gfv8sMuXXSo2Qq0M_ZzbXbg2c323G_bE4tVi7Ue7d_sW0CrnycpJ1CvV-OyrWryZw_TeQ3gLGDgzUuHD04MpSHquYZaSQ0_mIHLWjnu7fu8c7nb6_aGDb_H1Q-86fD2VmWluYA5jxRkC9U2NsSwSSXV4FPW9w1Q2T_Wjt6koJvLgtikd66MqwYiJPX2x9MwLhoGYlpTbKtkJuHwE9eM6wQgieChskow6tJCVjQ75I315dT8n3tUtasGdBkprOlUK9ibPrYr9HqRz4AwzEQaxAq9_EDcsSG_XW0CHuqi2lRKHw592MlGlhjyQibXKSZJh-v3KW4wIVqa-2x0k1wfbZdpaO3BZaKYCacLOxwUKTnXPbQqDPLQDeYgDBwaTLvaCN221H&si=APYL9bvoDGWmsM6h2lfKzIb8LfQg_oNQyUOQgna9TyfQHAoqUvvaXjJhb-NHEJtDKiWdK3OqRhtZNP2EtNq6veOxTLUq88TEa2J8JiXE33-xY1b8ohiuDLBeOOGhuI1U6V4mDc9jmZkDoxLC9b6s6V8MAjPhY-EC_g%3D%3D&sa=X&sqi=2&ved=2ahUKEwi05JSHnMWMAxUw8bsIHRRCDd0Qk8gLegQIHxAB&ictx=1&stq=1&cs=0&lei=o2bzZ_SGIrDi7_UPlIS16A0#ebo=1",
                }
            else:
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                # Safely access the next question
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if isinstance(next_question, dict) and "options" in next_question:
                        options = ", ".join(next_question["options"])
                        return {
                            "response": f"{general_assistant_response.content.strip()}",
                            "question": f"Let's try again: {next_question['question']}",
                            "options": options,
                        }
                    else:
                        question_text = (
                            next_question["question"]
                            if isinstance(next_question, dict)
                            else next_question
                        )
                        return {
                            "response": f"{general_assistant_response.content.strip()}",
                            "question": f"Let's try again: {question_text}",
                        }
                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": "It seems we have reached the end of the questions.",
                    }
        # elif "post a review" in user_message.lower():
        #     user_message = user_message.lower()  # Convert user_message to lowercase
        #     next_question = questions[conversation_state["current_question_index"]]
        #     if "post a review" in user_message:
        #         if "options" in next_question:
        #             options = ", ".join(next_question["options"])
        #             return {
        #                 return {
        #             "review_message": "If you are satisfied with Wehbe(Broker) services, please leave a review for sharing happiness to others!!😊",
        #             "review_link": "https://www.google.com/search?client=ms-android-samsung-ss&sca_esv=4eb717e6f42bf628&sxsrf=AHTn8zprabdPVFL3C2gXo4guY8besI3jqQ:1744004771562&q=wehbe+insurance+services+llc+reviews&uds=ABqPDvy-z0dcsfm2PY76_gjn-YWou9-AAVQ4iWjuLR6vmDV0vf3KpBMNjU5ZkaHGmSY0wBrWI3xO9O55WuDmXbDq6a3SqlwKf2NJ5xQAjebIw44UNEU3t4CpFvpLt9qFPlVh2F8Gfv8sMuXXSo2Qq0M_ZzbXbg2c323G_bE4tVi7Ue7d_sW0CrnycpJ1CvV-OyrWryZw_TeQ3gLGDgzUuHD04MpSHquYZaSQ0_mIHLWjnu7fu8c7nb6_aGDb_H1Q-86fD2VmWluYA5jxRkC9U2NsSwSSXV4FPW9w1Q2T_Wjt6koJvLgtikd66MqwYiJPX2x9MwLhoGYlpTbKtkJuHwE9eM6wQgieChskow6tJCVjQ75I315dT8n3tUtasGdBkprOlUK9ibPrYr9HqRz4AwzEQaxAq9_EDcsSG_XW0CHuqi2lRKHw592MlGlhjyQibXKSZJh-v3KW4wIVqa-2x0k1wfbZdpaO3BZaKYCacLOxwUKTnXPbQqDPLQDeYgDBwaTLvaCN221H&si=APYL9bvoDGWmsM6h2lfKzIb8LfQg_oNQyUOQgna9TyfQHAoqUvvaXjJhb-NHEJtDKiWdK3OqRhtZNP2EtNq6veOxTLUq88TEa2J8JiXE33-xY1b8ohiuDLBeOOGhuI1U6V4mDc9jmZkDoxLC9b6s6V8MAjPhY-EC_g%3D%3D&sa=X&sqi=2&ved=2ahUKEwi05JSHnMWMAxUw8bsIHRRCDd0Qk8gLegQIHxAB&ictx=1&stq=1&cs=0&lei=o2bzZ_SGIrDi7_UPlIS16A0#ebo=1"
        #         }
        #                 "question": f"Let's Move Back {question}",
        #                 "options": options
        #             }

        #         else:
        #                 return {
        #                 return {
        #             "review_message": "If you are satisfied with Wehbe(Broker) services, please leave a review for sharing happiness to others!!😊",
        #             "review_link": "https://www.google.com/search?client=ms-android-samsung-ss&sca_esv=4eb717e6f42bf628&sxsrf=AHTn8zprabdPVFL3C2gXo4guY8besI3jqQ:1744004771562&q=wehbe+insurance+services+llc+reviews&uds=ABqPDvy-z0dcsfm2PY76_gjn-YWou9-AAVQ4iWjuLR6vmDV0vf3KpBMNjU5ZkaHGmSY0wBrWI3xO9O55WuDmXbDq6a3SqlwKf2NJ5xQAjebIw44UNEU3t4CpFvpLt9qFPlVh2F8Gfv8sMuXXSo2Qq0M_ZzbXbg2c323G_bE4tVi7Ue7d_sW0CrnycpJ1CvV-OyrWryZw_TeQ3gLGDgzUuHD04MpSHquYZaSQ0_mIHLWjnu7fu8c7nb6_aGDb_H1Q-86fD2VmWluYA5jxRkC9U2NsSwSSXV4FPW9w1Q2T_Wjt6koJvLgtikd66MqwYiJPX2x9MwLhoGYlpTbKtkJuHwE9eM6wQgieChskow6tJCVjQ75I315dT8n3tUtasGdBkprOlUK9ibPrYr9HqRz4AwzEQaxAq9_EDcsSG_XW0CHuqi2lRKHw592MlGlhjyQibXKSZJh-v3KW4wIVqa-2x0k1wfbZdpaO3BZaKYCacLOxwUKTnXPbQqDPLQDeYgDBwaTLvaCN221H&si=APYL9bvoDGWmsM6h2lfKzIb8LfQg_oNQyUOQgna9TyfQHAoqUvvaXjJhb-NHEJtDKiWdK3OqRhtZNP2EtNq6veOxTLUq88TEa2J8JiXE33-xY1b8ohiuDLBeOOGhuI1U6V4mDc9jmZkDoxLC9b6s6V8MAjPhY-EC_g%3D%3D&sa=X&sqi=2&ved=2ahUKEwi05JSHnMWMAxUw8bsIHRRCDd0Qk8gLegQIHxAB&ictx=1&stq=1&cs=0&lei=o2bzZ_SGIrDi7_UPlIS16A0#ebo=1"
        #         }
        #                 "question": f"Let's Move Back {question}",
        #                 }

        #     else:
        #         general_assistant_prompt = f"user response: {user_message}. Please assist."
        #         general_assistant_response = llm.invoke([
        #             SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),
        #             HumanMessage(content=general_assistant_prompt)
        #         ])

        #         # Safely access the next question
        #         if conversation_state["current_question_index"] < len(questions):
        #             next_question = questions[conversation_state["current_question_index"]]
        #             if isinstance(next_question, dict) and "options" in next_question:
        #                 options = ", ".join(next_question["options"])
        #                 return {
        #                     "response": f"{general_assistant_response.content.strip()}",
        #                     "question": f"Let's try again: {next_question['question']}",
        #                     "options": options
        #                 }
        #             else:
        #                 question_text = next_question["question"] if isinstance(next_question, dict) else next_question
        #                 return {
        #                     "response": f"{general_assistant_response.content.strip()}",
        #                     "question": f"Let's try again: {question_text}"
        #                 }
        #         else:
        #             return {
        #                 "response": f"{general_assistant_response.content.strip()}",
        #                 "question": "It seems we have reached the end of the questions."
        #             }

        elif question == "Please Enter Your PassKey":
            responses[question] = user_message

            if user_message == "5514":
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    if "options" in next_question:
                        options = next_question["options"]
                        next_question_text = next_question["question"]
                        response_message = f"Great choice! {next_question_text}"
                        return format_response_in_language(
                            response_message, options, user_language
                        )
                    else:
                        next_question_text = (
                            next_question["question"]
                            if isinstance(next_question, dict)
                            else next_question
                        )
                        response_message = f"Great choice! {next_question_text}"
                        return format_response_in_language(
                            response_message, [], user_language
                        )
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                    result = format_response_in_language(
                        final_message, [], user_language
                    )
                    result["final_responses"] = responses
                    return result
            else:
                error_message = (
                    "Incorrect passkey. Please try again. Please Enter Your PassKey"
                )
                return format_response_in_language(error_message, [], user_language)

        elif question == "May I know your name, please?":
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                next_questions = next_question["question"]
                return {
                    "response": f"Thanks a lot for providing your name! Alright, moving on {next_questions}"
                }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        elif question == "May I kindly ask for your phone number, please?":
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    next_questions = next_question["question"]
                    return {
                        "response": f"Thank you so much! I'd really appreciate it {next_questions}",
                        "dropdown": options,
                    }
                else:
                    return {
                        "response": f"Thank you so much! I'd really appreciate it {next_question}"
                    }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        # Handled Emaf
        elif (
            question
            == "Could you kindly confirm the name of your insurance company, please?"
        ):
            # responses[question] =  user_message
            # conversation_state["current_question_index"]+=1

            valid_options = [
                "Takaful Emarat (Ecare)",
                "National Life & General Insurance (Innayah)",
                "Takaful Emarat (Aafiya)",
                "National Life & General Insurance (NAS)",
                "Orient UNB Takaful (Nextcare)",
                "Orient Mednet (Mednet)",
                "Al Sagr Insurance (Nextcare)",
                "RAK Insurance (Mednet)",
                "Dubai Insurance (Dubai Care)",
                "Fidelity United (Nextcare)",
                "Salama April International (Salama)",
                "Sukoon (Sukoon)",
                "Orient basic",
                "Daman",
                "Dubai insurance(Mednet)",
                "Takaful Emarat(NAS)",
                "Takaful emarat(Nextcare)",
            ]
            company_number_mapping = {
                "Takaful Emarat (Ecare)": 1,
                "National Life & General Insurance (Innayah)": 2,
                "Takaful Emarat (Aafiya)": 3,
                "National Life & General Insurance (NAS)": 4,
                "Orient UNB Takaful (Nextcare)": 6,
                "Orient Mednet (Mednet)": 7,
                "Al Sagr Insurance (Nextcare)": 8,
                "RAK Insurance (Mednet)": 9,
                "Dubai Insurance (Dubai Care)": 10,
                "Fidelity United (Nextcare)": 11,
                "Salama April International (Salama)": 12,
                "Sukoon (Sukoon)": 13,
                "Orient basic": 14,
                "Daman": 15,
                "Dubai insurance(Mednet)": 16,
                "Takaful Emarat(NAS)": 17,
                "Takaful emarat(Nextcare)": 18,
            }

            if user_message in valid_options:
                # Update the Response
                responses[question] = user_message
                selected_company_number = company_number_mapping.get(user_message)
                responses["emaf_company_id"] = selected_company_number
                conversation_state["current_question_index"] += 1
                emaf_id = emaf_document(responses)

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    del user_states[user_id]
                    return {
                        "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}"
                    }
                else:
                    if isinstance(emaf_id, int):
                        del user_states[user_id]
                        return {
                            "response": f"Thank you for sharing the details. Please find the link below to view your emaf document:",
                            "link": f"https://www.insuranceclub.ae/medical_form/view/{emaf_id}",
                        }
                    else:
                        return {
                            "response": "Thank you for sharing the details. If you have any questions, please contact support@insuranceclub.ae."
                        }
            else:
                general_assistant_prompt = (
                    f"The user entered '{user_message}', . Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        "options": options,
                    }

                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                    }

        elif question == "What type of plan are you looking for?":
            # Use multilingual validation for plan type selection
            valid_options = [
                "Basic Plan",
                "Enhanced Plan",
                "Enhanced Plan Standalone",
                "Flexi Plan",
            ]
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif question == "Please Upload Your Document":
            try:
                # Try to parse as JSON first
                document_data = json.loads(user_message)
                print(f"Parsed document data: {document_data}")

                # Store the document data
                if isinstance(document_data, dict):
                    responses[question] = document_data

                # Initialize flags if they don't exist
                if "back_page_received" not in responses:
                    responses["back_page_received"] = False
                if "front_page_received" not in responses:
                    responses["front_page_received"] = False

                back_page_question = {
                    "question": "Please Upload Back Page of Your Document"
                }
                front_page_question = {
                    "question": "Please Upload Front Page of Your Document"
                }

                # Check if card_number is present - indicates back page information
                if "card_number" in document_data and document_data.get("card_number"):
                    # Mark that we've received the back page information
                    responses["back_page_received"] = True
                    responses["Card Number"] = document_data.get("card_number")

                # Check if date_of_birth and name are present - indicates front page information
                if (
                    "date_of_birth" in document_data
                    and document_data.get("date_of_birth")
                    and "name" in document_data
                    and document_data.get("name")
                ):
                    # Mark that we've received the front page information
                    responses["front_page_received"] = True

                    # Store these important details in the main responses
                    responses[
                        "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                    ] = document_data.get("name")
                    responses["Date of Birth (DOB)"] = document_data.get(
                        "date_of_birth"
                    )
                    if "gender" in document_data and document_data.get("gender"):
                        responses["Please confirm this gender of"] = document_data.get(
                            "gender"
                        )

                # First document upload storage for reference
                if "first_document_upload" not in responses:
                    responses["first_document_upload"] = document_data

                # Determine if we need to ask for additional pages
                if not responses["back_page_received"]:
                    # Need to ask for back page
                    if back_page_question not in questions:
                        questions.insert(
                            conversation_state["current_question_index"] + 1,
                            back_page_question,
                        )
                        responses[back_page_question["question"]] = None

                    response_message = (
                        "We couldn't detect the Back Page in the uploaded document"
                    )
                    result = format_response_in_language(
                        response_message,
                        [],
                        user_language,
                        message_type="document_upload_request",
                        document_type="emirates_id_back",
                    )
                    result["question"] = translate_text(
                        back_page_question["question"], user_language
                    )
                    return result
                elif not responses["front_page_received"]:
                    # Need to ask for front page
                    if front_page_question not in questions:
                        questions.insert(
                            conversation_state["current_question_index"] + 1,
                            front_page_question,
                        )
                        responses[front_page_question["question"]] = None

                    response_message = (
                        "We couldn't detect the Front Page in the uploaded document"
                    )
                    result = format_response_in_language(
                        response_message,
                        [],
                        user_language,
                        message_type="document_upload_request",
                        document_type="emirates_id_front",
                    )
                    result["question"] = translate_text(
                        front_page_question["question"], user_language
                    )
                    return result

                # If both pages have been received, continue with normal flow
                conversation_state["current_question_index"] += 1

                # Remove the page questions if they exist in the question list
                for q in [back_page_question, front_page_question]:
                    if q in questions:
                        questions.remove(q)

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if isinstance(next_question, dict) and "options" in next_question:
                        options = next_question["options"]
                        next_question_text = next_question["question"]
                        response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                        # Detect document type from next question
                        msg_type, doc_type = detect_document_type_from_question(
                            next_question_text
                        )
                        return format_response_in_language(
                            response_message, options, user_language, msg_type, doc_type
                        )
                    else:
                        next_question_text = (
                            next_question["question"]
                            if isinstance(next_question, dict)
                            else next_question
                        )
                        response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                        # Detect document type from next question
                        msg_type, doc_type = detect_document_type_from_question(
                            next_question_text
                        )
                        return format_response_in_language(
                            response_message, [], user_language, msg_type, doc_type
                        )
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                    result = format_response_in_language(
                        final_message, [], user_language
                    )
                    result["final_responses"] = responses
                    return result

            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )
                return {
                    "response": f"{general_assistant_response.content.strip()} \n\n",
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )
                return {
                    "response": f"{general_assistant_response.content.strip()} \n\n",
                    "example": example_message,
                    "question": retry_question,
                }
        elif question == "Please Upload Back Page of Your Document":
            try:
                document_data = json.loads(user_message)
                responses["Card Number"] = document_data.get("card_number")

                print(user_message)
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    print(document_data)
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = next_question["question"]
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = (
                                next_question["question"]
                                if isinstance(next_question, dict)
                                else next_question
                            )
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
        elif question == "Please Upload Front Page of Your Document":
            try:
                document_data = json.loads(user_message)
                responses[
                    "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                ] = document_data.get("name")
                responses["Date of Birth (DOB)"] = document_data.get("date_of_birth")
                responses["Please confirm this gender of"] = document_data.get("gender")

                print(user_message)
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    print(document_data)
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = next_question["question"]
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = (
                                next_question["question"]
                                if isinstance(next_question, dict)
                                else next_question
                            )
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
        elif question == "Please Upload Your Driving license":
            try:
                document_data = json.loads(user_message)
                responses["driving license Name in the License"] = document_data.get(
                    "name"
                )
                responses["Date of Birth (DOB) in the License"] = document_data.get(
                    "date_of_birth"
                )
                responses["License No in the License"] = document_data.get("license_no")
                responses["Nationality in the License"] = document_data.get(
                    "nationality"
                )
                responses["Issue Date in the License"] = document_data.get("issue_date")
                responses["Expiry Date in the License"] = document_data.get(
                    "expiry_date"
                )
                responses["Place Of Issue in the License"] = document_data.get(
                    "place_of_issue"
                )

                print(user_message)
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    print(document_data)
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = next_question["question"]
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = (
                                next_question["question"]
                                if isinstance(next_question, dict)
                                else next_question
                            )
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                            del user_states[user_id]
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }

        elif question == "Please Upload Mulkiya":
            try:
                document_data = json.loads(user_message)
                responses["Owner in the Vehicle Mulkiya"] = document_data.get("owner")
                responses["Place of Issues in the Vehicle License"] = document_data.get(
                    "place_of_issue"
                )
                responses["Traffic Plate No in the Vehicle License"] = (
                    document_data.get("traffic_plate_no")
                )
                responses["T.C.NO in the Vehicle Mulkiya"] = document_data.get(
                    "nationality"
                )
                responses["Nationality in the Vehicle Mulkiya"] = document_data.get(
                    "nationality"
                )
                responses["Expiry Date in the Vehicle Mulkiya"] = document_data.get(
                    "expiry_date"
                )
                responses["Registertion Date in the Vehicle Mulkiya"] = (
                    document_data.get("reg_date")
                )
                responses["Issues Date in the Vehicle Mulkiya"] = document_data.get(
                    "ins_exp"
                )
                responses["Policy No in the Vehicle Mulkiya"] = document_data.get(
                    "policy_no"
                )
                responses["Model in the Vehicle Mulkiya"] = document_data.get(
                    "model_no"
                )
                responses["Origin in the Vehicle Mulkiya"] = document_data.get("origin")
                responses["Vehicle Type in the Vehicle Mulkiya"] = document_data.get(
                    "vehicle_type"
                )
                responses["Num of pass in the Vehicle Mulkiya"] = document_data.get(
                    "number_of_pass"
                )
                responses["G V M in the Vehicle Mulkiya"] = document_data.get("gvw")
                responses["Empty Weight in the Vehicle Mulkiya"] = document_data.get(
                    "empty_weight"
                )
                responses["Engine Number in the Vehicle Mulkiya"] = document_data.get(
                    "engine_no"
                )
                responses["Chassis Number in the Vehicle Mulkiya"] = document_data.get(
                    "chassis_no"
                )

                print(user_message)
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    print(document_data)
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        if "options" in next_question:
                            options = next_question["options"]
                            next_question_text = next_question["question"]
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message,
                                options,
                                user_language,
                                msg_type,
                                doc_type,
                            )
                        else:
                            next_question_text = (
                                next_question["question"]
                                if isinstance(next_question, dict)
                                else next_question
                            )
                            response_message = f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                            # Detect document type from next question
                            msg_type, doc_type = detect_document_type_from_question(
                                next_question_text
                            )
                            return format_response_in_language(
                                response_message, [], user_language, msg_type, doc_type
                            )
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                            del user_states[user_id]
                        final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure that the document is in JPEG format.", user_language
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }
            except ValueError as e:
                general_assistant_prompt = f"user response: {user_message}. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )
                example_message = translate_text(
                    "Please ensure the document is in the correct format and try uploading again.",
                    user_language,
                )

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": example_message,
                    "question": retry_question,
                }

        elif question in ["Please confirm this gender of"]:
            # Use multilingual validation for gender selection
            valid_options = ["Male", "Female"]
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif (
            question
            == "Next, we need the details of the member. Would you like to upload their Emirates ID or manually enter the information?"
        ):
            # Use the Emirates ID upload handler with multilingual support
            return handle_emirate_upload_document(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )
        elif (
            question
            == "Next, we need the details of the car owner. Would you like to upload their Emirates ID or manually enter the information?"
        ):
            # Use multilingual validation for Yes/No questions
            valid_options = ["Yes", "No"]
            return handle_emirate_upload_document_car_insurance(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif question == "You Wish to Buy":
            valid_options = ["Comprehensive (Full Cover)", "Third Party"]

            # Use multilingual validation
            validation_result = validate_response_multilingual(
                user_message, valid_options, user_language
            )

            if validation_result["is_valid"]:
                # Store the English version of the response
                english_value = validation_result["matched_value"]
                responses[question] = english_value
                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if "options" in next_question:
                        next_questions_text = next_question["question"]
                        next_options = next_question["options"]
                        response_message = (
                            f"Thank you! Now, let's move on to: {next_questions_text}"
                        )

                        # Format in user's language
                        return format_response_in_language(
                            response_message, next_options, user_language
                        )
                    else:
                        next_question_text = (
                            next_question
                            if isinstance(next_question, str)
                            else next_question.get("question", "")
                        )
                        response_message = (
                            f"Thank you. Now, let's move on to: {next_question_text}"
                        )
                        return format_response_in_language(
                            response_message, [], user_language
                        )
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    del user_states[user_id]

                    final_message = "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae."

                    result = format_response_in_language(
                        final_message, [], user_language
                    )
                    result["final_responses"] = responses
                    return result
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist in a helpful manner. "
                    f"Explain that they need to choose from: {', '.join(valid_options)}"
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly insurance assistant. Respond in {user_language}. "
                        "Help the user understand they need to select a valid option."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                error_message = general_assistant_response.content.strip()
                retry_question = translate_text(
                    f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                    user_language,
                )

                return {
                    "response": error_message,
                    "question": retry_question,
                }

        elif question == "Let me know the make of the car":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM for additional validation
                check_prompt = (
                    f"The user has responded with: '{user_message}'. Determine if this is a valid car make name. "
                    "Respond only with 'Yes' or 'No'."
                )
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                        "Your task is to act as a car maker agency. You can handle lots of car makes, so your job is to check if the given name is a car maker."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_valid_car_make = llm_response.content.strip().lower() == "yes"

                if is_valid_car_make:
                    # Store the name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the car make. Now, let's move on to: {next_question}"
                        }
                    else:
                        # All questions completed
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Use general assistant for invalid LLM validation
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which was not validated as a car make by Insura. "
                        "Please assist them in correcting their input."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": general_assistant_response.content.strip(),
                        "question": f"Let's try agin {question}",
                    }

        elif (
            question
            == "Now, let's gather some details about your bike. Let me know the make of the bike."
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM for additional validation
                check_prompt = (
                    f"The user has responded with: '{user_message}'. Determine if this is a valid bike make name. "
                    "Respond only with 'Yes' or 'No'."
                )
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                        "Your task is to act as a bike maker agency. You can handle lots of bike makes, so your job is to check if the given name is a bike maker."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_valid_bike_make = llm_response.content.strip().lower() == "yes"

                if is_valid_bike_make:
                    # Store the name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the bike make. Now, let's move on to: {next_question}"
                        }
                    else:
                        # All questions completed
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Use general assistant for invalid LLM validation
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which was not validated as a bike make by Insura. "
                        "Please assist them in correcting their input."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": general_assistant_response.content.strip(),
                        "question": f"Let's try agin {question}",
                    }

        elif question == "May I know the model number of your car, please?":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM for additional validation
                check_prompt = (
                    f"The user has responded with: '{user_message}'. Determine if this is a valid car model number. "
                    "Respond only with 'Yes' or 'No'."
                )
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                        "Your task is to act as a car maker agency. You can handle lots of car models, so your job is to check if the given name is a car model."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_valid_car_model = llm_response.content.strip().lower() == "yes"

                if is_valid_car_model:
                    # Store the model number
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the car model number. Now, let's move on to: {next_question}"
                        }
                    else:
                        # All questions completed
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Use general assistant for invalid LLM validation
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which was not validated as a car model number by Insura. "
                        "Please assist them in correcting their input."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": general_assistant_response.content.strip(),
                        "question": f"Let's try again: {question}",
                    }

        elif question == "Could you please tell me the model number of your bike":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM for additional validation
                check_prompt = (
                    f"The user has responded with: '{user_message}'. Determine if this is a valid bike model number. "
                    "Respond only with 'Yes' or 'No'."
                )
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                        "Your task is to act as a bike maker agency. You can handle lots of car models, so your job is to check if the given name is a bike model."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_valid_bike_model = llm_response.content.strip().lower() == "yes"

                if is_valid_bike_model:
                    # Store the model number
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the bike model number. Now, let's move on to: {next_question}"
                        }
                    else:
                        # All questions completed
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Use general assistant for invalid LLM validation
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which was not validated as a bike model number by Insura. "
                        "Please assist them in correcting their input."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": general_assistant_response.content.strip(),
                        "question": f"Let's try again: {question}",
                    }

        elif question == "May I know the variant of your car, please?":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM for additional validation
                check_prompt = (
                    f"The user has responded with: '{user_message}'. Determine if this is a valid car variant. "
                    "Respond only with 'Yes' or 'No'."
                )
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                        "Your task is to act as a car maker agency. You can handle lots of car variants, so your job is to check if the given name is a car variant."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_valid_car_variant = llm_response.content.strip().lower() == "yes"

                if is_valid_car_variant:
                    # Store the variant
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the car variant. Now, let's move on to: {next_question}"
                        }
                    else:
                        # All questions completed
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Use general assistant for invalid LLM validation
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which was not validated as a car variant by Insura. "
                        "Please assist them in correcting their input."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": general_assistant_response.content.strip(),
                        "question": f"Let's try again: {question}",
                    }

        elif question == "Could you please tell me the Variant of your bike":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM for additional validation
                check_prompt = (
                    f"The user has responded with: '{user_message}'. Determine if this is a valid bike variant. "
                    "Respond only with 'Yes' or 'No'."
                )
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                        "Your task is to act as a bike maker agency. You can handle lots of bike variants, so your job is to check if the given name is a bike variant."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_valid_bike_variant = llm_response.content.strip().lower() == "yes"

                if is_valid_bike_variant:
                    # Store the variant
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the bike variant. Now, let's move on to: {next_question}"
                        }
                    else:
                        # All questions completed
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Use general assistant for invalid LLM validation
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which was not validated as a bike variant by Insura. "
                        "Please assist them in correcting their input."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": general_assistant_response.content.strip(),
                        "question": f"Let's try again: {question}",
                    }

        elif question == "May I have the sponsor's mobile number, please?":
            is_mobile_number = is_valid_mobile_number(user_message)

            if is_mobile_number:
                # Store the mobile number
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    # Get the next question text
                    next_question_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )

                    # Translate response to user's language
                    response_msg = translate_text(
                        f"Thank you for providing the mobile number! 📱 Now, let's move on to: {next_question_text}",
                        user_language,
                    )

                    # Handle options if they exist
                    if isinstance(next_question, dict) and "options" in next_question:
                        translated_options = [
                            translate_text(opt, user_language)
                            for opt in next_question["options"]
                        ]
                        return {
                            "response": response_msg,
                            "options": ", ".join(translated_options),
                            "language": user_language,
                            "language_code": get_language_code(user_language),
                        }

                    return {
                        "response": response_msg,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)

                    completion_msg = translate_text(
                        "You're all set! 🎉 Thank you for providing your details. If you need further assistance, feel free to ask.",
                        user_language,
                    )
                    return {
                        "response": completion_msg,
                        "final_responses": responses,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }
            else:
                general_assistant_prompt = f"The user entered '{user_message}'. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                next_question = questions[conversation_state["current_question_index"]]
                if isinstance(next_question, dict) and "options" in next_question:
                    next_question_text = next_question["question"]
                    translated_options = [
                        translate_text(opt, user_language)
                        for opt in next_question["options"]
                    ]
                    retry_msg = translate_text(
                        f"Let's Move Back: {next_question_text}", user_language
                    )
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": retry_msg,
                        "options": options,
                    }

                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                    }

        elif question == "May I have the Client Name, please?":
            return handle_client_name_question(
                question,
                user_message,
                conversation_state,
                questions,
                responses,
                is_valid_name,
            )

        elif question == "May I have the Client mobile number, please?":
            is_mobile_number = is_valid_mobile_number(user_message)

            if is_mobile_number:
                # Store the mobile number
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    return {
                        "response": f"Thank you for providing the mobile number. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                general_assistant_prompt = (
                    f"The user entered '{user_message}', . Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    next_question = next_question["question"]
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {next_question}",
                        "options": options,
                    }

                else:
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                    }

        elif question == "What would you like to do today?":
            return handle_what_would_you_do_today_question(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif question == "Please choose which one would you like to":
            return handle_individual_sma_choice(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif question == "Please Confirm the marital status of":
            # Use multilingual validation for marital status
            valid_options = ["Single", "Married"]
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif question == "May I know sponsor's marital status?":
            valid_options = ["Single", "Married"]
            if user_message in valid_options:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you for your response. Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    return {
                        "response": f"Thank you for your response. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif question == "Tell me you Height in Cm":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Validate user input using regex (only numerical values)
                if not re.match(r"^\d+$", user_message):
                    height_assistant_prompt = (
                        f"The user entered '{user_message}', which does not appear to be a valid height in cm. "
                        "Please assist them in providing a valid height."
                    )
                    height_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=height_assistant_prompt),
                    ])
                    return {
                        "response": f"{height_assistant_response.content.strip()}",
                        "question": f" Let's try again: {question}",
                    }

                # Convert the height to an integer and check for a valid range
                try:
                    height = int(user_message)
                    if 50 <= height <= 300:  # Assuming a realistic height range in cm
                        # Store the height
                        responses[question] = user_message
                        conversation_state["current_question_index"] += 1

                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]
                            return {
                                "response": f"Thank you for providing your height. Now, let's move on to: {next_question}"
                            }
                        else:
                            # All questions completed
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)
                            return {
                                "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                                "final_responses": responses,
                            }
                    else:
                        return {
                            "response": "The height you entered seems unrealistic. Please enter your height in cm (e.g., 170).",
                            "question": f" Let's try again: {question}",
                        }
                except ValueError:
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which is not a valid numerical value for height. "
                        "Please assist them in providing a valid height in cm."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f" Let's try again: {question}",
                    }

        elif question == "Tell me you Weight in Kg":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Validate user input using regex (only numerical values)
                if not re.match(r"^\d+$", user_message):
                    weight_assistant_prompt = (
                        f"The user entered '{user_message}', which does not appear to be a valid weight in kg. "
                        "Please assist them in providing a valid weight."
                    )
                    weight_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=weight_assistant_prompt),
                    ])
                    return {
                        "response": f"{weight_assistant_response.content.strip()}",
                        "question": f" Let's try again: {question}",
                    }

                # Convert the weight to an integer and check for a valid range
                try:
                    weight = int(user_message)
                    if 20 <= weight <= 300:  # Assuming a realistic weight range in kg
                        # Store the weight
                        responses[question] = user_message
                        conversation_state["current_question_index"] += 1

                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]
                            if "options" in next_question:
                                options = ", ".join(next_question["options"])
                                next_questions = next_question["question"]
                                return {
                                    "response": f"Thank you! Now, let's move on to: {next_questions}",
                                    "options": options,
                                }
                            else:
                                return {
                                    "response": f"Thank you for providing your weight. Now, let's move on to: {next_question}"
                                }
                        else:
                            # All questions completed
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)
                            return {
                                "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                                "final_responses": responses,
                            }
                    else:
                        return {
                            "response": "The weight you entered seems unrealistic. Please enter your weight in kg (e.g., 70).",
                            "question": f" Let's try again: {question}",
                        }
                except ValueError:
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which is not a valid numerical value for weight. "
                        "Please assist them in providing a valid weight in kg."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f" Let's try again: {question}",
                    }

        elif question == "Can you please tell me the year your insurance expired?":
            # Store the user-provided year
            if (
                user_message.isdigit() and len(user_message) == 4
            ):  # Ensure valid year format
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions to ask
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                        "options": options,
                    }
                else:
                    # All questions have been answered
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!.",
                        "final_responses": responses,
                    }
            else:
                # Redirect to general assistant for help
                general_assistant_prompt = (
                    f"User response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurances assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's Move back to {question}",
                }

        elif question == "What company does the sponsor work for?":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Check if the input is a company name using LLM
                check_prompt = f"This is the company name: '{user_message}'. Please check if that name could be a company name and respond with 'Yes' or 'No'"
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are a friendly assistant working in Isuran's company department. Your primary task is to verify the user provided input could be a company name. The input might include examples such as 'Fallout Private Limited' or 'Fallout Technologies'. Your role is to validate and identify whether the given input is a valid company name "
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_company_name = llm_response.content.strip().lower() == "yes"

                if is_company_name:
                    # Store the company name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question}"
                        }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = (
                        f"User response: {user_message}. Please assist."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's move back to: {question}",
                    }
        elif question == "Which insurance company is your current policy with?":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Check if the input is a company name using LLM
                check_prompt = f"This is the company name: '{user_message}'. Please check if that name could be a company name and respond with 'Yes' or 'No'"
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are a friendly assistant working in Isuran's company department. Your primary task is to verify the user provided input could be a company name. The input might include examples such as 'Fallout Private Limited' or 'Fallout Technologies'. Your role is to validate and identify whether the given input is a valid company name "
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_company_name = llm_response.content.strip().lower() == "yes"

                if is_company_name:
                    # Store the company name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question}"
                        }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = (
                        f"User response: {user_message}. Please assist."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's move back to: {question}",
                    }

        elif question == "Have you been vaccinated for Covid-19?":
            valid_options = ["Yes", "No"]
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "Yes":
                    # Dynamically add follow-up questions for dose dates
                    first_dose_question = (
                        "Can you please tell me the date of your first dose?"
                    )
                    second_dose_question = (
                        "Can you please tell me the date of your second dose?"
                    )

                    # Insert follow-up questions into the list if not already present
                    if first_dose_question not in questions:
                        responses[first_dose_question] = None
                        questions.insert(
                            conversation_state["current_question_index"] + 1,
                            first_dose_question,
                        )

                    if second_dose_question not in questions:
                        responses[second_dose_question] = None
                        questions.insert(
                            conversation_state["current_question_index"] + 2,
                            second_dose_question,
                        )

                    # Move to the next question
                    conversation_state["current_question_index"] += 1
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
                elif user_message == "No":
                    # Remove the questions about first and second doses if they exist
                    first_dose_question = (
                        "Can you please tell me the date of your first dose?"
                    )
                    second_dose_question = (
                        "Can you please tell me the date of your second dose?"
                    )

                    if first_dose_question in questions:
                        questions.remove(first_dose_question)
                    if second_dose_question in questions:
                        questions.remove(second_dose_question)

                    # Proceed to the next predefined question
                    conversation_state["current_question_index"] += 1
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]

                        return {
                            "response": f"Thank you for your response. Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        # All predefined questions have been answered
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }

        elif question == "Can you please tell me the date of your first dose?":
            # Validate and store the first dose date
            if valid_date_format(
                user_message
            ):  # Replace with your date validation function
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions to ask
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
                else:
                    # All questions have been answered
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                        "final_responses": responses,
                    }
            else:
                return {
                    "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY or MM-DD-YYYY."
                }

        elif question == "Can you please tell me the date of your second dose?":
            # Validate and store the second dose date
            if valid_date_format(
                user_message
            ):  # Replace with your date validation function
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions to ask
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                        "options": options,
                    }
                else:
                    # All questions have been answered
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                        "final_responses": responses,
                    }
            else:
                return {
                    "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY or MM-DD-YYYY."
                }

        elif (
            question
            == "Your policy is up for renewal. Would you like to proceed with renewing it?"
        ):
            valid_options = ["Yes", "No"]
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "Yes":
                    # Proceed to the next predefined question
                    conversation_state["current_question_index"] += 1
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_question}"
                        }
                    else:
                        # All predefined questions have been answered
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }

                elif user_message == "No":
                    # Update the responses and return the final response
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for your response. Your request has been updated accordingly. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                return {
                    "response": "Invalid response. Please answer with 'Yes' or 'No'."
                }

        elif question in [
            "Now, let's move to the sponsor details. Please provide the Sponsor Name?",
            # "Next, we need the details of the member for whom the policy is being purchased. Please provide Name",
            "Please provide the member's details.Please tell me the Name",
            "Next, Please provide the member's details.Please tell me the Name",
            "Could you please provide your full name",
            "Could you kindly share your contact details with me? To start, may I know your name, please?",
        ]:
            return handle_validate_name(
                question,
                user_message,
                conversation_state,
                questions,
                responses,
                is_valid_name,
            )

        elif (
            question
            == "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
        ):
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    next_questions = next_question["question"]
                    member_name = responses.get[
                        "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                    ]
                    return {
                        "response": f"Thank you,May I know the {next_question} of {member_name}.Please ensure it is in the format DD/MM/YYYY.",
                        "options": options,
                    }
                else:
                    member_name = responses.get(
                        "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                    )

                    return {
                        "response": f"Thank you,May I know the {next_question} of {member_name}.Please ensure it is in the format DD/MM/YYYY."
                    }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }

            # if conversation_state["current_question_index"] == questions.index(question):
            #     # Prompt LLM to check if the input is a valid person name
            #     check_prompt = f"The user has responded with: '{user_message}'. Is this a valid person's name? Respond with 'Yes' or 'No'."
            #     llm_response = llm.invoke([
            #         SystemMessage(content="You are Insura, an AI assistant specialized in insurance-related tasks. Your task is to determine if the input provided by the user is a valid person's name.Make sure it a valide name for a person"),
            #         HumanMessage(content=check_prompt)
            #     ])
            #     is_person_name = llm_response.content.strip().lower() == "yes"

            #     if is_person_name:
            #         # Store the person's name
            #         responses[question] = user_message
            #         conversation_state["current_question_index"] += 1

            #         # Check if there are more questions
            #         if conversation_state["current_question_index"] < len(questions):
            #             next_question = questions[conversation_state["current_question_index"]]
            #             return {
            #                 "response": f"Thank you for providing the sponsor's name. Now, let's move on to: {next_question}"
            #             }
            #         else:
            #             # If all questions are completed, save responses and end conversation
            #             with open("user_responses.json", "w") as file:
            #                 json.dump(responses, file, indent=4)
            #             return {
            #                 "response": "Thank you for using Insura. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
            #                 "final_responses": responses
            #             }
            #     else:
            #         # Handle invalid or unrelated input
            #         general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a person's name. Please assist."
            #         general_assistant_response = llm.invoke([
            #             SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately."),
            #             HumanMessage(content=general_assistant_prompt)
            #         ])
            #         return {
            #             "response": f"{general_assistant_response.content.strip()}",
            #             "question": f"Let's move back to: {question}"
            #         }

        elif question == "How many years of driving experience do you have in the UAE?":
            valid_options = ["0-1 year", "1-2 years", "2+ years"]
            if user_message in valid_options:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    return {
                        "response": f"Thank you for your response. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif question == "Could you please let me know the year your car was made?":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Check if the input is a valid year
                try:
                    year = int(user_message)
                    current_year = datetime.now().year
                    if 1886 <= year <= current_year:  # Cars were invented in 1886
                        # Valid year
                        responses[question] = user_message
                        conversation_state["current_question_index"] += 1

                        # Check if there are more questions
                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]

                            return {
                                "response": f"Thank you for providing the year. Now, let's move on to: {next_question}",
                            }
                        else:
                            # Save responses and end the conversation
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)
                            return {
                                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                                "final_responses": responses,
                            }
                    else:
                        # Year is out of range
                        raise ValueError("Invalid year range")
                except ValueError:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}' when asked for the year their car was made. Please assist."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's revisit: {question}",
                    }

        elif (
            question
            == "Could you please provide the registration details? When was your car first registered?"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Check if the input is a valid year
                try:
                    year = int(user_message)
                    current_year = datetime.now().year
                    if 1886 <= year <= current_year:  # Cars were invented in 1886
                        # Valid year
                        responses[question] = user_message
                        conversation_state["current_question_index"] += 1

                        # Check if there are more questions
                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]

                            return {
                                "response": f"Thank you for providing the year. Now, let's move on to: {next_question}",
                            }
                        else:
                            # Save responses and end the conversation
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)
                            return {
                                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                                "final_responses": responses,
                            }
                    else:
                        # Year is out of range
                        raise ValueError("Invalid year range")
                except ValueError:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}' when asked for the year their car was made. Please assist."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's revisit: {question}",
                    }

        elif question == "Do you have a No Claim certificate?":
            valid_options = ["No", "1 Year", "2 Years", "3+ Years"]
            if user_message in valid_options:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    return {
                        "response": f"Thank you for your response. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif question == "Could you provide the sponsor's Emirates ID?":
            # Validate sponsor Emirates ID
            if valid_emirates_id(user_message):
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Move to the next question or finalize responses
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
                else:
                    # All questions answered
                    try:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your responses have been recorded. "
                            "Feel free to ask any other questions. Have a great day!",
                            "final_responses": responses,
                        }
                    except Exception as e:
                        return {
                            "response": f"An error occurred while saving your responses: {str(e)}"
                        }
            else:
                # Handle invalid Emirates ID or unrelated query
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])

                # Example of a valid Emirates ID
                emirates_id_example = "784-1990-1234567-0"

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": f"Here's an example of a valid Emirates ID for your reference: {emirates_id_example}.",
                    "question": f"Let's try again: {question}",
                }

        elif question == "Do you have a vehicle test passing certificate?":
            return handle_yes_or_no(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif question == "Does your current policy have comprehensive cover?":
            return handle_yes_or_no(
                user_message,
                conversation_state,
                questions,
                responses,
                question,
                user_language,
            )

        elif question == "Does your policy include agency repair?":
            valid_options = ["Yes", "No"]
            if user_message in valid_options:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    options = ", ".join(next_question["options"])
                    next_questions = next_question["question"]

                    return {
                        "response": f"Thank you for your response. Now, let's move on to: {next_questions}",
                        "options": options,
                    }

                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif (
            question
            == "Please enter your Insurance Advisor code for assigning your enquiry for further assistance"
        ):
            if valid_adivisor_code(user_message):
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Move to the next question or finalize responsess
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    response_message = (
                        f"Thank you! Now, let's move on to: {next_question}"
                    )
                    return format_response_in_language(
                        response_message, [], user_language
                    )
                else:
                    try:
                        if (
                            responses.get("Do you have an Insurance Advisor code?")
                            == "Yes"
                        ):
                            medical_detail_response = fetching_medical_detail(responses)
                            print(medical_detail_response)
                            del user_states[user_id]
                            if isinstance(medical_detail_response, int):
                                # Translate the success response to user's language
                                success_message = "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please find the link below to view your quotation:"
                                review_message = "If you are satisfied with Wehbe(Broker) services, please leave a review for sharing happiness to others!!😊"

                                translated_success = translate_text(
                                    success_message, user_language
                                )
                                translated_review = translate_text(
                                    review_message, user_language
                                )

                                # Build the customer plan link using the medical detail response ID
                                customer_plan_link = f"{INSURANCE_LAB_BASE_URL}/customer_plan/{medical_detail_response}"

                                # Reset conversation state to allow starting a new inquiry (same as SME flow)
                                saved_language = conversation_state.get(
                                    "preferred_language", "English"
                                )
                                saved_language_code = conversation_state.get(
                                    "language_code", "en"
                                )
                                saved_language_explicitly_set = conversation_state.get(
                                    "language_explicitly_set", False
                                )

                                user_states[user_id] = {
                                    "current_question_index": 0,
                                    "responses": {},
                                    "current_flow": "initial",
                                    "welcome_shown": False,  # Set to False to allow new greeting on restart
                                    "awaiting_document_name": False,
                                    "document_name": "",
                                    "last_takaful_query_time": None,
                                    "awaiting_takaful_followup": False,
                                    "last_chronic_conditions_time": None,
                                    "awaiting_chronic_conditions_followup": False,
                                    "takaful_emarat_asked": False,
                                    "preferred_language": saved_language,
                                    "language_code": saved_language_code,
                                    "language_explicitly_set": saved_language_explicitly_set,
                                }

                                return {
                                    "response": translated_success,
                                    "link": customer_plan_link,
                                    "review_message": translated_review,
                                    "review_link": "https://www.google.com/search?client=ms-android-samsung-ss&sca_esv=4eb717e6f42bf628&sxsrf=AHTn8zprabdPVFL3C2gXo4guY8besI3jqQ:1744004771562&q=wehbe+insurance+services+llc+reviews&uds=ABqPDvy-z0dcsfm2PY76_gjn-YWou9-AAVQ4iWjuLR6vmDV0vf3KpBMNjU5ZkaHGmSY0wBrWI3xO9O55WuDmXbDq6a3SqlwKf2NJ5xQAjebIw44UNEU3t4CpFvpLt9qFPlVh2F8Gfv8sMuXXSo2Qq0M_ZzbXbg2c323G_bE4tVi7Ue7d_sW0CrnycpJ1CvV-OyrWryZw_TeQ3gLGDgzUuHD04MpSHquYZaSQ0_mIHLWjnu7fu8c7nb6_aGDb_H1Q-86fD2VmWluYA5jxRkC9U2NsSwSSXV4FPW9w1Q2T_Wjt6koJvLgtikd66MqwYiJPX2x9MwLhoGYlpTbKtkJuHwE9eM6wQgieChskow6tJCVjQ75I315dT8n3tUtasGdBkprOlUK9ibPrYr9HqRz4AwzEQaxAq9_EDcsSG_XW0CHuqi2lRKHw592MlGlhjyQibXKSZJh-v3KW4wIVqa-2x0k1wfbZdpaO3BZaKYCacLOxwUKTnXPbQqDPLQDeYgDBwaTLvaCN221H&si=APYL9bvoDGWmsM6h2lfKzIb8LfQg_oNQyUOQgna9TyfQHAoqUvvaXjJhb-NHEJtDKiWdK3OqRhtZNP2EtNq6veOxTLUq88TEa2J8JiXE33-xY1b8ohiuDLBeOOGhuI1U6V4mDc9jmZkDoxLC9b6s6V8MAjPhY-EC_g%3D%3D&sa=X&sqi=2&ved=2ahUKEwi05JSHnMWMAxUw8bsIHRRCDd0Qk8gLegQIHxAB&ictx=1&stq=1&cs=0&lei=o2bzZ_SGIrDi7_UPlIS16A0#ebo=1",
                                    "language": user_language,
                                    "language_code": get_language_code(user_language),
                                    "restart_conversation": True,  # Signal to frontend to restart
                                }
                            else:
                                # Translate the fallback response to user's language
                                fallback_message = "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae."
                                translated_fallback = translate_text(
                                    fallback_message, user_language
                                )
                                return {
                                    "response": translated_fallback,
                                    "language": user_language,
                                    "language_code": get_language_code(user_language),
                                }
                    except Exception as e:
                        # Translate the error message to user's language
                        error_message = f"An error occurred while fetching medical details: {str(e)}"
                        translated_error = translate_text(error_message, user_language)
                        return {
                            "response": translated_error,
                            "language": user_language,
                            "language_code": get_language_code(user_language),
                        }

                    try:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        # Translate the no agent code message to user's language
                        no_agent_message = "Since you don't have an agent code, we will arrange a callback from the next available agent to assist you further. Thank you!"
                        translated_no_agent = translate_text(
                            no_agent_message, user_language
                        )
                        return {
                            "response": translated_no_agent,
                            "final_responses": responses,
                            "language": user_language,
                            "language_code": get_language_code(user_language),
                        }
                    except Exception as e:
                        # Translate the save error message to user's language
                        save_error_message = (
                            f"An error occurred while saving your responses: {str(e)}"
                        )
                        translated_save_error = translate_text(
                            save_error_message, user_language
                        )
                        return {
                            "response": translated_save_error,
                            "language": user_language,
                            "language_code": get_language_code(user_language),
                        }
            else:
                # Handle invalid advisor code or unrelated query
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist in {user_language}."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                # Translate the example message to user's language
                example_message = "The Advisor code should be a 4-digit numeric value. Please enter a valid code"
                translated_example = translate_text(example_message, user_language)

                retry_question = translate_text(
                    f"Let's try again: {question}", user_language
                )

                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "example": translated_example,
                    "question": retry_question,
                    "language": user_language,
                    "language_code": get_language_code(user_language),
                }
        elif question == "Could you please tell me the year your bike was made?":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Check if the input is a valid year
                try:
                    year = int(user_message)
                    current_year = datetime.now().year
                    if 1886 <= year <= current_year:  # Cars were invented in 1886
                        # Valid year
                        responses[question] = user_message
                        conversation_state["current_question_index"] += 1

                        # Check if there are more questions
                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]

                            return {
                                "response": f"Thank you for providing the year. Now, let's move on to: {next_question}",
                            }
                        else:
                            # Save responses and end the conversation
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)
                            return {
                                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                                "final_responses": responses,
                            }
                    else:
                        # Year is out of range
                        raise ValueError("Invalid year range")
                except ValueError:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}' when asked for the year their car was made. Please assist."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's revisit: {question}",
                    }

        elif (
            question
            == "Could you please provide the registration details? When was your bike first registered?"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Check if the input is a valid year
                try:
                    year = int(user_message)
                    current_year = datetime.now().year
                    if 1886 <= year <= current_year:  # Cars were invented in 1886
                        # Valid year
                        responses[question] = user_message
                        conversation_state["current_question_index"] += 1

                        # Check if there are more questions
                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]

                            return {
                                "response": f"Thank you for providing the year. Now, let's move on to: {next_question}",
                            }
                        else:
                            # Save responses and end the conversation
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)
                            return {
                                "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                                "final_responses": responses,
                            }
                    else:
                        # Year is out of range
                        raise ValueError("Invalid year range")
                except ValueError:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}' when asked for the year their car was made. Please assist."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's revisit: {question}",
                    }

        elif (
            question
            == "Could you kindly share your contact details with me? To start, may I know your name, please?"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM to check if the input is a valid person name
                check_prompt = f"The user has responded with: '{user_message}'. Is this a valid person's name? Respond with 'Yes' or 'No'."
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. Your task is to determine if the input provided by the user is a valid person's name.Make sure it a valide name for a person"
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_person_name = llm_response.content.strip().lower() == "yes"

                if is_person_name:
                    # Store the person's name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing your name. Now, let's move on to: {next_question}"
                        }
                    else:
                        # If all questions are completed, save responses and end conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a person's name. Please assist."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's move back to: {question}",
                    }

        elif (
            question
            == "Could you kindly provide me with the sponsor's Source of Income"
        ):
            valid_options = ["Business", "Salary"]
            if user_message in valid_options:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        return {
                            "response": f"Thank you. Now, let's move on to: {next_question}"
                        }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif (
            question == "Are you suffering from any pre-existing or chronic conditions?"
        ):
            valid_options = ["Yes", "No"]
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "No":
                    # Check if the follow-up question is already in the list
                    follow_up_question = "Please provide us with the details of your Chronic Conditions Medical Report"
                    if follow_up_question in questions:
                        # If the follow-up question exists, skip it and proceed
                        questions.remove(follow_up_question)

                    # Proceed to the next predefined question
                    conversation_state["current_question_index"] += 1
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]

                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        # All predefined questions have been answered
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }

                elif user_message == "Yes":
                    # Dynamically add the follow-up question if not already present
                    follow_up_question = "Please provide us with the details of your Chronic Conditions Medical Report"
                    if follow_up_question not in questions:
                        responses[follow_up_question] = None
                        # Insert the new question immediately after the current one
                        questions.insert(
                            conversation_state["current_question_index"] + 1,
                            follow_up_question,
                        )

                    # Move to the new follow-up question
                    conversation_state["current_question_index"] += 1

                    return {
                        "response": f"Thank you! Now, let's move on to: {follow_up_question}",
                    }
            else:
                pass

        elif (
            question
            == "Please provide us with the details of your Chronic Conditions Medical Report"
        ):
            # if conversation_state["current_question_index"] == questions.index(question):
            #     # Enhanced file path validation
            #     upload_pattern = re.compile(
            #         r"^uploads\/(?:[\w\s-]+\/)*[\w\s-]+\.(pdf|docx|jpg|png|jpeg)$", re.IGNORECASE
            #     )

            #     if upload_pattern.match(user_message):
            #         # Valid file format
            #         responses[question] = user_message
            #         conversation_state["current_question_index"] += 1

            #         # Check if there are more questions
            #         if conversation_state["current_question_index"] < len(questions):
            #             next_question = questions[conversation_state["current_question_index"]]
            #             return {
            #                 "response": f"Thank you for providing the document. Now, let's move on to: {next_question}",
            #             }
            #         else:
            #             # Save responses and end the conversation
            #             with open("user_responses.json", "w") as file:
            #                 json.dump(responses, file, indent=4)
            #             return {
            #                 "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
            #                 "final_responses": responses,
            #             }
            #     else:
            #         # Invalid file format
            #         return {
            #            "response": "The file format seems incorrect. Please upload a valid document."
            #         }

            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Store user message as is without validation
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    options = ", ".join(next_question["options"])
                    next_questions = next_question["question"]

                    return {
                        "response": f"Thank you! Now, let's move on to: {next_questions}",
                        "options": options,
                    }
                else:
                    # Save responses and end the conversation
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                        "final_responses": responses,
                    }
            else:
                # Handle other questions or general assistance
                pass
        elif question in [
            "May I have the sponsor's Email Address, please?",
            "May i know your Email address",
            "May I have the Client Email Address, please?",
        ]:
            # Regex pattern for validating email address - also accept numbers
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            # Check if it's an email or just numbers (accept both)
            is_valid_input = (
                re.match(email_pattern, user_message) or user_message.strip().isdigit()
            )

            if is_valid_input:
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    # Determine appropriate response based on current flow and question
                    if (
                        current_flow == "sma"
                        and question == "May I have the Client Email Address, please?"
                    ):
                        thank_you_message = (
                            "Thank you for providing the client's email! 📧"
                        )
                    else:
                        thank_you_message = (
                            "Thank you for providing the sponsor's email! 📧"
                        )

                    # Get next question text
                    next_question_text = (
                        next_question["question"]
                        if isinstance(next_question, dict)
                        else next_question
                    )

                    # Translate response to user's language
                    response_msg = translate_text(
                        f"{thank_you_message} Now, let's move on to: {next_question_text}",
                        user_language,
                    )

                    if isinstance(next_question, dict) and "options" in next_question:
                        translated_options = [
                            translate_text(opt, user_language)
                            for opt in next_question["options"]
                        ]
                        return {
                            "response": response_msg,
                            "options": ", ".join(translated_options),
                            "language": user_language,
                            "language_code": get_language_code(user_language),
                        }
                    else:
                        return {
                            "response": response_msg,
                            "language": user_language,
                            "language_code": get_language_code(user_language),
                        }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)

                    completion_msg = translate_text(
                        "Thank you for sharing the details! 🎉 We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae",
                        user_language,
                    )
                    return {
                        "response": completion_msg,
                        "final_responses": responses,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }
            else:
                # Handle invalid input
                general_assistant_prompt = f"The user entered '{user_message}'. Please assist them in {user_language}."
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's move back to: {question}",
                    "example": "The email address should be in this format: example@gmail.com",
                }
        elif (
            question
            == "Please upload an Excel file to get your medical insurance details"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Import user_states from excel_upload module
                from routes.excel_upload import user_states as excel_user_states
                import requests

                # Check if user has uploaded Excel file via the upload-excel endpoint
                excel_data_exists = (
                    user_id in excel_user_states
                    and "responses" in excel_user_states[user_id]
                    and "excel_employee_data" in excel_user_states[user_id]["responses"]
                )

                # Also check for file path patterns (backwards compatibility)
                upload_pattern = re.compile(
                    r"^(?:uploads\/)?(?:[\w\s\-\.\/]+\/)*[\w\s\-\.]+\.(xlsx|xls)$",
                    re.IGNORECASE,
                )

                is_valid_file_path = (
                    upload_pattern.match(user_message)
                    or user_message.lower().endswith((".xlsx", ".xls"))
                    or "uploaded" in user_message.lower()
                    or "excel" in user_message.lower()
                )

                if excel_data_exists or is_valid_file_path:
                    # Valid Excel file format or data exists
                    responses[question] = user_message

                    # Store the Excel file path/confirmation for processing
                    responses["excel_file_path"] = user_message

                    # If Excel data exists, store it in responses
                    if excel_data_exists:
                        responses["excel_employee_data"] = excel_user_states[user_id][
                            "responses"
                        ]["excel_employee_data"]

                        # Get the Excel employee data
                        excel_data = excel_user_states[user_id]["responses"][
                            "excel_employee_data"
                        ]
                        employees_list = excel_data.get("employees", [])
                        print(
                            f"Excel Employees List: {json.dumps(employees_list, indent=2)}"
                        )

                        # Build the members array from Excel data
                        members = []
                        for emp in employees_list:
                            member = {
                                "mem_name": emp.get("first_name", ""),
                                "mem_dob": emp.get("date_of_birth", ""),
                                "mem_gender": emp.get("gender", ""),
                                "mem_marital_status": emp.get("marital_status", ""),
                                "mem_relation": emp.get("relation", ""),
                                "mem_nationality": emp.get("nationality", ""),
                                "mem_emirate": emp.get("visa_issued_location", ""),
                            }
                            members.append(member)
                        print(f"Built Members Array: {json.dumps(members, indent=2)}")

                        # Get the other responses from the conversation
                        visa_issued_emirates = ""
                        plan = ""
                        client_name = ""
                        client_mobile = ""
                        client_email = ""

                        print(f"All Responses: {json.dumps(responses, indent=2)}")

                        # Find these from the responses dictionary
                        for key, value in responses.items():
                            if (
                                "Visa issued Emirate" in key
                                or key
                                == "Let's start with your Medical insurance details. Choose your Visa issued Emirate?"
                            ):
                                visa_issued_emirates = value
                            elif (
                                "type of plan" in key.lower()
                                or key == "What type of plan are you looking for?"
                            ):
                                plan = value
                            elif "Client Name" in key:
                                client_name = value
                            elif "Client mobile" in key:
                                client_mobile = value
                            elif "Client Email" in key:
                                client_email = value

                        print(f"Extracted Values:")
                        print(f"  visa_issued_emirates: {visa_issued_emirates}")
                        print(f"  plan: {plan}")
                        print(f"  client_name: {client_name}")
                        print(f"  client_mobile: {client_mobile}")
                        print(f"  client_email: {client_email}")

                        # Validate required fields
                        if (
                            not visa_issued_emirates
                            or not plan
                            or not client_name
                            or not client_mobile
                            or not client_email
                        ):
                            print("ERROR: Missing required fields!")
                            return {
                                "response": "Thank you for uploading the Excel file. However, some required information is missing. Please provide all client details.",
                                "missing_fields": {
                                    "visa_issued_emirates": not visa_issued_emirates,
                                    "plan": not plan,
                                    "client_name": not client_name,
                                    "client_mobile": not client_mobile,
                                    "client_email": not client_email,
                                },
                            }

                        # Prepare the JSON payload
                        payload = {
                            "visa_issued_emirates": visa_issued_emirates,
                            "plan": plan,
                            "client_name": client_name,
                            "client_mobile": client_mobile,
                            "client_email": client_email,
                            "currency": "",
                            "census_sheet": "",
                            "members": members,
                        }

                        # Submit to the API
                        try:
                            # Print the payload for debugging
                            print(f"API Payload: {json.dumps(payload, indent=2)}")
                            print(f"API URL: {INSURANCE_LAB_SME_ADD_API}")

                            # Set proper headers to avoid Mod_Security issues
                            headers = {
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                                "User-Agent": "InsuraBot/1.0",
                            }

                            # Try to send as JSON first
                            try:
                                api_response = requests.post(
                                    INSURANCE_LAB_SME_ADD_API,
                                    json=payload,
                                    headers=headers,
                                    timeout=30,
                                )
                            except:
                                # If JSON fails, try as form data
                                print("JSON request failed, trying form data...")
                                api_response = requests.post(
                                    INSURANCE_LAB_SME_ADD_API,
                                    data=payload,
                                    headers=headers,
                                    timeout=30,
                                )

                            print(f"API Response Status: {api_response.status_code}")
                            print(f"API Response Text: {api_response.text}")

                            if api_response.status_code == 200:
                                response_data = api_response.json()
                                print(
                                    f"API Response Data: {json.dumps(response_data, indent=2)}"
                                )

                                # Store the ID from response if it exists
                                response_id = response_data.get("id", "")
                                print(f"Extracted ID: {response_id}")

                                # Build the customer plan link using the ID
                                customer_plan_link = (
                                    f"{INSURANCE_LAB_SME_PLAN_BASE}/{response_id}"
                                )

                                # Store API response in user state
                                responses["api_response_id"] = response_id
                                responses["api_submission_status"] = "success"
                                responses["customer_plan_link"] = customer_plan_link

                                # Check if there are more questions after this
                                conversation_state["current_question_index"] += 1

                                if conversation_state["current_question_index"] < len(
                                    questions
                                ):
                                    next_question = questions[
                                        conversation_state["current_question_index"]
                                    ]
                                    if isinstance(next_question, dict):
                                        if "options" in next_question:
                                            options = ", ".join(
                                                next_question["options"]
                                            )
                                            next_questions = next_question["question"]
                                            return {
                                                "response": f"Thank you for uploading the Excel file. Your data has been processed successfully (ID: {response_id}). Now, let's move on to: {next_questions}",
                                                "options": options,
                                                "submission_id": response_id,
                                                "customer_plan_link": customer_plan_link,
                                            }
                                        else:
                                            next_questions = next_question["question"]
                                            return {
                                                "response": f"Thank you for uploading the Excel file. Your data has been processed successfully (ID: {response_id}). Now, let's move on to: {next_questions}",
                                                "submission_id": response_id,
                                                "customer_plan_link": customer_plan_link,
                                            }
                                    else:
                                        return {
                                            "response": f"Thank you for uploading the Excel file. Your data has been processed successfully (ID: {response_id}). Now, let's move on to: {next_question}",
                                            "submission_id": response_id,
                                            "customer_plan_link": customer_plan_link,
                                        }
                                else:
                                    # Save responses and end the conversation - SMA flow completion
                                    with open("user_responses.json", "w") as file:
                                        json.dump(responses, file, indent=4)

                                    # Format the response similar to individual flow
                                    success_message = "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please find the link below to view your quotation:"
                                    review_message = "If you are satisfied with Wehbe(Broker) services, please leave a review for sharing happiness to others!!😊"

                                    # Translate messages if needed
                                    translated_success = (
                                        translate_text(success_message, user_language)
                                        if user_language != "English"
                                        else success_message
                                    )
                                    translated_review = (
                                        translate_text(review_message, user_language)
                                        if user_language != "English"
                                        else review_message
                                    )

                                    # Reset conversation state to allow starting a new inquiry
                                    # Save the language preference before resetting
                                    saved_language = conversation_state.get(
                                        "preferred_language", "English"
                                    )
                                    saved_language_code = conversation_state.get(
                                        "language_code", "en"
                                    )
                                    saved_language_explicitly_set = (
                                        conversation_state.get(
                                            "language_explicitly_set", False
                                        )
                                    )

                                    user_states[user_id] = {
                                        "current_question_index": 0,
                                        "responses": {},
                                        "current_flow": "initial",
                                        "welcome_shown": False,  # Set to False to allow new greeting on restart
                                        "awaiting_document_name": False,
                                        "document_name": "",
                                        "last_takaful_query_time": None,
                                        "awaiting_takaful_followup": False,
                                        "last_chronic_conditions_time": None,
                                        "awaiting_chronic_conditions_followup": False,
                                        "takaful_emarat_asked": False,
                                        "preferred_language": saved_language,  # Preserve language
                                        "language_code": saved_language_code,
                                        "language_explicitly_set": saved_language_explicitly_set,  # Preserve explicit setting
                                    }

                                    return {
                                        "response": translated_success,
                                        "link": customer_plan_link,
                                        "review_message": translated_review,
                                        "review_link": "https://www.google.com/search?client=ms-android-samsung-ss&sca_esv=4eb717e6f42bf628&sxsrf=AHTn8zprabdPVFL3C2gXo4guY8besI3jqQ:1744004771562&q=wehbe+insurance+services+llc+reviews&uds=ABqPDvy-z0dcsfm2PY76_gjn-YWou9-AAVQ4iWjuLR6vmDV0vf3KpBMNjU5ZkaHGmSY0wBrWI3xO9O55WuDmXbDq6a3SqlwKf2NJ5xQAjebIw44UNEU3t4CpFvpLt9qFPlVh2F8Gfv8sMuXXSo2Qq0M_ZzbXbg2c323G_bE4tVi7Ue7d_sW0CrnycpJ1CvV-OyrWryZw_TeQ3gLGDgzUuHD04MpSHquYZaSQ0_mIHLWjnu7fu8c7nb6_aGDb_H1Q-86fD2VmWluYA5jxRkC9U2NsSwSSXV4FPW9w1Q2T_Wjt6koJvLgtikd66MqwYiJPX2x9MwLhoGYlpTbKtkJuHwE9eM6wQgieChskow6tJCVjQ75I315dT8n3tUtasGdBkprOlUK9ibPrYr9HqRz4AwzEQaxAq9_EDcsSG_XW0CHuqi2lRKHw592MlGlhjyQibXKSZJh-v3KW4wIVqa-2x0k1wfbZdpaO3BZaKYCacLOxwUKTnXPbQqDPLQDeYgDBwaTLvaCN221H&si=APYL9bvoDGWmsM6h2lfKzIb8LfQg_oNQyUOQgna9TyfQHAoqUvvaXjJhb-NHEJtDKiWdK3OqRhtZNP2EtNq6veOxTLUq88TEa2J8JiXE33-xY1b8ohiuDLBeOOGhuI1U6V4mDc9jmZkDoxLC9b6s6V8MAjPhY-EC_g%3D%3D&sa=X&sqi=2&ved=2ahUKEwi05JSHnMWMAxUw8bsIHRRCDd0Qk8gLegQIHxAB&ictx=1&stq=1&cs=0&lei=o2bzZ_SGIrDi7_UPlIS16A0#ebo=1",
                                        "language": user_language,
                                        "language_code": get_language_code(
                                            user_language
                                        ),
                                        "restart_conversation": True,  # Signal to frontend to restart
                                    }
                            else:
                                # API call failed
                                print(
                                    f"API Error - Status: {api_response.status_code}, Response: {api_response.text}"
                                )
                                responses["api_submission_status"] = "error"
                                responses["api_error_message"] = api_response.text
                                return {
                                    "response": f"Thank you for uploading the Excel file. However, there was an issue processing your data (Error: {api_response.status_code}). Please try again or contact support@insuranceclub.ae",
                                    "error_details": api_response.text,
                                }
                        except requests.exceptions.RequestException as e:
                            # Handle request exceptions
                            responses["api_submission_status"] = "error"
                            responses["api_error_message"] = str(e)
                            print(f"API request error: {e}")
                            # Continue with normal flow even if API fails
                            conversation_state["current_question_index"] += 1

                            if conversation_state["current_question_index"] < len(
                                questions
                            ):
                                next_question = questions[
                                    conversation_state["current_question_index"]
                                ]
                                if isinstance(next_question, dict):
                                    options = ", ".join(
                                        next_question.get("options", [])
                                    )
                                    next_questions = next_question.get("question", "")
                                    return {
                                        "response": f"Thank you for uploading the Excel file. There was a temporary issue, but we've saved your data. Now, let's move on to: {next_questions}",
                                        "options": options,
                                    }
                                else:
                                    return {
                                        "response": f"Thank you for uploading the Excel file. Now, let's move on to: {next_question}"
                                    }
                            else:
                                return {
                                    "response": "Thank you for uploading the Excel file. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry.",
                                    "final_responses": responses,
                                }
                    else:
                        # Excel data doesn't exist yet - wait for it
                        conversation_state["current_question_index"] += 1

                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]
                            if isinstance(next_question, dict):
                                options = ", ".join(next_question.get("options", []))
                                next_questions = next_question.get("question", "")
                                return {
                                    "response": f"Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Now, let's move on to: {next_questions}",
                                    "options": options,
                                }
                            else:
                                return {
                                    "response": f"Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Now, let's move on to: {next_question}"
                                }
                        else:
                            # Save responses and end the conversation - SMA flow completion
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)

                            # Reset conversation state to allow starting a new inquiry
                            user_states[user_id] = {
                                "current_question_index": 0,
                                "responses": {},
                                "current_flow": "initial",
                                "welcome_shown": True,
                                "awaiting_document_name": False,
                                "document_name": "",
                                "last_takaful_query_time": None,
                                "awaiting_takaful_followup": False,
                                "last_chronic_conditions_time": None,
                                "awaiting_chronic_conditions_followup": False,
                                "takaful_emarat_asked": False,
                            }

                            return {
                                "response": "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae",
                                "final_responses": responses,
                            }
                else:
                    # Invalid file format
                    return {
                        "response": "The file format seems incorrect. Please upload a valid Excel file (xlsx or xls format) using the upload button."
                    }
        elif question == "Do you have an Insurance Advisor code?":
            # Use the advisor code handler with multilingual support
            return handle_adiviosr_code(
                question,
                user_message,
                responses,
                conversation_state,
                questions,
                user_language,
            )
        elif question == "Could you kindly share your relationship with the sponsor?":
            # Use multilingual validation for relationship selection
            valid_options = [
                "Investor",
                "Employee",
                "Spouse",
                "Child",
                "4th Child",
                "Parent",
                "Domestic",
            ]
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )
        elif question == "Please upload photos of your driving license Front side":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Enhanced file path validation
                upload_pattern = re.compile(
                    r"^uploads\/(?:[\w\s-]+\/)*[\w\s-]+\.(pdf|docx|jpg|png|jpeg)$",
                    re.IGNORECASE,
                )

                if upload_pattern.match(user_message):
                    # Valid file format
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the driving license Front side. Now, let's move on to: {next_question}",
                        }
                    else:
                        # Save responses and end the conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Invalid file format
                    return {
                        "response": "The file format seems incorrect. Please upload a valid document."
                    }

        elif question == "Please upload photos of your driving license Back side":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Enhanced file path validation
                upload_pattern = re.compile(
                    r"^uploads\/(?:[\w\s-]+\/)*[\w\s-]+\.(pdf|docx|jpg|png|jpeg)$",
                    re.IGNORECASE,
                )

                if upload_pattern.match(user_message):
                    # Valid file format
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the driving license. Now, let's move on to: {next_question}",
                        }
                    else:
                        # Save responses and end the conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Invalid file format
                    return {
                        "response": "The file format seems incorrect. Please upload a valid document."
                    }

        elif (
            question
            == "Please upload photos of your vehicle registration (Mulkiya) Front side"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Enhanced file path validation
                upload_pattern = re.compile(
                    r"^uploads\/(?:[\w\s-]+\/)*[\w\s-]+\.(pdf|docx|jpg|png|jpeg)$",
                    re.IGNORECASE,
                )

                if upload_pattern.match(user_message):
                    # Valid file format
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing vehicle registration Front side. Now, let's move on to: {next_question}",
                        }
                    else:
                        # Save responses and end the conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Invalid file format
                    return {
                        "response": "The file format seems incorrect. Please upload a valid document."
                    }

        elif (
            question
            == "Please upload photos of your vehicle registration (Mulkiya)  Back side"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Enhanced file path validation
                upload_pattern = re.compile(
                    r"^uploads\/(?:[\w\s-]+\/)*[\w\s-]+\.(pdf|docx|jpg|png|jpeg)$",
                    re.IGNORECASE,
                )

                if upload_pattern.match(user_message):
                    # Valid file format
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the vehicle registration Back side. Now, let's move on to: {next_question}",
                        }
                    else:
                        # Save responses and end the conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Invalid file format
                    return {
                        "response": "The file format seems incorrect. Please upload a valid document."
                    }

        elif (
            question
            == "Please upload a copy of the police report related to the incident"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Enhanced file path validation
                upload_pattern = re.compile(
                    r"^uploads\/(?:[\w\s-]+\/)*[\w\s-]+\.(pdf|docx|jpg|png|jpeg)$",
                    re.IGNORECASE,
                )

                if upload_pattern.match(user_message):
                    # Valid file format
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you for providing the Policy Report. Now, let's move on to: {next_questions}",
                            "options": options,
                        }

                    else:
                        # Save responses and end the conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Invalid file format
                    return {
                        "response": "The file format seems incorrect. Please upload a valid document."
                    }

        elif question == "Could you please provide your full name":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Prompt LLM to check if the input is a valid person name
                check_prompt = f"The user has responded with: '{user_message}'. Is this a valid person's  name? Respond with 'Yes' or 'No'."
                llm_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, an AI assistant specialized in insurance-related tasks. Your task is to determine if the input provided by the user is a valid person's  name. Make sure it is a valid  name for a person."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_person_name = llm_response.content.strip().lower() == "yes"

                if is_person_name:
                    # Translate name to English for storage
                    name_in_english = translate_to_english_for_storage(
                        user_message, user_language
                    )
                    responses[question] = name_in_english
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        response_message = f"Thank you for providing the name. Now, let's move on to: {next_question}"

                        # Format in user's language
                        if isinstance(next_question, dict):
                            return format_response_in_language(
                                f"Thank you for providing the name. Now, let's move on to: {next_question['question']}",
                                next_question.get("options", []),
                                user_language,
                            )
                        else:
                            return format_response_in_language(
                                response_message, [], user_language
                            )
                    else:
                        # If all questions are completed, save responses and end conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)

                        final_message = "Thank you for using Insura. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!"
                        result = format_response_in_language(
                            final_message, [], user_language
                        )
                        result["final_responses"] = responses
                        return result
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a person's name. Please assist them in {user_language}."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content=f"You are Insura, an AI assistant created by CloudSubset. Respond in {user_language}. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])

                    retry_question = translate_text(
                        f"Let's move back to: {question}", user_language
                    )
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": retry_question,
                    }

        elif question in [
            "Please provide us with your job title",
            "Please provide us with the member job title.",
        ]:
            return handle_job_title_question(
                question,
                user_message,
                conversation_state,
                questions,
                responses,
                user_language,
            )

        elif (
            question
            == "Now, let's move to the sponsor details, Could you let me know the sponsor's type?"
        ):
            valid_options = ["Employee", "Investors"]
            # Use generic multilingual handler
            return handle_option_validation_multilingual(
                user_message,
                valid_options,
                question,
                user_language,
                conversation_state,
                questions,
                responses,
                user_id,
            )

        elif question == "May I kindly ask you to tell me the currency?":
            valid_options = ["AED", "USD"]
            if user_message in valid_options:
                responses["question"] = user_message
                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        return {
                            "response": f"Thank you. Now, let's move on to: {next_question}"
                        }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid responses or unrelated queries
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }

        elif question == "Could you please tell me your monthly salary?":
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                try:
                    # Check if the input is a valid numeric value
                    salary = float(
                        user_message
                    )  # Attempt to convert the input to a float
                    if salary > 0:  # Ensure it's a positive number
                        # Store the salary
                        responses[question] = salary
                        conversation_state["current_question_index"] += 1

                        # Check if there are more questions
                        if conversation_state["current_question_index"] < len(
                            questions
                        ):
                            next_question = questions[
                                conversation_state["current_question_index"]
                            ]
                            if "options" in next_question:
                                options = ", ".join(next_question["options"])
                                next_questions = next_question["question"]

                                # Translate response to user's language
                                thank_you_msg = translate_text(
                                    f"Thank you! 😊 Now, let's move on to: {next_questions}",
                                    user_language,
                                )
                                translated_options = [
                                    translate_text(opt, user_language)
                                    for opt in next_question["options"]
                                ]

                                return {
                                    "response": thank_you_msg,
                                    "options": translated_options,
                                    "language": user_language,
                                    "language_code": get_language_code(user_language),
                                }
                            else:
                                # Translate response to user's language
                                thank_you_msg = translate_text(
                                    f"Thank you for providing your salary! 💰 Now, let's move on to: {next_question}",
                                    user_language,
                                )
                                return {
                                    "response": thank_you_msg,
                                    "language": user_language,
                                    "language_code": get_language_code(user_language),
                                }
                        else:
                            # If all questions are completed, save responses and end conversation
                            with open("user_responses.json", "w") as file:
                                json.dump(responses, file, indent=4)

                            # Translate completion message to user's language
                            completion_msg = translate_text(
                                "Thank you for using Insura! 🎉 Your request has been processed. If you have any further questions, feel free to ask. Have a great day! 😊",
                                user_language,
                            )
                            return {
                                "response": completion_msg,
                                "final_responses": responses,
                                "language": user_language,
                                "language_code": get_language_code(user_language),
                            }
                    else:
                        # Handle invalid input for non-positive values
                        error_msg = translate_text(
                            "The salary must be a positive number. Could you please re-enter your monthly salary in AED?",
                            user_language,
                        )
                        translated_question = translate_text(question, user_language)
                        return {
                            "response": error_msg,
                            "question": translated_question,
                            "language": user_language,
                            "language_code": get_language_code(user_language),
                        }
                except ValueError:
                    # If invalid input, use the general assistant
                    general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a valid monetary amount. Please assist them in {user_language}."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])

                    retry_msg = translate_text(
                        f"Let's move back to: {question}", user_language
                    )
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": retry_msg,
                        "language": user_language,
                        "language_code": get_language_code(user_language),
                    }

        elif question in [
            "Tell me your Emirate",
            "Tell me your Emirate sponsor located in?",
            "In which emirate would you prefer your vehicle to be repaired?",
            "Let's start with your motor insurance details. Select the city of registration",
        ]:
            return handle_emirate_question(
                question,
                user_message,
                conversation_state,
                questions,
                responses,
                user_language,
            )

        elif (
            question
            == "Which area you prefer for the vehicle repair? Please type the name of the area"
        ):
            if conversation_state["current_question_index"] == questions.index(
                question
            ):
                # Fetch the emirate from the previous response
                emirate = (
                    responses.get(
                        "In which emirate would you prefer your vehicle to be repaired?",
                        "",
                    )
                    .strip()
                    .lower()
                )

                # Prompt LLM for additional validation
                check_prompt = (
                    f"The user has responded with: '{user_message}'. Determine if this is a valid area within the emirate '{emirate}'. "
                    "Respond only with 'Yes' or 'No'."
                )
                llm_response = llm.invoke([
                    SystemMessage(
                        content=f"You are Insura, an AI assistant specialized in identifying the area based on the {emirate}. "
                        "Your task is to verify if the provided input is a valid area within the specified emirate."
                    ),
                    HumanMessage(content=check_prompt),
                ])
                is_valid_area = llm_response.content.strip().lower() == "yes"

                if is_valid_area:
                    # Store the area
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        if "options" in next_question:
                            options = ", ".join(next_question["options"])
                            next_questions = next_question["question"]
                            return {
                                "response": f"Thank you! Now, let's move on to: {next_questions}",
                                "options": options,
                            }
                        else:
                            return {
                                "response": f"Thank you for providing the area. Now, let's move on to: {next_question['question']}"
                            }
                    else:
                        # All questions completed
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                            "final_responses": responses,
                        }
                else:
                    # Use general assistant for invalid LLM validation
                    general_assistant_prompt = (
                        f"The user entered '{user_message}', which was not validated as a valid area within the emirate '{emirate}' by Insura. "
                        "Please assist them in correcting their input."
                    )
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, an AI assistant created by CloudSubset. "
                            "Your role is to assist users with their inquiries and guide them appropriately."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's try again: {question}\n",
                    }
        elif question == "Date of Birth (DOB)":
            return handle_date_question(
                question,
                user_message,
                responses,
                conversation_state,
                questions,
                user_language,
            )
        # For other free-text questions - Use multilingual evaluation

        evaluation_prompt = f"Is the user's response '{user_message}' correct for the question '{question}'? The user is responding in {user_language}. Answer 'yes' or 'no'."
        evaluation_response = llm.invoke([
            SystemMessage(
                content=f"You are evaluating user responses in {user_language}. Consider language variations and cultural context."
            ),
            HumanMessage(content=evaluation_prompt),
        ])
        evaluation = evaluation_response.content.strip().lower()

        if evaluation == "yes":
            # Translate to English for storage if needed
            english_response = translate_to_english_for_storage(
                user_message, user_language
            )
            responses[question] = english_response
            conversation_state["current_question_index"] += 1

            # Check if there are more questions
            if conversation_state["current_question_index"] < len(questions):
                next_question_data = questions[
                    conversation_state["current_question_index"]
                ]
                if isinstance(next_question_data, dict):
                    next_question = next_question_data["question"]
                    next_options = next_question_data.get("options", [])
                    response_message = f"Thank you! That was helpful. Now, let's move on to: {next_question}"

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )
                else:
                    next_question = next_question_data
                    response_message = f"Thank you! That was helpful. Now, let's move on to: {next_question}"

                    # Translate to user's language
                    return format_response_in_language(
                        response_message, [], user_language
                    )
            else:
                # All questions answered
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)

                final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
                result = format_response_in_language(final_message, [], user_language)
                result["final_responses"] = responses
                return result
        else:
            # Redirect to general assistant for help in user's language
            general_assistant_prompt = (
                f"User response: {user_message}. Please assist them in {user_language}."
            )
            general_assistant_response = llm.invoke([
                SystemMessage(
                    content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                ),
                HumanMessage(content=general_assistant_prompt),
            ])

            retry_question = translate_text(
                f"Let's Move back to {question}", user_language
            )
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": retry_question,
                "language": user_language,
                "language_code": get_language_code(user_language),
            }
    else:
        # Get user language for general queries
        user_language = conversation_state.get("preferred_language", "English")

        general_assistant_prompt = f"General query: {user_message}."
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])

        return {
            "response": f"{general_assistant_response.content.strip()}",
            "language": user_language,
            "language_code": get_language_code(user_language),
        }


async def clear_user_states_task():
    while True:
        await asyncio.sleep(86400)  # Sleep for 24 hours
        user_states.clear()
        print(f"User states cleared at {datetime.utcnow()}")


def start_clear_user_states_task():
    loop = asyncio.get_event_loop()
    loop.create_task(clear_user_states_task())


# Ensure the task starts when the module is imported
start_clear_user_states_task()
