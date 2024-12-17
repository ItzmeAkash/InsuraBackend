from datetime import datetime
from utils.helper import get_user_name, valid_date_format, valid_emirates_id
from langchain_groq.chat_models import ChatGroq
from fastapi import FastAPI, File, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage
from models.user_input import UserInput
from random import choice
import requests
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=groq_api_key,
    groq_proxy=None
)

user_states = {}

def load_questions(file_path="questions/questions.json"):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found. Please ensure the file exists.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")

# Load questions from the questions folder
questions_data = load_questions()

initial_questions = questions_data["initial_questions"]
motor_claim_questions = questions_data.get("motor_claim_questions", [])

def process_user_input(user_input: UserInput):
    user_id = user_input.user_id.strip()
    user_message = user_input.message.strip()

    if user_id not in user_states:
        user_states[user_id] = {
            "current_question_index": 0,
            "responses": {},
            "current_flow": "initial",
            "welcome_shown": False
        }

    conversation_state = user_states[user_id]
    responses = conversation_state["responses"]
    current_index = conversation_state["current_question_index"]
    user_name = get_user_name(user_id)

    if not conversation_state["welcome_shown"]:
        conversation_state["welcome_shown"] = True
        if current_index == 0:
            first_question = initial_questions[0]
            next_options = first_question.get("options", [])
            greeting = choice(questions_data["greeting_templates"]).format(
                user_name=user_name, first_question=first_question['question']
            )
            return {"response": greeting, "options": ', '.join(next_options)}
        else:
            pending_question = initial_questions[current_index]["question"]
            return {
                "response": f"Welcome back, {user_name}! Let's continue with: {pending_question}",
                "options": ', '.join(initial_questions[current_index].get("options", []))
            }

def process_motor_claim_questions(user_input: UserInput):
    user_id = user_input.user_id.strip()
    user_message = user_input.message.strip()

    if user_id not in user_states:
        user_states[user_id] = {
            "current_question_index": 0,
            "responses": {},
            "current_flow": "motor_claim",
            "welcome_shown": False
        }

    conversation_state = user_states[user_id]
    responses = conversation_state["responses"]
    current_index = conversation_state["current_question_index"]

    if current_index < len(motor_claim_questions):
        question_data = motor_claim_questions[current_index]
        question = question_data["question"]
        options = question_data.get("options", [])

        if options and user_message not in options:
            return {
                "response": f"Please choose a valid option from: {', '.join(options)}"
            }

        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(motor_claim_questions):
            next_question_data = motor_claim_questions[conversation_state["current_question_index"]]
            next_question = next_question_data["question"]
            next_options = next_question_data.get("options", [])
            return {
                "response": f"Thank you! Now, let's move on to: {next_question}",
                "options": ', '.join(next_options) if next_options else ""
            }
        else:
            with open("motor_claim_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                "final_responses": responses
            }
    else:
        return {
            "response": "No more questions available for the motor claim process."
        }

def edit_motor_claim_response(user_id, updated_response):
    try:
        if user_id in user_states:
            responses = user_states[user_id]["responses"]
            responses.update(updated_response)
            with open("motor_claim_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "The motor claim response has been successfully updated.",
                "updated_responses": responses
            }
        else:
            return {
                "response": "User not found. Unable to update motor claim response."
            }
    except Exception as e:
        return {
            "response": f"An error occurred while updating the motor claim response: {str(e)}"
        }
