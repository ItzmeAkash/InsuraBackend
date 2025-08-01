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
    handle_date_question,
    handle_emaf_document,
    handle_emirate_question,
    handle_emirate_upload_document,
    handle_emirate_upload_document_car_insurance,
    handle_gender,
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

# Updated
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    groq_proxy=None,
)


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
        }

    conversation_state = user_states[user_id]
    user_name = get_user_name(user_id)

    # Show welcome message with the first question if not already shown
    if not conversation_state["welcome_shown"]:
        conversation_state["welcome_shown"] = True
        first_question = initial_questions[0]
        next_options = first_question.get("options", [])
        greeting = choice(greeting_templates).format(
            user_name=user_name, first_question=first_question["question"]
        )
        return {"response": greeting, "options": ", ".join(next_options)}

    # Determine the current flow and questions
    current_flow = conversation_state["current_flow"]
    if current_flow == "initial":
        questions = initial_questions
    elif current_flow == "medical_insurance":
        questions = medical_questions
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
            # Validate user response against options
            if user_message in options:
                responses[question] = user_message
                if user_message == "Purchase a Medical Insurance":
                    conversation_state["current_flow"] = "medical_insurance"
                    conversation_state["current_question_index"] = 0

                    # Check if medical_questions is a list of dictionaries
                    if isinstance(medical_questions[0], dict):
                        next_options = medical_questions[0].get("options", [])
                        return {
                            "response": f"Great choice! {medical_questions[0]['question']}",
                            "options": ", ".join(next_options),
                        }
                    # If medical_questions is a list of strings
                    else:
                        return {
                            "response": f"Great choice! {medical_questions[0]}",
                        }
                elif user_message == "Purchase a Motor Insurance":
                    conversation_state["current_flow"] = "motor_insurance"
                    conversation_state["current_question_index"] = 0
                    next_options = motor_insurance_questions[0].get("options", [])

                    return {
                        "response": f"Great choice! {motor_insurance_questions[0]['question']}",
                        "options": ", ".join(next_options),
                    }
                elif user_message == "Purchase a Car Insurance":
                    conversation_state["current_flow"] = "car_questions"
                    conversation_state["current_question_index"] = 0
                    # Check if medical_questions is a list of dictionaries
                    if isinstance(medical_questions[0], dict):
                        next_options = medical_questions[0].get("options", [])
                        return {
                            "response": f"Great choice! {medical_questions[0]['question']}",
                            "options": ", ".join(next_options),
                        }
                    # If medical_questions is a list of strings
                    else:
                        return {
                            "response": f"Great choice! {medical_questions[0]}",
                        }

                elif user_message == "Purchase a Bike Insurance":
                    conversation_state["current_flow"] = "bike_questions"
                    conversation_state["current_question_index"] = 0
                    # Check if medical_questions is a list of dictionaries
                    if isinstance(medical_questions[0], dict):
                        next_options = medical_questions[0].get("options", [])
                        return {
                            "response": f"Great choice! {medical_questions[0]['question']}",
                            "options": ", ".join(next_options),
                        }
                    # If medical_questions is a list of strings
                    else:
                        return {
                            "response": f"Great choice! {medical_questions[0]}",
                        }
                # elif user_message == "Purchase a new policy":
                #     conversation_state["current_flow"] = "new_policy"
                #     conversation_state["current_question_index"] = 0
                #     return {
                #         "response": f"Great choice! {new_policy_questions[0]}"
                #     }
                elif user_message == "Renew my existing policy":
                    conversation_state["current_flow"] = "existing_policy"
                    conversation_state["current_question_index"] = 0
                    # Check if medical_questions is a list of dictionaries
                    if isinstance(medical_questions[0], dict):
                        next_options = medical_questions[0].get("options", [])
                        return {
                            "response": f"Great choice! {medical_questions[0]['question']}",
                            "options": ", ".join(next_options),
                        }
                    # If medical_questions is a list of strings
                    else:
                        return {
                            "response": f"Great choice! {medical_questions[0]}",
                        }
                elif user_message == "Claim a Motor Insurance":
                    conversation_state["current_flow"] = "motor_claim"
                    conversation_state["current_question_index"] = 0
                    # Check if medical_questions is a list of dictionaries
                    if isinstance(medical_questions[0], dict):
                        next_options = medical_questions[0].get("options", [])
                        return {
                            "response": f"Great choice! {medical_questions[0]['question']}",
                            "options": ", ".join(next_options),
                        }
                    # If medical_questions is a list of strings
                    else:
                        return {
                            "response": f"Great choice! {medical_questions[0]}",
                        }

        if question in [
            "Let's start with your Medical insurance details. Choose your Visa issued Emirate?",
            "Tell me your Emirate sponsor located in?",
        ]:
            return handle_visa_issued_emirate_question(
                user_message, conversation_state, questions, responses, question
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

            if user_message == "9567":
                conversation_state["current_question_index"] += 1
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        next_questions = next_question["question"]
                        return {
                            "response": f"Great choice!{next_questions}",
                            "options": options,
                        }
                    else:
                        return {"response": f"Great choice!{next_question}"}
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                return {
                    "response": "Incorrect passkey. Please try again. Please Enter Your PassKey"
                }

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
            return handle_type_plan_question(
                user_message, conversation_state, questions, responses, question
            )

        elif question == "Please Upload Your Document":
            try:
                document_data = json.loads(user_message)

                # Store the document data
                if isinstance(document_data, dict):
                    responses[question] = document_data
                    print(document_data)

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

                    return {
                        "response": "We couldn't detect the Back Page in the uploaded document",
                        "question": back_page_question["question"],
                    }
                elif not responses["front_page_received"]:
                    # Need to ask for front page
                    if front_page_question not in questions:
                        questions.insert(
                            conversation_state["current_question_index"] + 1,
                            front_page_question,
                        )
                        responses[front_page_question["question"]] = None

                    return {
                        "response": "We couldn't detect the Front Page in the uploaded document",
                        "question": front_page_question["question"],
                    }

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
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you for uploading the document. Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        next_question_text = (
                            next_question["question"]
                            if isinstance(next_question, dict)
                            else next_question
                        )
                        return {
                            "response": f"Thank you for uploading the document. Now, let's move on to: {next_question_text}"
                        }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }

            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()} \n\n",
                    "example": "Please ensure that the document is in JPEG format.",
                    "question": f"Let's try again: {question}",
                }
            except ValueError as e:
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()} \n\n",
                    "example": "Please ensure the document is in the correct format and try uploading again.",
                    "question": f"Let's try again: {question}",
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
                            options = ", ".join(next_question["options"])
                            next_questions = next_question["question"]
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_questions}",
                                "options": options,
                            }
                        else:
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_question}"
                            }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                            "final_responses": responses,
                        }
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure that the document is in JPEG format.",
                    "question": f"Let's try again: {question}",
                }
            except ValueError as e:
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure the document is in the correct format and try uploading again.",
                    "question": f"Let's try again: {question}",
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
                            options = ", ".join(next_question["options"])
                            next_questions = next_question["question"]
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_questions}",
                                "options": options,
                            }
                        else:
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_question}"
                            }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                            "final_responses": responses,
                        }
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure that the document is in JPEG format.",
                    "question": f"Let's try again: {question}",
                }
            except ValueError as e:
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure the document is in the correct format and try uploading again.",
                    "question": f"Let's try again: {question}",
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
                            options = ", ".join(next_question["options"])
                            next_questions = next_question["question"]
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_questions}",
                                "options": options,
                            }
                        else:
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_question}"
                            }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                            del user_states[user_id]
                        return {
                            "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                            "final_responses": responses,
                        }
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure that the document is in JPEG format.",
                    "question": f"Let's try again: {question}",
                }
            except ValueError as e:
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure the document is in the correct format and try uploading again.",
                    "question": f"Let's try again: {question}",
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
                            options = ", ".join(next_question["options"])
                            next_questions = next_question["question"]
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_questions}",
                                "options": options,
                            }
                        else:
                            return {
                                "response": f"Thank you for uploading the document. Now, let's move on to: {next_question}"
                            }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                            del user_states[user_id]
                        return {
                            "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                            "final_responses": responses,
                        }
                else:
                    raise ValueError("Please Upload Again")
            except json.JSONDecodeError:
                # Handle invalid JSON input
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(
                        content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                    ),
                    HumanMessage(content=general_assistant_prompt),
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure that the document is in JPEG format.",
                    "question": f"Let's try again: {question}",
                }
            except ValueError as e:
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "Please ensure the document is in the correct format and try uploading again.",
                    "question": f"Let's try again: {question}",
                }

        elif question in ["Please confirm this gender of"]:
            return handle_gender(
                user_message, conversation_state, questions, responses, question
            )

        elif (
            question
            == "Next, we need the details of the member. Would you like to upload their Emirates ID or manually enter the information?"
        ):
            return handle_emirate_upload_document(
                user_message, conversation_state, questions, responses, question
            )
        elif (
            question
            == "Next, we need the details of the car owner. Would you like to upload their Emirates ID or manually enter the information?"
        ):
            return handle_emirate_upload_document_car_insurance(
                user_message, conversation_state, questions, responses, question
            )

        elif question == "You Wish to Buy":
            valid_options = ["Comprehensive (Full Cover)", "Third Party"]
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
                    del user_states[user_id]
                    return {
                        "response": "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae.",
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
                user_message, conversation_state, questions, responses, question
            )

        elif question == "Please Confirm the marital status of":
            return handle_marital_status(
                user_message, conversation_state, questions, responses, question
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
                user_message, conversation_state, questions, responses, question
            )

        elif question == "Does your current policy have comprehensive cover?":
            return handle_yes_or_no(
                user_message, conversation_state, questions, responses, question
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
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
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
                                return {
                                    "response": f"Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please find the link below to view your quotation:",
                                    "link": f"https://insuranceclub.ae/customer_plan/{medical_detail_response}",
                                    "review_message": "If you are satisfied with Wehbe(Broker) services, please leave a review for sharing happiness to others!!😊",
                                    "review_link": "https://www.google.com/search?client=ms-android-samsung-ss&sca_esv=4eb717e6f42bf628&sxsrf=AHTn8zprabdPVFL3C2gXo4guY8besI3jqQ:1744004771562&q=wehbe+insurance+services+llc+reviews&uds=ABqPDvy-z0dcsfm2PY76_gjn-YWou9-AAVQ4iWjuLR6vmDV0vf3KpBMNjU5ZkaHGmSY0wBrWI3xO9O55WuDmXbDq6a3SqlwKf2NJ5xQAjebIw44UNEU3t4CpFvpLt9qFPlVh2F8Gfv8sMuXXSo2Qq0M_ZzbXbg2c323G_bE4tVi7Ue7d_sW0CrnycpJ1CvV-OyrWryZw_TeQ3gLGDgzUuHD04MpSHquYZaSQ0_mIHLWjnu7fu8c7nb6_aGDb_H1Q-86fD2VmWluYA5jxRkC9U2NsSwSSXV4FPW9w1Q2T_Wjt6koJvLgtikd66MqwYiJPX2x9MwLhoGYlpTbKtkJuHwE9eM6wQgieChskow6tJCVjQ75I315dT8n3tUtasGdBkprOlUK9ibPrYr9HqRz4AwzEQaxAq9_EDcsSG_XW0CHuqi2lRKHw592MlGlhjyQibXKSZJh-v3KW4wIVqa-2x0k1wfbZdpaO3BZaKYCacLOxwUKTnXPbQqDPLQDeYgDBwaTLvaCN221H&si=APYL9bvoDGWmsM6h2lfKzIb8LfQg_oNQyUOQgna9TyfQHAoqUvvaXjJhb-NHEJtDKiWdK3OqRhtZNP2EtNq6veOxTLUq88TEa2J8JiXE33-xY1b8ohiuDLBeOOGhuI1U6V4mDc9jmZkDoxLC9b6s6V8MAjPhY-EC_g%3D%3D&sa=X&sqi=2&ved=2ahUKEwi05JSHnMWMAxUw8bsIHRRCDd0Qk8gLegQIHxAB&ictx=1&stq=1&cs=0&lei=o2bzZ_SGIrDi7_UPlIS16A0#ebo=1",
                                }
                            else:
                                return {
                                    "response": "Thank you for sharing the details. We will inform Shafeeque Shanavas from Wehbe Insurance to assist you further with your enquiry. Please wait for further assistance. If you have any questions, please contact support@insuranceclub.ae."
                                }
                    except Exception as e:
                        return {
                            "response": f"An error occurred while fetching medical details: {str(e)}"
                        }

                    try:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Since you don't have an agent code, we will arrange a callback from the next available agent to assist you further. Thank you!",
                            "final_responses": responses,
                        }
                    except Exception as e:
                        return {
                            "response": f"An error occurred while saving your responses: {str(e)}"
                        }
            else:
                # Handle invalid advisor code or unrelated query
                # Here the `llm.invoke` and `HumanMessage` are placeholders, replace them with actual logic
                general_assistant_prompt = (
                    f"user response: {user_message}. Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
                ])

                return {
                    "response": (f"{general_assistant_response.content.strip()} \n\n"),
                    "example": "The Advisor code should be a 4-digit numeric value. Please enter a valid code",
                    "question": f"Let's try again: {question}",
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
        ]:
            # Regex pattern for validating email address
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if re.match(email_pattern, user_message):
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
                            "response": f"Thank you for providing the sponsor's email. Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        return {
                            "response": f"Thank you for providing the sponsor's email. Now, let's move on to: {next_question}"
                        }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid input
                general_assistant_prompt = (
                    f"The user entered '{user_message}'. Please assist."
                )
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
        elif question == "Do you have an Insurance Advisor code?":
            return handle_adiviosr_code(
                question, user_message, responses, conversation_state, questions
            )
        elif question == "Could you kindly share your relationship with the sponsor?":
            valid_options = [
                "Investor",
                "Employee",
                "Spouse",
                "Child",
                "4th Child",
                "Parent",
                "Domestic",
            ]

            if user_message in valid_options:
                # Update the Response
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[
                        conversation_state["current_question_index"]
                    ]

                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        member_name = responses.get(
                            "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                        )
                        return {
                            "response": f"Thank you for providing the relationship. let's proceed with: {next_questions}",
                            "options": options,
                        }
                    else:
                        member_name = responses.get(
                            "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                        )
                        return {
                            "response": f"Thank you for providing the relationship.  Now, let's address: {next_question}"
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
                    # Store the person's name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[
                            conversation_state["current_question_index"]
                        ]
                        return {
                            "response": f"Thank you for providing the name. Now, let's move on to: {next_question}"
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

        elif question in [
            "Please provide us with your job title",
            "Please provide us with the member job title.",
        ]:
            return handle_job_title_question(
                question, user_message, conversation_state, questions, responses
            )

        elif (
            question
            == "Now, let's move to the sponsor details, Could you let me know the sponsor's type?"
        ):
            valid_options = ["Employee", "Investors"]
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
                                return {
                                    "response": f"Thank you! Now, let's move on to: {next_questions}",
                                    "options": options,
                                }
                            else:
                                return {
                                    "response": f"Thank you for providing your salary. Now, let's move on to: {next_question}"
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
                        # Handle invalid input for non-positive values
                        return {
                            "response": "The salary must be a positive number. Could you please re-enter your monthly salary in AED?",
                            "question": question,
                        }
                except ValueError:
                    # If invalid input, use the general assistant
                    general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a valid monetary amount. Please assist."
                    general_assistant_response = llm.invoke([
                        SystemMessage(
                            content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                        ),
                        HumanMessage(content=general_assistant_prompt),
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's move back to: {question}",
                    }

        elif question in [
            "Tell me your Emirate",
            "Tell me your Emirate sponsor located in?",
            "In which emirate would you prefer your vehicle to be repaired?",
            "Let's start with your motor insurance details. Select the city of registration",
        ]:
            return handle_emirate_question(
                question, user_message, conversation_state, questions, responses
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
                question, user_message, responses, conversation_state, questions
            )
        # For other free-text questions

        evaluation_prompt = f"Is the user's response '{user_message}' correct for the question '{question}'? Answer 'yes' or 'no'."
        evaluation_response = llm.invoke([HumanMessage(content=evaluation_prompt)])
        evaluation = evaluation_response.content.strip().lower()

        if evaluation == "yes":
            # Store the valid response
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            # Check if there are more questions
            if conversation_state["current_question_index"] < len(questions):
                next_question_data = questions[
                    conversation_state["current_question_index"]
                ]
                if isinstance(next_question_data, dict):
                    next_question = next_question_data["question"]
                    next_options = next_question_data.get("options", [])
                    if next_options:
                        return {
                            "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}.",
                            "options": ", ".join(next_options),
                        }
                else:
                    next_question = next_question_data
                return {
                    "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}"
                }
            else:
                # All questions answered
                # save_file = "claim.json" if conversation_state["current_flow"] in ["existing_policy", "claim"] else "user_responses.json"

                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        else:
            # Redirect to general assistant for help
            general_assistant_prompt = f"User response: {user_message}. Please assist."
            general_assistant_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
                ),
                HumanMessage(content=general_assistant_prompt),
            ])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's Move back to {question}",
            }
    else:
        general_assistant_prompt = f"General query: {user_message}."
        general_assistant_response = llm.invoke([
            SystemMessage(
                content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])
        return {"response": f"{general_assistant_response.content.strip()}"}


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
