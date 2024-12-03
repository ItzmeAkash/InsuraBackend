from utils.helper import get_user_name,valid_date_format
from langchain_groq.chat_models import ChatGroq

from langchain_core.messages import HumanMessage,SystemMessage
from models.user_input import UserInput
from random import choice
import json
import re
from dotenv import load_dotenv
load_dotenv()

import os
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

# Access questions in your logic
initial_questions = questions_data["initial_questions"]
medical_questions = questions_data["medical_questions"]
new_policy_questions = questions_data["new_policy_questions"]
existing_policy_questions = questions_data["existing_policy_questions"]
motor_insurance_questions = questions_data["motor_insurance_questions"]
car_questions = questions_data["car_questions"]
bike_questions = questions_data["bike_questions"]
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
            "welcome_shown": False
        }
        
    
    conversation_state = user_states[user_id]
    user_name = get_user_name(user_id)
    
    # Show welcome message with the first question if not already shown
    if not conversation_state["welcome_shown"]:
        conversation_state["welcome_shown"] = True
        first_question = initial_questions[0]
        next_options = first_question.get("options", [])
        greeting = choice(greeting_templates).format(user_name=user_name, first_question=first_question['question'])
        return {"response":greeting,"options": ', '.join(next_options)}


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
    elif current_flow == "new_policy":
        questions = new_policy_questions
    elif current_flow == "existing_policy":
        questions = existing_policy_questions
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
                    next_options = medical_questions[0].get("options", [])
                    
                    return {
                        "response": f"Great choice! {medical_questions[0]['question']}","options": ', '.join(next_options)
                    }
                elif user_message == "Purchase a Motor Insurance":
                    conversation_state["current_flow"] = "motor_insurance"
                    conversation_state["current_question_index"] = 0
                    next_options = motor_insurance_questions[0].get("options", [])
                    
                    return {
                        "response": f"Great choice! {motor_insurance_questions[0]['question']}","options": ', '.join(next_options)}
                elif user_message == "Purchase a Car Insurance":
                    conversation_state["current_flow"] = "car_questions"
                    conversation_state["current_question_index"] = 0
                    next_options = car_questions[0].get("options", [])
                    
                    return {
                        "response": f"Great choice! {car_questions[0]['question']}","options": ', '.join(next_options)}

                elif user_message == "Purchase a Bike Insurance":
                    conversation_state["current_flow"] = "bike_questions"
                    conversation_state["current_question_index"] = 0
                    return {
                        "response": f"Great choice! {bike_questions[0]}"
                    }                    
                elif user_message == "Purchase a new policy":
                    conversation_state["current_flow"] = "new_policy"
                    conversation_state["current_question_index"] = 0
                    return {
                        "response": f"Great choice! {new_policy_questions[0]}"
                    }
                elif user_message == "Renew my existing policy":
                    conversation_state["current_flow"] = "existing_policy"
                    conversation_state["current_question_index"] = 0
                    return {
                        "response": f"Great choice! {existing_policy_questions[0]}"
                    }
            else:
                return {
                    "response": f"Please choose a valid option from: {', '.join(options)}"
                }

        if question=="To whom are you purchasing this plan?":
            valid_options =[
                "Employee",
                "Dependents",
               "Small Investors",
             "Domestic Help",
              "4th Child Children above 18 years and Parents"
            ]
            if user_message in valid_options:
                responses[question]=user_message
                conversation_state["current_question_index"]+=1
                
                if conversation_state["current_question_index"]< len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                     "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}"
                     }
                else:
                    with open("user_responses.json", "w") as file:
                         json.dump(responses, file, indent=4)
                    return {
                     "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                     "final_responses": responses
                     }
            else:    
               general_assistant_prompt = f"user response: {user_message}. Please assist."
               general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
               return {
               "response": f"{general_assistant_response.content.strip()}",
               "question":f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}"
                }
    
        elif question == "May I have the sponsor's Email Address, please?":
            # Regex pattern for validating email address
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if re.match(email_pattern, user_message):
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses
                    }
            else:
                # Handle invalid input
                general_assistant_prompt = f"The user entered '{user_message}'. Please assist."
                general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's move back to: {question}"
                }    
                                                   
        elif question == "Please enter your Entry Date or Visa Change Status Date.":
            date_pattern = r"^\d{2}-\d{2}-\d{4}$"
            if re.match(date_pattern, user_message):
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}"
                    }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses
                    }
            else:
                # Handle invalid date format
                if re.search(r"[a-zA-Z]", user_message):  # Detect non-date queries
                    general_assistant_prompt = f"user response: {user_message}. Please assist."
                    
                    general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question":f"Let’s move back to: {question}"
                    }
                else:
                    return {
                        "response": "The date format should be DD-MM-YYYY. Please provide the correct format. For example: 25-12-2024."
                    }
       
        elif question == "Is accommodation provided to you?":
           valid_options = ["Yes", "No"]
           if user_message in valid_options:
               responses[question] = user_message
               conversation_state["current_question_index"] += 1

        # Check if there are more questions
               if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                return {
                "response": f"Thank you for your response. Now, let's move on to: {next_question}"
                }
               else:
                with open("user_responses.json", "w") as file:
                 json.dump(responses, file, indent=4)
                return {
                "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                "final_responses": responses
                 }
           else:
           # Handle invalid responses or unrelated queries
            general_assistant_prompt = f"user response: {user_message}. Please assist."
            general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
            return {
            "response": f"{general_assistant_response.content.strip()}",
             "question":f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}"
          }
            
        elif question == "Could you kindly provide me with the sponsor's Company Name":
            if conversation_state["current_question_index"] == questions.index(question):
                # Check if the input is a company name using LLM
                check_prompt = f"The user has responded with: '{user_message}'. Is this a valid company name? Respond with 'Yes' or 'No'."
                llm_response = llm.invoke([SystemMessage(content=f"Check {user_message} this message is a valid Company name not an general topic make sure check all the details "),HumanMessage(content=check_prompt)])
                is_company_name = llm_response.content.strip().lower() == "yes"

                if is_company_name:
                    # Store the company name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[conversation_state["current_question_index"]]
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question}"
                        }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                            "final_responses": responses
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
                    general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question":f"Let's Move Back {question}"
                    }

        elif question == "Is your car currently insured?":
            valid_options = ["Yes", "No"]
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "Yes":
                    # Check if the follow-up question is already in the list
                    follow_up_question = "Can you please tell me the year your insurance expired?"
                    if follow_up_question in questions:
                        # If the follow-up question exists, skip it and proceed
                        questions.remove(follow_up_question)
                    
                    # Proceed to the next predefined question
                    conversation_state["current_question_index"] += 1
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[conversation_state["current_question_index"]]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_question}"
                        }
                    else:
                        # All predefined questions have been answered
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses
                        }

                elif user_message == "No":
                    # Dynamically add the follow-up question if not already present
                    follow_up_question = "Can you please tell me the year your insurance expired?"
                    if follow_up_question not in questions:
                        responses[follow_up_question] = None
                        # Insert the new question immediately after the current one
                        questions.insert(conversation_state["current_question_index"] + 1, follow_up_question)
                    
                    # Move to the new follow-up question
                    conversation_state["current_question_index"] += 1
                    return {
                        "response": f"Thank you! Now, let's move on to: {follow_up_question}"
                    }
            else:
                return {
                    "response": "Invalid response. Please answer with 'Yes' or 'No'."
                }

        elif question == "Can you please tell me the year your insurance expired?":
            # Store the user-provided year
            if user_message.isdigit() and len(user_message) == 4:  # Ensure valid year format
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions to ask
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
                else:
                    # All questions have been answered
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!.",
                        "final_responses": responses
                    }
            else:
                
                # Redirect to general assistant for help
                general_assistant_prompt = f"User response: {user_message}. Please assist."
                general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurances assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question":f"Let's Move back to {question}"
                }
        
        elif question=="Are you suffering from any pre-existing or chronic conditions?":
            valid_options =[
                "Yes",
               "No"
            ]
            if user_message in valid_options:
                responses[question]=user_message
                conversation_state["current_question_index"]+=1
                
                if conversation_state["current_question_index"]< len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                     "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}",
                     
                     }
                else:
                    with open("user_responses.json", "w") as file:
                         json.dump(responses, file, indent=4)
                    return {
                     "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                     "final_responses": responses
                     }
            else:    
               general_assistant_prompt = f"user response: {user_message}. Please assist."
               general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
               return {
               "response": f"{general_assistant_response.content.strip()}",
               "question":f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}"
                }

        elif question == "What company does the sponsor work for?":
            if conversation_state["current_question_index"] == questions.index(question):
                # Check if the input is a company name using LLM
                check_prompt = f"The user has responded with: '{user_message}'. Is this a valid company name? Respond with 'Yes' or 'No'."
                llm_response = llm.invoke([SystemMessage(content=f"You are a friendly assistant working in Isuran's company department. Your primary task is to verify the user's input, which could be a company name. The input might include examples such as 'Fallout Private Limited' or 'Fallout Technologies'. Your role is to validate and identify whether the given input is a valid company name or needs clarification"),HumanMessage(content=check_prompt)])
                is_company_name = llm_response.content.strip().lower() == "yes"

                if is_company_name:
                    # Store the company name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[conversation_state["current_question_index"]]
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question}"
                        }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"User response: {user_message}. Please assist."
                    general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question":f"Let's Move back to {question}"
                    }

        elif question == "Have you been vaccinated for Covid-19?":
            valid_options = ["Yes","No"]
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "Yes":
                    # Dynamically add follow-up questions for dose dates
                    first_dose_question = "Can you please tell me the date of your first dose?"
                    second_dose_question = "Can you please tell me the date of your second dose?"

                    # Insert follow-up questions into the list if not already present
                    if first_dose_question not in questions:
                        responses[first_dose_question] = None
                        questions.insert(conversation_state["current_question_index"] + 1, first_dose_question)

                    if second_dose_question not in questions:
                        responses[second_dose_question] = None
                        questions.insert(conversation_state["current_question_index"] + 2, second_dose_question)

                    # Move to the next question
                    conversation_state["current_question_index"] += 1
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
                elif user_message == "No":
                    # Remove the questions about first and second doses if they exist
                    first_dose_question = "Please provide the date of the first dose"
                    second_dose_question = "Please provide the date of the second dose"

                    if first_dose_question in questions:
                        questions.remove(first_dose_question)
                    if second_dose_question in questions:
                        questions.remove(second_dose_question)

                    # Proceed to the next predefined question
                    conversation_state["current_question_index"] += 1
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[conversation_state["current_question_index"]]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_question["question"]}",
                            "options": f"{next_question['options']}"
                        }
                    else:
                        # All predefined questions have been answered
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses
                        }

        elif question == "Can you please tell me the date of your first dose?":
            # Validate and store the first dose date
            if valid_date_format(user_message):  # Replace with your date validation function
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions to ask
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question}"
                    }
                else:
                    # All questions have been answered
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                        "final_responses": responses
                    }
            else:
                return {
                    "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY or MM-DD-YYYY."
                }

        elif question == "Can you please tell me the date of your second dose?":
            # Validate and store the second dose date
            if valid_date_format(user_message):  # Replace with your date validation function
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions to ask
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"Thank you! Now, let's move on to: {next_question["question"]}",
                        "options":options
                    }
                else:
                    # All questions have been answered
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                        "final_responses": responses
                    }
            else:
                return {
                    "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY or MM-DD-YYYY."
                }
                
        elif question == "Your policy is up for renewal. Would you like to proceed with renewing it?":
            valid_options = ["Yes", "No"]
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "Yes":
                    # Proceed to the next predefined question
                    conversation_state["current_question_index"] += 1
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[conversation_state["current_question_index"]]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_question}"
                        }
                    else:
                        # All predefined questions have been answered
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses
                        }

                elif user_message == "No":
                    # Update the responses and return the final response
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for your response. Your request has been updated accordingly. If you need further assistance, feel free to ask.",
                        "final_responses": responses
                    }
            else:
                return {
                    "response": "Invalid response. Please answer with 'Yes' or 'No'."
                }

        elif question == "Now, let’s move to the sponsor details. Please provide the Sponsor Name?":
            if conversation_state["current_question_index"] == questions.index(question):
                # Prompt LLM to check if the input is a valid person name
                check_prompt = f"The user has responded with: '{user_message}'. Is this a valid person's name? Respond with 'Yes' or 'No'."
                llm_response = llm.invoke([
                    SystemMessage(content="You are Insura, an AI assistant specialized in insurance-related tasks. Your task is to determine if the input provided by the user is a valid person's name.Make sure it a valide name for a person"),
                    HumanMessage(content=check_prompt)
                ])
                is_person_name = llm_response.content.strip().lower() == "yes"

                if is_person_name:
                    # Store the person's name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[conversation_state["current_question_index"]]
                        return {
                            "response": f"Thank you for providing the sponsor's name. Now, let's move on to: {next_question}"
                        }
                    else:
                        # If all questions are completed, save responses and end conversation
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a person's name. Please assist."
                    general_assistant_response = llm.invoke([
                        SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately."),
                        HumanMessage(content=general_assistant_prompt)
                    ])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's move back to: {question}"
                    }

        elif question == "Which insurance company is your current policy with?":
            if conversation_state["current_question_index"] == questions.index(question):
                # Check if the input is a company name using LLM
                check_prompt = f"The user has responded with: '{user_message}'. Is this a valid company name? Respond with 'Yes' or 'No'."
                llm_response = llm.invoke([SystemMessage(content=f"You are a friendly assistant working in Isuran's company department. Your primary task is to verify the user's input, which could be a company name. The input might include examples such as 'Fallout Private Limited' or 'Fallout Technologies'. Your role is to validate and identify whether the given input is a valid company name or needs clarification"),HumanMessage(content=check_prompt)])
                is_company_name = llm_response.content.strip().lower() == "yes"

                if is_company_name:
                    # Store the company name
                    responses[question] = user_message
                    conversation_state["current_question_index"] += 1

                    # Check if there are more questions
                    if conversation_state["current_question_index"] < len(questions):
                        next_question = questions[conversation_state["current_question_index"]]
                        options = ", ".join(next_question["options"])
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question['question']}",
                            "options": options
                        }
                    else:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
                            "final_responses": responses
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
                    general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly AI assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question":f"Let's Move Back {question}"
                    }


   
       # For other free-text questions
        evaluation_prompt = f"Is the user's response '{user_message}' correct for the question '{question}'? Answer 'yes' or 'no'."
        evaluation_response = llm.invoke([HumanMessage(content=evaluation_prompt)])
        evaluation = evaluation_response.content.strip().lower()

        if evaluation == 'yes':
            # Store the valid response
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            # Check if there are more questions
            if conversation_state["current_question_index"] < len(questions):
                next_question_data = questions[conversation_state["current_question_index"]]
                if isinstance(next_question_data, dict):
                    next_question = next_question_data["question"]
                    next_options = next_question_data.get("options", [])
                    if next_options:
                        return {
                            "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}.",
                            "options": ', '.join(next_options)
                        }
                else:
                    next_question = next_question_data
                return {
                    "response": f"Thank you! That was helpful. Now, let's move on to: {next_question}"
                }
            else:
                # All questions answered
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses
                }
        else:
            # Redirect to general assistant for help
            general_assistant_prompt = f"User response: {user_message}. Please assist."
            general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question":f"Let's Move back to {question}"
            }
    else:
        
        general_assistant_prompt = f"General query: {user_message}."
        general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
        return {"response": f"{general_assistant_response.content.strip()}"}