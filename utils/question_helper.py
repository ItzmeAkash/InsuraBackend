import json
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain_groq.chat_models import ChatGroq
import re
from datetime import datetime
from utils.helper import fetching_medical_detail, is_valid_country, is_valid_nationality
from utils.helper import valid_date_format
from fastapi import Request

llm = ChatGroq(
    model=os.getenv("LLM_MODEL"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    groq_proxy=None,
)


# Question To whom are you purchasing this plan?
def handle_purchasing_plan_question(
    user_message, conversation_state, questions, responses, question
):
    valid_options = [
        "Employee",
        "Dependents",
        "Small Investors",
        "Domestic Help",
        "4th Child",
        "Children above 18 years",
        "Parents",
    ]

    if user_message in valid_options:
        # Update the Response
        responses["To whom are you purchasing this plan?"] = user_message
        conversation_state["current_question_index"] += 1

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


# Question Visa issued Emirate?
def handle_visa_issued_emirate_question(
    user_message, conversation_state, questions, responses, question
):
    valid_options = [
        "Abudhabi",
        "Ajman",
        "Dubai",
        "Fujairah",
        "Ras Al Khaimah",
        "Sharjah",
        "Umm Al Quwain",
    ]
    if user_message in valid_options:
        responses["question"] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = ", ".join(next_question["options"])
                next_questions = next_question["question"]
                return {
                    "response": f"Thank you! Now, let's move on to: {next_questions}",
                    "options": options,
                }
            else:
                return {
                    "response": f"Thank you for providing the visa emirate Now, let's move on to: {next_question}"
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


# Question What type of plan are you looking for?
def handle_type_plan_question(
    user_message, conversation_state, questions, responses, question
):
    valid_options = [
        "Basic Plan",
        "Enhanced Plan",
        "Enhanced Plan Standalone",
        "Flexi Plan",
    ]

    if user_message in valid_options:
        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]

            if "options" in next_question:
                options = ", ".join(next_question["options"])
                next_question = questions[conversation_state["current_question_index"]]
                next_questions = next_question["question"]
                return {"response": f"Thank you!{next_questions}", "options": options}
            else:
                return {"response": f"Thank you.{next_question}"}
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


# Question Is accommodation provided to you?
def handle_yes_or_no(
    user_message,
    conversation_state,
    questions,
    responses,
    question,
    user_language="English",
):
    from services.llm_services import (
        format_response_in_language,
        validate_response_multilingual,
    )

    valid_options = ["Yes", "No"]

    # Use multilingual validation instead of direct string matching
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if validation_result["is_valid"]:
        # Store the English version
        matched_option = validation_result["matched_value"]
        responses[question] = matched_option
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = next_question["options"]
                next_questions = next_question["question"]
                response_message = f"Thank you! Now, let's move on to: {next_questions}"
                return format_response_in_language(
                    response_message, options, user_language
                )
            else:
                response_message = f"Thank you. Now, let's move on to: {next_question}"
                return format_response_in_language(response_message, [], user_language)
        else:
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
            result = format_response_in_language(final_message, [], user_language)
            result["final_responses"] = responses
            return result
    else:
        # Handle invalid responses or unrelated queries
        general_assistant_prompt = f"user response: {user_message}. Please assist."
        general_assistant_response = llm.invoke([
            HumanMessage(content=general_assistant_prompt)
        ])
        return {
            "response": f"{general_assistant_response.content.strip()}",
            "question": f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
        }


def handle_validate_name(
    question, user_message, conversation_state, questions, responses, is_valid_name
):
    if conversation_state["current_question_index"] == questions.index(question):
        # Convert user input to title case
        user_message = user_message.strip().title()

        # Validate user input using regex
        if not is_valid_name(user_message):
            general_assistant_prompt = (
                f"The user entered '{user_message}', which does not appear to be a valid name. "
                "Please assist them in providing a valid name."
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
                "question": question,
                "example": "Please ensure that the member's name is written in its full form (e.g., Shefeek PS)",
            }

        # Prompt LLM for additional validation
        check_prompt = (
            f"The user has responded with: '{user_message}'. Determine if this is a valid person's name. "
            "Respond only with 'Yes' or 'No'."
        )
        llm_response = llm.invoke([
            SystemMessage(
                content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                "Your task is to verify if the provided input is a valid person's name."
            ),
            HumanMessage(content=check_prompt),
        ])
        is_person_name = llm_response.content.strip().lower() == "yes"

        if is_person_name:
            # Store the name
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                return {
                    "response": f"Thank you for providing the name. Now, let's move on to: {next_question}"
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
                f"The user entered '{user_message}', which was not validated as a name by Insura. "
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
                "question": question,
                "example": "Please ensure that the member's name is written in its full form (e.g., Shefeek PS)",
            }


def handle_client_name_question(
    question, user_message, conversation_state, questions, responses, is_valid_name
):
    """
    Handles the 'May I have the Client Name, please?' question.
    This function directly accepts the user's input as the client name without validation.
    """
    if conversation_state["current_question_index"] == questions.index(question):
        # Convert user input to title case and store the client name
        user_message = user_message.strip().title()
        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if isinstance(next_question, dict) and "options" in next_question:
                options = ", ".join(next_question["options"])
                next_question_text = next_question["question"]
                return {
                    "response": f"Thank you for providing the client name. Now, let's move on to: {next_question_text}",
                    "options": options,
                }
            else:
                return {
                    "response": f"Thank you for providing the client name. Now, let's move on to: {next_question}"
                }
        else:
            # All questions completed
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                "final_responses": responses,
            }


def handle_gender(user_message, conversation_state, questions, responses, question):
    valid_options = ["Male", "Female"]
    # member_name = responses.get("Next, we need the details of the member for whom the policy is being purchased. Please provide Name")
    if user_message in valid_options:
        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = ", ".join(next_question["options"])
                next_questions = next_question["question"]

                return {
                    #    "response": f"Thank you,{next_questions} {member_name}",
                    "response": f"Thank you,Now let's move on to: {next_questions}",
                    "options": options,
                }
            return {
                # "response": f"Thank you,{next_question} {member_name}"
                "response": f"Thank you,{next_question}"
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
        general_assistant_prompt = f"user response: {user_message}. Please assist."
        general_assistant_response = llm.invoke([
            HumanMessage(content=general_assistant_prompt)
        ])
        return {
            "response": f"{general_assistant_response.content.strip()}",
            "question": f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
        }


def is_valid_policy_number(policy_number: str) -> bool:
    """
    Validates the format of a policy number.
    Example: Policy numbers must be alphanumeric and 6-12 characters long.
    """
    return bool(re.match(r"^[A-Za-z0-9]{6,12}$", policy_number))


def handle_policy_question(
    user_message, conversation_state, questions, responses, question
):
    """
    Handles the 'tell your policy number' question by validating and processing the user's input.
    """
    if question == question:
        if conversation_state["current_question_index"] == questions.index(question):
            # Prompt LLM for additional validation
            check_prompt = (
                f"The user has responded with: '{user_message}'. Determine if this is a valid policy number. "
                "Respond only with 'Yes' or 'No'."
            )
            llm_response = llm.invoke([
                SystemMessage(
                    content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                    "Your task is to verify if the provided input is a valid policy number."
                ),
                HumanMessage(content=check_prompt),
            ])
            is_valid_policy = llm_response.content.strip().lower() == "yes"

            if is_valid_policy:
                # Store the policy number
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
                            "response": f"Thank you for providing the Policy Number. Now, let's move on to: {next_question}"
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
                    f"The user entered '{user_message}', which was not validated as a policy number by Insura. "
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
                    "question": f"Let’s try again: {question}\n",
                }


# def handle_date_question(question, user_message, responses, conversation_state, questions):


#     if question == question:
#         # Debugging: Check the type of user_messa

#         # Ensure user_message is a string
#         if isinstance(user_message, str):
#             if valid_date_format(user_message):
#                 responses[question] = user_message
#                 conversation_state["current_question_index"] += 1

#                 # Check if there are more questions to ask
#                 if conversation_state["current_question_index"] < len(questions):
#                     next_question = questions[conversation_state["current_question_index"]]
#                     return {
#                         "response": f"Thank you! Now, let's move on to: {next_question}"
#                     }
#                 else:
#                     # All questions have been answered
#                     with open("user_responses.json", "w") as file:
#                         json.dump(responses, file, indent=4)
#                     return {
#                         "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
#                         "final_responses": responses
#                     }
#             else:
#                 # Handle invalid date format
#                 try:
#                     datetime.strptime(user_message, "%d/%m/%Y")
#                 except ValueError:
#                     return {
#                         "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY.",
#                         "question": question
#                     }

#                 general_assistant_prompt = f"User response: {user_message}. Please assist."
#                 general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
#                 return {
#                     "response": general_assistant_response.content.strip(),
#                     "question": f"Let’s try again: {question}"
#                 }
#         else:
#             return {
#                 "response": "Error: Expected a string input for the date. Please provide a valid date.",
#                 "question": question
#             }


def handle_company_name_question(
    question, user_message, conversation_state, questions, responses
):
    if question == question:
        if conversation_state["current_question_index"] == questions.index(question):
            # Check if the input is a company name using LLM
            check_prompt = f"The user has responded with: '{user_message}'. Is this a valid company name? Respond with 'Yes' or 'No'."
            llm_response = llm.invoke([
                SystemMessage(
                    content=f"Check {user_message} this message is a valid Company name not an general topic make sure check all the details "
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
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options,
                        }
                    else:
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question}"
                        }
                else:
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                        "final_responses": responses,
                    }
            else:
                # Handle invalid or unrelated input
                general_assistant_prompt = (
                    f"The user entered '{user_message}', . Please assist."
                )
                general_assistant_response = llm.invoke([
                    HumanMessage(content=general_assistant_prompt)
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


def handle_job_title_question(
    question,
    user_message,
    conversation_state,
    questions,
    responses,
    user_language="English",
):
    from services.llm_services import (
        format_response_in_language,
        translate_text,
    )

    if conversation_state["current_question_index"] == questions.index(question):
        # Prompt LLM to check if the input is a valid job title
        check_prompt = f"The user has responded with: '{user_message}'. Is this a valid job title? Respond with 'Yes' or 'No'."
        llm_response = llm.invoke([
            SystemMessage(
                content="You are Insura, an AI assistant specialized in insurance-related tasks. You are in a job title validation section you task is when any job title comes make sure that is a job title or not"
            ),
            HumanMessage(content=check_prompt),
        ])
        is_job_title = llm_response.content.strip().lower() == "yes"

        if is_job_title:
            # Store the job title
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            # Check if there are more questions
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if isinstance(next_question, dict):
                    next_question_text = next_question["question"]
                    next_options = next_question.get("options", [])
                    response_message = (
                        f"Thank you for providing your job title. {next_question_text}"
                    )
                    return format_response_in_language(
                        response_message, next_options, user_language
                    )
                else:
                    response_message = (
                        f"Thank you for providing your job title. {next_question}"
                    )
                    return format_response_in_language(
                        response_message, [], user_language
                    )
            else:
                # If all questions are completed, save responses and end conversation
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                final_message = "Thank you for using Insura. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!"
                result = format_response_in_language(final_message, [], user_language)
                result["final_responses"] = responses
                return result
        else:
            # Handle invalid or unrelated input in user's language
            general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a valid job title. Please assist them in {user_language}."
            general_assistant_response = llm.invoke([
                SystemMessage(
                    content=f"You are Insura, an AI assistant created by CloudSubset. Respond in {user_language}. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately"
                ),
                HumanMessage(content=general_assistant_prompt),
            ])

            # Translate the retry question to user's language
            retry_question = translate_text(
                f"Let's move back to: {question}", user_language
            )

            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": retry_question,
            }


def handle_emirate_question(
    question,
    user_message,
    conversation_state,
    questions,
    responses,
    user_language="English",
):
    from services.llm_services import (
        format_response_in_language,
        validate_response_multilingual,
        translate_text,
    )

    valid_options = [
        "Abudhabi",
        "Ajman",
        "Dubai",
        "Fujairah",
        "Ras Al Khaimah",
        "Sharjah",
        "Umm Al Quwain",
    ]

    # Use multilingual validation instead of direct string matching
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if validation_result["is_valid"]:
        # Store the English version
        matched_option = validation_result["matched_value"]
        responses[question] = matched_option
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = next_question["options"]
                next_questions = next_question["question"]
                response_message = f"Thank you! Now, let's move on to: {next_questions}"
                return format_response_in_language(
                    response_message, options, user_language
                )
            else:
                response_message = f"Thank you! Now, let's move on to: {next_question}"
                return format_response_in_language(response_message, [], user_language)
        else:
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
            result = format_response_in_language(final_message, [], user_language)
            result["final_responses"] = responses
            return result
    else:
        # Handle invalid responses or unrelated queries in user's language
        general_assistant_prompt = (
            f"user response: {user_message}. Please assist them in {user_language}."
        )
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])

        # Translate the retry question and options to user's language
        retry_question = translate_text(f"Let's try again: {question}", user_language)
        translated_options = [
            translate_text(opt, user_language) for opt in valid_options
        ]

        return {
            "response": f"{general_assistant_response.content.strip()}",
            "question": retry_question,
            "options": ", ".join(translated_options),
        }


def handle_nationality_question(
    user_message, question, conversation_state, questions, responses
):
    if conversation_state["current_question_index"] == questions.index(question):
        # First check if the input is a valid nationality using the is_valid_nationality function
        is_nationality = is_valid_nationality(user_message)

        if is_nationality:
            # Store the nationality
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            # Check if there are more questions
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                return {
                    "response": f"Thank you for providing the nationality. Now, let's move on to: {next_question}"
                }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        else:
            # Handle invalid or unrelated input
            general_assistant_prompt = (
                f"The user entered '{user_message}'. Please assist."
            )
            general_assistant_response = llm.invoke([
                HumanMessage(content=general_assistant_prompt)
            ])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's move back: {question}",
            }


# Todo def handle_marital_status(user_message, conversation_state, questions, responses, question):
#     valid_options = ["Single", "Married"]
#     first_question = {"question": "May I kindly ask if you are currently pregnant?", "options": ["Yes", "No"]}
#     second_question = {"question": "Have you recently been preparing or planning for pregnancy?", "options": ["Yes", "No"]}
#     third_question = "Could you please share the date of your last menstrual period?"
#     if user_message in valid_options:
#         responses[question] = user_message
#         conversation_state["current_question_index"] += 1

#         if user_message == "Married":
#             gender = responses.get("May I Know member's gender.Please?")
#             if gender == "Female":
#                 if first_question not in questions:
#                     responses[first_question["question"]] = None
#                     questions.insert(conversation_state["current_question_index"], first_question)

#                 if second_question not in questions:
#                     responses[second_question["question"]] = None
#                     questions.insert(conversation_state["current_question_index"] + 1, second_question)

#                 if third_question not in questions:
#                     responses[third_question] = None
#                     questions.insert(conversation_state["current_question_index"] + 2, third_question)

#                 next_question = questions[conversation_state["current_question_index"]]

#                 if "options" in next_question:
#                     options = ", ".join(next_question["options"])
#                     return {
#                         "response": f"Thank you! Now, let's move on to: {next_question['question']}",
#                         "options": options
#                     }
#                 else:
#                     return {
#                         "response": f"Thank you for providing the Marital Status. Now, let's move on to: {next_question}"
#                     }
#         else:

#             if first_question in questions:
#                 questions.remove(first_question)
#             if second_question in questions:
#                 questions.remove(second_question)
#             if third_question in questions:
#                 questions.remove(third_question)

#             # Proceed to the next predefined question
#             conversation_state["current_question_index"] += 1
#             if conversation_state["current_question_index"] < len(questions):
#                 next_question = questions[conversation_state["current_question_index"]]


#                 return {
#                     "response": f"Thank you for your response. Now, let's move on to: {next_question}",

#                 }
#             else:
#                 # All predefined questions have been answered
#                 with open("user_responses.json", "w") as file:
#                     json.dump(responses, file, indent=4)
#                 return {
#                     "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
#                     "final_responses": responses
#                 }
#     else:
#         general_assistant_prompt = f"The user entered '{user_message}'. Please assist."
#         general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
#         return {
#             "response": f"{general_assistant_response.content.strip()}",
#             "question": f"Let's move back: {question}"
#         }

# Todo(female married condition) def handle_marital_status(user_message, conversation_state, questions, responses, question):
#     valid_options = ["Single", "Married"]
#     first_question = {"question": "May I kindly ask if you are currently pregnant?", "options": ["Yes", "No"]}
#     second_question = {"question": "Have you recently been preparing or planning for pregnancy?", "options": ["Yes", "No"]}
#     third_question = "Could you please share the date of your last menstrual period?"

#     if user_message in valid_options:
#         responses[question] = user_message
#         conversation_state["current_question_index"] += 1

#         if user_message == "Married":
#             gender = responses.get("May I Know member's gender.Please?")
#             if gender == "Female":
#                 if first_question not in questions:
#                     responses[first_question["question"]] = None
#                     questions.insert(conversation_state["current_question_index"], first_question)

#                 if second_question not in questions:
#                     responses[second_question["question"]] = None
#                     questions.insert(conversation_state["current_question_index"] + 1, second_question)

#                 if third_question not in questions:
#                     responses[third_question] = None
#                     questions.insert(conversation_state["current_question_index"] + 2, third_question)

#                 next_question = questions[conversation_state["current_question_index"]]

#                 if "options" in next_question:
#                     options = ", ".join(next_question["options"])
#                     return {
#                         "response": f"Thank you! Now, let's move on to: {next_question['question']}",
#                         "options": options
#                     }
#                 else:
#                     return {
#                         "response": f"Thank you for providing the Marital Status. Now, let's move on to: {next_question}"
#                     }
#             else:
#                 # Proceed to the next predefined question
#                 if conversation_state["current_question_index"] < len(questions):
#                     next_question = questions[conversation_state["current_question_index"]]

#                     if "options" in next_question:
#                         options = ", ".join(next_question["options"])
#                         return {
#                             "response": f"Thank you! Now, let's move on to: {next_question['question']}",
#                             "options": options
#                         }
#                     else:
#                         return {
#                             "response": f"Thank you for providing the Marital Status. Now, let's move on to: {next_question}"
#                         }
#                 else:
#                     # All predefined questions have been answered
#                     with open("user_responses.json", "w") as file:
#                         json.dump(responses, file, indent=4)
#                     return {
#                         "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
#                         "final_responses": responses
#                     }
#         else:
#             if first_question in questions:
#                 questions.remove(first_question)
#             if second_question in questions:
#                 questions.remove(second_question)
#             if third_question in questions:
#                 questions.remove(third_question)

#             # Proceed to the next predefined question
#             if conversation_state["current_question_index"] < len(questions):
#                 next_question = questions[conversation_state["current_question_index"]]
#                 next_question = questions[conversation_state["current_question_index"]]

#                 if "options" in next_question:
#                     options = ", ".join(next_question["options"])
#                     return {
#                         "response": f"Thank you! Now, let's move on to: {next_question['question']}",
#                         "options": options
#                     }
#                 else:
#                     return {
#                         "response": f"Thank you for providing the Marital Status. Now, let's move on to: {next_question}"
#                     }
#             else:
#                 # All predefined questions have been answered
#                 with open("user_responses.json", "w") as file:
#                     json.dump(responses, file, indent=4)
#                 return {
#                     "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
#                     "final_responses": responses
#                 }
#     else:
#         general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
#         general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
#         next_question = questions[conversation_state["current_question_index"]]
#         if "options" in next_question:
#             next_question = next_question['question']
#             options = ", ".join(next_question["options"])
#             return {
#                 "response": f"{general_assistant_response.content.strip()}",
#                 "question": f"Let's Move Back {next_question}",
#                 "options": options
#             }


#         else:
#                 return {
#                 "response": f"{general_assistant_response.content.strip()}",
#                 "question": f"Let's Move Back {question}",
#                 }


def handle_marital_status(
    user_message, conversation_state, questions, responses, question
):
    valid_options = ["Single", "Married"]
    if user_message in valid_options:
        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = ", ".join(next_question["options"])
                # next_questions = next_question["question"]
                member_name = responses.get(
                    "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                )
                next_questions = next_question["question"].replace(
                    "your", f"{member_name}'s"
                )

                return {
                    "response": f"Thank you Next, let's discuss. {next_questions}",
                    "options": options,
                }
            return {"response": f"Thank you Next, let's discuss. {next_question}"}
        else:
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                "final_responses": responses,
            }
    else:
        # Handle invalid responses or unrelated queries
        general_assistant_prompt = f"user response: {user_message}. Please assist."
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


def handle_pregant(user_message, conversation_state, questions, responses, question):
    valid_options = ["Yes", "No"]
    if user_message in valid_options:
        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
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
        general_assistant_prompt = f"user response: {user_message}. Please assist."
        general_assistant_response = llm.invoke([
            HumanMessage(content=general_assistant_prompt)
        ])
        return {
            "response": f"{general_assistant_response.content.strip()}",
            "question": f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
        }


def handle_sposor_type(
    user_message, conversation_state, questions, responses, question
):
    valid_options = ["Employee", "Investors"]
    if user_message in valid_options:
        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
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


def valid_date_format(date_str):
    # Example date validation function
    for fmt in ("%d/%m/%Y", "%m-%d-%Y"):
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue
    return False


# def handle_date_question(question, user_message, responses, conversation_state, questions):
#     if question == question:
#         # Validate and store the date
#         if valid_date_format(user_message):
#             responses[question] = user_message
#             conversation_state["current_question_index"] += 1

#             # Check if there are more questions to ask
#             if conversation_state["current_question_index"] < len(questions):
#                 next_question = questions[conversation_state["current_question_index"]]
#                 options = ", ".join(next_question["options"])
#                 next_questions = next_question["question"]

#                 return {
#                     "response": f"Thank you! Now, let's move on to: {next_questions}",
#                     "options": options
#                 }
#             else:
#                 # All questions have been answered
#                 try:
#                     with open("user_responses.json", "w") as file:
#                         json.dump(responses, file, indent=4)
#                 except Exception as e:
#                     return {
#                         "response": f"An error occurred while saving responses: {str(e)}"
#                     }
#                 return {
#                     "response": "Thank you for using Insuar. Your request has been processed. If you have any further questions, feel free to ask. Have a great day!",
#                     "final_responses": responses
#                 }
#         else:
#             return {
#                 "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY or MM-DD-YYYY."
#             }
#     else:
#         return {
#             "response": "Unexpected question."
#         }


def handle_country_question(
    user_message, question, conversation_state, questions, responses
):
    if conversation_state["current_question_index"] == questions.index(question):
        is_country = is_valid_country(user_message)

        if is_country:
            # Store the nationality
            responses[question] = user_message
            conversation_state["current_question_index"] += 1

            # Check if there are more questions
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                return {
                    "response": f"Thank you for providing the Country. Now, let's move on to: {next_question}"
                }
            else:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                return {
                    "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                    "final_responses": responses,
                }
        else:
            # Handle invalid or unrelated input
            general_assistant_prompt = (
                f"The user entered '{user_message}'. Please assist."
            )
            general_assistant_response = llm.invoke([
                HumanMessage(content=general_assistant_prompt)
            ])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's move back: {question}",
            }


def handle_individual_sma_choice(
    user_message,
    conversation_state,
    questions,
    responses,
    question,
    user_language="English",
):
    from services.llm_services import (
        individual_questions,
        sma_questions,
        format_response_in_language,
        validate_response_multilingual,
        translate_text,
    )

    valid_options = ["Individual", "SME"]

    # Use multilingual validation instead of direct string matching
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if validation_result["is_valid"]:
        # Store the English version
        matched_option = validation_result["matched_value"]
        responses[question] = matched_option
        conversation_state["current_question_index"] += 1

        if matched_option == "Individual":
            conversation_state["current_flow"] = "individual"
            conversation_state["current_question_index"] = 0

            if isinstance(individual_questions[0], dict):
                next_options = individual_questions[0].get("options", [])
                response_message = (
                    f"Great choice! {individual_questions[0]['question']}"
                )
                # Translate to user's language
                return format_response_in_language(
                    response_message, next_options, user_language
                )
            else:
                response_message = f"Great choice! {individual_questions[0]}"
                return format_response_in_language(response_message, [], user_language)
        elif matched_option == "SME":
            conversation_state["current_flow"] = "sma"
            conversation_state["current_question_index"] = 0

            if isinstance(sma_questions[0], dict):
                next_options = sma_questions[0].get("options", [])
                response_message = f"Great choice! {sma_questions[0]['question']}"
                # Translate to user's language
                return format_response_in_language(
                    response_message, next_options, user_language
                )
            else:
                response_message = f"Great choice! {sma_questions[0]}"
                return format_response_in_language(response_message, [], user_language)
    else:
        # Handle invalid responses or unrelated queries in user's language
        general_assistant_prompt = (
            f"user response: {user_message}. Please assist them in {user_language}."
        )
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])

        # Translate the retry question and options to user's language
        retry_question = translate_text(f"Let's try again: {question}", user_language)
        translated_options = [
            translate_text(opt, user_language) for opt in valid_options
        ]

        return {
            "response": f"{general_assistant_response.content.strip()}",
            "question": retry_question,
            "options": ", ".join(translated_options),
        }


def handle_what_would_you_do_today_question(
    user_message,
    conversation_state,
    questions,
    responses,
    question,
    user_language="English",
):
    from services.llm_services import (
        format_response_in_language,
        validate_response_multilingual,
    )

    valid_options = [
        "Purchase a Medical Insurance",
        "Purchase a Motor Insurance",
        "Claim a Motor Insurance",
    ]

    # Use multilingual validation instead of direct string matching
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if validation_result["is_valid"]:
        # Store the English version
        matched_option = validation_result["matched_value"]
        responses[question] = matched_option
        conversation_state["current_question_index"] += 1

        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = next_question["options"]
                next_questions = next_question["question"]
                response_message = f"Thank you! Now, let's move on to: {next_questions}"
                return format_response_in_language(
                    response_message, options, user_language
                )
            else:
                response_message = f"Thank you for providing the plan. Now, let's move on to: {next_question}"
                return format_response_in_language(response_message, [], user_language)
        else:
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            final_message = "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask."
            result = format_response_in_language(final_message, [], user_language)
            result["final_responses"] = responses
            return result
    else:
        # TODO
        # Handle invalid responses or unrelated queries
        general_assistant_prompt = f"user response: {user_message}. Please assist."
        general_assistant_response = llm.invoke([
            SystemMessage(
                content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])
        next_question = questions[conversation_state["current_question_index"]]
        if "options" in next_question:
            options = ", ".join(next_question["options"])
            print(question)
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let’s try again: {question}",
                "options": options,
            }
        else:
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
            }


# Todo
def handle_date_question(
    question,
    user_message,
    responses,
    conversation_state,
    questions,
    user_language="English",
):
    from services.llm_services import translate_text

    # Also accept numbers as valid input (in addition to date format)
    is_valid_input = valid_date_format(user_message) or user_message.strip().isdigit()

    if is_valid_input:
        responses[question] = user_message
        conversation_state["current_question_index"] += 1

        # Move to the next question or finalize responses
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if isinstance(next_question, dict) and "options" in next_question:
                next_questions = next_question["question"]
                member_name = responses.get(
                    "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
                )

                # Translate response to user's language
                response_text = translate_text(
                    f"Thanks! 📅 Let's continue. {next_questions} {member_name}. given below",
                    user_language,
                )
                translated_options = [
                    translate_text(opt, user_language)
                    for opt in next_question["options"]
                ]

                return {
                    "response": response_text,
                    "options": ", ".join(translated_options),
                    "language": user_language,
                }
            else:
                next_question_text = (
                    next_question
                    if isinstance(next_question, str)
                    else next_question.get("question", "")
                )
                response_text = translate_text(
                    f"Thanks! 😊 Let's continue with {next_question_text}",
                    user_language,
                )
                return {
                    "response": response_text,
                    "language": user_language,
                }
        else:
            # All questions answered
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)

                completion_msg = translate_text(
                    "Thank you for using Insura! 🎉 Your responses have been recorded. Feel free to ask any other questions. Have a great day!",
                    user_language,
                )
                return {
                    "response": completion_msg,
                    "final_responses": responses,
                    "language": user_language,
                }
            except Exception as e:
                error_msg = translate_text(
                    f"An error occurred while saving your responses: {str(e)}",
                    user_language,
                )
                return {
                    "response": error_msg,
                    "language": user_language,
                }
    else:
        # Handle invalid date or unrelated query
        general_assistant_prompt = (
            f"user response: {user_message}. Please assist them in {user_language}."
        )
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])

        example_msg = translate_text(
            "Please provide the date in the format DD/MM/YYYY.", user_language
        )
        retry_msg = translate_text(f"Let's try again: {question}", user_language)

        return {
            "response": (f"{general_assistant_response.content.strip()} \n\n"),
            "example": example_msg,
            "question": retry_msg,
            "language": user_language,
        }


def handle_adiviosr_code(
    question,
    user_message,
    responses,
    conversation_state,
    questions,
    user_language="English",
):
    from services.llm_services import (
        validate_response_multilingual,
        translate_text,
    )

    valid_options = ["Yes", "No"]
    first_question = "Please enter your Insurance Advisor code for assigning your enquiry for further assistance"

    # Use multilingual validation
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if validation_result["is_valid"]:
        # Store the English value
        responses[question] = validation_result["matched_value"]

        if validation_result["matched_value"] == "Yes":
            # Dynamically add follow-up questions
            # Insert follow-up questions into the list if not already present
            if first_question not in questions:
                responses[first_question] = None
                questions.insert(
                    conversation_state["current_question_index"] + 1, first_question
                )

            # Move to the next question
            conversation_state["current_question_index"] += 1
            next_question = questions[conversation_state["current_question_index"]]

            # Translate response to user's language
            response_msg = translate_text(
                f"Thank you for the responses! 👍 Now, {next_question}", user_language
            )
            return {
                "response": response_msg,
                "language": user_language,
            }
        elif validation_result["matched_value"] == "No":
            # Remove the advisor code question if it exists
            if first_question in questions:
                questions.remove(first_question)

            # Proceed to the next predefined question
            conversation_state["current_question_index"] += 1
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                next_questions = (
                    next_question["question"]
                    if isinstance(next_question, dict)
                    else next_question
                )

                # Translate response to user's language
                response_text = translate_text(
                    f"Thank you for your response! 😊 Now, let's move on to: {next_questions}",
                    user_language,
                )

                # Handle options if they exist
                if isinstance(next_question, dict) and "options" in next_question:
                    translated_options = [
                        translate_text(opt, user_language)
                        for opt in next_question["options"]
                    ]
                    return {
                        "response": response_text,
                        "options": ", ".join(translated_options),
                        "language": user_language,
                    }
                return {
                    "response": response_text,
                    "language": user_language,
                }
            else:
                # All predefined questions have been answered
                if responses.get("Do you have an Insurance Advisor code?") == "Yes":
                    medical_deatil_response = fetching_medical_detail(responses)
                    completion_msg = translate_text(
                        f"Thank you for sharing the details! 🎉 We will inform the agent to assist you further with your enquiry. Please find the link below to view your quotation: {medical_deatil_response}",
                        user_language,
                    )
                    return {
                        "response": completion_msg,
                        "language": user_language,
                    }

                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)

                no_agent_msg = translate_text(
                    "Since you don't have an agent code, we will arrange a callback from the next available agent to assist you further. Thank you! 📞",
                    user_language,
                )
                return {
                    "response": no_agent_msg,
                    "final_responses": responses,
                    "language": user_language,
                }
    else:
        general_assistant_prompt = (
            f"user response: {user_message}. Please assist them in {user_language}."
        )
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])

        retry_question = translate_text(f"Let's try again: {question}", user_language)
        return {
            "response": (f"{general_assistant_response.content.strip()} \n\n"),
            "question": retry_question,
            "language": user_language,
        }


def handle_emirate_upload_document(
    user_message,
    conversation_state,
    questions,
    responses,
    question,
    user_language="English",
):
    from services.llm_services import (
        validate_response_multilingual,
        translate_text,
    )

    valid_options = ["Yes", "No"]

    # Define all questions upfront for better maintainability
    QUESTIONS = {
        "upload": {"question": "Please Upload Your Document"},
        "name": {
            "question": "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
        },
        "dob": "Date of Birth (DOB)",  # Simple string format as requested
        "gender": {
            "question": "Please confirm this gender of",
            "options": ["Male", "Female"],
        },
    }

    # Use multilingual validation instead of direct string matching
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if not validation_result["is_valid"]:
        general_assistant_prompt = (
            f"user response: {user_message}. Please assist them in {user_language}."
        )
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])
        next_question = questions[conversation_state["current_question_index"]]
        if isinstance(next_question, dict) and "options" in next_question:
            options = next_question["options"]
            retry_question = translate_text(
                f"Let's try again: {next_question['question']}", user_language
            )
            translated_options = [translate_text(opt, user_language) for opt in options]
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": retry_question,
                "options": ", ".join(translated_options),
                "language": user_language,
            }
        else:
            question_text = (
                question["question"] if isinstance(question, dict) else question
            )
            retry_question = translate_text(
                f"Let's try again: {question_text}", user_language
            )
            translated_options = [
                translate_text(opt, user_language) for opt in valid_options
            ]
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"{retry_question}\nPlease choose from the following options: {', '.join(translated_options)}",
                "language": user_language,
            }

    # Store the response - handle both string and dict question formats
    # Use the matched English value from validation result
    question_text = question["question"] if isinstance(question, dict) else question
    responses[question_text] = validation_result["matched_value"]

    # Handle "Yes" path - use the matched English value
    if validation_result["matched_value"] == "Yes":
        upload_question = QUESTIONS["upload"]
        name_question = QUESTIONS["name"]
        dob_question = QUESTIONS["dob"]
        gender_question = QUESTIONS["gender"]

        # Remove other questions if they exist
        for q in [name_question, dob_question, gender_question]:
            if q in questions:
                questions.remove(q)

        if upload_question not in questions:
            questions.insert(
                conversation_state["current_question_index"] + 1, upload_question
            )
            responses[upload_question["question"]] = None

        conversation_state["current_question_index"] += 1
        next_question = questions[conversation_state["current_question_index"]]
        next_question_text = (
            next_question["question"]
            if isinstance(next_question, dict)
            else next_question
        )

        # Translate response to user's language
        response_msg = translate_text(
            f"Thank you for the responses! 📄 Now, {next_question_text}", user_language
        )
        return {
            "response": response_msg,
            "language": user_language,
        }

    # Handle "No" path - use the matched English value
    elif validation_result["matched_value"] == "No":
        # Remove upload document question if it exists
        upload_question = QUESTIONS["upload"]
        if upload_question in questions:
            questions.remove(upload_question)

        # Add all required questions in sequence
        next_index = conversation_state["current_question_index"] + 1
        for key in ["name", "dob", "gender"]:
            question_dict = QUESTIONS[key]
            if question_dict not in questions:
                questions.insert(next_index, question_dict)
                responses[
                    question_dict["question"]
                    if isinstance(question_dict, dict)
                    else question_dict
                ] = None
                next_index += 1

        # Move to next question
        conversation_state["current_question_index"] += 1

        # Check if there are more questions
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            next_question_text = (
                next_question["question"]
                if isinstance(next_question, dict)
                else next_question
            )

            # Translate response to user's language
            response_text = translate_text(
                f"Thank you for your response! 😊 Now, let's move on to: {next_question_text}",
                user_language,
            )

            # Add options if they exist
            if isinstance(next_question, dict) and "options" in next_question:
                translated_options = [
                    translate_text(opt, user_language)
                    for opt in next_question["options"]
                ]
                return {
                    "response": response_text,
                    "options": ", ".join(translated_options),
                    "language": user_language,
                }
            return {
                "response": response_text,
                "language": user_language,
            }

        # Handle end of questions
        else:
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


# def handle_emaf_document(question, user_message, responses, conversation_state, questions):
#     required_questions = [
#         "Tell me your document name",
#         "Phone number",
#         "Company name"
#     ]

#     # Store the current response
#     responses[question] = user_message

#     # If it's the first time, add required questions in order
#     if not conversation_state.get("initialized"):
#         for req_question in required_questions:
#             if req_question not in questions:
#                 responses[req_question] = None
#                 questions.append(req_question)
#         conversation_state["initialized"] = True

#     # Move to next question
#     conversation_state["current_question_index"] += 1

#     # Check if there are more questions to ask
#     if conversation_state["current_question_index"] < len(questions):
#         next_question = questions[conversation_state["current_question_index"]]

#         # Handle questions with options
#         if isinstance(next_question, dict) and "options" in next_question:
#             options = ", ".join(next_question["options"])
#             return {
#                 "response": f"Thank you for your response. Now, let's move on to: {next_question['question']}",
#                 "options": options
#             }
#         # Handle regular questions
#         else:
#             return {
#                 "response": f"Thank you for the responses! Now, {next_question}"
#             }

#     # Handle completion of all questions
#     if responses.get("Do you have an Insurance Advisor code?") == "Yes":
#         medical_detail_response = fetching_medical_detail(responses)
#         return {
#             "response": f"Thank you for sharing the details. We will inform (Agent Name) to assist you further with your enquiry. Please find the link below to view your quotation: {medical_detail_response}"
#         }

#     # Save responses if no agent code is provided
#     with open("emaf_document.json", "w") as file:
#         json.dump(responses, file, indent=4)
#     return {
#         "response": "Since you don't have an agent code, we will arrange a callback from the next available agent to assist you further. Thank you!",
#         "final_responses": responses
#     }


def handle_emaf_document(
    question, user_message, responses, conversation_state, questions
):
    QUESTIONS = {
        "name": {
            "question": "May I know your name, please?",
        },
        "phone": {"question": "May I kindly ask for your phone number, please?"},
        "company": {
            "question": "Could you kindly confirm the name of your insurance company, please?",
            "options": [
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
            ],
        },
    }

    # Store the current response
    question_text = question["question"] if isinstance(question, dict) else question
    responses[question_text] = user_message

    # Add all required questions in sequence
    next_index = conversation_state["current_question_index"] + 1
    for key in ["name", "phone", "company"]:
        question_dict = QUESTIONS[key]
        if question_dict not in questions:
            questions.insert(next_index, question_dict)
            responses[
                question_dict["question"]
                if isinstance(question_dict, dict)
                else question_dict
            ] = None
            next_index += 1

    # Move to next question
    conversation_state["current_question_index"] += 1

    # Check if there are more questions
    if conversation_state["current_question_index"] < len(questions):
        next_question = questions[conversation_state["current_question_index"]]
        next_question_text = (
            next_question["question"]
            if isinstance(next_question, dict)
            else next_question
        )
        response_text = (
            f"Thank you for your response! I appreciate it. Now, {next_question_text}"
        )

        # Add options if they exist
        if isinstance(next_question, dict) and "options" in next_question:
            return {
                "response": response_text,
                "options": ", ".join(next_question["options"]),
            }
        return {"response": response_text}

    # Handle end of questions
    else:
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
                "response": f"An error occurred while saving your responses: {str(e)}",
                "final_responses": responses,
            }


def handle_emirate_upload_document_car_insurance(
    user_message,
    conversation_state,
    questions,
    responses,
    question,
    user_language="English",
):
    from services.llm_services import (
        format_response_in_language,
        validate_response_multilingual,
        translate_text,
    )

    valid_options = ["Yes", "No"]

    # Define all questions upfront for better maintainability
    QUESTIONS = {
        "upload": {"question": "Please Upload Your Document"},
        "name": {
            "question": "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"
        },
        "dob": "Date of Birth (DOB)",  # Simple string format as requested
        "gender": {
            "question": "Please confirm this gender of",
            "options": ["Male", "Female"],
        },
    }

    # Use multilingual validation instead of direct string matching
    validation_result = validate_response_multilingual(
        user_message, valid_options, user_language
    )

    if not validation_result["is_valid"]:
        general_assistant_prompt = (
            f"user response: {user_message}. Please assist them in {user_language}."
        )
        general_assistant_response = llm.invoke([
            SystemMessage(
                content=f"You are Insura, a friendly Insurance assistant created by CloudSubset. Respond in {user_language}. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."
            ),
            HumanMessage(content=general_assistant_prompt),
        ])
        next_question = questions[conversation_state["current_question_index"]]
        if isinstance(next_question, dict) and "options" in next_question:
            options = next_question["options"]
            retry_question = translate_text(
                f"Let's try again: {next_question['question']}", user_language
            )
            translated_options = [translate_text(opt, user_language) for opt in options]
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": retry_question,
                "options": ", ".join(translated_options),
            }
        else:
            question_text = (
                question["question"] if isinstance(question, dict) else question
            )
            retry_question = translate_text(
                f"Let's try again: {question_text}", user_language
            )
            translated_options = [
                translate_text(opt, user_language) for opt in valid_options
            ]
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"{retry_question}\nPlease choose from the following options: {', '.join(translated_options)}",
            }

    # Store the response - handle both string and dict question formats
    question_text = question["question"] if isinstance(question, dict) else question
    matched_value = validation_result["matched_value"]
    responses[question_text] = matched_value

    # Handle "Yes" path
    if matched_value == "Yes":
        upload_question = QUESTIONS["upload"]
        name_question = QUESTIONS["name"]
        dob_question = QUESTIONS["dob"]
        gender_question = QUESTIONS["gender"]

        # Remove other questions if they exist
        for q in [name_question, dob_question, gender_question]:
            if q in questions:
                questions.remove(q)

        if upload_question not in questions:
            questions.insert(
                conversation_state["current_question_index"] + 1, upload_question
            )
            responses[upload_question["question"]] = None

        conversation_state["current_question_index"] += 1
        next_question = questions[conversation_state["current_question_index"]]
        next_question_text = (
            next_question["question"]
            if isinstance(next_question, dict)
            else next_question
        )
        response_message = f"Thank you for the responses! Now, {next_question_text}"
        return format_response_in_language(response_message, [], user_language)

    # Handle "No" path
    elif matched_value == "No":
        # Remove upload document question if it exists
        upload_question = QUESTIONS["upload"]
        if upload_question in questions:
            questions.remove(upload_question)

        # Add all required questions in sequence
        next_index = conversation_state["current_question_index"] + 1
        for key in ["name", "dob", "gender"]:
            question_dict = QUESTIONS[key]
            if question_dict not in questions:
                questions.insert(next_index, question_dict)
                responses[
                    question_dict["question"]
                    if isinstance(question_dict, dict)
                    else question_dict
                ] = None
                next_index += 1

        # Move to next question
        conversation_state["current_question_index"] += 1

        # Check if there are more questions
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            next_question_text = (
                next_question["question"]
                if isinstance(next_question, dict)
                else next_question
            )
            response_text = f"Thank you for your response. Now, let's move on to: {next_question_text}"

            # Add options if they exist
            if isinstance(next_question, dict) and "options" in next_question:
                options = next_question["options"]
                return format_response_in_language(
                    response_text, options, user_language
                )
            return format_response_in_language(response_text, [], user_language)

        # Handle end of questions
        else:
            try:
                with open("user_responses.json", "w") as file:
                    json.dump(responses, file, indent=4)
                final_message = "Thank you for using Insura. Your responses have been recorded. Feel free to ask any other questions. Have a great day!"
                result = format_response_in_language(final_message, [], user_language)
                result["final_responses"] = responses
                return result
            except Exception as e:
                error_message = (
                    f"An error occurred while saving your responses: {str(e)}"
                )
                return format_response_in_language(error_message, [], user_language)
