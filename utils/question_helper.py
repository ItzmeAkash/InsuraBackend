import json
from langchain_core.messages import HumanMessage,SystemMessage
import os
from langchain_groq.chat_models import ChatGroq
import re
from datetime import datetime
from utils.helper import is_valid_nationality
from utils.helper import valid_date_format

llm = ChatGroq(
    model=os.getenv('LLM_MODEL'),
    temperature=0,
    api_key=os.getenv('GROQ_API_KEY'),
    groq_proxy=None
)

# Question To whom are you purchasing this plan?
def handle_purchasing_plan_question(user_message,conversation_state,questions,responses):
    valid_options = [
            "Employee",
            "Dependents",
            "Small Investors",
            "Domestic Help",
            "4th Child",
            "Children above 18 years",
            "Parents"
    ]   
    
    if user_message in valid_options:
        #Update the Response 
        responses["To whom are you purchasing this plan?"] = user_message
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
        "question":f"Let’s try again: To whom are you purchasing this plan? \nPlease choose from the following options: {', '.join(valid_options)}"
        }
                  
#Question Let's start with your Medical insurance details. Chosse your Visa issued Emirate? 
def handle_visa_issued_emirate_question(user_message,conversation_state,questions,responses): 
    valid_options =[
            "Abudhabi",
            "Ajman",
            "Dubai",
            "Fujairah",
            "Ras Al Khaimah",
            "Sharjah",
            "Umm Al Quwain"
        ]
    if user_message in valid_options:
        responses["Let's start with your Medical insurance details. Chosse your Visa issued Emirate?"]=user_message
        conversation_state["current_question_index"]+=1
        
        if conversation_state["current_question_index"]< len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            options = ", ".join(next_question["options"])
            next_questions = next_question["question"]
                
            
            return {
                    "response": f"Thank you! Now, let's move on to: {next_questions}",
                    "options":options

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
        "question":f"Let’s try again: Tell your Visa issued Emirate? \choose from the following options: {', '.join(valid_options)}"
        } 
             
# Question What type of plan are you looking for?     
def handle_type_plan_question(user_message,conversation_state,questions,responses):
    valid_options =[
               "Basic Plan",
               "Enhanced Plan"
            ]
    if user_message in valid_options:
        responses["What type of plan are you looking for?"]=user_message
        conversation_state["current_question_index"]+=1
        
        if conversation_state["current_question_index"]< len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            options = ", ".join(next_question["options"])
            next_questions = next_question["question"]
                
            
            return {
                    "response": f"Thank you! Now, let's move on to: {next_questions}",
                    "options":options

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
        "question":f"Let’s try again: What type of plan are you looking for? Please choose from the following options: {', '.join(valid_options)}"
        } 
        
# Question Is accommodation provided to you?        
def handle_yes_or_no(user_message,conversation_state,questions,responses,question):
    valid_options = ["Yes", "No"]
    if user_message in valid_options:
        responses["Is accommodation provided to you?"] = user_message
        conversation_state["current_question_index"] += 1
        
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


def handle_validate_name(question, user_message, conversation_state, questions, responses, is_valid_name):
    """
    Handles the name validation in a conversation flow.

    Parameters:
    - question (str): The current question being asked.
    - user_message (str): The user's response to the question.
    - conversation_state (dict): Dictionary maintaining the state of the conversation.
    - questions (list): List of all questions in the flow.
    - responses (dict): Dictionary to store user responses.
    - llm (object): Instance of the LLM for generating prompts and responses.
    - is_valid_name (callable): Function to validate names using regex.

    Returns:
    - dict: Contains the response message and optionally the next question or final responses.
    """
    if conversation_state["current_question_index"] == questions.index(question):
        # Validate user input using regex
        if not is_valid_name(user_message):
            general_assistant_prompt = (
                f"The user entered '{user_message}', which does not appear to be a valid name. "
                "Please assist them in providing a valid name."
            )
            general_assistant_response = llm.invoke([
                SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. "
                                          "Your role is to assist users with their inquiries and guide them appropriately."),
                HumanMessage(content=general_assistant_prompt)
            ])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": question
            }

        # Prompt LLM for additional validation
        check_prompt = (
            f"The user has responded with: '{user_message}'. Determine if this is a valid person's name. "
            "Respond only with 'Yes' or 'No'."
        )
        llm_response = llm.invoke([
            SystemMessage(content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                                      "Your task is to verify if the provided input is a valid person's name."),
            HumanMessage(content=check_prompt)
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
                    "final_responses": responses
                }
        else:
            # Use general assistant for invalid LLM validation
            general_assistant_prompt = (
                f"The user entered '{user_message}', which was not validated as a name by Insura. "
                "Please assist them in correcting their input."
            )
            general_assistant_response = llm.invoke([
                SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. "
                                          "Your role is to assist users with their inquiries and guide them appropriately."),
                HumanMessage(content=general_assistant_prompt)
            ])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": question
            }
def handle_gender(user_message,conversation_state,questions,responses,question):
    valid_options = [ "Male","Female"]
    if user_message in valid_options:
        responses[question] = user_message
        conversation_state["current_question_index"] += 1
        
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



def is_valid_policy_number(policy_number: str) -> bool:
    """
    Validates the format of a policy number.
    Example: Policy numbers must be alphanumeric and 6-12 characters long.
    """
    return bool(re.match(r'^[A-Za-z0-9]{6,12}$', policy_number))


def handle_policy_question(user_message,conversation_state,questions,responses,question):
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
                SystemMessage(content="You are Insura, an AI assistant specialized in insurance-related tasks. "
                                      "Your task is to verify if the provided input is a valid policy number."),
                HumanMessage(content=check_prompt)
            ])
            is_valid_policy = llm_response.content.strip().lower() == "yes"

            if is_valid_policy:
                # Store the policy number
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    options = ", ".join(next_question["options"])
                    next_questions = next_question["question"]
                    
                    return {
                        "response": f"Thank you for your response. Now, let's move on to: {next_questions}",
                        "options":options 
                    }
                else:
                    # All questions completed
                    with open("user_responses.json", "w") as file:
                        json.dump(responses, file, indent=4)
                    return {
                        "response": "Thank you for using Insura. Your request has been processed. Have a great day!",
                        "final_responses": responses
                    }
            else:
                # Use general assistant for invalid LLM validation
                general_assistant_prompt = (
                    f"The user entered '{user_message}', which was not validated as a policy number by Insura. "
                    "Please assist them in correcting their input."
                )
                general_assistant_response = llm.invoke([
                    SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. "
                                          "Your role is to assist users with their inquiries and guide them appropriately."),
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question":f"Let’s try again: {question}\n"
                }



def handle_date_question(question, user_message, responses, conversation_state, questions):
    

    if question == question:
        # Debugging: Check the type of user_messa

        # Ensure user_message is a string
        if isinstance(user_message, str):
            if valid_date_format(user_message):
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
                # Handle invalid date format
                try:
                    datetime.strptime(user_message, "%d/%m/%Y")
                except ValueError:
                    return {
                        "response": "Invalid date format. Please provide the date in the format DD/MM/YYYY.",
                        "question": question
                    }

                general_assistant_prompt = f"User response: {user_message}. Please assist."
                general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                return {
                    "response": general_assistant_response.content.strip(),
                    "question": f"Let’s try again: {question}"
                }
        else:
            return {
                "response": "Error: Expected a string input for the date. Please provide a valid date.",
                "question": question
            }
            
            

def handle_company_name_question(question, user_message, conversation_state, questions, responses):
    if question == question:
        if conversation_state["current_question_index"] == questions.index(question):
            # Check if the input is a company name using LLM
            check_prompt = f"The user has responded with: '{user_message}'. Is this a valid company name? Respond with 'Yes' or 'No'."
            llm_response = llm.invoke([SystemMessage(content=f"Check {user_message} this message is a valid Company name not an general topic make sure check all the details "), HumanMessage(content=check_prompt)])
            is_company_name = llm_response.content.strip().lower() == "yes"

            if is_company_name:
                # Store the company name
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_question['question']}",
                            "options": options
                        }
                    else:
                        return {
                            "response": f"Thank you for providing the company name. Now, let's move on to: {next_question['question']}"
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
                    "question": f"Let's Move Back {question}"
                }


def handle_job_title_question(question, user_message, conversation_state, questions, responses):
     if question == question:
        if conversation_state["current_question_index"] == questions.index(question):
            # Prompt LLM to check if the input is a valid job title
            check_prompt = f"The user has responded with: '{user_message}'. Is this a valid job title? Respond with 'Yes' or 'No'."
            llm_response = llm.invoke([
                SystemMessage(content="You are Insura, an AI assistant specialized in insurance-related tasks. Your task is to determine if the input provided by the user is a valid job title. Make sure it is a recognizable and appropriate job title."),
                HumanMessage(content=check_prompt)
            ])
            is_job_title = llm_response.content.strip().lower() == "yes"

            if is_job_title:
                # Store the job title
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Check if there are more questions
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you for providing your job title.{next_question}"
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
                general_assistant_prompt = f"The user entered '{user_message}', which does not appear to be a valid job title. Please assist."
                general_assistant_response = llm.invoke([
                    SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately."),
                    HumanMessage(content=general_assistant_prompt)
                ])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's move back to: {question}"
                }
                

def handle_emirate_question(question, user_message, conversation_state, questions, responses):
    if question == question:
            valid_options =[
         "Abudhabi",
              "Ajman",
              "Dubai",
              "Fujairah",
              "Ras Al Khaimah",
              "Sharjah",
              "Umm Al Quwain"
            ]
            if user_message in valid_options:
                responses[question]=user_message
                conversation_state["current_question_index"]+=1
                
                if conversation_state["current_question_index"]< len(questions):
                    next_question = questions[conversation_state["current_question_index"]]

                        
                    
                    return {
                         "response": f"Thank you! Now, let's move on to: {next_question}",
                      

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
               
               
def handle_nationality_question(user_message,question, conversation_state, questions, responses):
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
                            "final_responses": responses
                        }
                else:
                    # Handle invalid or unrelated input
                    general_assistant_prompt = f"The user entered '{user_message}'. Please assist."
                    general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's move back: {question}"
                    }
