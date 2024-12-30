import json
from langchain_core.messages import HumanMessage,SystemMessage
import os
from langchain_groq.chat_models import ChatGroq

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
