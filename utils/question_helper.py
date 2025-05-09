import json
from langchain_core.messages import HumanMessage,SystemMessage
import os
from langchain_groq.chat_models import ChatGroq
import re
from datetime import datetime
from utils.helper import fetching_medical_detail, is_valid_country, is_valid_nationality
from utils.helper import valid_date_format
from fastapi import Request
llm = ChatGroq(
    model=os.getenv('LLM_MODEL'),
    temperature=0,
    api_key=os.getenv('GROQ_API_KEY'),
    groq_proxy=None
)

# Question To whom are you purchasing this plan?
def handle_purchasing_plan_question(user_message,conversation_state,questions,responses,question):
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
                general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
                general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        "options": options
                    }


                else:
                        return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        }      
                  
                  
#Question Visa issued Emirate? 
def handle_visa_issued_emirate_question(user_message,conversation_state,questions,responses,question): 
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
        responses["question"]=user_message
        conversation_state["current_question_index"]+=1
        
        if conversation_state["current_question_index"]< len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = ", ".join(next_question["options"])
                next_questions = next_question["question"]
                return {
                    "response": f"Thank you! Now, let's move on to: {next_questions}",
                    "options": options
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
                "final_responses": responses
                }
    else:    
                general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
                general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        "options": options
                    }


                else:
                        return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        } 
# Question What type of plan are you looking for?     
def handle_type_plan_question(user_message, conversation_state, questions, responses,question):
    valid_options = [
          "Basic Plan",
          "Enhanced Plan",
          "Enhanced Plan Standalone",
          "Flexi Plan"
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
                return {
                    "response": f"Thank you!{next_questions}",
                    "options": options
                }
            else:
                return {
                    "response": f"Thank you.{next_question}"
                }
        else:
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
                "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
                "final_responses": responses
            }
    else:
        general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
        general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
        next_question = questions[conversation_state["current_question_index"]]
        if "options" in next_question:
            options = ", ".join(next_question["options"])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's Move Back {question}",
                "options": options
            }


        else:
                return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's Move Back {question}",
                }     
# Question Is accommodation provided to you?        
def handle_yes_or_no(user_message,conversation_state,questions,responses,question):
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
                    "options": options
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
                SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. "
                                      "Your role is to assist users with their inquiries and guide them appropriately."),
                HumanMessage(content=general_assistant_prompt)
            ])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": question,
                "example":"Please ensure that the member's name is written in its full form (e.g., Shefeek PS)"
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
                "question": question,
                 "example":"Please ensure that the member's name is written in its full form (e.g., Shefeek PS)"
            }
def handle_gender(user_message,conversation_state,questions,responses,question):
    valid_options = [ "Male","Female"]
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
                   
                    "options": options
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
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options
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
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options
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
                        "final_responses": responses
                    }
            else:
                # Handle invalid or unrelated input
                general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
                general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    next_question = next_question['question']
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {next_question}",
                        "options": options
                    }


                else:
                        return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        }

def handle_job_title_question(question, user_message, conversation_state, questions, responses):
    if question == question:
        if conversation_state["current_question_index"] == questions.index(question):
            # Prompt LLM to check if the input is a valid job title
            check_prompt = f"The user has responded with: '{user_message}'. Is this a valid job title? Respond with 'Yes' or 'No'."
            llm_response = llm.invoke([
                SystemMessage(content="You are Insura, an AI assistant specialized in insurance-related tasks. You are in a job title validation section you task is when any job title comes make sure that is a job title or not"),
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
                        "response": f"Thank you for providing your job title. {next_question}"
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
                    SystemMessage(content="You are Insura, an AI assistant created by CloudSubset. Your role is to assist users with their inquiries. Your task here is to redirect or assist the user appropriately"),
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
                    if "options" in next_question:
                        options = ", ".join(next_question["options"])
                        next_questions = next_question["question"]
                        return {
                            "response": f"Thank you! Now, let's move on to: {next_questions}",
                            "options": options
                        }
                    
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


#Todo def handle_marital_status(user_message, conversation_state, questions, responses, question):
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

#Todo(female married condition) def handle_marital_status(user_message, conversation_state, questions, responses, question):
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

def handle_marital_status(user_message, conversation_state, questions, responses, question):
        valid_options = ["Single", "Married"]
        if user_message in valid_options:
            responses[question] = user_message
            conversation_state["current_question_index"] += 1
            
            if conversation_state["current_question_index"] < len(questions):
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    # next_questions = next_question["question"]
                    member_name = responses.get("Next, we need the details of the member for whom the policy is being purchased. Please provide Name")
                    next_questions = next_question["question"].replace("your", f"{member_name}'s")

                    return {
                    "response": f"Thank you Next, let's discuss. {next_questions}",
                        "options": options
                    }
                return {
                    "response": f"Thank you Next, let's discuss. {next_question}"
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
                general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
                next_question = questions[conversation_state["current_question_index"]]
                if "options" in next_question:
                    options = ", ".join(next_question["options"])
                    return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        "options": options
                    }


                else:
                        return {
                        "response": f"{general_assistant_response.content.strip()}",
                        "question": f"Let's Move Back {question}",
                        } 
def handle_pregant(user_message,conversation_state,questions,responses,question):
    valid_options = ["Yes","No"]
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
                    "options": options
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

def handle_sposor_type(user_message,conversation_state,questions,responses,question):
        valid_options = ["Employee","Investors"]
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
                        "options": options
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
                "final_responses": responses
                    }
        else:
            # Handle invalid responses or unrelated queries
            general_assistant_prompt = f"The user entered '{user_message}', . Please assist."
            general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = ", ".join(next_question["options"])
                return {
                    "response": f"{general_assistant_response.content.strip()}",
                    "question": f"Let's Move Back {question}",
                    "options": options
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
  
def handle_country_question(user_message,question, conversation_state, questions, responses):
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

def handle_what_would_you_do_today_question(user_message,conversation_state,questions,responses,question):
    valid_options = [ "Purchase a Medical Insurance","Purchase a Motor Insurance","Claim a Motor Insurance"]
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
                    "options": options
                }
            else:
                return {
                    "response": f"Thank you for providing the plan. Now, let's move on to: {next_question}"
                }
        else:
            with open("user_responses.json", "w") as file:
                json.dump(responses, file, indent=4)
            return {
            "response": "You're all set! Thank you for providing your details. If you need further assistance, feel free to ask.",
            "final_responses": responses
                }
    else:  
        #TODO 
           # Handle invalid responses or unrelated queries
            general_assistant_prompt = f"user response: {user_message}. Please assist."
            general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
            next_question = questions[conversation_state["current_question_index"]]
            if "options" in next_question:
                options = ", ".join(next_question["options"])
                print(question)
                return {
                "response": f"{general_assistant_response.content.strip()}",
                "question":f"Let’s try again: {question}",
                "options": options
               }
            else:
                return {
                "response": f"{general_assistant_response.content.strip()}",
                "question":f"Let’s try again: {question}\nPlease choose from the following options: {', '.join(valid_options)}",
                }
                
                
#Todo
def  handle_date_question(question, user_message, responses, conversation_state, questions):
    if valid_date_format(user_message):
                responses[question] = user_message
                conversation_state["current_question_index"] += 1

                # Move to the next question or finalize responses
                if conversation_state["current_question_index"] < len(questions):
                    next_question = questions[conversation_state["current_question_index"]]
                    if 'options' in next_question:
                         options = ", ".join(next_question["options"])
                         next_questions = next_question["question"]
                         member_name = responses.get("Next, we need the details of the member for whom the policy is being purchased. Please provide Name")
                         return {
                              "response": f"Thanks! Let’s continue.{next_questions} {member_name}.given below",
                              "options": options
                             }
                    else:
                        return {
                               "response": f"Thanks! Let’s continue with {next_question}"
                             }     
                else:
                    # All questions answered
                    try:
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Thank you for using Insura. Your responses have been recorded. "
                                        "Feel free to ask any other questions. Have a great day!",
                            "final_responses": responses
                        }
                    except Exception as e:
                        return {
                            "response": f"An error occurred while saving your responses: {str(e)}"
                        }
    else:
        # Handle invalid Emirates ID or unrelated query
        general_assistant_prompt = f"user response: {user_message}. Please assist."
        general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
        

        return {
            "response": (
                f"{general_assistant_response.content.strip()} \n\n"
                
            ),
            "example": "Please provide the date in the format DD/MM/YYYY.",
            "question": f"Let’s try again: {question}"
        }
        
        
def handle_adiviosr_code(question, user_message, responses, conversation_state, questions):
            valid_options = ["Yes","No"]
            first_question = "Please enter your Insurance Advisor code for assigning your enquiry for further assistance"
            if user_message in valid_options:
                responses[question] = user_message  # Store the response

                if user_message == "Yes":
                    # Dynamically add follow-up questions
                    # Insert follow-up questions into the list if not already present
                    if first_question not in questions:
                        responses[first_question] = None
                        questions.insert(conversation_state["current_question_index"] + 1, first_question)

                    # Move to the next question
                    conversation_state["current_question_index"] += 1
                    next_question = questions[conversation_state["current_question_index"]]
                    return {
                        "response": f"Thank you for the responses! Now,{next_question}"
                    }
                elif user_message == "No":
                    # Remove the questions about first and second doses if they exist

                    if first_question in questions:
                        questions.remove(first_question)


                    # Proceed to the next predefined question
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
                        # All predefined questions have been answered
                        if responses.get("Do you have an Insurance Advisor code?")=="Yes":
                            medical_deatil_response = fetching_medical_detail(responses)
                            return {
                                "response":f"Thank you for sharing the details We will inform (Agent Name) to assist you further with your enquiry.Please find the link below to view your quotation: {medical_deatil_response}",
                            }
                          
                        with open("user_responses.json", "w") as file:
                            json.dump(responses, file, indent=4)
                        return {
                            "response": "Since you don't have an agent code, we will arrange a callback from the next available agent to assist you further Thank you!",
                            "final_responses": responses
                        }
            else:
                
                general_assistant_prompt = f"user response: {user_message}. Please assist."
                general_assistant_response = llm.invoke([HumanMessage(content=general_assistant_prompt)])
                

                return {
                    "response": (
                        f"{general_assistant_response.content.strip()} \n\n"
                        
                    ),
                    "question": f"Let’s try again: {question}"
                }
                
                
                
                
def handle_emirate_upload_document(user_message, conversation_state, questions, responses, question):
    valid_options = ["Yes", "No"]
    
    # Define all questions upfront for better maintainability
    QUESTIONS = {
        "upload": {"question": "Please Upload Your Document"},
        "name": {"question": "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"},
        "dob": "Date of Birth (DOB)",  # Simple string format as requested
        "gender": {"question": "Please confirm this gender of", "options": ["Male", "Female"]},
    }

    # Validate user input
    if user_message not in valid_options:
        general_assistant_prompt = f"user response: {user_message}. Please assist."
        general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
        next_question = questions[conversation_state["current_question_index"]]
        if isinstance(next_question, dict) and "options" in next_question:
            options = ", ".join(next_question["options"])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's try again: {next_question['question']}",
                "options": options
            }
        else:
            question_text = question["question"] if isinstance(question, dict) else question
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's try again: {question_text}\nPlease choose from the following options: {', '.join(valid_options)}",
            }

    # Store the response - handle both string and dict question formats
    question_text = question["question"] if isinstance(question, dict) else question
    responses[question_text] = user_message

    # Handle "Yes" path
    if user_message == "Yes":
        upload_question = QUESTIONS["upload"]
        name_question = QUESTIONS["name"]
        dob_question = QUESTIONS["dob"]
        gender_question = QUESTIONS["gender"]
        
        # Remove other questions if they exist
        for q in [name_question, dob_question, gender_question]:
            if q in questions:
                questions.remove(q)
            
        if upload_question not in questions:
            questions.insert(conversation_state["current_question_index"] + 1, upload_question)
            responses[upload_question["question"]] = None
        
        conversation_state["current_question_index"] += 1
        next_question = questions[conversation_state["current_question_index"]]
        next_question_text = next_question["question"] if isinstance(next_question, dict) else next_question
        return {
            "response": f"Thank you for the responses! Now, {next_question_text}"
        }

    # Handle "No" path
    elif user_message == "No":
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
                responses[question_dict["question"] if isinstance(question_dict, dict) else question_dict] = None
                next_index += 1

        # Move to next question
        conversation_state["current_question_index"] += 1
        
        # Check if there are more questions
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            next_question_text = next_question["question"] if isinstance(next_question, dict) else next_question
            response_text = f"Thank you for your response. Now, let's move on to: {next_question_text}"
            
            # Add options if they exist
            if isinstance(next_question, dict) and "options" in next_question:
                return {
                    "response": response_text,
                    "options": ", ".join(next_question["options"])
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
                    "final_responses": responses
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
    
    
def handle_emaf_document(question, user_message, responses, conversation_state, questions):
    QUESTIONS = {
        "name": {
            "question": "May I know your name, please?",
        },
        "phone": {
            "question": "May I kindly ask for your phone number, please?"
        },
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
                    ]

        }
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
            responses[question_dict["question"] if isinstance(question_dict, dict) else question_dict] = None
            next_index += 1
            
    # Move to next question
    conversation_state["current_question_index"] += 1
    
    # Check if there are more questions
    if conversation_state["current_question_index"] < len(questions):
        next_question = questions[conversation_state["current_question_index"]]
        next_question_text = next_question["question"] if isinstance(next_question, dict) else next_question
        response_text = f"Thank you for your response! I appreciate it. Now, {next_question_text}"
        
        # Add options if they exist
        if isinstance(next_question, dict) and "options" in next_question:
            return {
                "response": response_text,
                "options": ", ".join(next_question["options"])
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
                "final_responses": responses
            }
        except Exception as e:
            return {
                "response": f"An error occurred while saving your responses: {str(e)}",
                "final_responses": responses
            }
            
            

            
def handle_emirate_upload_document_car_insurance(user_message, conversation_state, questions, responses, question):
    valid_options = ["Yes", "No"]
    
    # Define all questions upfront for better maintainability
    QUESTIONS = {
        "upload": {"question": "Please Upload Your Document"},
        "name": {"question": "Next, we need the details of the member for whom the policy is being purchased. Please provide Name"},
        "dob": "Date of Birth (DOB)",  # Simple string format as requested
        "gender": {"question": "Please confirm this gender of", "options": ["Male", "Female"]},
    }

    # Validate user input
    if user_message not in valid_options:
        general_assistant_prompt = f"user response: {user_message}. Please assist."
        general_assistant_response = llm.invoke([SystemMessage(content="You are Insura, a friendly Insurance assistant created by CloudSubset. Your role is to assist with any inquiries using your vast knowledge base. Provide helpful, accurate, and user-friendly responses to all questions or requests. Do not mention being a large language model; you are Insura."),HumanMessage(content=general_assistant_prompt)])
        next_question = questions[conversation_state["current_question_index"]]
        if isinstance(next_question, dict) and "options" in next_question:
            options = ", ".join(next_question["options"])
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's try again: {next_question['question']}",
                "options": options
            }
        else:
            question_text = question["question"] if isinstance(question, dict) else question
            return {
                "response": f"{general_assistant_response.content.strip()}",
                "question": f"Let's try again: {question_text}\nPlease choose from the following options: {', '.join(valid_options)}",
            }

    # Store the response - handle both string and dict question formats
    question_text = question["question"] if isinstance(question, dict) else question
    responses[question_text] = user_message

    # Handle "Yes" path
    if user_message == "Yes":
        upload_question = QUESTIONS["upload"]
        name_question = QUESTIONS["name"]
        dob_question = QUESTIONS["dob"]
        gender_question = QUESTIONS["gender"]
        
        # Remove other questions if they exist
        for q in [name_question, dob_question, gender_question]:
            if q in questions:
                questions.remove(q)
            
        if upload_question not in questions:
            questions.insert(conversation_state["current_question_index"] + 1, upload_question)
            responses[upload_question["question"]] = None
        
        conversation_state["current_question_index"] += 1
        next_question = questions[conversation_state["current_question_index"]]
        next_question_text = next_question["question"] if isinstance(next_question, dict) else next_question
        return {
            "response": f"Thank you for the responses! Now, {next_question_text}"
        }

    # Handle "No" path
    elif user_message == "No":
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
                responses[question_dict["question"] if isinstance(question_dict, dict) else question_dict] = None
                next_index += 1

        # Move to next question
        conversation_state["current_question_index"] += 1
        
        # Check if there are more questions
        if conversation_state["current_question_index"] < len(questions):
            next_question = questions[conversation_state["current_question_index"]]
            next_question_text = next_question["question"] if isinstance(next_question, dict) else next_question
            response_text = f"Thank you for your response. Now, let's move on to: {next_question_text}"
            
            # Add options if they exist
            if isinstance(next_question, dict) and "options" in next_question:
                return {
                    "response": response_text,
                    "options": ", ".join(next_question["options"])
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
                    "final_responses": responses
                }
            except Exception as e:
                return {
                    "response": f"An error occurred while saving your responses: {str(e)}"
                }
                