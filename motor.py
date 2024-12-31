# import json
# from rapidfuzz import process, fuzz

# # Initialize a set of valid nationalities
# valid_nationalities = [
#     "Afghan", "Albanian", "Algerian", "American", "Andorran", "Angolan", 
#     "Antiguan or Barbudan", "Argentine", "Armenian", "Australian", "Austrian", 
#     "Azerbaijani", "Bahamian", "Bahraini", "Bangladeshi", "Barbadian", 
#     "Belarusian", "Belgian", "Belizean", "Beninese", "Bhutanese", "Bolivian", 
#     "Bosnian or Herzegovinian", "Motswana", "Brazilian", "Bruneian", 
#     "Bulgarian", "Burkinabé", "Burundian", "Cabo Verdean", "Cambodian", 
#     "Cameroonian", "Canadian", "Central African", "Chadian", "Chilean", 
#     "Chinese", "Colombian", "Comoran", "Congolese (Democratic Republic)", 
#     "Congolese (Republic)", "Costa Rican", "Croatian", "Cuban", "Cypriot", 
#     "Czech", "Danish", "Djiboutian", "Dominican", "Timorese", "Ecuadorian", 
#     "Egyptian", "Salvadoran", "Equatorial Guinean", "Eritrean", "Estonian", 
#     "Swazi", "Ethiopian", "Fijian", "Finnish", "French", "Gabonese", 
#     "Gambian", "Georgian", "German", "Ghanaian", "Greek", "Grenadian", 
#     "Guatemalan", "Guinean", "Bissau-Guinean", "Guyanese", "Haitian", 
#     "Honduran", "Hungarian", "Icelander", "Indian", "Indonesian", "Iranian", 
#     "Iraqi", "Irish", "Israeli", "Italian", "Jamaican", "Japanese", 
#     "Jordanian", "Kazakhstani", "Kenyan", "I-Kiribati", "North Korean", 
#     "South Korean", "Kosovar", "Kuwaiti", "Kyrgyzstani", "Laotian", "Latvian", 
#     "Lebanese", "Basotho", "Liberian", "Libyan", "Liechtensteiner", 
#     "Lithuanian", "Luxembourger", "Malagasy", "Malawian", "Malaysian", 
#     "Maldivian", "Malian", "Maltese", "Marshallese", "Mauritanian", 
#     "Mauritian", "Mexican", "Micronesian", "Moldovan", "Monegasque", 
#     "Mongolian", "Montenegrin", "Moroccan", "Mozambican", "Burmese", 
#     "Namibian", "Nauruan", "Nepali", "Dutch", "New Zealander", "Nicaraguan", 
#     "Nigerien", "Nigerian", "Macedonian", "Norwegian", "Omani", "Pakistani", 
#     "Palauan", "Panamanian", "Papua New Guinean", "Paraguayan", "Peruvian", 
#     "Filipino", "Polish", "Portuguese", "Qatari", "Romanian", "Russian", 
#     "Rwandan", "Kittitian or Nevisian", "Saint Lucian", "Vincentian", 
#     "Samoan", "Sammarinese", "São Toméan", "Saudi or Saudi Arabian", 
#     "Senegalese", "Serbian", "Seychellois", "Sierra Leonean", "Singaporean", 
#     "Slovak", "Slovenian", "Solomon Islander", "Somali", "South African", 
#     "South Sudanese", "Spanish", "Sri Lankan", "Sudanese", "Surinamese", 
#     "Swedish", "Swiss", "Syrian", "Taiwanese", "Tajikistani", "Tanzanian", 
#     "Thai", "Togolese", "Tongan", "Trinidadian or Tobagonian", "Tunisian", 
#     "Turkish", "Turkmen", "Tuvaluan", "Ugandan", "Ukrainian", "Emirati", 
#     "British", "Uruguayan", "Uzbekistani", "Vanuatuan", "Venezuelan", 
#     "Vietnamese", "Yemeni", "Zambian", "Zimbabwean"
# ]

# def is_valid_nationality(user_input):
#     """
#     Validates whether the provided user input is a recognized nationality,
#     including fuzzy matching for similar words.

#     Args:
#         user_input (str): The user's input to validate.

#     Returns:
#         bool: True if the input is a valid or similar nationality, False otherwise.
#     """
#     # Convert user input to title case
#     user_input = user_input.strip().title()
    
#     # Exact match
#     if user_input in valid_nationalities:
#         return True
    
#     # Fuzzy match with a threshold
#     matches = process.extractOne(user_input, valid_nationalities, scorer=fuzz.ratio)
#     if matches and matches[1] >= 85:  # You can adjust the threshold as needed
#         return True
    
#     return False

# # Example usage
# user_message = "IndiAN"  # Lowercase nationality
# if is_valid_nationality(user_message):
#     print(f"'{user_message}' is recognized as a valid nationality.")
# else:
#     print(f"'{user_message}' is not recognized as a valid nationality.")

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
import json

from dotenv import load_dotenv
load_dotenv()

def handle_validate_name(question, user_message, conversation_state, questions, responses):
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
