from datetime import datetime
from pickle import NONE
import re
from urllib import response
from fuzzywuzzy import process, fuzz
import requests
from sqlalchemy import null
import subprocess
import json
from rapidfuzz import process, fuzz

def get_user_name(user_id: str) -> str:

    return f"{user_id}"  


def is_valid_mobile_number(number):
    # This is a simple regex for validating a mobile number
    pattern = re.compile(r'^\+?\d{10,15}$')
    return pattern.match(number) is not None

def valid_date_format(date_string, date_format="%d/%m/%Y"):
    """
    Validates if a given string is a valid date in the specified format.
    :param date_string: The date string to validate.
    :param date_format: The expected date format (default is DD/MM/YYYY).
    :return: True if valid, False otherwise.
    """
    if not isinstance(date_string, str):  # Ensure the input is a string
        return False
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False
def valid_adivisor_code(code):
    """
    Validates if the input is a 4-digit number.
    
    Args:
        emirates_id (str): The input to validate.
    
    Returns:
        bool: True if the input is a 4-digit number, False otherwise.
    """
    return code.isdigit() and len(code) == 4
       
def valid_emirates_id(emirates_id):
    # Pattern: Starts with 784, followed by a birth year (4 digits), 7 digits, and ends with 1 digit
    pattern = r"784-\d{4}-\d{7}-\d"
    return bool(re.fullmatch(pattern, emirates_id))


# def is_valid_name(name):
#     # Allow names with alphabets, spaces, and hyphens
#     pattern = r"^[A-Za-z]+(?: [A-Za-z]+)*$"
#     return bool(re.match(pattern, name.strip()))



def download_model():
    """
    Download the spaCy model if it's not installed.
    """
    try:
        # Try loading the model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not found, download it
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")  # Load the model again after download
    return nlp

def is_valid_name(text):
    """
    Detect if the text contains a person's name.
    """
    nlp = download_model()  
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return True
    return False




# Initialize a set of valid nationalities
valid_nationalities = [
    "Afghan", "Albanian", "Algerian", "American", "Andorran", "Angolan", 
    "Antiguan or Barbudan", "Argentine", "Armenian", "Australian", "Austrian", 
    "Azerbaijani", "Bahamian", "Bahraini", "Bangladeshi", "Barbadian", 
    "Belarusian", "Belgian", "Belizean", "Beninese", "Bhutanese", "Bolivian", 
    "Bosnian or Herzegovinian", "Motswana", "Brazilian", "Bruneian", 
    "Bulgarian", "Burkinabé", "Burundian", "Cabo Verdean", "Cambodian", 
    "Cameroonian", "Canadian", "Central African", "Chadian", "Chilean", 
    "Chinese", "Colombian", "Comoran", "Congolese (Democratic Republic)", 
    "Congolese (Republic)", "Costa Rican", "Croatian", "Cuban", "Cypriot", 
    "Czech", "Danish", "Djiboutian", "Dominican", "Timorese", "Ecuadorian", 
    "Egyptian", "Salvadoran", "Equatorial Guinean", "Eritrean", "Estonian", 
    "Swazi", "Ethiopian", "Fijian", "Finnish", "French", "Gabonese", 
    "Gambian", "Georgian", "German", "Ghanaian", "Greek", "Grenadian", 
    "Guatemalan", "Guinean", "Bissau-Guinean", "Guyanese", "Haitian", 
    "Honduran", "Hungarian", "Icelander", "Indian", "Indonesian", "Iranian", 
    "Iraqi", "Irish", "Israeli", "Italian", "Jamaican", "Japanese", 
    "Jordanian", "Kazakhstani", "Kenyan", "I-Kiribati", "North Korean", 
    "South Korean", "Kosovar", "Kuwaiti", "Kyrgyzstani", "Laotian", "Latvian", 
    "Lebanese", "Basotho", "Liberian", "Libyan", "Liechtensteiner", 
    "Lithuanian", "Luxembourger", "Malagasy", "Malawian", "Malaysian", 
    "Maldivian", "Malian", "Maltese", "Marshallese", "Mauritanian", 
    "Mauritian", "Mexican", "Micronesian", "Moldovan", "Monegasque", 
    "Mongolian", "Montenegrin", "Moroccan", "Mozambican", "Burmese", 
    "Namibian", "Nauruan", "Nepali", "Dutch", "New Zealander", "Nicaraguan", 
    "Nigerien", "Nigerian", "Macedonian", "Norwegian", "Omani", "Pakistani", 
    "Palauan", "Panamanian", "Papua New Guinean", "Paraguayan", "Peruvian", 
    "Filipino", "Polish", "Portuguese", "Qatari", "Romanian", "Russian", 
    "Rwandan", "Kittitian or Nevisian", "Saint Lucian", "Vincentian", 
    "Samoan", "Sammarinese", "São Toméan", "Saudi or Saudi Arabian", 
    "Senegalese", "Serbian", "Seychellois", "Sierra Leonean", "Singaporean", 
    "Slovak", "Slovenian", "Solomon Islander", "Somali", "South African", 
    "South Sudanese", "Spanish", "Sri Lankan", "Sudanese", "Surinamese", 
    "Swedish", "Swiss", "Syrian", "Taiwanese", "Tajikistani", "Tanzanian", 
    "Thai", "Togolese", "Tongan", "Trinidadian or Tobagonian", "Tunisian", 
    "Turkish", "Turkmen", "Tuvaluan", "Ugandan", "Ukrainian", "Emirati", 
    "British", "Uruguayan", "Uzbekistani", "Vanuatuan", "Venezuelan", 
    "Vietnamese", "Yemeni", "Zambian", "Zimbabwean"
]

def is_valid_nationality(user_input):
    """
    Validates whether the provided user input is a recognized nationality,
    including fuzzy matching for similar words.

    Args:
        user_input (str): The user's input to validate.

    Returns:
        bool: True if the input is a valid or similar nationality, False otherwise.
    """
    # Convert user input to title case
    user_input = user_input.strip().title()
    
    # Exact match
    if user_input in valid_nationalities:
        return True
    
    # Fuzzy match with a threshold
    matches = process.extractOne(user_input, valid_nationalities, scorer=fuzz.ratio)
    if matches and matches[1] >= 85:  # You can adjust the threshold as needed
        return True
    
    return False



marital_statuses = [
    "Single",
    "Married",
    "Divorced",
    "Widowed",
    "Separated",
    "Domestic Partnership",
    "Engaged",
    "In a Relationship",
    "Complicated Relationship",
    "Annulled",
    "Not Disclosed",
    "Civil Union"
]

def is_valid_marital_status(user_input):
    """
    Validates whether the provided user input is a recognized marital status,
    including fuzzy matching for similar words.

    Args:
        user_input (str): The user's input to validate.

    Returns:
        bool: True if the input is a valid or similar marital status, False otherwise.
    """
    # Convert user input to title case
    user_input = user_input.strip().title()
    
    # Exact match
    if user_input in marital_statuses:
        return True
    
    # Fuzzy match with a threshold
    matches = process.extractOne(user_input, marital_statuses, scorer=fuzz.ratio)
    if matches and matches[1] >= 85:  # You can adjust the threshold as needed
        return True
    
    return False

valid_countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", 
    "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", 
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", 
    "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", 
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", 
    "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", 
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", 
    "China", "Colombia", "Comoros", "Congo (Democratic Republic)", 
    "Congo (Republic)", "Costa Rica", "Croatia", "Cuba", "Cyprus", 
    "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", 
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", 
    "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", 
    "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", 
    "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", 
    "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", 
    "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", 
    "Jordan", "Kazakhstan", "Kenya", "Kiribati", "North Korea", 
    "South Korea", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", 
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", 
    "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", 
    "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", 
    "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", 
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", 
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", 
    "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", 
    "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", 
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", 
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", 
    "Saint Vincent and the Grenadines", "Samoa", "San Marino", 
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", 
    "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", 
    "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", 
    "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", 
    "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", 
    "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", 
    "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "UAE", 
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", 
    "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]

def is_valid_country(user_input):
    """
    Validates whether the provided user input is a recognized country,
    including fuzzy matching for similar words.

    Args:
        user_input (str): The user's input to validate.

    Returns:
        bool: True if the input is a valid or similar country, False otherwise.
    """
    # Convert user input to title case
    user_input = user_input.strip().title()
    
    # Exact match
    if user_input in valid_countries:
        return True
    
    # Fuzzy match with a threshold
    matches = process.extractOne(user_input, valid_countries, scorer=fuzz.ratio)
    if matches and matches[1] >= 85:  # You can adjust the threshold as needed
        return True
    
    return False

import requests

def fetching_medical_detail(responses_dict):
    def convert_date_format(date_str):
        try:
            return datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            return date_str 

    policy_type_question = responses_dict.get("What would you like to do today?", "").lower()

    marital_status_member_question = responses_dict.get("May I know your marital status?", "").capitalize()

    if policy_type_question == "purchase a new policy":
        policy_type = "New"

    else:
        policy_type = "Renewal"

    payload = {
        "visa_issued_emirates": responses_dict.get("Let's start with your Medical insurance details. Chosse your Visa issued Emirate?", "").capitalize(),
        "plan": responses_dict.get("What type of plan are you looking for?", "").capitalize(),

        "monthly_salary": responses_dict.get("Could you please tell me your monthly salary?", ""),
         "currency": responses_dict.get("May I kindly ask you to tell me the currency?",""),
        "sponsor_type": responses_dict.get("Now, let’s move to the sponsor details.Could you let me know the sponsor's type?", "").capitalize(),
        "sponsor_mobile": responses_dict.get("May I have the sponsor's mobile number, please?", "").capitalize(),
        "sponsor_email": responses_dict.get("May I have the sponsor's Email Address, please?", "").lower(),
        "members":[
            {
        "name": responses_dict.get("Next, we need the details of the member for whom the policy is being purchased. Please provide Name", "").capitalize(),
        "dob": convert_date_format(responses_dict.get("Date of Birth (DOB)", "")),
        "gender":responses_dict.get("May I Know member's gender.Please?",""),
        "marital_status": marital_status_member_question,
        "relation": responses_dict.get("Tell your relationship with the Sponsor", "").capitalize(),

            }
        ]
    
    }

    api = "https://www.insuranceclub.ae/Api/medical_insert"

    res = requests.post(api, json=payload)
    id = res.json()["id"]
    print(payload)
    return id

