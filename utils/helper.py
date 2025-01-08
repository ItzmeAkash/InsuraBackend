from datetime import datetime
import re
from urllib import response
from fuzzywuzzy import process, fuzz
import requests
from sqlalchemy import null
def get_user_name(user_id: str) -> str:

    return f"{user_id}"  


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
        
def valid_emirates_id(emirates_id):
    # Pattern: Starts with 784, followed by a birth year (4 digits), 7 digits, and ends with 1 digit
    pattern = r"784-\d{4}-\d{7}-\d"
    return bool(re.fullmatch(pattern, emirates_id))


def is_valid_name(name):
    # Allow names with alphabets, spaces, and hyphens
    pattern = r"^[A-Za-z]+(?: [A-Za-z]+)*$"
    return bool(re.match(pattern, name.strip()))


import json
from rapidfuzz import process, fuzz

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
    policy_type_question = responses_dict.get("What would you like to do today?", "").lower()
    plan_question = responses_dict.get("What type of plan are you looking for?", "").lower()
    covid_dose_question = responses_dict.get("Have you been vaccinated for Covid-19?", "").lower()
    gender_question = responses_dict.get("May I Know member's gender.Please?", "").lower()
    marital_status_member_question = responses_dict.get("May I know your marital status?", "").lower()
    
    if policy_type_question == "purchase a new policy":
        policy_type = "new"
    else:
        policy_type = "renewal"
    
    payload = {
        "visa_issued_emirates": responses_dict.get("Let's start with your Medical insurance details. Chosse your Visa issued Emirate?", "").lower(),
        "policy_type": policy_type,
        "visa_date": responses_dict.get("Please enter your Entry Date or Visa Change Status Date.", ""),
        "expiry_date": responses_dict.get("Please enter your Entry Date or Visa Change Status Date.", ""),
        "insurance_company": responses_dict.get("insurance_company", ""),
        "plan": plan_question,
        "basic_plan": responses_dict.get("To whom are you purchasing this plan?", ""),
        "sponsor": responses_dict.get("Could you let me know the sponsor's type?", "").lower(),
        "monthly_salary": responses_dict.get("Could you please tell me your monthly salary (in AED)?", ""),
        "accomodation": responses_dict.get("Is accommodation provided to you?", "").lower(),
        "job_title": responses_dict.get("Please provide us with your job title", "").lower(),
        "sponsor_type": responses_dict.get("Could you let me know the sponsor's type?", "").lower(),
        "sponsor_name": responses_dict.get("Now, let’s move to the sponsor details. Please provide the Sponsor Name?", "").lower(),
        "sponsor_gender": responses_dict.get("May I Know sponsor's gender.Please", "").lower(),
        "sponsor_marital_status": responses_dict.get("May I know sponsor's marital status?", "").lower(),
        "sponsor_po": responses_dict.get("Could you share your PO Box number, please?", "").lower(),
        "sponsor_emirate": responses_dict.get("Tell me your Emirate sponsor located in?", "").lower(),
        "sponsor_nationality": responses_dict.get("Could you let me know the sponsor's nationality?", "").lower(),
        "sponsor_country": responses_dict.get("May I have the sponsor's Country, please?", "").lower(),
        "sponsor_mobile": responses_dict.get("May I have the sponsor's mobile number, please?", "").lower(),
        "sponsor_email": responses_dict.get("May I have the sponsor's Email Address, please?", "").lower(),
        "sponsor_company": responses_dict.get("What company does the sponsor work for?", "").lower(),
        "sponsor_emirates_id": responses_dict.get("Could you provide the sponsor's Emirates ID?", ""),
        "sponsor_vat": responses_dict.get("Tell me Sponsor's  VAT Number", ""),
        "sponsor_income_source": responses_dict.get("Could you kindly provide me with the sponsor's Source of Income", ""),
        "name": responses_dict.get("Next, we need the details of the member for whom the policy is being purchased. Please provide Name", "").lower(),
        "dob": responses_dict.get("Date of Birth (DOB)", ""),
        "gender": gender_question,
        "marital_status": marital_status_member_question,
        "height": responses_dict.get("Tell me you Height in Cm", ""),
        "weight": responses_dict.get("Tell me you Weight in Kg", ""),
        "relation": responses_dict.get("Tell your relationship with the Sponsor", "").lower(),
        "covid_dose": covid_dose_question,
        "first_dose": responses_dict.get("Can you please tell me the date of your first dose?", "") if covid_dose_question == "yes" else None,
        "second_dose": responses_dict.get("Can you please tell me the date of your second dose?", "") if covid_dose_question == "yes" else None,
        "chronic_condition": responses_dict.get("Are you suffering from any pre-existing or chronic conditions?", "").lower(),
        "chronic_note": None,
        "pregnant": responses_dict.get("May I kindly ask if you are currently pregnant?", "").lower() if gender_question == "female" and marital_status_member_question == "married" else "no",
        "pregnant_note": None,
        "pregnant_referral": None,
        "pregnant_planning": responses_dict.get("Have you recently been preparing or planning for pregnancy?", "").lower() if gender_question == "female" and marital_status_member_question == "married" else "no",
        "menstrual_date": responses_dict.get("Could you please share the date of your last menstrual period?", "").lower() if gender_question == "female" and marital_status_member_question == "married" else None
    }

    api = "https://www.insuranceclub.ae/Api/medical_insert"
    
    res = requests.post(api, json=payload)
    
    print(payload)
    return res

responses = {
    "What would you like to do today?": "Purchase a new policy",
    "Let's start with your Medical insurance details. Chosse your Visa issued Emirate?": "Ras Al Khaimah",
    "question": "Sharjah",
    "Please enter your Entry Date or Visa Change Status Date?": "21/10/2025",
    "What type of plan are you looking for?": "Enhanced Plan",
    "To whom are you purchasing this plan?": "Children above 18 years",
    "Could you please tell me your monthly salary (in AED)?": 20000.0,
    "Is accommodation provided to you?": "Yes",
    "Please provide us with your job title": "software engineer",
    "Now, let\u2019s move to the sponsor details. Please provide the Sponsor Name?": "Jeffin",
    "Could you share your PO Box number, please?": "POB5678926",
    "Tell me your Emirate sponsor located in?": "Sharjah",
    "Could you let me know the sponsor's nationality?": "Indian",
    "May I have the sponsor's Country, please?": "India",
    "May I have the sponsor's mobile number, please?": "+919567551494",
    "Could you let me know the sponsor's type?": "Employee",
    "May I have the sponsor's Email Address, please?": "akash@gmail.co",
    "What company does the sponsor work for?": "Giza Systems",
    "Could you provide the sponsor's Emirates ID?": "784-1990-1234567-0",
    "Tell me Sponsor's  VAT Number": "VAT456781267",
    "Could you kindly provide me with the sponsor's Source of Income": "Salary",
    "May I Know sponsor's gender.Please": "Male",
    "May I know sponsor's marital status?": "Single",
    "Next, we need the details of the member for whom the policy is being purchased. Please provide Name": "Jeffin",
    "Date of Birth (DOB)": "21/10/1999",
    "May I Know member's gender.Please?": "Female",
    "May I know your marital status?": "Married",
    "May I kindly ask if you are currently pregnant?": "Yes",
    "Have you recently been preparing or planning for pregnancy?": "Yes",
    "Could you please share the date of your last menstrual period?": "21/10/1999",
    "Tell me you Height in Cm": "175",
    "Tell me you Weight in Kg": "78",
    "Tell your relationship with the Sponsor": "Wife",
    "Have you been vaccinated for Covid-19?": "Yes",
    "Can you please tell me the date of your first dose?": "21/10/2024",
    "Can you please tell me the date of your second dose?": "22/10/2025",
    "Are you suffering from any pre-existing or chronic conditions?": "No"
}

print(fetching_medical_detail(responses))