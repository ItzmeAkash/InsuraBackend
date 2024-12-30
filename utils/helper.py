from datetime import datetime
import re
def get_user_name(user_id: str) -> str:

    return f"{user_id}"  


def valid_date_format(date_string, date_format="%d/%m/%Y"):
    """
    Validates if a given string is a valid date in the specified format.
    :param date_string: The date string to validate.
    :param date_format: The expected date format (default is DD/MM/YYYY).
    :return: True if valid, False otherwise.
    """
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

