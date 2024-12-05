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