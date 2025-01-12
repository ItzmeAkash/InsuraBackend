import subprocess
import spacy
import sys

def download_model():
    """
    Download the spaCy model if it's not installed.
    """
    try:
        # Try loading the model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not found, download it
        print("Model not found, downloading...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while downloading the model: {e}")
            sys.exit(1)  # Exit if the download fails
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

print(is_valid_name("babu pk"))