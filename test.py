import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Initialize the ChatGroq model
chat_model = ChatGroq(
    model="llama-3.2-11b-vision-preview",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Define OCR prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an OCR assistant. Extract the text and translate it into English. Return only the extracted text."),
    ("user", "{image}")
])

def create_text_extract_chain(chat_model):
    """Create an OCR text extraction chain."""
    return prompt | chat_model

def main(chat_model, image_path: Path):
    """Extract text from an image using the LLaMA 3 Vision model."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()  # Read binary image data

    chain = create_text_extract_chain(chat_model)
    
    # Send the actual image file instead of base64 text
    input_data = {
        "image": image_data
    }

    result = chain.invoke(input_data)
    return result

if __name__ == "__main__":
    image_path = Path("test.jpeg")
    extracted_text = main(chat_model, image_path)
    print("Extracted Text:", extracted_text)
