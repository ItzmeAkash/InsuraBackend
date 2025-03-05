from json import load
import os
import base64
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
import io



#Vison model class
class ChatGroqVisionOCR:
    def __init__(self, api_key=None):
        """
        Initialize ChatGroq Vision OCR client
        
        Args:
            api_key (str, optional): Groq API key
        """
        # Use API key from environment or passed parameter
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        # Initialize ChatGroq with vision-capable model
        self.chat = ChatGroq(
            groq_api_key=self.api_key,
            model="llama-3.2-11b-vision-preview",  # Vision-capable model
            temperature=0.2,  # Lower temperature for more precise OCR
            max_tokens=300
        )
    
    def encode_image(self, image_path):
        """
        Encode image to base64
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Base64 encoded image
        """
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def extract_english_text(self, image_path):
        """
        Extract English text from the image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Extracted English text from the image
        """
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Specific prompt for English text extraction
        english_text_prompt = (
            "Extract ONLY the English text from this image. "
            "Ignore any non-English text or symbols. "
            "Preserve the original formatting and reading order. "
            "If there are multiple text sections, organize them clearly. "
            "Ensure the output contains ONLY English text."
        )
        
        try:
            # Prepare message with image and prompt
            msg = HumanMessage(
                content=[
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": english_text_prompt
                    }
                ]
            )
            
            # Invoke OCR
            response = self.chat.invoke([msg])
            
            return response.content
        
        except Exception as e:
            print(f"English Text Extraction Error: {e}")
            return None
# Example usage
def main():
    # Initialize ChatGroq OCR client
    ocr_client = ChatGroqVisionOCR(
        api_key=os.getenv("GROQ_API_KEY")  # Replace with your Groq API key
    )
    
    # Path to the image
    image_path = "WhatsApp Image 2025-03-05 at 11.21.20 AM.jpeg"
    
    # Perform basic OCR
    english_text = ocr_client.extract_english_text(image_path)
    print("Extracted English Text:")
    print(english_text)

if __name__ == "__main__":
    main()