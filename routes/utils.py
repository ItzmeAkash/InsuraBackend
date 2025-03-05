from typing import Dict, Any, Optional
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain.chains import create_extraction_chain
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
import pytesseract
from pdf2image import convert_from_path
from PIL import Image,ImageFilter
import os
import io
import base64

#Extract text from the  give emirate pdf
async def extract_pdf_info(file_path: str):
    """
    Extract information from PDF and return as JSON
    """
    try:
        # Convert PDF to images and extract text
        raw_text = ""
        for page in convert_from_path(file_path):
            raw_text += pytesseract.image_to_string(page, lang='eng')
        print(raw_text)    

        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )

        schema = {
            "properties": {
                "name": {"type": "string", "description": "Full name of the person"},
                "id_number": {"type": "string", "description": "ID number"},
                "date_of_birth": {"type": "string", "description": "Date of birth"},
                "nationality": {"type": "string", "description": "Nationality"},
                "issue_date": {"type": "string", "description": "Document issue date"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "gender": {"type": "string", "description": "Gender"},
                "card_number": {"type": "string", "description": "Document identification number or Card Number"}
            },
            "required": ["name","id_number","gender"]
        }

        # Extract information
        extracted_content = create_extraction_chain(schema, llm).run(raw_text)

        # Combine results into single dictionary
        result = {}
        for item in extracted_content:
            for key, value in item.items():
                if value and value.strip():
                    if key == "gender":
                        # Map gender values
                        if value.lower() == 'm' or value.lower() == 'male':
                            value = 'male'
                        elif value.lower() == 'f' or value.lower() == 'female': 
                            value = 'female'
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value)
                        else:
                            result[key] = [result[key], value]
                    else:
                        result[key] = value


        return result or {"error": "No information extracted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
   
#Extract text from give  emirate image
async def extract_image_info(file_path: str) -> Dict:
    """
    Extract information from JPG and return as JSON
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        
        # Extract text from JPG image
        raw_text = pytesseract.image_to_string(image, lang='eng')
        print("Extracted text:", raw_text)  # Debugging step to verify OCR extraction
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        schema = {
            "properties": {
                "name": {"type": "string", "description": "Full name of the person"},
                "id_number": {"type": "string", "description": "ID number"},
                "date_of_birth": {"type": "string", "description": "Date of birth"},
                "nationality": {"type": "string", "description": "Nationality"},
                "issue_date": {"type": "string", "description": "Document issue date"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "gender": {"type": "string", "description": "Gender"},
                 "card_number": {"type": "string", "description": "Document identification number or Card Number"}
            },
            "required": ["name","id_number","gender"]
        }
        
        # Extract information
        extraction_chain = create_extraction_chain(schema, llm)
        extracted_content = extraction_chain.run(raw_text)
        
        print("Extracted content:", extracted_content)  # Debugging step to verify LLM extraction
        
        # Combine results into single dictionary
        result = {}
        for item in extracted_content:
            for key, value in item.items():
                if value and value.strip():
                    if key == "gender":
                        # Map gender values
                        if value.lower() == 'm':
                            value = 'male'
                        elif value.lower() == 'f':
                            value = 'female'
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value)
                        else:
                            result[key] = [result[key], value]
                    else:
                        result[key] = value
                        
        return result or {"error": "No information extracted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
     
#Extract text from license image
async def extract_image_drving_license(file_path: str) -> Dict:
    """
    Extract information from JPG and return as JSON
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        
        # Extract text from JPG image
        raw_text = pytesseract.image_to_string(image, lang='eng')
        print("Extracted text:", raw_text)  # Debugging step to verify OCR extraction
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        schema = {
            "properties": {
                "name": {"type": "string", "description": "Full name of the person"},
                "license_no": {"type": "string", "description": "License No only six numbers"},
                "date_of_birth": {"type": "string", "description": "Date of birth"},
                "nationality": {"type": "string", "description": "Nationality"},
                "issue_date": {"type": "string", "description": "Document issue date"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "place_of_issue": {"type": "string", "description": "Please select the Place of issue"},
            },
            "required": ["name", "license_no", "date_of_birth", "nationality", "issue_date", "expiry_date", "place_of_issue"]
        }

        
        # Extract information
        extraction_chain = create_extraction_chain(schema, llm)
        extracted_content = extraction_chain.run(raw_text)
        
        print("Extracted content:", extracted_content)  # Debugging step to verify LLM extraction
        
        # Combine results into single dictionary
        result = {}
        for item in extracted_content:
            for key, value in item.items():
                if value and value.strip():
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value)
                        else:
                            result[key] = [result[key], value]
                    else:
                        result[key] = value

        return result or {"error": "No information extracted"}

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
# Extract information from license pdf
async def extract_pdf_drving_license(file_path: str):
    """
    Extract information from PDF and return as JSON
    """
    try:
        # Convert PDF to images and extract text
        raw_text = ""
        for page in convert_from_path(file_path):
            raw_text += pytesseract.image_to_string(page, lang='eng') + "\n"
        print(raw_text)    

        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )

        schema = {
            "properties": {
                "name": {"type": "string", "description": "Full name of the person"},
                "license_no": {"type": "string", "description": "License No only six numbers"},
                "date_of_birth": {"type": "string", "description": "Date of birth"},
                "nationality": {"type": "string", "description": "Nationality"},
                "issue_date": {"type": "string", "description": "Document issue date"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "place_of_issue": {"type": "string", "description": "Please select the Place of issue"},
            },
            "required": ["name", "license_no", "date_of_birth", "nationality", "issue_date", "expiry_date", "place_of_issue"]
        }

        # Extract information
        extracted_content = create_extraction_chain(schema, llm).run(raw_text)

        # Combine results into a single dictionary
        result = {}
        for item in extracted_content:
            for key, value in item.items():
                if value and value.strip():
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value)
                        else:
                            result[key] = [result[key], value]
                    else:
                        result[key] = value

        return result or {"error": "No information extracted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    

    
#Extract text from license image
async def extract_image_mulkiya(file_path: str) -> Dict:
    """
    Extract information from JPG and return as JSON
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image to improve OCR accuracy
        
        # Extract text from JPG image
        raw_text = pytesseract.image_to_string(image, lang='eng')
        print("Extracted text:", raw_text)  # Debugging step to verify OCR extraction
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        schema = {
            "properties": {
                "owner": {"type": "string", "description": "Full name of the owner"},
                "traffic_plate_no": {"type": "string", "description": "The Traffic Plate Number example:1/6233"},
                "tc_no": {"type": "string", "description": "T C Number"},
                "nationality": {"type": "string", "description": "Nationality"},
                "reg_date": {"type": "string", "description": "Registeration Date (Reg. Date)"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "ins_exp": {"type": "string", "description": "insurances expiry date"},
                "policy_no": {"type": "string", "description": "Policy number"},
                "place_of_issue": {"type": "string", "description": "Please select the Place of issue"},
                "model_no": {"type": "string", "description": "Vehicle Model number"},
                "number_of_pass": {"type": "string", "description": "Number of Pass"},
                "origin": {"type": "string", "description": "Vehicle Origin"},
                "vehicle_type": {"type": "string", "description": "Veh, Type"},
                "vehicle_empty_weight": {"type": "string", "description": "Vehicle Empty Weight"},
                "engine_no": {"type": "string", "description": "Vehicle Engine number"},
                "chassis_no": {"type": "string", "description": "Vehicle Chassis number"},
            },
            "required": ["owner", "traffic_plate_no","tc_no","nationality","reg_date","expiry_date","ins_exp","policy_no","place_of_issue","model_no","number_of_pass","origin","vehicle_type","vehicle_empty_weight","engine_no","chassis_no"]
        }

        # Extract information
        extraction_chain = create_extraction_chain(schema, llm)
        extracted_content = extraction_chain.run(raw_text)
        
        print("Extracted content:", extracted_content)  # Debugging step to verify LLM extraction
        
        # Combine results into single dictionary
        result = {}
        for item in extracted_content:
            for key, value in item.items():
                if value and value.strip():
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value)
                        else:
                            result[key] = [result[key], value]
                    else:
                        result[key] = value

        return result or {"error": "No information extracted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
# Extract information from license pdf
async def extract_pdf_mulkiya(file_path: str):
    """
    Extract information from PDF and return as JSON
    """
    try:
        # Convert PDF to images and extract text
        raw_text = ""
        for page in convert_from_path(file_path):
            raw_text += pytesseract.image_to_string(page, lang='eng') + "\n"
        print(raw_text)    

        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )

        schema = {
            "properties": {
                "owner": {"type": "string", "description": "Full name of the owner"},
                "traffic_plate_no": {"type": "string", "description": "The Traffic Plate Number example:1/6233"},
                "tc_no": {"type": "string", "description": "T C Number"},
                "nationality": {"type": "string", "description": "Nationality"},
                "reg_date": {"type": "string", "description": "Registeration Date (Reg. Date)"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "ins_exp": {"type": "string", "description": "insurances expiry date"},
                "policy_no": {"type": "string", "description": "Policy number"},
                "place_of_issue": {"type": "string", "description": "Please select the Place of issue"},
                "model_no": {"type": "string", "description": "Vehicle Model number"},
                "number_of_pass": {"type": "string", "description": "Number of Pass"},
                "origin": {"type": "string", "description": "Vehicle Origin"},
                "vehicle_type": {"type": "string", "description": "Vehicle Type"},
                "vehicle_empty_weight": {"type": "string", "description": "Vehicle Empty Weight"},
                "engine_no": {"type": "string", "description": "Vehicle Engine number"},
                "chassis_no": {"type": "string", "description": "Vehicle Chassis number"},
            },
            "required": ["owner", "traffic_plate_no","tc_no","nationality","reg_date","expiry_date","ins_exp","policy_no","place_of_issue","model_no","number_of_pass","origin","vehicle_type","vehicle_empty_weight","engine_no","chassis_no"]
        }
        # Extract information
        extracted_content = create_extraction_chain(schema, llm).run(raw_text)

        # Combine results into a single dictionary
        result = {}
        for item in extracted_content:
            for key, value in item.items():
                if value and value.strip():
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value)
                        else:
                            result[key] = [result[key], value]
                    else:
                        result[key] = value

        return result or {"error": "No information extracted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class ChatGroqVisionOCR:
    def __init__(self, api_key=None):
        """
        Initialize ChatGroq Vision OCR client
        
        Args:
            api_key (str, optional): Groq API key
        """
        # Use API key from environment or passed parameter
        self.api_key = "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"
        
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
    
    def extract_and_fill_schema(self, image_path):
        """
        Extract text from image and fill schema
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Filled schema
        """
        # Perform OCR to extract text
        extracted_text = self.perform_ocr(image_path)
        
        # Initialize schema
        filled_schema = {
            "owner": "",
            "traffic_plate_no": "",
            "tc_no": "",
            "nationality": "",
            "reg_date": "",
            "expiry_date": "",
            "ins_exp": "",
            "policy_no": "",
            "place_of_issue": "",
            "model_no": "",
            "number_of_pass": "",
            "origin": "",
            "vehicle_type": "",
            "vehicle_empty_weight": "",
            "engine_no": "",
            "chassis_no": ""
        }
        
        # Parse extracted text to fill schema (this is a simplified example)
        for line in extracted_text.split('\n'):
            if "owner" in line.lower():
                filled_schema["owner"] = line.split(":")[-1].strip()
            elif "traffic plate no" in line.lower():
                filled_schema["traffic_plate_no"] = line.split(":")[-1].strip()
            elif "tc no" in line.lower():
                filled_schema["tc_no"] = line.split(":")[-1].strip()
            elif "nationality" in line.lower():
                filled_schema["nationality"] = line.split(":")[-1].strip()
            elif "reg date" in line.lower():
                filled_schema["reg_date"] = line.split(":")[-1].strip()
            elif "expiry date" in line.lower():
                filled_schema["expiry_date"] = line.split(":")[-1].strip()
            elif "ins exp" in line.lower():
                filled_schema["ins_exp"] = line.split(":")[-1].strip()
            elif "policy no" in line.lower():
                filled_schema["policy_no"] = line.split(":")[-1].strip()
            elif "place of issue" in line.lower():
                filled_schema["place_of_issue"] = line.split(":")[-1].strip()
            elif "model" in line.lower():
                filled_schema["model_no"] = line.split(":")[-1].strip()
            elif "number of pass" in line.lower():
                filled_schema["number_of_pass"] = line.split(":")[-1].strip()
            elif "origin" in line.lower():
                filled_schema["origin"] = line.split(":")[-1].strip()
            elif "vehicle type" in line.lower():
                filled_schema["vehicle_type"] = line.split(":")[-1].strip()
            elif "empty weight" in line.lower():
                filled_schema["vehicle_empty_weight"] = line.split(":")[-1].strip()
            elif "engine no" in line.lower():
                filled_schema["engine_no"] = line.split(":")[-1].strip()
            elif "chassis no" in line.lower():
                filled_schema["chassis_no"] = line.split(":")[-1].strip()
        
        return filled_schema
    
    def perform_ocr(self, image_path, prompt=None):
        """
        Perform OCR using ChatGroq
        
        Args:
            image_path (str): Path to the image file
            prompt (str, optional): Custom OCR instruction
        
        Returns:
            str: Extracted text from the image
        """
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Default prompt if not provided
        default_prompt = (
            "Carefully extract all text from this image. "
            "Preserve the original formatting and reading order. "
            "If there are multiple text sections, organize them clearly."
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
                        "text": prompt or default_prompt
                    }
                ]
            )
            
            # Invoke OCR
            response = self.chat.invoke([msg])
            
            return response.content
        
        except Exception as e:
            print(f"OCR Error: {e}")
            return None
