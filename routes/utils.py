import logging
import re
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
import json

from routes.VisionModel import DocumentVisionOCR

# #Extract text from the  give emirate pdf
# async def extract_pdf_info(file_path: str):
#     """
#     Extract information from PDF and return as JSON
#     """
#     try:
#         # Convert PDF to images and extract text
#         raw_text = ""
#         for page in convert_from_path(file_path):
#             raw_text += pytesseract.image_to_string(page, lang='eng')
#         print(raw_text)    

#         # Initialize LLM and create extraction chain
#         llm = ChatGroq(
#             model=os.getenv('LLM_MODEL'),
#             temperature=0,
#             api_key=os.getenv('GROQ_API_KEY')
#         )

#         schema = {
#             "properties": {
#                 "name": {"type": "string", "description": "Full name of the person"},
#                 "id_number": {"type": "string", "description": "ID number"},
#                 "date_of_birth": {"type": "string", "description": "Date of birth"},
#                 "nationality": {"type": "string", "description": "Nationality"},
#                 "issue_date": {"type": "string", "description": "Document issue date"},
#                 "expiry_date": {"type": "string", "description": "Document expiry date"},
#                 "gender": {"type": "string", "description": "Gender"},
#                 "card_number": {"type": "string", "description": "Document identification number or Card Number"}
#             },
#             "required": ["name","id_number","gender"]
#         }

#         # Extract information
#         extracted_content = create_extraction_chain(schema, llm).run(raw_text)

#         # Combine results into single dictionary
#         result = {}
#         for item in extracted_content:
#             for key, value in item.items():
#                 if value and value.strip():
#                     if key == "gender":
#                         # Map gender values
#                         if value.lower() == 'm' or value.lower() == 'male':
#                             value = 'male'
#                         elif value.lower() == 'f' or value.lower() == 'female': 
#                             value = 'female'
#                     if key in result:
#                         if isinstance(result[key], list):
#                             result[key].append(value)
#                         else:
#                             result[key] = [result[key], value]
#                     else:
#                         result[key] = value


#         return result or {"error": "No information extracted"}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
   
   
# #Extract text from give  emirate image
# async def extract_image_info(file_path: str) -> Dict:
#     """
#     Extract information from JPG and return as JSON
#     """
#     try:
#         # Preprocess the image
#         image = Image.open(file_path)
#         image = image.convert('L')  # Convert to grayscale
#         image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        
#         # Extract text from JPG image
#         raw_text = pytesseract.image_to_string(image, lang='eng')
#         print("Extracted text:", raw_text)  # Debugging step to verify OCR extraction
        
#         # Initialize LLM and create extraction chain
#         llm = ChatGroq(
#             model=os.getenv('LLM_MODEL'),
#             temperature=0,
#             api_key=os.getenv('GROQ_API_KEY')
#         )
        
#         schema = {
#             "properties": {
#                 "name": {"type": "string", "description": "Full name of the person"},
#                 "id_number": {"type": "string", "description": "ID number"},
#                 "date_of_birth": {"type": "string", "description": "Date of birth"},
#                 "nationality": {"type": "string", "description": "Nationality"},
#                 "issue_date": {"type": "string", "description": "Document issue date"},
#                 "expiry_date": {"type": "string", "description": "Document expiry date"},
#                 "gender": {"type": "string", "description": "Gender"},
#                 "card_number": {"type": "string", "description": "Document identification number or Card Number"}
#             },
#             "required": ["name","id_number","gender"]
#         }
        
#         # Extract information
#         extraction_chain = create_extraction_chain(schema, llm)
#         extracted_content = extraction_chain.run(raw_text)
        
#         print("Extracted content:", extracted_content)  # Debugging step to verify LLM extraction
        
#         # Combine results into single dictionary
#         result = {}
#         for item in extracted_content:
#             for key, value in item.items():
#                 if value and value.strip():
#                     if key == "gender":
#                         # Map gender values
#                         if value.lower() == 'm':
#                             value = 'male'
#                         elif value.lower() == 'f':
#                             value = 'female'
#                     if key in result:
#                         if isinstance(result[key], list):
#                             result[key].append(value)
#                         else:
#                             result[key] = [result[key], value]
#                     else:
#                         result[key] = value
                        
#         return result or {"error": "No information extracted"}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
     
# #Extract text from license image
    
    
    
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
    
 #?Image
 
 
#?Tod just image extract
async def extract_image_info1(file_path: str) -> Dict:
    """
    Extract information from  document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the document
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image to improve OCR accuracy
        
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        license_prompt = """
        Extract ALL English text from this license.
        Pay special attention to:
        - Name
        - Id Number
        - Date of Birth
        - Nationality
        - Issuing Date
        - Expiry Date
        - Sex
        - Card Number
        - Occupation
        - Employer
        - Issuing Place
        
        Capture all text exactly as shown, preserving numbers and codes precisely.
        If any mentioned information is missing, recheck and extract everything accurately.
        """
        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_from_image(image, prompt=license_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this document.
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
        {{
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "nationality": "",
                "issue_date": "",
                "expiry_date": "",
                "gender": "",
                "card_number": "",
                "occupation": "",
                "employer": "",
                "issuing_place": "",
            }}
        
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result = {
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "nationality": "",
                "issue_date": "",
                "expiry_date": "",
                "gender": "",
                "card_number": "",
                "occupation": "",
                "employer": "",
                "issuing_place": "",
            }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
       

async def extract_front_page_emirate(file_path: str) -> Dict:
    """
    Extract information from  document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the document
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image to improve OCR accuracy
        
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        license_prompt = """
        Extract ALL English text from this license.
        Pay special attention to:
        - Name
        - Id Number
        - Date of Birth
        - Nationality
        - Issuing Date
        - Expiry Date
        - Sex
       
        
        Capture all text exactly as shown, preserving numbers and codes precisely.
        If any mentioned information is missing, recheck and extract everything accurately.
        """
        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_from_image(image, prompt=license_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this document.
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
        {{
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "nationality": "",
                "issue_date": "",
                "expiry_date": "",
                "gender": "",
              
            }}
        
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result = {
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "nationality": "",
                "issue_date": "",
                "expiry_date": "",
                "gender": "",

            }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
       
       
async def extract_back_page_emirate(file_path: str) -> Dict:
    """
    Extract information from back page document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the document
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image to improve OCR accuracy
        
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        license_prompt = """
        Extract ALL English text from this license.
        Pay special attention to:
        - Card Number
        - Occupation
        - Employer
        - Issuing Place
        
        Capture all text exactly as shown, preserving numbers and codes precisely.
        If any mentioned information is missing, recheck and extract everything accurately.
        """
        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_from_image(image, prompt=license_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this document.
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
        {{
                "card_number": "",
                "occupation": "",
                "employer": "",
                "issuing_place": "",
            }}
        
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result = {

                "card_number": "",
                "occupation": "",
                "employer": "",
                "issuing_place": "",
            }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
           

async def extract_pdf_info1(file_path: str) -> Dict:
    """
    Extract information from JPG License document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the driving license document
    """
    try:
        # Preprocess the image
       
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        emirate_prompt = """
        Extract all English text from this license. 

        Pay special attention to the following details:
        - Name  
        - ID Number (Ensure the ID number starts in the format: 784-YYYY-123456-9. This is an example.)  
        - Date of Birth  
        - Nationality  
        - Issuing Date  
        - Expiry Date  
        - Sex  
        - Card Number  
        - Occupation  
        - Employer  
        - Issuing Place  

        Capture all text exactly as shown, preserving numbers and codes precisely.  
        Ensure that all the listed details are extracted from the given document.  
        If any mentioned information is missing, recheck and extract everything accurately.
        """

        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_to_string(file_path, prompt=emirate_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this Emirate document
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
       {{
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "nationality": "",
                "issue_date": "",
                "expiry_date": "",
                "gender": "",
                "card_number": "",
                "occupation": "",
                "employer": "",
                "issuing_place": "",
            }}
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result =  {
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "nationality": "",
                "issue_date": "",
                "expiry_date": "",
                "gender": "",
                "card_number": "",
                "occupation": "",
                "employer": "",
                "issuing_place": "",
            }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



#Todo
async def extract_image_driving_license(file_path: str) -> Dict:
    """
    Extract information from JPG License document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the driving license document
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image to improve OCR accuracy
        
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        license_prompt = """
        Extract ALL English text from this license.
        Pay special attention to:
        - Name
        - License No
        - Date of Birth
        - Issue Date
        - Expiry Date
        - Nationality
        - Place of Issue
        - Traffic Code No
        - Permitted Vehicles
        
        Capture all text exactly as shown, preserving numbers and codes precisely.
        If any mentioned information is missing, recheck and extract everything accurately.
        """
        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_from_image(image, prompt=license_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this Driving License.
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
        {{
            "name": "",
            "license_no": "",
            "date_of_birth": "",
            "nationality": "",
            "issue_date": "",
            "expiry_date": "",
            "traffic_code_no": "",
            "place_of_issue": "",
            "permitted_vehicles": ""
        }}
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result = {
            "name": "",
            "license_no": "",
            "date_of_birth": "",
            "nationality": "",
            "issue_date": "",
            "expiry_date": "",
            "traffic_code_no": "",
            "place_of_issue": "",
            "permitted_vehicles": ""
        }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
    
    
    

#Todo
async def extract_pdf_driving_license(file_path: str) -> Dict:
    """
    Extract information from JPG License document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the driving license document
    """
    try:
        # Preprocess the image
       
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        license_prompt = """
        Extract ALL English text from this license.
        Pay special attention to:
        - Name
        - License No
        - Date of Birth
        - Issue Date
        - Expiry Date
        - Nationality
        - Place of Issue
        - Traffic Code No
        - Permitted Vehicles
        
        Capture all text exactly as shown, preserving numbers and codes precisely.
        make sure to extract all provided information in the give document
        If any mentioned information is missing, recheck and extract everything accurately.
        """
        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_to_string(file_path, prompt=license_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this Driving License.
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
        {{
            "name": "",
            "license_no": "",
            "date_of_birth": "",
            "nationality": "",
            "issue_date": "",
            "expiry_date": "",
            "traffic_code_no": "",
            "place_of_issue": "",
            "permitted_vehicles": ""
        }}
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result = {
            "name": "",
            "license_no": "",
            "date_of_birth": "",
            "nationality": "",
            "issue_date": "",
            "expiry_date": "",
            "traffic_code_no": "",
            "place_of_issue": "",
            "permitted_vehicles": ""
        }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
    
    
    


#Todo Mulkiya
async def extract_image_mulkiya(file_path: str) -> Dict:
    """
    Extract information from JPG License document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the driving license document
    """
    try:
        # Preprocess the image
        image = Image.open(file_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((image.width * 2, image.height * 2))  # Resize to improve OCR accuracy
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image to improve OCR accuracy
        
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        license_prompt = """
        Extract ALL English text from this driving license or mulkiya.
        Pay special attention to:
        - Owner
        - Traffic Plate No 
        - T.C. No.
        - Place of Issue
        - Nationality
        - Exp Date
        - Reg Date
        - Ins Exp
        - Policy No
        - Mortgage By
        - Model
        - Num of Pass
        - Origin
        - Vechile Type
        - G V W
        - Empty Weight
        - Engine No
        - Chassis No
        
        Capture all text exactly as shown, preserving numbers and codes precisely.
        If any mentioned information is missing, recheck and extract everything accurately.
        """
        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_from_image(image, prompt=license_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this Driving License.
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
        {{
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
         "empty_weight": "",
         "engine_no": "",
         "chassis_no": ""
         "gvw": "",
        }}
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result = {
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
         "empty_weight": "",
         "engine_no": "",
         "chassis_no": ""
        }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
    
    
    
    


async def extract_pdf_mulkiya(file_path: str) -> Dict:
    """
    Extract information from JPG License document and return as JSON
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        Dict: Structured information extracted from the driving license document
    """
    try:
        # Preprocess the image
       
        # Extract text from JPG image
        vision_model = DocumentVisionOCR()
        
        # Create a specialized prompt for license documents
        mulkiya_prompt = """
        Extract ALL English text from this driving license or mulkiya.
        Pay special attention to:
        - Owner
        - Traffic Plate No  
        - T.C. No.
        - Place of Issue
        - Nationality
        - Exp Date
        - Reg Date
        - Ins Exp
        - Policy No
        - Mortgage By
        - Model
        - Num of Pass
        - Origin
        - Vechile Type
        - G V W
        - Empty Weight
        - Engine No
        - Chassis No
        
        Capture all text exactly as shown, preserving numbers and codes precisely.
        If any mentioned information is missing, recheck and extract everything accurately.
        """
        
        # Use the extract_text_from_image method with the preprocessed image
        vision_text = vision_model.extract_text_to_string(file_path, prompt=mulkiya_prompt)
        logging.info("Extracted text from license document")
        
        # Initialize LLM and create extraction chain
        llm = ChatGroq(
            model=os.getenv('LLM_MODEL'),
            temperature=0,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Enhanced extraction prompt to ensure structured JSON output
        extraction_prompt = f"""
        Extract the following information from this Driving License.
        Respond with ONLY a valid JSON object - no explanations, no markdown formatting.
        
        For dates, use format DD-MM-YYYY if possible.
        For numbers and codes, preserve exact formatting including any special characters.
        If a piece of information is not found, use an empty string.
        
        Text to extract from:
        {vision_text}
        
        JSON format:
        {{
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
         "empty_weight": "",
         "engine_no": "",
         "chassis_no": ""
         "gvw": "",
        }}
        
        IMPORTANT: Return ONLY the JSON object with no additional text, code blocks, or explanations.
        """
        
        # Directly use the LLM to extract structured information
        extraction_response = llm.invoke(extraction_prompt)
        extracted_content = extraction_response.content
        logging.info("LLM extraction completed")
        
        # Create default empty result structure
        default_result = {
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
         "empty_weight": "",
         "engine_no": "",
         "chassis_no": ""
        }
        
        # Try to parse the response as JSON
        try:
            # First, attempt to parse as a JSON string
            result = json.loads(extracted_content)
            logging.info("Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            logging.warning(f"Direct JSON parsing failed: {e}")
            try:
                # Find JSON-like content between curly braces
                start = extracted_content.find('{')
                end = extracted_content.rfind('}') + 1
                
                if start >= 0 and end > start:
                    cleaned_content = extracted_content[start:end]
                    # Replace potential newlines, tabs and fix common JSON format issues
                    cleaned_content = cleaned_content.replace('\n', ' ').replace('\t', ' ')
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas
                    
                    result = json.loads(cleaned_content)
                    logging.info("Successfully parsed JSON after cleaning")
                else:
                    raise ValueError("No valid JSON structure found")
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(f"JSON parsing failed: {e}. Creating empty structure.")
                result = default_result
        
        # Ensure all expected keys are present in the result
        for key in default_result:
            if key not in result:
                result[key] = ""
        
        return result
    except Exception as e:
        logging.error(f"Error in extract_image_driving_license: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    