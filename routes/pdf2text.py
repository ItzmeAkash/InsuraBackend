from ast import Dict
from fastapi import APIRouter, File, UploadFile, HTTPException
from langchain_groq import ChatGroq
from langchain.chains import create_extraction_chain
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
import os
import tempfile
load_dotenv()


# Initialize Router
router = APIRouter()

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
                "date_of_birth": {"type": "string", "description": "Date of birth"},
                "passport_number": {"type": "string", "description": "Passport or ID number"},
                "nationality": {"type": "string", "description": "Nationality"},
                "issue_date": {"type": "string", "description": "Document issue date"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "gender": {"type": "string", "description": "Gender"},
                "document_number": {"type": "string", "description": "Document identification number"}
            },
            "required": ["name"]
        }

        # Extract information
        extracted_content = create_extraction_chain(schema, llm).run(raw_text)

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
                "date_of_birth": {"type": "string", "description": "Date of birth"},
                "passport_number": {"type": "string", "description": "Passport or ID number"},
                "nationality": {"type": "string", "description": "Nationality"},
                "issue_date": {"type": "string", "description": "Document issue date"},
                "expiry_date": {"type": "string", "description": "Document expiry date"},
                "gender": {"type": "string", "description": "Gender"},
                "document_number": {"type": "string", "description": "Document identification number"}
            },
            "required": ["name"]
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
    
    
@router.post("/extract-pdf/", tags=["PDF Processing"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and extract information from it.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            # Process the PDF and extract information
            result = await extract_pdf_info(temp_file.name)
            
            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
@router.post("/extract-image/", tags=["Image Processing"])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file and extract information from it.
    """
    # Check for valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Only image files ({', '.join(valid_extensions)}) are allowed"
        )

    # Create a temporary file with correct image extension
    file_extension = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the image and extract information
            result = await extract_image_info(temp_file.name)
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)