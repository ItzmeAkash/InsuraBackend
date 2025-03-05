from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse,JSONResponse
from routes.utils import ChatGroqVisionOCR
from routes.utils import extract_image_drving_license, extract_image_info, extract_image_mulkiya, extract_pdf_drving_license, extract_pdf_info, extract_pdf_mulkiya
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import os
import tempfile
import asyncio


# Initialize Route
router = APIRouter()

user_states = {}

async def clear_user_states():
    while True:
        await asyncio.sleep(86400)  # 24 hours in seconds
        user_states.clear()
        print("User states cleared")

@router.on_event("startup")
async def startup_event():
    asyncio.create_task(clear_user_states())
    
@router.post("/extract-pdf/", tags=["PDF Processing"])
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...)):
    user_id = user_id.strip()
    print("pdf",user_id)
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file
        }
    
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
async def upload_image(file: UploadFile = File(...), user_id: str = Form(...)):
    user_id = user_id.strip()
    print("image",user_id)
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file
        }
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
            


#Get the pdf document
 
@router.get("/pdf/{document_name}")
def get_pdf(document_name: str):
    pdf_path = f"pdf/{document_name}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type='application/pdf', filename=f"{document_name}.pdf")
    else:
        raise HTTPException(status_code=404, detail="Document not found")
    
#Get all trhe pdf
@router.get("/pdfs", tags=["PDF Processing"])
def get_all_pdfs():
    pdf_directory = "pdf"
    try:
        # List all files in the pdf directory
        files = os.listdir(pdf_directory)
        # Filter to include only .pdf files
        pdf_files = [file for file in files if file.endswith('.pdf')]
        return {"pdf_files": pdf_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
    
    
@router.post("/extract-pdf-licence/", tags=["PDF Processing"])
async def upload_lience_pdf(file: UploadFile = File(...),user_id: str = Form(...)):
    user_id = user_id.strip()
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file
        }
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
            result = await extract_pdf_drving_license(temp_file.name)
            
            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            





@router.post("/extract-image-licence/", tags=["Image Processing"])
async def upload_liences_image(file: UploadFile = File(...),user_id: str = Form(...)):
    user_id = user_id.strip()
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file
        }
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
            result = await extract_image_drving_license(temp_file.name)
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            
       
 
   
@router.post("/extract-pdf-mulkiya/", tags=["PDF Processing"])
async def upload_lience_pdf(file: UploadFile = File(...),user_id: str = Form(...)):
    user_id = user_id.strip()
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file
        }
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
            result = await extract_pdf_mulkiya(temp_file.name)
            
            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            




@router.post("/extract-image-mulkiya/", tags=["Image Processing"])
async def upload_licence_images(files: List[UploadFile] = File(...),user_id: str = Form(...)):
    user_id = user_id.strip()
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file
        }
    """
    Upload multiple image files and extract information from them.
    
    Args:
        files (List[UploadFile]): List of image files to process
    
    Returns:
        List[dict]: Extracted information from each image
    """
    # Check for valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    # List to store results
    extraction_results = []
    
    # Process each file
    for file in files:
        # Validate file extension
        if not file.filename.lower().endswith(valid_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {file.filename}. Only {', '.join(valid_extensions)} are allowed."
            )
        
        # Create a temporary file with correct image extension
        file_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            try:
                # Write the uploaded file content to temporary file
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                ocr_client = ChatGroqVisionOCR()
                # Process the image and extract information
                result = ocr_client.extract_and_fill_schema(temp_file.name)
                
                # Add filename to the result for reference
                result['filename'] = file.filename
                extraction_results.append(result)
            
            except Exception as e:
                # Log the error and continue processing other files
                print(f"Error processing {file.filename}: {str(e)}")
                extraction_results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
            
            finally:
                # Clean up the temporary file
                os.unlink(temp_file.name)
    
    return extraction_results
