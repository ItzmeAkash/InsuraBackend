from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse,JSONResponse
from routes.utils import extract_back_page_emirate, extract_front_page_emirate, extract_image_driving_license,extract_image_info1, extract_image_mulkiya, extract_pdf_driving_license,extract_pdf_info1, extract_pdf_mulkiya
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import os
import tempfile
import asyncio
import logging
from cachetools import TTLCache

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
# Initialize Route
router = APIRouter()

user_states = TTLCache(maxsize=1000, ttl=3600)

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
            if len(await file.read()) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            file.file.seek(0)
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            # Process the PDF and extract information
            result = await extract_pdf_info1(temp_file.name)
            
            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
@router.post("/extract-image/", tags=["Image Processing"])
async def upload_mulkiya_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a PDF or image file and extract licence information from it.
    Supports PDF, JPG, JPEG, and PNG formats.
    """
    user_id = user_id.strip()
    
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file.filename
        }
    
    # Get file extension and convert to lowercase
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file type
    valid_extensions = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png']
    }
    
    # Check if extension is valid
    if file_extension in valid_extensions['pdf']:
        file_type = 'pdf'
    elif file_extension in valid_extensions['image']:
        file_type = 'image'
    else:
        valid_all = valid_extensions['pdf'] + valid_extensions['image']
        raise HTTPException(
            status_code=400,
            detail=f"Only these file types are allowed: {', '.join(valid_all)}"
        )
    
    # Create a temporary file with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the file based on its type
            if file_type == 'pdf':
                result = await extract_pdf_mulkiya(temp_file.name)
            else:  # file_type == 'image'
                result = await extract_image_info1(temp_file.name)
                
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            
@router.post("/extract-emirate/", tags=["Image Processing"])
async def upload_emirate_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a PDF or image file and extract licence information from it.
    Supports PDF, JPG, JPEG, and PNG formats.
    """
    user_id = user_id.strip()
    
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file.filename
        }
    
    # Get file extension and convert to lowercase
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file type
    valid_extensions = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png']
    }
    
    # Check if extension is valid
    if file_extension in valid_extensions['pdf']:
        file_type = 'pdf'
    elif file_extension in valid_extensions['image']:
        file_type = 'image'
    else:
        valid_all = valid_extensions['pdf'] + valid_extensions['image']
        raise HTTPException(
            status_code=400,
            detail=f"Only these file types are allowed: {', '.join(valid_all)}"
        )
    
    # Create a temporary file with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the file based on its type
            if file_type == 'pdf':
                result = await extract_pdf_info1(temp_file.name)
            else:  # file_type == 'image'
                result = await extract_image_info1(temp_file.name)
                
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)

          
@router.post("/extract-back-page-emirate/", tags=["Image Processing"])
async def upload_back_emirate_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a PDF or image file and extract licence information from it.
    Supports PDF, JPG, JPEG, and PNG formats.
    """
    user_id = user_id.strip()
    
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file.filename
        }
    
    # Get file extension and convert to lowercase
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file type
    valid_extensions = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png']
    }
    
    # Check if extension is valid
    if file_extension in valid_extensions['pdf']:
        file_type = 'pdf'
    elif file_extension in valid_extensions['image']:
        file_type = 'image'
    else:
        valid_all = valid_extensions['pdf'] + valid_extensions['image']
        raise HTTPException(
            status_code=400,
            detail=f"Only these file types are allowed: {', '.join(valid_all)}"
        )
    
    # Create a temporary file with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the file based on its type
            if file_type == 'pdf':
                result = await extract_pdf_info1(temp_file.name)
            else:  # file_type == 'image'
                result = await extract_back_page_emirate(temp_file.name)
                
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
@router.post("/extract-front-page-emirate/", tags=["Image Processing"])
async def upload_front_emirate_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a PDF or image file and extract licence information from it.
    Supports PDF, JPG, JPEG, and PNG formats.
    """
    user_id = user_id.strip()
    
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file.filename
        }
    
    # Get file extension and convert to lowercase
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file type
    valid_extensions = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png']
    }
    
    # Check if extension is valid
    if file_extension in valid_extensions['pdf']:
        file_type = 'pdf'
    elif file_extension in valid_extensions['image']:
        file_type = 'image'
    else:
        valid_all = valid_extensions['pdf'] + valid_extensions['image']
        raise HTTPException(
            status_code=400,
            detail=f"Only these file types are allowed: {', '.join(valid_all)}"
        )
    
    # Create a temporary file with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the file based on its type
            if file_type == 'pdf':
                result = await extract_pdf_info1(temp_file.name)
            else:  # file_type == 'image'
                result = await extract_front_page_emirate(temp_file.name)
                
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

    
    
    


@router.post("/extract-licence/", tags=["Document Processing"])
async def upload_licence_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a PDF or image file and extract licence information from it.
    Supports PDF, JPG, JPEG, and PNG formats.
    """
    user_id = user_id.strip()
    
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file.filename
        }
    
    # Get file extension and convert to lowercase
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file type
    valid_extensions = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png']
    }
    
    # Check if extension is valid
    if file_extension in valid_extensions['pdf']:
        file_type = 'pdf'
    elif file_extension in valid_extensions['image']:
        file_type = 'image'
    else:
        valid_all = valid_extensions['pdf'] + valid_extensions['image']
        raise HTTPException(
            status_code=400,
            detail=f"Only these file types are allowed: {', '.join(valid_all)}"
        )
    
    # Create a temporary file with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the file based on its type
            if file_type == 'pdf':
                result = await extract_pdf_driving_license(temp_file.name)
            else:  # file_type == 'image'
                result = await extract_image_driving_license(temp_file.name)
                
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            

@router.post("/extract-mulkiya/", tags=["Mulkiya Document Processing"])
async def upload_mulkiya_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a PDF or image file and extract licence information from it.
    Supports PDF, JPG, JPEG, and PNG formats.
    """
    user_id = user_id.strip()
    
    # Initialize user state if not already present
    if user_id not in user_states:
        user_states[user_id] = {
            "document_name": file.filename
        }
    
    # Get file extension and convert to lowercase
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Validate file type
    valid_extensions = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png']
    }
    
    # Check if extension is valid
    if file_extension in valid_extensions['pdf']:
        file_type = 'pdf'
    elif file_extension in valid_extensions['image']:
        file_type = 'image'
    else:
        valid_all = valid_extensions['pdf'] + valid_extensions['image']
        raise HTTPException(
            status_code=400,
            detail=f"Only these file types are allowed: {', '.join(valid_all)}"
        )
    
    # Create a temporary file with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write the uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the file based on its type
            if file_type == 'pdf':
                result = await extract_pdf_mulkiya(temp_file.name)
            else:  # file_type == 'image'
                result = await extract_image_mulkiya(temp_file.name)
                
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            