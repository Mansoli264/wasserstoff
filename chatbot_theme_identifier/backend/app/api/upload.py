from fastapi import APIRouter, UploadFile, File
from app.services import ocr_parser
import os
from app.config import UPLOAD_DIR
from app.services import chunking, embedding

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/", summary="Upload a file", response_description="Preview of extracted content")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for processing.

    - **file**: The file to upload (PDF).
    - **returns**: A preview of the extracted text content.
    """
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    extracted_text = ocr_parser.extract_text(file_location)

    chunks = chunking.chunk_text(extracted_text)     
    embedding.store_document_chunks(chunks)         

    return {
        "filename": file.filename,
        "content": extracted_text[:1000]  # preview only
    }