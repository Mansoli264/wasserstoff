from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..services.pdf_text_extractor import extract_text_from_pdf
from ..services.nlp_model import analyze_text

router = APIRouter()

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    file_bytes = await file.read()
    extracted_text = extract_text_from_pdf(file_bytes)

    if not extracted_text.strip():
        return JSONResponse(status_code=400, content={"detail": "No text found in PDF"})
    
    analysis_result = analyze_text(extracted_text)

    return {"analysis": analysis_result}
