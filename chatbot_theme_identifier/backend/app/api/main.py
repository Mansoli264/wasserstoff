import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("wasserstoff")

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .extractor import extract_text
from .preprocessor import clean_text
from .theme_analyzer import detect_themes
import os

app = FastAPI()

# File size limit middleware (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.headers.get("content-length"):
        if int(request.headers["content-length"]) > MAX_FILE_SIZE:
            logger.warning("Upload rejected: file too large")
            raise HTTPException(status_code=413, detail="File too large")
    return await call_next(request)

# ✅ CORS: Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Use the route expected by frontend: /upload/
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("uploaded", exist_ok=True)
    os.makedirs("extracted", exist_ok=True)

    # Save uploaded file
    file_location = f"uploaded/{file.filename}"
    with open(file_location, "wb") as f:
        file_bytes = await file.read()
        f.write(file_bytes)
    logger.info(f"File uploaded and saved to {file_location}")

    # Extract and clean text
    extracted_text = extract_text(file_location)
    logger.info(f"Text extracted from {file_location} (length: {len(extracted_text)})")
    cleaned_text = clean_text(extracted_text)
    logger.info(f"Text cleaned for {file_location} (length: {len(cleaned_text)})")

    # Save cleaned text
    output_path = f"extracted/{file.filename}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    logger.info(f"Cleaned text saved to {output_path}")

    # Detect themes
    themes = detect_themes(cleaned_text)
    logger.info(f"Themes detected for {file.filename}: {themes}")

    return {
        "message": "File uploaded, processed, and analyzed!",
        "filename": file.filename,
        "text_preview": cleaned_text[:300],
        "themes_detected": themes
    }