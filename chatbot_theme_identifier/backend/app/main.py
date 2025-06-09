# backend/app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .extractor import extract_text
from .preprocessor import clean_text
from .theme_analyzer import detect_themes
import os

app = FastAPI()

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
        f.write(await file.read())

    # Extract and clean text
    extracted_text = extract_text(file_location)
    cleaned_text = clean_text(extracted_text)

    # Save cleaned text
    output_path = f"extracted/{file.filename}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # Detect themes
    themes = detect_themes(cleaned_text)

            # Detect themes
    themes = detect_themes(cleaned_text)

    return {
        "message": "File uploaded, processed, and analyzed!",
        "filename": file.filename,
        "text_preview": cleaned_text[:300],
        "themes_detected": themes
    }