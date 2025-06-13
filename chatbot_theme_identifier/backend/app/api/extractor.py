import os
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
from typing import Optional

def extract_text_from_pdf(filepath: str, ocr: bool = False) -> str:
    """
    Extracts text from a PDF file.
    If ocr is True, applies OCR to each page image.
    """
    text = ""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if ocr:
        # OCR each page as image
        with fitz.open(filepath) as doc:
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
    else:
        # Try extracting text directly
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        # Fallback to OCR if no text found
        if not text.strip():
            return extract_text_from_pdf(filepath, ocr=True)
    return text.strip()

def extract_text(filepath: str, ocr: bool = False) -> str:
    """
    General entry point for extracting text from a file.
    Currently supports PDF files.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath, ocr=ocr)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Example usage:
# text = extract_text("path/to/file.pdf")