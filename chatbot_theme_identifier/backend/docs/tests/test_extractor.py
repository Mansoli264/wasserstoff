import pytest
from app.api.extractor import extract_text

def test_extract_text_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        extract_text("nonexistent.pdf")

def test_extract_text_empty_pdf(tmp_path):
    # Create an empty PDF file
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    text = extract_text(str(pdf_path))
    assert isinstance(text, str)