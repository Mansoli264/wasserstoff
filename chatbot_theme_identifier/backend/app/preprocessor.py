import re

def clean_text(text: str) -> str:
    """
    Cleans extracted text by:
    - Removing extra whitespace and line breaks
    - Removing non-printable/control characters
    - Optionally, normalizing unicode
    """
    # Remove non-printable/control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

# Example usage:
# cleaned = clean_text(raw_text)