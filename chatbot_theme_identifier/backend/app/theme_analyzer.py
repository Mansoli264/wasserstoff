import os
import openai
from typing import List

openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_themes(text: str, max_themes: int = 5) -> List[str]:
    """
    Uses OpenAI GPT to extract main themes from the provided text.
    Returns a list of themes.
    """
    prompt = (
        "Analyze the following document and list the main themes or topics it covers. "
        f"Limit your answer to {max_themes} concise themes, separated by commas.\n\n"
        f"Document:\n{text}\n\nThemes:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts themes from documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.3,
        n=1,
        stop=None,
    )
    # Extract and clean the response
    answer = response.choices[0].message["content"]
    # Split by commas and strip whitespace
    themes = [theme.strip() for theme in answer.split(",") if theme.strip()]
    return themes

# Example usage:
# themes = analyze_themes("Your document text here")