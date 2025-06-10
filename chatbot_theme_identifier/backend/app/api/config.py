import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
UPLOAD_DIR = os.path.join(BASE_DIR, "../../data")

# Models/API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")