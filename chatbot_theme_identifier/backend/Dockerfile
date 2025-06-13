# Use an official Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements file
COPY chatbot_theme_identifier/backend/requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the full source code
COPY . /app

# Expose app port (optional; adjust if needed)
EXPOSE 8000

# Run your app (adjust this to your main script or app entry point)
CMD ["python", "chatbot_theme_identifier/backend/main.py"]
