# Use lightweight official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for spacy + nltk
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy model + NLTK resources
RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; [__import__('nltk').download(x) for x in ['punkt','stopwords']]"

# Copy app files into container
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app when the container starts
CMD ["streamlit", "run", "smartresearch_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
