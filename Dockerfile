FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy app source
COPY . .

# Streamlit config
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
