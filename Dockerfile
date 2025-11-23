# Base stage
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
# Added 'build-essential' to provide limits.h and other C headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application stage
COPY src/ /app/src/
RUN mkdir -p /app/models
# COPY models/ /app/models/ 

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]