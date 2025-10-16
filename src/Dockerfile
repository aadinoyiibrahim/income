FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/

VOLUME ["/app/data", "/app/models", "/app/result"]

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]