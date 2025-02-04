FROM python:3.8-slim

#working dir
WORKDIR /app 

RUN apt-get update && apt-get install -y \
    gcc \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirement.txt

RUN mkdir -p /app/result

CMD ["python", "predict.py"]
