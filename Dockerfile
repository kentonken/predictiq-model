# 1. Base Image
FROM python:3.11-slim

# 2. Install the missing library (libgomp1) for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup App
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 4. Correct Port Mapping
# We use the shell form here so Railway can inject the $PORT integer correctly
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
