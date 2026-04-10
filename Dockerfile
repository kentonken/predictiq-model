# 1. Use a stable Python base
FROM python:3.11-slim

# 2. Install the missing C++ library for LightGBM/XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set up the working directory
WORKDIR /app

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your PredictIQ Pro source code
COPY . .

# 6. Expose the port Railway expects
EXPOSE 8080

# 7. Start the server using a shell to correctly map the Railway $PORT variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
