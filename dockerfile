# Dockerfile
FROM python:3.11-slim

# Helpful defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Workdir
WORKDIR /app

# (Optional) bump to force a rebuild if cache gets sticky
ARG BUILD_ID=1

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Railway sets $PORT at runtime; default to 8000 for local dev
ENV PORT=8000
EXPOSE 8000

# Start FastAPI with Uvicorn at **runtime** (not build time!)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
