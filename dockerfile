# Use Python slim image
FROM python:3.10-slim

# Environment defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Working dir
WORKDIR /app

# Cache-buster (bump if rebuild needed)
ARG BUILD_ID=1

# Install system libs required by DeepFace/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default port for local dev (Railway overrides at runtime)
ENV PORT=8000
EXPOSE 8000

# No start command here – Railway Start Command will be used
