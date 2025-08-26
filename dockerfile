FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
ARG BUILD_ID=9

# OS libs for OpenCV/DeepFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Let Railway set Start Command in UI, or uncomment a CMD:
# CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","${PORT}"]
