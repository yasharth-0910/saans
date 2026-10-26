# Saans API for Google Cloud Run. Build from the repo root.
FROM python:3.11-slim

# ffmpeg for video; libgl1 and libglib2.0-0 for opencv-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY data/grap_rules.json data/grap_rules.json

# MODEL_API_KEY, USE_MOCK and FRONTEND_ORIGIN come from the Cloud Run service config.
# Cloud Run sets PORT; 8080 is the fallback for local runs.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn --app-dir backend main:app --host 0.0.0.0 --port ${PORT:-8080}"]
