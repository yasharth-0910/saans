# Saans API for Hugging Face Spaces (Docker SDK). Build from the repo root.
FROM python:3.11-slim

# ffmpeg for video; libgl1 and libglib2.0-0 for opencv-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000, so the app (and data/, where the db and uploads go) is owned by it.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --user -r backend/requirements.txt

COPY --chown=user backend backend
COPY --chown=user data/grap_rules.json data/grap_rules.json

# MODEL_API_KEY, USE_MOCK and FRONTEND_ORIGIN come from the Space's secrets and variables.
EXPOSE 7860
CMD ["uvicorn", "--app-dir", "backend", "main:app", "--host", "0.0.0.0", "--port", "7860"]
