# Hugging Face Docker Space image for Aldigram / IACT.
# Build context: repository root. App listens on port 7860 (Space default).

FROM node:20-alpine AS frontend
WORKDIR /app
COPY IACTDashboard.jsx ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
WORKDIR /app/frontend
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Empty value = same-origin API (the backend serves this build).
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch keeps the image far smaller than the CUDA default.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r backend/requirements.txt

# Bake the multilingual model into the image for fast, offline startup.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY backend ./backend
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN useradd -m -u 1000 user \
 && chown -R user:user /app /opt/hf-cache
USER user

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
