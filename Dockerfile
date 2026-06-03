# Multi-stage build is used to keep the final image clean and small.
# Stage 1: Build dependencies and wheels
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to builder
COPY requirements.txt .

# Install dependencies into a wheels directory to avoid re-compilation in final stage
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# Stage 2: Final minimal runtime image
FROM python:3.11-slim

LABEL maintainer="Enterprise RAG Team"
LABEL description="Docker image for Enterprise RAG System containing FastAPI Backend and Streamlit Frontend"

WORKDIR /app

# Install runtime dependencies (e.g. git or curl if needed, but not strictly required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder stage
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

# Create non-root user for security compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10000 -g appgroup -m -s /bin/bash appuser

# Create upload, parent store, and database directories with correct ownership
RUN mkdir -p /app/data /app/chroma_db /app/parent_store /app/logs && \
    chown -R appuser:appgroup /app

# Copy application source code
COPY --chown=appuser:appgroup backend/ /app/backend/
COPY --chown=appuser:appgroup frontend/ /app/frontend/
COPY --chown=appuser:appgroup .env /app/.env

# Switch to non-root user
USER appuser

# Set environmental defaults for container runs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    UPLOAD_DIR=/app/data \
    PARENT_STORE_DIR=/app/parent_store \
    SEMANTIC_CACHE_DB=/app/data/semantic_cache.db

# Expose backend (8000) and frontend (8501) ports
EXPOSE 8000
EXPOSE 8501

# The actual command is overridden in docker-compose for each service
CMD ["python", "-m", "backend.main"]
