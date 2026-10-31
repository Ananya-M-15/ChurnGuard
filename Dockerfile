# ==============================================================================
# ChurnGuard — Multi-Service Production Dockerfile
# Base image: Python 3.11-slim (high stability, lightweight, fast build)
# ==============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY api/ /app/api/
COPY src/ /app/src/
COPY dashboard/ /app/dashboard/
COPY monitoring/ /app/monitoring/
COPY models/ /app/models/
COPY reports/ /app/reports/
COPY data/processed/ /app/data/processed/
COPY .streamlit/ /app/.streamlit/

# Expose API (8000) and Dashboard (8501) ports
EXPOSE 8000 8501

# Default command: launch FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
