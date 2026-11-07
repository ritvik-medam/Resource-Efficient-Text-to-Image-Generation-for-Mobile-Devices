# Dockerfile for Resource-Efficient Text-to-Image Generation (P3)
FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (OpenCV headless & image libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ ./src/
COPY static/ ./static/
COPY eval_P3_T01.md .
COPY DESIGN.md .
COPY docs/ ./docs/

# Expose ports (FastAPI / Render default)
EXPOSE 8000 10000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/api/system-status || exit 1

# Start the application server
CMD ["python", "run.py", "--device", "cpu"]

