# API Gateway for Cube3D Pipeline
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY shared/ /app/shared/

# Copy temporal workflows (needed for type hints)
COPY temporal/ /app/temporal/

# Copy API code
COPY api/ /app/api/

# Set Python path
ENV PYTHONPATH=/app

# Environment variables (override in docker-compose)
ENV TEMPORAL_HOST=temporal:7233
ENV TASK_QUEUE=cube3d-mesh-generation
ENV TEMPORAL_NAMESPACE=default

# Expose port
EXPOSE 8080

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
