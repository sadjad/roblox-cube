# Temporal Worker for Cube3D Pipeline
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY temporal/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY shared/ /app/shared/

# Copy temporal code
COPY temporal/ /app/temporal/

# Set Python path
ENV PYTHONPATH=/app

# Environment variables (override in docker-compose)
ENV TEMPORAL_HOST=temporal:7233
ENV TASK_QUEUE=cube3d-mesh-generation
ENV TEMPORAL_NAMESPACE=default
ENV CLIP_TRITON_URL=triton-clip:8001
ENV GPT_TRITON_URL=triton-gpt:8001
ENV MESH_TRITON_URL=triton-mesh:8001

# Run worker
CMD ["python", "temporal/worker.py"]
