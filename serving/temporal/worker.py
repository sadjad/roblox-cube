"""
Temporal Worker for Cube3D Pipeline.

This worker executes the text-to-mesh workflow activities,
calling Triton servers for each inference stage.
"""

import asyncio
import logging
import os
import sys

from temporalio.client import Client
from temporalio.worker import Worker

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal.workflows import TextToMeshWorkflow
from temporal.workflows.text_to_mesh import TokensToMeshWorkflow
from temporal.activities import encode_text, generate_tokens, decode_mesh

# Configuration
TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "temporal:7233")
TASK_QUEUE = os.getenv("TASK_QUEUE", "cube3d-mesh-generation")
NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Start the Temporal worker."""
    logger.info(f"Connecting to Temporal at {TEMPORAL_HOST}...")

    # Connect to Temporal
    client = await Client.connect(
        TEMPORAL_HOST,
        namespace=NAMESPACE,
    )

    logger.info(f"Connected to Temporal. Starting worker on queue: {TASK_QUEUE}")

    # Create worker
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            TextToMeshWorkflow,
            TokensToMeshWorkflow,
        ],
        activities=[
            encode_text,
            generate_tokens,
            decode_mesh,
        ],
    )

    # Run worker
    logger.info("Worker started. Waiting for tasks...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
