"""
Temporal Worker for Cube3D Pipeline.

This worker can be configured to handle:
  - All activities (unified mode) - simpler, for development
  - Specific stage activities (per-stage mode) - for production scaling

Environment Variables:
  TEMPORAL_HOST: Temporal server address (default: temporal:7233)
  TEMPORAL_NAMESPACE: Temporal namespace (default: default)
  WORKER_MODE: "unified" or "per-stage" (default: unified)

  For unified mode:
    TASK_QUEUE: Single queue name (default: cube3d-mesh-generation)

  For per-stage mode:
    WORKER_STAGE: Which stage to handle - "clip", "gpt", "mesh", or "workflow"
    CLIP_TASK_QUEUE: Queue for CLIP activities (default: cube3d-clip)
    GPT_TASK_QUEUE: Queue for GPT activities (default: cube3d-gpt)
    MESH_TASK_QUEUE: Queue for mesh activities (default: cube3d-mesh)
    WORKFLOW_TASK_QUEUE: Queue for workflow orchestration (default: cube3d-mesh-generation)

Scaling Example (per-stage mode):
  Deploy different numbers of workers per stage:

  # 1 workflow worker (orchestration only, no GPU needed)
  WORKER_MODE=per-stage WORKER_STAGE=workflow

  # 1 CLIP worker (fast, ~20ms per request)
  WORKER_MODE=per-stage WORKER_STAGE=clip

  # 10 GPT workers (slow, ~10s per request - BOTTLENECK)
  WORKER_MODE=per-stage WORKER_STAGE=gpt

  # 3 Mesh workers (medium, ~3s per request)
  WORKER_MODE=per-stage WORKER_STAGE=mesh
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

# =============================================================================
# CONFIGURATION
# =============================================================================

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "temporal:7233")
NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")

# Worker mode: "unified" (all activities) or "per-stage" (specific activities)
WORKER_MODE = os.getenv("WORKER_MODE", "unified")

# Unified mode config
UNIFIED_TASK_QUEUE = os.getenv("TASK_QUEUE", "cube3d-mesh-generation")

# Per-stage mode config
WORKER_STAGE = os.getenv("WORKER_STAGE", "")  # clip, gpt, mesh, or workflow
CLIP_TASK_QUEUE = os.getenv("CLIP_TASK_QUEUE", "cube3d-clip")
GPT_TASK_QUEUE = os.getenv("GPT_TASK_QUEUE", "cube3d-gpt")
MESH_TASK_QUEUE = os.getenv("MESH_TASK_QUEUE", "cube3d-mesh")
WORKFLOW_TASK_QUEUE = os.getenv("WORKFLOW_TASK_QUEUE", "cube3d-mesh-generation")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================

async def run_unified_worker(client: Client):
    """
    Run a worker that handles all activities on a single queue.

    This is simpler but doesn't allow independent scaling of stages.
    """
    logger.info(f"UNIFIED mode: all activities on queue '{UNIFIED_TASK_QUEUE}'")

    worker = Worker(
        client,
        task_queue=UNIFIED_TASK_QUEUE,
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

    logger.info("Worker started. Waiting for tasks...")
    await worker.run()


async def run_stage_worker(client: Client, stage: str):
    """
    Run a worker for a specific pipeline stage.

    This allows independent scaling - run more GPT workers for the bottleneck.
    """
    stage_configs = {
        "workflow": {
            "task_queue": WORKFLOW_TASK_QUEUE,
            "workflows": [TextToMeshWorkflow, TokensToMeshWorkflow],
            "activities": [],
            "description": "workflow orchestration (no GPU needed)",
        },
        "clip": {
            "task_queue": CLIP_TASK_QUEUE,
            "workflows": [],
            "activities": [encode_text],
            "description": "CLIP text encoding (~20ms, 1 worker = 50 req/s)",
        },
        "gpt": {
            "task_queue": GPT_TASK_QUEUE,
            "workflows": [],
            "activities": [generate_tokens],
            "description": "GPT token generation (~10s, BOTTLENECK)",
        },
        "mesh": {
            "task_queue": MESH_TASK_QUEUE,
            "workflows": [],
            "activities": [decode_mesh],
            "description": "Mesh decoding (~3s)",
        },
    }

    if stage not in stage_configs:
        raise ValueError(
            f"Unknown stage: {stage}. Valid stages: {list(stage_configs.keys())}"
        )

    config = stage_configs[stage]
    logger.info(
        f"PER-STAGE mode: {stage.upper()} - {config['description']}\n"
        f"  Task queue: {config['task_queue']}"
    )

    worker = Worker(
        client,
        task_queue=config["task_queue"],
        workflows=config["workflows"],
        activities=config["activities"],
    )

    logger.info("Worker started. Waiting for tasks...")
    await worker.run()


async def main():
    """Start the Temporal worker."""
    logger.info(f"Connecting to Temporal at {TEMPORAL_HOST}...")

    client = await Client.connect(
        TEMPORAL_HOST,
        namespace=NAMESPACE,
    )
    logger.info("Connected to Temporal")

    if WORKER_MODE == "unified":
        await run_unified_worker(client)
    elif WORKER_MODE == "per-stage":
        if not WORKER_STAGE:
            raise ValueError(
                "WORKER_STAGE must be set when WORKER_MODE=per-stage. "
                "Valid values: workflow, clip, gpt, mesh"
            )
        await run_stage_worker(client, WORKER_STAGE)
    else:
        raise ValueError(f"Unknown WORKER_MODE: {WORKER_MODE}")


if __name__ == "__main__":
    asyncio.run(main())
