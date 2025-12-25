"""
Text-to-Mesh Temporal Workflow.

This workflow orchestrates the three-stage Cube3D pipeline:
1. CLIP text encoding
2. GPT token generation
3. Mesh decoding

SCALING: Each stage runs on a separate task queue, allowing independent
scaling by deploying different numbers of workers per queue.
"""

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple, List

from temporalio import workflow

# Import activity stubs
with workflow.unsafe.imports_passed_through():
    from ..activities import (
        encode_text,
        generate_tokens,
        decode_mesh,
    )


# =============================================================================
# TASK QUEUE CONFIGURATION
# =============================================================================

# Each activity runs on its own task queue for independent scaling
# These can be overridden via environment variables in the workflow starter
CLIP_TASK_QUEUE = os.getenv("CLIP_TASK_QUEUE", "cube3d-clip")
GPT_TASK_QUEUE = os.getenv("GPT_TASK_QUEUE", "cube3d-gpt")
MESH_TASK_QUEUE = os.getenv("MESH_TASK_QUEUE", "cube3d-mesh")

# For simpler deployments, use a single queue for all activities
UNIFIED_TASK_QUEUE = os.getenv("UNIFIED_TASK_QUEUE", "cube3d-mesh-generation")
USE_UNIFIED_QUEUE = os.getenv("USE_UNIFIED_QUEUE", "false").lower() == "true"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TextToMeshInput:
    """Input parameters for the text-to-mesh workflow."""
    prompt: str
    guidance_scale: float = 3.0
    resolution_base: float = 8.0
    top_p: Optional[float] = None
    bounding_box: Optional[Tuple[float, float, float]] = None


@dataclass
class TextToMeshOutput:
    """Output from the text-to-mesh workflow."""
    vertices: List[List[float]]  # [[x, y, z], ...]
    faces: List[List[int]]  # [[v0, v1, v2], ...]
    num_vertices: int
    num_faces: int
    token_ids: Optional[List[int]] = None


# =============================================================================
# WORKFLOWS
# =============================================================================

@workflow.defn
class TextToMeshWorkflow:
    """
    Temporal workflow that orchestrates text-to-mesh generation.

    This workflow coordinates three Triton inference calls, each on
    a separate task queue for independent scaling:

    1. CLIP encoder (cube3d-clip queue): text -> embeddings (~20ms)
    2. GPT generator (cube3d-gpt queue): embeddings -> tokens (~10s) ← BOTTLENECK
    3. Mesh decoder (cube3d-mesh queue): tokens -> mesh (~3s)

    Scaling Example:
      - 1 worker on cube3d-clip
      - 10 workers on cube3d-gpt (bottleneck)
      - 3 workers on cube3d-mesh
    """

    @workflow.run
    async def run(self, input: TextToMeshInput) -> TextToMeshOutput:
        """
        Execute the text-to-mesh pipeline.

        Args:
            input: TextToMeshInput with prompt and generation parameters

        Returns:
            TextToMeshOutput with mesh vertices and faces
        """
        workflow.logger.info(f"Starting text-to-mesh: {input.prompt[:50]}...")

        # Determine task queues
        clip_queue = UNIFIED_TASK_QUEUE if USE_UNIFIED_QUEUE else CLIP_TASK_QUEUE
        gpt_queue = UNIFIED_TASK_QUEUE if USE_UNIFIED_QUEUE else GPT_TASK_QUEUE
        mesh_queue = UNIFIED_TASK_QUEUE if USE_UNIFIED_QUEUE else MESH_TASK_QUEUE

        # Stage 1: Encode text with CLIP
        workflow.logger.info(f"Stage 1: CLIP encoding (queue: {clip_queue})")
        encoding_result = await workflow.execute_activity(
            encode_text,
            args=[
                input.prompt,
                input.guidance_scale,
                input.bounding_box,
            ],
            task_queue=clip_queue,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=3,
            ),
        )

        # Stage 2: Generate tokens with GPT (BOTTLENECK)
        workflow.logger.info(f"Stage 2: GPT generation (queue: {gpt_queue})")
        token_result = await workflow.execute_activity(
            generate_tokens,
            args=[
                encoding_result["condition_embeddings"],
                encoding_result["uncond_embeddings"],
                input.guidance_scale,
                input.top_p,
            ],
            task_queue=gpt_queue,
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=2),
                maximum_interval=timedelta(seconds=60),
                maximum_attempts=2,  # GPT is expensive, limit retries
            ),
        )

        # Stage 3: Decode mesh
        workflow.logger.info(f"Stage 3: Mesh decoding (queue: {mesh_queue})")
        mesh_result = await workflow.execute_activity(
            decode_mesh,
            args=[
                token_result["token_ids"],
                input.resolution_base,
            ],
            task_queue=mesh_queue,
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=3,
            ),
        )

        workflow.logger.info(
            f"Done: {mesh_result['num_vertices']} verts, {mesh_result['num_faces']} faces"
        )

        return TextToMeshOutput(
            vertices=mesh_result["vertices"],
            faces=mesh_result["faces"],
            num_vertices=mesh_result["num_vertices"],
            num_faces=mesh_result["num_faces"],
            token_ids=token_result["token_ids"],
        )


@workflow.defn
class TokensToMeshWorkflow:
    """
    Workflow that takes pre-generated tokens and decodes them to mesh.

    Useful for:
    - Replaying cached token sequences
    - Debugging mesh generation
    - Regenerating meshes at different resolutions
    """

    @workflow.run
    async def run(
        self,
        token_ids: List[int],
        resolution_base: float = 8.0,
    ) -> TextToMeshOutput:
        """
        Decode tokens to mesh.

        Args:
            token_ids: Pre-generated token IDs (1024 integers)
            resolution_base: Grid resolution as power of 2

        Returns:
            TextToMeshOutput with mesh data
        """
        mesh_queue = UNIFIED_TASK_QUEUE if USE_UNIFIED_QUEUE else MESH_TASK_QUEUE

        workflow.logger.info(f"Decoding tokens to mesh (queue: {mesh_queue})")

        mesh_result = await workflow.execute_activity(
            decode_mesh,
            args=[token_ids, resolution_base],
            task_queue=mesh_queue,
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=3,
            ),
        )

        return TextToMeshOutput(
            vertices=mesh_result["vertices"],
            faces=mesh_result["faces"],
            num_vertices=mesh_result["num_vertices"],
            num_faces=mesh_result["num_faces"],
            token_ids=token_ids,
        )
