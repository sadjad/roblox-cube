"""
Text-to-Mesh Temporal Workflow.

This workflow orchestrates the three-stage Cube3D pipeline:
1. CLIP text encoding
2. GPT token generation
3. Mesh decoding

Each stage is an activity that calls a Triton model.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple, List

from temporalio import workflow

# Import activity stubs - these will be defined in activities module
with workflow.unsafe.imports_passed_through():
    from ..activities import (
        encode_text,
        generate_tokens,
        decode_mesh,
    )


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
    # Include intermediate results for debugging/caching
    token_ids: Optional[List[int]] = None


@workflow.defn
class TextToMeshWorkflow:
    """
    Temporal workflow that orchestrates text-to-mesh generation.

    This workflow coordinates three Triton inference calls:
    1. CLIP encoder: text -> embeddings
    2. GPT generator: embeddings -> tokens
    3. Mesh decoder: tokens -> mesh

    The workflow is durable - if it fails partway through, it will
    resume from the last completed activity.
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
        workflow.logger.info(f"Starting text-to-mesh generation for: {input.prompt[:50]}...")

        # Stage 1: Encode text with CLIP
        workflow.logger.info("Stage 1: Encoding text with CLIP...")
        encoding_result = await workflow.execute_activity(
            encode_text,
            args=[
                input.prompt,
                input.guidance_scale,
                input.bounding_box,
            ],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=3,
            ),
        )

        # Stage 2: Generate tokens with GPT
        workflow.logger.info("Stage 2: Generating tokens with GPT...")
        token_result = await workflow.execute_activity(
            generate_tokens,
            args=[
                encoding_result["condition_embeddings"],
                encoding_result["uncond_embeddings"],
                input.guidance_scale,
                input.top_p,
            ],
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=2),
                maximum_interval=timedelta(seconds=60),
                maximum_attempts=2,  # GPT is expensive, limit retries
            ),
        )

        # Stage 3: Decode mesh
        workflow.logger.info("Stage 3: Decoding mesh...")
        mesh_result = await workflow.execute_activity(
            decode_mesh,
            args=[
                token_result["token_ids"],
                input.resolution_base,
            ],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=workflow.RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=3,
            ),
        )

        workflow.logger.info(
            f"Mesh generation complete: {mesh_result['num_vertices']} vertices, "
            f"{mesh_result['num_faces']} faces"
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
        workflow.logger.info("Decoding tokens to mesh...")

        mesh_result = await workflow.execute_activity(
            decode_mesh,
            args=[token_ids, resolution_base],
            start_to_close_timeout=timedelta(minutes=5),
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
