"""
Temporal Activities for Cube3D Inference Pipeline.

Each activity calls a Triton Inference Server model and transforms
the results for the next stage.

SCALING STRATEGY:
  Each activity is registered on a separate task queue, allowing
  independent scaling of each pipeline stage by running different
  numbers of workers per queue.

  Example:
    - 1 worker on "cube3d-clip" queue (fast stage)
    - 10 workers on "cube3d-gpt" queue (slow stage - BOTTLENECK)
    - 3 workers on "cube3d-mesh" queue (medium stage)
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from temporalio import activity

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


# =============================================================================
# CONFIGURATION
# =============================================================================

# Triton server URLs - can point to single server or separate servers
CLIP_TRITON_URL = os.getenv("CLIP_TRITON_URL", "triton:8001")
GPT_TRITON_URL = os.getenv("GPT_TRITON_URL", "triton:8001")
MESH_TRITON_URL = os.getenv("MESH_TRITON_URL", "triton:8001")

# Model names in Triton
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip_encoder")
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt_generator")
MESH_MODEL_NAME = os.getenv("MESH_MODEL_NAME", "mesh_decoder")

# Task queue names - workers register to specific queues based on their role
CLIP_TASK_QUEUE = os.getenv("CLIP_TASK_QUEUE", "cube3d-clip")
GPT_TASK_QUEUE = os.getenv("GPT_TASK_QUEUE", "cube3d-gpt")
MESH_TASK_QUEUE = os.getenv("MESH_TASK_QUEUE", "cube3d-mesh")


# =============================================================================
# TRITON CLIENT HELPERS
# =============================================================================

def _create_triton_client(url: str) -> grpcclient.InferenceServerClient:
    """Create a Triton gRPC client."""
    return grpcclient.InferenceServerClient(url=url, verbose=False)


# =============================================================================
# ACTIVITIES - Each on its own task queue for independent scaling
# =============================================================================

@activity.defn
async def encode_text(
    prompt: str,
    guidance_scale: float = 3.0,
    bounding_box: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Encode text prompt using CLIP encoder via Triton.

    Task Queue: cube3d-clip (configurable via CLIP_TASK_QUEUE env var)
    Latency: ~20ms
    Scaling: 1 worker handles ~50 req/s (usually 1 worker is enough)

    Args:
        prompt: Text prompt to encode
        guidance_scale: Classifier-free guidance scale
        bounding_box: Optional bounding box dimensions (x, y, z)

    Returns:
        Dictionary with condition_embeddings, uncond_embeddings, guidance_scale
    """
    activity.logger.info(f"[CLIP] Encoding: {prompt[:50]}...")

    client = _create_triton_client(CLIP_TRITON_URL)

    try:
        inputs = []

        # Prompt input (string)
        prompt_data = np.array([[prompt]], dtype=object)
        prompt_input = grpcclient.InferInput("prompt", [1, 1], "BYTES")
        prompt_input.set_data_from_numpy(prompt_data)
        inputs.append(prompt_input)

        # Guidance scale input
        guidance_data = np.array([[guidance_scale]], dtype=np.float32)
        guidance_input = grpcclient.InferInput("guidance_scale", [1, 1], "FP32")
        guidance_input.set_data_from_numpy(guidance_data)
        inputs.append(guidance_input)

        # Bounding box input (optional)
        if bounding_box is not None:
            bbox_data = np.array([list(bounding_box)], dtype=np.float32)
            bbox_input = grpcclient.InferInput("bounding_box", [1, 3], "FP32")
            bbox_input.set_data_from_numpy(bbox_data)
            inputs.append(bbox_input)

        outputs = [
            grpcclient.InferRequestedOutput("condition_embeddings"),
            grpcclient.InferRequestedOutput("uncond_embeddings"),
            grpcclient.InferRequestedOutput("guidance_scale_out"),
        ]

        result = client.infer(
            model_name=CLIP_MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        cond_embeddings = result.as_numpy("condition_embeddings")
        uncond_embeddings = result.as_numpy("uncond_embeddings")
        guidance_out = result.as_numpy("guidance_scale_out")[0]

        activity.logger.info(f"[CLIP] Done: shape={cond_embeddings.shape}")

        return {
            "condition_embeddings": cond_embeddings.tolist(),
            "uncond_embeddings": uncond_embeddings.tolist(),
            "guidance_scale": float(guidance_out),
        }

    except InferenceServerException as e:
        activity.logger.error(f"[CLIP] Triton error: {e}")
        raise
    finally:
        client.close()


@activity.defn
async def generate_tokens(
    condition_embeddings: List[List[float]],
    uncond_embeddings: List[List[float]],
    guidance_scale: float = 3.0,
    top_p: Optional[float] = None,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Generate shape tokens using GPT model via Triton.

    Task Queue: cube3d-gpt (configurable via GPT_TASK_QUEUE env var)
    Latency: ~10 seconds
    Scaling: THIS IS THE BOTTLENECK - need 10 workers per 1 req/s throughput

    Args:
        condition_embeddings: CLIP embeddings for the prompt
        uncond_embeddings: CLIP embeddings for empty prompt (for CFG)
        guidance_scale: Classifier-free guidance scale
        top_p: Nucleus sampling threshold (None for argmax)
        max_tokens: Maximum number of tokens to generate

    Returns:
        Dictionary with token_ids list
    """
    activity.logger.info("[GPT] Starting token generation...")

    client = _create_triton_client(GPT_TRITON_URL)

    try:
        cond_np = np.array(condition_embeddings, dtype=np.float16)
        uncond_np = np.array(uncond_embeddings, dtype=np.float16)

        inputs = []

        cond_input = grpcclient.InferInput(
            "condition_embeddings", list(cond_np.shape), "FP16"
        )
        cond_input.set_data_from_numpy(cond_np)
        inputs.append(cond_input)

        uncond_input = grpcclient.InferInput(
            "uncond_embeddings", list(uncond_np.shape), "FP16"
        )
        uncond_input.set_data_from_numpy(uncond_np)
        inputs.append(uncond_input)

        guidance_data = np.array([guidance_scale], dtype=np.float32)
        guidance_input = grpcclient.InferInput("guidance_scale", [1], "FP32")
        guidance_input.set_data_from_numpy(guidance_data)
        inputs.append(guidance_input)

        if top_p is not None:
            top_p_data = np.array([top_p], dtype=np.float32)
            top_p_input = grpcclient.InferInput("top_p", [1], "FP32")
            top_p_input.set_data_from_numpy(top_p_data)
            inputs.append(top_p_input)

        max_tokens_data = np.array([max_tokens], dtype=np.int32)
        max_tokens_input = grpcclient.InferInput("max_tokens", [1], "INT32")
        max_tokens_input.set_data_from_numpy(max_tokens_data)
        inputs.append(max_tokens_input)

        outputs = [grpcclient.InferRequestedOutput("token_ids")]

        # Heartbeat for long-running inference
        activity.heartbeat("Starting GPT inference...")

        result = client.infer(
            model_name=GPT_MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        token_ids = result.as_numpy("token_ids")

        activity.logger.info(f"[GPT] Done: {len(token_ids)} tokens")

        return {"token_ids": token_ids.tolist()}

    except InferenceServerException as e:
        activity.logger.error(f"[GPT] Triton error: {e}")
        raise
    finally:
        client.close()


@activity.defn
async def decode_mesh(
    token_ids: List[int],
    resolution_base: float = 8.0,
) -> Dict[str, Any]:
    """
    Decode tokens to mesh using shape decoder via Triton.

    Task Queue: cube3d-mesh (configurable via MESH_TASK_QUEUE env var)
    Latency: ~3 seconds
    Scaling: 3 workers per 1 req/s throughput

    Args:
        token_ids: List of 1024 token IDs
        resolution_base: Grid resolution as power of 2

    Returns:
        Dictionary with vertices, faces, num_vertices, num_faces
    """
    activity.logger.info(f"[MESH] Decoding at resolution 2^{resolution_base}...")

    client = _create_triton_client(MESH_TRITON_URL)

    try:
        tokens_np = np.array(token_ids, dtype=np.int32)

        inputs = []

        tokens_input = grpcclient.InferInput("token_ids", list(tokens_np.shape), "INT32")
        tokens_input.set_data_from_numpy(tokens_np)
        inputs.append(tokens_input)

        resolution_data = np.array([resolution_base], dtype=np.float32)
        resolution_input = grpcclient.InferInput("resolution_base", [1], "FP32")
        resolution_input.set_data_from_numpy(resolution_data)
        inputs.append(resolution_input)

        outputs = [
            grpcclient.InferRequestedOutput("vertices"),
            grpcclient.InferRequestedOutput("faces"),
            grpcclient.InferRequestedOutput("num_vertices"),
            grpcclient.InferRequestedOutput("num_faces"),
        ]

        # Heartbeat for mesh extraction
        activity.heartbeat("Extracting mesh...")

        result = client.infer(
            model_name=MESH_MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        vertices = result.as_numpy("vertices")
        faces = result.as_numpy("faces")
        num_vertices = int(result.as_numpy("num_vertices")[0])
        num_faces = int(result.as_numpy("num_faces")[0])

        activity.logger.info(f"[MESH] Done: {num_vertices} verts, {num_faces} faces")

        return {
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "num_vertices": num_vertices,
            "num_faces": num_faces,
        }

    except InferenceServerException as e:
        activity.logger.error(f"[MESH] Triton error: {e}")
        raise
    finally:
        client.close()
