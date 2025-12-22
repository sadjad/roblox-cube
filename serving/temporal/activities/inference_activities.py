"""
Temporal Activities for Cube3D Inference Pipeline.

Each activity calls a Triton Inference Server model and transforms
the results for the next stage.
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from temporalio import activity

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


# Configuration from environment
CLIP_TRITON_URL = os.getenv("CLIP_TRITON_URL", "triton-clip:8001")
GPT_TRITON_URL = os.getenv("GPT_TRITON_URL", "triton-gpt:8001")
MESH_TRITON_URL = os.getenv("MESH_TRITON_URL", "triton-mesh:8001")

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip_encoder")
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt_generator")
MESH_MODEL_NAME = os.getenv("MESH_MODEL_NAME", "mesh_decoder")


def _create_triton_client(url: str) -> grpcclient.InferenceServerClient:
    """Create a Triton gRPC client."""
    return grpcclient.InferenceServerClient(url=url, verbose=False)


def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
    """Convert numpy dtype to Triton dtype string."""
    dtype_map = {
        np.float32: "FP32",
        np.float16: "FP16",
        np.int32: "INT32",
        np.int64: "INT64",
        np.uint8: "UINT8",
        np.bool_: "BOOL",
    }
    for np_type, triton_type in dtype_map.items():
        if np.issubdtype(dtype, np_type):
            return triton_type
    raise ValueError(f"Unsupported dtype: {dtype}")


@activity.defn
async def encode_text(
    prompt: str,
    guidance_scale: float = 3.0,
    bounding_box: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Encode text prompt using CLIP encoder via Triton.

    Args:
        prompt: Text prompt to encode
        guidance_scale: Classifier-free guidance scale
        bounding_box: Optional bounding box dimensions (x, y, z)

    Returns:
        Dictionary with condition_embeddings, uncond_embeddings, guidance_scale
    """
    activity.logger.info(f"Encoding text: {prompt[:50]}...")

    client = _create_triton_client(CLIP_TRITON_URL)

    try:
        # Prepare inputs
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

        # Prepare outputs
        outputs = [
            grpcclient.InferRequestedOutput("condition_embeddings"),
            grpcclient.InferRequestedOutput("uncond_embeddings"),
            grpcclient.InferRequestedOutput("guidance_scale_out"),
        ]

        # Call Triton
        result = client.infer(
            model_name=CLIP_MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        # Extract results
        cond_embeddings = result.as_numpy("condition_embeddings")
        uncond_embeddings = result.as_numpy("uncond_embeddings")
        guidance_out = result.as_numpy("guidance_scale_out")[0]

        activity.logger.info(
            f"Text encoded: cond_shape={cond_embeddings.shape}, "
            f"uncond_shape={uncond_embeddings.shape}"
        )

        # Return as serializable dict (convert numpy to lists for Temporal)
        return {
            "condition_embeddings": cond_embeddings.tolist(),
            "uncond_embeddings": uncond_embeddings.tolist(),
            "guidance_scale": float(guidance_out),
        }

    except InferenceServerException as e:
        activity.logger.error(f"Triton inference failed: {e}")
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

    Args:
        condition_embeddings: CLIP embeddings for the prompt
        uncond_embeddings: CLIP embeddings for empty prompt (for CFG)
        guidance_scale: Classifier-free guidance scale
        top_p: Nucleus sampling threshold (None for argmax)
        max_tokens: Maximum number of tokens to generate

    Returns:
        Dictionary with token_ids list
    """
    activity.logger.info("Generating tokens with GPT...")

    client = _create_triton_client(GPT_TRITON_URL)

    try:
        # Convert lists back to numpy arrays
        cond_np = np.array(condition_embeddings, dtype=np.float16)
        uncond_np = np.array(uncond_embeddings, dtype=np.float16)

        # Prepare inputs
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

        # Prepare outputs
        outputs = [grpcclient.InferRequestedOutput("token_ids")]

        # Call Triton (this is the slow part - GPT generation)
        # Send heartbeats during long-running inference
        activity.heartbeat("Starting GPT inference...")

        result = client.infer(
            model_name=GPT_MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        # Extract results
        token_ids = result.as_numpy("token_ids")

        activity.logger.info(f"Generated {len(token_ids)} tokens")

        return {
            "token_ids": token_ids.tolist(),
        }

    except InferenceServerException as e:
        activity.logger.error(f"Triton inference failed: {e}")
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

    Args:
        token_ids: List of 1024 token IDs
        resolution_base: Grid resolution as power of 2

    Returns:
        Dictionary with vertices, faces, num_vertices, num_faces
    """
    activity.logger.info(f"Decoding mesh at resolution 2^{resolution_base}...")

    client = _create_triton_client(MESH_TRITON_URL)

    try:
        # Convert to numpy
        tokens_np = np.array(token_ids, dtype=np.int32)

        # Prepare inputs
        inputs = []

        tokens_input = grpcclient.InferInput("token_ids", list(tokens_np.shape), "INT32")
        tokens_input.set_data_from_numpy(tokens_np)
        inputs.append(tokens_input)

        resolution_data = np.array([resolution_base], dtype=np.float32)
        resolution_input = grpcclient.InferInput("resolution_base", [1], "FP32")
        resolution_input.set_data_from_numpy(resolution_data)
        inputs.append(resolution_input)

        # Prepare outputs
        outputs = [
            grpcclient.InferRequestedOutput("vertices"),
            grpcclient.InferRequestedOutput("faces"),
            grpcclient.InferRequestedOutput("num_vertices"),
            grpcclient.InferRequestedOutput("num_faces"),
        ]

        # Call Triton
        result = client.infer(
            model_name=MESH_MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        # Extract results
        vertices = result.as_numpy("vertices")
        faces = result.as_numpy("faces")
        num_vertices = int(result.as_numpy("num_vertices")[0])
        num_faces = int(result.as_numpy("num_faces")[0])

        activity.logger.info(f"Mesh decoded: {num_vertices} vertices, {num_faces} faces")

        return {
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "num_vertices": num_vertices,
            "num_faces": num_faces,
        }

    except InferenceServerException as e:
        activity.logger.error(f"Triton inference failed: {e}")
        raise
    finally:
        client.close()
