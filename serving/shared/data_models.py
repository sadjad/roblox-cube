"""
Shared data models for the Cube3D serving pipeline.

These dataclasses define the interface contracts between pipeline stages.
Modify these if you want to change how stages communicate.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import json
import base64


@dataclass
class TextEncodingInput:
    """Input to the CLIP text encoding stage."""
    prompt: str
    guidance_scale: float = 3.0
    bounding_box: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "bounding_box": list(self.bounding_box) if self.bounding_box else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TextEncodingInput":
        bbox = tuple(d["bounding_box"]) if d.get("bounding_box") else None
        return cls(
            prompt=d["prompt"],
            guidance_scale=d.get("guidance_scale", 3.0),
            bounding_box=bbox,
        )


@dataclass
class TextEncodingOutput:
    """Output from CLIP encoding stage / Input to GPT generation stage."""
    # Condition embeddings: [seq_len, embed_dim] - typically [77-78, 1536]
    condition_embeddings: np.ndarray
    # Unconditional embeddings for classifier-free guidance (optional)
    uncond_embeddings: Optional[np.ndarray] = None
    guidance_scale: float = 3.0

    def to_bytes(self) -> bytes:
        """Serialize for transmission between services."""
        return json.dumps({
            "condition_embeddings": _ndarray_to_b64(self.condition_embeddings),
            "uncond_embeddings": _ndarray_to_b64(self.uncond_embeddings) if self.uncond_embeddings is not None else None,
            "guidance_scale": self.guidance_scale,
        }).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "TextEncodingOutput":
        d = json.loads(data.decode())
        return cls(
            condition_embeddings=_b64_to_ndarray(d["condition_embeddings"]),
            uncond_embeddings=_b64_to_ndarray(d["uncond_embeddings"]) if d.get("uncond_embeddings") else None,
            guidance_scale=d.get("guidance_scale", 3.0),
        )


@dataclass
class TokenGenerationInput:
    """Input to GPT token generation stage."""
    condition_embeddings: np.ndarray
    uncond_embeddings: Optional[np.ndarray] = None
    guidance_scale: float = 3.0
    top_p: Optional[float] = None  # None = deterministic (argmax)
    max_tokens: int = 1024

    def to_bytes(self) -> bytes:
        return json.dumps({
            "condition_embeddings": _ndarray_to_b64(self.condition_embeddings),
            "uncond_embeddings": _ndarray_to_b64(self.uncond_embeddings) if self.uncond_embeddings is not None else None,
            "guidance_scale": self.guidance_scale,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "TokenGenerationInput":
        d = json.loads(data.decode())
        return cls(
            condition_embeddings=_b64_to_ndarray(d["condition_embeddings"]),
            uncond_embeddings=_b64_to_ndarray(d["uncond_embeddings"]) if d.get("uncond_embeddings") else None,
            guidance_scale=d.get("guidance_scale", 3.0),
            top_p=d.get("top_p"),
            max_tokens=d.get("max_tokens", 1024),
        )


@dataclass
class TokenGenerationOutput:
    """Output from GPT stage / Input to mesh decoding stage."""
    # Token IDs: [num_tokens] - typically [1024] integers in range [0, 16383]
    token_ids: np.ndarray  # dtype: int32 or int64

    def to_bytes(self) -> bytes:
        """Compact serialization - just the token IDs as int32."""
        return self.token_ids.astype(np.int32).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "TokenGenerationOutput":
        token_ids = np.frombuffer(data, dtype=np.int32)
        return cls(token_ids=token_ids)

    def to_list(self) -> List[int]:
        return self.token_ids.tolist()

    @classmethod
    def from_list(cls, tokens: List[int]) -> "TokenGenerationOutput":
        return cls(token_ids=np.array(tokens, dtype=np.int32))


@dataclass
class MeshDecodingInput:
    """Input to mesh decoding stage."""
    token_ids: np.ndarray
    resolution_base: float = 8.0  # 2^8 = 256 grid resolution

    def to_bytes(self) -> bytes:
        # Pack resolution as 4-byte float, then token IDs
        resolution_bytes = np.array([self.resolution_base], dtype=np.float32).tobytes()
        token_bytes = self.token_ids.astype(np.int32).tobytes()
        return resolution_bytes + token_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "MeshDecodingInput":
        resolution = np.frombuffer(data[:4], dtype=np.float32)[0]
        token_ids = np.frombuffer(data[4:], dtype=np.int32)
        return cls(token_ids=token_ids, resolution_base=float(resolution))


@dataclass
class MeshOutput:
    """Final mesh output from the pipeline."""
    vertices: np.ndarray  # [N, 3] float32
    faces: np.ndarray     # [M, 3] int32

    def to_bytes(self) -> bytes:
        """Serialize mesh data."""
        return json.dumps({
            "vertices": _ndarray_to_b64(self.vertices),
            "faces": _ndarray_to_b64(self.faces),
        }).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "MeshOutput":
        d = json.loads(data.decode())
        return cls(
            vertices=_b64_to_ndarray(d["vertices"]),
            faces=_b64_to_ndarray(d["faces"]),
        )

    def to_obj_string(self) -> str:
        """Convert to OBJ format string."""
        lines = []
        for v in self.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for f in self.faces:
            # OBJ uses 1-indexed faces
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        return "\n".join(lines)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


# --- Serialization helpers ---

def _ndarray_to_b64(arr: np.ndarray) -> dict:
    """Serialize numpy array to base64 with metadata."""
    return {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _b64_to_ndarray(d: dict) -> np.ndarray:
    """Deserialize numpy array from base64."""
    data = base64.b64decode(d["data"])
    arr = np.frombuffer(data, dtype=np.dtype(d["dtype"]))
    return arr.reshape(d["shape"])


# --- Pipeline configuration ---

@dataclass
class PipelineConfig:
    """Configuration for the serving pipeline.

    Modify this to change how stages are split or combined.
    """
    # Triton server endpoints
    clip_triton_url: str = "triton-clip:8001"
    gpt_triton_url: str = "triton-gpt:8001"
    mesh_triton_url: str = "triton-mesh:8001"

    # Model names in Triton
    clip_model_name: str = "clip_encoder"
    gpt_model_name: str = "gpt_generator"
    mesh_model_name: str = "mesh_decoder"

    # Timeouts (in seconds)
    clip_timeout: float = 30.0
    gpt_timeout: float = 300.0  # GPT can take a while
    mesh_timeout: float = 120.0

    # Default generation parameters
    default_guidance_scale: float = 3.0
    default_resolution_base: float = 8.0
    default_top_p: Optional[float] = None

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables."""
        import os
        return cls(
            clip_triton_url=os.getenv("CLIP_TRITON_URL", "triton-clip:8001"),
            gpt_triton_url=os.getenv("GPT_TRITON_URL", "triton-gpt:8001"),
            mesh_triton_url=os.getenv("MESH_TRITON_URL", "triton-mesh:8001"),
            clip_model_name=os.getenv("CLIP_MODEL_NAME", "clip_encoder"),
            gpt_model_name=os.getenv("GPT_MODEL_NAME", "gpt_generator"),
            mesh_model_name=os.getenv("MESH_MODEL_NAME", "mesh_decoder"),
            clip_timeout=float(os.getenv("CLIP_TIMEOUT", "30")),
            gpt_timeout=float(os.getenv("GPT_TIMEOUT", "300")),
            mesh_timeout=float(os.getenv("MESH_TIMEOUT", "120")),
            default_guidance_scale=float(os.getenv("DEFAULT_GUIDANCE_SCALE", "3.0")),
            default_resolution_base=float(os.getenv("DEFAULT_RESOLUTION_BASE", "8.0")),
        )
