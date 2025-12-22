"""
Triton Inference Server client utilities.

Provides a clean interface for calling Triton models from Temporal activities.
"""

import numpy as np
from typing import Dict, List, Optional, Union
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class TritonClient:
    """Wrapper around Triton gRPC client with convenience methods."""

    def __init__(self, url: str, verbose: bool = False):
        """
        Initialize Triton client.

        Args:
            url: Triton server URL (e.g., "localhost:8001")
            verbose: Enable verbose logging
        """
        self.url = url
        self.verbose = verbose
        self._client: Optional[grpcclient.InferenceServerClient] = None

    def _get_client(self) -> grpcclient.InferenceServerClient:
        """Get or create the gRPC client."""
        if self._client is None:
            self._client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
            )
        return self._client

    def is_server_ready(self) -> bool:
        """Check if Triton server is ready."""
        try:
            return self._get_client().is_server_ready()
        except Exception:
            return False

    def is_model_ready(self, model_name: str) -> bool:
        """Check if a specific model is ready."""
        try:
            return self._get_client().is_model_ready(model_name)
        except Exception:
            return False

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: List[str],
        timeout: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perform inference on a Triton model.

        Args:
            model_name: Name of the model to call
            inputs: Dictionary mapping input names to numpy arrays
            outputs: List of output names to request
            timeout: Request timeout in seconds

        Returns:
            Dictionary mapping output names to numpy arrays

        Raises:
            TritonInferenceError: If inference fails
        """
        client = self._get_client()

        # Prepare inputs
        triton_inputs = []
        for name, data in inputs.items():
            inp = grpcclient.InferInput(
                name,
                list(data.shape),
                _numpy_to_triton_dtype(data.dtype),
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        # Prepare outputs
        triton_outputs = [
            grpcclient.InferRequestedOutput(name) for name in outputs
        ]

        # Perform inference
        try:
            result = client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=triton_outputs,
                client_timeout=timeout,
            )
        except InferenceServerException as e:
            raise TritonInferenceError(f"Inference failed for {model_name}: {e}") from e

        # Extract outputs
        output_dict = {}
        for name in outputs:
            output_dict[name] = result.as_numpy(name)

        return output_dict

    def close(self):
        """Close the client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None


class TritonInferenceError(Exception):
    """Raised when Triton inference fails."""
    pass


def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
    """Convert numpy dtype to Triton dtype string."""
    dtype_map = {
        np.float32: "FP32",
        np.float64: "FP64",
        np.float16: "FP16",
        np.int32: "INT32",
        np.int64: "INT64",
        np.int16: "INT16",
        np.int8: "INT8",
        np.uint8: "UINT8",
        np.uint16: "UINT16",
        np.uint32: "UINT32",
        np.uint64: "UINT64",
        np.bool_: "BOOL",
        np.object_: "BYTES",
    }

    # Handle numpy dtype objects
    if hasattr(dtype, 'type'):
        dtype = dtype.type

    for np_type, triton_type in dtype_map.items():
        if np.issubdtype(dtype, np_type):
            return triton_type

    raise ValueError(f"Unsupported dtype: {dtype}")


def create_client_pool(
    urls: Dict[str, str],
    verbose: bool = False,
) -> Dict[str, TritonClient]:
    """
    Create a pool of Triton clients for multiple servers.

    Args:
        urls: Dictionary mapping service names to URLs
        verbose: Enable verbose logging

    Returns:
        Dictionary mapping service names to TritonClient instances

    Example:
        clients = create_client_pool({
            "clip": "triton-clip:8001",
            "gpt": "triton-gpt:8001",
            "mesh": "triton-mesh:8001",
        })
    """
    return {name: TritonClient(url, verbose) for name, url in urls.items()}
