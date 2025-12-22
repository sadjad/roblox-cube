"""
Triton Python Backend for Mesh Decoder.

This model takes token IDs and produces a 3D mesh by:
1. VQ codebook lookup
2. VAE decoding
3. Occupancy field querying on dense grid
4. Marching cubes isosurface extraction
"""

import json
import numpy as np
import sys
from pathlib import Path

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Mesh decoder for Cube3D pipeline."""

    def initialize(self, args):
        """
        Initialize the model.

        Args:
            args: Dictionary containing model configuration
        """
        self.model_config = json.loads(args["model_config"])

        # Get parameters from config
        params = {p["key"]: p["value"]["string_value"]
                  for p in self.model_config.get("parameters", [])}

        self.cube3d_path = params.get("CUBE3D_PATH", "/models/cube3d")
        self.shape_config_path = params.get("SHAPE_CONFIG_PATH")
        self.shape_checkpoint_path = params.get("SHAPE_CHECKPOINT_PATH")
        self.default_resolution_base = float(params.get("DEFAULT_RESOLUTION_BASE", "8.0"))
        self.use_warp = params.get("USE_WARP", "true").lower() == "true"
        self.chunk_size = int(params.get("CHUNK_SIZE", "100000"))

        # Add cube3d to path
        sys.path.insert(0, self.cube3d_path)

        # Import dependencies
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load shape model
        pb_utils.Logger.log_info("Loading shape model...")
        self._load_models()
        pb_utils.Logger.log_info("Mesh decoder initialized successfully")

    def _load_models(self):
        """Load the shape autoencoder model."""
        from cube3d.inference.utils import load_config, load_model_weights, parse_structured
        from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder

        # Load config
        cfg = load_config(self.shape_config_path)

        # Initialize shape model
        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, cfg.shape_model)
        )
        load_model_weights(self.shape_model, self.shape_checkpoint_path)
        self.shape_model = self.shape_model.eval().to(self.device)

        # Store model config values
        self.num_codes = self.shape_model.cfg.num_codes
        self.num_encoder_latents = self.shape_model.cfg.num_encoder_latents

        pb_utils.Logger.log_info(
            f"Shape model loaded: num_codes={self.num_codes}, "
            f"num_latents={self.num_encoder_latents}"
        )

    def execute(self, requests):
        """
        Execute inference for a batch of requests.

        Args:
            requests: List of pb_utils.InferenceRequest

        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []

        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                pb_utils.Logger.log_error(f"Error processing request: {e}")
                import traceback
                pb_utils.Logger.log_error(traceback.format_exc())
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def _process_request(self, request):
        """Process a single inference request."""
        # Get input tensors
        token_tensor = pb_utils.get_input_tensor_by_name(request, "token_ids")
        token_ids = self.torch.from_numpy(token_tensor.as_numpy()).to(self.device)

        # Get optional resolution
        resolution_base = self.default_resolution_base
        resolution_tensor = pb_utils.get_input_tensor_by_name(request, "resolution_base")
        if resolution_tensor is not None:
            resolution_base = float(resolution_tensor.as_numpy()[0])

        # Decode mesh
        vertices, faces = self._decode_mesh(token_ids, resolution_base)

        # Create output tensors
        # Handle variable-size outputs
        vertices_np = vertices.astype(np.float32)
        faces_np = faces.astype(np.int32)

        vertices_tensor = pb_utils.Tensor("vertices", vertices_np)
        faces_tensor = pb_utils.Tensor("faces", faces_np)
        num_verts_tensor = pb_utils.Tensor(
            "num_vertices",
            np.array([len(vertices)], dtype=np.int32)
        )
        num_faces_tensor = pb_utils.Tensor(
            "num_faces",
            np.array([len(faces)], dtype=np.int32)
        )

        return pb_utils.InferenceResponse(
            output_tensors=[vertices_tensor, faces_tensor, num_verts_tensor, num_faces_tensor]
        )

    def _decode_mesh(self, token_ids, resolution_base):
        """
        Decode token IDs to mesh vertices and faces.

        Args:
            token_ids: Tensor of shape [num_tokens]
            resolution_base: Grid resolution as power of 2

        Returns:
            Tuple of (vertices, faces) numpy arrays
        """
        with self.torch.inference_mode():
            with self.torch.autocast(self.device.type, dtype=self.torch.bfloat16):
                # Ensure token IDs are in valid range
                shape_ids = token_ids[:self.num_encoder_latents].clamp(0, self.num_codes - 1)
                shape_ids = shape_ids.view(1, -1)  # [1, 1024]

                # VQ lookup + VAE decoding
                latents = self.shape_model.decode_indices(shape_ids)  # [1, 1024, 768]

                # Extract geometry (occupancy query + marching cubes)
                mesh_v_f, has_surface = self.shape_model.extract_geometry(
                    latents,
                    resolution_base=resolution_base,
                    chunk_size=self.chunk_size,
                    use_warp=self.use_warp,
                )

        # Get mesh data
        vertices, faces = mesh_v_f[0]

        if vertices is None or faces is None:
            # Return empty mesh if extraction failed
            pb_utils.Logger.log_warning("Mesh extraction failed, returning empty mesh")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

        return vertices, faces

    def finalize(self):
        """Clean up resources."""
        pb_utils.Logger.log_info("Mesh decoder finalized")
