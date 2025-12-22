"""
Triton Python Backend for CLIP Text Encoder.

This model takes text prompts and produces embeddings for the GPT stage.
It handles both conditional and unconditional embeddings for classifier-free guidance.
"""

import json
import numpy as np
import sys
from pathlib import Path

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """CLIP text encoder for Cube3D pipeline."""

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

        self.clip_model_name = params.get("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")
        self.cube3d_path = params.get("CUBE3D_PATH", "/models/cube3d")
        self.gpt_config_path = params.get("GPT_CONFIG_PATH")
        self.gpt_checkpoint_path = params.get("GPT_CHECKPOINT_PATH")

        # Add cube3d to path
        sys.path.insert(0, self.cube3d_path)

        # Import dependencies
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model
        pb_utils.Logger.log_info(f"Loading CLIP model: {self.clip_model_name}")
        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            self.clip_model_name,
            device_map=self.device,
        ).eval()

        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.clip_model_name)

        # Load GPT model config to get projection dimensions
        # We need the text_proj layer from GPT to project CLIP outputs
        from cube3d.inference.utils import load_config, parse_structured
        from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer

        cfg = load_config(self.gpt_config_path)
        gpt_cfg = parse_structured(DualStreamRoformer.Config, cfg.gpt_model)

        # Create the projection layer (768 -> 1536)
        self.text_proj = torch.nn.Linear(
            gpt_cfg.text_model_embed_dim,  # 768
            gpt_cfg.n_embd,  # 1536
            bias=gpt_cfg.bias,
        ).to(self.device)

        # Load projection weights from GPT checkpoint
        from cube3d.inference.utils import load_model_weights
        import safetensors.torch

        pb_utils.Logger.log_info(f"Loading text_proj weights from: {self.gpt_checkpoint_path}")
        state_dict = safetensors.torch.load_file(self.gpt_checkpoint_path)
        text_proj_state = {
            k.replace("text_proj.", ""): v
            for k, v in state_dict.items()
            if k.startswith("text_proj.")
        }
        self.text_proj.load_state_dict(text_proj_state)
        self.text_proj.eval()

        # Check if model uses bbox
        self.use_bbox = gpt_cfg.use_bbox
        if self.use_bbox:
            self.bbox_proj = torch.nn.Linear(3, gpt_cfg.n_embd).to(self.device)
            bbox_proj_state = {
                k.replace("bbox_proj.", ""): v
                for k, v in state_dict.items()
                if k.startswith("bbox_proj.")
            }
            if bbox_proj_state:
                self.bbox_proj.load_state_dict(bbox_proj_state)
            self.bbox_proj.eval()

        pb_utils.Logger.log_info("CLIP encoder initialized successfully")

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
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def _process_request(self, request):
        """Process a single inference request."""
        # Get input tensors
        prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
        prompt = prompt_tensor.as_numpy()[0][0].decode("utf-8")

        # Get optional inputs
        guidance_scale = 3.0
        guidance_tensor = pb_utils.get_input_tensor_by_name(request, "guidance_scale")
        if guidance_tensor is not None:
            guidance_scale = float(guidance_tensor.as_numpy()[0][0])

        bounding_box = None
        bbox_tensor = pb_utils.get_input_tensor_by_name(request, "bounding_box")
        if bbox_tensor is not None:
            bounding_box = bbox_tensor.as_numpy()[0]

        # Encode text
        with self.torch.inference_mode():
            # Tokenize
            text_inputs = self.tokenizer(
                [prompt],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            # Run CLIP encoder (in full precision)
            with self.torch.autocast(device_type=self.device.type, enabled=False):
                encoded = self.text_model(**text_inputs)
                clip_embeds = encoded.last_hidden_state  # [1, 77, 768]

            # Project to GPT embedding dimension
            with self.torch.autocast(device_type=self.device.type, dtype=self.torch.bfloat16):
                cond_embeds = self.text_proj(clip_embeds)  # [1, 77, 1536]

                # Add bounding box if provided and supported
                if self.use_bbox and bounding_box is not None:
                    bbox_tensor = self.torch.tensor(
                        bounding_box, dtype=cond_embeds.dtype, device=self.device
                    ).unsqueeze(0)
                    bbox_embed = self.bbox_proj(bbox_tensor).unsqueeze(1)  # [1, 1, 1536]
                    cond_embeds = self.torch.cat([cond_embeds, bbox_embed], dim=1)  # [1, 78, 1536]

            # Generate unconditional embeddings for classifier-free guidance
            if guidance_scale > 0.0:
                empty_inputs = self.tokenizer(
                    [""],
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                empty_inputs = {k: v.to(self.device) for k, v in empty_inputs.items()}

                with self.torch.autocast(device_type=self.device.type, enabled=False):
                    uncond_encoded = self.text_model(**empty_inputs)
                    uncond_clip = uncond_encoded.last_hidden_state

                with self.torch.autocast(device_type=self.device.type, dtype=self.torch.bfloat16):
                    uncond_embeds = self.text_proj(uncond_clip)

                    if self.use_bbox:
                        zero_bbox = self.torch.zeros(1, 3, dtype=uncond_embeds.dtype, device=self.device)
                        zero_bbox_embed = self.bbox_proj(zero_bbox).unsqueeze(1)
                        uncond_embeds = self.torch.cat([uncond_embeds, zero_bbox_embed], dim=1)
            else:
                # No CFG, just return zeros for uncond
                uncond_embeds = self.torch.zeros_like(cond_embeds)

            # Convert to FP16 for output
            cond_out = cond_embeds.squeeze(0).half().cpu().numpy()  # [seq_len, 1536]
            uncond_out = uncond_embeds.squeeze(0).half().cpu().numpy()
            guidance_out = np.array([guidance_scale], dtype=np.float32)

        # Create output tensors
        cond_tensor = pb_utils.Tensor("condition_embeddings", cond_out)
        uncond_tensor = pb_utils.Tensor("uncond_embeddings", uncond_out)
        guidance_tensor = pb_utils.Tensor("guidance_scale_out", guidance_out)

        return pb_utils.InferenceResponse(
            output_tensors=[cond_tensor, uncond_tensor, guidance_tensor]
        )

    def finalize(self):
        """Clean up resources."""
        pb_utils.Logger.log_info("CLIP encoder finalized")
