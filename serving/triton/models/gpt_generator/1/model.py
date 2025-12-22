"""
Triton Python Backend for GPT Token Generator.

This model performs autoregressive token generation using the DualStreamRoformer.
It takes text embeddings and generates 1024 discrete shape tokens.
"""

import json
import numpy as np
import sys
from pathlib import Path

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """GPT-based token generator for Cube3D pipeline."""

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
        self.gpt_config_path = params.get("GPT_CONFIG_PATH")
        self.gpt_checkpoint_path = params.get("GPT_CHECKPOINT_PATH")
        self.shape_checkpoint_path = params.get("SHAPE_CHECKPOINT_PATH")
        self.use_kv_cache = params.get("USE_KV_CACHE", "true").lower() == "true"
        self.default_max_tokens = int(params.get("DEFAULT_MAX_TOKENS", "1024"))

        # Add cube3d to path
        sys.path.insert(0, self.cube3d_path)

        # Import dependencies
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        pb_utils.Logger.log_info("Loading GPT model...")
        self._load_models()
        pb_utils.Logger.log_info("GPT generator initialized successfully")

    def _load_models(self):
        """Load the GPT model and copy VQ codebook."""
        from cube3d.inference.utils import load_config, load_model_weights, parse_structured
        from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer
        from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
        from cube3d.inference.logits_postprocesses import process_logits

        self.process_logits = process_logits

        # Load config
        cfg = load_config(self.gpt_config_path)

        # Initialize GPT model
        self.gpt_model = DualStreamRoformer(
            parse_structured(DualStreamRoformer.Config, cfg.gpt_model)
        )
        load_model_weights(self.gpt_model, self.gpt_checkpoint_path)
        self.gpt_model = self.gpt_model.eval().to(self.device)

        # Initialize shape model (just for codebook)
        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, cfg.shape_model)
        )
        load_model_weights(self.shape_model, self.shape_checkpoint_path)
        self.shape_model = self.shape_model.eval().to(self.device)

        # Copy VQ codebook to GPT token embeddings
        with self.torch.no_grad():
            codebook = self.shape_model.bottleneck.block.get_codebook()
            codebook = self.gpt_model.shape_proj(codebook).detach()
        self.gpt_model.transformer.wte.weight.data[:codebook.shape[0]] = codebook

        # Store model config
        self.max_new_tokens = self.shape_model.cfg.num_encoder_latents  # 1024
        self.min_id = 0
        self.max_id = self.shape_model.cfg.num_codes  # 16384

        pb_utils.Logger.log_info(
            f"GPT model loaded: max_tokens={self.max_new_tokens}, "
            f"vocab_size={self.max_id}, kv_cache={self.use_kv_cache}"
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
        cond_tensor = pb_utils.get_input_tensor_by_name(request, "condition_embeddings")
        uncond_tensor = pb_utils.get_input_tensor_by_name(request, "uncond_embeddings")
        guidance_tensor = pb_utils.get_input_tensor_by_name(request, "guidance_scale")

        cond_embeds = self.torch.from_numpy(cond_tensor.as_numpy()).to(self.device)
        uncond_embeds = self.torch.from_numpy(uncond_tensor.as_numpy()).to(self.device)
        guidance_scale = float(guidance_tensor.as_numpy()[0])

        # Get optional inputs
        max_tokens = self.default_max_tokens
        max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
        if max_tokens_tensor is not None:
            max_tokens = int(max_tokens_tensor.as_numpy()[0])
        max_tokens = min(max_tokens, self.max_new_tokens)

        top_p = None
        top_p_tensor = pb_utils.get_input_tensor_by_name(request, "top_p")
        if top_p_tensor is not None:
            top_p_val = float(top_p_tensor.as_numpy()[0])
            if top_p_val > 0:
                top_p = top_p_val

        # Generate tokens
        token_ids = self._generate_tokens(
            cond_embeds,
            uncond_embeds,
            guidance_scale,
            max_tokens,
            top_p,
        )

        # Create output tensor
        output_tensor = pb_utils.Tensor(
            "token_ids",
            token_ids.cpu().numpy().astype(np.int32)
        )

        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    def _generate_tokens(
        self,
        cond_embeds,
        uncond_embeds,
        guidance_scale,
        max_tokens,
        top_p,
    ):
        """
        Generate tokens autoregressively.

        This implements the core generation loop with KV-cache optimization.
        """
        with self.torch.inference_mode():
            # Prepare condition tensor
            # cond_embeds: [seq_len, 1536], uncond_embeds: [seq_len, 1536]
            cond = cond_embeds.unsqueeze(0)  # [1, seq_len, 1536]
            uncond = uncond_embeds.unsqueeze(0)

            # For classifier-free guidance, we concatenate cond and uncond
            if guidance_scale > 0:
                cond = self.torch.cat([cond, uncond], dim=0)  # [2, seq_len, 1536]
                batch_size = 2
            else:
                batch_size = 1

            cond_len = cond.shape[1]

            # Initialize BOS token embedding
            bos_embed = self.gpt_model.encode_token(
                self.torch.full(
                    (batch_size, 1),
                    fill_value=self.gpt_model.shape_bos_id,
                    dtype=self.torch.long,
                    device=self.device,
                )
            )  # [batch_size, 1, 1536]

            # Initialize embed buffer
            embed_dim = bos_embed.shape[-1]
            max_seq_len = 1 + max_tokens
            embed_buffer = self.torch.zeros(
                (batch_size, max_seq_len, embed_dim),
                dtype=bos_embed.dtype,
                device=self.device,
            )
            embed_buffer[:, :1, :].copy_(bos_embed)

            # Initialize KV-cache if enabled
            kv_cache = None
            if self.use_kv_cache:
                kv_cache = self.gpt_model.init_kv_cache(
                    batch_size,
                    cond_len,
                    max_tokens + 1,
                    self.torch.bfloat16,
                    self.device,
                )

            output_ids = []

            with self.torch.autocast(self.device.type, dtype=self.torch.bfloat16):
                for i in range(max_tokens):
                    curr_pos_id = self.torch.tensor([i], dtype=self.torch.long, device=self.device)

                    # Forward pass
                    logits = self.gpt_model(
                        embed_buffer,
                        cond,
                        kv_cache=kv_cache,
                        curr_pos_id=curr_pos_id if self.use_kv_cache else None,
                        decode=(i > 0) if self.use_kv_cache else False,
                    )

                    # Extract logits for current position
                    if self.use_kv_cache:
                        logits = logits[:, 0, ...]
                    else:
                        logits = logits[:, i, ...]

                    # Slice to valid token range
                    logits = logits[..., self.min_id:self.max_id]

                    # Apply classifier-free guidance
                    if guidance_scale > 0:
                        logits, uncond_logits = logits.float().chunk(2, dim=0)
                        gamma = guidance_scale * (max_tokens - i) / max_tokens
                        logits = (1 + gamma) * logits - gamma * uncond_logits

                    # Sample next token
                    next_id = self.process_logits(logits, top_p=top_p)
                    output_ids.append(next_id)

                    # Encode next token and add to buffer
                    next_embed = self.gpt_model.encode_token(next_id)
                    if guidance_scale > 0:
                        next_embed = self.torch.cat([next_embed, next_embed], dim=0)
                    embed_buffer[:, i + 1, :].copy_(next_embed.squeeze(1))

            # Concatenate all token IDs
            token_ids = self.torch.cat(output_ids, dim=1).squeeze(0)  # [max_tokens]

        return token_ids

    def finalize(self):
        """Clean up resources."""
        pb_utils.Logger.log_info("GPT generator finalized")
