# Cube3D Model Serving Architecture Analysis

This document provides a detailed analysis of the Cube3D text-to-3D model from a serving perspective, identifying all pipeline stages, their characteristics, and options for decomposition.

---

## 1. Executive Summary

The Cube3D model transforms text prompts into 3D meshes through a **7-stage pipeline**:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  Text Prompt → CLIP → GPT Transformer → VQ Decode → VAE Decode → Occupancy → Marching   │
│                                                                   Query      Cubes      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

Each stage has distinct computational characteristics, making the model a candidate for both monolithic and decomposed serving strategies.

---

## 2. Complete Pipeline Stages

### Stage 1: Text Encoding (CLIP)

**Component:** `CLIPTextModelWithProjection` (HuggingFace Transformers)

| Attribute | Value |
|-----------|-------|
| **Model** | `openai/clip-vit-large-patch14` |
| **Input** | Text string(s) |
| **Output** | `[batch, 77, 768]` embeddings |
| **Parameters** | ~124M (ViT-L/14 text encoder) |
| **Compute Type** | Dense matrix operations, runs in FP32 |
| **Latency** | ~10-20ms on A100 |

**Key Code Path:**
```
engine.py:158-185 → run_clip()
  ├── text_tokenizer() → [batch, 77] token IDs
  ├── text_model() → [batch, 77, 768] hidden states
  └── gpt_model.encode_text() → [batch, 77, 1536] projected
```

**Serving Characteristics:**
- Stateless, embarrassingly parallel
- Can be batched efficiently
- Standard transformer inference - well-supported by TensorRT, ONNX

---

### Stage 2: Condition Preparation

**Component:** Projection layers + optional bounding box encoding

| Attribute | Value |
|-----------|-------|
| **Input** | CLIP embeddings `[batch, 77, 768]`, optional bbox `[batch, 3]` |
| **Output** | Condition tensor `[batch, 77-78, 1536]` |
| **Parameters** | ~1.2M (text_proj) + ~4.6K (bbox_proj) |
| **Compute Type** | Simple linear projections |
| **Latency** | <1ms |

**Key Code Path:**
```
engine.py:116-155 → prepare_inputs()
  ├── gpt_model.text_proj() → Linear(768→1536)
  └── gpt_model.bbox_proj() → Linear(3→1536) [if use_bbox]
```

**Serving Characteristics:**
- Trivial compute, usually combined with Stage 1 or Stage 3
- For classifier-free guidance: duplicates embeddings and adds unconditional path

---

### Stage 3: Autoregressive Token Generation (GPT)

**Component:** `DualStreamRoformer`

| Attribute | Value |
|-----------|-------|
| **Architecture** | 23 dual-stream + 1 single-stream transformer layers |
| **Hidden Dim** | 1536 |
| **Heads** | 12 |
| **Vocabulary** | 16,387 (16,384 VQ codes + 3 special tokens) |
| **Input** | BOS token + condition embeddings |
| **Output** | 1024 discrete tokens (shape codes) |
| **Parameters** | ~470M |
| **Compute Type** | Autoregressive, 1024 sequential forward passes |
| **Latency** | ~8-15 seconds on A100 (with KV-cache) |

**Key Code Path:**
```
engine.py:210-285 → run_gpt()
  └── for i in range(1024):  # autoregressive loop
        ├── gpt_model() → forward pass
        │     ├── dual_blocks[0:23] → DualStreamDecoderLayer
        │     │     └── DualStreamAttention with RoPE
        │     └── single_blocks[0:1] → DecoderLayer
        ├── lm_head → [batch, 16384] logits
        ├── classifier_free_guidance()
        ├── process_logits() → next token ID
        └── gpt_model.encode_token() → next embedding
```

**Serving Characteristics:**
- **MOST COMPUTE-INTENSIVE STAGE** (~80-90% of total inference time)
- Sequential dependency: each token depends on previous tokens
- KV-cache optimization reduces O(n²) to O(n) per step
- CUDA graphs (`EngineFast`) provide 2-3x speedup
- Batching limited: autoregressive nature means all samples in batch must sync
- Classifier-free guidance doubles effective batch size (cond + uncond)

---

### Stage 4: VQ Codebook Lookup

**Component:** `SphericalVectorQuantizer.lookup_codebook()`

| Attribute | Value |
|-----------|-------|
| **Codebook Size** | 16,384 codes × 32 dimensions |
| **Input** | 1024 discrete token IDs `[batch, 1024]` |
| **Output** | Quantized embeddings `[batch, 1024, 768]` |
| **Parameters** | ~525K (codebook) + ~50K (projections) |
| **Compute Type** | Embedding lookup + linear projection |
| **Latency** | <5ms |

**Key Code Path:**
```
one_d_autoencoder.py:447-458 → decode_indices()
  ├── bottleneck.block.lookup_codebook()
  │     ├── F.embedding(ids, normalized_codebook)
  │     └── c_out projection → [batch, 1024, 768]
  └── decode() → VAE decoder
```

**Serving Characteristics:**
- Very fast, memory-bound operation
- Stateless lookup table
- Could be merged with Stage 3 or Stage 5

---

### Stage 5: VAE Latent Decoding

**Component:** `OneDDecoder`

| Attribute | Value |
|-----------|-------|
| **Architecture** | 24 transformer encoder layers |
| **Hidden Dim** | 768 |
| **Heads** | 12 |
| **Input** | VQ embeddings `[batch, 1024, 768]` |
| **Output** | Latent features `[batch, 1024, 768]` |
| **Parameters** | ~113M |
| **Compute Type** | Standard transformer (non-autoregressive) |
| **Latency** | ~200-500ms on A100 |

**Key Code Path:**
```
one_d_autoencoder.py:284-309 → OneDDecoder.forward()
  ├── Add positional encodings
  └── for block in blocks[0:24]:
        └── EncoderLayer() → self-attention + FFN
```

**Serving Characteristics:**
- Fully parallelizable (non-autoregressive)
- Standard transformer - optimizable with TensorRT/ONNX
- Moderate compute, good batching characteristics

---

### Stage 6: Occupancy Field Query

**Component:** `OneDOccupancyDecoder` + dense grid generation

| Attribute | Value |
|-----------|-------|
| **Grid Resolution** | 2^8 = 256 per axis (default) |
| **Total Query Points** | ~16.7 million (257³) |
| **Chunk Size** | 100,000 points per forward pass |
| **Input** | 3D coordinates `[N, 3]` + latents `[batch, 1024, 768]` |
| **Output** | Occupancy logits `[batch, 257, 257, 257]` |
| **Parameters** | ~19M (occupancy decoder) |
| **Compute Type** | Cross-attention queries, highly parallelizable |
| **Latency** | ~2-5 seconds on A100 |

**Key Code Path:**
```
one_d_autoencoder.py:570-655 → extract_geometry()
  ├── generate_dense_grid_points() → [16.7M, 3]
  └── for chunk in chunks:  # ~167 chunks
        ├── PhaseModulatedFourierEmbedder → [chunk, 771]
        └── occupancy_decoder.forward()
              ├── query_in → MLPEmbedder
              ├── attn_out → cross-attention with latents
              └── c_head → [chunk, 1] occupancy logits
```

**Serving Characteristics:**
- **SECOND MOST COMPUTE-INTENSIVE STAGE**
- Embarrassingly parallel across query points
- Memory bandwidth bound at high resolutions
- Chunking required to fit in GPU memory
- Resolution is configurable (2^resolution_base)

---

### Stage 7: Mesh Extraction (Marching Cubes)

**Component:** NVIDIA Warp or scikit-image marching cubes

| Attribute | Value |
|-----------|-------|
| **Algorithm** | Marching Cubes (isosurface extraction) |
| **Input** | Occupancy grid `[257, 257, 257]` |
| **Output** | Vertices `[N_v, 3]`, Faces `[N_f, 3]` |
| **Typical Output** | 50K-200K faces |
| **Parameters** | None (algorithmic) |
| **Compute Type** | GPU-accelerated isosurface extraction |
| **Latency** | ~100-500ms on A100 |

**Key Code Path:**
```
grid.py:41-84 → marching_cubes_with_warp()
  ├── wp.MarchingCubes(nx=257, ny=257, nz=257)
  └── iso.surface(field, threshold=0.0)

Fallback (CPU): skimage.measure.marching_cubes()
```

**Serving Characteristics:**
- GPU-accelerated with NVIDIA Warp
- Falls back to CPU scikit-image if Warp unavailable
- Pure compute, no learned parameters
- Deterministic given same input grid

---

### Stage 8: Post-Processing (Optional)

**Component:** PyMeshLab operations

| Attribute | Value |
|-----------|-------|
| **Operations** | cleanup, remove_floaters, simplify_mesh |
| **Target Faces** | max(10000, original × 0.1) |
| **Input** | Raw mesh `[N_v, 3]`, `[N_f, 3]` |
| **Output** | Cleaned mesh (typically ~10K faces) |
| **Compute Type** | CPU-based mesh processing |
| **Latency** | ~500ms-2s on CPU |

**Key Code Path:**
```
postprocessing.py:78-84 → postprocess_mesh()
  ├── cleanup() → remove degenerate geometry
  ├── remove_floaters() → remove disconnected components
  └── simplify_mesh() → quadric edge collapse
```

**Serving Characteristics:**
- CPU-only (PyMeshLab)
- Optional - can be skipped if full-resolution mesh needed
- Good candidate for async/background processing

---

## 3. Component Dependency Graph

```
                    ┌─────────────────┐
                    │   Text Prompt   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  STAGE 1: CLIP  │ ← Stateless, batchable
                    │   Text Encoder  │
                    └────────┬────────┘
                             │ [batch, 77, 768]
                    ┌────────▼────────┐
                    │ STAGE 2: Cond   │ ← Trivial projection
                    │   Preparation   │
                    └────────┬────────┘
                             │ [batch, 77-78, 1536]
           ┌─────────────────┼─────────────────┐
           │ (classifier-free guidance split) │
           ▼                                   ▼
    ┌──────────────┐                 ┌──────────────┐
    │  Conditional │                 │Unconditional │
    │   Embeddings │                 │  Embeddings  │
    └──────┬───────┘                 └──────┬───────┘
           └─────────────────┬─────────────────┘
                             │
                    ┌────────▼────────┐
                    │ STAGE 3: GPT    │ ← BOTTLENECK (~80% time)
                    │  Autoregressive │   Sequential, 1024 steps
                    │  Generation     │   KV-cache + CUDA graphs
                    └────────┬────────┘
                             │ [batch, 1024] token IDs
                    ┌────────▼────────┐
                    │ STAGE 4: VQ     │ ← Fast lookup
                    │ Codebook Lookup │
                    └────────┬────────┘
                             │ [batch, 1024, 768]
                    ┌────────▼────────┐
                    │ STAGE 5: VAE    │ ← Parallelizable
                    │    Decoder      │
                    └────────┬────────┘
                             │ [batch, 1024, 768] latents
                    ┌────────▼────────┐
                    │ STAGE 6: Occ.   │ ← Heavy compute
                    │ Field Query     │   Embarrassingly parallel
                    └────────┬────────┘
                             │ [batch, 257, 257, 257]
                    ┌────────▼────────┐
                    │ STAGE 7: March  │ ← GPU-accelerated
                    │ Cubes           │   NVIDIA Warp
                    └────────┬────────┘
                             │ [vertices, faces]
                    ┌────────▼────────┐
                    │ STAGE 8: Post   │ ← CPU, optional
                    │ Processing      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   3D Mesh .obj  │
                    └─────────────────┘
```

---

## 4. Serving Architecture Options

### Option A: Monolithic Serving

**Description:** Single service handles entire pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Single GPU Service                       │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ │
│  │CLIP │→│Cond │→│ GPT │→│ VQ │→│VAE │→│Occ │→│ MC │→│Post│ │
│  └─────┘ └─────┘ └─────┘ └────┘ └────┘ └────┘ └────┘ └────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Simple deployment and orchestration
- No inter-service network latency
- Shared GPU memory for all models
- Single codebase, easier debugging

**Cons:**
- GPU underutilized during GPT autoregressive steps
- Can't scale individual components
- All models must fit on one GPU (~2-3GB VRAM total)
- Single point of failure

**Best For:**
- Low-volume deployments
- Latency-critical applications (no network hops)
- Development/testing environments

---

### Option B: Two-Stage Decomposition (Generation + Rendering)

**Description:** Separate GPT token generation from mesh rendering

```
┌────────────────────────────────┐    ┌─────────────────────────────┐
│      Stage 1: Token Gen        │    │    Stage 2: Mesh Render     │
│  ┌─────┐ ┌─────┐ ┌─────┐       │    │  ┌────┐ ┌────┐ ┌────┐ ┌────┐│
│  │CLIP │→│Cond │→│ GPT │───────┼───→│  │ VQ │→│VAE │→│Occ │→│ MC ││
│  └─────┘ └─────┘ └─────┘       │    │  └────┘ └────┘ └────┘ └────┘│
│         GPU-heavy              │    │       GPU-heavy             │
│    (autoregressive compute)    │    │    (parallel compute)       │
└────────────────────────────────┘    └─────────────────────────────┘

           [1024 token IDs]  →→→→→→→→
            (4KB payload)
```

**Pros:**
- Clear separation of concerns
- Tiny intermediate representation (4KB = 1024 × 4-byte ints)
- Token generation can be scaled independently
- Mesh rendering can use different GPU types
- Can cache/reuse token sequences

**Cons:**
- Network hop adds ~1-5ms latency
- Requires service orchestration
- More complex deployment

**Interface Contract:**
```python
# Stage 1 Output / Stage 2 Input
TokenGenerationResult = {
    "token_ids": List[int],  # 1024 integers in [0, 16383]
    "guidance_scale": float,
    "resolution_base": float,
    "bounding_box": Optional[Tuple[float, float, float]]
}
```

**Best For:**
- Medium to high volume deployments
- When caching token sequences is valuable
- Heterogeneous GPU fleet (use A100 for GPT, cheaper GPUs for rendering)

---

### Option C: Four-Stage Microservices

**Description:** Maximum decomposition into independent services

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Text Encoder    │   │  Token Generator │   │  Latent Decoder  │   │  Mesh Extractor  │
│  ┌─────┐ ┌─────┐ │   │      ┌─────┐     │   │ ┌────┐ ┌────┐    │   │  ┌────┐ ┌────┐   │
│  │CLIP │→│Cond │─┼──→│      │ GPT │─────┼──→│ │ VQ │→│VAE │────┼──→│  │Occ │→│ MC │   │
│  └─────┘ └─────┘ │   │      └─────┘     │   │ └────┘ └────┘    │   │  └────┘ └────┘   │
│     CPU/GPU      │   │    GPU A100      │   │    GPU T4/A10    │   │   GPU + CPU      │
└──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
    [77×1536]              [1024 ints]           [1024×768]            [257³] → mesh
    ~472KB                    4KB                   ~3MB               ~67MB → ~1MB
```

**Pros:**
- Maximum flexibility in scaling
- Each service optimized for its workload
- Can use specialized hardware per stage
- Fault isolation
- Independent deployment and versioning

**Cons:**
- Significant network overhead (especially Stage 3→4)
- Complex orchestration (Kubernetes, service mesh)
- Increased end-to-end latency
- More operational complexity

**Interface Contracts:**

```python
# Service 1 → Service 2
TextEncodingResult = {
    "condition_embeddings": np.ndarray,  # [77-78, 1536] float16
    "bounding_box": Optional[Tuple[float, float, float]],
    "guidance_scale": float
}

# Service 2 → Service 3
TokenGenerationResult = {
    "token_ids": List[int],  # 1024 integers
}

# Service 3 → Service 4
LatentDecodingResult = {
    "latents": np.ndarray,  # [1024, 768] float16
    "resolution_base": float
}

# Service 4 Output
MeshResult = {
    "vertices": np.ndarray,  # [N_v, 3] float32
    "faces": np.ndarray,     # [N_f, 3] int32
}
```

**Best For:**
- Very high volume (millions of requests/day)
- Teams with strong microservices infrastructure
- Need for fine-grained autoscaling

---

### Option D: Hybrid with Async Post-Processing

**Description:** Synchronous generation + async mesh optimization

```
┌──────────────────────────────────────────────────────────┐
│              Synchronous Generation Service              │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐     │
│  │CLIP │→│Cond │→│ GPT │→│ VQ │→│VAE │→│Occ │→│ MC │─────┼──→ Raw Mesh
│  └─────┘ └─────┘ └─────┘ └────┘ └────┘ └────┘ └────┘     │      (immediate)
└──────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                            ┌───────────────┐
                                            │   Message     │
                                            │    Queue      │
                                            └───────┬───────┘
                                                    │
                                            ┌───────▼───────┐
                                            │  CPU Workers  │
                                            │  ┌────────┐   │
                                            │  │  Post  │   │──→ Optimized Mesh
                                            │  │Process │   │      (delayed)
                                            │  └────────┘   │
                                            └───────────────┘
```

**Pros:**
- Fast initial response (skip post-processing)
- CPU-bound work offloaded from GPU service
- Can use cheap CPU instances for post-processing
- Progressive mesh delivery (raw → optimized)

**Cons:**
- Two-phase delivery more complex for clients
- Need message queue infrastructure

**Best For:**
- Real-time preview use cases
- When mesh quality can be progressive

---

## 5. Component Analysis: Compute & Memory

| Stage | GPU Memory | Compute Intensity | Batching Efficiency | Parallelizable |
|-------|------------|-------------------|---------------------|----------------|
| 1. CLIP | ~300MB | Low | Excellent | Yes |
| 2. Condition | <1MB | Negligible | N/A | Yes |
| 3. GPT | ~1.2GB | **Very High** | Poor (sequential) | No |
| 4. VQ Lookup | ~2MB | Low | Excellent | Yes |
| 5. VAE Decoder | ~450MB | Medium | Good | Yes |
| 6. Occupancy | ~100MB | **High** | Excellent | Yes |
| 7. Marching Cubes | ~500MB | Medium | Good | Yes |
| 8. Post-Process | CPU only | Low | Good | Yes |

**Total GPU Memory:** ~2.5-3GB (all models loaded)

---

## 6. Latency Breakdown (A100, batch=1)

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| 1. CLIP Text Encoding | 15 | 0.1% |
| 2. Condition Preparation | 1 | <0.1% |
| 3. GPT Generation (1024 tokens) | 8,000-12,000 | **75-85%** |
| 4. VQ Codebook Lookup | 3 | <0.1% |
| 5. VAE Decoding | 300 | 2-3% |
| 6. Occupancy Query | 2,000-4,000 | **15-25%** |
| 7. Marching Cubes | 200 | 1-2% |
| 8. Post-Processing | 500-1,500 | 4-10% |
| **Total** | **11,000-18,000** | 100% |

---

## 7. Key Optimizations Currently Implemented

### 7.1 KV-Cache for GPT
**File:** `engine.py:244-251`, `cache.py`

Reduces per-token compute from O(sequence_length) to O(1) by caching key/value tensors.

### 7.2 CUDA Graphs (EngineFast)
**File:** `engine.py:352-569`

Captures the GPT forward pass as a CUDA graph, eliminating CPU-GPU synchronization overhead. Provides 2-3x speedup.

### 7.3 Classifier-Free Guidance Batching
**File:** `engine.py:149-153, 269-274`

Runs conditional and unconditional paths in a single batched forward pass, avoiding redundant computation.

### 7.4 Chunked Occupancy Queries
**File:** `one_d_autoencoder.py:633-647`

Processes the 16.7M grid points in 100K chunks to fit in GPU memory while maintaining parallelism.

### 7.5 GPU-Accelerated Marching Cubes
**File:** `grid.py:41-84`

Uses NVIDIA Warp for GPU-native isosurface extraction, 10-100x faster than CPU alternatives.

---

## 8. Recommendations by Use Case

### Low Latency, Single Request
- **Architecture:** Monolithic with EngineFast
- **GPU:** A100 or H100
- **Optimizations:** CUDA graphs, KV-cache, bf16

### High Throughput, Batch Processing
- **Architecture:** Option B (Two-Stage)
- **GPU:** A100 for GPT, A10/T4 for rendering
- **Optimizations:** Batch GPT requests, parallelize rendering

### Production at Scale
- **Architecture:** Option C (Four-Stage) or Option D (Hybrid)
- **Infrastructure:** Kubernetes, service mesh, message queues
- **Considerations:** Auto-scaling policies per service, monitoring per stage

### Edge/Consumer Devices
- **Architecture:** Monolithic with quantization
- **Optimizations:** INT8/INT4 quantization, reduced resolution
- **Hardware:** RTX 4090, RTX 3080

---

## 9. Data Serialization Recommendations

| Interface | Recommended Format | Size (typical) |
|-----------|-------------------|----------------|
| CLIP → GPT | Shared memory (monolithic) or MessagePack | 472KB |
| GPT → VQ | JSON or Protobuf | 4KB |
| VQ → VAE | Arrow/Parquet or NumPy binary | 3MB |
| VAE → Occupancy | Shared tensor (monolithic) | 3MB |
| Occupancy → MC | GPU tensor (in-memory) | 67MB |
| MC → Client | OBJ, GLB, or Draco-compressed | 1-5MB |

---

## 10. Monitoring & Observability

Key metrics to track per stage:

| Stage | Metrics |
|-------|---------|
| CLIP | Tokenization time, embedding extraction time |
| GPT | Tokens/second, KV-cache hit rate, guidance scale |
| VQ Decode | Lookup latency, codebook usage distribution |
| VAE Decode | Forward pass time, memory utilization |
| Occupancy | Queries/second, chunk processing time |
| Marching Cubes | Vertices extracted, triangles generated |
| Post-Process | Simplification ratio, cleanup operations count |

---

## 11. Appendix: Model Parameter Counts

| Component | Parameters | Size (FP16) |
|-----------|------------|-------------|
| CLIP Text Encoder | ~124M | 248MB |
| DualStreamRoformer (GPT) | ~470M | 940MB |
| OneDEncoder | ~75M | 150MB |
| SphericalVectorQuantizer | ~0.6M | 1.2MB |
| OneDDecoder | ~113M | 226MB |
| OneDOccupancyDecoder | ~19M | 38MB |
| **Total** | **~802M** | **~1.6GB** |

---

## 12. Appendix: File Reference

| Stage | Primary Files |
|-------|---------------|
| Entry Point | `cube3d/generate.py`, `cube3d/inference/engine.py` |
| CLIP | HuggingFace `transformers` (external) |
| GPT Model | `cube3d/model/gpt/dual_stream_roformer.py` |
| Attention | `cube3d/model/transformers/dual_stream_attention.py`, `roformer.py` |
| RoPE | `cube3d/model/transformers/rope.py` |
| KV Cache | `cube3d/model/transformers/cache.py` |
| Autoencoder | `cube3d/model/autoencoder/one_d_autoencoder.py` |
| VQ | `cube3d/model/autoencoder/spherical_vq.py` |
| Grid/MC | `cube3d/model/autoencoder/grid.py` |
| Embedder | `cube3d/model/autoencoder/embedder.py` |
| Post-Process | `cube3d/mesh_utils/postprocessing.py` |
| Config | `cube3d/configs/open_model_v0.5.yaml` |
