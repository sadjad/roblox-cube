# Cube3D Serving Infrastructure

This directory contains the infrastructure for serving the Cube3D text-to-mesh model using Triton Inference Server and Temporal for workflow orchestration.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  Client                                          │
│                            POST /generate                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API Gateway (FastAPI)                               │
│                              localhost:8080                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Temporal Server                                        │
│                     Workflow Orchestration                                       │
│                     localhost:7233 (gRPC)                                        │
│                     localhost:8088 (Web UI)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Temporal Worker                                        │
│                   Executes workflow activities                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
           │                          │                          │
           ▼                          ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   Triton: CLIP      │  │   Triton: GPT       │  │   Triton: Mesh      │
│   Text Encoder      │  │   Token Generator   │  │   Decoder           │
│   :8001             │  │   :8011             │  │   :8021             │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

## Directory Structure

```
serving/
├── docker-compose.yml           # Multi-GPU setup (3 Triton servers)
├── docker-compose.single-gpu.yml # Single GPU setup (1 Triton server)
├── temporal-config/             # Temporal dynamic configuration
├── triton/
│   └── models/
│       ├── clip_encoder/        # CLIP text encoding model
│       │   ├── config.pbtxt
│       │   └── 1/model.py
│       ├── gpt_generator/       # GPT token generation model
│       │   ├── config.pbtxt
│       │   └── 1/model.py
│       └── mesh_decoder/        # Mesh decoding model
│           ├── config.pbtxt
│           └── 1/model.py
├── temporal/
│   ├── workflows/               # Temporal workflow definitions
│   ├── activities/              # Temporal activities (Triton calls)
│   ├── worker.py                # Worker entry point
│   ├── Dockerfile
│   └── requirements.txt
├── api/
│   ├── main.py                  # FastAPI application
│   ├── Dockerfile
│   └── requirements.txt
└── shared/
    ├── data_models.py           # Shared data classes
    └── triton_client.py         # Triton client utilities
```

## Prerequisites

1. **NVIDIA GPU** with CUDA support (A100 recommended for best performance)
2. **Docker** with NVIDIA Container Toolkit
3. **Model weights** (see "Exporting Weights" section)

## Quick Start

### 1. Export Model Weights

Before starting the services, you need to export the model weights to a format Triton can load:

```bash
# Create weights directory
mkdir -p serving/weights

# Export GPT weights (safetensors format)
# You'll need to copy or export your trained weights here
cp path/to/your/gpt_weights.safetensors serving/weights/gpt.safetensors
cp path/to/your/shape_weights.safetensors serving/weights/shape.safetensors
```

### 2. Start Services (Single GPU)

For development or single-GPU setups:

```bash
cd serving

# Set weights path
export WEIGHTS_PATH=./weights

# Start all services
docker-compose -f docker-compose.single-gpu.yml up -d

# Watch logs
docker-compose -f docker-compose.single-gpu.yml logs -f
```

### 3. Start Services (Multi-GPU)

For production with multiple GPUs:

```bash
cd serving

# Set weights path
export WEIGHTS_PATH=./weights

# Start all services
docker-compose up -d
```

### 4. Verify Services

```bash
# Check Temporal UI
open http://localhost:8088

# Check API health
curl http://localhost:8080/health

# Check Triton health
curl http://localhost:8000/v2/health/ready
```

## API Usage

### Generate Mesh (Synchronous)

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a wooden chair",
    "guidance_scale": 3.0,
    "resolution_base": 8.0
  }'
```

### Generate Mesh (Asynchronous)

```bash
# Start generation
curl -X POST http://localhost:8080/generate/async \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red sports car"}'

# Response: {"workflow_id": "mesh-abc123", "status": "running", ...}

# Check status
curl http://localhost:8080/status/mesh-abc123

# Get result when complete
curl http://localhost:8080/result/mesh-abc123

# Download as OBJ file
curl http://localhost:8080/result/mesh-abc123/obj > mesh.obj
```

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the 3D object |
| `guidance_scale` | float | 3.0 | Classifier-free guidance scale (0-20) |
| `resolution_base` | float | 8.0 | Grid resolution as 2^n (6-9) |
| `top_p` | float | null | Nucleus sampling (null=deterministic) |
| `bounding_box` | [x,y,z] | null | Object dimensions |
| `wait` | bool | true | Wait for completion or return immediately |
| `timeout_seconds` | int | 600 | Timeout for sync requests |

## Modifying the Pipeline Split

The current architecture splits the pipeline into 3 stages:

1. **CLIP Encoder**: Text → Embeddings
2. **GPT Generator**: Embeddings → Tokens
3. **Mesh Decoder**: Tokens → Mesh

To change how stages are split:

### Option A: Combine CLIP + GPT

Edit `triton/models/gpt_generator/1/model.py` to include CLIP encoding.
Then remove `triton-clip` from docker-compose and update worker environment variables.

### Option B: Split Mesh Decoder Further

Create separate models for:
- `vq_decoder`: Tokens → Latents
- `vae_decoder`: Latents → Occupancy Grid
- `marching_cubes`: Grid → Mesh

Update `temporal/activities/inference_activities.py` to call the new models.

### Option C: Use Triton Ensemble

Instead of Temporal, use Triton's built-in ensemble feature by creating a `config.pbtxt` that chains models:

```protobuf
name: "text_to_mesh_ensemble"
platform: "ensemble"
ensemble_scheduling {
  step [
    { model_name: "clip_encoder" ... },
    { model_name: "gpt_generator" ... },
    { model_name: "mesh_decoder" ... }
  ]
}
```

## Monitoring

### Temporal Web UI

Access the Temporal UI at http://localhost:8088 to:
- View running workflows
- Inspect workflow history
- Debug failed workflows
- Cancel stuck workflows

### Triton Metrics

Triton exposes Prometheus metrics on port 8002:

```bash
curl http://localhost:8002/metrics
```

Key metrics:
- `nv_inference_request_success`: Successful inferences
- `nv_inference_request_failure`: Failed inferences
- `nv_inference_queue_duration_us`: Time spent in queue
- `nv_inference_compute_infer_duration_us`: Inference compute time

### API Logs

```bash
docker-compose logs -f api
docker-compose logs -f worker
```

## Performance Tuning

### GPU Memory

Adjust `shm_size` in docker-compose.yml based on your GPU:
- 16GB GPU: `shm_size: '4gb'`
- 24GB GPU: `shm_size: '6gb'`
- 40GB+ GPU: `shm_size: '8gb'`

### Batching

Edit `config.pbtxt` for each model to tune batching:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: 100000
}
```

### Model Instances

Add more model instances for higher throughput:

```protobuf
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [ 0 ] }
]
```

## Troubleshooting

### Triton won't start

Check model loading:
```bash
docker-compose logs triton-gpt | grep -i error
```

Common issues:
- Missing weight files in `/models/weights`
- Python dependencies not installed
- GPU memory insufficient

### Temporal workflow fails

Check the Temporal UI for error details, or:
```bash
temporal workflow show -w <workflow-id>
```

### API returns 503

Temporal not connected:
```bash
docker-compose logs api | grep -i temporal
```

## Development

### Running Worker Locally

```bash
cd serving
pip install -r temporal/requirements.txt

export TEMPORAL_HOST=localhost:7233
export CLIP_TRITON_URL=localhost:8001
export GPT_TRITON_URL=localhost:8011
export MESH_TRITON_URL=localhost:8021

python temporal/worker.py
```

### Running API Locally

```bash
cd serving
pip install -r api/requirements.txt

export TEMPORAL_HOST=localhost:7233
uvicorn api.main:app --reload --port 8080
```

## License

See the main repository LICENSE file.
