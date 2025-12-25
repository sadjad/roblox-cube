# Scaling the Cube3D Serving Pipeline

This document explains how to scale each component of the pipeline to maximize throughput.

## Pipeline Latency Analysis

| Stage | Latency (A100) | Throughput/replica |
|-------|----------------|-------------------|
| CLIP  | ~20ms          | ~50 req/s         |
| GPT   | ~10s           | ~0.1 req/s        |
| Mesh  | ~3s            | ~0.33 req/s       |

**Key insight**: GPT is 500x slower than CLIP and 30x slower than Mesh decoding. This is your bottleneck.

## Calculating Replica Counts

For a target throughput of **T requests/second**:

```
CLIP_REPLICAS = ceil(T / 50)      # Usually 1 is enough
GPT_REPLICAS  = ceil(T / 0.1)     # 10 replicas per 1 req/s
MESH_REPLICAS = ceil(T / 0.33)    # 3 replicas per 1 req/s
WORKERS       = max(GPT_REPLICAS, MESH_REPLICAS)
```

### Example: 2 requests/second target

```bash
CLIP_REPLICAS=1    # 1 replica handles 50 req/s (way more than needed)
GPT_REPLICAS=20    # 20 * 0.1 = 2 req/s capacity
MESH_REPLICAS=6    # 6 * 0.33 = 2 req/s capacity
WORKER_REPLICAS=20 # Match GPT to keep pipeline fed

docker-compose -f docker-compose.scaled.yml up -d
```

## Three Levels of Scaling

### Level 1: Triton Instance Groups (Model Parallelism)

Multiple model instances on the **same GPU**. Useful when GPU memory allows and you want to hide memory transfer latency.

```protobuf
# In config.pbtxt
instance_group [
  {
    count: 2          # 2 instances on GPU 0
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**When to use:**
- GPU memory is underutilized
- Model is small relative to GPU memory
- Want to overlap compute with memory transfers

**Limitations:**
- Bounded by GPU memory
- Diminishing returns after 2-4 instances

### Level 2: Triton Server Replicas (Container Scaling)

Multiple Triton containers, each on its own GPU (or sharing GPUs with instance groups).

```bash
# Scale GPT to 8 replicas
docker-compose -f docker-compose.scaled.yml up -d --scale triton-gpt=8
```

**When to use:**
- Multi-GPU systems
- Need more throughput than one GPU provides
- Kubernetes cluster with GPU nodes

**Architecture:**
```
            ┌─────────────────┐
            │  Load Balancer  │
            │  (lb-gpt:8001)  │
            └────────┬────────┘
                     │ gRPC (least_conn)
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ triton-gpt │ │ triton-gpt │ │ triton-gpt │
│  replica 1 │ │  replica 2 │ │  replica 3 │
│   GPU 0    │ │   GPU 1    │ │   GPU 2    │
└────────────┘ └────────────┘ └────────────┘
```

### Level 3: Temporal Workers (Activity Parallelism)

Multiple Temporal workers to parallelize activity execution and keep all Triton replicas busy.

```bash
# Scale workers
docker-compose -f docker-compose.scaled.yml up -d --scale worker=20
```

**When to use:**
- Always scale workers to match or exceed your Triton GPT replicas
- Workers are lightweight (just gRPC calls), so over-provision is fine

**How it works:**
```
┌──────────────────────────────────────────────────────────┐
│                    Temporal Server                        │
│                                                          │
│  Task Queue: cube3d-mesh-generation                      │
│  Pending Activities: [encode_text, generate_tokens, ...] │
└────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ Worker 1│       │ Worker 2│       │ Worker 3│
   │         │       │         │       │         │
   │ Running:│       │ Running:│       │ Running:│
   │ gen_tok │       │ decode  │       │ encode  │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                 │                 │
        ▼                 ▼                 ▼
   Triton GPT        Triton Mesh      Triton CLIP
```

## Load Balancing Strategy

The nginx load balancers use `least_conn` algorithm:

```nginx
upstream triton_gpt_grpc {
    least_conn;  # Route to replica with fewest active connections
    server triton-gpt:8001;
}
```

**Why `least_conn`:**
- GPT requests take 10+ seconds
- Round-robin would create head-of-line blocking
- `least_conn` routes new requests to idle replicas

**Alternative: Client-side load balancing**

For even better performance, you can use gRPC's built-in load balancing:

```python
# In inference_activities.py
import grpc

channel = grpc.insecure_channel(
    'triton-gpt:8001',
    options=[
        ('grpc.lb_policy_name', 'round_robin'),
        ('grpc.enable_retries', 1),
    ]
)
```

## Kubernetes Deployment (Production)

For production, use Kubernetes with Horizontal Pod Autoscaling:

```yaml
# kubernetes/gpt-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-gpt
spec:
  replicas: 10  # Base replicas
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-gpt-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-gpt
  minReplicas: 4
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: triton_queue_duration_us
      target:
        type: AverageValue
        averageValue: 100000  # Scale up if queue time > 100ms
```

## Monitoring for Scaling Decisions

### Triton Metrics

```bash
# Check queue depth (if growing, need more replicas)
curl http://localhost:8002/metrics | grep nv_inference_pending_request_count

# Check inference latency
curl http://localhost:8002/metrics | grep nv_inference_compute_infer_duration_us
```

### Key metrics to watch:

| Metric | Meaning | Action if High |
|--------|---------|----------------|
| `nv_inference_pending_request_count` | Requests waiting in queue | Add more Triton replicas |
| `nv_inference_queue_duration_us` | Time spent waiting | Add more Triton replicas |
| `nv_inference_compute_infer_duration_us` | Actual compute time | Optimize model or use faster GPU |

### Temporal Metrics

```bash
# Check activity task queue depth
temporal task-queue describe --task-queue cube3d-mesh-generation
```

If `backlogCountHint` is growing, add more workers.

## Quick Reference

```bash
# Start with default scaling (1 CLIP, 4 GPT, 2 Mesh)
docker-compose -f docker-compose.scaled.yml up -d

# Scale GPT to handle more throughput
docker-compose -f docker-compose.scaled.yml up -d --scale triton-gpt=16

# Scale all components for 5 req/s
CLIP_REPLICAS=1 \
GPT_REPLICAS=50 \
MESH_REPLICAS=15 \
WORKER_REPLICAS=50 \
docker-compose -f docker-compose.scaled.yml up -d

# Check replica counts
docker-compose -f docker-compose.scaled.yml ps
```

## Cost Optimization

GPT dominates both compute time AND cost. Consider:

1. **Spot/Preemptible GPUs** for GPT replicas (Temporal handles retries)
2. **Mixed instance types**: A100 for GPT, cheaper T4/A10 for CLIP/Mesh
3. **Batch requests** when latency allows (Triton dynamic batching)
4. **Cache token sequences** for repeated prompts
