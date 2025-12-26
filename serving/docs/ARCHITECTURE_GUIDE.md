# From Triton to Production: A Complete Guide to Scalable ML Pipelines

This document explains how to scale a multi-stage ML pipeline (like Cube3D's text-to-3D generation) from a single Triton server to a fully autoscaling Kubernetes deployment. If you're familiar with Triton Inference Server but new to workflow orchestration, Kubernetes, or autoscaling, this guide is for you.

## Table of Contents

1. [The Challenge: Multi-Stage ML Pipelines](#the-challenge-multi-stage-ml-pipelines)
2. [Why Triton Alone Isn't Enough](#why-triton-alone-isnt-enough)
3. [Enter Temporal: Workflow Orchestration](#enter-temporal-workflow-orchestration)
4. [The Two-Layer Architecture](#the-two-layer-architecture)
5. [Scaling Strategies](#scaling-strategies)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Autoscaling with KEDA](#autoscaling-with-keda)
8. [Putting It All Together](#putting-it-all-together)
9. [Monitoring and Operations](#monitoring-and-operations)

---

## The Challenge: Multi-Stage ML Pipelines

### The Cube3D Pipeline

Cube3D generates 3D meshes from text prompts. This isn't a single model inference—it's a pipeline of three distinct stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Cube3D Pipeline                                      │
│                                                                              │
│   "a wooden chair"                                                           │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐   │
│   │   CLIP    │      │    GPT    │      │   Mesh    │      │   .OBJ    │   │
│   │  Encoder  │─────▶│ Generator │─────▶│  Decoder  │─────▶│   File    │   │
│   │           │      │           │      │           │      │           │   │
│   │  ~20ms    │      │  ~10sec   │      │   ~3sec   │      │           │   │
│   └───────────┘      └───────────┘      └───────────┘      └───────────┘   │
│                                                                              │
│   Text → Embeddings   Embeddings → Tokens   Tokens → Mesh                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The key insight**: Each stage has dramatically different characteristics:

| Stage | Latency | GPU Memory | Throughput (1 GPU) |
|-------|---------|------------|-------------------|
| CLIP  | ~20ms   | ~2GB       | ~50 req/s         |
| GPT   | ~10s    | ~16GB      | ~0.1 req/s        |
| Mesh  | ~3s     | ~8GB       | ~0.33 req/s       |

GPT is **500x slower** than CLIP. This asymmetry is the core scaling challenge.

---

## Why Triton Alone Isn't Enough

### What Triton Does Well

Triton Inference Server excels at:

1. **Model hosting**: Load models once, serve many requests
2. **Batching**: Combine multiple requests for GPU efficiency
3. **Model formats**: Support for TensorRT, ONNX, PyTorch, TensorFlow
4. **Metrics**: Built-in Prometheus metrics for monitoring
5. **Concurrent execution**: Multiple model instances on one GPU

### The Single-Server Approach

You might start with a single Triton server hosting all models:

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton Server                             │
│                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │ clip_encoder│  │gpt_generator│  │mesh_decoder │        │
│   │  instance 1 │  │  instance 1 │  │  instance 1 │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
│                      GPU 0 (24GB)                           │
└─────────────────────────────────────────────────────────────┘
```

**Problems with this approach:**

1. **Bottleneck starvation**: GPT takes 10 seconds. While it's running, CLIP and Mesh are idle.
2. **No independent scaling**: You can't add more GPT capacity without also scaling CLIP/Mesh.
3. **Memory contention**: All models compete for the same GPU memory.
4. **Single point of failure**: One Triton server crash = complete outage.

### Triton Ensembles: Partial Solution

Triton supports [ensembles](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)—pipelines defined in config:

```protobuf
# ensemble_model/config.pbtxt
name: "text_to_mesh_ensemble"
platform: "ensemble"
ensemble_scheduling {
  step {
    model_name: "clip_encoder"
    model_version: -1
    input_map { key: "prompt" value: "prompt" }
    output_map { key: "embeddings" value: "clip_output" }
  }
  step {
    model_name: "gpt_generator"
    model_version: -1
    input_map { key: "embeddings" value: "clip_output" }
    output_map { key: "tokens" value: "gpt_output" }
  }
  step {
    model_name: "mesh_decoder"
    model_version: -1
    input_map { key: "tokens" value: "gpt_output" }
    output_map { key: "mesh" value: "mesh" }
  }
}
```

**Why ensembles still aren't enough:**

| Feature | Triton Ensemble | What We Need |
|---------|----------------|--------------|
| Cross-server routing | ❌ Same server only | ✅ Route to any server |
| Independent scaling | ❌ All steps together | ✅ Scale GPT separately |
| Failure recovery | ❌ Request fails | ✅ Automatic retry |
| Long-running jobs | ❌ HTTP timeout issues | ✅ Async with status |
| Progress tracking | ❌ No visibility | ✅ Per-stage status |

---

## Enter Temporal: Workflow Orchestration

### What is Temporal?

[Temporal](https://temporal.io) is a workflow orchestration platform. Think of it as a reliable, distributed state machine that coordinates work across multiple services.

**Key concepts:**

1. **Workflow**: A function that orchestrates a sequence of steps. Workflows are durable—if a server crashes, the workflow resumes where it left off.

2. **Activity**: A single unit of work (like calling Triton). Activities can fail and be retried.

3. **Task Queue**: A queue of work items. Workers poll queues for tasks to execute.

4. **Worker**: A process that executes workflows or activities.

### Why Temporal for ML Pipelines?

| Challenge | Temporal Solution |
|-----------|------------------|
| Coordinating 3 inference calls | Workflow defines the sequence |
| 10-second GPT inference | Async execution, no HTTP timeouts |
| GPT server crashes mid-inference | Automatic retry on another server |
| Need 10x more GPT capacity | Add workers to GPT queue only |
| User wants progress updates | Query workflow state anytime |
| Request spike overwhelms GPUs | Queue absorbs burst, processes at capacity |

### Temporal Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Temporal Server                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Task Queues                                   │   │
│  │                                                                      │   │
│  │   cube3d-workflow    cube3d-clip    cube3d-gpt    cube3d-mesh      │   │
│  │   ┌──┬──┬──┬──┐     ┌──┬──┬──┐    ┌──┬──┬──┐    ┌──┬──┬──┐       │   │
│  │   │W1│W2│W3│W4│     │A1│A2│A3│    │A1│A2│A3│    │A1│A2│A3│       │   │
│  │   └──┴──┴──┴──┘     └──┴──┴──┘    └──┴──┴──┘    └──┴──┴──┘       │   │
│  │   Workflows          Activities    Activities    Activities        │   │
│  │   pending...         pending...    pending...    pending...        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Workflow History                                 │   │
│  │   workflow-123: Started → CLIP done → GPT running...                │   │
│  │   workflow-124: Started → CLIP done → GPT done → Mesh running...    │   │
│  │   workflow-125: Completed (13.2s total)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**How it works:**

1. Client submits a request → Temporal creates a workflow
2. Workflow worker picks up the workflow, schedules CLIP activity
3. CLIP worker picks up activity, calls Triton, returns result
4. Workflow schedules GPT activity
5. GPT worker picks up activity, calls Triton, returns result
6. Workflow schedules Mesh activity
7. Mesh worker picks up activity, calls Triton, returns result
8. Workflow completes, client gets result

---

## The Two-Layer Architecture

Our solution uses two complementary layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                        LAYER 1: ORCHESTRATION                               │
│                           (Temporal)                                         │
│                                                                              │
│   • Workflow coordination        • Failure recovery                         │
│   • Task queuing                 • Progress tracking                        │
│   • Independent scaling          • Request buffering                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        LAYER 2: INFERENCE                                   │
│                          (Triton)                                           │
│                                                                              │
│   • GPU scheduling               • Model optimization                       │
│   • Dynamic batching             • Memory management                        │
│   • Model versioning             • Metrics                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Concern | Temporal | Triton |
|---------|----------|--------|
| "Which GPU runs this?" | ❌ | ✅ |
| "What order do stages run?" | ✅ | ❌ |
| "Retry failed inference?" | ✅ | ❌ |
| "Batch similar requests?" | ❌ | ✅ |
| "Add more GPT capacity?" | ✅ (add workers) | ✅ (add servers) |
| "Track request progress?" | ✅ | ❌ |

### Code Example: The Workflow

```python
# temporal/workflows/text_to_mesh.py

@workflow.defn
class TextToMeshWorkflow:
    """Orchestrates the text-to-mesh pipeline."""

    @workflow.run
    async def run(self, input: TextToMeshInput) -> TextToMeshOutput:
        # Stage 1: CLIP encoding
        # Routes to cube3d-clip queue → picked up by CLIP worker → calls Triton
        encoding = await workflow.execute_activity(
            encode_text,
            args=[input.prompt],
            task_queue="cube3d-clip",           # Separate queue!
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Stage 2: GPT generation (BOTTLENECK)
        # Routes to cube3d-gpt queue → picked up by GPT worker → calls Triton
        tokens = await workflow.execute_activity(
            generate_tokens,
            args=[encoding["embeddings"]],
            task_queue="cube3d-gpt",            # Separate queue!
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Stage 3: Mesh decoding
        # Routes to cube3d-mesh queue → picked up by Mesh worker → calls Triton
        mesh = await workflow.execute_activity(
            decode_mesh,
            args=[tokens["token_ids"]],
            task_queue="cube3d-mesh",           # Separate queue!
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return TextToMeshOutput(
            vertices=mesh["vertices"],
            faces=mesh["faces"],
        )
```

### Code Example: The Activity (Triton Call)

```python
# temporal/activities/inference_activities.py

@activity.defn
async def generate_tokens(
    embeddings: List[float],
    guidance_scale: float = 3.0,
) -> Dict[str, Any]:
    """
    Call Triton to generate shape tokens.

    This activity runs on cube3d-gpt queue.
    Multiple workers can process these in parallel.
    """
    # Create Triton client
    client = grpcclient.InferenceServerClient(url="triton-gpt:8001")

    try:
        # Prepare inputs
        inputs = [
            grpcclient.InferInput("embeddings", embeddings.shape, "FP16"),
            grpcclient.InferInput("guidance_scale", [1], "FP32"),
        ]
        inputs[0].set_data_from_numpy(embeddings)
        inputs[1].set_data_from_numpy(np.array([guidance_scale]))

        # Heartbeat for long-running inference
        # Tells Temporal "I'm still working, don't time me out"
        activity.heartbeat("Generating tokens...")

        # Call Triton
        result = client.infer(model_name="gpt_generator", inputs=inputs)

        return {"token_ids": result.as_numpy("token_ids").tolist()}

    except InferenceServerException as e:
        activity.logger.error(f"Triton error: {e}")
        raise  # Temporal will retry based on RetryPolicy
```

### Code Example: The Worker

```python
# temporal/worker.py

async def main():
    # Connect to Temporal
    client = await Client.connect("temporal:7233")

    # Create a worker for GPT activities
    # This worker ONLY handles GPT - other workers handle CLIP/Mesh
    worker = Worker(
        client,
        task_queue="cube3d-gpt",        # Only pulls from GPT queue
        activities=[generate_tokens],    # Only runs GPT activity
    )

    # Start polling for tasks
    await worker.run()
```

---

## Scaling Strategies

### The Core Insight: Scale the Bottleneck

```
Throughput Goal: 1 request/second

CLIP:  1 req/s ÷ 50 req/s/worker  = 0.02 workers → 1 worker
GPT:   1 req/s ÷ 0.1 req/s/worker = 10 workers   → 10 workers  ← BOTTLENECK
Mesh:  1 req/s ÷ 0.33 req/s/worker = 3 workers   → 3 workers
```

**For 1 request/second, you need 10x more GPT workers than CLIP workers.**

### Three Levels of Scaling

#### Level 1: Triton Instance Groups (Same GPU)

Multiple model instances share one GPU. Useful when GPU memory allows.

```protobuf
# config.pbtxt
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
- Want to hide memory transfer latency
- Limited by: GPU memory, diminishing returns after 2-4 instances

#### Level 2: Triton Server Replicas (Multiple GPUs)

Multiple Triton containers, each with its own GPU.

```
┌────────────────────────────────────────────────────────────────┐
│                    Triton GPT Replicas                          │
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│   │ triton-gpt  │  │ triton-gpt  │  │ triton-gpt  │           │
│   │  replica 1  │  │  replica 2  │  │  replica 3  │           │
│   │    GPU 0    │  │    GPU 1    │  │    GPU 2    │           │
│   └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
│   Each processes ~0.1 req/s → Total: ~0.3 req/s               │
└────────────────────────────────────────────────────────────────┘
```

**When to use:**
- Multi-GPU systems
- Need more throughput than one GPU provides

#### Level 3: Temporal Workers (Activity Parallelism)

Multiple workers processing activities from the same queue.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Temporal Task Queue: cube3d-gpt              │
│                                                                 │
│   Pending Activities: [gen_tok_1, gen_tok_2, gen_tok_3, ...]   │
│                              │                                  │
│              ┌───────────────┼───────────────┐                 │
│              ▼               ▼               ▼                 │
│        ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│        │ Worker 1 │    │ Worker 2 │    │ Worker 3 │           │
│        │          │    │          │    │          │           │
│        │ Calls    │    │ Calls    │    │ Calls    │           │
│        │ Triton   │    │ Triton   │    │ Triton   │           │
│        └────┬─────┘    └────┬─────┘    └────┬─────┘           │
│             │               │               │                  │
│             ▼               ▼               ▼                  │
│        Triton GPT      Triton GPT      Triton GPT             │
│        replica 1       replica 2       replica 3              │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Workers are cheap (just Python processes making gRPC calls). Scale them to match or exceed Triton replicas to keep GPUs busy.

### Scaling Formula

For target throughput **T** requests/second:

```python
def calculate_replicas(target_throughput: float) -> dict:
    """Calculate required replicas for each component."""

    # Throughput per replica (from benchmarks)
    CLIP_THROUGHPUT = 50      # req/s per replica
    GPT_THROUGHPUT = 0.1      # req/s per replica
    MESH_THROUGHPUT = 0.33    # req/s per replica

    return {
        "triton_clip": max(1, ceil(target_throughput / CLIP_THROUGHPUT)),
        "triton_gpt": max(1, ceil(target_throughput / GPT_THROUGHPUT)),
        "triton_mesh": max(1, ceil(target_throughput / MESH_THROUGHPUT)),
        # Workers should match or exceed Triton replicas
        "worker_clip": max(1, ceil(target_throughput / CLIP_THROUGHPUT)),
        "worker_gpt": max(1, ceil(target_throughput / GPT_THROUGHPUT)),
        "worker_mesh": max(1, ceil(target_throughput / MESH_THROUGHPUT)),
    }

# Example: 2 requests/second
calculate_replicas(2.0)
# {
#     "triton_clip": 1,    # 1 replica handles 50 req/s
#     "triton_gpt": 20,    # 20 replicas × 0.1 = 2 req/s
#     "triton_mesh": 6,    # 6 replicas × 0.33 = 2 req/s
#     "worker_clip": 1,
#     "worker_gpt": 20,
#     "worker_mesh": 6,
# }
```

---

## Kubernetes Deployment

### Why Kubernetes?

| Need | Kubernetes Solution |
|------|---------------------|
| Run 20 GPT containers | Deployment with replicas: 20 |
| Distribute across GPUs | NVIDIA device plugin |
| Service discovery | Services and DNS |
| Load balancing | Service + Endpoints |
| Rolling updates | Deployment strategy |
| Auto-restart on crash | Pod restartPolicy |
| Resource limits | Resource requests/limits |

### Core Kubernetes Concepts

#### 1. Deployment

A Deployment manages identical Pod replicas:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-gpt
spec:
  replicas: 10                    # 10 identical pods
  selector:
    matchLabels:
      app: triton-gpt
  template:
    metadata:
      labels:
        app: triton-gpt
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        resources:
          limits:
            nvidia.com/gpu: 1     # Each pod gets 1 GPU
        ports:
        - containerPort: 8001     # gRPC port
```

#### 2. Service

A Service provides a stable endpoint for pods:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: triton-gpt              # DNS: triton-gpt.namespace.svc.cluster.local
spec:
  selector:
    app: triton-gpt             # Routes to all pods with this label
  ports:
  - name: grpc
    port: 8001
    targetPort: 8001
```

Now workers can call `triton-gpt:8001` and Kubernetes load-balances across all 10 replicas.

#### 3. GPU Scheduling

Kubernetes uses the NVIDIA device plugin to schedule GPUs:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1    # Request exactly 1 GPU
```

Kubernetes ensures each pod gets exclusive access to its GPU.

### Our Kubernetes Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          Services                                    │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ triton-clip  │  │  triton-gpt  │  │ triton-mesh  │              │   │
│  │  │    :8001     │  │    :8001     │  │    :8001     │              │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │   │
│  │         │                 │                 │                       │   │
│  └─────────┼─────────────────┼─────────────────┼───────────────────────┘   │
│            │                 │                 │                            │
│  ┌─────────▼─────────────────▼─────────────────▼───────────────────────┐   │
│  │                       Deployments                                    │   │
│  │                                                                      │   │
│  │  Triton CLIP (1 replica)     Triton GPT (10 replicas)               │   │
│  │  ┌─────┐                     ┌─────┬─────┬─────┬─────┬─────┐       │   │
│  │  │ Pod │                     │ Pod │ Pod │ Pod │ Pod │ ... │       │   │
│  │  │GPU 0│                     │GPU 1│GPU 2│GPU 3│GPU 4│     │       │   │
│  │  └─────┘                     └─────┴─────┴─────┴─────┴─────┘       │   │
│  │                                                                      │   │
│  │  Triton Mesh (3 replicas)                                           │   │
│  │  ┌─────┬─────┬─────┐                                                │   │
│  │  │ Pod │ Pod │ Pod │                                                │   │
│  │  │GPU11│GPU12│GPU13│                                                │   │
│  │  └─────┴─────┴─────┘                                                │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Worker Deployments                              │   │
│  │                                                                       │   │
│  │  Workflow Workers (2)  CLIP Workers (1)  GPT Workers (10)  Mesh (3)  │   │
│  │  ┌───┬───┐            ┌───┐             ┌───┬───┬...┐     ┌───┬───┐ │   │
│  │  │Pod│Pod│            │Pod│             │Pod│Pod│   │     │Pod│Pod│ │   │
│  │  └───┴───┘            └───┘             └───┴───┴───┘     └───┴───┘ │   │
│  │  (no GPU)             (no GPU)          (no GPU)          (no GPU)   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Note**: Workers don't need GPUs—they just make gRPC calls to Triton. This makes workers cheap to scale.

---

## Autoscaling with KEDA

### The Problem with Standard Autoscaling

Kubernetes has built-in autoscaling (HPA - Horizontal Pod Autoscaler), but it scales based on CPU/memory:

```yaml
# Standard HPA - scales on CPU
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Why this doesn't work for ML inference:**

1. GPU utilization isn't exposed to HPA
2. Queue depth is a better signal than CPU
3. Triton pods are idle waiting for requests, not spinning CPU

### KEDA: Kubernetes Event-Driven Autoscaling

[KEDA](https://keda.sh) extends Kubernetes autoscaling to support external metrics—including Temporal queue depth.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KEDA Architecture                               │
│                                                                              │
│                         ┌─────────────────┐                                 │
│                         │  Temporal Queue │                                 │
│                         │   cube3d-gpt    │                                 │
│                         │                 │                                 │
│                         │  Backlog: 15    │◄──────────────┐                │
│                         └────────┬────────┘               │                │
│                                  │                        │                │
│                                  │ Query                  │                │
│                                  ▼                        │                │
│                         ┌─────────────────┐               │                │
│                         │  KEDA Metrics   │               │                │
│                         │    Adapter      │               │                │
│                         └────────┬────────┘               │                │
│                                  │                        │                │
│                                  │ Scale                  │ Reduce         │
│                                  ▼ Up                     │ Backlog        │
│                         ┌─────────────────┐               │                │
│                         │   Deployment    │               │                │
│                         │   triton-gpt    │               │                │
│                         │                 │               │                │
│                         │ replicas: 4→8  │───────────────┘                │
│                         └─────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### KEDA ScaledObject

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: triton-gpt-scaler
spec:
  scaleTargetRef:
    name: triton-gpt                    # Deployment to scale
  minReplicaCount: 4                    # Never go below 4
  maxReplicaCount: 50                   # Never exceed 50
  pollingInterval: 10                   # Check queue every 10s
  cooldownPeriod: 30                    # Wait 30s before scaling down
  triggers:
  - type: temporal                      # KEDA Temporal trigger
    metadata:
      address: temporal:7233
      namespace: default
      taskQueue: cube3d-gpt             # Queue to monitor
      targetBacklogCount: "2"           # Scale up when backlog > 2
```

**How it works:**

1. KEDA polls Temporal every 10 seconds
2. If `cube3d-gpt` queue has >2 pending activities, KEDA increases replicas
3. Formula: `replicas = ceil(backlog / targetBacklogCount)`
4. If backlog is 20, KEDA scales to 10 replicas
5. When backlog drops, KEDA waits 30s then scales down

### Scaling Both Workers and Triton

For optimal performance, scale **both** the workers and Triton pods based on queue depth:

```yaml
# Worker ScaledObject
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: worker-gpt-scaler
spec:
  scaleTargetRef:
    name: worker-gpt
  triggers:
  - type: temporal
    metadata:
      taskQueue: cube3d-gpt
      targetBacklogCount: "2"

---
# Triton ScaledObject
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: triton-gpt-scaler
spec:
  scaleTargetRef:
    name: triton-gpt
  triggers:
  - type: temporal
    metadata:
      taskQueue: cube3d-gpt
      targetBacklogCount: "2"
```

Both scale together based on the same queue, keeping workers and Triton replicas in sync.

---

## Putting It All Together

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Complete Request Flow                               │
│                                                                              │
│  1. User Request                                                            │
│     POST /generate {"prompt": "a wooden chair"}                             │
│                          │                                                   │
│                          ▼                                                   │
│  2. API Gateway                                                             │
│     ┌──────────────────────────────────────────┐                           │
│     │  Validates request                        │                           │
│     │  Starts Temporal workflow                 │                           │
│     │  Returns workflow_id immediately          │                           │
│     └──────────────────────────────────────────┘                           │
│                          │                                                   │
│                          ▼                                                   │
│  3. Temporal Server                                                         │
│     ┌──────────────────────────────────────────┐                           │
│     │  Creates workflow execution               │                           │
│     │  Persists state to database               │                           │
│     │  Adds workflow task to queue              │                           │
│     └──────────────────────────────────────────┘                           │
│                          │                                                   │
│                          ▼                                                   │
│  4. Workflow Worker                                                         │
│     ┌──────────────────────────────────────────┐                           │
│     │  Polls workflow queue                     │                           │
│     │  Picks up workflow task                   │                           │
│     │  Schedules CLIP activity                  │                           │
│     └──────────────────────────────────────────┘                           │
│                          │                                                   │
│                          ▼                                                   │
│  5. CLIP Worker                                                             │
│     ┌──────────────────────────────────────────┐                           │
│     │  Polls cube3d-clip queue                  │                           │
│     │  Picks up encode_text activity            │                           │
│     │  Calls Triton CLIP → returns embeddings   │                           │
│     └──────────────────────────────────────────┘                           │
│                          │                                                   │
│                          ▼                                                   │
│  6. GPT Worker (BOTTLENECK)                                                 │
│     ┌──────────────────────────────────────────┐                           │
│     │  Polls cube3d-gpt queue                   │                           │
│     │  Picks up generate_tokens activity        │                           │
│     │  Calls Triton GPT → returns tokens        │                           │
│     │  (This takes ~10 seconds)                 │                           │
│     └──────────────────────────────────────────┘                           │
│                          │                                                   │
│                          ▼                                                   │
│  7. Mesh Worker                                                             │
│     ┌──────────────────────────────────────────┐                           │
│     │  Polls cube3d-mesh queue                  │                           │
│     │  Picks up decode_mesh activity            │                           │
│     │  Calls Triton Mesh → returns vertices     │                           │
│     └──────────────────────────────────────────┘                           │
│                          │                                                   │
│                          ▼                                                   │
│  8. Workflow Completes                                                      │
│     ┌──────────────────────────────────────────┐                           │
│     │  Result stored in Temporal                │                           │
│     │  Client polls GET /status/{workflow_id}   │                           │
│     │  Returns mesh data                        │                           │
│     └──────────────────────────────────────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Helm Chart Structure

The Helm chart packages everything for easy deployment:

```
helm/cube3d/
├── Chart.yaml                    # Chart metadata
├── values.yaml                   # Configuration
└── templates/
    ├── triton-clip-deployment.yaml
    ├── triton-gpt-deployment.yaml
    ├── triton-mesh-deployment.yaml
    ├── worker-workflow-deployment.yaml
    ├── worker-clip-deployment.yaml
    ├── worker-gpt-deployment.yaml
    ├── worker-mesh-deployment.yaml
    ├── api-deployment.yaml
    ├── services.yaml
    ├── keda-scaledobjects.yaml   # Autoscaling rules
    └── ...
```

### Installation

```bash
# Prerequisites
# 1. Kubernetes cluster with GPUs
# 2. NVIDIA device plugin installed
# 3. KEDA installed

# Install Cube3D
helm install cube3d ./helm/cube3d -n cube3d --create-namespace

# Scale for 5 requests/second
helm upgrade cube3d ./helm/cube3d -n cube3d \
  --set triton.gpt.replicaCount=50 \
  --set workers.gpt.replicaCount=50
```

---

## Monitoring and Operations

### Key Metrics to Watch

#### Triton Metrics (port 8002)

```bash
# Queue depth - are requests waiting?
curl http://triton-gpt:8002/metrics | grep nv_inference_pending_request_count

# Inference latency - how long is inference taking?
curl http://triton-gpt:8002/metrics | grep nv_inference_compute_infer_duration_us

# GPU utilization
curl http://triton-gpt:8002/metrics | grep nv_gpu_utilization
```

| Metric | Healthy | Action if Unhealthy |
|--------|---------|---------------------|
| `nv_inference_pending_request_count` | 0-2 | Add more Triton replicas |
| `nv_inference_queue_duration_us` | <100ms | Add more Triton replicas |
| `nv_gpu_utilization` | >80% | GPU is well-utilized |

#### Temporal Metrics

```bash
# Check queue backlog
temporal task-queue describe --task-queue cube3d-gpt

# Output:
# pollers: 10            ← Number of workers polling
# backlogCountHint: 5    ← Pending activities
```

| Metric | Healthy | Action if Unhealthy |
|--------|---------|---------------------|
| `backlogCountHint` | 0-5 | Add more workers |
| `pollers` | ≥ Triton replicas | Add more workers |

### Debugging Common Issues

#### 1. High latency but GPUs idle

**Symptom**: Requests take forever, but GPU utilization is low.

**Cause**: Not enough workers to keep GPUs busy.

**Fix**: Scale workers to match Triton replicas.

```bash
kubectl scale deployment worker-gpt --replicas=20
```

#### 2. Queue backlog growing

**Symptom**: `backlogCountHint` keeps increasing.

**Cause**: Incoming requests > processing capacity.

**Fix**: Scale both workers and Triton.

```bash
kubectl scale deployment triton-gpt --replicas=20
kubectl scale deployment worker-gpt --replicas=20
```

#### 3. Workers can't connect to Triton

**Symptom**: Workers log connection errors.

**Check**:
```bash
# Is Triton healthy?
kubectl exec -it deploy/triton-gpt -- curl localhost:8000/v2/health/ready

# Can workers reach Triton?
kubectl exec -it deploy/worker-gpt -- curl triton-gpt:8000/v2/health/ready
```

#### 4. Pods stuck in Pending

**Symptom**: Pods won't start.

**Cause**: Not enough GPUs available.

**Check**:
```bash
kubectl describe pod <pod-name>
# Look for: "Insufficient nvidia.com/gpu"

kubectl describe nodes | grep nvidia.com/gpu
# Check available GPUs
```

### Grafana Dashboard

Create a dashboard with these panels:

1. **Request Rate**: Temporal workflow starts per second
2. **Queue Depth per Stage**: Backlog for each task queue
3. **Inference Latency**: P50/P95/P99 from Triton metrics
4. **GPU Utilization**: Per Triton pod
5. **Replica Count**: Current replicas per deployment
6. **Error Rate**: Failed activities per minute

---

## Summary

### Architecture Layers

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| Ingress | Kubernetes Ingress | External traffic routing |
| API | FastAPI | Request validation, workflow start |
| Orchestration | Temporal | Workflow coordination, queuing, retries |
| Workers | Python + Temporal SDK | Bridge between Temporal and Triton |
| Inference | Triton | GPU scheduling, model execution |
| Autoscaling | KEDA | Queue-based replica scaling |

### Key Takeaways

1. **Separate concerns**: Temporal handles coordination, Triton handles inference
2. **Scale the bottleneck**: GPT needs 10x more replicas than CLIP
3. **Workers are cheap**: Scale them generously to keep GPUs busy
4. **Queue-based scaling**: KEDA + Temporal queues = automatic right-sizing
5. **Monitor queue depth**: It's the best signal for scaling decisions

### Quick Reference

```bash
# Deploy
helm install cube3d ./helm/cube3d -n cube3d --create-namespace

# Scale for N requests/second
GPT_REPLICAS=$((N * 10))
helm upgrade cube3d ./helm/cube3d -n cube3d \
  --set triton.gpt.replicaCount=$GPT_REPLICAS \
  --set workers.gpt.replicaCount=$GPT_REPLICAS

# Check status
kubectl get pods -n cube3d
kubectl get scaledobjects -n cube3d

# Monitor queues
temporal task-queue describe --task-queue cube3d-gpt

# View Triton metrics
kubectl port-forward svc/triton-gpt 8002:8002 -n cube3d
curl http://localhost:8002/metrics
```

---

## Further Reading

- [Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
- [Temporal Documentation](https://docs.temporal.io/)
- [KEDA Documentation](https://keda.sh/docs/)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
