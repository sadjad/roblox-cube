# Cube3D Helm Chart

Kubernetes Helm chart for deploying the Cube3D text-to-3D mesh generation pipeline with Triton Inference Server and Temporal workflow orchestration.

## Architecture

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                      Temporal Server                         │
                                    │  ┌──────────────┬──────────────┬──────────────┬──────────┐  │
                                    │  │  Workflow Q  │   CLIP Q     │    GPT Q     │  Mesh Q  │  │
                                    │  └──────┬───────┴──────┬───────┴──────┬───────┴────┬─────┘  │
                                    └─────────┼──────────────┼──────────────┼────────────┼────────┘
                                              │              │              │            │
                              ┌───────────────┘              │              │            │
                              ▼                              ▼              ▼            ▼
┌───────────┐         ┌─────────────┐              ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    API    │────────▶│  Workflow   │              │    CLIP     │  │     GPT     │  │    Mesh     │
│  Gateway  │         │   Workers   │              │   Workers   │  │   Workers   │  │   Workers   │
└───────────┘         └─────────────┘              └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
                                                          │                │                │
                                                          ▼                ▼                ▼
                                                   ┌────────────┐   ┌────────────┐   ┌────────────┐
                                                   │   Triton   │   │   Triton   │   │   Triton   │
                                                   │    CLIP    │   │    GPT     │   │    Mesh    │
                                                   │   (GPU)    │   │   (GPU)    │   │   (GPU)    │
                                                   └────────────┘   └────────────┘   └────────────┘
```

**Key Features:**
- Per-stage scaling with separate Triton deployments
- KEDA autoscaling based on Temporal queue depth
- Independent worker scaling per pipeline stage
- GPU resource management with proper tolerations

## Prerequisites

- Kubernetes 1.24+
- Helm 3.0+
- NVIDIA GPU Operator installed
- KEDA 2.10+ (for autoscaling)
- Temporal server (can be deployed as subchart)

## Installation

### 1. Add Dependencies

```bash
# Add Temporal Helm repo
helm repo add temporal https://go.temporal.io/helm-charts
helm repo update
```

### 2. Install KEDA (if not installed)

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace
```

### 3. Deploy Cube3D

```bash
# Install with default values
helm install cube3d ./cube3d -n cube3d --create-namespace

# Or with custom values
helm install cube3d ./cube3d -n cube3d --create-namespace \
  --set triton.gpt.replicaCount=10 \
  --set workers.gpt.replicaCount=10
```

### 4. Using External Temporal

If you have an existing Temporal cluster:

```bash
helm install cube3d ./cube3d -n cube3d --create-namespace \
  --set temporal.enabled=false \
  --set temporal.external.enabled=true \
  --set temporal.external.host=temporal.example.com \
  --set temporal.external.port=7233
```

## Configuration

### Scaling Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `triton.clip.replicaCount` | CLIP Triton replicas | 1 |
| `triton.gpt.replicaCount` | GPT Triton replicas (BOTTLENECK) | 4 |
| `triton.mesh.replicaCount` | Mesh Triton replicas | 2 |
| `workers.workflow.replicaCount` | Workflow workers | 2 |
| `workers.clip.replicaCount` | CLIP activity workers | 1 |
| `workers.gpt.replicaCount` | GPT activity workers | 4 |
| `workers.mesh.replicaCount` | Mesh activity workers | 2 |

### Autoscaling (KEDA)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `keda.enabled` | Enable KEDA autoscaling | true |
| `triton.gpt.autoscaling.minReplicas` | Min GPT replicas | 4 |
| `triton.gpt.autoscaling.maxReplicas` | Max GPT replicas | 50 |
| `triton.gpt.autoscaling.kedaTriggers[0].metadata.targetBacklogCount` | Scale when queue > N | 2 |

### GPU Resources

| Parameter | Description | Default |
|-----------|-------------|---------|
| `triton.clip.resources.limits.nvidia.com/gpu` | GPUs for CLIP | 1 |
| `triton.gpt.resources.limits.nvidia.com/gpu` | GPUs for GPT | 1 |
| `triton.mesh.resources.limits.nvidia.com/gpu` | GPUs for Mesh | 1 |

## Scaling Guidelines

GPT is the bottleneck (~10s per request). For target throughput T requests/second:

```
GPT_REPLICAS = ceil(T / 0.1)     # 10 replicas per 1 req/s
MESH_REPLICAS = ceil(T / 0.33)   # 3 replicas per 1 req/s
CLIP_REPLICAS = ceil(T / 50)     # Usually 1 is enough
```

**Example: 5 requests/second:**

```bash
helm upgrade cube3d ./cube3d -n cube3d \
  --set triton.gpt.replicaCount=50 \
  --set triton.mesh.replicaCount=15 \
  --set workers.gpt.replicaCount=50 \
  --set workers.mesh.replicaCount=15
```

## Monitoring

### Check Deployment Status

```bash
kubectl get pods -n cube3d
kubectl get scaledobjects -n cube3d
kubectl get hpa -n cube3d
```

### View Triton Metrics

```bash
kubectl port-forward svc/cube3d-triton-gpt 8002:8002 -n cube3d
curl http://localhost:8002/metrics | grep nv_inference
```

### Check Temporal Queues

```bash
# If using temporal CLI
temporal task-queue describe --task-queue cube3d-gpt
```

## Upgrading

```bash
# Update values
helm upgrade cube3d ./cube3d -n cube3d -f custom-values.yaml

# Scale specific component
helm upgrade cube3d ./cube3d -n cube3d \
  --reuse-values \
  --set triton.gpt.replicaCount=20
```

## Uninstalling

```bash
helm uninstall cube3d -n cube3d
kubectl delete namespace cube3d
```

## Troubleshooting

### Pods stuck in Pending

Check for GPU availability:
```bash
kubectl describe node | grep nvidia.com/gpu
```

### Workers not connecting to Triton

Check Triton health:
```bash
kubectl exec -it deploy/cube3d-triton-gpt -n cube3d -- curl localhost:8000/v2/health/ready
```

### High queue depth but no scaling

Verify KEDA is working:
```bash
kubectl get scaledobjects -n cube3d
kubectl describe scaledobject cube3d-triton-gpt-scaler -n cube3d
```

## License

Apache 2.0
