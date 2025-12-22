"""
FastAPI Gateway for Cube3D Text-to-Mesh Service.

This API provides HTTP endpoints that start Temporal workflows
for 3D mesh generation.
"""

import asyncio
import os
import uuid
from datetime import timedelta
from typing import Optional, List, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from temporalio.client import Client, WorkflowExecutionStatus

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal.workflows import TextToMeshWorkflow, TextToMeshInput

# Configuration
TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "temporal:7233")
TASK_QUEUE = os.getenv("TASK_QUEUE", "cube3d-mesh-generation")
TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")

# FastAPI app
app = FastAPI(
    title="Cube3D Text-to-Mesh API",
    description="Generate 3D meshes from text prompts using the Cube3D model",
    version="0.1.0",
)

# Temporal client (initialized on startup)
temporal_client: Optional[Client] = None


# --- Request/Response Models ---

class GenerateMeshRequest(BaseModel):
    """Request to generate a 3D mesh from text."""
    prompt: str = Field(..., description="Text description of the 3D object")
    guidance_scale: float = Field(3.0, ge=0.0, le=20.0, description="Classifier-free guidance scale")
    resolution_base: float = Field(8.0, ge=6.0, le=9.0, description="Grid resolution as power of 2 (6=64, 7=128, 8=256, 9=512)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling threshold (None for deterministic)")
    bounding_box: Optional[Tuple[float, float, float]] = Field(None, description="Bounding box dimensions (x, y, z)")
    wait: bool = Field(True, description="Wait for completion (sync) or return immediately (async)")
    timeout_seconds: int = Field(600, ge=10, le=3600, description="Timeout in seconds (only for sync requests)")


class WorkflowStatus(BaseModel):
    """Status of a workflow execution."""
    workflow_id: str
    status: str
    prompt: str


class MeshResponse(BaseModel):
    """Response containing the generated mesh."""
    workflow_id: str
    num_vertices: int
    num_faces: int
    vertices: List[List[float]]
    faces: List[List[int]]
    token_ids: Optional[List[int]] = None


class AsyncGenerateResponse(BaseModel):
    """Response for async generation request."""
    workflow_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    temporal_connected: bool


# --- Startup/Shutdown ---

@app.on_event("startup")
async def startup():
    """Connect to Temporal on startup."""
    global temporal_client
    try:
        temporal_client = await Client.connect(
            TEMPORAL_HOST,
            namespace=TEMPORAL_NAMESPACE,
        )
        print(f"Connected to Temporal at {TEMPORAL_HOST}")
    except Exception as e:
        print(f"Failed to connect to Temporal: {e}")
        temporal_client = None


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global temporal_client
    if temporal_client:
        await temporal_client.close()


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy" if temporal_client else "degraded",
        temporal_connected=temporal_client is not None,
    )


@app.post("/generate", response_model=MeshResponse)
async def generate_mesh_sync(request: GenerateMeshRequest):
    """
    Generate a 3D mesh from a text prompt (synchronous).

    This endpoint starts a workflow and waits for completion.
    """
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal not connected")

    workflow_id = f"mesh-{uuid.uuid4().hex[:12]}"

    try:
        # Start workflow
        handle = await temporal_client.start_workflow(
            TextToMeshWorkflow.run,
            TextToMeshInput(
                prompt=request.prompt,
                guidance_scale=request.guidance_scale,
                resolution_base=request.resolution_base,
                top_p=request.top_p,
                bounding_box=request.bounding_box,
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=request.timeout_seconds),
        )

        if request.wait:
            # Wait for completion
            result = await handle.result()
            return MeshResponse(
                workflow_id=workflow_id,
                num_vertices=result.num_vertices,
                num_faces=result.num_faces,
                vertices=result.vertices,
                faces=result.faces,
                token_ids=result.token_ids,
            )
        else:
            return AsyncGenerateResponse(
                workflow_id=workflow_id,
                status="running",
                message="Workflow started. Use /status/{workflow_id} to check progress.",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/async", response_model=AsyncGenerateResponse)
async def generate_mesh_async(request: GenerateMeshRequest):
    """
    Start mesh generation asynchronously.

    Returns immediately with a workflow ID. Use /status/{workflow_id}
    to check progress and /result/{workflow_id} to get the result.
    """
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal not connected")

    workflow_id = f"mesh-{uuid.uuid4().hex[:12]}"

    try:
        await temporal_client.start_workflow(
            TextToMeshWorkflow.run,
            TextToMeshInput(
                prompt=request.prompt,
                guidance_scale=request.guidance_scale,
                resolution_base=request.resolution_base,
                top_p=request.top_p,
                bounding_box=request.bounding_box,
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=3600),
        )

        return AsyncGenerateResponse(
            workflow_id=workflow_id,
            status="running",
            message=f"Workflow started. Poll /status/{workflow_id} for progress.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{workflow_id}", response_model=WorkflowStatus)
async def get_workflow_status(workflow_id: str):
    """Get the status of a workflow."""
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal not connected")

    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        describe = await handle.describe()

        status_map = {
            WorkflowExecutionStatus.RUNNING: "running",
            WorkflowExecutionStatus.COMPLETED: "completed",
            WorkflowExecutionStatus.FAILED: "failed",
            WorkflowExecutionStatus.CANCELED: "canceled",
            WorkflowExecutionStatus.TERMINATED: "terminated",
            WorkflowExecutionStatus.TIMED_OUT: "timed_out",
        }

        return WorkflowStatus(
            workflow_id=workflow_id,
            status=status_map.get(describe.status, "unknown"),
            prompt="",  # Would need to extract from history
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {e}")


@app.get("/result/{workflow_id}", response_model=MeshResponse)
async def get_workflow_result(workflow_id: str):
    """Get the result of a completed workflow."""
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal not connected")

    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        result = await handle.result()

        return MeshResponse(
            workflow_id=workflow_id,
            num_vertices=result.num_vertices,
            num_faces=result.num_faces,
            vertices=result.vertices,
            faces=result.faces,
            token_ids=result.token_ids,
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Result not available: {e}")


@app.get("/result/{workflow_id}/obj")
async def get_workflow_result_as_obj(workflow_id: str):
    """Get the result as an OBJ file."""
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal not connected")

    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        result = await handle.result()

        # Build OBJ string
        lines = []
        for v in result.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for f in result.faces:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

        obj_content = "\n".join(lines)

        return Response(
            content=obj_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={workflow_id}.obj"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Result not available: {e}")


@app.delete("/workflow/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow."""
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal not connected")

    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        await handle.cancel()
        return {"status": "canceled", "workflow_id": workflow_id}

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to cancel: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
