# main.py
# FastAPI application — HTTP routing, middleware, and exception handling.
#
# This file intentionally contains NO business logic.  Every route is a
# thin adapter: it validates the request, calls a service function, and
# formats the response using the shared APIResponse envelope.
#
# Adding a new MCP tool = add one entry to tools/mcp_manifest.py and one
# route here.  No other files need to change.

from __future__ import annotations

import uuid
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import services.k8s_service as svc
from config import settings
from models.schema import APIResponse, ErrorDetail, JobStatus
from tools.mcp_manifest import build_manifest

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("kubeflow_mcp")


# ──────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown hooks)
# ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: validate config and log server readiness.
    Shutdown: release any resources (e.g., close SDK connections).
    """
    mode = "MOCK" if settings.mock_mode else "LIVE CLUSTER"
    logger.info("──────────────────────────────────────────────")
    logger.info(f"  {settings.app_name}  v{settings.app_version}")
    logger.info(f"  Mode        : {mode}")
    logger.info(f"  Namespace   : {settings.default_namespace}")
    logger.info(f"  Lifecycle   : {'enabled' if settings.enable_lifecycle_actions else 'disabled'}")
    logger.info("──────────────────────────────────────────────")
    yield
    logger.info("MCP server shutting down.")


# ──────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# Middleware — request ID injection + latency logging
# ──────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Attaches a unique X-Request-ID to every response and logs method,
    path, status code, and latency.  Mirrors production observability
    practices used in Kubernetes operator deployments.
    """
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({elapsed_ms:.1f} ms) [req={request_id}]"
    )
    return response


# ──────────────────────────────────────────────────────────────────
# Exception handlers
# ──────────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=404,
        content=ErrorDetail(error="Resource not found.", code="NOT_FOUND", request_id=rid).model_dump(),
    )


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content=ErrorDetail(error=str(exc), code="VALIDATION_ERROR", request_id=rid).model_dump(),
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    rid = getattr(request.state, "request_id", None)
    logger.exception(f"Unhandled exception [req={rid}]", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(error="Internal server error.", code="INTERNAL_ERROR", request_id=rid).model_dump(),
    )


# ──────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────

def ok(request: Request, data: dict) -> JSONResponse:
    """Wrap a success payload in the MCP-style APIResponse envelope."""
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(
        content=APIResponse(success=True, data=data, request_id=rid).model_dump()
    )


def resource_not_found(request: Request, resource: str, identifier: str) -> HTTPException:
    """Raise a structured 404 with a machine-readable error code."""
    rid = getattr(request.state, "request_id", None)
    raise HTTPException(
        status_code=404,
        detail=ErrorDetail(
            error=f"{resource} '{identifier}' not found.",
            code=f"{resource.upper().replace(' ', '_')}_NOT_FOUND",
            request_id=rid,
        ).model_dump(),
    )


# ══════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════

# ── System endpoints ──────────────────────────────────────────────

@app.get("/", tags=["System"], summary="Health check")
async def root():
    """Server liveness probe — returns name, version, and mode."""
    return {
        "server":  settings.app_name,
        "version": settings.app_version,
        "mode":    "mock" if settings.mock_mode else "live",
        "docs":    "/docs",
        "tools":   "/tools",
    }


# ── MCP Tool Manifest ─────────────────────────────────────────────

@app.get("/tools", tags=["MCP Protocol"], summary="List available MCP tools")
async def list_tools(request: Request):
    """
    Returns the complete MCP tool manifest for this server.

    This is the first endpoint an LLM agent calls when connecting to an
    MCP server — it discovers available tools and their input schemas
    before invoking any of them.  Equivalent to the tools/list RPC in
    the MCP specification.
    """
    manifest = build_manifest()
    return ok(request, manifest.model_dump())


# ── Namespace endpoints ───────────────────────────────────────────

@app.get("/namespaces", tags=["MCP Tools"], summary="list_namespaces")
async def list_namespaces(request: Request):
    """
    MCP Tool: **list_namespaces**

    Lists all Kubernetes namespaces visible to the MCP server,
    with per-namespace job and active-job counts.
    """
    namespaces = await svc.list_namespaces()
    return ok(request, {
        "namespaces": [ns.model_dump() for ns in namespaces],
        "total": len(namespaces),
    })


# ── TrainJob endpoints ────────────────────────────────────────────

@app.get("/jobs", tags=["MCP Tools"], summary="list_training_jobs")
async def list_jobs(
    request: Request,
    namespace: Optional[str] = Query(default=None, description="Filter by namespace"),
    status:    Optional[JobStatus] = Query(default=None, description="Filter by job status"),
):
    """
    MCP Tool: **list_training_jobs**

    Returns a summary list of all TrainJobs. Supports optional filtering
    by namespace and status. An LLM agent calls this to orient itself
    before drilling into a specific job.
    """
    jobs = await svc.list_jobs(namespace=namespace, status_filter=status)
    return ok(request, {
        "jobs":  [j.model_dump() for j in jobs],
        "total": len(jobs),
        "filters": {"namespace": namespace, "status": status},
    })


@app.get("/jobs/{job_id}", tags=["MCP Tools"], summary="get_training_job")
async def get_job(job_id: str, request: Request):
    """
    MCP Tool: **get_training_job**

    Returns full details for a single TrainJob: configuration, pod
    topology, current metrics, and status message.
    """
    job = await svc.get_job(job_id)
    if job is None:
        resource_not_found(request, "Job", job_id)
    return ok(request, {"job": job.model_dump()})


@app.get("/jobs/{job_id}/logs", tags=["MCP Tools"], summary="get_job_logs")
async def get_logs(
    job_id: str,
    request: Request,
    tail_lines: Optional[int] = Query(
        default=None, ge=1, le=500,
        description="Maximum number of recent log lines to return",
    ),
):
    """
    MCP Tool: **get_job_logs**

    Returns structured log output from the primary pod of a TrainJob.
    Useful for LLM-assisted debugging without requiring kubectl access.
    """
    logs = await svc.get_logs(job_id, tail_lines=tail_lines)
    if logs is None:
        resource_not_found(request, "Job", job_id)
    return ok(request, {"logs": logs.model_dump()})


@app.get("/jobs/{job_id}/events", tags=["MCP Tools"], summary="get_job_events")
async def get_events(job_id: str, request: Request):
    """
    MCP Tool: **get_job_events**

    Returns Kubernetes Events for a TrainJob, ordered newest-first.
    Events surface scheduling failures, OOMKills, and pod restarts that
    may not appear in application logs — critical for LLM debugging.
    """
    events = await svc.get_events(job_id)
    if events is None:
        resource_not_found(request, "Job", job_id)
    return ok(request, {"events": events.model_dump()})


@app.get("/jobs/{job_id}/metrics", tags=["MCP Tools"], summary="get_job_metrics")
async def get_metrics(job_id: str, request: Request):
    """
    MCP Tool: **get_job_metrics**

    Returns the latest training metrics snapshot for a job.
    Allows an LLM to assess convergence without parsing raw log lines.
    """
    if not await svc.job_exists(job_id):
        resource_not_found(request, "Job", job_id)

    metrics = await svc.get_metrics(job_id)
    return ok(request, {
        "job_id":  job_id,
        "metrics": metrics.model_dump() if metrics else None,
        "available": metrics is not None,
    })


@app.get("/jobs/{job_id}/validate", tags=["MCP Tools"], summary="validate_job_exists")
async def validate_job(job_id: str, request: Request):
    """
    MCP Tool: **validate_job_exists**

    Lightweight pre-validation check.  Returns immediately without fetching
    a full job record — use this before any expensive operation.
    """
    exists = await svc.job_exists(job_id)
    return ok(request, {"job_id": job_id, "exists": exists})


# ── Lifecycle mutations ───────────────────────────────────────────

@app.post("/jobs/{job_id}/suspend", tags=["MCP Tools"], summary="suspend_training_job")
async def suspend_job(job_id: str, request: Request):
    """
    MCP Tool: **suspend_training_job**

    Safely suspends a Running or Pending TrainJob.  The Training Operator
    saves a checkpoint before terminating pods, so training can resume
    from the last saved step.

    This is a mutating action and is only available when
    `enable_lifecycle_actions=True` in server config.
    """
    if not settings.enable_lifecycle_actions:
        raise HTTPException(
            status_code=403,
            detail=ErrorDetail(
                error="Lifecycle actions are disabled on this server.",
                code="LIFECYCLE_ACTIONS_DISABLED",
            ).model_dump(),
        )

    if not await svc.job_exists(job_id):
        resource_not_found(request, "Job", job_id)

    updated_job = await svc.suspend_job(job_id)

    if updated_job is None:
        raise HTTPException(
            status_code=409,
            detail=ErrorDetail(
                error=f"Job '{job_id}' cannot be suspended in its current state.",
                code="INVALID_STATE_TRANSITION",
            ).model_dump(),
        )

    logger.info(f"Job '{job_id}' suspended via MCP tool call.")
    return ok(request, {"job": updated_job.model_dump(), "action": "suspended"})
