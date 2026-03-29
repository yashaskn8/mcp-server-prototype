# models/schema.py
# All Pydantic v2 data models for the Kubeflow MCP Server.
#
# Design philosophy:
#   Every model here doubles as both an API response schema AND an MCP
#   tool output schema.  Field descriptions are first-class citizens because
#   they surface directly in the /tools manifest — an LLM reads those
#   descriptions to understand how to call each tool correctly.
#
# In a full implementation these models would be generated from (or
# validated against) the upstream TrainJob CRD OpenAPI spec.

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field, model_validator
from pydantic.generics import GenericModel   # Pydantic v2 approach


# ══════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════

class JobStatus(str, Enum):
    """
    Lifecycle states taken directly from the Kubeflow Training Operator
    condition types. Maps 1-to-1 with TrainJob .status.conditions[].type.
    """
    PENDING   = "Pending"
    RUNNING   = "Running"
    SUCCEEDED = "Succeeded"
    FAILED    = "Failed"
    SUSPENDED = "Suspended"


class Framework(str, Enum):
    """Supported distributed training frameworks (subset of Training Operator)."""
    PYTORCH    = "PyTorch"
    TENSORFLOW = "TensorFlow"
    JAX        = "JAX"
    MPI        = "MPI"


class EventType(str, Enum):
    """Kubernetes event severity — mirrors core/v1 Event .type field."""
    NORMAL  = "Normal"
    WARNING = "Warning"


class PodPhase(str, Enum):
    """Kubernetes Pod phases."""
    PENDING   = "Pending"
    RUNNING   = "Running"
    SUCCEEDED = "Succeeded"
    FAILED    = "Failed"
    UNKNOWN   = "Unknown"


# ══════════════════════════════════════════════════════════════════
# Sub-models (composable building blocks)
# ══════════════════════════════════════════════════════════════════

class ResourceSpec(BaseModel):
    """Compute resources requested per replica — mirrors containers[].resources."""
    cpu:    Annotated[str, Field(description="CPU request (e.g. '4', '500m')")]
    memory: Annotated[str, Field(description="Memory request (e.g. '16Gi', '4Gi')")]
    gpu:    Annotated[int, Field(ge=0, description="Number of nvidia.com/gpu requested")]


class PodInfo(BaseModel):
    """
    Slim representation of a pod associated with a TrainJob.
    Full pod spec would come from the Kubernetes core/v1 Pod resource.
    """
    name:       Annotated[str,      Field(description="Pod name")]
    role:       Annotated[str,      Field(description="Training role: master | worker")]
    phase:      Annotated[PodPhase, Field(description="Current pod lifecycle phase")]
    node:       Annotated[Optional[str], Field(default=None, description="Node the pod is scheduled on")]
    restart_count: Annotated[int,   Field(ge=0, default=0, description="Container restart count")]
    start_time: Annotated[Optional[str], Field(default=None, description="ISO-8601 pod start time")]


class TrainingMetrics(BaseModel):
    """
    Latest training metrics scraped from pod stdout / Prometheus.
    In production these would be populated via the Kubeflow Training Operator
    SDK's metrics collection or a sidecar exporter.
    """
    step:          Annotated[int,           Field(ge=0,  description="Global training step")]
    epoch:         Annotated[Optional[int], Field(default=None, description="Current epoch")]
    total_epochs:  Annotated[Optional[int], Field(default=None, description="Total epochs configured")]
    loss:          Annotated[Optional[float], Field(default=None, description="Current training loss")]
    accuracy:      Annotated[Optional[float], Field(ge=0, le=1, default=None,
                                                    description="Current training accuracy [0-1]")]
    learning_rate: Annotated[Optional[float], Field(default=None, description="Current learning rate")]
    gpu_utilization_pct: Annotated[Optional[float], Field(ge=0, le=100, default=None,
                                                          description="Mean GPU utilisation across workers")]
    throughput_samples_per_sec: Annotated[Optional[float], Field(default=None,
                                                                  description="Training throughput")]
    recorded_at:   Annotated[str, Field(description="ISO-8601 timestamp of this metric snapshot")]


class JobEvent(BaseModel):
    """
    A Kubernetes Event linked to a TrainJob.
    Mirrors core/v1 Event — critical for LLM-assisted debugging workflows.
    """
    reason:    Annotated[str,       Field(description="Short CamelCase reason string")]
    message:   Annotated[str,       Field(description="Human-readable event message")]
    event_type:Annotated[EventType, Field(description="Normal or Warning")]
    count:     Annotated[int,       Field(ge=1, description="Number of times this event was seen")]
    first_time:Annotated[str,       Field(description="ISO-8601 first occurrence")]
    last_time: Annotated[str,       Field(description="ISO-8601 most recent occurrence")]


# ══════════════════════════════════════════════════════════════════
# Top-level TrainJob models
# ══════════════════════════════════════════════════════════════════

class TrainJobSummary(BaseModel):
    """
    Lightweight row returned by list_training_jobs.
    Keeps list responses fast — detail fields require a separate call.
    """
    job_id:     Annotated[str,       Field(description="Unique job identifier")]
    name:       Annotated[str,       Field(description="TrainJob .metadata.name")]
    namespace:  Annotated[str,       Field(description="Kubernetes namespace")]
    status:     Annotated[JobStatus, Field(description="Current lifecycle status")]
    framework:  Annotated[Framework, Field(description="Training framework")]
    created_at: Annotated[str,       Field(description="ISO-8601 creation timestamp")]
    message:    Annotated[Optional[str], Field(default=None,
                                               description="Latest status message")]


class TrainJobDetail(TrainJobSummary):
    """
    Full TrainJob record returned by get_training_job.
    Includes resource config, pod topology, metrics, and events.
    """
    # Runtime configuration
    num_workers:    Annotated[int, Field(ge=1,  description="Number of worker replicas")]
    num_masters:    Annotated[int, Field(ge=1, default=1, description="Number of master replicas")]
    resources:      Annotated[ResourceSpec, Field(description="Compute resources per replica")]
    image:          Annotated[str,           Field(description="Container image used by workers")]

    # Runtime state
    pods:           Annotated[List[PodInfo],   Field(default_factory=list,
                                                     description="Pods associated with this job")]
    metrics:        Annotated[Optional[TrainingMetrics], Field(default=None,
                                                              description="Latest training metrics")]
    duration_mins:  Annotated[Optional[int],   Field(default=None,
                                                     description="Elapsed duration in minutes; null if still running")]
    completed_at:   Annotated[Optional[str],   Field(default=None,
                                                     description="ISO-8601 completion timestamp; null if running")]

    @model_validator(mode="after")
    def completed_requires_duration(self) -> "TrainJobDetail":
        """Ensure completed jobs always carry a duration."""
        if self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            if self.duration_mins is None:
                raise ValueError(
                    f"Job with status {self.status} must have a duration_mins value."
                )
        return self


# ══════════════════════════════════════════════════════════════════
# Logs, Events, Metrics response wrappers
# ══════════════════════════════════════════════════════════════════

class JobLogs(BaseModel):
    """Structured log payload from a specific pod."""
    job_id:    Annotated[str,       Field(description="Parent job identifier")]
    pod:       Annotated[str,       Field(description="Pod whose logs are included")]
    lines:     Annotated[List[str], Field(description="Ordered log lines, oldest first")]
    total_lines: Annotated[int,     Field(ge=0,  description="Total lines returned")]
    truncated: Annotated[bool,      Field(default=False,
                                         description="True if log was capped by max_log_lines")]


class JobEventsResponse(BaseModel):
    """All Kubernetes events recorded for a TrainJob."""
    job_id:  Annotated[str,           Field(description="Parent job identifier")]
    events:  Annotated[List[JobEvent],Field(description="Events, newest first")]
    total:   Annotated[int,           Field(ge=0)]


class NamespaceSummary(BaseModel):
    """Kubernetes namespace visible to the MCP server."""
    name:      str
    job_count: int
    active_jobs: int   # Running | Pending


# ══════════════════════════════════════════════════════════════════
# MCP Tool Manifest models
# These precisely mirror the MCP protocol spec for tool definitions.
# See: https://spec.modelcontextprotocol.io/specification/server/tools/
# ══════════════════════════════════════════════════════════════════

class MCPToolInputSchema(BaseModel):
    """JSON Schema describing a tool's accepted input parameters."""
    type:       str = "object"
    properties: dict[str, Any]
    required:   List[str] = []


class MCPTool(BaseModel):
    """
    A single MCP tool definition.
    This is the exact structure an LLM agent (e.g. Claude, GPT-4) receives
    when discovering available tools on this server.
    """
    name:        Annotated[str,              Field(description="Unique tool name (snake_case)")]
    description: Annotated[str,              Field(description="What the tool does and when to use it")]
    inputSchema: Annotated[MCPToolInputSchema, Field(description="JSON Schema of accepted parameters")]


class MCPToolManifest(BaseModel):
    """Full tool manifest returned by GET /tools — the MCP server's capability advertisement."""
    server_name:    str
    server_version: str
    protocol:       str = "mcp/0.1"
    tools:          List[MCPTool]
    total_tools:    int


# ══════════════════════════════════════════════════════════════════
# Generic API response envelope
# ══════════════════════════════════════════════════════════════════

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """
    Uniform JSON envelope wrapping every endpoint response.
    Mirrors the structured-output contract required by MCP tool calls:
    every tool invocation returns { success, data } or { success, error }.
    """
    success:    bool
    data:       Optional[T]   = None
    error:      Optional[str] = None
    request_id: Optional[str] = None   # Injected by middleware for traceability


class ErrorDetail(BaseModel):
    """Structured error body for 4xx / 5xx responses."""
    success:    bool = False
    error:      str
    code:       str   # machine-readable error code (e.g. "JOB_NOT_FOUND")
    request_id: Optional[str] = None
