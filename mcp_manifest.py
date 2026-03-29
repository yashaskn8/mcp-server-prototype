# tools/mcp_manifest.py
# MCP Tool Manifest — the capability advertisement of this server.
#
# The Model Context Protocol specifies that every MCP server MUST expose
# its available tools as structured JSON Schema definitions.  When an LLM
# agent (Claude, GPT-4, Gemini) connects to an MCP server it first calls
# the tool-listing endpoint to discover what operations are available and
# what parameters each operation accepts — exactly like OpenAPI, but
# designed for machine-to-machine tool calling.
#
# Reference: https://spec.modelcontextprotocol.io/specification/server/tools/
#
# Each MCPTool entry here has a direct 1-to-1 mapping to an endpoint in
# main.py.  This file is the authoritative source of truth for what an
# LLM is allowed to invoke and with what arguments.

from config import settings
from models.schema import MCPTool, MCPToolInputSchema, MCPToolManifest


# ──────────────────────────────────────────────────────────────────
# Tool Definitions
# Each inputSchema follows the JSON Schema Draft-7 specification.
# ──────────────────────────────────────────────────────────────────

TOOLS: list[MCPTool] = [

    MCPTool(
        name="list_training_jobs",
        description=(
            "List all Kubeflow TrainJobs visible to the MCP server. "
            "Use this as the entry point when you need to discover what "
            "training runs exist before inspecting a specific job. "
            "Supports optional filtering by namespace and job status."
        ),
        inputSchema=MCPToolInputSchema(
            properties={
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace to query. Omit to list across all namespaces.",
                    "default": "kubeflow",
                },
                "status_filter": {
                    "type": "string",
                    "enum": ["Pending", "Running", "Succeeded", "Failed", "Suspended"],
                    "description": "Filter results to jobs in a specific lifecycle state.",
                },
            },
            required=[],
        ),
    ),

    MCPTool(
        name="get_training_job",
        description=(
            "Retrieve full details for a single TrainJob by its ID. "
            "Returns configuration (replicas, image, resources), "
            "current pod topology, the latest training metrics snapshot, "
            "and any status message set by the Training Operator. "
            "Call this after list_training_jobs to drill into a specific job."
        ),
        inputSchema=MCPToolInputSchema(
            properties={
                "job_id": {
                    "type": "string",
                    "description": "The unique job identifier (e.g. 'job-003').",
                },
            },
            required=["job_id"],
        ),
    ),

    MCPTool(
        name="get_job_logs",
        description=(
            "Fetch structured log output from the primary pod (master/chief) "
            "of a TrainJob. Use this to diagnose failures, track training "
            "progress, or surface error messages without needing kubectl access. "
            "Supports tail_lines to limit the number of lines returned."
        ),
        inputSchema=MCPToolInputSchema(
            properties={
                "job_id": {
                    "type": "string",
                    "description": "The unique job identifier.",
                },
                "tail_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Maximum number of recent log lines to return. Defaults to 500.",
                },
            },
            required=["job_id"],
        ),
    ),

    MCPTool(
        name="get_job_events",
        description=(
            "Return all Kubernetes Events recorded for a TrainJob, ordered "
            "newest-first. Events expose scheduling failures, OOMKill notices, "
            "pod restarts, and lifecycle transitions that do not always appear "
            "in logs. Especially useful for diagnosing Pending or Failed jobs."
        ),
        inputSchema=MCPToolInputSchema(
            properties={
                "job_id": {
                    "type": "string",
                    "description": "The unique job identifier.",
                },
            },
            required=["job_id"],
        ),
    ),

    MCPTool(
        name="get_job_metrics",
        description=(
            "Return the latest training metrics snapshot for a job "
            "(loss, accuracy, learning rate, GPU utilisation, throughput). "
            "Use to assess convergence or compare runs without reading raw logs."
        ),
        inputSchema=MCPToolInputSchema(
            properties={
                "job_id": {
                    "type": "string",
                    "description": "The unique job identifier.",
                },
            },
            required=["job_id"],
        ),
    ),

    MCPTool(
        name="validate_job_exists",
        description=(
            "Lightweight check that confirms whether a given job_id exists "
            "in the cluster. Use as a pre-validation step before attempting "
            "more expensive operations like log fetching or suspend actions."
        ),
        inputSchema=MCPToolInputSchema(
            properties={
                "job_id": {
                    "type": "string",
                    "description": "The unique job identifier to check.",
                },
            },
            required=["job_id"],
        ),
    ),

    MCPTool(
        name="suspend_training_job",
        description=(
            "Safely suspend a Running or Pending TrainJob. "
            "The Training Operator will checkpoint the job before terminating "
            "its pods, allowing the run to be resumed later without data loss. "
            "Only available when lifecycle actions are enabled on the server."
        ),
        inputSchema=MCPToolInputSchema(
            properties={
                "job_id": {
                    "type": "string",
                    "description": "The unique job identifier to suspend.",
                },
            },
            required=["job_id"],
        ),
    ),

    MCPTool(
        name="list_namespaces",
        description=(
            "List all Kubernetes namespaces visible to the MCP server, "
            "along with per-namespace job counts. Use this to scope subsequent "
            "list_training_jobs calls to the correct namespace."
        ),
        inputSchema=MCPToolInputSchema(
            properties={},
            required=[],
        ),
    ),
]


# ──────────────────────────────────────────────────────────────────
# Manifest builder
# ──────────────────────────────────────────────────────────────────

def build_manifest() -> MCPToolManifest:
    """Return the full MCP tool manifest for this server instance."""
    return MCPToolManifest(
        server_name=settings.app_name,
        server_version=settings.app_version,
        protocol="mcp/0.1",
        tools=TOOLS,
        total_tools=len(TOOLS),
    )
