# Kubeflow MCP Server — Prototype

> **GSoC 2025 Prototype** · A minimal but production-structured MCP Server for the Kubeflow Training SDK, demonstrating LLM-driven interaction with distributed training workflows.

---

## Overview

This repository is a working prototype built to support a Google Summer of Code 2025 proposal for the Kubeflow project. The proposal centres on building a first-class **Model Context Protocol (MCP) Server** for the Kubeflow Training Operator SDK — enabling LLM agents (Claude, GPT-4, Gemini) to observe, query, and manage `TrainJob` resources in a Kubernetes cluster through structured, typed tool calls.

### The Problem

Kubeflow's Training Operator manages distributed ML training jobs (`PyTorchJob`, `TFJob`, `MPIJob`) on Kubernetes. Today, there is **no direct, structured interface** through which an LLM can:
- discover what training jobs are running in a cluster,
- read job status or logs without raw `kubectl` access,
- receive structured error context when a job fails,
- take lifecycle actions (suspend, resume) in response to observed state.

This forces ML engineers to switch contexts between their LLM assistant and their terminal, which breaks the debugging flow that conversational AI is best positioned to accelerate.

### The Solution

The MCP (Model Context Protocol) defines a standard wire protocol for exposing structured tools to LLM agents. A Kubeflow MCP Server would wrap the Training Operator SDK and expose each operation — list jobs, get status, fetch logs, suspend — as a first-class typed tool an LLM can invoke directly, parse deterministically, and act upon.

This prototype implements that architecture in miniature. It is not a toy demo — it is designed with the exact separation of concerns, schema discipline, and protocol fidelity that a production implementation would require.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LLM Agent (e.g. Claude)                    │
│                                                                     │
│  1. GET /tools         ← Discovers available tool schemas            │
│  2. GET /jobs          ← list_training_jobs(namespace="kubeflow")   │
│  3. GET /jobs/job-003  ← get_training_job(job_id="job-003")         │
│  4. GET /jobs/job-003/events ← get_job_events(job_id="job-003")     │
│  5. POST /jobs/job-003/suspend ← suspend_training_job(...)          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  HTTP + JSON (MCP-style envelope)
┌─────────────────────────────▼───────────────────────────────────────┐
│                   Kubeflow MCP Server  (this repo)                  │
│                                                                     │
│  main.py          FastAPI routes + middleware + exception handlers  │
│  tools/           MCP tool manifest — JSON Schema per tool          │
│  services/        Async service layer (mock ↔ real SDK toggle)      │
│  models/          Pydantic v2 schemas (API + MCP output types)      │
│  config.py        Pydantic Settings — env-var driven configuration  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  SDK calls (when MOCK_MODE=False)
┌─────────────────────────────▼───────────────────────────────────────┐
│           Kubeflow Training Operator SDK                            │
│                                                                     │
│  TrainingClient.list_jobs()     TrainingClient.get_job_logs()       │
│  TrainingClient.get_job()       kubernetes.CoreV1Api (events)       │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Decisions

**Why FastAPI?** The Training Operator SDK's forthcoming gRPC interface is the long-term target for a production MCP server, but HTTP/JSON is the correct choice for a prototype — it is directly inspectable, compatible with every LLM agent framework, and maps cleanly to the REST-over-HTTP profile of the MCP specification.

**Why an explicit `/tools` manifest endpoint?** The MCP specification requires every server to expose its tool definitions as JSON Schema objects so an LLM can discover the contract before invoking anything. Building this from the start — rather than bolting it on — keeps the prototype honest about what MCP actually is.

**Why async throughout?** The service layer uses `async def` functions so the server remains non-blocking when waiting on the Kubernetes API. This is not premature optimisation — it is the baseline requirement for a server that may fan out calls across multiple namespaces or stream log lines.

**Why is mock data isolated?** `services/mock_data.py` is completely isolated from `services/k8s_service.py`'s function signatures. Switching to a real cluster is a one-line change (`KF_MCP_MOCK_MODE=false`) plus filling in four `raise NotImplementedError` stubs. Nothing in `main.py` or the models changes.

---

## Project Structure

```
kubeflow-mcp-prototype/
│
├── main.py                    # FastAPI app — routes, middleware, exception handlers
├── config.py                  # Pydantic Settings — env-var driven configuration
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   └── schema.py              # All Pydantic v2 models (jobs, logs, events, MCP types)
│
├── services/
│   ├── __init__.py
│   ├── mock_data.py           # Realistic mock TrainJobs, logs, and K8s events
│   └── k8s_service.py        # Async service layer — mock today, real SDK tomorrow
│
├── tools/
│   ├── __init__.py
│   └── mcp_manifest.py        # MCP tool definitions (JSON Schema per tool)
│
└── tests/
    ├── __init__.py
    └── test_api.py            # Comprehensive pytest test suite (60+ assertions)
```

---

## Available MCP Tools

The server exposes **8 tools** via the `/tools` manifest endpoint. Each maps directly to one HTTP route.

| Tool Name | HTTP Route | Description |
|---|---|---|
| `list_training_jobs` | `GET /jobs` | Lists all TrainJobs with optional namespace/status filtering |
| `get_training_job` | `GET /jobs/{id}` | Full job record: config, pod topology, metrics, status message |
| `get_job_logs` | `GET /jobs/{id}/logs` | Structured log lines from the master pod; supports `tail_lines` |
| `get_job_events` | `GET /jobs/{id}/events` | Kubernetes events (scheduling failures, OOMKills, restarts) |
| `get_job_metrics` | `GET /jobs/{id}/metrics` | Training metrics: loss, accuracy, GPU utilisation, throughput |
| `validate_job_exists` | `GET /jobs/{id}/validate` | Lightweight pre-validation check (no full record fetch) |
| `suspend_training_job` | `POST /jobs/{id}/suspend` | Safely suspends a running job (checkpoint → terminate pods) |
| `list_namespaces` | `GET /namespaces` | Lists namespaces visible to the server with job counts |

---

## Mock Data

The prototype ships with **6 realistic TrainJobs** spanning all lifecycle states:

| Job ID | Name | Framework | Status | Notable Feature |
|---|---|---|---|---|
| `job-001` | `resnet50-imagenet` | PyTorch | Running | 4 workers, worker-2 had a transient restart |
| `job-002` | `bert-sst2-finetune` | TensorFlow | Succeeded | 3-epoch fine-tune, val_accuracy=0.935 |
| `job-003` | `gpt2-124m-pretrain` | PyTorch | **Failed** | OOMKill on worker-3, realistic error logs and K8s events |
| `job-004` | `stable-diffusion-lora` | PyTorch | Pending | GPU quota exhausted, `FailedScheduling` events |
| `job-005` | `llama3-8b-sft` | PyTorch | Suspended | 16 workers, suspended mid-run at step 52,000 |
| `job-006` | `vit-large-jax` | JAX | Running | JAX distributed on TPU nodes, JIT compile time in logs |

---

## Setup and Running

### Prerequisites

- Python 3.10 or higher
- pip
- VS Code (recommended) with the Python extension

### Step 1 — Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the server

```bash
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`. The `--reload` flag restarts automatically on file saves.

### Step 4 — Explore

- **Interactive API docs (Swagger UI):** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc
- **MCP Tool Manifest:** http://127.0.0.1:8000/tools

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=term-missing
```

The test suite covers all 8 tool endpoints, error cases (404, 409, 422), the MCP manifest structure, response envelope contract, and the X-Request-ID middleware header.

---

## Configuration

All settings can be overridden via environment variable using the `KF_MCP_` prefix:

| Variable | Default | Description |
|---|---|---|
| `KF_MCP_MOCK_MODE` | `true` | Set to `false` to use a real Kubernetes cluster |
| `KF_MCP_DEFAULT_NAMESPACE` | `kubeflow` | Default namespace for SDK queries |
| `KF_MCP_ENABLE_LIFECYCLE_ACTIONS` | `true` | Enables mutating tools (suspend) |
| `KF_MCP_LOG_LEVEL` | `INFO` | Logging verbosity |
| `KF_MCP_MAX_LOG_LINES` | `500` | Maximum log lines per request |

Example — disable lifecycle actions:
```bash
KF_MCP_ENABLE_LIFECYCLE_ACTIONS=false uvicorn main:app
```

---

## Example API Responses

### `GET /tools` — MCP Tool Manifest (excerpt)

```json
{
  "success": true,
  "data": {
    "server_name": "Kubeflow MCP Server",
    "protocol": "mcp/0.1",
    "total_tools": 8,
    "tools": [
      {
        "name": "get_job_logs",
        "description": "Fetch structured log output from the primary pod ...",
        "inputSchema": {
          "type": "object",
          "properties": {
            "job_id":     { "type": "string", "description": "The unique job identifier." },
            "tail_lines": { "type": "integer", "minimum": 1, "maximum": 500 }
          },
          "required": ["job_id"]
        }
      }
    ]
  }
}
```

### `GET /jobs?status=Failed`

```json
{
  "success": true,
  "request_id": "a3f1c8b2",
  "data": {
    "total": 1,
    "filters": { "namespace": null, "status": "Failed" },
    "jobs": [
      {
        "job_id": "job-003",
        "name": "gpt2-124m-pretrain",
        "namespace": "research",
        "status": "Failed",
        "framework": "PyTorch",
        "message": "Worker-3 OOMKilled (exit 137). Exceeded 32Gi memory limit."
      }
    ]
  }
}
```

### `GET /jobs/job-003/events`

```json
{
  "success": true,
  "data": {
    "events": {
      "job_id": "job-003",
      "total": 4,
      "events": [
        {
          "reason": "TrainJobFailed",
          "message": "TrainJob gpt2-124m-pretrain failed: worker-3 exited with code 1",
          "event_type": "Warning",
          "count": 1,
          "last_time": "2025-03-19T10:28:02Z"
        },
        {
          "reason": "OOMKilled",
          "message": "Container pytorch-worker in pod gpt2-124m-pretrain-pytorchjob-worker-3 OOMKilled",
          "event_type": "Warning",
          "count": 3,
          "last_time": "2025-03-19T10:28:01Z"
        }
      ]
    }
  }
}
```

---

## Connecting to a Real Kubeflow Cluster

When a real cluster is available, open `services/k8s_service.py` and replace the four `raise NotImplementedError` stubs with real SDK calls. The only change required elsewhere is setting the environment variable:

```bash
export KF_MCP_MOCK_MODE=false
pip install kubeflow-training kubernetes
uvicorn main:app --reload
```

The service layer is designed so that `main.py`, all schemas, and all tests continue to work without modification.

---

## Relation to the GSoC Proposal

This prototype validates the technical feasibility of the proposed GSoC project and demonstrates fluency with the relevant stack. The proposal targets the following deliverables over the summer:

1. A production-grade MCP Server integrated with the upstream `kubeflow-training` Python SDK
2. MCP tool definitions for `PyTorchJob`, `TFJob`, and the new `TrainJob` abstraction
3. Real-time log streaming via Server-Sent Events
4. Resume and cancel lifecycle operations
5. Prometheus metrics integration for GPU utilisation and throughput reporting
6. Upstream documentation and examples contributed to the Kubeflow repository

This prototype covers items 1, 2, and the skeleton of items 4 and 5 — in mock form but with production-grade structure.

---

## License

MIT — free to use, modify, and reference in your own proposals.
