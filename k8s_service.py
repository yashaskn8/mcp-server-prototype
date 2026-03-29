# services/k8s_service.py
# Service layer — the single place that touches cluster data.
#
# Architecture note:
#   All functions are async.  When MOCK_MODE=True (default) they resolve
#   immediately from in-memory dictionaries.  When MOCK_MODE=False each
#   function delegates to the Kubeflow Training Operator SDK.  No changes
#   are required in main.py or the models when switching modes.
#
# Real SDK equivalents are shown in comments using the upstream client:
#   pip install kubeflow-training
#   from kubeflow.training import TrainingClient
#
# Reference: https://github.com/kubeflow/training-operator/tree/master/sdk/python

from typing import List, Optional
from config import settings

from models.schema import (
    Framework, JobStatus, EventType,
    TrainJobSummary, TrainJobDetail,
    ResourceSpec, PodInfo, TrainingMetrics, JobEvent,
    JobLogs, JobEventsResponse, NamespaceSummary,
)

# Imported only in mock mode — not in the real-cluster path
if settings.mock_mode:
    from services.mock_data import JOBS, LOGS, EVENTS


# ══════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════

def _build_summary(job: dict) -> TrainJobSummary:
    """Extract the summary-level fields from a full job record."""
    return TrainJobSummary(
        job_id=job["job_id"],
        name=job["name"],
        namespace=job["namespace"],
        status=job["status"],
        framework=job["framework"],
        created_at=job["created_at"],
        message=job.get("message"),
    )


def _build_detail(job: dict) -> TrainJobDetail:
    """Build a full TrainJobDetail from a raw job record dict."""
    return TrainJobDetail(**{
        k: v for k, v in job.items()
        if k in TrainJobDetail.model_fields
    })


# ══════════════════════════════════════════════════════════════════
# Public service functions
# ══════════════════════════════════════════════════════════════════

async def list_jobs(
    namespace: Optional[str] = None,
    status_filter: Optional[JobStatus] = None,
) -> List[TrainJobSummary]:
    """
    Return a summary list of TrainJobs, with optional filtering.

    Real SDK equivalent:
        client = TrainingClient(namespace=namespace or settings.default_namespace)
        jobs = client.list_jobs(job_kind="PyTorchJob")
        # Repeat for TFJob, MPIJob, etc. — Training Operator supports multiple CRDs
        return [_build_summary(j) for j in jobs]
    """
    if settings.mock_mode:
        results = list(JOBS.values())

        if namespace:
            results = [j for j in results if j["namespace"] == namespace]

        if status_filter:
            results = [j for j in results if j["status"] == status_filter]

        return [_build_summary(j) for j in results]

    # ── Real cluster path (not executed in this prototype) ──────────
    raise NotImplementedError("Real cluster mode not implemented in this prototype.")


async def get_job(job_id: str) -> Optional[TrainJobDetail]:
    """
    Return full details for a single TrainJob, or None if it does not exist.

    Real SDK equivalent:
        client = TrainingClient()
        # job_kind is inferred from the framework field stored in annotations
        job = client.get_job(name=job_id, job_kind="PyTorchJob",
                             namespace=settings.default_namespace)
        return _build_detail(job) if job else None
    """
    if settings.mock_mode:
        raw = JOBS.get(job_id)
        return _build_detail(raw) if raw else None

    raise NotImplementedError("Real cluster mode not implemented in this prototype.")


async def get_logs(
    job_id: str,
    tail_lines: Optional[int] = None,
) -> Optional[JobLogs]:
    """
    Return structured logs for the primary pod of a TrainJob.

    Real SDK equivalent:
        client = TrainingClient()
        raw_logs = client.get_job_logs(
            name=job_id,
            job_kind="PyTorchJob",
            container="pytorch-master",
            follow=False,
        )
        lines = raw_logs.splitlines()
    """
    if settings.mock_mode:
        if job_id not in JOBS:
            return None

        all_lines = LOGS.get(job_id, ["[INFO] No log output captured yet."])

        cap = tail_lines or settings.max_log_lines
        truncated = len(all_lines) > cap
        lines = all_lines[-cap:]  # most-recent lines when capping

        job_name = JOBS[job_id]["name"]
        pod_name = f"{job_name}-pytorchjob-master-0"

        return JobLogs(
            job_id=job_id,
            pod=pod_name,
            lines=lines,
            total_lines=len(lines),
            truncated=truncated,
        )

    raise NotImplementedError("Real cluster mode not implemented in this prototype.")


async def get_events(job_id: str) -> Optional[JobEventsResponse]:
    """
    Return Kubernetes Events associated with a TrainJob.

    Real SDK equivalent:
        from kubernetes import client as k8s_client, config as k8s_config
        k8s_config.load_kube_config()
        v1 = k8s_client.CoreV1Api()
        events = v1.list_namespaced_event(
            namespace=settings.default_namespace,
            field_selector=f"involvedObject.name={job_id}",
        )
        return [_map_event(e) for e in events.items]
    """
    if settings.mock_mode:
        if job_id not in JOBS:
            return None

        raw_events = EVENTS.get(job_id, [])
        event_objects = [JobEvent(**e) for e in raw_events]
        # Return newest first (mirrors kubectl describe output)
        event_objects.sort(key=lambda e: e.last_time, reverse=True)

        return JobEventsResponse(
            job_id=job_id,
            events=event_objects,
            total=len(event_objects),
        )

    raise NotImplementedError("Real cluster mode not implemented in this prototype.")


async def get_metrics(job_id: str) -> Optional[TrainingMetrics]:
    """
    Return the latest training metrics snapshot for a job.

    In production metrics would be scraped from a Prometheus sidecar
    or read from TensorBoard event files stored in object storage.

    Real SDK equivalent:
        # Not yet in the Training Operator SDK — planned feature.
        # In practice: query Prometheus with PromQL targeting the job label.
    """
    if settings.mock_mode:
        raw = JOBS.get(job_id)
        if raw is None:
            return None
        return raw.get("metrics")  # May be None for Pending jobs

    raise NotImplementedError("Real cluster mode not implemented in this prototype.")


async def suspend_job(job_id: str) -> Optional[TrainJobDetail]:
    """
    Suspend a running TrainJob (safe checkpoint-and-pause).

    Real SDK equivalent:
        client = TrainingClient()
        client.update_job(
            name=job_id,
            job_kind="PyTorchJob",
            patch={"spec": {"suspend": True}},
        )
    """
    if settings.mock_mode:
        raw = JOBS.get(job_id)
        if raw is None:
            return None

        if raw["status"] not in (JobStatus.RUNNING, JobStatus.PENDING):
            return None   # Caller should surface a 409 Conflict

        # Mutate in-memory state (simulates the K8s API patch)
        JOBS[job_id]["status"]  = JobStatus.SUSPENDED
        JOBS[job_id]["message"] = "Job suspended via MCP tool call. Checkpoint saved."
        JOBS[job_id]["pods"]    = []   # Pods are deleted on suspend

        return _build_detail(JOBS[job_id])

    raise NotImplementedError("Real cluster mode not implemented in this prototype.")


async def list_namespaces() -> List[NamespaceSummary]:
    """
    Return namespaces visible to the MCP server with job-count statistics.

    Real SDK equivalent:
        from kubernetes import client as k8s_client, config as k8s_config
        k8s_config.load_kube_config()
        v1 = k8s_client.CoreV1Api()
        return [ns.metadata.name for ns in v1.list_namespace().items]
    """
    if settings.mock_mode:
        from collections import defaultdict
        ns_map: dict[str, dict] = defaultdict(lambda: {"total": 0, "active": 0})

        for job in JOBS.values():
            ns = job["namespace"]
            ns_map[ns]["total"] += 1
            if job["status"] in (JobStatus.RUNNING, JobStatus.PENDING):
                ns_map[ns]["active"] += 1

        return [
            NamespaceSummary(name=ns, job_count=v["total"], active_jobs=v["active"])
            for ns, v in sorted(ns_map.items())
        ]

    raise NotImplementedError("Real cluster mode not implemented in this prototype.")


async def job_exists(job_id: str) -> bool:
    """Lightweight existence check — avoids fetching a full job record."""
    if settings.mock_mode:
        return job_id in JOBS
    raise NotImplementedError("Real cluster mode not implemented in this prototype.")
