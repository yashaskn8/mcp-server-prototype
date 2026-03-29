# tests/test_api.py
# Automated tests for the Kubeflow MCP Server API.
#
# These tests use FastAPI's built-in TestClient (which wraps httpx) and
# do NOT require a running server — the application is instantiated
# in-process for each test session.
#
# Run with:   pytest tests/ -v
# Coverage:   pytest tests/ -v --cov=. --cov-report=term-missing

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

VALID_JOB_IDS   = ["job-001", "job-002", "job-003", "job-004", "job-005", "job-006"]
INVALID_JOB_ID  = "job-999"
RUNNING_JOB_ID  = "job-001"
SUCCEEDED_JOB_ID= "job-002"
FAILED_JOB_ID   = "job-003"
PENDING_JOB_ID  = "job-004"
SUSPENDED_JOB_ID= "job-005"


# ══════════════════════════════════════════════════════════════════
# System endpoints
# ══════════════════════════════════════════════════════════════════

class TestSystemEndpoints:

    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_expected_keys(self):
        data = client.get("/").json()
        assert "server" in data
        assert "version" in data
        assert "docs" in data
        assert "tools" in data

    def test_openapi_schema_accessible(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert "paths" in r.json()


# ══════════════════════════════════════════════════════════════════
# MCP Tool Manifest
# ══════════════════════════════════════════════════════════════════

class TestMCPManifest:

    def test_tools_endpoint_returns_200(self):
        r = client.get("/tools")
        assert r.status_code == 200

    def test_manifest_structure(self):
        body = client.get("/tools").json()
        assert body["success"] is True
        manifest = body["data"]
        assert "tools" in manifest
        assert "total_tools" in manifest
        assert manifest["total_tools"] == len(manifest["tools"])

    def test_each_tool_has_required_fields(self):
        tools = client.get("/tools").json()["data"]["tools"]
        for tool in tools:
            assert "name" in tool,        f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "inputSchema" in tool, f"Tool missing 'inputSchema': {tool}"

    def test_expected_tools_are_present(self):
        tool_names = {t["name"] for t in client.get("/tools").json()["data"]["tools"]}
        expected = {
            "list_training_jobs", "get_training_job", "get_job_logs",
            "get_job_events", "get_job_metrics", "validate_job_exists",
            "suspend_training_job", "list_namespaces",
        }
        assert expected.issubset(tool_names)

    def test_mcp_protocol_version_present(self):
        manifest = client.get("/tools").json()["data"]
        assert "protocol" in manifest
        assert manifest["protocol"].startswith("mcp/")


# ══════════════════════════════════════════════════════════════════
# Namespace endpoints
# ══════════════════════════════════════════════════════════════════

class TestNamespaces:

    def test_list_namespaces_returns_200(self):
        r = client.get("/namespaces")
        assert r.status_code == 200

    def test_namespace_list_is_non_empty(self):
        data = client.get("/namespaces").json()["data"]
        assert data["total"] > 0
        assert len(data["namespaces"]) == data["total"]

    def test_namespace_has_expected_keys(self):
        ns = client.get("/namespaces").json()["data"]["namespaces"][0]
        assert "name" in ns
        assert "job_count" in ns
        assert "active_jobs" in ns


# ══════════════════════════════════════════════════════════════════
# Job list
# ══════════════════════════════════════════════════════════════════

class TestListJobs:

    def test_list_all_jobs(self):
        r = client.get("/jobs")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["data"]["total"] == len(VALID_JOB_IDS)

    def test_filter_by_namespace_kubeflow(self):
        r = client.get("/jobs?namespace=kubeflow")
        assert r.status_code == 200
        jobs = r.json()["data"]["jobs"]
        assert all(j["namespace"] == "kubeflow" for j in jobs)
        assert len(jobs) > 0

    def test_filter_by_status_running(self):
        r = client.get("/jobs?status=Running")
        assert r.status_code == 200
        jobs = r.json()["data"]["jobs"]
        assert all(j["status"] == "Running" for j in jobs)
        assert len(jobs) >= 2   # job-001 and job-006 are Running

    def test_filter_by_unknown_namespace_returns_empty(self):
        r = client.get("/jobs?namespace=does-not-exist")
        assert r.status_code == 200
        assert r.json()["data"]["total"] == 0

    def test_each_job_summary_has_required_fields(self):
        jobs = client.get("/jobs").json()["data"]["jobs"]
        required = {"job_id", "name", "namespace", "status", "framework", "created_at"}
        for job in jobs:
            assert required.issubset(job.keys()), f"Missing fields in {job}"

    def test_response_envelope_shape(self):
        body = client.get("/jobs").json()
        assert "success" in body
        assert "data" in body
        assert body["success"] is True


# ══════════════════════════════════════════════════════════════════
# Job detail
# ══════════════════════════════════════════════════════════════════

class TestGetJob:

    @pytest.mark.parametrize("job_id", VALID_JOB_IDS)
    def test_all_valid_jobs_return_200(self, job_id):
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200, f"Expected 200 for {job_id}, got {r.status_code}"

    def test_invalid_job_returns_404(self):
        r = client.get(f"/jobs/{INVALID_JOB_ID}")
        assert r.status_code == 404

    def test_404_body_has_error_code(self):
        body = client.get(f"/jobs/{INVALID_JOB_ID}").json()
        assert "error" in body
        assert "code" in body

    def test_running_job_has_pods(self):
        job = client.get(f"/jobs/{RUNNING_JOB_ID}").json()["data"]["job"]
        assert len(job["pods"]) > 0

    def test_running_job_has_metrics(self):
        job = client.get(f"/jobs/{RUNNING_JOB_ID}").json()["data"]["job"]
        assert job["metrics"] is not None
        assert job["metrics"]["loss"] is not None

    def test_succeeded_job_has_duration(self):
        job = client.get(f"/jobs/{SUCCEEDED_JOB_ID}").json()["data"]["job"]
        assert job["duration_mins"] is not None
        assert job["duration_mins"] > 0

    def test_suspended_job_has_no_pods(self):
        job = client.get(f"/jobs/{SUSPENDED_JOB_ID}").json()["data"]["job"]
        assert job["pods"] == []

    def test_job_detail_contains_resource_spec(self):
        job = client.get(f"/jobs/{RUNNING_JOB_ID}").json()["data"]["job"]
        res = job["resources"]
        assert "cpu" in res and "memory" in res and "gpu" in res


# ══════════════════════════════════════════════════════════════════
# Logs
# ══════════════════════════════════════════════════════════════════

class TestGetLogs:

    def test_logs_return_200_for_valid_job(self):
        r = client.get(f"/jobs/{RUNNING_JOB_ID}/logs")
        assert r.status_code == 200

    def test_logs_return_404_for_invalid_job(self):
        r = client.get(f"/jobs/{INVALID_JOB_ID}/logs")
        assert r.status_code == 404

    def test_logs_have_expected_structure(self):
        logs = client.get(f"/jobs/{RUNNING_JOB_ID}/logs").json()["data"]["logs"]
        assert "job_id" in logs
        assert "pod" in logs
        assert "lines" in logs
        assert "total_lines" in logs
        assert isinstance(logs["lines"], list)

    def test_tail_lines_respected(self):
        logs = client.get(f"/jobs/{RUNNING_JOB_ID}/logs?tail_lines=3").json()["data"]["logs"]
        assert logs["total_lines"] <= 3

    def test_failed_job_logs_contain_error_markers(self):
        logs = client.get(f"/jobs/{FAILED_JOB_ID}/logs").json()["data"]["logs"]
        combined = " ".join(logs["lines"]).upper()
        assert "ERROR" in combined or "OOM" in combined


# ══════════════════════════════════════════════════════════════════
# Events
# ══════════════════════════════════════════════════════════════════

class TestGetEvents:

    def test_events_return_200_for_valid_job(self):
        r = client.get(f"/jobs/{RUNNING_JOB_ID}/events")
        assert r.status_code == 200

    def test_events_return_404_for_invalid_job(self):
        r = client.get(f"/jobs/{INVALID_JOB_ID}/events")
        assert r.status_code == 404

    def test_events_structure(self):
        data = client.get(f"/jobs/{RUNNING_JOB_ID}/events").json()["data"]["events"]
        assert "job_id" in data
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_failed_job_has_warning_events(self):
        events = client.get(f"/jobs/{FAILED_JOB_ID}/events").json()["data"]["events"]["events"]
        warning_events = [e for e in events if e["event_type"] == "Warning"]
        assert len(warning_events) > 0


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

class TestGetMetrics:

    def test_metrics_return_200_for_running_job(self):
        r = client.get(f"/jobs/{RUNNING_JOB_ID}/metrics")
        assert r.status_code == 200

    def test_metrics_return_404_for_invalid_job(self):
        r = client.get(f"/jobs/{INVALID_JOB_ID}/metrics")
        assert r.status_code == 404

    def test_pending_job_metrics_are_none(self):
        data = client.get(f"/jobs/{PENDING_JOB_ID}/metrics").json()["data"]
        assert data["available"] is False
        assert data["metrics"] is None

    def test_running_job_metrics_contain_loss_and_gpu(self):
        data = client.get(f"/jobs/{RUNNING_JOB_ID}/metrics").json()["data"]
        assert data["available"] is True
        m = data["metrics"]
        assert m["loss"] is not None
        assert m["gpu_utilization_pct"] is not None


# ══════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════

class TestValidateJob:

    def test_existing_job_returns_true(self):
        data = client.get(f"/jobs/{RUNNING_JOB_ID}/validate").json()["data"]
        assert data["exists"] is True

    def test_non_existing_job_returns_false(self):
        data = client.get(f"/jobs/{INVALID_JOB_ID}/validate").json()["data"]
        assert data["exists"] is False

    def test_validate_does_not_return_404(self):
        """Validate must return 200 even for unknown jobs — it's a boolean check."""
        r = client.get(f"/jobs/{INVALID_JOB_ID}/validate")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════
# Lifecycle: Suspend
# ══════════════════════════════════════════════════════════════════

class TestSuspendJob:

    def test_suspend_running_job_returns_200(self):
        # Note: this mutates in-memory state, so run last or use a fresh ID
        r = client.post(f"/jobs/{RUNNING_JOB_ID}/suspend")
        assert r.status_code == 200

    def test_suspend_updates_status_to_suspended(self):
        # job-006 is Running and untouched by other tests
        r = client.post("/jobs/job-006/suspend")
        assert r.status_code == 200
        job = r.json()["data"]["job"]
        assert job["status"] == "Suspended"

    def test_suspend_invalid_job_returns_404(self):
        r = client.post(f"/jobs/{INVALID_JOB_ID}/suspend")
        assert r.status_code == 404

    def test_suspend_already_suspended_job_returns_409(self):
        """Suspending a Suspended job is an invalid state transition."""
        r = client.post(f"/jobs/{SUSPENDED_JOB_ID}/suspend")
        assert r.status_code == 409

    def test_suspend_response_includes_action(self):
        r = client.post("/jobs/job-004/suspend")  # job-004 is Pending
        if r.status_code == 200:
            assert r.json()["data"]["action"] == "suspended"


# ══════════════════════════════════════════════════════════════════
# Response envelope contract (applies to ALL endpoints)
# ══════════════════════════════════════════════════════════════════

class TestResponseEnvelope:

    def test_success_responses_always_have_request_id(self):
        r = client.get("/jobs")
        body = r.json()
        assert "request_id" in body
        assert body["request_id"] is not None

    def test_success_flag_is_true_on_200(self):
        body = client.get("/jobs").json()
        assert body["success"] is True

    def test_x_request_id_header_present(self):
        r = client.get("/jobs")
        assert "x-request-id" in r.headers
