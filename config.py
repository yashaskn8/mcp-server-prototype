# config.py
# Central configuration for the MCP server.
#
# Every setting can be overridden via environment variable using the
# KF_MCP_ prefix, e.g.:
#
#   export KF_MCP_MOCK_MODE=false
#   export KF_MCP_DEFAULT_NAMESPACE=research
#
# This mirrors how a production Kubeflow operator would inject cluster
# credentials and namespace settings via pod environment variables.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KF_MCP_", case_sensitive=False)

    # ── Application metadata ──────────────────────────────────────────
    app_name:    str = "Kubeflow MCP Server"
    app_version: str = "0.1.0"
    description: str = (
        "An MCP-style server exposing structured tools for LLM-driven "
        "interaction with Kubeflow TrainJob resources."
    )

    # ── Cluster settings ──────────────────────────────────────────────
    mock_mode:         bool = True          # Flip to False with a real cluster
    default_namespace: str  = "kubeflow"
    kubeconfig_path:   str  = "~/.kube/config"   # Ignored in mock mode

    # ── Feature flags ────────────────────────────────────────────────
    enable_lifecycle_actions: bool = True   # Allow suspend/resume mutations
    log_level:                str  = "INFO"
    max_log_lines:            int  = 500    # Cap on lines returned per request


# Module-level singleton — import `settings` everywhere.
settings = Settings()
