"""
Databricks Workspace Client.
"""

from databricks.sdk import WorkspaceClient


_workspace_client = None


def get_workspace_client() -> WorkspaceClient:
    """
    Returns a singleton WorkspaceClient.

    The Databricks SDK automatically authenticates using
    the credentials provided by the Databricks App.
    """

    global _workspace_client

    if _workspace_client is None:
        _workspace_client = WorkspaceClient()

    return _workspace_client