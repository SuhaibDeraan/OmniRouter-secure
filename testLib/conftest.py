"""Shared pytest fixtures for the focused (mock-based) test suite."""
import pytest


@pytest.fixture(autouse=True)
def _reset_sse_starlette_appstatus():
    """Reset sse-starlette's module-level exit Event between tests.

    ``EventSourceResponse`` caches an ``anyio.Event`` on ``AppStatus`` bound to
    the first event loop that used it. Each Starlette ``TestClient`` runs on its
    own loop, so without this reset the second streaming response in a run
    raises "bound to a different event loop".
    """
    try:
        from sse_starlette.sse import AppStatus

        AppStatus.should_exit_event = None
    except Exception:
        pass
    yield
    try:
        from sse_starlette.sse import AppStatus

        AppStatus.should_exit_event = None
    except Exception:
        pass
