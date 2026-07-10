import pytest
import requests
import subprocess
import time
import os


@pytest.fixture(scope="module")
def api_server():
    """Start the actual API server for testing."""
    # Start the server in a subprocess
    proc = subprocess.Popen(
        ["python", "api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PORT": "9999"}
    )
    
    # Wait for server to start
    time.sleep(2)
    
    yield proc
    
    # Cleanup
    proc.terminate()
    proc.wait()


@pytest.mark.parametrize("payload", [
    # Missing token (exact exploit case)
    {"headers": {}, "command": "restart"},
    # Malformed token
    {"headers": {"Authorization": "Bearer invalid_token"}, "command": "kill"},
    # Expired token (simulated)
    {"headers": {"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MDAwMDAwMDB9.invalid"}, "command": "status"},
    # Valid input with no auth (boundary case)
    {"headers": {}, "command": None},
])
def test_unauthenticated_requests_rejected(api_server, payload):
    """Invariant: Protected endpoints reject unauthenticated requests with 401/403."""
    
    # Send request to actual endpoint
    url = "http://localhost:9999/control"
    
    if payload["command"] is not None:
        # POST request
        response = requests.post(
            url,
            json={"command": payload["command"]},
            headers=payload["headers"],
            timeout=5
        )
    else:
        # GET request
        response = requests.get(
            url,
            params={"command": ""},
            headers=payload["headers"],
            timeout=5
        )
    
    # Assert authentication failure
    assert response.status_code in [401, 403], \
        f"Expected 401/403 for unauthenticated request, got {response.status_code}"