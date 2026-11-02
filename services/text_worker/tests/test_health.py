from fastapi.testclient import TestClient

from gale_text_worker.app import build_app
from galecore.config import Settings


def test_health_ready_metrics() -> None:
    client = TestClient(build_app(Settings(env="test", bus_backend="memory")))
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["service"] == "text_worker"
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"bus": True}
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "# HELP" in metrics.text
