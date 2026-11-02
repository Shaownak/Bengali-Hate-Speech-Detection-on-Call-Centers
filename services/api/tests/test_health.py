from fastapi.testclient import TestClient

from gale_api.app import build_app
from galecore.config import Settings


def test_health_ready_metrics(db_url: str) -> None:
    s = Settings(env="test", bus_backend="memory", workers=False, db_url=db_url)
    with TestClient(build_app(s)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["service"] == "api"
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["checks"] == {"bus": True, "db": True}
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "# HELP" in metrics.text


def test_not_ready_without_db() -> None:
    s = Settings(
        env="test",
        bus_backend="memory",
        workers=False,
        db_url="postgresql+psycopg://nobody@127.0.0.1:1/none?connect_timeout=1",
    )
    with TestClient(build_app(s)) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["checks"]["db"] is False
