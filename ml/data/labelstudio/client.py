"""Minimal Label Studio REST client (self-hosted, in-country; docs/02 §3.1, docs/05).

Token from ``$GALE_LS_TOKEN`` (legacy API token: ``Authorization: Token …``). The base URL must be
our own Label Studio host; call audio and transcripts never go to a hosted service.
"""

from __future__ import annotations

from typing import Any

import httpx


class LabelStudioClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ):
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            transport=transport,
            timeout=timeout,
            headers={"Authorization": f"Token {token}"},
        )

    def _ok(self, r: httpx.Response) -> Any:
        r.raise_for_status()
        return r.json() if r.content else None

    def create_project(self, title: str, label_config: str, instruction: str = "") -> int:
        body = {
            "title": title,
            "label_config": label_config,
            "expert_instruction": instruction,
            "show_instruction": True,
            "maximum_annotations": 2,
        }  # 20 % double annotation is set per task
        return int(self._ok(self.http.post("/api/projects", json=body))["id"])

    def find_project(self, title: str) -> int | None:
        data = self._ok(self.http.get("/api/projects", params={"title": title, "page_size": 100}))
        items = data.get("results", data) if isinstance(data, dict) else data
        return next((int(p["id"]) for p in items if p.get("title") == title), None)

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]], batch: int = 500) -> int:
        n = 0
        for i in range(0, len(tasks), batch):
            self._ok(
                self.http.post(f"/api/projects/{project_id}/import", json=tasks[i : i + batch])
            )
            n += len(tasks[i : i + batch])
        return n

    def export(self, project_id: int) -> list[dict[str, Any]]:
        out = self._ok(
            self.http.get(f"/api/projects/{project_id}/export", params={"exportType": "JSON"})
        )
        return list(out or [])
