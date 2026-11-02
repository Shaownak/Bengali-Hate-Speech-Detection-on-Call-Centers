"""Fetch datasets whose register entry has ``download_allowed: true`` (docs/09 §6.1).

Writes ``<raw>/<key>/<file>`` plus ``SOURCE.json`` (url, sha256, retrieval date). If the register
pins a sha256 for a file, a mismatch is an error (the upstream file changed).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ml.data.open.registry import Dataset, sha256_file


class ChecksumError(ValueError):
    pass


def download(
    ds: Dataset, raw_root: Path, transport: httpx.BaseTransport | None = None
) -> dict[str, str]:
    if not ds.download_allowed or not ds.download:
        raise PermissionError(f"{ds.id} {ds.key}: not downloadable; place files from {ds.url}")
    dest = raw_root / ds.key
    dest.mkdir(parents=True, exist_ok=True)
    got: dict[str, str] = {}
    with httpx.Client(transport=transport, timeout=120.0, follow_redirects=True) as http:
        for split, url in ds.download.items():
            name = ds.files[split]
            tmp = dest / f"{name}.part"
            with http.stream("GET", url) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            digest = sha256_file(tmp)
            want = ds.sha256.get(split)
            if want and want != digest:
                tmp.unlink()
                raise ChecksumError(f"{ds.key}/{name}: sha256 {digest[:12]}… ≠ pinned {want[:12]}…")
            tmp.replace(dest / name)
            got[split] = digest
    (dest / "SOURCE.json").write_text(
        json.dumps(
            {
                "dataset": ds.id,
                "key": ds.key,
                "version": ds.version,
                "urls": ds.download,
                "sha256": got,
                "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return got
