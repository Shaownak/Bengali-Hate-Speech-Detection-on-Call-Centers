"""Fetch datasets whose register entry has ``download_allowed: true`` (docs/09 §6.1).

Writes ``<raw>/<key>/<file>`` plus ``SOURCE.json`` (url, sha256, retrieval date). If the register
pins a sha256 for a file, a mismatch is an error (the upstream file changed).

Datasets published as a Mendeley Data folder tree (``mendeley: {id, version}``, e.g. BAAD's
per-term audio folders) are mirrored file by file through the public API, each file checked
against the sha256 the API reports; files already in place with that hash are skipped, so an
interrupted download resumes. Folder and file names are never printed or logged (BAAD's folders
are the abusive terms themselves); ``SOURCE.json`` records counts and a manifest hash only.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ml.data.open.registry import Dataset, sha256_file

MENDELEY_API = "https://data.mendeley.com/public-api/datasets"


class ChecksumError(ValueError):
    pass


def _unpack(archive: Path, dest: Path) -> None:
    """Unzip into ``dest``, refusing members that would land outside it (zip slip)."""
    root = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"{archive.name}: member escapes the dataset folder")
        z.extractall(dest)


def _safe_part(name: str) -> str:
    part = name.strip()
    if not part or part in (".", "..") or any(c in part for c in "/\\:"):
        raise ValueError("unsafe file or folder name in the dataset listing")
    return part


def mendeley_tree(ds: Dataset, dest: Path, http: httpx.Client) -> dict[str, str]:
    """Mirror a Mendeley dataset version's folder tree into ``dest``; returns counts + a hash of
    the (relative path, sha256) manifest."""
    did, ver = ds.mendeley["id"], ds.mendeley["version"]
    r = http.get(f"{MENDELEY_API}/{did}/folders/{ver}")
    r.raise_for_status()
    folders = {f["id"]: f for f in r.json()}

    def rel(fid: str) -> Path:
        parts: list[str] = []
        while fid in folders:
            parts.append(_safe_part(folders[fid]["name"]))
            fid = folders[fid].get("parent_id") or ""
        return Path(*reversed(parts)) if parts else Path()

    manifest: list[tuple[str, str]] = []
    fetched = skipped = 0
    for fid in ["root", *folders]:
        r = http.get(f"{MENDELEY_API}/{did}/files", params={"folder_id": fid, "version": ver})
        r.raise_for_status()
        for f in r.json():
            want = f["content_details"]["sha256_hash"]
            target = dest / (Path() if fid == "root" else rel(fid)) / _safe_part(f["filename"])
            if not target.resolve().is_relative_to(dest.resolve()):
                raise ValueError("listing escapes the dataset folder")
            manifest.append((target.relative_to(dest).as_posix(), want))
            if target.is_file() and sha256_file(target) == want:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            for attempt in range(3):  # thousands of small files: ride out dropped connections
                try:
                    with http.stream("GET", f["content_details"]["download_url"]) as resp:
                        resp.raise_for_status()
                        with tmp.open("wb") as out:
                            for chunk in resp.iter_bytes():
                                out.write(chunk)
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
            if sha256_file(tmp) != want:
                tmp.unlink()
                raise ChecksumError(f"{ds.key}: a file's sha256 differs from the API listing")
            tmp.replace(target)
            fetched += 1
    digest = hashlib.sha256(json.dumps(sorted(manifest)).encode()).hexdigest()
    pinned = ds.sha256.get("manifest")
    if pinned and pinned != digest:
        raise ChecksumError(f"{ds.key}: manifest {digest[:12]}… ≠ pinned {pinned[:12]}…")
    return {
        "files": str(len(manifest)),
        "fetched": str(fetched),
        "skipped": str(skipped),
        "manifest": digest,
    }


def download(
    ds: Dataset, raw_root: Path, transport: httpx.BaseTransport | None = None
) -> dict[str, str]:
    if ds.download_allowed and ds.mendeley:
        dest = raw_root / ds.key
        dest.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "gal-e-open-data/1"}  # Mendeley refuses some default agents
        with httpx.Client(
            transport=transport, timeout=120.0, follow_redirects=True, headers=headers
        ) as http:
            got = mendeley_tree(ds, dest, http)
        _source(ds, dest, {"mendeley": ds.mendeley}, got)
        return got
    if not ds.download_allowed or not ds.download:
        raise PermissionError(f"{ds.id} {ds.key}: not downloadable; place files from {ds.url}")
    dest = raw_root / ds.key
    dest.mkdir(parents=True, exist_ok=True)
    got: dict[str, str] = {}
    headers = {"User-Agent": "gal-e-open-data/1"}
    with httpx.Client(
        transport=transport, timeout=120.0, follow_redirects=True, headers=headers
    ) as http:
        for split, url in ds.download.items():
            name = ds.archives.get(split) or ds.files[split]
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
            if split in ds.archives:
                _unpack(dest / name, dest)
            got[split] = digest
    _source(ds, dest, {"urls": ds.download}, got)
    return got


def _source(ds: Dataset, dest: Path, origin: dict[str, object], got: dict[str, str]) -> None:
    (dest / "SOURCE.json").write_text(
        json.dumps(
            {
                "dataset": ds.id,
                "key": ds.key,
                "version": ds.version,
                **origin,
                "sha256": got,
                "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
