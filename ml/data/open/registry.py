"""Dataset register (``ml/data/datasets.yaml``) and the licence gate (docs/09 header).

Run tags:
- ``prod``: only datasets with ``commercial_ok: true``; anything else is refused.
- ``research``: every dataset may be used; outputs carry ``research_only = True`` and the model
  registry refuses to promote them (ADR-012: the owner runs the prototype on ``research``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REGISTER = Path(__file__).resolve().parents[1] / "datasets.yaml"
DATASETS_MD = Path(__file__).resolve().parents[1] / "DATASETS.md"
RunTag = Literal["prod", "research"]
CommercialOk = Literal["true", "false", "unknown"]


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    key: str
    name: str
    url: str
    licence: str
    commercial_ok: CommercialOk = "unknown"
    download_allowed: bool = False
    kind: Literal["text", "translit", "audio"]
    files: dict[str, str]
    columns: dict[str, str | list[str]] = Field(default_factory=dict)
    format: str = "VERIFY"
    version: str | None = None
    doi: str | None = None
    download: dict[str, str] = Field(default_factory=dict)
    sha256: dict[str, str] = Field(default_factory=dict)
    label_names: list[str] = Field(default_factory=list)
    eval_only: bool = False


class Register(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    datasets: list[Dataset]

    def get(self, key_or_id: str) -> Dataset:
        for d in self.datasets:
            if key_or_id in (d.key, d.id):
                return d
        raise KeyError(f"unknown dataset {key_or_id!r}")


class LicenceError(PermissionError):
    """A dataset may not be used in a run with this tag."""


def load_register(path: Path = REGISTER) -> Register:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    for d in raw.get("datasets", []):  # YAML reads true/false as bools; the register uses strings
        if isinstance(d.get("commercial_ok"), bool):
            d["commercial_ok"] = str(d["commercial_ok"]).lower()
    reg = Register.model_validate(raw)
    keys = [d.key for d in reg.datasets]
    if len(set(keys)) != len(keys):
        raise ValueError("datasets.yaml: duplicate keys")
    return reg


def check_allowed(ds: Dataset, run_tag: RunTag) -> bool:
    """Return ``research_only`` for rows from ``ds``; raise if the run may not use it at all."""
    if run_tag == "prod":
        if ds.commercial_ok != "true":
            raise LicenceError(
                f"{ds.id} {ds.key}: commercial_ok={ds.commercial_ok}; not allowed in prod runs"
            )
        if ds.eval_only:
            raise LicenceError(f"{ds.id} {ds.key} is evaluation-only")
        return False
    return ds.commercial_ok != "true"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def register_markdown(reg: Register) -> str:
    rows = []
    for d in reg.datasets:
        files = ", ".join(f"{k}: `{v}`" for k, v in d.files.items())
        src = f"[{d.doi or 'link'}]({d.url})" if d.url.startswith("http") else d.url
        rows.append(
            f"| {d.id} | `{d.key}` | {d.name} | {src} | {d.licence} | **{d.commercial_ok}** | "
            f"{'yes' if d.download_allowed else 'manual'} | {files} | {d.format} |"
        )
    return (
        "# Open-dataset register\n\n"
        "Generated from `ml/data/datasets.yaml` by `python -m ml.data.open register --write-md`; edit the YAML.\n\n"
        "- `commercial_ok`: `true` | `false` | `unknown`. **Runs tagged `prod` refuse anything not `true`.** "
        "`research` runs may use every dataset; their outputs are tagged `research_only` and blocked from "
        "model promotion (docs/09, ADR-012).\n"
        "- `download`: `yes` = fetched by `python -m ml.data.open download`; `manual` = place the files under "
        "`$GALE_DATA_DIR/data/raw/<key>/`.\n"
        "- `format: VERIFY` = column names come from the paper/README; check them when the files are first "
        "placed (the loader names the columns it found if they don't match).\n\n"
        "| id | key | dataset | source | licence | commercial_ok | download | files | format |\n"
        "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
