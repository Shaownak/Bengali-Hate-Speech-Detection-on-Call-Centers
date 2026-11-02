"""Docker-free local dev stack (ADR-009).

    python -m tools.devstack up [--web]   # start bus, db, all services (stub mode) [+ web]
    python -m tools.devstack status
    python -m tools.devstack down

Bus = fakeredis TCP server, DB = embedded Postgres (pgserver), object store = local folder,
auth = dev JWT mode. Everything lives under ``.gale/`` on the repo drive.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / ".gale" / "devstack"
LOG_DIR = ROOT / ".gale" / "logs"
STATE_FILE = STATE_DIR / "state.json"
DB_URI_FILE = STATE_DIR / "db_uri"
BUS_PORT = 6380
WEB_PORT = 3000

SERVICES: dict[str, int] = {
    "api": 8000,
    "ingest_gateway": 8001,
    "audio_worker": 8002,
    "asr_worker": 8003,
    "text_worker": 8004,
    "llm_judge": 8005,
    "fusion_engine": 8006,
}
IS_WINDOWS = os.name == "nt"


# -- infrastructure processes (run in their own detached process) ----------------


def serve_bus(port: int) -> None:
    from fakeredis import TcpFakeServer

    server = TcpFakeServer(("127.0.0.1", port), server_type="redis")
    server.serve_forever()


PGDATA = ROOT / ".gale" / "pgdata"


def _pg_ctl() -> Path:
    import pgserver

    exe = "pg_ctl.exe" if IS_WINDOWS else "pg_ctl"
    return Path(pgserver.__file__).parent / "pginstall" / "bin" / exe


def start_db() -> str:
    """Start embedded Postgres (pg_ctl daemonises it) and return the SQLAlchemy URL."""
    import pgserver

    srv = pgserver.get_server(str(PGDATA), cleanup_mode=None)  # keep running after we exit
    exists = srv.psql("SELECT 1 FROM pg_database WHERE datname = 'gale';")
    if "1 row" not in exists:
        srv.psql("CREATE DATABASE gale;")
    uri = srv.get_uri().rsplit("/", 1)[0] + "/gale"
    return uri.replace("postgresql://", "postgresql+psycopg://")


def stop_db() -> None:
    if (PGDATA / "postmaster.pid").exists():
        subprocess.run(
            [str(_pg_ctl()), "-D", str(PGDATA), "stop", "-m", "fast", "-w"],
            capture_output=True,
            check=False,
            timeout=60,
        )


# -- process management -------------------------------------------------------------


def _spawn(name: str, args: list[str], env: dict[str, str]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOG_DIR / f"{name}.log").open("ab")
    kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "cwd": ROOT, "env": env}
    if IS_WINDOWS:
        kwargs["creationflags"] = (
            # not DETACHED_PROCESS: console-less children crash Windows DLL init (0xC0000142)
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(args, **kwargs).pid


def _kill_tree(pid: int) -> None:
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _load_state() -> dict[str, int]:
    if not STATE_FILE.exists():
        return {}
    data: dict[str, int] = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return data


def _save_state(state: dict[str, int]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - localhost only
            return bool(r.status == 200)
    except OSError:
        return False


def _wait(pred: Any, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.3)
    raise SystemExit(f"devstack: timed out waiting for {what} (see {LOG_DIR})")


def _bus_up() -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", BUS_PORT)) == 0


# -- commands -------------------------------------------------------------------------


def up(web: bool) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in _load_state().items() if _alive(v)}
    py = sys.executable
    base_env = {**os.environ, "PYTHONUTF8": "1"}

    if "bus" not in state:
        state["bus"] = _spawn("bus", [py, "-m", "tools.devstack", "_bus"], base_env)
        _save_state(state)
    _wait(_bus_up, 20, "fakeredis bus")

    db_url = start_db()
    DB_URI_FILE.write_text(db_url, encoding="utf-8")

    env = {
        **base_env,
        "GALE_ENV": "dev",
        "GALE_MODEL_MODE": "stub",
        "GALE_AUTH_MODE": "dev",
        "GALE_BUS_BACKEND": "redis",
        "GALE_BUS_URL": f"redis://127.0.0.1:{BUS_PORT}/0",
        "GALE_DB_URL": db_url,
        "GALE_OBJECT_STORE": "local",
    }
    for name, port in SERVICES.items():
        if name not in state:
            state[name] = _spawn(name, [py, "-m", f"gale_{name}"], {**env, "GALE_PORT": str(port)})
            _save_state(state)
    if web and "web" not in state:
        if not (ROOT / "web" / "node_modules").exists():
            raise SystemExit("devstack: web/node_modules missing; run `make setup` first")
        pnpm = shutil.which("pnpm")
        if pnpm is None:
            raise SystemExit("devstack: pnpm not found on PATH")
        state["web"] = _spawn("web", [pnpm, "--dir", "web", "dev", "--port", str(WEB_PORT)], env)
    _save_state(state)

    for name, port in SERVICES.items():
        _wait(lambda p=port: _http_ok(f"http://127.0.0.1:{p}/healthz"), 60, name)
    status()


def status() -> None:
    state = _load_state()
    print(f"{'component':<16}{'pid':>8}  status")
    db = "running" if (PGDATA / "postmaster.pid").exists() else "stopped"
    url = DB_URI_FILE.read_text(encoding="utf-8") if db == "running" else ""
    print(f"{'db':<16}{'-':>8}  {db}  {url}")
    for name in ["bus", *SERVICES, "web"]:
        pid = state.get(name)
        if pid is None:
            print(f"{name:<16}{'-':>8}  stopped")
            continue
        if name in SERVICES:
            ok = _http_ok(f"http://127.0.0.1:{SERVICES[name]}/readyz")
            st = f"ready  http://127.0.0.1:{SERVICES[name]}" if ok else "not ready"
        elif name == "web":
            st = f"running  http://127.0.0.1:{WEB_PORT}" if _alive(pid) else "dead"
        else:
            st = "running" if _alive(pid) else "dead"
        print(f"{name:<16}{pid:>8}  {st}")


def down() -> None:
    state = _load_state()
    for name in ["web", *reversed(list(SERVICES))]:
        pid = state.pop(name, None)
        if pid is not None:
            _kill_tree(pid)
            print(f"stopped {name} ({pid})")
    stop_db()
    print("stopped db")
    pid = state.pop("bus", None)
    if pid is not None:
        _kill_tree(pid)
        print(f"stopped bus ({pid})")
    _save_state(state)
    DB_URI_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="devstack")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("up")
    p_up.add_argument("--web", action="store_true", help="also start the Next.js dev server")
    sub.add_parser("status")
    sub.add_parser("down")
    sub.add_parser("_bus")
    a = parser.parse_args()
    if a.cmd == "up":
        up(a.web)
    elif a.cmd == "status":
        status()
    elif a.cmd == "down":
        down()
    elif a.cmd == "_bus":
        serve_bus(BUS_PORT)


if __name__ == "__main__":
    main()
