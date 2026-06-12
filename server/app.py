"""
Pinnae dashboard server.

Routes:
  GET  /                           → static/index.html (dashboard)
  GET  /api/usv/events             → SQLite history query
  WS   /usv                        → live event stream via UDP fanout

Start with:
  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from server.broadcast import manager, start_udp_listener

log = logging.getLogger(__name__)

DB_PATH = "events.db"
STATIC_DIR = Path(__file__).parent.parent / "static"
HISTORY_LIMIT_MAX = 50_000


def _open_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp_abs)"
    )
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_udp_listener()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/usv/events")
async def get_events(
    since: float = Query(0.0, description="Epoch milliseconds lower bound (inclusive)"),
    max_ts: Optional[float] = Query(None, description="Epoch milliseconds upper bound (exclusive)"),
    limit: int = Query(2000, ge=1, le=HISTORY_LIMIT_MAX),
) -> JSONResponse:
    since_s = since / 1000.0
    try:
        db = _open_db()
    except sqlite3.OperationalError:
        return JSONResponse({"events": []})

    try:
        if max_ts is not None:
            max_ts_s = max_ts / 1000.0
            rows = db.execute(
                "SELECT timestamp_abs, band FROM events "
                "WHERE timestamp_abs >= ? AND timestamp_abs < ? "
                "ORDER BY timestamp_abs ASC LIMIT ?",
                (since_s, max_ts_s, limit),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT timestamp_abs, band FROM events "
                "WHERE timestamp_abs >= ? "
                "ORDER BY timestamp_abs ASC LIMIT ?",
                (since_s, limit),
            ).fetchall()
    finally:
        db.close()

    events = [{"ts": row["timestamp_abs"] * 1000, "type": row["band"]} for row in rows]
    return JSONResponse({"events": events})


@app.websocket("/usv")
async def usv_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "hello", "server_now_ms": time.time() * 1000})
    try:
        await manager.serve(ws)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("WS error: %s", exc)
    finally:
        if ws.client_state != WebSocketState.DISCONNECTED:
            await ws.close()


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
