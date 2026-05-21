"""
UDP datagram listener and WebSocket connection manager.

The detector process sends one UDP packet per event to 127.0.0.1:8765.
This module receives those packets and fans them out to every connected
WebSocket client via per-connection asyncio queues.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket

log = logging.getLogger(__name__)

UDP_HOST = "127.0.0.1"
UDP_PORT = 8765
_QUEUE_MAX = 1000


class ConnectionManager:
    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()

    def _add(self, q: asyncio.Queue) -> None:
        self._queues.add(q)

    def _remove(self, q: asyncio.Queue) -> None:
        self._queues.discard(q)

    async def broadcast(self, event: dict) -> None:
        for q in list(self._queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest, then enqueue latest so a slow client never stalls.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    async def serve(self, ws: WebSocket) -> None:
        """Feed events to a single WebSocket until it disconnects."""
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._add(q)
        try:
            while True:
                event = await q.get()
                await ws.send_json(event)
        finally:
            self._remove(q)


manager = ConnectionManager()


class _UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def connection_made(self, transport) -> None:
        self._loop = asyncio.get_event_loop()

    def datagram_received(self, data: bytes, addr) -> None:
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if self._loop:
            asyncio.ensure_future(manager.broadcast(event), loop=self._loop)

    def error_received(self, exc: Exception) -> None:
        log.warning("UDP error: %s", exc)


async def start_udp_listener() -> None:
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(
        _UDPProtocol,
        local_addr=(UDP_HOST, UDP_PORT),
    )
    log.info("UDP listener bound to %s:%d", UDP_HOST, UDP_PORT)
