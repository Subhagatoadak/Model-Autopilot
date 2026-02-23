from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Map experiment_id -> set of connected WebSockets
_connections: dict[str, set[WebSocket]] = defaultdict(set)


async def broadcast(experiment_id: str, message: dict) -> None:
    """Broadcast a message to all WebSocket clients watching an experiment."""
    dead = set()
    for ws in _connections.get(experiment_id, set()):
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    _connections[experiment_id] -= dead


@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_ws(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for real-time experiment updates."""
    await websocket.accept()
    _connections[experiment_id].add(websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections[experiment_id].discard(websocket)
