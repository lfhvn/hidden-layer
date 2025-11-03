"""WebSocket endpoint for real-time updates during training."""

import logging
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: int):
        """Accept new WebSocket connection.

        Args:
            websocket: WebSocket connection
            experiment_id: Experiment ID to subscribe to
        """
        await websocket.accept()

        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = set()

        self.active_connections[experiment_id].add(websocket)

        logger.info(f"WebSocket connected for experiment {experiment_id}")

    def disconnect(self, websocket: WebSocket, experiment_id: int):
        """Remove WebSocket connection.

        Args:
            websocket: WebSocket connection
            experiment_id: Experiment ID
        """
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].discard(websocket)

            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]

        logger.info(f"WebSocket disconnected for experiment {experiment_id}")

    async def broadcast(self, experiment_id: int, message: dict):
        """Broadcast message to all connections for an experiment.

        Args:
            experiment_id: Experiment ID
            message: Message to broadcast
        """
        if experiment_id not in self.active_connections:
            return

        disconnected = set()

        for connection in self.active_connections[experiment_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection, experiment_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/experiments/{experiment_id}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: int):
    """WebSocket endpoint for experiment updates.

    Clients can connect to receive real-time updates during SAE training,
    including loss values, sparsity metrics, and progress updates.

    Args:
        websocket: WebSocket connection
        experiment_id: Experiment ID to subscribe to
    """
    await manager.connect(websocket, experiment_id)

    try:
        while True:
            # Keep connection alive and listen for ping/pong
            data = await websocket.receive_text()

            # Echo back for testing
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)
        logger.info(f"Client disconnected from experiment {experiment_id}")


async def send_training_update(
    experiment_id: int,
    epoch: int,
    total_epochs: int,
    metrics: dict,
):
    """Send training progress update to connected clients.

    Args:
        experiment_id: Experiment ID
        epoch: Current epoch
        total_epochs: Total number of epochs
        metrics: Training metrics
    """
    message = {
        "type": "training_update",
        "experiment_id": experiment_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "metrics": metrics,
        "progress": (epoch + 1) / total_epochs,
    }

    await manager.broadcast(experiment_id, message)


async def send_completion_update(experiment_id: int, status: str, final_metrics: dict):
    """Send training completion update.

    Args:
        experiment_id: Experiment ID
        status: Final status (completed/failed)
        final_metrics: Final training metrics
    """
    message = {
        "type": "training_complete",
        "experiment_id": experiment_id,
        "status": status,
        "metrics": final_metrics,
    }

    await manager.broadcast(experiment_id, message)
