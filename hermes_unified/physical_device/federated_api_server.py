"""
Federated Learning REST API Server

FastAPI-based server for federated learning that communicates with edge devices.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hermes Federated Learning API")


class ModelUpdate(BaseModel):
    """Model update submitted by a client."""
    device_id: str
    round: int
    update: List[float]
    stats: Optional[Dict[str, Any]] = None


class ServerConfig:
    """Server configuration."""
    def __init__(self):
        self.model_size = 100
        self.global_model = np.zeros(self.model_size)
        self.round = 0
        self.updates_buffer: List[Dict[str, Any]] = []
        self.client_info: Dict[str, Dict[str, Any]] = {}
        self.round_history: List[Dict[str, Any]] = []
        self.aggregation_method = "fedavg"

    def get_model_data(self) -> Dict[str, Any]:
        """Get model data for client."""
        return {
            'weights': self.global_model.tolist(),
            'round': self.round,
            'model_size': self.model_size,
            'timestamp': datetime.now().isoformat()
        }

    def receive_update(self, update: ModelUpdate) -> Dict[str, Any]:
        """Receive and buffer an update from a client."""
        self.updates_buffer.append({
            'device_id': update.device_id,
            'round': update.round,
            'update': np.array(update.update),
            'stats': update.stats or {},
            'timestamp': time.time()
        })

        self.client_info[update.device_id] = {
            'last_seen': time.time(),
            'round': update.round,
            'stats': update.stats or {}
        }

        return {'status': 'received', 'buffer_size': len(self.updates_buffer)}


config = ServerConfig()


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "Hermes Federated Learning API",
        "version": "1.0",
        "status": "running"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "round": config.round,
        "connected_clients": len(config.client_info),
        "updates_buffered": len(config.updates_buffer)
    }


@app.get("/get_model")
def get_model():
    """Get the current global model."""
    return config.get_model_data()


@app.post("/submit_update")
def submit_update(update: ModelUpdate):
    """Submit a model update from a client."""
    if update.round < config.round:
        raise HTTPException(
            status_code=400,
            detail=f"Stale update: client round {update.round} < server round {config.round}"
        )

    result = config.receive_update(update)
    logger.info(f"Received update from {update.device_id} for round {update.round}")

    return result


@app.get("/status")
def get_status():
    """Get server status and connected clients."""
    return {
        'round': config.round,
        'global_model_norm': float(np.linalg.norm(config.global_model)),
        'connected_clients': len(config.client_info),
        'updates_buffered': len(config.updates_buffer),
        'clients': [
            {
                'device_id': device_id,
                'last_seen': info['last_seen'],
                'round': info['round'],
                'stats': info['stats']
            }
            for device_id, info in config.client_info.items()
        ]
    }


@app.post("/aggregate")
def aggregate():
    """Aggregate buffered updates and update global model."""
    if not config.updates_buffer:
        return {'status': 'no_updates', 'message': 'No updates to aggregate'}

    round_updates = [u for u in config.updates_buffer if u['round'] == config.round]

    if not round_updates:
        return {
            'status': 'partial',
            'message': f'No updates for round {config.round}',
            'buffered_rounds': list(set(u['round'] for u in config.updates_buffer))
        }

    updates = [u['update'] for u in round_updates]

    if config.aggregation_method == "fedavg":
        avg_update = np.mean(updates, axis=0)
        config.global_model += avg_update
    elif config.aggregation_method == "median":
        stacked = np.stack(updates, axis=0)
        median_update = np.median(stacked, axis=0)
        config.global_model += median_update
    else:
        avg_update = np.mean(updates, axis=0)
        config.global_model += avg_update

    config.updates_buffer = [u for u in config.updates_buffer if u['round'] > config.round]

    config.round_history.append({
        'round': config.round,
        'num_updates': len(round_updates),
        'timestamp': time.time()
    })

    logger.info(f"Round {config.round}: Aggregated {len(round_updates)} updates")

    config.round += 1

    return {
        'status': 'aggregated',
        'round': config.round - 1,
        'num_updates': len(round_updates),
        'new_round': config.round
    }


@app.post("/reset")
def reset():
    """Reset the server state."""
    config.global_model = np.zeros(config.model_size)
    config.round = 0
    config.updates_buffer = []
    config.client_info = {}
    config.round_history = []

    return {'status': 'reset', 'message': 'Server state has been reset'}


@app.get("/history")
def get_history():
    """Get aggregation history."""
    return {
        'history': config.round_history,
        'total_rounds': len(config.round_history)
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    print("Starting Hermes Federated Learning API Server on http://0.0.0.0:8000")
    run_server()
