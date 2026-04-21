from __future__ import annotations

from openenv.core.env_server import create_fastapi_app

from models import BioMedAction, BioMedObservation
from server.environment import BioMedEnvironment


def build_environment() -> BioMedEnvironment:
    """
    Construct the singleton environment instance used by the FastAPI app.

    Step 1/2 keeps server wiring intentionally minimal:
    one environment instance, one typed app, no extra runtime indirection.
    """
    return BioMedEnvironment()


env = build_environment()
app = create_fastapi_app(env, BioMedAction, BioMedObservation)

__all__ = ["app", "env", "build_environment"]
