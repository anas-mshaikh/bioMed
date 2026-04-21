from __future__ import annotations

from openenv.core.env_server import create_fastapi_app
import uvicorn

from models import BioMedAction, BioMedObservation
from server.bioMed_environment import BioMedEnvironment


def build_environment() -> BioMedEnvironment:
    """
    Construct the singleton environment instance used by the FastAPI app.

    Step 1/2 keeps server wiring intentionally minimal:
    one environment instance, one typed app, no extra runtime indirection.
    """
    return BioMedEnvironment()


env = build_environment()
app = create_fastapi_app(env, BioMedAction, BioMedObservation)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()


__all__ = ["app", "env", "build_environment", "main"]
