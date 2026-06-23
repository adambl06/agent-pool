"""Compatibility entrypoint for uvicorn path agents.agent:app."""

from agents.api.app import app


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8091))
    uvicorn.run(app, host="0.0.0.0", port=port)
