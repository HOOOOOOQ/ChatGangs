"""Allow ``python -m chatgangs`` to launch the API server."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ChatGangs API server")
    parser.add_argument("--host", default=os.getenv("CHATGANGS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHATGANGS_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="auto-reload for development")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WEB_CONCURRENCY", "1")))
    args = parser.parse_args()
    uvicorn.run(
        "chatgangs.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=os.getenv("CHATGANGS_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
