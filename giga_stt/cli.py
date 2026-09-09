"""Console entry point for the ``giga-stt`` command."""

import uvicorn

from .config import Settings


def main() -> None:
    """Start the HTTP service (settings are read from GIGAAM_* environment variables)."""
    settings = Settings.from_env()
    uvicorn.run(
        "giga_stt.app:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
