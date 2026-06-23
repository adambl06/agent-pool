import logging
import os


def configure_logging() -> logging.Logger:
    """
    Configure process-wide logging and return a module logger.

    Called during module import in API bootstrap before request handling starts.
    """
    log_level = logging.DEBUG if os.environ.get("LOG_LEVEL") == "DEBUG" else logging.INFO
    log_file = os.environ.get("LOG_FILE")

    if log_file:
        logging.basicConfig(
            level=log_level,
            filename=log_file,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    return logging.getLogger(__name__)
