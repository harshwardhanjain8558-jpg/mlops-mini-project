import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Turn every log record into one line of JSON."""

    def format(self, record):
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # extra fields passed as logger.info("...", extra={"fields": {...}})
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def get_logger(name="mp289"):
    """Logger that writes JSON lines to stdout (never to a file)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # avoid duplicate lines via the root logger
    return logger
