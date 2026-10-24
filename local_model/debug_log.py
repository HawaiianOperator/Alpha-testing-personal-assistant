"""Rotating performance log; excludes request text and response content."""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def create_debug_logger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger("assistant.performance")
    logger.propagate = False
    handler = RotatingFileHandler(path, maxBytes=1048576, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger

def record_performance(logger, result):
    keys = ("status", "command", "elapsed_ms", "elapsed_seconds", "process_ram_mib", "peak_process_ram_mib")
    record = {"timestamp_utc":datetime.now(timezone.utc).isoformat()}
    record.update({key:result[key] for key in keys if key in result})
    logger.info(json.dumps(record))
