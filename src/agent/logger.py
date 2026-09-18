# agent/logger.py
import json
import logging
import logging.config
import logging.handlers
import os
from datetime import datetime
from typing import Optional

import yaml

_config_loaded = False


class SafeJSONFormatter(logging.Formatter):
    """Fallback JSON formatter that doesn't require python-json-logger."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        created = datetime.fromtimestamp(record.created)
        asctime = created.strftime(self.datefmt) if self.datefmt else created.isoformat()

        payload = {
            "asctime": asctime,
            "levelname": record.levelname,
            "name": record.name,
            "message": message,
            "correlation_id": getattr(record, "correlation_id", "no-cor-id"),
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        for k, v in record.__dict__.items():
            if k in payload or k.startswith("_"):
                continue
            if k in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "asctime",
            ):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except Exception:
                payload[k] = str(v)

        return json.dumps(payload, indent=2, ensure_ascii=False)


class CorrelationIdFallbackFilter(logging.Filter):
    def __init__(self, name: str = "", uuid_length: int = 10, default_value: str = "no-cor-id"):
        super().__init__(name)
        self.default_value = default_value
        self.uuid_length = uuid_length

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id") or not record.correlation_id:
            record.correlation_id = self.default_value
        else:
            try:
                record.correlation_id = str(record.correlation_id)[: self.uuid_length]
            except Exception:
                record.correlation_id = self.default_value
        return True


def _ensure_log_dir(config: dict) -> None:
    """Create directory for any RotatingFileHandler with a filename."""
    for _, handler_cfg in config.get("handlers", {}).items():
        if handler_cfg.get("class") == "logging.handlers.RotatingFileHandler":
            path = handler_cfg.get("filename")
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _load_config(explicit_path: Optional[str] = None) -> None:
    global _config_loaded
    if _config_loaded:
        return

    config_path = explicit_path or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "logging-config.yaml")
    )

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            _ensure_log_dir(config)

            logging.config.dictConfig(config)
            _config_loaded = True
            return

        except Exception as e:
            print(f"[LOGGER] Failed to load logging-config.yaml: {e}")

    # === Fallback logging ===
    os.makedirs("logs", exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename="logs/app.log",
        maxBytes=10_485_760,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        SafeJSONFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s %(funcName)s %(lineno)d"
        )
    )
    file_handler.addFilter(CorrelationIdFallbackFilter(uuid_length=10))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        SafeJSONFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s %(funcName)s %(lineno)d"
        )
    )
    console_handler.addFilter(CorrelationIdFallbackFilter(uuid_length=10))

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
    )
    _config_loaded = True


_RESERVED = {
    "name",
    "msg",
    "message",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "asctime",
}


class SanitizingAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        if isinstance(extra, dict):
            fixed = {}
            for k, v in extra.items():
                fixed_key = k if k not in _RESERVED else f"extra_{k}"
                fixed[fixed_key] = v
            kwargs["extra"] = fixed
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    _load_config()
    base = logging.getLogger(name)
    return SanitizingAdapter(base, {})
