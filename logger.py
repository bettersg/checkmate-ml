import logging
import json
import traceback
from datetime import datetime
from context import request_id_var
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
            "request_id": request_id_var.get(),
        }

        # Add exception info if available
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            formatted_tb = traceback.format_exception(
                exc_type, exc_value, exc_traceback
            )
            log_record["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": "".join(formatted_tb),  # Convert list to string
            }

        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)

        # Remove raw exc_info attribute to avoid double inclusion
        if hasattr(record, "exc_info"):
            record.exc_info = None

        return json.dumps(log_record)


class StructuredLogger(logging.Logger):
    def __init__(self, name="pino_logger", level=logging.INFO, context=None):
        super().__init__(name, level)

        self.context = context or {}

        # Add handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.addHandler(handler)

    def log(self, level, message, **kwargs):
        """
        Log a message at a specific level with contextual data.
        """
        # ensure kwargs are JSON serializable
        for key, value in kwargs.items():
            try:
                json.dumps(value)
            except TypeError:
                kwargs[key] = str(value)
        merged_context = {**self.context, **kwargs}
        super().log(level, message, extra={"extra_data": merged_context})

    def info(self, message, **kwargs):
        self.log(logging.INFO, message, **kwargs)

    def error(self, message, exc_info=None, **kwargs):
        """
        Log an error message, automatically including traceback information.
        """
        if exc_info is None and sys.exc_info()[0] is not None:
            exc_info = sys.exc_info()
        kwargs["exc_info"] = exc_info
        self.log(logging.ERROR, message, **kwargs)

    def debug(self, message, **kwargs):
        self.log(logging.DEBUG, message, **kwargs)

    def warn(self, message, **kwargs):
        self.log(logging.WARNING, message, **kwargs)

    def child(self, **new_context):
        """
        Returns a child logger with additional context.
        """
        # Create a child logger that inherits the context of the parent
        merged_context = {**self.context, **new_context}
        return StructuredLogger(
            name=self.name, level=self.level, context=merged_context
        )
