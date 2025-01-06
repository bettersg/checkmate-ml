import logging
import json
from datetime import datetime
from context import request_id_var


import logging
import json


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
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)
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

    def error(self, message, **kwargs):
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
