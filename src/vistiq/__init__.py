try:
    from ._version import __version__  # type: ignore
except Exception:
    __version__ = "0.0.0"

# Automatically configure logging when package is imported
# This ensures logging works in Jupyter notebooks and interactive environments
import logging

# Only configure if root logger has no handlers (not already configured)
_root_logger = logging.getLogger()
if len(_root_logger.handlers) == 0:
    # Configure with INFO level by default
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# Export configure_logger for users who want to customize logging
from .app import configure_logger

__all__ = ["__version__", "configure_logger"]

