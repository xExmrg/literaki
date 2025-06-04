# Common utility functions shared across modules.
from __future__ import annotations

import os
from datetime import datetime


def ensure_dir(path: str) -> None:
    """Create directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current time formatted as a string."""
    return datetime.now().strftime(fmt)
