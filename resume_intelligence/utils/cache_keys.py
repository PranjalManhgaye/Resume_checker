"""Cache key helpers for session-scoped result caching."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def make_cache_key(prefix: str, *parts: Any) -> str:
    """Build a stable cache key from structured parts."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.md5(payload.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"
