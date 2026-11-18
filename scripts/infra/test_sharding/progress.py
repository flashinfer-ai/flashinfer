from __future__ import annotations

import json
from typing import Any


PYTEST_EVENT_PREFIX = "@@flashinfer-pytest-event@@ "


def encode_pytest_event(event: str, **fields: Any) -> str:
    return PYTEST_EVENT_PREFIX + json.dumps(
        {"event": event, **fields}, sort_keys=True, separators=(",", ":")
    )


def decode_pytest_event(line: str) -> dict[str, Any] | None:
    position = line.find(PYTEST_EVENT_PREFIX)
    if position < 0:
        return None
    try:
        value = json.loads(line[position + len(PYTEST_EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
