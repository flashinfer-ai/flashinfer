"""
Environment-derived FlashInfer runtime settings.

Values in this module are cached so they are detected once per Python process
context, matching the behavior expected for global runtime switches.
"""

from __future__ import annotations

import functools
import os

FLASHINFER_SPECIALIZED_KERNEL_DISABLE = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

_TRUE_ENV_VALUES = frozenset(("1", "true", "t", "yes", "y", "on"))


def _env_flag_is_true(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUE_ENV_VALUES


@functools.cache
def is_specialized_kernel_disabled() -> bool:
    return _env_flag_is_true(os.getenv(FLASHINFER_SPECIALIZED_KERNEL_DISABLE))


def reset_specialized_kernel_env_cache() -> None:
    is_specialized_kernel_disabled.cache_clear()
