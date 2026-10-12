# SPDX-License-Identifier: Apache-2.0
"""Env selector for SM120 pipeline32 GDN prefill overlay."""

from __future__ import annotations

import os


def _env_truthy(keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        if raw.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if raw.strip().lower() in ("0", "false", "no", "off"):
            return False
    return None


def sm120_pipeline32_prefill_enabled() -> bool:
    """Enable SM120 32-token 4-WG pipeline when env is truthy."""
    return (
        _env_truthy(
            ("FLASHINFER_GDN_SM120_PIPELINE32", "VLLM_GDN_FLASHINFER_PIPELINE32")
        )
        is True
    )
