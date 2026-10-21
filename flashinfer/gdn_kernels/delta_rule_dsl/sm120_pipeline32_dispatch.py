# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in for the SM120 pipeline32 GDN prefill kernel."""

import os


def sm120_pipeline32_prefill_enabled() -> bool:
    """Read the FlashInfer opt-in; downstream environment aliases are ignored."""
    return os.environ.get("FLASHINFER_GDN_SM120_PIPELINE32", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
