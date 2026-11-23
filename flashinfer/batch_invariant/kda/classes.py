# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The registry entry of the KDA class (``flashinfer.batch_invariant.fingerprint("recurrent_kda")``)."""

from __future__ import annotations

from typing import Any, Dict


def _recurrent_kda_fingerprint() -> str:
    from .spec import spec_fingerprint

    return spec_fingerprint()


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the KDA class to the mode's registry (called by ``flashinfer.batch_invariant`` at import): one specification,
    no dimension (the head count and the batch are not dimensions of the per-head term)."""
    classes["recurrent_kda"] = class_info(
        "kda.bf16.c16", (), _recurrent_kda_fingerprint
    )
