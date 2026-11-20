# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The registry entry of the GDN class (``flashinfer.batch_invariant.fingerprint("chunk_gated_delta_rule")``)."""

from __future__ import annotations

from typing import Any, Dict


def _chunk_gated_delta_rule_fingerprint() -> str:
    from .spec import spec_fingerprint

    return spec_fingerprint()


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the GDN class to the mode's registry (called by ``flashinfer.batch_invariant`` at import): one specification,
    no dimension (the head counts and the batch are not dimensions of the per-head term)."""
    classes["chunk_gated_delta_rule"] = class_info(
        "gdn.bf16.c64", (), _chunk_gated_delta_rule_fingerprint
    )
