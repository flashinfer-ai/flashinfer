# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The registry entry of the KDA class (``flashinfer.batch_invariant.fingerprint("recurrent_kda")``)."""

from __future__ import annotations

from typing import Any, Dict

from .spec import CLASS


def _recurrent_kda_fingerprint(kda_class: str = CLASS) -> str:
    from .spec import spec_fingerprint

    if kda_class != CLASS:
        raise ValueError(
            f"no batch-invariant KDA class {kda_class!r}; this build has {CLASS!r}"
        )
    return spec_fingerprint()


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the KDA class to the mode's registry (called by ``flashinfer.batch_invariant`` at import): no fixed
    dimension (the head count and the batch are not dimensions of the per-head term); ``kda_class`` names the class."""
    classes["recurrent_kda"] = class_info(
        CLASS, (), _recurrent_kda_fingerprint, ("kda_class",)
    )
