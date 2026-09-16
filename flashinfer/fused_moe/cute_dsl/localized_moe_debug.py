# SPDX-License-Identifier: Apache-2.0
"""One-shot debug tracing for the MoE path localized across locality domains.

Everything the caller can observe about localization happens BEFORE the call
into FlashInfer -- resources created, weights sharded, kwargs assembled. Nothing
downstream of that boundary is visible, so "the kernel engaged" has to be
inferred rather than seen, and the grid sizing (which has no correctness
signature at all) cannot be checked from outside.

These traces sit at the three points that matter: the fan-out in
_moe_core_impl, and each dispatcher's offset/grid computation.

Opt-in via FLASHINFER_LOCALIZED_MOE_DEBUG=1, and each tag prints once per process, so
this stays quiet in production and does not multiply by 58 MoE layers.
"""

import os
import sys

_SEEN: set = set()


def localized_moe_debug_enabled() -> bool:
    return os.environ.get("FLASHINFER_LOCALIZED_MOE_DEBUG", "0") == "1"


def localized_moe_trace(tag: str, msg: str) -> None:
    """Print `msg` once per process for this `tag`. No-op unless enabled."""
    if not localized_moe_debug_enabled() or tag in _SEEN:
        return
    _SEEN.add(tag)
    print(f"[locality-domains-fi] {msg}", file=sys.stderr, flush=True)
