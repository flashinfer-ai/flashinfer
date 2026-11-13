# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The numerical specification of batch-invariant MegaMoE.

The fingerprint identifies a compositional arithmetic specification. It is
independent of batch shape and scheduling, and is not a hash of CUDA code or a
certificate of memory-safety or hardware-instruction semantics.
"""

from .spec import (
    ALLOWED_PLANS,
    OrderedTermAlgebra,
    component_terms,
    fc1_activation,
    fc1_activation_accumulator,
    spec_fingerprint,
)

__all__ = [
    "ALLOWED_PLANS",
    "OrderedTermAlgebra",
    "component_terms",
    "fc1_activation",
    "fc1_activation_accumulator",
    "spec_fingerprint",
]
