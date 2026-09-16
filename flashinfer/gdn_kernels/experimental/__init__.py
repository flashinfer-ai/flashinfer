# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental GDN kernels and the fused decode step they serve.

"Experimental" describes where this code *lives*, not how it is called:
the public API is exported at the top level like the other GDN APIs, so
consumers write::

    import flashinfer

    if flashinfer.gdn_fused_decode_step_supported(batch, ..., conv_state_layout="SD"):
        flashinfer.gdn_fused_decode_step(hidden_states, w_ba, ..., out=core_attn_out)
    else:
        ...  # keep your own composition

:func:`~flashinfer.gdn_kernels.experimental.gdn_fused_decode.gdn_fused_decode_step`
fuses one decode step of a gated-delta-net linear-attention layer (b/a
projection GEMV, causal conv1d state update, q/k/v split, gated delta-rule
decode) for the traced layer geometries in
``gdn_fused_decode_registry.json``.  The interface -- this re-export, the
dispatch/registry module and the routing probe -- lives here; the kernels
that serve the registered geometries live in :mod:`.kernel` (see README.md).

**The op takes no backend argument and no environment gate.**  Whether a
call runs a specialized kernel or the composable torch path is decided by
the library from the workload registry and the device;
:func:`gdn_fused_decode_step_supported` reports that decision up front,
cheaply, for consumers that keep their own path for unsupported shapes.
The registry and the probe answer *support*; whether to use a supported op
at all is the calling framework's decision, taken with the framework's own
configuration surface -- this op is new, so there is no pre-existing
in-FlashInfer alternative for an environment variable here to fall back to.

Importing this package pulls in :mod:`torch` and nothing else: the
specialized dispatch module, the kernels, the JIT machinery and the
optional CuTe-DSL dependency are all imported lazily, on the first probe or
call.  ``flashinfer/__init__.py`` therefore re-exports the two public names
unconditionally, and a consumer's capability check (``getattr(flashinfer,
"gdn_fused_decode_step", None)``) costs nothing and never compiles.
"""

from .gdn_fused_decode import (
    gdn_fused_decode_step as gdn_fused_decode_step,
    gdn_fused_decode_step_supported as gdn_fused_decode_step_supported,
)

__all__ = [
    "gdn_fused_decode_step",
    "gdn_fused_decode_step_supported",
]
