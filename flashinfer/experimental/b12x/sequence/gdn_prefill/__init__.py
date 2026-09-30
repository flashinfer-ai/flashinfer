"""Scalar-gated GDN prefill over a paged recurrent-state pool.

The implemented API consumes projected/convolved BF16 Q/K/V and raw scalar
projections a and b. It writes BF16 recurrence output and FP32 final states
using the shared delta-rule CuTe kernels with a compile-time GDN recipe.
Projection, convolution, output RMSNorm, and output gating are external.

Q/K heads are 128-wide; each has three 128-wide value heads. Q/K/V may be
row-contiguous views of a packed projection. State uses [slot, value_head,
value_dim, key_dim] with a possibly padded slot stride, shared with GDN decode.
Pool-scaled offsets use 64-bit arithmetic.

Requests occupy cu_seqlens[r]:cu_seqlens[r+1]. Device scalars num_tokens and
num_seqs select live work within fixed planned capacity. Each request names
an initial and final state slot and optionally a checkpoint at a multiple of
16 tokens. A null initial slot means zero state, unlike the null-request
sentinel in gdn_decode; null destinations are not written. Requests spanning
pipeline windows require a non-null final slot for their running state.

Use ``plan(Caps(...), invocation=invocation_from_tensors(...))`` to declare
geometry and parameter dtypes. ``PreparationSession`` selects and primes the
native plan; ``bind`` requires the resulting prepared ``Plan``.
Capture or run only after preparation. Replay allocates no tensor storage.
Packed metadata is not checked on the device: the caller supplies in-range
slots, one writer per slot, and chunk-aligned checkpoint offsets. Padded
output rows are left untouched.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="gdn_prefill", group="sequence", api_style="planned",
    entry_points=("Binding", "Caps", "GdnPrefillConfig", "GdnPrefillQuery", "Plan",
                  "bind", "clear_caches", "is_supported", "plan", "invocation_from_tensors",
                  "reference", "run", "staging_memory"),
    dtypes=("bf16", "fp32", "int32", "int64"), recipes=("scalar_gdn",),
    provenance=Provenance(repo="https://github.com/lukealonso/b12x", commit="79e228ec6",
                          paths=("b12x/sequence/kda_prefill/_cute_kernels.py",
                                 "b12x/sequence/gdn_decode/_cute_kernels.py")),
    test_path="tests/experimental/b12x/preparation/test_delta_prefill.py", since="1.4.0",
    notes=("BF16 activations, FP32 state, "
           "head dim 128, three value heads per Q/K head, 16-token chunks."),
)

if TYPE_CHECKING:
    from .api import (Binding, Caps, GdnPrefillConfig, GdnPrefillQuery, Plan, bind,
                      clear_caches, is_supported, plan, invocation_from_tensors, reference,
                      run, staging_memory)

install_lazy_api(globals(), META)
