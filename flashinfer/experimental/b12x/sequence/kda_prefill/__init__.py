"""Chunked lower-bounded KDA prefill over a paged recurrent-state pool.

The op consumes already-projected and convolved packed Q/K/V, the raw gate and
beta projections, and the per-head ``A_log`` and ``dt_bias`` parameters. It
computes the chunked gated delta rule with a lower-bounded gate, writes each
token's output, and advances a caller-owned recurrent-state pool in place.
Projection GEMMs, the causal convolution, and output gating are outside this
package.

The recurrent-state pool uses the ``gdn_decode`` physical layout
``[slot, head, value_dim, key_dim]`` in fp32, so a prefill and a decode of the
same request share one pool without conversion. State slots are addressed by
index rather than gathered: a request names its initial slot, its final slot,
and optionally one checkpoint slot with a chunk-aligned token offset, and the
op reads and writes those slots directly. ``Caps.null_state_index`` may reserve
one index meaning "zero initial state" and "do not write".

Requests are packed. Request ``r`` covers tokens
``cu_seqlens[r]:cu_seqlens[r + 1]``; ``num_seqs`` and ``num_tokens`` are device
scalars, so one plan serves every batch shape within its capacity. Tokens are
processed in sixteen-token chunks, ordered so that one pipeline window advances
every live request, which keeps the prepare and recurrence kernels overlapped.

Declare geometry and parameter dtypes with ``plan(Caps(...), invocation=...)``;
``invocation_from_tensors`` supplies the immutable dtype coordinates.
``PreparationSession`` selects and primes the plan before ``bind`` may
consume it. Runtime launches use caller-owned scratch, allocate no tensor
storage, and are capture safe. Packed metadata is not checked on the device:
the caller supplies in-range slots, one writer per slot, and chunk-aligned
checkpoint offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="kda_prefill",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Binding",
        "Caps",
        "KdaPrefillConfig",
        "KdaPrefillQuery",
        "Plan",
        "bind",
        "clear_caches",
        "is_supported",
        "plan",
        "invocation_from_tensors",
        "reference",
        "run",
    ),
    dtypes=("bf16", "fp32", "int32", "int64"),
    recipes=("lower_bounded_kda",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="f6a46f4cc",
        paths=("b12x/sequence/kda_prefill/_cute_kernels.py",),
    ),
    test_path="tests/experimental/b12x/sequence/test_kda_prefill.py",
    since="1.4.0",
    notes=(
        "Chunk size is sixteen tokens, head dim 128, bf16 activations, and "
        "fp32 recurrent state. Checkpoint offsets must be chunk aligned. The "
        "gate lower bound must lie in [-5, 0). Requests whose chunks span more "
        "than one pipeline window keep their running state in their final "
        "slot, so those requests require a non-null final slot."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        KdaPrefillConfig,
        KdaPrefillQuery,
        Plan,
        bind,
        clear_caches,
        is_supported,
        plan,
        invocation_from_tensors,
        reference,
        run,
    )

install_lazy_api(globals(), META)
