"""Stateful packed Gated Delta Network decode.

The op consumes already-projected and convolved packed Q/K/V plus decay,
update, and output-gate projections. It updates a caller-owned recurrent-state
pool in place, then applies per-value-head RMSNorm and either a SiLU or sigmoid
output gate. Projection GEMMs and causal-convolution state are intentionally
outside this package.

``bind`` / ``run`` implements scalar per-head Qwen GDN decay. ``bind_kda`` /
``run_kda`` implements GLM/Kimi lower-bounded KDA decay from a per-key-coordinate
raw gate while preserving the same state and serving lifecycle. Bindings accept
live tensor capacities within the plan, so serving runtimes can bind
projection, metadata, and output tensors directly without staging.

With ``Caps.recover_speculative_state=True``, KDA verification instead reads
column zero, leaves its FP32 checkpoint unchanged, and writes caller-owned
``correction_cache`` (FP32) and ``kg_cache`` (BF16 original keys/raw gates).
The record shapes are ``[slot, head, window, 128]`` and
``[slot, head, window, 256]``; ``state_index_columns`` sets the maximum window.
``bind_kda_commit`` / ``run_kda_commit`` apply accepted records across a group
of layers and optionally export an aligned boundary checkpoint. This mode uses
CuTe for both recurrence and recovery. Gated output normalization is enabled by
default; ``run_kda(..., apply_output_norm=False)`` returns the unnormalized
recurrence output. Only recovery mode supports disabling output normalization.

The recurrent-state pool uses the optimized physical layout
``[slot, value_head, value_dim, key_dim]``. This is the transpose of the
``[batch, head, key_dim, value_dim]`` state used by slow mathematical PyTorch
references; importing such a state requires transposing its final two axes.
The three inner dimensions must be contiguous. The outer slot stride may be
larger than one logical state to accommodate an aligned paged cache; binding
preserves that stride and never compacts or copies the caller-owned pool.
Pool-scaled slot offsets are computed with 64-bit arithmetic.

Packed requests use fixed-capacity device metadata that the caller guarantees
is well formed: counts within the bound capacities, monotone ``query_start_loc``,
accepted-token counts within the column capacity, in-range state indices, and
unique active state cells. Request ``r`` consumes
``query_start_loc[r]:query_start_loc[r + 1]`` and reads its initial checkpoint
from state-index column ``num_accepted_tokens[r] - 1``. Tokens execute
sequentially per request and persist their post-token checkpoints to columns
starting at zero. A one-column plan with one token per request is ordinary
decode. ``Caps.null_state_index`` may reserve one index as a null checkpoint.
Requests whose selected initial checkpoint is null produce zero output without
reading or writing recurrent state; null destination cells are not written.
The default ``None`` leaves every in-range slot, including slot zero, usable.

``plan(Caps(...), invocation=...)`` declares immutable geometry and layouts.
``invocation_from_tensors`` describes actual parameter dtypes and buffer layouts.
``PreparationSession`` resolves and primes the declaration; ``bind`` /
``bind_kda`` require its ready prepared ``Plan``. Runtime launches consume
only stored launchers, allocate no tensor storage, and are opaque to
``torch.compile``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="gdn_decode",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Binding",
        "Caps",
        "GdnConfig",
        "GdnQuery",
        "KdaBinding",
        "KdaCommitBinding",
        "Plan",
        "bind",
        "bind_kda",
        "bind_kda_commit",
        "is_supported",
        "plan",
        "invocation_from_tensors",
        "reference",
        "run",
        "run_kda",
        "run_kda_commit",
    ),
    dtypes=("bf16", "fp32", "int32", "int64"),
    recipes=("silu", "sigmoid", "lower_bounded_kda"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="e8d02602f",
        paths=(
            "serve/kernels/fla/fused_recurrent.py",
            "serve/kernels/fla/fused_norm_gate.py",
        ),
    ),
    test_path="tests/experimental/b12x/sequence/test_gdn_decode.py",
    since="1.3.0",
    notes=(
        "Qwen3.8 Flash Next uses the CuTeDSL recurrence for any planned "
        "capacity with three value heads per Q/K head. BF16 and FP32 recurrent "
        "state and int32 or int64 state indices are supported. Triton is used "
        "only for the gated RMSNorm auxiliary. The separately named GLM/KDA "
        "API uses Triton for full-checkpoint recurrence and CuTe for "
        "speculative-state recovery, with equal 128-wide Q/K/V head counts."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        GdnConfig,
        GdnQuery,
        KdaBinding,
        KdaCommitBinding,
        Plan,
        bind,
        bind_kda,
        bind_kda_commit,
        is_supported,
        plan,
        invocation_from_tensors,
        reference,
        run,
        run_kda,
        run_kda_commit,
    )

install_lazy_api(globals(), META)
