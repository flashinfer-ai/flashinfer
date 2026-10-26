"""Learned low-rank HyperConnection primitives for flattened multi-stream state.

The residual state is BF16 ``[T, S*H]`` and is logically ``[T, S, H]``.
Projection GEMMs stay outside this op; this package owns the grouped norm,
scaled activation, gate reduction, and residual injection kernels around them.

Declare one immutable operation with ``plan(Caps(...), invocation=...)``;
``PreparationSession`` admits its plan before ``bind`` attaches capacity
storage. The combine launches return live-sized tensors owned by
PyTorch so compiled graphs do not functionalize capacity-sized mutations. The
explicitly named ``reference`` module contains PyTorch correctness oracles and
is never a runtime fallback for the public GPU entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="hyperconnection",
    group="norm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "HyperConnectionConfig",
        "HyperConnectionQuery",
        "plan",
        "bind",
        "run_grouped_rmsnorm",
        "run_scaled_silu",
        "run_gate_mean",
        "run_combine",
        "run_combine_norm",
        "run_engram_mix",
        "run_swiglu",
        "run_add",
        "run_sigmoid",
        "reference",
        "is_supported",
    ),
    dtypes=("bf16",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="3a437ab51680",
        paths=("b12x/norm/mhc/",),
    ),
    test_path="tests/experimental/b12x/norm/test_hyperconnection.py",
    since="1.3.0",
    notes=(
        "The Qwen3.8 Flash Next S=4, H=2560, R=320 BF16 contract uses the "
        "CuTeDSL combine+norm kernel for every non-empty live token count. "
        "Unsupported geometry or layout fails instead of falling back. Triton "
        "is used only for the auxiliary normalization, "
        "activation, gate-reduction, and final residual-injection stages."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        HyperConnectionConfig,
        HyperConnectionQuery,
        Plan,
        bind,
        is_supported,
        plan,
        reference,
        run_combine,
        run_combine_norm,
        run_gate_mean,
        run_grouped_rmsnorm,
        run_scaled_silu,
        run_engram_mix,
        run_swiglu,
        run_add,
        run_sigmoid,
    )

install_lazy_api(globals(), META)
