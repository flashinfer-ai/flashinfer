"""BF16 -> NVFP4 TMA tile quantizer for SM12x.

Quantizes a [M, K] BF16 tensor (M, K multiples of 128) into packed FP4
values plus e4m3 scales in the dense-GEMM MMA layout (sf vec 16), using a
128x128 TMA tile kernel. ``plan(m, k)`` returns a metadata-only declaration;
``allocate_outputs`` provisions its output pair before preparation.
``PreparationSession`` selects, compiles, and primes the quantizer with a
``PreparedCall`` over the caller's real activation/scale/output tensors.
``run(plan=..., x=..., global_scale=..., outputs=...)`` consumes the
resulting prepared plan without lookup or compilation. There is no
separate bind step: the outputs container owns the explicit output buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="nvfp4",
    group="quantization",
    api_style="planned",
    entry_points=(
        "Outputs",
        "Plan",
        "Nvfp4QuantizationConfig",
        "Nvfp4QuantizationQuery",
        "plan",
        "allocate_outputs",
        "run",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("nvfp4",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/quantization/",),
    ),
    test_path="tests/experimental/b12x/quantization/test_nvfp4.py",
    since="0.7.0",
    notes=(
        "No bind step: outputs are caller-allocated via allocate_outputs. "
        "Dead at the original port baseline; revived by the CUTLASS DSL 4.6 "
        "migration."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Nvfp4QuantizationConfig,
        Nvfp4QuantizationQuery,
        Outputs,
        Plan,
        allocate_outputs,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
