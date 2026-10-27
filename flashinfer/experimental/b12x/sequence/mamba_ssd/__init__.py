"""Prepared Mamba-2 selective scan on SM120/SM121.

Packed BF16 inputs update caller-owned BF16 or FP32 state and output through
tensor-core tiles and parallel 128-token chunks. Projection, convolution, and gated normalization
belong to the caller. Importing this experimental operation is the opt-in.
"""

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mamba_ssd",
    group="sequence",
    api_style="planned",
    entry_points=(
        "MambaQuery",
        "MambaConfig",
        "plan",
        "query_from_call",
        "run",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp32", "int32", "int64"),
    recipes=("mamba2",),
    provenance=Provenance(
        repo="https://github.com/flashinfer-ai/flashinfer",
        commit="b1f081e6e381c333718054cc2cd1c3d82c855a79",
        paths=(
            "flashinfer/experimental/b12x/sequence/_shared/delta_prefill/_cute_kernels.py",
        ),
    ),
    test_path="tests/experimental/b12x/sequence/test_mamba_ssd.py",
)

if TYPE_CHECKING:
    from .api import (
        MambaQuery,
        MambaConfig,
        clear_caches,
        is_supported,
        plan,
        query_from_call,
        run,
    )

install_lazy_api(globals(), META)
