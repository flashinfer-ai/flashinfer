# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""PCIe collectives for SM12x multi-GPU boxes (no NVLink).

Stateful class API (collectives own CUDA-IPC handles, mapped peer buffers,
and their JIT-built extension): kwargs-only constructors with the shared
vocabulary (rank / world_size / device / ...), CUDA-graph-capturable methods,
pools via ``<Class>Pool``.

- ``OneshotAllReduce``: one-shot all-reduce (+ ``all_reduce_fused_add_rms_norm``).
- ``DmaAllReduce``: CE-copy ring reduce-scatter + all-gather for prefill
  sizes, with a runtime crossover autotuner (``autotune_dma_crossovers``).
- ``TwoShotReduceScatter``: two-shot sequence-parallel collectives with
  per-token FP8-e4m3 transport.
- ``DcpAllToAll``: DCP attention exchange with fused LSE reduce-scatter.

Raw CUDA (not CuTe): each class JIT-builds its colocated ``.cu`` via
torch.utils.cpp_extension, so nvcc must be available at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="pcie",
    group="comm",
    api_style="stateful",
    entry_points=(
        "OneshotAllReduce",
        "OneshotAllReducePool",
        "DmaAllReduce",
        "TwoShotReduceScatter",
        "DcpAllToAll",
        "DcpAllToAllPool",
        "autotune_dma_crossovers",
        "parse_oneshot_max_size",
        "lse_reduce_scatter_reference",
        "is_supported",
    ),
    dtypes=("bf16", "fp32", "fp8_e4m3"),
    requires=("multi_gpu",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/distributed/",),
    ),
    test_path="tests/experimental/comm/test_pcie.py",
    since="0.7.0",
    notes="Raw CUDA JIT-built via torch cpp_extension; needs nvcc at runtime.",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        DcpAllToAll,
        DcpAllToAllPool,
        DmaAllReduce,
        OneshotAllReduce,
        OneshotAllReducePool,
        TwoShotReduceScatter,
        autotune_dma_crossovers,
        is_supported,
        lse_reduce_scatter_reference,
        parse_oneshot_max_size,
    )

install_lazy_api(globals(), META)
