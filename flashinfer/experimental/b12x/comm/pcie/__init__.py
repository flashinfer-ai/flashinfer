"""PCIe collectives for SM12x multi-GPU boxes (no NVLink).

Caller-owned runtimes retain CUDA-IPC handles and mapped peer buffers.
``query_from_runtime`` and ``plan`` declare work for an established channel;
``PreparationSession`` prepares the plan required by mathematical methods.
``AllReduce.plan`` performs its existing world/message routing during planning.
Pools select only channels provisioned before preparation.

- ``AllReduce``: peer-safe world-size dispatch. TP2-TP8 use the all-peer
  oneshot path; TP9/TP10/TP12/TP16 use bounded-degree islands of up to four GPUs.
- ``OneshotAllReduce``: low-level one-shot all-reduce
  (+ ``all_reduce_fused_add_rms_norm``).
- ``DmaAllReduce``: CE-copy ring reduce-scatter + all-gather for prefill
  sizes, preserving explicitly configured transport crossovers.
- ``PCIeTwoShotBF16``: single-rounding BF16 reduce-scatter, all-gather, and
  all-reduce with FP32 accumulation followed by one BF16 rounding.
- ``TwoShotReduceScatter``: two-shot sequence-parallel collectives with
  per-token FP8-e4m3 transport.
- ``DcpAllToAll``: DCP attention exchange with fused LSE reduce-scatter.
- ``DcpTopKOwnerExchange``: exact DCP candidate owner staging.
- ``VocabParallelArgmax``: TP8/TP12/TP16 fused BF16 add and exact global
  greedy argmax.

Every device kernel is authored in Python with CuTe DSL. Host-side CUDA
Runtime/Driver calls are also made from Python; this package contains no
repo-authored C++ or CUDA source and never invokes a native extension build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="pcie",
    group="comm",
    api_style="planned",
    entry_points=(
        "Plan",
        "PcieConfig",
        "PcieQuery",
        "plan",
        "query_from_runtime",
        "AllReduce",
        "OneshotAllReduce",
        "OneshotAllReducePool",
        "DmaAllReduce",
        "PCIeTwoShotBF16",
        "TwoShotReduceScatter",
        "DcpAllToAll",
        "DcpAllToAllPool",
        "DcpTopKOwnerExchange",
        "VocabParallelArgmax",
        "kimi_topk16",
        "parse_oneshot_max_size",
        "lse_reduce_scatter_reference",
        "owner_stage_reference",
        "is_supported",
    ),
    dtypes=("bf16", "fp32", "fp8_e4m3", "int32", "int64"),
    requires=("multi_gpu",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/distributed/",),
    ),
    test_path="tests/experimental/b12x/comm/test_pcie.py",
    since="0.7.0",
    notes="Python/CuTe DSL collectives with Python CUDA Runtime/Driver control.",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Plan,
        PcieConfig,
        PcieQuery,
        plan,
        query_from_runtime,
        AllReduce,
        DcpAllToAll,
        DcpAllToAllPool,
        DcpTopKOwnerExchange,
        DmaAllReduce,
        OneshotAllReduce,
        OneshotAllReducePool,
        PCIeTwoShotBF16,
        TwoShotReduceScatter,
        VocabParallelArgmax,
        is_supported,
        kimi_topk16,
        lse_reduce_scatter_reference,
        owner_stage_reference,
        parse_oneshot_max_size,
    )

install_lazy_api(globals(), META)
