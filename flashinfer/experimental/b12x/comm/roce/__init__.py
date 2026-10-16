"""``b12x.comm.roce`` (RoCEnante): one-shot all-reduce and all-gather over RoCE for multi-node TP.

Target: clusters of DGX Spark nodes joined by their ConnectX-7 200 GbE ports,
one GPU per node.  The GB10's unified memory lets the NIC RDMA-write straight
into pinned host memory that the GPU kernel then reads in place, so no
GPUDirect RDMA (dmabuf/peermem) support is required.

``AllReduce`` mirrors the ``comm.pcie.AllReduce`` surface (``from_exchange_group``,
``should_allreduce``, ``all_reduce``, ``for_stream``, ``capture``, ``close``) plus
``should_all_gather``/``all_gather`` for dim-0 and last-dim concatenation, so
integrations can dispatch to it behind the same adapter.  See
``roce_oneshot.py`` for the protocol and constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="roce",
    group="comm",
    api_style="stateful",
    entry_points=(
        "API_VERSION",
        "AllReduce",
        "DEFAULT_MAX_GATHER_BYTES",
        "DEFAULT_MAX_SIZE",
        "SUPPORTED_DTYPES",
        "SUPPORTED_WORLD_SIZES",
        "RoceQuery",
        "plan",
        "query_from_runtime",
        "default_gid_index",
        "discover_hcas",
        "is_supported",
    ),
    archs=("sm121a",),
    dtypes=("bf16", "fp16", "fp32", "int32", "int64"),
    requires=("multi_node", "rdma"),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/b12x",
        commit="b9e450f1",
        paths=("b12x/comm/roce/",),
    ),
    test_path="tests/experimental/b12x/comm/test_roce_oneshot_gpu.py",
    since="1.3.0",
    notes=(
        "RoCEnante. Python/CuTe DSL kernels; the RDMA proxy is a small C file "
        "built with the host C compiler at first use (no CUDA extension build)."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        API_VERSION,
        DEFAULT_MAX_GATHER_BYTES,
        DEFAULT_MAX_SIZE,
        SUPPORTED_DTYPES,
        SUPPORTED_WORLD_SIZES,
        AllReduce,
        RoceQuery,
        default_gid_index,
        discover_hcas,
        is_supported,
        plan,
        query_from_runtime,
    )

install_lazy_api(globals(), META)
