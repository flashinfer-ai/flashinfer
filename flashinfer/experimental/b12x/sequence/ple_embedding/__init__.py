"""Prime-hashed PLE embedding lookup.

``compute_geometry`` and ``storage_layout`` calculate host-only table metadata;
``allocate_geometry`` and ``allocate_storage`` create checkpoint-owned buffers.
``plan`` declares the table plan without loading kernels or allocating device memory.
A ``PreparationSession`` admits the declaration and returns the prepared plan
required by ``bind``. Geometry tensors are borrowed, not replaced.
``run`` hashes tokens, gathers selected rows from the local table shard, applies
inline dequantization for FP8 or NVFP4, and writes a BF16 embedding contribution.
``io_uring`` uses ``DiskTable`` with O_DIRECT to stage only the current batch.
Its host-I/O ``run`` produces fixed GPU outputs outside capture; graphs may
then consume those outputs without performing disk reads.

The expressed operation is one hash, gather, and dequantization call. Its
binding exposes only caller-owned inputs and the output; intermediate
embedding IDs remain private so implementations may fuse the operation
without changing the integration API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="ple_embedding",
    group="sequence",
    api_style="planned",
    entry_points=(
        "QuantMode",
        "TableMemory",
        "TableStorage",
        "DiskTable",
        "Caps",
        "Plan",
        "TableLayout",
        "storage_layout",
        "Geometry",
        "GeometryTensors",
        "compute_geometry",
        "allocate_geometry",
        "invocation_from_tensors",
        "Binding",
        "PleEmbeddingConfig",
        "PleEmbeddingQuery",
        "plan",
        "allocate_storage",
        "bind",
        "run",
        "is_supported",
    ),
    dtypes=("bf16", "fp8_e4m3", "nvfp4", "int64"),
    recipes=(
        "packed_eos_bounded_bf16",
        "packed_eos_bounded_fp8_per_tensor",
        "packed_eos_bounded_nvfp4_group16",
    ),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="74d2023e6cdb705568011110ad3f8e9b0806b647",
        paths=("b12x/sequence/ple_embedding/",),
    ),
    test_path="tests/experimental/b12x/sequence/test_ple_embedding.py",
    since="1.3.0",
    notes=(
        "FP8 E4M3 and NVFP4 tables remain quantized in persistent storage; "
        "only selected local rows are dequantized. The expressed API is one "
        "opaque hash, local-shard gather, and inline-dequantization operation. "
        "Device-resident, CUDA-mapped host, and io_uring-staged checkpoint "
        "storage share the gather/dequantization implementation. Its Triton "
        "implementation is functional but not throughput-qualified."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        DiskTable,
        Plan,
        TableLayout,
        storage_layout,
        Geometry,
        GeometryTensors,
        compute_geometry,
        allocate_geometry,
        invocation_from_tensors,
        PleEmbeddingConfig,
        PleEmbeddingQuery,
        QuantMode,
        TableMemory,
        TableStorage,
        allocate_storage,
        bind,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
