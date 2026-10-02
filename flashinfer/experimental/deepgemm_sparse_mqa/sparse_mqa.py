"""Prepared compressed sparse MQA metadata and logits on SM100a/SM103a.

Generated programs serve every architecture: per KV layout a runtime metadata
kernel (SM count, sparse capacity, split width, block size and page size are
runtime arguments) and an exact-geometry metadata kernel for the production
geometry (``LIMITS["exact_capacity"]`` sparse blocks of ``LIMITS["sparse_block_kv"]``
tokens, ``LIMITS["page_kv"]``-token pages) compiled for one exported device SM
count each (``LIMITS["exact_num_sms"]``; route key ``metadata:<layout>:exact:<sms>``),
plus the block-scaled logits
kernel (one per format and layout).  ``MODULES`` registers each program once
with the architectures it compiles for; ``ROUTES`` maps a logical kernel key
(``metadata_route_key``, ``logits:<fmt>:<layout>``) to its program; ``LIMITS``
carries the exported geometry.  Plans bind user tensors once; ``run()`` submits
on the current PyTorch stream without allocations.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

# Populated verbatim by the generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_deepgemm_sparse_mqa_28babcca92f8d4cfc322": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_28babcca92f8d4cfc322_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_28babcca92f8d4cfc322_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "SF_Q"],
            ["tma_buffer", "Weights"],
            ["tma_buffer", "KV_TMA"],
            ["tma_buffer", "SF_KV_TMA"],
            ["buffer", "KV"],
            ["buffer", "SF_KV"],
            ["buffer", "Metadata"],
            ["buffer", "Logits"],
            ["parameter", "logits_stride"],
            ["parameter", "kv_page_stride_bytes"],
            ["parameter", "num_sms"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_sparse_mqa_3bfcb411f01cc76113be": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_3bfcb411f01cc76113be_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_3bfcb411f01cc76113be_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "Starts"],
            ["buffer", "Ends"],
            ["buffer", "Context"],
            ["buffer", "BlockTable"],
            ["buffer", "Requests"],
            ["buffer", "Sparse"],
            ["buffer", "Metadata"],
            ["buffer", "Workspace"],
            ["parameter", "num_q_tokens"],
            ["parameter", "num_kv_tokens"],
            ["parameter", "block_table_stride"],
            ["parameter", "num_ctas"],
            ["parameter", "num_sms"],
            ["parameter", "sms_divmod"],
            ["parameter", "num_max_sparse_blocks"],
            ["parameter", "blocks_per_split"],
            ["parameter", "split_divmod"],
            ["parameter", "sparse_block_kv"],
            ["parameter", "block_shift"],
            ["parameter", "blocks_per_page"],
            ["parameter", "page_shift"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_sparse_mqa_45284ac27434cdff81f2": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_45284ac27434cdff81f2_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_45284ac27434cdff81f2_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "SF_Q"],
            ["tma_buffer", "Weights"],
            ["tma_buffer", "KV_TMA"],
            ["tma_buffer", "SF_KV_TMA"],
            ["buffer", "KV"],
            ["buffer", "SF_KV"],
            ["buffer", "Metadata"],
            ["buffer", "Logits"],
            ["parameter", "logits_stride"],
            ["parameter", "kv_page_stride_bytes"],
            ["parameter", "num_sms"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_sparse_mqa_63de295a148bfd61dbca": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_63de295a148bfd61dbca_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_63de295a148bfd61dbca_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "Starts"],
            ["buffer", "Ends"],
            ["buffer", "Context"],
            ["buffer", "BlockTable"],
            ["buffer", "Requests"],
            ["buffer", "Sparse"],
            ["buffer", "Metadata"],
            ["buffer", "Workspace"],
            ["parameter", "num_q_tokens"],
            ["parameter", "num_kv_tokens"],
            ["parameter", "block_table_stride"],
            ["parameter", "num_ctas"],
            ["parameter", "num_sms"],
            ["parameter", "sms_divmod"],
            ["parameter", "num_max_sparse_blocks"],
            ["parameter", "blocks_per_split"],
            ["parameter", "split_divmod"],
            ["parameter", "sparse_block_kv"],
            ["parameter", "block_shift"],
            ["parameter", "blocks_per_page"],
            ["parameter", "page_shift"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a"],
    },
    "cake_deepgemm_sparse_mqa_7824bf2a4a902225d6f8": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_7824bf2a4a902225d6f8_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_7824bf2a4a902225d6f8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "SF_Q"],
            ["tma_buffer", "Weights"],
            ["tma_buffer", "KV_TMA"],
            ["tma_buffer", "SF_KV_TMA"],
            ["buffer", "KV"],
            ["buffer", "SF_KV"],
            ["buffer", "Metadata"],
            ["buffer", "Logits"],
            ["parameter", "logits_stride"],
            ["parameter", "kv_page_stride_bytes"],
            ["parameter", "num_sms"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_sparse_mqa_902ec6b7ae479462d214": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_902ec6b7ae479462d214_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_902ec6b7ae479462d214_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "Starts"],
            ["buffer", "Ends"],
            ["buffer", "Context"],
            ["buffer", "BlockTable"],
            ["buffer", "Requests"],
            ["buffer", "Sparse"],
            ["buffer", "Metadata"],
            ["buffer", "Workspace"],
            ["parameter", "num_q_tokens"],
            ["parameter", "num_kv_tokens"],
            ["parameter", "block_table_stride"],
            ["parameter", "num_ctas"],
            ["parameter", "num_sms"],
            ["parameter", "sms_divmod"],
            ["parameter", "num_max_sparse_blocks"],
            ["parameter", "blocks_per_split"],
            ["parameter", "split_divmod"],
            ["parameter", "sparse_block_kv"],
            ["parameter", "block_shift"],
            ["parameter", "blocks_per_page"],
            ["parameter", "page_shift"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_sparse_mqa_eddc11afc3a31783d4ab": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_eddc11afc3a31783d4ab_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_eddc11afc3a31783d4ab_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "Starts"],
            ["buffer", "Ends"],
            ["buffer", "Context"],
            ["buffer", "BlockTable"],
            ["buffer", "Requests"],
            ["buffer", "Sparse"],
            ["buffer", "Metadata"],
            ["buffer", "Workspace"],
            ["parameter", "num_q_tokens"],
            ["parameter", "num_kv_tokens"],
            ["parameter", "block_table_stride"],
            ["parameter", "num_ctas"],
            ["parameter", "num_sms"],
            ["parameter", "sms_divmod"],
            ["parameter", "num_max_sparse_blocks"],
            ["parameter", "blocks_per_split"],
            ["parameter", "split_divmod"],
            ["parameter", "sparse_block_kv"],
            ["parameter", "block_shift"],
            ["parameter", "blocks_per_page"],
            ["parameter", "page_shift"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a"],
    },
    "cake_deepgemm_sparse_mqa_f9c966dba9fa31868dbb": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_f9c966dba9fa31868dbb_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_f9c966dba9fa31868dbb_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "Starts"],
            ["buffer", "Ends"],
            ["buffer", "Context"],
            ["buffer", "BlockTable"],
            ["buffer", "Requests"],
            ["buffer", "Sparse"],
            ["buffer", "Metadata"],
            ["buffer", "Workspace"],
            ["parameter", "num_q_tokens"],
            ["parameter", "num_kv_tokens"],
            ["parameter", "block_table_stride"],
            ["parameter", "num_ctas"],
            ["parameter", "num_sms"],
            ["parameter", "sms_divmod"],
            ["parameter", "num_max_sparse_blocks"],
            ["parameter", "blocks_per_split"],
            ["parameter", "split_divmod"],
            ["parameter", "sparse_block_kv"],
            ["parameter", "block_shift"],
            ["parameter", "blocks_per_page"],
            ["parameter", "page_shift"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_103a"],
    },
    "cake_deepgemm_sparse_mqa_ff8c1a20aeba51091b1e": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_ff8c1a20aeba51091b1e_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_ff8c1a20aeba51091b1e_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "Starts"],
            ["buffer", "Ends"],
            ["buffer", "Context"],
            ["buffer", "BlockTable"],
            ["buffer", "Requests"],
            ["buffer", "Sparse"],
            ["buffer", "Metadata"],
            ["buffer", "Workspace"],
            ["parameter", "num_q_tokens"],
            ["parameter", "num_kv_tokens"],
            ["parameter", "block_table_stride"],
            ["parameter", "num_ctas"],
            ["parameter", "num_sms"],
            ["parameter", "sms_divmod"],
            ["parameter", "num_max_sparse_blocks"],
            ["parameter", "blocks_per_split"],
            ["parameter", "split_divmod"],
            ["parameter", "sparse_block_kv"],
            ["parameter", "block_shift"],
            ["parameter", "blocks_per_page"],
            ["parameter", "page_shift"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_103a"],
    },
    "cake_deepgemm_sparse_mqa_ff9376991950f1fcab1d": {
        "sources": [
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_ff9376991950f1fcab1d_kernel.cu",
            "experimental/deepgemm_sparse_mqa/generated/cake_deepgemm_sparse_mqa_ff9376991950f1fcab1d_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "SF_Q"],
            ["tma_buffer", "Weights"],
            ["tma_buffer", "KV_TMA"],
            ["tma_buffer", "SF_KV_TMA"],
            ["buffer", "KV"],
            ["buffer", "SF_KV"],
            ["buffer", "Metadata"],
            ["buffer", "Logits"],
            ["parameter", "logits_stride"],
            ["parameter", "kv_page_stride_bytes"],
            ["parameter", "num_sms"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
}
ROUTES: dict[str, str] = {
    "logits:mxfp4:contiguous": "cake_deepgemm_sparse_mqa_45284ac27434cdff81f2",
    "logits:mxfp4:paged": "cake_deepgemm_sparse_mqa_ff9376991950f1fcab1d",
    "logits:mxfp8:contiguous": "cake_deepgemm_sparse_mqa_28babcca92f8d4cfc322",
    "logits:mxfp8:paged": "cake_deepgemm_sparse_mqa_7824bf2a4a902225d6f8",
    "metadata:contiguous": "cake_deepgemm_sparse_mqa_902ec6b7ae479462d214",
    "metadata:contiguous:exact:148": "cake_deepgemm_sparse_mqa_63de295a148bfd61dbca",
    "metadata:contiguous:exact:152": "cake_deepgemm_sparse_mqa_f9c966dba9fa31868dbb",
    "metadata:paged": "cake_deepgemm_sparse_mqa_3bfcb411f01cc76113be",
    "metadata:paged:exact:148": "cake_deepgemm_sparse_mqa_eddc11afc3a31783d4ab",
    "metadata:paged:exact:152": "cake_deepgemm_sparse_mqa_ff8c1a20aeba51091b1e",
}
LIMITS: dict[str, Any] = {
    "max_sparse_blocks": 2048,
    "max_sms": 160,
    "sparse_block_kv": 8,
    "page_kv": 64,
    "heads": 32,
    "exact_capacity": 2048,
    "exact_num_sms": [148, 152],
}

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


@functools.cache
def device_facts(device_index: int) -> tuple[str, int]:
    """``(architecture, SM count)`` of CUDA device ``device_index``, resolved once per process."""
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    arch = _ARCHES.get((properties.major, properties.minor))
    if arch is None or not any(arch in record["arches"] for record in MODULES.values()):
        raise RuntimeError(
            f"Sparse MQA has no exported programs for compute capability "
            f"{(properties.major, properties.minor)}; exported architectures: "
            f"{sorted({a for record in MODULES.values() for a in record['arches']})}"
        )
    sms = int(properties.multi_processor_count)
    if not 1 <= sms <= LIMITS["max_sms"]:
        raise RuntimeError(
            f"Sparse MQA metadata schedules cover 1..{LIMITS['max_sms']} SMs, this device has {sms}"
        )
    return arch, sms


def device_arch(device) -> str:
    """Generated-program architecture of ``device`` (raises when none is exported)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Sparse MQA requires a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return device_facts(index)[0]


@functools.cache
def program_spec(arch: str, name: str):
    """JIT spec of program ``name`` for architecture ``arch`` (one of the program's exported architectures)."""
    from ...jit import env
    from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not exported for {arch}")
    flags = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=[
            env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]
        ],
        extra_cuda_cflags=[
            *flags,
            *record["compile_flags"],
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,  # Only the program's own compile flags select fast math.
    )


def load_program(arch: str, name: str):
    """Compile ``name`` for the exact architecture of the attached device; returns ``(module, spec)``."""
    spec = program_spec(arch, name)
    return spec.build_and_load(), spec


def fast_divmod(divisor: int) -> tvm_ffi.Shape:
    """Host-precomputed divmod constants of a positive divisor (the generated binding's three-field carrier)."""
    if not 1 <= divisor <= 0x7FFF_FFFF:
        raise ValueError(f"divisor must be in [1, 2147483647], got {divisor}")
    if divisor == 1:
        return tvm_ffi.Shape((1, 0, 0))
    p = 31 + (divisor - 1).bit_length()
    return tvm_ffi.Shape(
        (divisor, (((1 << p) + divisor - 1) // divisor) & 0xFFFF_FFFF, p - 32)
    )


def blocks_per_split(fmt: str, sparse_block_kv: int) -> int:
    """Sparse blocks per logits split: 640 (MXFP4) or 512 (MXFP8) KV tokens over the block size."""
    if fmt not in ("mxfp4", "mxfp8"):
        raise ValueError("fmt must be mxfp4 or mxfp8")
    if sparse_block_kv not in (8, 16):
        raise ValueError("sparse_block_kv must be 8 or 16")
    return (640 if fmt == "mxfp4" else 512) // sparse_block_kv


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _log2_exact(name: str, value: int) -> int:
    """log2 of a power of two; the kernel splits tokens and pages with shifts."""
    if value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")
    return value.bit_length() - 1


def metadata_max_splits(queries, capacity, *, fmt, sparse_block_kv, paged):
    """Upper bound of the logits splits one metadata buffer can carry."""
    nb = blocks_per_split(fmt, sparse_block_kv)
    return (
        queries * _ceildiv(capacity, nb)
        if paged
        else _ceildiv(queries, 2) * _ceildiv(2 * capacity, nb)
    )


def metadata_size_bytes(queries, capacity, *, fmt, sparse_block_kv, paged, num_sms):
    """Size the packed split/schedule ABI, including its maximum extent."""
    nb = blocks_per_split(fmt, sparse_block_kv)
    splits = metadata_max_splits(
        queries, capacity, fmt=fmt, sparse_block_kv=sparse_block_kv, paged=paged
    )
    return 16 + splits * (16 + nb * 8) + _ceildiv(splits, num_sms) * num_sms * 16


def metadata_workspace_words(queries, capacity, *, fmt, sparse_block_kv, paged):
    """int32 words of the metadata workspace: three counters (words 0/32/64), two words of split info
    per query, and for contiguous rows one 64-bit owner record per possible split."""
    words = 96 + queries * 2
    if not paged:
        words += (
            2
            * _ceildiv(queries, 2)
            * _ceildiv(2 * capacity, blocks_per_split(fmt, sparse_block_kv))
        )
    return words


def metadata_route_key(*, paged, capacity, sparse_block_kv, page_kv, num_sms, use_unaligned_ks=False):
    """``ROUTES`` key of the metadata program for one geometry: the exact-geometry program at the exported
    production geometry (capacity ``LIMITS["exact_capacity"]``, ``LIMITS["sparse_block_kv"]``-token blocks and,
    for paged rows, ``LIMITS["page_kv"]``-token pages) compiled for this device's SM count
    (``LIMITS["exact_num_sms"]``), the runtime program otherwise."""
    layout = "paged" if paged else "contiguous"
    exact = (
        not use_unaligned_ks
        and num_sms in LIMITS["exact_num_sms"]
        and capacity == LIMITS["exact_capacity"]
        and sparse_block_kv == LIMITS["sparse_block_kv"]
        and (not paged or page_kv == LIMITS["page_kv"])
    )
    return f"metadata:{layout}" + (f":exact:{num_sms}" if exact else "")


def _arguments(record, bindings):
    arguments = []
    for kind, key in record["arg_plan"]:
        if kind not in ("buffer", "tma_buffer", "parameter", "grid"):
            raise NotImplementedError(
                f"unsupported argument kind {kind!r} in the exported plan"
            )
        arguments.append(bindings[key])
    return tuple(arguments)


def _check_tensors(tensors, device):
    if any(
        t.device != device or not t.is_contiguous() for t in tensors if t is not None
    ):
        raise ValueError("inputs must be contiguous tensors on one CUDA device")


class SparseMetadataPlan:
    """Prepare the packed metadata of one sparse index set for repeated generation.

    ``sparse_indices`` is sorted int32 ``[Q, capacity]`` with duplicate padding in
    unused slots (capacity a multiple of 4, at most ``LIMITS["max_sparse_blocks"]``).
    Contiguous inputs pass ``starts``/``ends`` int32 ``[Q]`` and ``num_kv_tokens``;
    paged inputs pass ``context_lens``/``request_indices`` int32 ``[Q]`` and a
    per-query ``block_table`` int32 ``[Q, pages]``.  ``sparse_block_kv`` is 8 or 16
    and ``page_kv`` a power-of-two multiple of it.  The packed metadata is uint8 of
    ``metadata_size_bytes()``; the caller-provided int32 workspace of
    ``metadata_workspace_words()`` entries must initially be zero, and the kernel
    restores its three counters after every submission.  Split allocation order is
    not a canonical serialization; consume metadata through this ABI.
    """

    def __init__(
        self,
        sparse_indices,
        *,
        fmt="mxfp4",
        sparse_block_kv=8,
        page_kv=64,
        use_unaligned_ks=False,
        starts=None,
        ends=None,
        num_kv_tokens=0,
        context_lens=None,
        block_table=None,
        request_indices=None,
        metadata=None,
        workspace=None,
    ):
        import torch

        if sparse_indices.dtype != torch.int32 or sparse_indices.ndim != 2:
            raise ValueError("sparse_indices must be int32[Q,capacity]")
        queries, capacity = sparse_indices.shape
        if (
            queries < 1
            or capacity < 1
            or capacity % 4
            or capacity > LIMITS["max_sparse_blocks"]
        ):
            raise ValueError(
                f"sparse capacity must be a positive multiple of 4 at most {LIMITS['max_sparse_blocks']}, got {capacity}"
            )
        nb = blocks_per_split(fmt, sparse_block_kv)
        paged = context_lens is not None
        if paged and (use_unaligned_ks or page_kv % sparse_block_kv):
            raise ValueError(
                "paged sparse MQA requires aligned blocks dividing the page"
            )
        blocks_per_page = max(1, page_kv // sparse_block_kv)
        block_shift = _log2_exact("sparse_block_kv", sparse_block_kv)
        page_shift = _log2_exact("blocks per page", blocks_per_page)
        vectors = (context_lens, request_indices) if paged else (starts, ends)
        if any(
            t is None or t.dtype != torch.int32 or tuple(t.shape) != (queries,)
            for t in vectors
        ):
            raise ValueError("window/request vectors must be int32[Q]")
        if paged:
            if (
                block_table is None
                or block_table.dtype != torch.int32
                or block_table.ndim != 2
                or block_table.shape[0] != queries
            ):
                raise ValueError("paged block_table must be int32[Q,pages]")
        elif num_kv_tokens <= 0:
            raise ValueError("contiguous metadata requires positive num_kv_tokens")
        device = sparse_indices.device
        self.arch, self.num_sms = device_facts(device.index)
        max_splits = metadata_max_splits(
            queries, capacity, fmt=fmt, sparse_block_kv=sparse_block_kv, paged=paged
        )
        if (max_splits + 1) * self.num_sms > 0x7FFF_FFFF:
            raise ValueError(
                f"metadata schedule of {max_splits} splits over {self.num_sms} SMs exceeds the 32-bit split range"
            )
        self.config = dict(
            fmt=fmt,
            paged=paged,
            num_max_sparse_blocks=capacity,
            sparse_block_kv=sparse_block_kv,
            page_kv=page_kv,
            use_unaligned_ks=bool(use_unaligned_ks),
        )
        if use_unaligned_ks:
            raise NotImplementedError(
                "unaligned contiguous windows are not an exported sparse MQA specialization"
            )
        self.program = ROUTES[
            metadata_route_key(
                paged=paged,
                capacity=capacity,
                sparse_block_kv=sparse_block_kv,
                page_kv=page_kv,
                num_sms=self.num_sms,
                use_unaligned_ks=use_unaligned_ks,
            )
        ]
        size = metadata_size_bytes(
            queries,
            capacity,
            fmt=fmt,
            sparse_block_kv=sparse_block_kv,
            paged=paged,
            num_sms=self.num_sms,
        )
        words = metadata_workspace_words(
            queries, capacity, fmt=fmt, sparse_block_kv=sparse_block_kv, paged=paged
        )
        if metadata is None:
            metadata = torch.empty(size, dtype=torch.uint8, device=device)
        if workspace is None:
            workspace = torch.zeros(words, dtype=torch.int32, device=device)
        if metadata.dtype != torch.uint8 or tuple(metadata.shape) != (size,):
            raise ValueError(
                "metadata must be a uint8 vector with the required packed extent"
            )
        if workspace.dtype != torch.int32 or tuple(workspace.shape) != (words,):
            raise ValueError(
                "workspace must be an int32 vector with metadata_workspace_words() entries"
            )
        self._retained = (
            sparse_indices,
            starts,
            ends,
            context_lens,
            block_table,
            request_indices,
            metadata,
            workspace,
        )
        _check_tensors(self._retained, device)
        placeholder = sparse_indices.view(torch.uint32)
        ctas = min(queries if paged else (queries + 1) // 2, self.num_sms * 4)
        self.bindings = dict(
            Starts=starts.view(torch.uint32) if starts is not None else placeholder,
            Ends=ends.view(torch.uint32) if ends is not None else placeholder,
            Context=context_lens.view(torch.uint32) if paged else placeholder,
            BlockTable=block_table.view(torch.uint32) if paged else placeholder,
            Requests=request_indices.view(torch.uint32) if paged else placeholder,
            Sparse=placeholder,
            Metadata=metadata.view(torch.uint32),
            Workspace=workspace.view(torch.uint32),
            num_q_tokens=queries,
            num_kv_tokens=num_kv_tokens,
            block_table_stride=block_table.stride(0) if paged else 0,
            num_ctas=ctas,
            num_sms=self.num_sms,
            sms_divmod=fast_divmod(self.num_sms),
            num_max_sparse_blocks=capacity,
            blocks_per_split=nb,
            split_divmod=fast_divmod(nb),
            sparse_block_kv=sparse_block_kv,
            block_shift=block_shift,
            blocks_per_page=blocks_per_page,
            page_shift=page_shift,
            grid_x=ctas,
            grid_y=1,
            grid_z=1,
        )
        module, _spec = load_program(self.arch, self.program)
        self._entry = module[MODULES[self.program]["ffi_entry"]]
        self._args = _arguments(MODULES[self.program], self.bindings)
        self.metadata, self.workspace = metadata, workspace
        self.queries, self.capacity = queries, capacity

    def run(self):
        with tvm_ffi.use_torch_stream():
            self._entry(*self._args)
        return self.metadata


class SparseMqaPlan:
    """Bind quantized q/kv and emit compressed BF16 logits, metadata generation included.

    ``q`` is packed E2M1 ``[Q, 32, 64]`` or E4M3 ``[Q, 32, 128]``; ``sf_q`` int32
    ``[Q, 32]`` holds four UE8M0 scale bytes per row; ``weights`` is BF16 ``[Q, 32]``.
    Contiguous ``kv`` is packed rows ``[K, 64|128]`` with int32 ``[K]`` scales; paged
    ``kv`` is uint8 ``[pages, stride]`` with each page's row bytes followed by its
    scale bytes, padded to a 512-byte stride (``sf_kv`` unused).  The exported
    logits programs use ``LIMITS["sparse_block_kv"]``-token blocks and
    ``LIMITS["page_kv"]``-token pages.  Output is BF16 ``[Q, capacity * block]``;
    only selected valid slots are written.  ``run()`` submits metadata then logits
    on the current stream.  Do not use one plan's mutable buffers concurrently on
    different streams.
    """

    def __init__(self, q, sf_q, kv, sf_kv, weights, metadata_plan, *, output=None):
        import torch

        config = metadata_plan.config
        fmt, paged, block = config["fmt"], config["paged"], config["sparse_block_kv"]
        queries, capacity = metadata_plan.queries, metadata_plan.capacity
        layout = "paged" if paged else "contiguous"
        if (
            block != LIMITS["sparse_block_kv"]
            or config["page_kv"] != LIMITS["page_kv"]
            or config["use_unaligned_ks"]
        ):
            raise NotImplementedError(
                f"No exported {fmt} {layout} logits program for sparse_block_kv={block}, page_kv={config['page_kv']}, "
                f"use_unaligned_ks={config['use_unaligned_ks']}; exported: sparse_block_kv={LIMITS['sparse_block_kv']}, "
                f"page_kv={LIMITS['page_kv']}, aligned windows"
            )
        row = 64 if fmt == "mxfp4" else 128
        heads = LIMITS["heads"]
        if tuple(q.shape) != (queries, heads, row):
            raise ValueError(
                "Q shape does not match metadata query count and precision"
            )
        supported_q = (
            (torch.uint8, torch.int8)
            if fmt == "mxfp4"
            else (torch.float8_e4m3fn, torch.uint8)
        )
        if (
            q.dtype not in supported_q
            or sf_q.dtype != torch.int32
            or tuple(sf_q.shape) != (queries, heads)
        ):
            raise ValueError("Q/scales have the wrong precision or packed scale layout")
        if weights.dtype != torch.bfloat16 or tuple(weights.shape) != (queries, heads):
            raise ValueError(f"weights must be BF16[Q,{heads}]")
        if paged:
            minimum = (config["page_kv"] * (row + 4) + 511) // 512 * 512
            if (
                kv.dtype != torch.uint8
                or kv.ndim != 2
                or kv.shape[1] < minimum
                or kv.stride(0) % 512
            ):
                raise ValueError(
                    "paged KV must use the packed row/scales layout and 512-byte stride"
                )
        elif (
            kv.dtype not in supported_q
            or kv.ndim != 2
            or kv.shape[1] != row
            or sf_kv is None
            or sf_kv.dtype != torch.int32
            or tuple(sf_kv.shape) != (kv.shape[0],)
        ):
            raise ValueError("contiguous KV/scales have the wrong packed row layout")
        if output is None:
            output = torch.empty(
                (queries, capacity * block), device=q.device, dtype=torch.bfloat16
            )
        if output.dtype != torch.bfloat16 or tuple(output.shape) != (
            queries,
            capacity * block,
        ):
            raise ValueError("output must be BF16[Q,capacity*sparse_block_kv]")
        self._retained = (
            q,
            sf_q,
            kv,
            sf_kv,
            weights,
            output,
            metadata_plan.metadata,
            metadata_plan.workspace,
        )
        _check_tensors(self._retained, q.device)
        kv_tma = (kv if not paged else q).view(torch.uint8).reshape(-1, row)
        sf_tma = (sf_kv if not paged else sf_q).view(torch.int32).reshape(1, -1)
        bindings = dict(
            Q=q.view(torch.uint8).reshape(-1, row),
            SF_Q=sf_q.view(torch.int32).reshape(-1, heads),
            Weights=weights.reshape(-1, heads),
            KV_TMA=kv_tma,
            SF_KV_TMA=sf_tma,
            KV=kv.view(torch.uint8),
            SF_KV=(sf_kv if not paged else sf_q).view(torch.uint32),
            Metadata=metadata_plan.metadata.view(torch.uint32),
            Logits=output,
            logits_stride=output.stride(0),
            kv_page_stride_bytes=kv.stride(0) if paged else 0,
            num_sms=metadata_plan.num_sms,
            grid_x=metadata_plan.num_sms,
            grid_y=1,
            grid_z=1,
        )
        program = ROUTES[f"logits:{fmt}:{layout}"]
        module, _spec = load_program(metadata_plan.arch, program)
        self.programs = {"metadata": metadata_plan.program, "logits": program}
        self._submissions = (
            (metadata_plan._entry, metadata_plan._args),
            (
                module[MODULES[program]["ffi_entry"]],
                _arguments(MODULES[program], bindings),
            ),
        )
        self.metadata_plan, self.output = metadata_plan, output

    def run(self):
        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.output
