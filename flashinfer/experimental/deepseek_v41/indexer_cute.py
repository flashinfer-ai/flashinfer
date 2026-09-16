# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Block8 gather adapter for the shared native CuTe FP4 scorer."""

import functools
from pathlib import Path

import torch
import triton as tr
import triton.language as tl

from ._utils import _overlaps


def workspace_size(batch, count):
    align16 = lambda x: (x + 15) // 16 * 16
    encoded = align16(4 * batch * count)
    # Each candidate has eight validity bits; pad rows for int32 loads.
    mask = align16(batch * tr.cdiv(count, 4) * 4)
    return encoded, mask


@functools.cache
def _compile_candidate(page, count, num_sms, arch, offset32=False):
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.experimental import primitives  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cute_dsl candidate scoring requires CUTLASS DSL>=4.7"
        ) from exc
    from ...attn_scores.kernels import fp4_paged_mqa_logits as native
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    batch = cute.sym_int()
    physical = cute.sym_int()
    width = count * 8
    mask_stride = tr.cdiv(count, 4) * 4
    kv = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (physical, page * 68),
        stride=(cute.sym_int64(), 1),
        assumed_align=16,
    )
    q = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (32, 64, batch),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    qs = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (32, batch),
        stride_order=(0, 1),
        assumed_align=16,
    )
    weights = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (32, batch),
        stride_order=(0, 1),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (batch, width),
        stride=(cute.sym_int64(), 1),
        assumed_align=16,
    )
    encoded = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch, count),
        stride=(count, 1),
        assumed_align=16,
    )
    lengths = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch,),
        stride=(1,),
        assumed_align=4,
    )
    mask = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (batch, mask_stride),
        stride=(mask_stride, 1),
        assumed_align=16,
    )
    kernel = native.FP4MQALogitsKernel(
        block_kv=128,
        phys_block_kv=page,
        num_heads=32,
        head_dim=128,
        next_n=1,
        num_sms=num_sms,
        epi_dtype=cutlass.Float32,
        output_dtype=cutlass.BFloat16,
        arch=arch,
        candidate_block8=True,
        candidate_offset32=offset32,
        weight_dtype=cutlass.BFloat16,
    )

    def compile_fn():
        return cute.compile(
            kernel,
            kv,
            q,
            qs,
            weights,
            out,
            encoded,
            lengths,
            None,
            cutlass.Int32(1),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            mask,
            options=f"--gpu-arch {arch} --enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "ds41_candidate_indexer",
        f"page{page}_c{count}_sms{num_sms}_{arch}_offset32{int(offset32)}",
        compile_fn,
        extra_key_files=(str(Path(__file__)), str(Path(native.__file__))),
    )


def _check_candidate_layout(q, qs, kv, weights, out):
    batch, page = q.shape[0], kv.shape[1]
    if any(x.data_ptr() % 16 for x in (q, qs, kv, weights)):
        raise ValueError("cute_dsl requires 16-byte-aligned Q, scales, KV and weights")
    if kv.shape[0] * (page // 8) >= 2**31:
        raise ValueError("cute_dsl encoded page IDs must fit int32; use triton")
    if batch * out.stride(0) >= 2**31:
        raise ValueError(
            "cute_dsl padded output element span must fit int32; use triton"
        )


def _launch_candidate(q, qs, kv, weights, out, encoded, lengths, mask):
    batch, count = encoded.shape
    page = kv.shape[1]
    major, minor = torch.cuda.get_device_capability(q.device)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    # Bound the complete strided pool, including its final page. This depends
    # only on tensor metadata and remains safe under changed-input graph replay.
    span = ((kv.shape[0] - 1) * kv.stride(0) + page * 68) * kv.element_size()
    compiled = _compile_candidate(page, count, sms, f"sm_{major}{minor}a", span < 2**32)
    compiled(
        kv.flatten(1),
        q.permute(1, 2, 0),
        qs.view(torch.int32).reshape(batch, 32).t(),
        weights.t(),
        out,
        encoded,
        lengths,
        None,
        kv.shape[0],
        batch,
        mask,
    )
    return out


def candidate_scores(
    q, qs, kv, weights, lengths, table, candidates, out, context, workspace
):
    batch, count = candidates.shape
    page = kv.shape[1]
    _check_candidate_layout(q, qs, kv, weights, out)
    encoded_bytes, mask_bytes = workspace_size(batch, count)
    size = encoded_bytes + mask_bytes
    if workspace is None:
        workspace = torch.empty(size, dtype=torch.uint8, device=q.device)
    if (
        workspace.dtype != torch.uint8
        or workspace.device != q.device
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.numel() < size
        or workspace.data_ptr() % 16
    ):
        raise ValueError(
            f"cute_dsl workspace needs a contiguous aligned uint8 CUDA vector of at least {size} bytes"
        )
    if any(
        _overlaps(workspace, x)
        for x in (q, qs, kv, weights, lengths, table, candidates, out)
    ):
        raise ValueError("workspace must not overlap inputs or output")
    encoded = workspace[: batch * count * 4].view(torch.int32).view(batch, count)
    mask_stride = tr.cdiv(count, 4) * 4
    mask = workspace[encoded_bytes : encoded_bytes + batch * mask_stride].view(
        batch, mask_stride
    )
    _candidate_metadata[(batch, tr.cdiv(count, 256))](
        candidates,
        table,
        lengths,
        encoded,
        mask,
        count,
        table.shape[1],
        page,
        kv.shape[0],
        context,
        256,
        num_warps=4,
    )
    return _launch_candidate(q, qs, kv, weights, out, encoded, lengths, mask)


@tr.jit
def _candidate_metadata(
    C,
    T,
    L,
    E,
    M,
    WIDTH: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE: tl.constexpr,
    PHYS: tl.constexpr,
    CONTEXT: tl.constexpr,
    BLOCK: tl.constexpr,
    SNAPSHOT=None,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    candidate = tl.load(C + b.to(tl.int64) * WIDTH + c, c < WIDTH, -1).to(tl.int64)
    first = candidate * 8
    visible = tl.load(L + b)
    if SNAPSHOT is not None:
        if tl.program_id(1) == 0:
            tl.store(SNAPSHOT + b, visible)
    valid = (c < WIDTH) & (first >= 0) & (first < CONTEXT) & (first < visible)
    page = tl.load(T + b.to(tl.int64) * PAGES + first // PAGE, valid, -1)
    valid &= (page >= 0) & (page < PHYS)
    encoded = page.to(tl.int64) * (PAGE // 8) + candidate % (PAGE // 8)
    tl.store(
        E + b.to(tl.int64) * WIDTH + c,
        tl.where(valid, encoded, PHYS * (PAGE // 8)),
        c < WIDTH,
    )
    # One validity byte per block8; four bytes cover 32 output tokens.
    # Zero the padded bytes so every packed int32 mask load is initialized.
    # Layer-local scales are gathered by the scorer's copy warps.
    visible_count = tl.minimum(
        tl.maximum(tl.minimum(visible, CONTEXT) - first, 0), 8
    ).to(tl.uint32)
    packed = tl.where(valid, (1 << visible_count) - 1, 0).to(tl.uint8)
    tl.store(
        M + b.to(tl.int64) * (tl.cdiv(WIDTH, 4) * 4) + c,
        packed,
        c < tl.cdiv(WIDTH, 4) * 4,
    )
