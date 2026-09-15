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
    mask = align16(8 * batch * count)
    scales = 4 * batch * tr.cdiv(count * 8, 128) * 128
    return encoded, mask, scales


@functools.cache
def _compile_candidate(page, count, num_sms, arch):
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
    padded = tr.cdiv(width, 128) * 128
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
        cutlass.Boolean,
        (batch, width),
        stride=(width, 1),
        assumed_align=16,
    )
    sf = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (batch, padded),
        stride=(padded, 1),
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
            sf,
            options=f"--gpu-arch {arch} --enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "ds41_candidate_indexer",
        f"page{page}_c{count}_sms{num_sms}_{arch}",
        compile_fn,
        extra_key_files=(str(Path(__file__)), str(Path(native.__file__))),
    )


def candidate_scores(
    q, qs, kv, weights, lengths, table, candidates, out, context, workspace
):
    batch, count = candidates.shape
    page = kv.shape[1]
    if any(x.data_ptr() % 16 for x in (q, qs, kv, weights)):
        raise ValueError("cute_dsl requires 16-byte-aligned Q, scales, KV and weights")
    if kv.shape[0] * (page // 8) >= 2**31:
        raise ValueError("cute_dsl encoded page IDs must fit int32; use triton")
    if batch * out.stride(0) >= 2**31:
        raise ValueError(
            "cute_dsl padded output element span must fit int32; use triton"
        )
    encoded_bytes, mask_bytes, scale_bytes = workspace_size(batch, count)
    size = encoded_bytes + mask_bytes + scale_bytes
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
    mask = (
        workspace[encoded_bytes : encoded_bytes + batch * count * 8]
        .view(torch.bool)
        .view(batch, count * 8)
    )
    padded = tr.cdiv(count * 8, 128) * 128
    sf = (
        workspace[encoded_bytes + mask_bytes : size]
        .view(torch.int32)
        .view(batch, padded)
    )
    major, minor = torch.cuda.get_device_capability(q.device)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    compiled = _compile_candidate(page, count, sms, f"sm_{major}{minor}a")
    _candidate_metadata[(batch, tr.cdiv(count, 64))](
        candidates,
        table,
        lengths,
        encoded,
        mask,
        kv,
        sf,
        count,
        padded,
        table.shape[1],
        page,
        kv.stride(0),
        kv.shape[0],
        context,
        64,
        num_warps=4,
    )
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
        sf,
    )
    return out


@tr.jit
def _candidate_metadata(
    C,
    T,
    L,
    E,
    M,
    KV,
    SF,
    WIDTH: tl.constexpr,
    PAD: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE: tl.constexpr,
    PHYS: tl.constexpr,
    CONTEXT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    candidate = tl.load(C + b.to(tl.int64) * WIDTH + c, c < WIDTH, -1).to(tl.int64)
    first = candidate * 8
    visible = tl.load(L + b)
    valid = (c < WIDTH) & (first >= 0) & (first < CONTEXT) & (first < visible)
    page = tl.load(T + b.to(tl.int64) * PAGES + first // PAGE, valid, -1)
    valid &= (page >= 0) & (page < PHYS)
    encoded = page.to(tl.int64) * (PAGE // 8) + candidate % (PAGE // 8)
    tl.store(
        E + b.to(tl.int64) * WIDTH + c,
        tl.where(valid, encoded, PHYS * (PAGE // 8)),
        c < WIDTH,
    )
    lane = tl.arange(0, 8)
    token = first[:, None] + lane[None, :]
    valid_token = valid[:, None] & (token < visible) & (token < CONTEXT)
    tl.store(
        M + (b.to(tl.int64) * WIDTH + c[:, None]) * 8 + lane[None, :],
        valid_token,
        c[:, None] < WIDTH,
    )
    sf_ptr = KV + page.to(tl.int64)[:, None] * STRIDE + PAGE * 64 + (token % PAGE) * 4
    words = tl.load(sf_ptr.to(tl.pointer_type(tl.int32)), valid_token, 0x7F7F7F7F)
    tl.store(
        SF + b.to(tl.int64) * PAD + c[:, None] * 8 + lane[None, :],
        words,
        c[:, None] * 8 < PAD,
    )
