# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Plan ownership and JIT dispatch for the single CuTe DSL decode backend."""

from dataclasses import dataclass
import functools
import torch
from ._utils import _overlaps


@functools.cache
def _compile(device_index, compressed_k, split):
    import cutlass
    import cutlass.cute as cute
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from ...cute_dsl.attention.dsa import hca_helpers
    from . import hca_v41, hca_v41_primitives

    sb, sq, sh, spw, spc, sws = (cute.sym_int() for _ in range(6))
    sd = cute.sym_int(divisibility=16)
    f = cute.runtime.make_fake_compact_tensor
    q = f(
        cutlass.BFloat16, (sb, sq, sh, sd), stride_order=(3, 2, 1, 0), assumed_align=16
    )
    w = f(cutlass.Uint8, (spw, 64 * 528), stride_order=(1, 0), assumed_align=16)
    c = f(cutlass.Uint8, (spc, 64 * 288), stride_order=(1, 0), assumed_align=16)
    wi = f(cutlass.Int32, (sb, 128), stride_order=(1, 0), assumed_align=16)
    ci = f(
        cutlass.Int32,
        (sb, max(64, compressed_k)),
        stride_order=(1, 0),
        assumed_align=16,
    )
    o = f(
        cutlass.BFloat16, (sb, sq, sh, sd), stride_order=(3, 2, 1, 0), assumed_align=16
    )
    lse = f(cutlass.Float32, (sb, sq, sh), stride_order=(2, 1, 0), assumed_align=16)
    ws = f(cutlass.Int8, (sws,), assumed_align=32) if split else None
    lengths = f(cutlass.Int32, (sb,), assumed_align=16)
    valid = f(cutlass.Int32, (sb,), assumed_align=16)
    sink = f(cutlass.Float32, (64,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = hca_v41.BlackwellV41MixedCacheDecode(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(64, 64),
        mma_pv_tiler_mn=(64, 128),
        max_active_clusters=1,
        page_size_cmp=64,
        skip_correction_threshold=0.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        is_causal=True,
        seq_len_q=1,
        hca_compress_ratio=128,
        window_page_size=64,
        compressed_page_size=64,
        compressed_is_fp4=True,
    )
    return build_and_load_cute_dsl_kernel(
        "deepseek_v41_decode",
        f"h64_s1_c{compressed_k}_split{int(split)}",
        lambda: cute.compile(
            kernel,
            q,
            w,
            c,
            wi,
            ci,
            o,
            lse,
            ws,
            cutlass.Int32(1),
            lengths,
            None,
            lengths,
            valid,
            cutlass.Float32(512**-0.5),
            cutlass.Float32(1.0),
            sink,
            stream,
            None,
            options="--enable-tvm-ffi --opt-level 2",
        ),
        extra_key_files=(
            __file__,
            hca_v41.__file__,
            hca_v41_primitives.__file__,
            hca_helpers.__file__,
        ),
    )


@dataclass(frozen=True)
class _Plan:
    signature: tuple
    kernel: object
    split: int
    out: torch.Tensor
    lse: torch.Tensor
    workspace: object
    lengths: torch.Tensor
    window_valid: torch.Tensor
    empty_cache: torch.Tensor
    empty_indices: torch.Tensor


def decode(q, swa_cache, global_cache, swa_indices, global_indices, sink, *, plan=None):
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (10, 0):
        raise ValueError("CuTe DS4.1 decode currently requires SM100")
    if (
        q.ndim != 4
        or q.shape[1:] != (1, 64, 512)
        or not 1 <= q.shape[0] <= 65535
        or q.dtype != torch.bfloat16
    ):
        raise ValueError("Q must be BF16 [B,1,64,512], with 1 <= B <= 65535")
    if (global_cache is None) != (global_indices is None):
        raise ValueError("global cache and indices must both be present or absent")
    inputs = tuple(
        x
        for x in (q, swa_cache, global_cache, swa_indices, global_indices, sink)
        if x is not None
    )
    if any(
        x.device != q.device or not x.is_contiguous() or x.data_ptr() % 16
        for x in inputs
    ):
        raise ValueError("inputs must be contiguous and 16-byte aligned on Q's device")
    for cache, width in ((swa_cache, 528), (global_cache, 288)):
        if cache is not None and (
            cache.ndim != 4
            or cache.shape[1:] != (64, 1, width)
            or cache.shape[0] < 1
            or cache.dtype != torch.uint8
        ):
            raise ValueError(
                "cache must be a nonempty uint8 page pool with page size 64"
            )
    batch = q.shape[0]
    if swa_indices.shape != (batch, 1, 128) or swa_indices.dtype != torch.int32:
        raise ValueError("SWA indices must be int32 [B,1,128]")
    if global_indices is not None and (
        global_indices.ndim != 3
        or global_indices.shape[:2] != (batch, 1)
        or not 64 <= global_indices.shape[2] <= 512
        or global_indices.shape[2] % 64
        or global_indices.dtype != torch.int32
    ):
        raise ValueError(
            "global indices must be int32 [B,1,K], K a multiple of64 up to512"
        )
    if sink.shape != (64,) or sink.dtype != torch.float32:
        raise ValueError("sink must be FP32 [64]")
    if torch.is_grad_enabled() and any(x.requires_grad for x in inputs):
        raise ValueError("DS4.1 decode is inference-only")
    signature = tuple(
        None if x is None else (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
        for x in (q, swa_cache, global_cache, swa_indices, global_indices, sink)
    )
    if plan is not None and (
        not isinstance(plan, _Plan) or plan.signature != signature
    ):
        raise ValueError("decode plan declaration mismatch")
    with torch.cuda.device(q.device):
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("prepare a decode plan before CUDA Graph capture")
            ck = 0 if global_indices is None else global_indices.shape[2]
            # Frozen small-batch schedule; tuning stays inside this kernel family.
            desired = (
                10
                if batch <= 4
                else 8
                if batch <= 16
                else 4
                if batch <= 32
                else 2
                if batch <= 64
                else 1
            )
            split = min(desired, (128 + ck) // 64)
            kernel = _compile(q.device.index, ck, split > 1)
            workspace = (
                torch.empty(
                    batch * 64 * split * 513 * 4, device=q.device, dtype=torch.int8
                )
                if split > 1
                else None
            )
            plan = _Plan(
                signature,
                kernel,
                split,
                torch.empty_like(q),
                torch.empty((batch, 64, 1), device=q.device, dtype=torch.float32),
                workspace,
                torch.full((batch,), 128 + ck, device=q.device, dtype=torch.int32),
                torch.full((batch,), 128, device=q.device, dtype=torch.int32),
                torch.empty((1, 64, 1, 288), device=q.device, dtype=torch.uint8),
                torch.full((batch, 1, 64), -1, device=q.device, dtype=torch.int32),
            )
        owned = (
            plan.out,
            plan.lse,
            plan.lengths,
            plan.window_valid,
            plan.empty_cache,
            plan.empty_indices,
        )
        if plan.workspace is not None:
            owned += (plan.workspace,)
        if any(_overlaps(dst, src) for dst in owned for src in inputs):
            raise ValueError("decode inputs must not overlap plan-owned buffers")
        if torch.is_grad_enabled() and any(x.requires_grad for x in owned):
            raise ValueError("DS4.1 decode is inference-only")
        from cutlass import Int32, Float32

        main = plan.empty_cache if global_cache is None else global_cache
        ids = plan.empty_indices if global_indices is None else global_indices
        plan.kernel(
            q,
            swa_cache.view(swa_cache.shape[0], -1),
            main.view(main.shape[0], -1),
            swa_indices.view(batch, 128),
            ids.view(batch, -1),
            plan.out,
            plan.lse.view(batch, 1, 64),
            plan.workspace,
            Int32(plan.split),
            plan.lengths,
            None,
            plan.lengths,
            plan.window_valid,
            Float32(512**-0.5),
            Float32(1.0),
            sink,
            None,
        )
    return plan.out, plan.lse, plan
