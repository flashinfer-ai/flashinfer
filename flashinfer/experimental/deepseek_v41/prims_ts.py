# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Joint shared-unit FP8 SWA/main attention for the V4.1 adapter."""

import functools
import hashlib
from pathlib import Path

import torch

from ._workspace import bind_inputs, device_sm_count


def _make_fp8_attention(compressed_k, sm_count, native_fp4):
    from ...attention.prims_ts.kernels.mla_decode.sparse_policy import SparseMlaProfile
    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
    )
    from .prims_ts_resources import MixedFp8KvProducer

    # A fixed SWA HEAD precedes ceil(K/128) main LOOP tiles, including when
    # every SWA entry is invalid. One online softmax/PV state spans both pools.
    capacity = max(256, 128 + ((compressed_k + 127) // 128) * 128)
    attention = ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=64,
        seq_len_q=1,
        seq_len_k=capacity,
        latent_dim=512,
        rope_dim=0,
        page_size=1,
        max_active_clusters=sm_count,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=MlaProfile(
            name="dsv41_shared_pertensor_joint",
            kernel_variant="keeps_mma_ab",
            tile_size_q=64,
            use_persistent_scheduler=0,
        ),
        logical_num_heads=64,
        logical_seq_len_q=1,
        tile_size_q=64,
        explicit_split_kv=1,
        explicit_persistent=False,
        reduction_mode="gmem_separate",
        mask_type="dense",
        sparse_profile=SparseMlaProfile(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            scheduler="nonpersistent",
            head_dim_ctas=1,
            gather_issue_warps=4,
            reuse_kv=True,
            reuse_kv_stages=8,
            kv_tile_size=128,
            page_pipeline_stages=1,
        ),
        finalize_output=True,
        kv_producer=MixedFp8KvProducer(native_fp4=native_fp4),
    )
    # Static here means host scalars, not device scales or value-specialized
    # binaries; both BMM factors remain runtime Float32.
    attention.direct_sparse = True
    attention.direct_static_scales = True
    attention.direct_split_softmax_scale = True
    attention.direct_sparse_capacities = (128, compressed_k)
    return attention


@functools.cache
def _compile(device_index, compressed_k, *, has_sink=False, native_fp4=None):
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("warm up the decode specialization before CUDA Graph capture")

    import cutlass
    import cutlass.cute as cute
    from packaging.version import Version

    if Version(cutlass.__version__) < Version("4.8.0"):
        raise RuntimeError("V4.1 prims-ts requires CuTe-DSL >= 4.8.0")
    compiler = (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor)
    if compiler[0] < 13:
        raise RuntimeError("V4.1 prims-ts requires the CuTe-DSL CUDA 13 compiler")
    if native_fp4 is None:
        native_fp4 = compiler >= (13, 2)
    if native_fp4 and compiler < (13, 2):
        raise RuntimeError("native FP4 conversion requires bundled CUDA >= 13.2")

    from ...cute_dsl.availability import require_cute_dsl_arch

    require_cute_dsl_arch(device_index)

    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import kernel
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from ...cute_dsl import fp4_common
    from . import (
        _sparse_views,
        _workspace,
        hca_v41_primitives,
        prims_ts_fp8,
        prims_ts_resources,
    )

    sm_count = device_sm_count(device_index)
    attention = _make_fp8_attention(compressed_k, sm_count, native_fp4)
    decode_kernel = prims_ts_fp8.Fp8SwaDecodeKernel(attention)
    batch = cute.sym_int()
    f = cute.runtime.make_fake_compact_tensor

    def query(dtype):
        return cute.runtime.make_fake_tensor(
            dtype,
            (64, 512, 1, batch),
            (512, 1, 64 * 512, 64 * 512),
            assumed_align=16,
        )

    q = query(cutlass.Float8E4M3FN)
    out = query(cutlass.BFloat16)
    swa = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, 512, cute.sym_int64(divisibility=64)),
        (512, 1, 512),
        assumed_align=16,
    )
    main_cache = f(
        cutlass.Uint8,
        (cute.sym_int64(divisibility=64 * 288),),
        assumed_align=16,
    )

    def indices(width):
        return f(cutlass.Int32, (batch, width), stride_order=(1, 0), assumed_align=16)

    def lengths():
        return f(cutlass.Int32, (batch,), assumed_align=16)

    combined = indices(128 + compressed_k)
    topk_lens, seq_lens = lengths(), lengths()
    lse = f(cutlass.Float32, (batch, 64), stride_order=(1, 0), assumed_align=16)
    sinks = f(cutlass.Float32, (64,), assumed_align=16) if has_sink else None
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    signature = (
        "dsv41_shared_pertensor_joint_e4m3_swa512_fp4_inline_metadata_v2",
        attention.compile_signature(),
        has_sink,
        "public_o_bf16_internal_lse_ln_sink_exclusive",
        "runtime_bmm1_bmm2_f32",
        cutlass.__version__,
        compiler,
    )
    key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    native_root = Path(kernel.__file__).resolve().parents[3]
    return build_and_load_cute_dsl_kernel(
        "deepseek_v41_prims_ts_fp8",
        f"h64_d512_swa128_c{compressed_k}_keep_bk128_kv8_split1_i64"
        f"_shared_pertensor_joint_sink{int(has_sink)}"
        f"_nativefp4{int(native_fp4)}_sm{sm_count}_{key}",
        lambda: cute.compile[cute.FrontendNext](
            decode_kernel,
            q,
            swa,
            main_cache,
            combined,
            topk_lens,
            seq_lens,
            sinks,
            out,
            lse,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        ),
        extra_key_files=(
            __file__,
            _workspace.__file__,
            _sparse_views.__file__,
            prims_ts_fp8.__file__,
            prims_ts_resources.__file__,
            hca_v41_primitives.__file__,
            fp4_common.__file__,
            *sorted(native_root.rglob("*.py")),
        ),
    )


def decode_mixed(
    query,
    swa_kv_cache,
    workspace_buffer,
    sparse_indices=None,
    compressed_kv_cache=None,
    sparse_topk_lens=None,
    seq_lens=None,
    out=None,
    bmm1_scale=1.0,
    bmm2_scale=1.0,
    sinks=None,
    kv_layout="HND",
):
    if query.dtype != torch.float8_e4m3fn:
        raise ValueError("Prims-TS decode requires E4M3 Q")
    bound = bind_inputs(
        query,
        swa_kv_cache,
        workspace_buffer,
        sparse_indices,
        compressed_kv_cache,
        sparse_topk_lens,
        seq_lens,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        kv_layout,
        backend="prims-ts",
    )
    views = bound.views
    with torch.cuda.device(query.device):
        kernel = _compile(
            query.device.index, bound.compressed_k, has_sink=bound.sinks is not None
        )
        kernel(
            query.permute(2, 3, 1, 0),
            bound.swa_cache.view(-1, 1, 512).permute(1, 2, 0),
            bound.main_cache.view(-1),
            bound.indices,
            bound.topk_lens,
            bound.seq_lens,
            bound.sinks,
            bound.out.permute(2, 3, 1, 0),
            views.lse.view(bound.batch, 64),
            bound.bmm1_scale,
            bound.bmm2_scale,
        )
    return bound.out


def decode_window(
    query,
    swa_kv_cache,
    workspace_buffer,
    sparse_indices=None,
    sparse_topk_lens=None,
    seq_lens=None,
    out=None,
    bmm1_scale=1.0,
    bmm2_scale=1.0,
    sinks=None,
    kv_layout="HND",
):
    return decode_mixed(
        query,
        swa_kv_cache,
        workspace_buffer,
        sparse_indices,
        None,
        sparse_topk_lens,
        seq_lens,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        kv_layout,
    )
