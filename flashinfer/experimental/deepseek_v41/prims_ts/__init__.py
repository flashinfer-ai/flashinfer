# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Host entry points and JIT dispatch for V4.1 Prims-TS decode.

Native attention configuration and ABI adaptation live in kernel.py; KV staging
and inline metadata views live in resources.py and _sparse_views.py. GPU modules
are imported only during compilation so workspace queries stay lightweight.
"""

import functools
import hashlib
from pathlib import Path

import torch

from ._workspace import bind_inputs, device_sm_count


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

    from ....cute_dsl.availability import require_cute_dsl_arch

    require_cute_dsl_arch(device_index)

    from ....attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import (
        kernel as native_kernel,
    )
    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from ....cute_dsl import fp4_common
    from .. import hca_v41_primitives
    from . import _sparse_views, _workspace, resources
    from . import kernel as adapter_kernel

    sm_count = device_sm_count(device_index)
    attention = adapter_kernel._make_fp8_attention(compressed_k, sm_count, native_fp4)
    decode_kernel = adapter_kernel.Fp8SwaDecodeKernel(attention)
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
    native_root = Path(native_kernel.__file__).resolve().parents[3]
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
            adapter_kernel.__file__,
            resources.__file__,
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
