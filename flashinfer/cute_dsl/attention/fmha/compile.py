# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JIT-compilers for the trtllm CuTe DSL FMHA kernels.

``compile_cute_dsl_fmha_kernel`` / ``compile_cute_dsl_fmha_blockscaled_kernel``
``cute.compile`` (and cache) the trtllm ``BlackwellFusedMultiHead[BlockScaled]AttentionForward``
classes with symbolic batch/seqlen dims, returning a TVM-FFI-callable kernel handle.
"""

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32

# torch dtype -> cutlass dtype
_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}

# Block-scaling mode -> (qk cutlass dtype, sf cutlass dtype, sf_vec_size)
_BLOCKSCALED_MODES = {
    "mxfp8": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
    "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
}


def _mask_type(is_causal: bool):
    from .helpers import fmha_helpers as fmha_utils

    # Bottom-right aligned causal, matching FlashInfer convention.
    return (
        fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
        if is_causal
        else fmha_utils.MaskEnum.RESIDUAL_MASK
    )


def _ex2_emulation_enabled(device: torch.device) -> bool:
    # ex2 emulation is only beneficial / correct on CC 10.0 (sm_100/sm_100a/sm_100f);
    # not on sm_103 (CC 10.3) or later archs.
    return torch.cuda.get_device_capability(device) == (10, 0)


# =============================================================================
# Non-block-scaled
# =============================================================================


@functools.cache
def compile_cute_dsl_fmha_kernel(
    qk_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_v: int,
    is_causal: bool,
    with_lse: bool,
    enable_sink: bool,
    enable_skip_softmax: bool,
    use_pdl: bool,
    device: torch.device,
):
    """Compile (and cache) the trtllm FMHA kernel for a static config.

    Compiles with the TVM-FFI ABI (the handle takes ``torch.Tensor`` args on the env
    stream) and symbolic batch/seqlen dims, so a single compile serves all shapes for the
    given dtypes / head counts / head_dim / mask.
    """
    from .fmha import BlackwellFusedMultiHeadAttentionForward

    enable_ex2_emulation = _ex2_emulation_enabled(device)
    qk = _CUTLASS_DTYPE[qk_dtype]
    pv = _CUTLASS_DTYPE[v_dtype]
    out = _CUTLASS_DTYPE[out_dtype]
    d = head_dim
    dv = head_dim_v
    head_dim_arg = d if d == dv else (d, dv)

    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=(128, 128),
        head_dim=head_dim_arg,
        is_persistent=False,  # varlen ragged
        mask_type=_mask_type(is_causal),
        enable_ex2_emulation=enable_ex2_emulation,
        enable_skip_correction=True,
        use_tma_store=False,  # varlen -> STG
    )

    sym_b = cute.sym_int()
    sym_s_q = cute.sym_int()
    sym_s_k = cute.sym_int()
    sym_hk = cute.sym_int()
    sym_hr = cute.sym_int()
    sym_seq = cute.sym_int()
    q_fake = make_fake_compact_tensor(
        qk,
        (sym_b, sym_s_q, sym_hk, sym_hr, d),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    k_fake = make_fake_compact_tensor(
        qk,
        (sym_b, sym_s_k, sym_hk, 1, d),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    v_fake = make_fake_compact_tensor(
        pv,
        (sym_b, sym_s_k, sym_hk, 1, dv),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    o_fake = make_fake_compact_tensor(
        out,
        (sym_b, sym_s_q, sym_hk, sym_hr, dv),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=32,
    )
    lse_fake = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_b, sym_s_q, sym_hk, sym_hr),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        if with_lse
        else None
    )
    sink_fake = (
        make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(),), stride_order=(0,), assumed_align=16
        )
        if enable_sink
        else None
    )
    cum_q_fake = make_fake_compact_tensor(Int32, (sym_seq,), assumed_align=16)
    cum_k_fake = make_fake_compact_tensor(Int32, (sym_seq,), assumed_align=16)

    problem_size = (1, 1, 1, 1, num_qo_heads, num_kv_heads, d, dv)
    scale_softmax = 1.0 / math.sqrt(d)
    skip_threshold_log2 = Float32(0.0) if enable_skip_softmax else None
    ws_right = Int32(0) if is_causal else None
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        q_fake,
        k_fake,
        v_fake,
        o_fake,
        problem_size,
        cum_q_fake,
        cum_k_fake,
        lse_fake,
        sink_fake,
        scale_softmax * math.log2(math.e),
        scale_softmax,
        1.0,
        skip_threshold_log2,
        None,
        ws_right,
        None,
        None,
        stream_fake,
        use_pdl,
        options="--enable-tvm-ffi --opt-level 2",
    )


# =============================================================================
# Block-scaled (MXFP8 / NVFP4)
# =============================================================================


@functools.cache
def compile_cute_dsl_fmha_blockscaled_kernel(
    qk_mode: str,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    with_lse: bool,
    enable_skip_softmax: bool,
    use_pdl: bool,
    device: torch.device,
):
    """Compile (and cache) the trtllm block-scaled FMHA kernel (batched, non-varlen)."""
    from .fmha_blockscaled import BlackwellFusedMultiHeadBlockScaledAttentionForward

    enable_ex2_emulation = _ex2_emulation_enabled(device)
    qk, sf_dt, sf_vec = _BLOCKSCALED_MODES[qk_mode]
    out = _CUTLASS_DTYPE[out_dtype]
    d = dv = head_dim

    fmha = BlackwellFusedMultiHeadBlockScaledAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=(128, 128),
        head_dim=d,
        is_persistent=False,
        mask_type=_mask_type(is_causal),
        enable_ex2_emulation=enable_ex2_emulation,
        enable_skip_correction=True,
        qk_sf_vec_size=sf_vec,
        use_tma_store=True,  # non-varlen
    )

    sym_b = cute.sym_int()
    sym_s_q = cute.sym_int()
    sym_s_k = cute.sym_int()
    sym_hk = cute.sym_int()
    sym_hr = cute.sym_int()
    # Sub-byte (fp4) stride-1 dim must be static.
    sym_d = d if qk.width < 8 else cute.sym_int()
    sym_dv = dv if qk.width < 8 else cute.sym_int()
    q_fake = make_fake_compact_tensor(
        qk,
        (sym_b, sym_s_q, sym_hk, sym_hr, sym_d),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    k_fake = make_fake_compact_tensor(
        qk,
        (sym_b, sym_s_k, sym_hk, 1, sym_d),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    v_fake = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_b, sym_s_k, sym_hk, 1, sym_dv),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    o_fake = make_fake_compact_tensor(
        out,
        (sym_b, sym_s_q, sym_hk, sym_hr, sym_dv),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=32,
    )
    # SF: the kernel reconstructs the blocked layout from q.shape via
    # tile_atom_to_shape_SF and only reads the SF pointer, so a flat 1D tensor
    # (the fused quantizer's SF, flattened) is sufficient.
    q_sf_fake = make_fake_compact_tensor(sf_dt, (cute.sym_int(),), assumed_align=16)
    k_sf_fake = make_fake_compact_tensor(sf_dt, (cute.sym_int(),), assumed_align=16)
    lse_fake = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_b, sym_s_q, sym_hk, sym_hr),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        if with_lse
        else None
    )

    problem_size = (1, 1, 1, 1, num_qo_heads, num_kv_heads, d, dv)
    scale_softmax = 1.0 / math.sqrt(d)
    skip_threshold_log2 = Float32(0.0) if enable_skip_softmax else None
    ws_right = Int32(0) if is_causal else None
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        q_fake,
        k_fake,
        q_sf_fake,
        k_sf_fake,
        v_fake,
        o_fake,
        problem_size,
        None,  # cum_seqlen_q (non-varlen)
        None,  # cum_seqlen_k
        lse_fake,
        None,  # sink
        scale_softmax * math.log2(math.e),
        scale_softmax,
        1.0,
        None,  # scale_v_channels
        skip_threshold_log2,
        None,
        ws_right,
        None,
        None,
        stream_fake,
        use_pdl,
        options="--enable-tvm-ffi --opt-level 2",
    )
