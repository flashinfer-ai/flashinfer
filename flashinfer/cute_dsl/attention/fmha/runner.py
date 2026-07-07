# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch runner for the trtllm CuTe DSL FMHA kernels (JIT-compiled).

JIT-compiles the trtllm BlackwellFusedMultiHead[BlockScaled]AttentionForward classes and exposes
PyTorch-friendly ragged/batched prefill entry points. This is the JIT runner invoked by
flashinfer.attention.cute_dsl.fmha
"""

import functools
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32

from .fmha import BlackwellFusedMultiHeadAttentionForward
from .helpers import fmha_helpers as fmha_utils

# torch dtype -> cutlass dtype
_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}

_LOG2_E = math.log2(math.e)

# Block-scaling mode -> (qk cutlass dtype, sf cutlass dtype, sf_vec_size)
_BLOCKSCALED_MODES = {
    "mxfp8": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
    "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
}


def _mask_type(is_causal: bool):
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
def _get_compiled_fmha(
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
    enable_ex2_emulation: bool,
):
    """Compile (and cache) the vendored FMHA kernel for a static config.

    Uses symbolic batch/seqlen dims (TVM-FFI ABI) so a single compile serves all
    shapes with the given dtypes / head counts / head_dim / mask.
    """
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
        scale_softmax * _LOG2_E,
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


def cute_dsl_fmha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    *,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    window_right: int = -1,
    lse: Optional[torch.Tensor] = None,
    attention_sinks: Optional[torch.Tensor] = None,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    scale_o: float = 1.0,
    max_qo_len: Optional[int] = None,
    max_kv_len: Optional[int] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool = False,
) -> None:
    """Ragged (varlen) prefill via the JIT-compiled vendored FMHA kernel."""
    total_q, H_q, D = q.shape
    total_kv, H_k, _ = k.shape
    D_v = v.shape[-1]
    h_r = H_q // H_k
    batch_size = len(qo_indptr) - 1

    use_skip = (
        skip_softmax_threshold_scale_factor is not None
        and skip_softmax_threshold_scale_factor > 0
    )
    kernel_fn = _get_compiled_fmha(
        q.dtype,
        v.dtype,
        o.dtype,
        H_q,
        H_k,
        D,
        D_v,
        is_causal,
        lse is not None,
        attention_sinks is not None,
        use_skip,
        enable_pdl,
        _ex2_emulation_enabled(q.device),
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scale_softmax = scale_q * scale_k * sm_scale
    scale_softmax_log2 = scale_softmax * _LOG2_E
    scale_output = scale_v / scale_o

    if max_qo_len is None:
        max_qo_len = int((qo_indptr[1:] - qo_indptr[:-1]).max().item())
    if max_kv_len is None:
        max_kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max().item())
    problem_size = (batch_size, max_qo_len, total_q, max_kv_len, H_q, H_k, D, D_v)

    skip_threshold_log2 = None
    if use_skip:
        skip_threshold_log2 = Float32(
            math.log2(skip_softmax_threshold_scale_factor / max_kv_len)
        )

    ws_left = None if window_left == -1 else Int32(window_left)
    ws_right = None if window_right == -1 else Int32(window_right)
    if is_causal and ws_right is None:
        ws_right = Int32(0)

    q_5d = q.view(1, total_q, H_k, h_r, D)
    k_5d = k.view(1, total_kv, H_k, 1, D)
    v_5d = v.view(1, total_kv, H_k, 1, D_v)
    assert o.data_ptr() % 32 == 0, "o must be 32-byte aligned (256-bit stores)"
    o_5d = o.view(1, total_q, H_k, h_r, D_v)
    lse_4d = lse.view(1, total_q, H_k, h_r) if lse is not None else None

    kernel_fn(
        q_5d,
        k_5d,
        v_5d,
        o_5d,
        problem_size,
        qo_indptr.to(torch.int32),
        kv_indptr.to(torch.int32),
        lse_4d,
        attention_sinks,
        Float32(scale_softmax_log2),
        Float32(scale_softmax),
        Float32(scale_output),
        skip_threshold_log2,
        ws_left,
        ws_right,
        None,  # skip_softmax_count
        None,  # total_softmax_count
        enable_pdl,
    )
