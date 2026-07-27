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
"""
Block-scaled CuTe DSL FMHA entry (JIT)
======================================

PyTorch-friendly wrapper for the trtllm block-scaled (MXFP8 / NVFP4) FMHA kernel.
Unlike :mod:`flashinfer.attention.cute_dsl.fmha`, there is no published cubin for the
block-scaled kernel, so this path is always JIT-compiled: the kernel handle comes from
``flashinfer.cute_dsl.attention.fmha.compile.compile_cute_dsl_fmha_blockscaled_kernel``
and this module only does the tensor marshalling. Pre-quantized Q/K + scale-factor tensors
are produced by ``flashinfer.cute_dsl.attention.fmha.quantize.quantize_blockscaled_qk``.
"""

import math
from typing import Optional

from cutlass.cute.typing import Float32, Int32

import torch

from flashinfer.cute_dsl.attention.fmha.compile import (
    _BLOCKSCALED_MODES,
    compile_cute_dsl_fmha_blockscaled_kernel,
)


def cute_dsl_fmha_blockscaled_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    q_sf: torch.Tensor,
    k_sf: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    *,
    qk_mode: str,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    window_right: int = -1,
    lse: Optional[torch.Tensor] = None,
    scale_q: float | torch.Tensor = 1.0,
    scale_k: float | torch.Tensor = 1.0,
    scale_v: float | torch.Tensor = 1.0,
    scale_o: float | torch.Tensor = 1.0,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool = False,
) -> None:
    """Batched (non-varlen) block-scaled prefill via the JIT-compiled trtllm kernel.

    Inputs are batched:
    - q (b, s_q, H_q, D), q_sf (quantized block-scaled format)
    - k (b, s_k, H_k, D), k_sf (quantized block-scaled format)
    - v/o: (b, s, H, D_v)

    The per-tensor scales accept a Python float or a 0-d tensor (as returned by the
    quantizer); tensors are converted to floats here at the eager boundary.
    """
    if qk_mode not in _BLOCKSCALED_MODES:
        raise ValueError(
            f"qk_mode must be one of {tuple(_BLOCKSCALED_MODES)}, got {qk_mode!r}"
        )
    batch_size, s_q, H_q, _ = q.shape
    _, s_k, H_k, _ = k.shape
    D = D_v = v.shape[-1]
    h_r = H_q // H_k

    use_skip = (
        skip_softmax_threshold_scale_factor is not None
        and skip_softmax_threshold_scale_factor > 0
    )
    kernel_fn = compile_cute_dsl_fmha_blockscaled_kernel(
        qk_mode,
        o.dtype,
        H_q,
        H_k,
        D,
        is_causal,
        lse is not None,
        use_skip,
        enable_pdl,
        q.device,
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    # The block-scale quantizer returns scales as 0-d tensors (no ``.item()`` sync, so it
    # stays torch.compile-friendly); materialize to floats here at the eager boundary.
    scale_q, scale_k, scale_v, scale_o = map(
        lambda x: x.item() if isinstance(x, torch.Tensor) else x,
        (scale_q, scale_k, scale_v, scale_o),
    )
    scale_softmax = scale_q * scale_k * sm_scale
    scale_softmax_log2 = scale_softmax * math.log2(math.e)
    scale_output = scale_v / scale_o
    problem_size = (batch_size, s_q, s_q, s_k, H_q, H_k, D, D_v)

    skip_threshold_log2 = None
    if use_skip:
        skip_threshold_log2 = Float32(
            math.log2(skip_softmax_threshold_scale_factor / s_k)
        )

    ws_left = None if window_left == -1 else Int32(window_left)
    ws_right = None if window_right == -1 else Int32(window_right)
    if is_causal and ws_right is None:
        ws_right = Int32(0)

    q_5d = q.reshape(batch_size, s_q, H_k, h_r, q.shape[-1])
    k_5d = k.reshape(batch_size, s_k, H_k, 1, k.shape[-1])
    v_5d = v.reshape(batch_size, s_k, H_k, 1, D_v)
    assert o.data_ptr() % 32 == 0, "o must be 32-byte aligned (256-bit stores)"
    o_5d = o.reshape(batch_size, s_q, H_k, h_r, D_v)
    lse_4d = lse.reshape(batch_size, s_q, H_k, h_r) if lse is not None else None

    kernel_fn(
        q_5d,
        k_5d,
        q_sf.reshape(-1),
        k_sf.reshape(-1),
        v_5d,
        o_5d,
        problem_size,
        None,  # cum_seqlen_q
        None,  # cum_seqlen_k
        lse_4d,
        None,  # attention_sinks
        Float32(scale_softmax_log2),
        Float32(scale_softmax),
        Float32(scale_output),
        None,  # scale_v_channels
        skip_threshold_log2,
        ws_left,
        ws_right,
        None,  # skip_softmax_count
        None,  # total_softmax_count
        enable_pdl,
    )
