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
Block-scaled CuTe DSL FMHA entry
================================

PyTorch-friendly wrapper for the trtllm block-scaled (MXFP8 / NVFP4) FMHA kernel.
Published cubins are preferred, with JIT compilation as a fallback for variants that
are not in the artifact matrix. Pre-quantized Q/K + scale-factor tensors are produced
by ``flashinfer.cute_dsl.attention.fmha.quantize.quantize_blockscaled_qk``.
"""

import functools
import logging
import math
import os
from typing import Optional

from cutlass.cute.typing import Float32, Int32

import torch

from flashinfer.attention.cute_dsl.fmha import (
    _dtype_to_str,
    _get_gpu_arch,
    _load_from_artifact,
    _load_from_local,
)
from flashinfer.cute_dsl.attention.fmha.compile import (
    _BLOCKSCALED_MODES,
    compile_cute_dsl_fmha_blockscaled_kernel,
)

logger = logging.getLogger("flashinfer.attention.cute_dsl.fmha_blockscaled")


def _reshape_swizzled_sf(
    sf: torch.Tensor,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    sf_vec_size: int,
) -> torch.Tensor:
    """Expose a flat 128x4-swizzled SF buffer through the kernel's 6D ABI."""
    seq_tiles = (seq_len + 127) // 128
    sf_k_tiles = (head_dim // sf_vec_size + 3) // 4
    expected_numel = batch_size * num_heads * seq_tiles * sf_k_tiles * 32 * 4 * 4
    if sf.numel() != expected_numel:
        raise ValueError(
            f"scale-factor tensor has {sf.numel()} elements, expected "
            f"{expected_numel} for shape (B={batch_size}, S={seq_len}, "
            f"H={num_heads}, D={head_dim}) and sf_vec_size={sf_vec_size}"
        )
    return sf.reshape(batch_size * num_heads, seq_tiles, sf_k_tiles, 32, 4, 4)


def _get_variant_name(
    qk_mode: str,
    pv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    is_persistent: bool,
    varlen: bool = False,
    with_lse: bool = False,
    enable_tvm_ffi: bool = False,
    enable_skip_softmax: bool = False,
    enable_sink: bool = False,
    use_pdl: bool = False,
) -> str:
    """Generate the block-scaled variant name used by the artifact compiler."""
    pv_str = _dtype_to_str(pv_dtype)
    out_str = _dtype_to_str(out_dtype)
    causal_str = "causal" if is_causal else "nocausal"
    persist_str = "persistent" if is_persistent else "nonpersistent"
    varlen_str = "_varlen" if varlen else ""
    lse_str = "_lse" if with_lse else ""
    skip_str = "_skipsm" if enable_skip_softmax else ""
    sink_str = "_sink" if enable_sink else ""
    pdl_str = "_pdl" if use_pdl else ""
    ffi_str = "_tvmffi" if enable_tvm_ffi else ""
    return f"cute_dsl_fmha_blockscaled_{qk_mode}_{pv_str}_{out_str}_h{head_dim}_{causal_str}_{persist_str}{varlen_str}{lse_str}{skip_str}{sink_str}{pdl_str}{ffi_str}"


@functools.cache
def get_cute_dsl_fmha_blockscaled_kernel(
    gpu_arch: str,
    qk_mode: str,
    pv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    is_persistent: bool = False,
    enable_tvm_ffi: bool = False,
    varlen: bool = False,
    with_lse: bool = False,
    enable_skip_softmax: bool = False,
    enable_sink: bool = False,
    use_pdl: bool = False,
):
    """Load a compiled block-scaled FMHA kernel from local or remote artifacts."""
    if qk_mode not in _BLOCKSCALED_MODES:
        raise ValueError(
            f"qk_mode must be one of {tuple(_BLOCKSCALED_MODES)}, got {qk_mode!r}"
        )

    variant_name = _get_variant_name(
        qk_mode,
        pv_dtype,
        out_dtype,
        head_dim,
        is_causal,
        is_persistent,
        varlen=varlen,
        with_lse=with_lse,
        enable_tvm_ffi=enable_tvm_ffi,
        enable_skip_softmax=enable_skip_softmax,
        enable_sink=enable_sink,
        use_pdl=use_pdl,
    )

    # Check for local .so directory (development mode)
    local_dir = os.environ.get("FLASHINFER_DSL_FMHA_LOCAL_DIR")
    if local_dir:
        logger.info(
            f"Loading block-scaled DSL FMHA kernel from local dir: {local_dir} (tvm_ffi={enable_tvm_ffi})"
        )
        return _load_from_local(variant_name, local_dir, enable_tvm_ffi=enable_tvm_ffi)

    logger.info(
        f"Loading block-scaled DSL FMHA kernel variant: {variant_name} (tvm_ffi={enable_tvm_ffi})"
    )
    return _load_from_artifact(variant_name, gpu_arch, enable_tvm_ffi=enable_tvm_ffi)


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
    """Batched (non-varlen) block-scaled prefill via the trtllm kernel.

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
    batch_size, s_q, H_q, D = q.shape
    _, s_k, H_k, _ = k.shape
    D_v = v.shape[-1]
    num_store_bits = q.element_size() * 8
    num_dtype_bits = _BLOCKSCALED_MODES[qk_mode][0].width
    D *= num_store_bits // num_dtype_bits
    h_r = H_q // H_k

    use_skip_softmax = (
        skip_softmax_threshold_scale_factor is not None
        and skip_softmax_threshold_scale_factor > 0
    )
    try:
        if window_left != -1 or window_right != -1:
            # The artifact name has no window axis, so only its traced
            # no-window/causal-right-bound signatures are safe to load.
            raise RuntimeError(
                "windowed block-scaled DSL FMHA variants are not exported"
            )
        kernel_fn = get_cute_dsl_fmha_blockscaled_kernel(
            _get_gpu_arch(q.device),
            qk_mode,
            v.dtype,
            o.dtype,
            D,
            is_causal,
            is_persistent=False,
            enable_tvm_ffi=True,
            with_lse=lse is not None,
            enable_skip_softmax=use_skip_softmax,
            use_pdl=enable_pdl,
        )
    except (RuntimeError, FileNotFoundError) as e:
        logger.info(
            f"Block-scaled DSL FMHA cubin unavailable ({e}); JIT-compiling the kernel."
        )
        kernel_fn = compile_cute_dsl_fmha_blockscaled_kernel(
            qk_mode,
            v.dtype,
            o.dtype,
            H_q,
            H_k,
            D,
            is_causal,
            lse is not None,
            use_skip_softmax,
            enable_pdl,
            q.device,
            has_window_left=window_left != -1,
            has_window_right=is_causal or window_right != -1,
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
    if use_skip_softmax:
        skip_threshold_log2 = Float32(
            math.log2(skip_softmax_threshold_scale_factor / s_k)
        )

    ws_left = None if window_left == -1 else Int32(window_left)
    ws_right = None if window_right == -1 else Int32(window_right)
    if is_causal and ws_right is None:
        ws_right = Int32(0)

    q_5d = q.reshape(batch_size, s_q, H_k, h_r, q.shape[-1])
    k_5d = k.reshape(batch_size, s_k, H_k, 1, k.shape[-1])
    sf_vec_size = _BLOCKSCALED_MODES[qk_mode][2]
    q_sf_6d = _reshape_swizzled_sf(q_sf, batch_size, H_q, s_q, D, sf_vec_size)
    k_sf_6d = _reshape_swizzled_sf(k_sf, batch_size, H_k, s_k, D, sf_vec_size)
    v_5d = v.reshape(batch_size, s_k, H_k, 1, D_v)
    assert o.data_ptr() % 32 == 0, "o must be 32-byte aligned (256-bit stores)"
    o_5d = o.reshape(batch_size, s_q, H_k, h_r, D_v)
    lse_4d = lse.reshape(batch_size, s_q, H_k, h_r) if lse is not None else None

    kernel_fn(
        q_5d,
        k_5d,
        q_sf_6d,
        k_sf_6d,
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
