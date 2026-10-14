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
"""Block-scale Q/K quantization for the trtllm block-scaled FMHA kernel.

Calls FlashInfer's fused ``mxfp8_quantize`` / ``fp4_quantize`` kernels: transpose to
(B, H, S, D), pad S to a multiple of 128 (the SF atom row size), quantize as a 2D
``[B*H*S_pad, D]`` matrix, then unpad the *data* back to S. The scale-factor tensor is
kept at S_pad and passed to the kernel as-is in the swizzled (128x4) layout, which
matches what the kernel's ``tile_atom_to_shape_SF`` consumes. NVFP4's per-tensor global
scale is returned to be folded into ``sm_scale``.
"""

from typing import Tuple

import torch

from flashinfer.quantization import fp4_quantize, mxfp8_quantize

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_SF_VEC = {"mxfp8": 32, "nvfp4": 16}


def _quantize_one(
    x_bshd: torch.Tensor, qk_mode: str, quant_backend: str
) -> Tuple[torch.Tensor, torch.Tensor, float | torch.Tensor]:
    """Quantize one (B, S, H, D) bf16/fp16 tensor -> (store, sf, scale)."""
    b, s, h, d = x_bshd.shape
    # (B, H, S, D): num_heads is batch-like for the GEMM-style quantizer.
    x_bhsd = x_bshd.transpose(1, 2).contiguous()
    s_pad = ((s + 127) // 128) * 128
    if s_pad != s:
        pad = torch.zeros(b, h, s_pad - s, d, dtype=x_bhsd.dtype, device=x_bhsd.device)
        x_bhsd = torch.cat([x_bhsd, pad], dim=2)
    x_2d = x_bhsd.reshape(b * h * s_pad, d)

    if qk_mode == "mxfp8":
        data, sf = mxfp8_quantize(
            x_2d, is_sf_swizzled_layout=True, alignment=32, backend=quant_backend
        )
        x_q = data.view(b, h, s_pad, d)[:, :, :s, :].transpose(1, 2).contiguous()
        # E8M0 scale factors (op returns uint8 storage); kernel reads only the SF pointer.
        return x_q, sf.view(torch.float8_e8m0fnu), 1.0  # per-tensor scale is trivial

    if qk_mode == "nvfp4":
        amax = x_2d.float().abs().amax().clamp(min=1e-6)
        global_sf = (_FP8_E4M3_MAX * _FP4_E2M1_MAX) / amax
        data, sf = fp4_quantize(
            x_2d,
            global_sf.to(torch.float32).reshape(1),
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
            backend=quant_backend,
        )
        x_q = (
            data.view(torch.float4_e2m1fn_x2)
            .view(b, h, s_pad, d // 2)[:, :, :s, :]
            .transpose(1, 2)
            .contiguous()
        )
        # Keep the dequant scale as a 0-d tensor (no ``.item()`` -> torch.compile safe);
        # the runner converts it to a Python float at the eager call boundary.
        return x_q, sf.view(torch.float8_e4m3fn), amax / (_FP8_E4M3_MAX * _FP4_E2M1_MAX)

    raise ValueError(f"qk_mode must be one of {tuple(_SF_VEC)}, got {qk_mode!r}")


def quantize_blockscaled_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    qk_mode: str,
    quant_backend: str = "cute-dsl",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float | torch.Tensor,
    float | torch.Tensor,
]:
    """Quantize Q/K ``(B, S, H, D)`` to the block-scaled format the FMHA kernel needs.

    ``quant_backend`` selects the fused quantizer backend (``"cuda"`` stable, or
    ``"cute-dsl"`` for nvcc-free environments). Returns
    ``(q_store, k_store, q_sf, k_sf, q_scale, k_scale)``: the quantized Q/K, the padded
    SF tensors, and the per-tensor dequant scales (fold ``q_scale * k_scale`` into
    ``sm_scale``; both are 1.0 for MXFP8).
    """
    if qk_mode not in _SF_VEC:
        raise ValueError(f"qk_mode must be one of {tuple(_SF_VEC)}, got {qk_mode!r}")
    q_store, q_sf, q_scale = _quantize_one(q, qk_mode, quant_backend)
    k_store, k_sf, k_scale = _quantize_one(k, qk_mode, quant_backend)
    return q_store, k_store, q_sf, k_sf, q_scale, k_scale
