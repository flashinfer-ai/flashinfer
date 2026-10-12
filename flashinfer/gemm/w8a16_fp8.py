"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.gemm.w8a16_fp8 import (
    gen_w8a16_fp8_gemm_module,
    gen_w8a16_fp8_gemm_sm100_module,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op


@functools.cache
def _get_module_for_major(major: int):
    if major == 9:
        return gen_w8a16_fp8_gemm_module().build_and_load()
    if major == 10:
        return gen_w8a16_fp8_gemm_sm100_module().build_and_load()
    raise RuntimeError(
        f"w8a16_fp8_gemm needs Hopper (sm_90a) or Blackwell (sm_100a), got SM {major}x"
    )


def get_w8a16_fp8_gemm_module(device: Optional[torch.device] = None):
    """Kernel module for ``device`` (current device when omitted): Hopper or Blackwell build."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _get_module_for_major(get_compute_capability(device)[0])


@register_custom_op("flashinfer::w8a16_fp8_gemm", mutates_args=("out",))
def _w8a16_fp8_gemm_impl(
    out: torch.Tensor,
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_prepared: bool,
    workspace: Optional[torch.Tensor],
) -> None:
    get_w8a16_fp8_gemm_module(act.device).w8a16_fp8_gemm(
        out, act, weight, weight_scale, bias, weight_prepared, workspace
    )


@register_fake_op("flashinfer::w8a16_fp8_gemm")
def _w8a16_fp8_gemm_impl_fake(
    out: torch.Tensor,
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_prepared: bool,
    workspace: Optional[torch.Tensor],
) -> None:
    pass


@functools.cache
def _workspace_bytes(m: int, n: int, k: int, device_index: int) -> int:
    """Bytes of workspace this problem needs on this device (0 = none)."""
    return int(
        get_w8a16_fp8_gemm_module(
            torch.device("cuda", device_index)
        ).w8a16_fp8_gemm_workspace_bytes(m, n, k, device_index)
    )


def prepare_w8a16_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    r"""Return a copy of an FP8 ``[N, K]`` weight in the kernel's prepared layout.

    Inside every 16-wide block of K the columns are reordered to
    ``[2c, 2c+1, 2c+8, 2c+9]`` for ``c = 0..3``, which lets the kernel fetch a
    whole wgmma A fragment with one ``ldmatrix``.  Shape, dtype and the
    per-tensor scale are unchanged; the result is only valid with
    ``w8a16_fp8_gemm(..., weight_prepared=True)``.  Do this once at load time.
    """
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        raise ValueError("expected a 2-D float8_e4m3fn weight [N, K]")
    k = weight.shape[1]
    if k % 16 != 0:
        raise ValueError(f"K must be a multiple of 16, got {k}")
    base = torch.arange(0, k, 16, device=weight.device).repeat_interleave(16)
    within = torch.tensor(
        [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15], device=weight.device
    ).repeat(k // 16)
    perm = base + within
    return weight.index_select(1, perm).contiguous()


@flashinfer_api
def w8a16_fp8_gemm(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    weight_prepared: bool = False,
) -> torch.Tensor:
    r"""Linear layer with an FP8 e4m3 weight and a 16-bit activation (W8A16).

    ``out = act @ (weight.float() * weight_scale).T + bias``, computed with the
    weight dequantized inside the kernel; no 16-bit copy of the weight is
    materialised.  This is the fused form of TRT-LLM's
    ``W8A16FP8LinearMethod`` (``weight.to(dtype).mul_(scale)`` then
    ``F.linear``) and consumes the same checkpoint tensors: an FP8QDQ weight
    with one per-tensor scale.

    Parameters
    ----------
    act: torch.Tensor
        Activation, shape ``(..., K)``, ``float16`` or ``bfloat16``,
        contiguous in the last dimension.  Leading dimensions are flattened.
    weight: torch.Tensor
        Weight, shape ``(N, K)`` (``torch.nn.Linear`` layout),
        ``float8_e4m3fn``, contiguous.
    weight_scale: torch.Tensor
        One ``float32`` value (shape ``()`` or ``(1,)``) on ``act``'s device:
        the per-tensor dequantization scale.
    bias: Optional[torch.Tensor]
        Optional bias, shape ``(N,)``, same dtype as ``act``.
    out: Optional[torch.Tensor]
        Optional output, shape ``(..., N)``, same dtype as ``act``.  Allocated
        when omitted.
    weight_prepared: bool
        ``True`` when ``weight`` came from :func:`prepare_w8a16_fp8_weight`
        (opt-in ldmatrix fragment path).  ``False`` (default) takes the plain
        checkpoint layout, which is also the faster path on H100 / H200.

    Notes
    -----
    Hopper (``sm_90a``: wgmma + TMA) and Blackwell (``sm_100a``: tcgen05 + TMA,
    a FlashInfer fork of CUTLASS's mixed-input mainloop).  ``K`` must be a
    multiple of 64, ``N`` a multiple of 8, and ``act`` / ``weight`` / ``out``
    16-byte aligned (the launcher rejects anything else).  On Hopper the
    launcher picks the tile shape (128x256 or 128x128) and the split-K count
    from a wave-fill cost model; a split run goes through an fp32 workspace that
    this function allocates, otherwise one persistent block per SM walks the
    tiles.  On Blackwell it picks one of three 2SM tiles (256 weight rows by
    128, 192 or 256 activation rows) from the same kind of model; ``weight_prepared``
    is Hopper-only there.

    Returns
    -------
    out: torch.Tensor
        ``(..., N)`` in ``act``'s dtype.
    """
    if act.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"act must be float16 or bfloat16, got {act.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [N, K], got shape {tuple(weight.shape)}")
    k = act.shape[-1]
    n = weight.shape[0]
    if weight.shape[1] != k:
        raise ValueError(f"K mismatch: act[..., {k}] vs weight[{n}, {weight.shape[1]}]")
    if weight_scale.numel() != 1:
        raise ValueError("weight_scale is one per-tensor value")
    if weight_prepared and get_compute_capability(act.device)[0] != 9:
        raise ValueError("weight_prepared=True is a Hopper-only layout")
    if weight_scale.dtype != torch.float32:
        weight_scale = weight_scale.to(torch.float32)
    weight_scale = weight_scale.reshape(1)

    lead_shape = act.shape[:-1]
    act2d = act.reshape(-1, k)
    if out is None:
        out = torch.empty(*lead_shape, n, dtype=act.dtype, device=act.device)
    elif out.dtype != act.dtype or out.shape[-1] != n:
        raise ValueError(
            f"out must be {tuple(lead_shape) + (n,)} in {act.dtype}, "
            f"got {tuple(out.shape)} in {out.dtype}"
        )
    out2d = out.reshape(-1, n)
    # Under-filled grids run split-K through an fp32 workspace; the kernel
    # decides, this only sizes the buffer (0 bytes = single pass).
    ws_bytes = _workspace_bytes(act2d.shape[0], n, k, act.device.index or 0)
    workspace = (
        torch.empty(ws_bytes // 4, dtype=torch.float32, device=act.device)
        if ws_bytes > 0
        else None
    )
    _w8a16_fp8_gemm_impl(
        out2d, act2d, weight, weight_scale, bias, weight_prepared, workspace
    )
    return out
