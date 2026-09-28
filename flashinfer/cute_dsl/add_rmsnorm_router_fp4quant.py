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
"""Private experimental router epilogue; not a supported serving API yet."""

import torch

from .add_rmsnorm_fp4quant import _add_rmsnorm_fp4quant_impl


def _experimental_add_rmsnorm_router_fp4quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    router_bias: torch.Tensor | None = None,
    global_scale: torch.Tensor,
    output_hidden_size: int | None = None,
    eps: float = 1e-6,
    y_fp4: torch.Tensor | None = None,
    block_scale: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    y_out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse add, RMSNorm, NVFP4, tail zeroing and router projection in one CTA.

    ``input`` and in-place ``residual`` are contiguous FP16/BF16 ``[M, H]``.
    ``weight`` is ``[H]``, ``router_weight`` is ``[E, H]`` and optional bias is
    ``[E]``, all in the input dtype. The returned logits are ``[M, E]`` in that
    dtype, computed from the *rounded normalized* values, not FP4 values.
    Optional ``y_out`` materializes those same values with logical width H.

    Packed FP4 and E4M3 scales have physical width ``output_hidden_size`` (K),
    default H: shapes ``[M, K/2]`` and ``[M, K/16]``. The tail is zeroed.
    Residual, normalization and quantization follow ``add_rmsnorm_fp4quant``.
    A device FP32 ``global_scale[1]`` is required to avoid a setup fill launch.
    All buffers must be on the input device, contiguous and non-overlapping.
    Only single-CTA NVFP4 with row-major scales is supported.

    This SIMT FP32 router reduction is an experimental correctness/performance
    vehicle, not a tensor-core GEMM replacement. It does not promise bitwise
    equality with cuBLAS/TRT-LLM router GEMMs or preservation of near-tied
    routing choices. No model/backend selects it automatically. Trace/replay
    registration and a stable public API are deferred until that contract and
    the performance crossover are established.
    """
    if input.ndim != 2 or input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("input must be a two-dimensional FP16/BF16 tensor")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    m, h = input.shape
    if h < 64 or h % 16:
        raise ValueError("hidden_size must be >= 64 and divisible by 16")
    if (
        router_weight.ndim != 2
        or router_weight.shape[1] != h
        or router_weight.shape[0] < 1
    ):
        raise ValueError("router_weight must have shape [num_experts, hidden_size]")
    e = router_weight.shape[0]
    k = h if output_hidden_size is None else output_hidden_size
    if not isinstance(k, int) or isinstance(k, bool) or k < h or k % 16:
        raise ValueError(
            "output_hidden_size must be an integer >= hidden_size divisible by 16"
        )

    def check(name, tensor, shape, dtype):
        if tensor is not None and (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != input.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous {dtype} with shape {shape} on the input device"
            )

    check("input", input, (m, h), input.dtype)
    check("residual", residual, (m, h), input.dtype)
    check("weight", weight, (h,), input.dtype)
    check("router_weight", router_weight, (e, h), input.dtype)
    check("router_bias", router_bias, (e,), input.dtype)
    check("global_scale", global_scale, (1,), torch.float32)
    check("y_fp4", y_fp4, (m, k // 2), torch.float4_e2m1fn_x2)
    check("block_scale", block_scale, (m, k // 16), torch.float8_e4m3fn)
    check("router_logits", router_logits, (m, e), input.dtype)
    check("y_out", y_out, (m, h), input.dtype)
    if global_scale is None:
        raise ValueError("global_scale must be a device FP32 tensor of shape [1]")
    # Reject overlapping storage before residual is modified. Read/read aliasing
    # is harmless, but any overlap involving an output can race across CTAs.
    readers = [input, weight, router_weight, router_bias, global_scale]
    writers = [residual, y_fp4, block_scale, router_logits, y_out]
    for i, out in enumerate(writers):
        if out is None or out.numel() == 0:
            continue
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for other in readers + writers[i + 1 :]:
            if other is not None and other.numel():
                start = other.data_ptr()
                end = start + other.numel() * other.element_size()
                if lo < end and start < hi:
                    raise ValueError(
                        "Router fusion requires non-overlapping input/output buffers"
                    )
    if router_logits is None:
        router_logits = torch.empty((m, e), device=input.device, dtype=input.dtype)
    fp4, scales = _add_rmsnorm_fp4quant_impl(
        input,
        residual,
        weight,
        y_fp4=y_fp4,
        block_scale=block_scale,
        global_scale=global_scale,
        eps=eps,
        enable_pdl=enable_pdl,
        y_out=y_out,
        output_hidden_size=k,
        router_weight=router_weight,
        router_bias=router_bias,
        router_logits=router_logits,
    )
    return fp4, scales, router_logits
