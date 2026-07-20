# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/quantization/__init__.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""BF16 → NVFP4 TMA quantization kernel API."""

from dataclasses import dataclass
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    compile as sm12x_compile,
)
from flashinfer.experimental.sm12x._lib.intrinsics import align_up
from flashinfer.experimental.sm12x._lib.utils import (
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from flashinfer.experimental.sm12x.quantization.nvfp4._kernel import TestKernel
from flashinfer.experimental.sm12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)

_TILE_M = 128
_TILE_K = 128
_SF_VEC_SIZE = 16
_KERNEL_CACHE: Dict[Tuple, object] = {}


def _validate_tiled_shape(M: int, K: int) -> None:
    if M <= 0 or K <= 0:
        raise ValueError(f"M and K must be positive, got M={M}, K={K}")
    if M % _TILE_M != 0 or K % _TILE_K != 0:
        raise ValueError(
            f"M and K must be multiples of ({_TILE_M}, {_TILE_K}), got M={M}, K={K}"
        )


def _validate_launch_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device or tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _contiguous_byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    begin = (
        tensor.untyped_storage().data_ptr()
        + tensor.storage_offset() * tensor.element_size()
    )
    return begin, begin + tensor.numel() * tensor.element_size()


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    lhs_begin, lhs_end = _contiguous_byte_interval(lhs)
    rhs_begin, rhs_end = _contiguous_byte_interval(rhs)
    return lhs_begin < rhs_end and rhs_begin < lhs_end


@dataclass
class BF16ToFP4TMAOutputs:
    packed_a_storage: torch.Tensor
    scale_storage: torch.Tensor
    packed_a_view: object
    sfa_ptr: object

    @property
    def packed_a_flat(self) -> torch.Tensor:
        return self.packed_a_storage.view(-1)

    @property
    def scale_flat(self) -> torch.Tensor:
        return self.scale_storage.view(-1)


def allocate_bf16_to_fp4_tma_outputs(
    M: int,
    K: int,
    *,
    device: torch.device = torch.device("cuda"),
) -> BF16ToFP4TMAOutputs:
    _validate_tiled_shape(M, K)
    rows_pad = align_up(M, _TILE_M)
    cols_pad_sf = align_up(K // _SF_VEC_SIZE, 4)
    packed_a_storage = torch.zeros(1, M, K // 2, dtype=torch.uint8, device=device)
    scale_storage = torch.zeros(
        rows_pad * cols_pad_sf, dtype=torch.uint8, device=device
    )
    packed_a_view = packed_a_storage.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    sfa_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        scale_storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return BF16ToFP4TMAOutputs(
        packed_a_storage=packed_a_storage,
        scale_storage=scale_storage,
        packed_a_view=packed_a_view,
        sfa_ptr=sfa_ptr,
    )


def compile_bf16_to_fp4_tma(M: int, K: int):
    """Compile the BF16→FP4 TMA kernel for (M, K). Returns a launch callable.

    The callable signature is: ``launch(bf16_input, global_scale, packed_a_flat, scale_flat)``
    where packed_a_flat and scale_flat come from ``BF16ToFP4TMAOutputs``.
    """
    _validate_tiled_shape(M, K)
    ab = cutlass.Float4E2M1FN
    bf = cutlass.BFloat16
    bf16_fake = cute.runtime.make_fake_compact_tensor(
        bf, (M, K), stride_order=(1, 0), assumed_align=16
    )
    gs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    pa_fake = cute.runtime.make_fake_compact_tensor(
        ab, (M, K, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (M * K // _SF_VEC_SIZE,),
        assumed_align=16,
    )
    mac = min(get_max_active_clusters(1), get_num_sm(torch.device("cuda")))
    if M == _TILE_M:
        # CUTLASS DSL 4.6 already lowers the register count for these shapes;
        # retain their original instruction schedule.
        liveness_strategy = "retain"
    else:
        # CUTLASS DSL 4.6 otherwise keeps two BF16-derived FP32 values live
        # across exact scale division.  Preserve the pair losslessly in one
        # raw register to shorten that live range without adding memory work.
        liveness_strategy = "packed"
    cache_key = (M, K, liveness_strategy, mac)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kernel = TestKernel(liveness_strategy)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    from cutlass.base_dsl.compiler import OptLevel

    raw = sm12x_compile(
        kernel,
        bf16_fake,
        gs_fake,
        pa_fake,
        scale_fake,
        mac,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quantization.bf16_to_fp4_tma",
            2,
            cache_key,
            labels=("M", "K", "liveness_strategy", "mac"),
        ),
        dsl_compile_options=OptLevel(1),
    )

    def launch(bf16_input, global_scale, packed_a_flat, scale_flat):
        device = bf16_input.device
        _validate_launch_tensor(
            bf16_input,
            name="bf16_input",
            dtype=torch.bfloat16,
            shape=(M, K),
            device=device,
        )
        _validate_launch_tensor(
            global_scale,
            name="global_scale",
            dtype=torch.float32,
            shape=(1,),
            device=device,
        )
        _validate_launch_tensor(
            packed_a_flat,
            name="packed_a_flat",
            dtype=torch.uint8,
            shape=(M * K // 2,),
            device=device,
        )
        _validate_launch_tensor(
            scale_flat,
            name="scale_flat",
            dtype=torch.uint8,
            shape=(M * K // _SF_VEC_SIZE,),
            device=device,
        )
        if _overlaps(packed_a_flat, scale_flat):
            raise ValueError("packed_a_flat and scale_flat must not overlap")
        if _overlaps(bf16_input, packed_a_flat) or _overlaps(bf16_input, scale_flat):
            raise ValueError("bf16_input must not overlap either output")
        if _overlaps(global_scale, packed_a_flat) or _overlaps(
            global_scale, scale_flat
        ):
            raise ValueError("global_scale must not overlap either output")
        pa_view = (
            packed_a_flat.view(1, M, K // 2)
            .permute(1, 2, 0)
            .view(torch.float4_e2m1fn_x2)
        )
        raw(bf16_input, global_scale, pa_view, scale_flat, current_cuda_stream())

    _KERNEL_CACHE[cache_key] = launch
    return launch
