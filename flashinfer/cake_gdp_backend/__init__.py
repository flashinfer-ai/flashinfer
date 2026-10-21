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

Backend of :mod:`flashinfer.cake_gdp`: the Cake Gated DeltaProduct (GDP) decode
and MTP-verification kernels for Nemotron 4 (``K = 128``, ``V = 64``, up to four
Householder updates per token, FP32 state pool; SM100 / SM103) as one
self-contained package (the host API body, the ``cake_jit`` registry and loader
glue and the generated CUDA sources under ``csrc/``).  The stable,
``@flashinfer_api``-decorated entry point and the full user-facing
documentation live in :mod:`flashinfer.cake_gdp`; import that module.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    HEAD_SIZE_K,
    HEAD_SIZE_V,
    NUM_HOUSEHOLDER,
    STATE_ALIGNMENT_BYTES,
    STATE_STRIDE_ELEMENTS,
    arch_for_capability,
    max_num_tokens,
    route_applies,
    route_key,
    selected_module,
    supported_num_householder,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# The kernels read q / k / v / a / b and write the output in BF16; FP16 inputs are
# converted to BF16 before the launch and an FP16 output is converted after it,
# as FlashInfer's MTP decode path does.
_MODEL_DTYPES = (torch.bfloat16, torch.float16)
_INDEX_DTYPES = (torch.int32, torch.int64)
_KERNEL_DTYPE = torch.bfloat16
# Widest vector accesses the kernels make along a tensor's dense last dimension: q and k
# are read in 8-byte pieces (four BF16 elements); the output is written in 8-byte pieces
# by the lane-owns-rows programs and in 4-byte pieces by the row-partition program.  A
# view used in place must keep those pieces naturally aligned (its base address and its
# outer strides); tests/gdn/test_cake_gdp_decode.py checks the widths against the shipped
# kernel sources.
_QK_ALIGNMENT_BYTES = 8
_OUTPUT_ALIGNMENT_BYTES = 8


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"Cake GDP decode requires CUDA tensors, got {device}")
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = arch_for_capability(capability)
    if arch is None:
        raise ValueError(
            "Cake GDP decode supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {capability}"
        )
    return arch


def is_cake_gdp_supported(
    device: torch.device,
    num_householder: Optional[int] = None,
    num_tokens: Optional[int] = None,
    *,
    use_qk_l2norm: bool = True,
) -> bool:
    """Whether ``device`` (and optionally ``num_householder`` / ``num_tokens``) has an exported route."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    if num_householder is None:
        return bool(supported_num_householder(arch, use_qk_l2norm))
    if int(num_householder) not in supported_num_householder(arch, use_qk_l2norm):
        return False
    if num_tokens is None:
        return True
    return 1 <= int(num_tokens) <= max_num_tokens(arch, num_householder, use_qk_l2norm)


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype not in dtypes:
        wanted = " or ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} must be {wanted}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} must have shape {list(shape)}, got {list(tensor.shape)}"
        )


def _aligned(tensor: torch.Tensor, nbytes: int) -> bool:
    """True when ``tensor`` can be accessed in ``nbytes``-byte pieces along its dense last
    dimension: its base address and the strides of its outer dimensions of size > 1 are
    multiples of ``nbytes``."""

    itemsize = tensor.element_size()
    if tensor.data_ptr() % nbytes:
        return False
    return all(
        (stride * itemsize) % nbytes == 0
        for stride, size in zip(tensor.stride()[:-1], tensor.shape[:-1], strict=True)
        if size > 1
    )


def _model_input(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    align: int = 1,
) -> torch.Tensor:
    """A BF16 / FP16 model input of ``dtype`` (the dtype of ``q``) the kernel can read.

    Strided views are read in place when the innermost dimension is contiguous
    (the kernels take every other stride) and the view is ``align``-byte aligned
    (:func:`_aligned`); any other layout is refused.  An FP16 input is converted
    to a BF16 copy.
    """

    _check_tensor(tensor, name, shape=shape, device=device, dtypes=(dtype,))
    if tensor.numel() and tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} must be contiguous in its last dimension (stride {tensor.stride(-1)}); "
            "pass a view whose innermost dimension is dense"
        )
    if tensor.numel() and dtype == _KERNEL_DTYPE and not _aligned(tensor, align):
        raise ValueError(
            f"{name} must be {align}-byte aligned (the kernels read it in {align}-byte "
            f"pieces): its base address and outer strides must be multiples of "
            f"{align // tensor.element_size()} elements; got strides "
            f"{list(tensor.stride())} and a base offset of {tensor.data_ptr() % align} "
            "bytes -- pass an aligned view or a copy"
        )
    return tensor if dtype == _KERNEL_DTYPE else tensor.to(_KERNEL_DTYPE)


def _vector_input(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """A small per-head vector (``A_log``, ``dt_bias``) as the kernel reads it: ``dtype``, contiguous.

    A contiguous vector already in ``dtype`` passes through untouched (no copy, no extra launch); another
    dtype or layout is converted / copied once, which costs one small elementwise launch per call.
    """

    _check_tensor(tensor, name, shape=shape, device=device, dtypes=dtypes)
    if tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _index_input(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...] = _INDEX_DTYPES,
) -> torch.Tensor:
    """A slot-index tensor as the kernel reads it: int32, contiguous; int64 is converted once."""

    _check_tensor(tensor, name, shape=shape, device=device, dtypes=dtypes)
    if tensor.dtype != torch.int32:
        return tensor.to(torch.int32)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _state_pool(tensor: torch.Tensor, *, heads: int, device: torch.device) -> None:
    """The FP32 state pool ``[pool_size, HV, V, K]`` is read and written in place: dense rows of ``K`` elements,
    strides that are multiples of :data:`STATE_STRIDE_ELEMENTS` and a :data:`STATE_ALIGNMENT_BYTES`-aligned base
    (the kernels stage a head's rows with 16-byte asynchronous copies).  A pool of another layout is refused: its
    slots are the caller's state and cannot be copied behind the caller's back."""

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError("initial_state must be a [pool_size, HV, 64, 128] tensor")
    if tensor.device != device:
        raise ValueError(f"initial_state must live on {device}, got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"initial_state must be float32, got {tensor.dtype}")
    pool_size = int(tensor.shape[0])
    if tuple(tensor.shape[1:]) != (heads, HEAD_SIZE_V, HEAD_SIZE_K):
        raise ValueError(
            f"initial_state must have shape [pool_size, HV={heads}, {HEAD_SIZE_V}, {HEAD_SIZE_K}], "
            f"got {list(tensor.shape)}"
        )
    if pool_size == 0:
        raise ValueError("initial_state must hold at least one slot")
    strides = tuple(int(s) for s in tensor.stride())
    if strides[3] != 1 or strides[2] < HEAD_SIZE_K:
        raise ValueError(
            f"initial_state rows must be dense [V, K] rows (strides {strides})"
        )
    if any(stride % STATE_STRIDE_ELEMENTS for stride in strides[:3]):
        raise ValueError(
            f"initial_state strides must be multiples of {STATE_STRIDE_ELEMENTS} elements, got {strides}"
        )
    if tensor.data_ptr() % STATE_ALIGNMENT_BYTES:
        raise ValueError(
            f"initial_state must have a {STATE_ALIGNMENT_BYTES}-byte aligned base address"
        )


def cake_gated_delta_product_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    disable_state_update: Optional[bool] = None,
    use_qk_l2norm: bool = True,
    output_state_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaProduct decode / MTP verification over an FP32 state pool.

    The body of :func:`flashinfer.cake_gdp.cake_gated_delta_product_mtp`; see
    that function for the full documentation of the arguments and semantics.
    """

    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError("q must be a [B, T, H, 128] tensor")
    device = q.device
    arch = _device_arch(device)
    B, T, H, K = (int(x) for x in q.shape)
    if K != HEAD_SIZE_K:
        raise ValueError(f"q must have head size {HEAD_SIZE_K}, got {K}")
    if q.dtype not in _MODEL_DTYPES:
        raise ValueError(f"q must be bfloat16 or float16, got {q.dtype}")
    model_dtype = q.dtype
    if not isinstance(k, torch.Tensor) or k.ndim != 5:
        raise ValueError("k must be a [B, T, num_householder, H, 128] tensor")
    n = int(k.shape[2])
    if n not in NUM_HOUSEHOLDER:
        raise ValueError(
            f"num_householder (k.shape[2]) must be one of {NUM_HOUSEHOLDER}, got {n}"
        )
    if not isinstance(v, torch.Tensor) or v.ndim != 5:
        raise ValueError("v must be a [B, T, num_householder, HV, 64] tensor")
    HV = int(v.shape[3])
    if HV != H:
        raise ValueError(
            f"Cake GDP decode requires H == HV (no GQA / GVA), got H={H} query heads and HV={HV} value heads"
        )
    if disable_state_update is None:
        disable_state_update = False
    disable_state_update = bool(disable_state_update)
    use_qk_l2norm = bool(use_qk_l2norm)
    scale = float(K**-0.5) if scale is None else float(scale)

    q = _model_input(
        q,
        "q",
        shape=(B, T, H, K),
        device=device,
        dtype=model_dtype,
        align=_QK_ALIGNMENT_BYTES,
    )
    k = _model_input(
        k,
        "k",
        shape=(B, T, n, H, K),
        device=device,
        dtype=model_dtype,
        align=_QK_ALIGNMENT_BYTES,
    )
    v = _model_input(
        v, "v", shape=(B, T, n, HV, HEAD_SIZE_V), device=device, dtype=model_dtype
    )
    a = _model_input(a, "a", shape=(B, T, HV), device=device, dtype=model_dtype)
    b = _model_input(b, "b", shape=(B, T, n, HV), device=device, dtype=model_dtype)
    _state_pool(initial_state, heads=HV, device=device)
    A_log = _vector_input(
        A_log,
        "A_log",
        shape=(HV,),
        device=device,
        dtypes=(torch.float32,),
        dtype=torch.float32,
    )
    dt_bias = _vector_input(
        dt_bias,
        "dt_bias",
        shape=(HV,),
        device=device,
        dtypes=(torch.float32, torch.bfloat16, torch.float16),
        dtype=torch.float32,
    )
    read_slots = _index_input(
        initial_state_indices, "initial_state_indices", shape=(B,), device=device
    )
    if output_state_indices is None:
        write_slots = read_slots
    else:
        write_slots = _index_input(
            output_state_indices, "output_state_indices", shape=(B,), device=device
        )
    if ssm_state_indices is not None:
        if disable_state_update:
            raise ValueError(
                "ssm_state_indices requires state writes; disable_state_update must be False"
            )
        scatter_slots = _index_input(
            ssm_state_indices,
            "ssm_state_indices",
            shape=(B, T),
            device=device,
            dtypes=(torch.int32,),
        )
        use_scatter = 1
    else:
        # The kernel takes the scatter table unconditionally and reads it only under ``use_ssm_state_indices != 0``
        # (0 here); the binding still wants a dense [B, T] int32 table, so an uninitialized one stands in: no fill
        # launch, no device synchronization.
        scatter_slots = torch.empty((B, T), dtype=torch.int32, device=device)
        use_scatter = 0
    if T >= 1 and not route_applies(
        device_capability=tuple(torch.cuda.get_device_capability(device)),
        num_householder=n,
        num_tokens=T,
        use_qk_l2norm=use_qk_l2norm,
    ):
        raise ValueError(
            f"no exported Cake GDP decode program for num_householder={n}, T={T}, "
            f"use_qk_l2norm={use_qk_l2norm} on {arch}: the route {route_key(n, use_qk_l2norm)!r} admits T up to "
            f"{max_num_tokens(arch, n, use_qk_l2norm)}; exported routes: {sorted(_loader.ROUTES.get(arch, {}))}"
        )

    out_shape = (B, T, HV, HEAD_SIZE_V)
    target_dtype = model_dtype if output is None else output.dtype
    if output is None:
        # Rows whose initial_state_indices entry is negative are skipped by the kernel and read as zeros.
        out = torch.zeros(out_shape, dtype=_KERNEL_DTYPE, device=device)
        user_out = None
    else:
        _check_tensor(
            output, "output", shape=out_shape, device=device, dtypes=_MODEL_DTYPES
        )
        if output.dtype == _KERNEL_DTYPE and (
            output.numel() == 0
            or (output.stride(-1) == 1 and _aligned(output, _OUTPUT_ALIGNMENT_BYTES))
        ):
            out, user_out = output, None
        else:
            # Another dtype, layout or alignment: the kernel writes a dense, freshly allocated BF16 copy that
            # keeps the caller's values in the rows it skips, copied back after the launch.
            out = output.to(
                _KERNEL_DTYPE, copy=True, memory_format=torch.contiguous_format
            )
            user_out = output
    if B == 0 or T == 0:
        result = output if output is not None else out.to(target_dtype)
        return result, initial_state

    name = selected_module(
        arch,
        n,
        T,
        B * HV,
        sm_count=_loader.sm_count(_device_index(device)),
        use_qk_l2norm=use_qk_l2norm,
    )
    values: dict[str, object] = {
        "q": q,
        "k": k.flatten(1, 2),
        "v": v.flatten(1, 2),
        "state": initial_state,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b.flatten(1, 2),
        "out": out,
        "initial_state_indices": read_slots,
        "output_state_indices": write_slots,
        "ssm_state_indices": scatter_slots,
        "heads": HV,
        "num_tokens": T,
        "scale": scale,
        "disable_state_update": int(disable_state_update),
        "use_ssm_state_indices": use_scatter,
    }
    # The loader binds the current torch stream of the *current* device; make that the
    # tensors' device so a call made while another device is current keeps the order
    # of the stream its inputs were produced on.
    with torch.cuda.device(device):
        _loader.run_module(
            name, values, batch=B, heads=HV, device_index=_device_index(device)
        )
    if user_out is not None:
        user_out.copy_(out)
        return user_out, initial_state
    if output is None and target_dtype != _KERNEL_DTYPE:
        out = out.to(target_dtype)
    return out, initial_state


__all__ = [
    "ARCHES",
    "HEAD_SIZE_K",
    "HEAD_SIZE_V",
    "NUM_HOUSEHOLDER",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "cake_gated_delta_product_mtp",
    "is_cake_gdp_supported",
    "route_key",
]
