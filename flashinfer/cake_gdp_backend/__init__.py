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

Backend of the Cake Gated DeltaProduct (GDP) kernels (``K = 128``,
``V = 64``, up to four Householder updates per token, FP32 state) as one
self-contained package (the host API bodies, the ``cake_jit`` registry and
loader glue and the generated CUDA sources under ``csrc/``): the decode /
MTP-verification kernels behind :mod:`flashinfer.cake_gdp`
(:func:`cake_gated_delta_product_mtp`; SM90 / SM100 / SM103) and the chunked varlen
prefill kernels behind
``flashinfer.gdp_prefill.chunk_gated_delta_product(..., backend="cake")``
(:func:`cake_chunk_gated_delta_product`; SM90).  The stable,
``@flashinfer_api``-decorated entry points and the full user-facing
documentation live in those modules; import them.
"""

from __future__ import annotations

from typing import Optional, Union

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
    prefill_route_applies,
    route_applies,
    route_key,
    selected_module,
    selected_prefill_module,
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
        raise ValueError(f"Cake GDP requires CUDA tensors, got {device}")
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = arch_for_capability(capability)
    if arch is None:
        raise ValueError(
            "Cake GDP supports compute capability "
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


# ---------------------------------------------------------------------------
# Prefill: chunked varlen GDP (``flashinfer.gdp_prefill.chunk_gated_delta_product(..., backend="cake")``)
# ---------------------------------------------------------------------------

# The prefill kernels read q / k / v in BF16 and write O in BF16; FP16 inputs are
# converted before the launch and an FP16 output after it.  The gates are FP32.
_PREFILL_IO_DTYPES = (torch.bfloat16, torch.float16)
_GATE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
# Tensor-map staging the persistent kernel keeps per CTA (four 128-byte slots).
_TENSORMAP_BYTES_PER_CTA = 512
# One (sequence, head) state record: V x K FP32 elements.
_STATE_ELEMENTS = HEAD_SIZE_V * HEAD_SIZE_K
# Bits of the kernel's ``metadata_i64_mask``: int64 ``cu_seqlens``, int64 slot indices.
_CU_SEQLENS_INT64 = 1
_INDICES_INT64 = 2
# The TMA descriptors want 16-byte multiples in every stride but the innermost.
_TMA_STRIDE_BYTES = 16
# ``use_qk_l2norm_in_kernel``: ``x * rsqrt(sum(x^2) + eps)`` over the head dimension.
_L2NORM_EPS = 1e-6


def _prefill_capabilities() -> tuple[tuple[int, int], ...]:
    """The device capabilities with an exported prefill route, ascending."""

    return tuple(
        sorted(
            capability
            for capability, arch in ARCH_BY_CAPABILITY.items()
            if _loader.ROUTES.get(arch, {}).get(_loader.PREFILL_ROUTE)
        )
    )


def is_cake_gdp_prefill_supported(
    device: torch.device, num_householder: Optional[int] = None
) -> bool:
    """Whether ``device`` (and optionally ``num_householder``) has an exported prefill route."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    capability = tuple(torch.cuda.get_device_capability(device))
    if num_householder is None:
        return any(
            prefill_route_applies(device_capability=capability, num_householder=n)
            for n in NUM_HOUSEHOLDER
        )
    return prefill_route_applies(
        device_capability=capability, num_householder=int(num_householder)
    )


# ---- the inputs as the kernel reads them ----


def _tma_operand(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A q / k / v operand the kernel reads through a TMA descriptor: BF16 (an FP16 input is converted once), a
    dense innermost dimension and 16-byte multiples in the outer strides; any other layout is copied once."""

    _check_tensor(tensor, name, shape=shape, device=device, dtypes=(dtype,))
    if tensor.dtype != _KERNEL_DTYPE:
        return tensor.to(_KERNEL_DTYPE).contiguous()
    if tensor.numel() == 0 or tensor.is_contiguous():
        return tensor
    if tensor.stride(-1) == 1 and _aligned(tensor, _TMA_STRIDE_BYTES):
        return tensor
    return tensor.contiguous()


def _gate_input(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> torch.Tensor:
    """A gate (``g`` / ``beta``) as the kernel reads it: FP32, contiguous (converted / copied once otherwise)."""

    _check_tensor(tensor, name, shape=shape, device=device, dtypes=dtypes)
    if tensor.dtype != torch.float32:
        return tensor.to(torch.float32).contiguous()
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _state_rows(
    tensor: torch.Tensor,
    name: str,
    *,
    rows: Optional[int],
    heads: int,
    device: torch.device,
    writable: bool,
) -> torch.Tensor:
    """A prefill state tensor ``[rows, H, 64, 128]`` (``rows`` ``None``: a pool of any positive size) as the kernel
    addresses it: FP32 and contiguous (the records of a (sequence, head) are dense ``V x K`` rows; the kernel takes
    one stride between slots, so a padded pool is refused).  A BF16 per-sequence initial state is converted to an
    FP32 copy; a pool or an output buffer is the caller's and is refused when it is not FP32."""

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(
            f"{name} must be a [{'num_seqs' if rows is not None else 'pool_size'}, H, {HEAD_SIZE_V}, {HEAD_SIZE_K}] tensor"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    wanted_rows = int(tensor.shape[0]) if rows is None else int(rows)
    if tuple(tensor.shape) != (wanted_rows, heads, HEAD_SIZE_V, HEAD_SIZE_K):
        raise ValueError(
            f"{name} must have shape [{'pool_size' if rows is None else rows}, {heads}, {HEAD_SIZE_V}, "
            f"{HEAD_SIZE_K}], got {list(tensor.shape)}"
        )
    if rows is None and wanted_rows == 0:
        raise ValueError(f"{name} must hold at least one slot")
    if tensor.dtype == torch.float32:
        if tensor.is_contiguous():
            return tensor
        if writable:
            raise ValueError(
                f"{name} must be contiguous: the kernel writes its records in place and takes one stride between "
                f"slots (strides {list(tensor.stride())})"
            )
        return tensor.contiguous()
    if writable or rows is None or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be float32, got {tensor.dtype}")
    return tensor.to(torch.float32).contiguous()


def _state_scratch(
    rows: int, heads: int, device: torch.device, *, zeros: bool
) -> torch.Tensor:
    """A contiguous FP32 buffer the kernel addresses with a slot stride of 0 (every slot is the same ``H x V x K``
    record) whose leading extent is ``rows`` (the kernel bounds a pool slot by ``.size(0)``): the discarded final
    state of a call without ``output_final_state`` or the zero initial state of a write-only pool.  ``torch.empty``
    launches nothing; the zero buffer is one memset."""

    columns = -(-(heads * _STATE_ELEMENTS) // max(rows, 1))
    maker = torch.zeros if zeros else torch.empty
    return maker((max(rows, 1), columns), dtype=torch.float32, device=device)


def _int32_view(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """A contiguous index tensor as the kernel takes it: an int32 tensor as is, an int64 tensor as its int32 byte
    view (the kernel reads it as int64 under the metadata mask bit); no conversion copy. The fork-order (V6) programs
    read int32 metadata only: ``_run_single_pass`` / ``_run_two_classes`` cast an int64 ``cu_seqlens`` / slot table to
    int32 for them."""

    tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
    if tensor.dtype == torch.int32:
        return tensor, False
    return tensor.view(torch.int32), True


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """``x * rsqrt(sum(x^2) + eps)`` over the last dimension in FP32, back in ``x``'s dtype."""

    y = x.to(torch.float32)
    return (y * torch.rsqrt((y * y).sum(-1, keepdim=True) + _L2NORM_EPS)).to(x.dtype)


def cake_chunk_gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    num_householder: int = 1,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    *,
    state_indices: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Chunked varlen Gated DeltaProduct prefill on the Cake kernels.

    The body of ``flashinfer.gdp_prefill.chunk_gated_delta_product(..., backend="cake")``; see that function for
    the full documentation of the arguments and semantics.  Every check here reads shapes, dtypes, strides and
    Python scalars only.  The launches follow the body route of record (``prefill_body_classes`` over the host
    sequence lengths): every sequence is classed by its own chunk count, ``H``, ``n`` and the SM count -- the V6
    fork-order body for a sequence under ``ROUTE_MIN_CHUNKS_P3`` (384) chunks (an empty one included) and for a
    call reading no initial state and keeping no final state, the merged chunked body for a unit-gate / unit-beta
    call and for a long sequence whose gate-saturation split items are the cheaper.  A uniform batch (every
    performance row) is one Cake kernel launch: the V6 program of the call's flags over the ``N x H`` (sequence,
    head) tiles on ``min(SM count, N x H)`` persistent CTAs, or, for the merged class, the exact unit program over
    those tiles or the split program of the flags over the items of the split rule of record
    (``prefill_split_table``, which always cuts a merged-class sequence; the item-table upload and a workspace
    clear added).  A mixed batch is one launch per body present, the merged body first, each on the indexed
    program of its body over its own sequences alone: their tokens are gathered into compact copies with their own
    offsets and scattered back afterwards, the dense state tensors are addressed as pools with ``slot =
    sequence``, and every output row and every state slot is written by exactly one launch (the gather / scatter
    copies and the split table / workspace prepare added).  With ``cu_seqlens_cpu`` given, no device-to-host
    synchronization is issued (the route reads the host copy); without it the sequence offsets are copied to the
    host once.  ``cu_seqlens_cpu`` is the host copy of ``cu_seqlens`` (a CPU int32 / int64 tensor of ``N + 1``
    entries, as vLLM keeps ``query_start_loc_cpu`` beside the device tensor); it is checked for its device, rank,
    length and dtype only, never compared value by value with ``cu_seqlens``.  When it is None the call reads
    ``cu_seqlens.cpu()`` -- one device-to-host synchronization, which CUDA Graph capture does not allow -- so a
    graph-captured caller passes the host copy.  A device without an exported prefill route, and a flag set no
    exported program spells (``g=None`` together with ``beta=None``; on SM90 a unit-gate or unit-beta call with an
    ``initial_state`` or a ``state_indices`` pool), is refused with ``ValueError`` rather than approximated.
    """

    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must be a [total_seq_len, H, 128] tensor")
    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Cake GDP prefill requires CUDA tensors, got {device}")
    T, H, K = (int(x) for x in q.shape)
    if K != HEAD_SIZE_K:
        raise ValueError(f"q must have head size {HEAD_SIZE_K}, got {K}")
    if q.dtype not in _PREFILL_IO_DTYPES:
        raise ValueError(f"q must be bfloat16 or float16, got {q.dtype}")
    io_dtype = q.dtype
    n = int(num_householder)
    if isinstance(num_householder, bool) or n not in NUM_HOUSEHOLDER:
        raise ValueError(
            f"num_householder must be one of {NUM_HOUSEHOLDER}, got {num_householder!r}"
        )
    if not isinstance(k, torch.Tensor) or k.ndim != 3:
        raise ValueError("k must be a [total_seq_len * num_householder, H, 128] tensor")
    if not isinstance(v, torch.Tensor) or v.ndim != 3:
        raise ValueError("v must be a [total_seq_len * num_householder, H, 64] tensor")
    if int(k.shape[1]) != H or int(v.shape[1]) != H:
        raise ValueError(
            f"Cake GDP prefill requires one head count for q, k and v (no GQA / GVA), got "
            f"{H} query heads, {int(k.shape[1])} key heads and {int(v.shape[1])} value heads"
        )
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for varlen mode")
    if (
        not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 1
    ):
        raise ValueError("cu_seqlens must be a one-dimensional [num_seqs + 1] tensor")
    if cu_seqlens.dtype not in _INDEX_DTYPES:
        raise ValueError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
    if cu_seqlens.device != device:
        raise ValueError(f"cu_seqlens must live on {device}, got {cu_seqlens.device}")
    N = int(cu_seqlens.numel()) - 1
    capability = tuple(torch.cuda.get_device_capability(device))
    if not prefill_route_applies(device_capability=capability, num_householder=n):
        raise ValueError(
            f"no exported Cake GDP prefill program for num_householder={n} on compute capability {capability}; "
            f"the Cake GDP prefill kernels are exported for the capabilities {list(_prefill_capabilities())} "
            f"(architectures {sorted(arch for arch, routes in _loader.ROUTES.items() if routes.get(_loader.PREFILL_ROUTE))})"
        )
    arch = arch_for_capability(capability)
    scale_value = 0.0 if scale is None else float(scale)
    output_final_state = bool(output_final_state)
    pool_mode = state_indices is not None

    # -- the model inputs
    q_in = _tma_operand(q, "q", shape=(T, H, K), device=device, dtype=io_dtype)
    k_in = _tma_operand(k, "k", shape=(T * n, H, K), device=device, dtype=io_dtype)
    v_in = _tma_operand(
        v, "v", shape=(T * n, H, HEAD_SIZE_V), device=device, dtype=io_dtype
    )
    if use_qk_l2norm_in_kernel and T:
        q_in, k_in = _l2norm(q_in), _l2norm(k_in)
    # g=None runs the unit-gate program (no decay), beta=None the unit-beta program (unit update gates): those builds
    # never read the formal, which takes a one-row placeholder.  Both None is no exported program (its undamped
    # state leaves the final-state tolerance), so the call is refused rather than approximated.  An empty batch (no
    # sequence or no token) launches nothing and is not refused.
    if g is None and beta is None and N > 0 and T > 0:
        raise ValueError(
            f"no exported Cake GDP prefill program for g=None with beta=None on {arch}; pass g or beta as a tensor"
        )
    unit_gates = g is None and T > 0
    unit_beta = beta is None and T > 0
    if g is None:
        gate = torch.ones(
            (1 if unit_gates else T, H), dtype=torch.float32, device=device
        )
    else:
        gate = _gate_input(g, "g", shape=(T, H), device=device, dtypes=_GATE_DTYPES)
    if beta is None:
        beta_in = torch.ones(
            (1 if unit_beta else T * n, H), dtype=torch.float32, device=device
        )
    else:
        beta_in = _gate_input(
            beta,
            "beta",
            shape=(T * n, H),
            device=device,
            dtypes=(torch.float32, io_dtype),
        )
    cu_in, cu_int64 = _int32_view(cu_seqlens)
    mask = _CU_SEQLENS_INT64 if cu_int64 else 0

    # -- the state
    if pool_mode:
        slots = _index_input(
            state_indices,
            "state_indices",
            shape=(N,),
            device=device,
            dtypes=_INDEX_DTYPES,
        )
        if initial_state is not None:
            initial_in = _state_rows(
                initial_state,
                "initial_state",
                rows=None,
                heads=H,
                device=device,
                writable=True,
            )
        else:
            initial_in = None
        if output_final_state:
            if output_state is None:
                raise ValueError(
                    "output_final_state with state_indices needs an explicit output_state pool "
                    "(the identical view of initial_state or disjoint memory)"
                )
            output_in = _state_rows(
                output_state,
                "output_state",
                rows=None,
                heads=H,
                device=device,
                writable=True,
            )
        else:
            output_in = None
    else:
        slots = None
        if initial_state is not None:
            initial_in = _state_rows(
                initial_state,
                "initial_state",
                rows=N,
                heads=H,
                device=device,
                writable=False,
            )
        else:
            initial_in = None
        if output_final_state:
            if output_state is None:
                output_in = torch.empty(
                    (N, H, HEAD_SIZE_V, HEAD_SIZE_K), dtype=torch.float32, device=device
                )
            else:
                output_in = _state_rows(
                    output_state,
                    "output_state",
                    rows=N,
                    heads=H,
                    device=device,
                    writable=True,
                )
        else:
            output_in = None
    if N == 0:
        # No sequence: nothing is read or written; the kernel wants at least one.
        out = (
            output
            if output is not None
            else torch.empty((T, H, HEAD_SIZE_V), dtype=io_dtype, device=device)
        )
        if output is not None:
            _check_tensor(
                output,
                "output",
                shape=(T, H, HEAD_SIZE_V),
                device=device,
                dtypes=_PREFILL_IO_DTYPES,
            )
        if not output_final_state:
            return out
        return out, (
            output_in
            if output_in is not None
            else torch.empty(
                (0, H, HEAD_SIZE_V, HEAD_SIZE_K), dtype=torch.float32, device=device
            )
        )

    # -- the output
    out_shape = (T, H, HEAD_SIZE_V)
    if output is None:
        out = torch.empty(out_shape, dtype=_KERNEL_DTYPE, device=device)
        user_out = None
    else:
        _check_tensor(
            output, "output", shape=out_shape, device=device, dtypes=_PREFILL_IO_DTYPES
        )
        if output.dtype == _KERNEL_DTYPE and output.is_contiguous():
            out, user_out = output, None
        else:
            # Another dtype or layout: the kernel writes a dense BF16 copy, copied back after the launch (the rows no
            # sequence covers keep the caller's values).
            out = output.to(
                _KERNEL_DTYPE, copy=True, memory_format=torch.contiguous_format
            )
            user_out = output
    if T == 0:
        # A batch of empty sequences: every sequence is an identity state update.  The TMA descriptors need one
        # row; ``total_tokens`` = 0 keeps the kernel from reading it.
        q_in = torch.empty((1, H, K), dtype=_KERNEL_DTYPE, device=device)
        k_in = torch.empty((n, H, K), dtype=_KERNEL_DTYPE, device=device)
        v_in = torch.empty((n, H, HEAD_SIZE_V), dtype=_KERNEL_DTYPE, device=device)
        kernel_out = torch.empty(
            (1, H, HEAD_SIZE_V), dtype=_KERNEL_DTYPE, device=device
        )
        gate = torch.ones((1, H), dtype=torch.float32, device=device)
        beta_in = torch.ones((n, H), dtype=torch.float32, device=device)
    else:
        kernel_out = out
    if cu_seqlens_cpu is None:
        # The host copy of the offsets for the split rule: one device-to-host synchronization (a graph-captured
        # caller passes ``cu_seqlens_cpu`` instead).
        cu_seqlens_cpu = cu_seqlens.detach().cpu()
    elif (
        not isinstance(cu_seqlens_cpu, torch.Tensor)
        or cu_seqlens_cpu.device.type != "cpu"
        or cu_seqlens_cpu.ndim != 1
        or cu_seqlens_cpu.numel() < 2
        or cu_seqlens_cpu.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "cu_seqlens_cpu must be a CPU int32/int64 1-D tensor with at least two entries (the host copy of cu_seqlens)"
        )
    if cu_seqlens_cpu.numel() != N + 1:
        raise ValueError(
            f"cu_seqlens_cpu must hold the {N + 1} entries of cu_seqlens, got {cu_seqlens_cpu.numel()}"
        )
    cu_cpu: tuple[int, ...] = tuple(int(x) for x in cu_seqlens_cpu.tolist())
    device_index = _device_index(device)
    sm_count = _loader.sm_count(device_index)
    _run_single_pass(
        arch,
        device,
        device_index,
        sm_count,
        q_in,
        k_in,
        v_in,
        kernel_out,
        gate,
        beta_in,
        cu_in,
        mask,
        slots,
        initial_in,
        output_in,
        scale_value,
        N,
        T,
        H,
        n,
        unit_gates,
        unit_beta,
        cu_cpu,
    )

    if user_out is not None:
        user_out.copy_(out)
        result = user_out
    elif io_dtype != _KERNEL_DTYPE:
        result = out.to(io_dtype)
    else:
        result = out
    if not output_final_state:
        return result
    return result, output_in


def _split_table(arch, cu_cpu, N, T, H, n, sm_count, unit_gates, unit_beta):
    """The split rule of record over the host sequence lengths: the item table (``(sequence, head, cut, span)`` per
    item) when the rule cuts a sequence and the registry routes split programs on ``arch``, else None (the exact
    program; also for a unit-gate / unit-beta call, which has no split program)."""

    if unit_gates or unit_beta:
        return None
    split_flag = getattr(_loader, "PREFILL_SPLIT_FLAG", None)
    if split_flag is None or not any(
        _loader.MODULES[name]["select_rule"].get(split_flag)
        for name in _loader.ROUTES[arch][_loader.PREFILL_ROUTE]
    ):
        return None
    return _loader.prefill_split_table(
        cu_cpu, num_heads=H, num_householder=n, sm_count=sm_count, total_tokens=T
    )


def _launch_prefill(
    name: str,
    *,
    device: torch.device,
    device_index: int,
    sm_count: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu: torch.Tensor,
    slot_in: torch.Tensor,
    table_in: torch.Tensor,
    initial: torch.Tensor,
    initial_stride: int,
    final: torch.Tensor,
    final_stride: int,
    ainv_ws: torch.Tensor,
    scale_value: float,
    num_seqs: int,
    total_tokens: int,
    H: int,
    n: int,
    tiles: int,
    mask: int,
) -> None:
    """One launch of the prefill program ``name`` over the 23 logical arguments of the prefill records; the
    persistent grid of ``min(sm_count, tiles)`` CTAs sizes the tensor-map staging."""

    grid = min(sm_count, tiles)
    values = {
        "q": q,
        "k": k,
        "v": v,
        "o": o,
        "gate": gate,
        "beta": beta,
        "cu_seqlens": cu,
        "state_indices": slot_in,
        "output_slot_indices": table_in,
        "initial_state": initial,
        "output_state": final,
        "tensormap_workspace": torch.empty(
            (grid * _TENSORMAP_BYTES_PER_CTA,), dtype=torch.uint8, device=device
        ),
        "initial_state_stride_slot": initial_stride,
        "output_state_stride_slot": final_stride,
        "scale": scale_value,
        "num_seqs": num_seqs,
        "total_tokens": total_tokens,
        "num_heads": H,
        "num_householder": n,
        "total_tiles": tiles,
        "metadata_i64_mask": mask,
        "ainv_workspace": ainv_ws,
        "ainv_chunk_stride": 0,
    }
    # The loader binds the current torch stream of the *current* device; make that the tensors' device.
    with torch.cuda.device(device):
        _loader.run_module(
            name,
            values,
            batch=num_seqs,
            heads=H,
            device_index=device_index,
            tiles=tiles,
            sm_count=sm_count,
        )


def _split_items(
    table: list, num_seqs: int, H: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The device inputs of a split program over the item ``table``: the int32 item table (formal 9, four words per
    item), the A-inverse workspace (the ``num_seqs x H`` seed-done flags and two plan-echo words per item, zero at
    launch) and the tile count (the items)."""

    items = len(table)
    table_in = torch.tensor(table, dtype=torch.int32, device=device).reshape(-1)
    ainv_ws = torch.zeros(
        (4 * (num_seqs * H + 2 * items),), dtype=torch.uint8, device=device
    )
    return table_in, ainv_ws, items


def _run_single_pass(
    arch,
    device,
    device_index,
    sm_count,
    q_in,
    k_in,
    v_in,
    out,
    gate,
    beta_in,
    cu_in,
    mask,
    slots,
    initial_in,
    output_in,
    scale_value,
    N,
    T,
    H,
    n,
    unit_gates,
    unit_beta,
    cu_cpu,
) -> None:
    """The launches of one call under the body route of record: ``prefill_body_classes`` over the host offsets
    ``cu_cpu`` classes every sequence and ``prefill_launch_plan`` orders the launches.  One class (every performance
    row) is one launch of the whole batch: the V6 program of the call's flags over the ``N x H`` tiles on
    ``min(SM count, N x H)`` persistent CTAs (every V6 program stores a final state; a call keeping none gets the
    discard record), or, for the merged class, the exact unit program over those tiles or the split program of the
    flags over the items of the split rule (tiles = the items; a merged-class sequence is always cut).  Two classes
    are two launches on the same stream (``_run_two_classes``)."""

    use_indices = slots is not None
    store_final = output_in is not None
    if use_indices and not store_final and initial_in is None:
        use_initial, use_indices = (
            False,
            False,
        )  # nothing read, nothing written: the plain program
    else:
        use_initial = (
            initial_in is not None or use_indices
        )  # a write-only pool reads a zero record
    # The shipped flag sets: a program reading an initial state and a unit program store a final state (a call
    # wanting none gets a discard record); the plain call without a final state is V6-class at every length, and
    # the V6 plain program stores one too.
    stores = store_final or use_initial or unit_gates or unit_beta
    classes = _loader.prefill_body_classes(
        cu_cpu,
        n,
        H,
        sm_count,
        store_final_state=stores,
        unit_gates=unit_gates,
        unit_beta=unit_beta,
        total_tokens=T,
    )
    plan = _loader.prefill_launch_plan(classes)
    pool_rows = N
    if use_indices:
        pool_rows = max(
            int(initial_in.shape[0]) if initial_in is not None else 0,
            int(output_in.shape[0]) if output_in is not None else 0,
        )
    launch = dict(
        device=device,
        device_index=device_index,
        sm_count=sm_count,
        scale_value=scale_value,
        H=H,
        n=n,
    )
    if len(plan) == 2:
        _run_two_classes(
            arch,
            plan,
            launch,
            q_in,
            k_in,
            v_in,
            out,
            gate,
            beta_in,
            slots,
            initial_in,
            output_in,
            store_final,
            pool_rows,
            T,
            cu_cpu,
        )
        return
    body = plan[0][0]
    flags = dict(
        use_initial_state=use_initial,
        use_state_indices=use_indices,
        store_final_state=stores,
        unit_gates=unit_gates,
        unit_beta=unit_beta,
    )
    table = None
    if body == _loader.ROUTE_BODY_V6:
        # Every V6 program stores a final state: a call keeping none gets the discard record below.
        name = selected_prefill_module(
            arch,
            body=_loader.ROUTE_BODY_V6,
            use_initial_state=use_initial,
            use_state_indices=use_indices,
            store_final_state=True,
            unit_gates=False,
            unit_beta=False,
        )
        if mask & _CU_SEQLENS_INT64:
            # Every V6 program reads int32 metadata only (the plain program is the body of record; the i64 select in the
            # flag programs' tile prologue re-scheduled their chunk loops): an int64 cu_seqlens is cast to int32 for it
            # (N + 1 elements, one small copy); the slot table below likewise.
            cu_in = cu_in.view(torch.int64).to(torch.int32)
            mask &= ~_CU_SEQLENS_INT64
    elif unit_gates or unit_beta:
        try:
            name = selected_prefill_module(arch, body=_loader.ROUTE_BODY_P3, **flags)
        except ValueError as exc:
            # The registry routes no program of these flags on this architecture (on SM90: a unit-gate or unit-beta
            # call reading an initial state or a slot pool); the call is refused rather than approximated.
            raise ValueError(
                f"no exported Cake GDP prefill program on {arch} for this call: initial state "
                f"{'read' if use_initial else 'none'}, state pool {'indexed' if use_indices else 'none'}, final "
                f"state {'stored' if stores else 'dropped'}, g={'None' if unit_gates else 'tensor'}, "
                f"beta={'None' if unit_beta else 'tensor'} ({exc})"
            ) from None
    else:
        table = _split_table(arch, cu_cpu, N, T, H, n, sm_count, unit_gates, unit_beta)
        if table is None:
            raise ValueError(
                "Cake GDP prefill body route invariant broken: a merged-class sequence has at least "
                f"{_loader.ROUTE_MIN_CHUNKS_P3} chunks and the split rule of record cuts it, yet the rule left the "
                f"batch whole (classes {classes}) or {arch} routes no split program"
            )
        name = selected_prefill_module(
            arch,
            body=_loader.ROUTE_BODY_P3,
            **flags,
            **{_loader.PREFILL_SPLIT_FLAG: True},
        )
    if use_initial:
        if initial_in is not None:
            initial, initial_stride = initial_in, H * _STATE_ELEMENTS
        else:
            initial, initial_stride = (
                _state_scratch(pool_rows, H, device, zeros=True),
                0,
            )
    else:
        initial, initial_stride = _state_scratch(1, H, device, zeros=False), 0
    if store_final:
        final, final_stride = output_in, H * _STATE_ELEMENTS
    else:
        # Every routed program stores a final state; the call wants none: a discard record every slot maps to.
        final, final_stride = _state_scratch(pool_rows, H, device, zeros=False), 0
    if use_indices:
        slot_in, slot_int64 = _int32_view(slots)
        if slot_int64 and body == _loader.ROUTE_BODY_V6:
            # the V6 programs read int32 slots only: one small copy (N elements)
            slot_in = slots.to(torch.int32)
        elif slot_int64:
            mask |= _INDICES_INT64
    else:
        slot_in = torch.empty((1,), dtype=torch.int32, device=device)
    if table is not None:
        # The split program: formal 9 carries the item table, the tiles are the items.
        table_in, ainv_ws, tiles = _split_items(table, N, H, device)
    else:
        table_in = slot_in
        ainv_ws = torch.empty((16,), dtype=torch.uint8, device=device)
        tiles = N * H
    _launch_prefill(
        name,
        q=q_in,
        k=k_in,
        v=v_in,
        o=out,
        gate=gate,
        beta=beta_in,
        cu=cu_in,
        slot_in=slot_in,
        table_in=table_in,
        initial=initial,
        initial_stride=initial_stride,
        final=final,
        final_stride=final_stride,
        ainv_ws=ainv_ws,
        num_seqs=N,
        total_tokens=T,
        tiles=tiles,
        mask=mask,
        **launch,
    )


def _run_two_classes(
    arch,
    plan,
    launch,
    q_in,
    k_in,
    v_in,
    out,
    gate,
    beta_in,
    slots,
    initial_in,
    output_in,
    store_final,
    pool_rows,
    T,
    cu_cpu,
) -> None:
    """The two launches of a mixed-class batch in ``plan`` order (the merged body, then the V6 body), each on the
    indexed program of its body over its class's sequences alone: the class's tokens are gathered into compact
    q / k / v / gate / beta copies with their own int32 offsets (one upload of the gather indices, one of the
    offsets and sequence indices), the state tensors are addressed as pools -- the user pool through the class's
    slots, a dense ``[N, H, 64, 128]`` tensor with ``slot = sequence``, the zero record / the discard record where
    the call reads no initial / keeps no final state -- and the launch's output rows are scattered back into
    ``out`` afterwards.  No output row and no state slot is written by both launches.  A unit call and a call
    without an initial and a final state are never mixed (the route classes every sequence of such a call alike),
    so the gates and betas are tensors here."""

    device, H, n, sm_count = (
        launch["device"],
        launch["H"],
        launch["n"],
        launch["sm_count"],
    )
    use_indices = slots is not None
    if initial_in is not None:
        initial, initial_stride = initial_in, H * _STATE_ELEMENTS
    else:
        initial, initial_stride = _state_scratch(pool_rows, H, device, zeros=True), 0
    if store_final:
        final, final_stride = output_in, H * _STATE_ELEMENTS
    else:
        final, final_stride = _state_scratch(pool_rows, H, device, zeros=False), 0
    for body, seqs in plan:
        num_seqs = len(seqs)
        pieces, cu_b = [], [0]
        for seq in seqs:
            start, end = _loader.split_sequence_bounds(cu_cpu[seq], cu_cpu[seq + 1], T)
            pieces.append(torch.arange(start, end, dtype=torch.int64))
            cu_b.append(cu_b[-1] + end - start)
        tok_cpu = torch.cat(pieces)
        total_tokens = int(tok_cpu.numel())
        # One upload of the gather indices: the class's tokens, then their ``n`` expanded k / v / beta rows.
        tok_n_cpu = tok_cpu[:, None] * n + torch.arange(n, dtype=torch.int64)
        tok_n_cpu = tok_n_cpu.reshape(-1)
        gather = torch.cat((tok_cpu, tok_n_cpu)).to(device)
        tok, tok_n = gather[:total_tokens], gather[total_tokens:]
        # One upload of the int32 metadata: the compact offsets, then the class's sequence indices.
        meta = torch.tensor([*cu_b, *seqs], dtype=torch.int32).to(device)
        cu_in, seq_in = meta[: num_seqs + 1], meta[num_seqs + 1 :]
        # The state slots of the class: the user pool's through its indices; else the dense tensors are pools
        # whose slot of sequence ``s`` is ``s``.
        slot_src = slots.index_select(0, seq_in) if use_indices else seq_in
        if body == _loader.ROUTE_BODY_V6 and slot_src.dtype == torch.int64:
            # the V6 programs read int32 slots only (the compact offsets are int32 already)
            slot_in, mask = slot_src.to(torch.int32), 0
        else:
            slot_in, slot_int64 = _int32_view(slot_src)
            mask = _INDICES_INT64 if slot_int64 else 0  # the compact offsets are int32
        if total_tokens:
            q_b = q_in.index_select(0, tok)
            k_b = k_in.index_select(0, tok_n)
            v_b = v_in.index_select(0, tok_n)
            gate_b = gate.index_select(0, tok)
            beta_b = beta_in.index_select(0, tok_n)
            o_b = torch.empty(
                (total_tokens, H, HEAD_SIZE_V), dtype=_KERNEL_DTYPE, device=device
            )
        else:
            # Every sequence of the class is empty (the V6 class beside merged-class long sequences): the one-row
            # placeholders of an empty call; ``total_tokens`` = 0 keeps the kernel from reading them.
            q_b = torch.empty((1, H, HEAD_SIZE_K), dtype=_KERNEL_DTYPE, device=device)
            k_b = torch.empty((n, H, HEAD_SIZE_K), dtype=_KERNEL_DTYPE, device=device)
            v_b = torch.empty((n, H, HEAD_SIZE_V), dtype=_KERNEL_DTYPE, device=device)
            gate_b = torch.ones((1, H), dtype=torch.float32, device=device)
            beta_b = torch.ones((n, H), dtype=torch.float32, device=device)
            o_b = torch.empty((1, H, HEAD_SIZE_V), dtype=_KERNEL_DTYPE, device=device)
        if body == _loader.ROUTE_BODY_P3:
            table = _loader.prefill_split_table(
                tuple(cu_b),
                num_heads=H,
                num_householder=n,
                sm_count=sm_count,
                total_tokens=total_tokens,
            )
            if table is None:
                raise ValueError(
                    "Cake GDP prefill body route invariant broken: a merged-class sequence has at least "
                    f"{_loader.ROUTE_MIN_CHUNKS_P3} chunks and the split rule of record cuts it, yet the rule "
                    f"left the merged class (sequences {seqs}) whole"
                )
            name = selected_prefill_module(
                arch,
                body=_loader.ROUTE_BODY_P3,
                use_initial_state=True,
                use_state_indices=True,
                store_final_state=True,
                **{_loader.PREFILL_SPLIT_FLAG: True},
            )
            table_in, ainv_ws, tiles = _split_items(table, num_seqs, H, device)
        else:
            name = selected_prefill_module(
                arch,
                body=_loader.ROUTE_BODY_V6,
                use_initial_state=True,
                use_state_indices=True,
                store_final_state=True,
            )
            table_in = slot_in
            ainv_ws = torch.empty((16,), dtype=torch.uint8, device=device)
            tiles = num_seqs * H
        _launch_prefill(
            name,
            q=q_b,
            k=k_b,
            v=v_b,
            o=o_b,
            gate=gate_b,
            beta=beta_b,
            cu=cu_in,
            slot_in=slot_in,
            table_in=table_in,
            initial=initial,
            initial_stride=initial_stride,
            final=final,
            final_stride=final_stride,
            ainv_ws=ainv_ws,
            num_seqs=num_seqs,
            total_tokens=total_tokens,
            tiles=tiles,
            mask=mask,
            **launch,
        )
        if total_tokens:
            out.index_copy_(0, tok, o_b)


__all__ = [
    "ARCHES",
    "HEAD_SIZE_K",
    "HEAD_SIZE_V",
    "NUM_HOUSEHOLDER",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "cake_chunk_gated_delta_product",
    "cake_gated_delta_product_mtp",
    "is_cake_gdp_prefill_supported",
    "is_cake_gdp_supported",
    "route_key",
]
