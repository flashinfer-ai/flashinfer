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
``V = 64``, up to four Householder updates per token, FP32 recurrence) as one
self-contained package (the host API bodies, the ``cake_jit`` registry and
loader glue and the generated CUDA sources under ``csrc/``): the decode /
MTP-verification kernels behind :mod:`flashinfer.cake_gdp`
(:func:`cake_gated_delta_product_mtp`; SM90 / SM100 / SM103 / SM107) and the chunked varlen
prefill kernels behind
``flashinfer.gdp_prefill.chunk_gated_delta_product(..., backend="cake")``
(:func:`cake_chunk_gated_delta_product`; SM90 / SM100 / SM103).  The stable,
``@flashinfer_api``-decorated entry points and the full user-facing
documentation live in those modules; import them.
"""

from __future__ import annotations

import array
import collections
import functools
from typing import NamedTuple, Optional, Union

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    F16_STATE_ALIGNMENT_BYTES,
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
_INT32_MAX = (1 << 31) - 1
_KERNEL_DTYPE = torch.bfloat16
# Widest vector accesses the kernels make along a tensor's dense last dimension: q and k
# are read in 8-byte pieces (four BF16 elements); the output is written in 8-byte pieces
# by the lane-owns-rows programs and in 4-byte pieces by the row-partition program.  A
# view used in place must keep those pieces naturally aligned (its base address and its
# outer strides); tests/gdn/test_cake_gdp_decode.py checks the widths against the shipped
# kernel sources.
_QK_ALIGNMENT_BYTES = 8
_OUTPUT_ALIGNMENT_BYTES = 8
# The state pool dtypes: FP32 everywhere, FP16 where the registry ships FP16-pool
# programs (SM100 / SM103 / SM107: the stores round with the hardware ``cvt.rs``).
_STATE_DTYPES = (torch.float32, torch.float16)
# Stochastic rounding of an FP16 pool: the accepted Philox round counts.
PHILOX_ROUNDS_MIN = 5
PHILOX_ROUNDS_MAX = 16


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
    state_dtype: torch.dtype = torch.float32,
) -> bool:
    """Whether ``device`` (and optionally ``num_householder`` / ``num_tokens``) has an exported route for a
    ``state_dtype`` state pool (``torch.float32``, or ``torch.float16`` on SM100 / SM103 / SM107)."""

    device = torch.device(device)
    if device.type != "cuda" or state_dtype not in _STATE_DTYPES:
        return False
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    supported = supported_num_householder(arch, use_qk_l2norm, state_dtype=state_dtype)
    if num_householder is None:
        return bool(supported)
    if int(num_householder) not in supported:
        return False
    if num_tokens is None:
        return True
    return (
        1
        <= int(num_tokens)
        <= max_num_tokens(arch, num_householder, use_qk_l2norm, state_dtype=state_dtype)
    )


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
    """A slot-index tensor as the kernel reads it: int32, contiguous; int64 is converted once (an entry outside
    the int32 range becomes -1)."""

    _check_tensor(tensor, name, shape=shape, device=device, dtypes=dtypes)
    if tensor.dtype != torch.int32:
        # An int64 slot outside the int32 range would wrap onto a row of the pool: it becomes -1, a slot outside
        # every pool, on the device (no host synchronization; graph-capturable).  Every other value is kept, and the
        # kernel checks it against the pool it addresses.
        in_range = (tensor >= 0) & (tensor <= _INT32_MAX)
        return torch.where(in_range, tensor, -1).to(
            torch.int32, memory_format=torch.contiguous_format
        )
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _state_pool(
    tensor: torch.Tensor, *, heads: int, device: torch.device, arch: str
) -> None:
    """The state pool ``[pool_size, HV, V, K]`` is read and written in place: dense rows of ``K`` elements,
    strides that are multiples of :data:`STATE_STRIDE_ELEMENTS` and an aligned base -- FP32:
    :data:`STATE_ALIGNMENT_BYTES` (the kernels stage a head's rows with 16-byte asynchronous copies); FP16:
    :data:`F16_STATE_ALIGNMENT_BYTES` (8-byte pieces), on an architecture with FP16-pool programs only.  A pool of
    another dtype or layout is refused: its slots are the caller's state and cannot be copied behind the caller's
    back."""

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError("initial_state must be a [pool_size, HV, 64, 128] tensor")
    if tensor.device != device:
        raise ValueError(f"initial_state must live on {device}, got {tensor.device}")
    if tensor.dtype not in _STATE_DTYPES:
        raise ValueError(
            f"initial_state must be float32 or float16, got {tensor.dtype}"
        )
    if tensor.dtype == torch.float16 and not supported_num_householder(
        arch, state_dtype=torch.float16
    ):
        raise ValueError(
            "a float16 state pool needs the hardware stochastic rounding of SM100 / SM103 / SM107 "
            f"(cvt.rs); {arch} has no float16-pool program"
        )
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
    alignment = (
        STATE_ALIGNMENT_BYTES
        if tensor.dtype == torch.float32
        else F16_STATE_ALIGNMENT_BYTES
    )
    if tensor.data_ptr() % alignment:
        raise ValueError(
            f"a {tensor.dtype} initial_state must have a {alignment}-byte aligned base address"
        )


def _rounding(
    pool: torch.Tensor, rand_seed: Optional[torch.Tensor], philox_rounds: int
) -> tuple[torch.Tensor, int]:
    """The seed tensor and round count of a float16 pool's launch.  With ``rand_seed`` the stores round
    stochastically with ``philox_rounds`` Philox rounds (an ``int`` from :data:`PHILOX_ROUNDS_MIN` to
    :data:`PHILOX_ROUNDS_MAX`); the seed is a one-element int64 tensor on the pool's device, read by the kernel.
    Without one they round to nearest even (0 rounds), and the seed argument is never read: the pool's first eight
    bytes viewed as int64 when the pool's storage offset allows it (no allocation), else a one-element int64
    placeholder."""

    if rand_seed is None:
        # The kernel loads the seed only when philox_rounds != 0, so this argument is never read.  The int64 view
        # of the pool's first four elements relies on the 8-byte base alignment _state_pool has checked; torch
        # also needs the storage offset to be a multiple of four elements, which fails only for a storage whose
        # own base is not 8-byte aligned (an UntypedStorage slice, DLPack): that pool gets a per-call placeholder.
        # An empty state tensor (a dense call over no sequence) has no element to view: it takes the placeholder.
        if pool.numel() and pool.storage_offset() % 4 == 0:
            return pool.as_strided((4,), (1,)).view(torch.int64), 0
        return pool.new_empty((1,), dtype=torch.int64), 0
    if not isinstance(rand_seed, torch.Tensor):
        raise TypeError("rand_seed must be a torch.Tensor")
    if rand_seed.dtype != torch.int64 or rand_seed.numel() != 1:
        raise ValueError(
            f"rand_seed must be a one-element int64 tensor, got {rand_seed.dtype} with "
            f"{rand_seed.numel()} elements"
        )
    if rand_seed.device != pool.device:
        raise ValueError(
            f"rand_seed must live on the state pool's device {pool.device}, got {rand_seed.device}"
        )
    if isinstance(philox_rounds, bool) or not isinstance(philox_rounds, int):
        raise TypeError(
            f"philox_rounds must be an int, got {type(philox_rounds).__name__}"
        )
    if not PHILOX_ROUNDS_MIN <= philox_rounds <= PHILOX_ROUNDS_MAX:
        raise ValueError(
            f"philox_rounds must be {PHILOX_ROUNDS_MIN} to {PHILOX_ROUNDS_MAX} with a rand_seed, got {philox_rounds}"
        )
    return rand_seed.reshape(1), philox_rounds


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
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaProduct decode / MTP verification over an FP32 or FP16 state pool.

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
    _state_pool(initial_state, heads=HV, device=device, arch=arch)
    state_dtype = initial_state.dtype
    rounds = 0  # the launch class of an FP32 pool: its stores do not round
    if state_dtype == torch.float16:
        seed, rounds = _rounding(initial_state, rand_seed, philox_rounds)
    elif rand_seed is not None:
        raise ValueError(
            "rand_seed selects the stochastic rounding of a float16 state pool; "
            "a float32 pool stores the state without rounding"
        )
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
        state_dtype=state_dtype,
    ):
        raise ValueError(
            f"no exported Cake GDP decode program for num_householder={n}, T={T}, "
            f"use_qk_l2norm={use_qk_l2norm} on {arch}: the route {route_key(n, use_qk_l2norm)!r} admits T up to "
            f"{max_num_tokens(arch, n, use_qk_l2norm, state_dtype=state_dtype)} for a {state_dtype} pool; "
            f"exported routes: {sorted(_loader.ROUTES.get(arch, {}))}"
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
        state_dtype=state_dtype,
        # The launch class (host values, so a captured call keeps its program): an FP16 pool's round count and
        # whether the call stores per-token snapshots pick a program's sibling (snapshot_noise or draw_share)
        # where it is faster.
        philox_rounds=rounds,
        use_ssm_state_indices=bool(use_scatter),
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
    if state_dtype == torch.float16:
        values["rand_seed"] = seed
        values["philox_rounds"] = rounds
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
# One (sequence, head) state record: V x K elements.
_STATE_ELEMENTS = HEAD_SIZE_V * HEAD_SIZE_K
# The prefill state dtypes: FP32 everywhere; BF16 and FP16 on the architectures whose registry ships their programs
# (SM100 / SM103: the state is read widened exactly and its final value rounded once, with the hardware ``cvt.rs``).
_PREFILL_STATE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
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


def _prefill_pool(
    tensor: torch.Tensor,
    name: str,
    *,
    rows: Optional[int],
    heads: int,
    device: torch.device,
    dtype: torch.dtype,
    writable: bool,
) -> torch.Tensor:
    """A state tensor of the Blackwell pool programs: ``[rows, H, 64, 128]`` (``rows`` ``None``: a pool of any
    positive size) of ``dtype``, dense ``[V, K]`` records of one head after another (strides ``(.., 8192, 128, 1)``)
    with a slot stride that is a multiple of :data:`STATE_STRIDE_ELEMENTS` (a padded pool is read and written in
    place) and an aligned base (16 bytes for FP32, 8 for BF16 / FP16).  Another layout is refused when the kernel
    writes the tensor (it is the caller's state) and copied once when it only reads a per-sequence state."""

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(
            f"{name} must be a [{'num_seqs' if rows is not None else 'pool_size'}, H, {HEAD_SIZE_V}, {HEAD_SIZE_K}] tensor"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(
            f"{name} must be {dtype} (the call's state dtype), got {tensor.dtype}"
        )
    wanted_rows = int(tensor.shape[0]) if rows is None else int(rows)
    if tuple(tensor.shape) != (wanted_rows, heads, HEAD_SIZE_V, HEAD_SIZE_K):
        raise ValueError(
            f"{name} must have shape [{'pool_size' if rows is None else rows}, {heads}, {HEAD_SIZE_V}, "
            f"{HEAD_SIZE_K}], got {list(tensor.shape)}"
        )
    if rows is None and wanted_rows == 0:
        raise ValueError(f"{name} must hold at least one slot")
    strides = tuple(int(x) for x in tensor.stride())
    alignment = (
        STATE_ALIGNMENT_BYTES if dtype == torch.float32 else F16_STATE_ALIGNMENT_BYTES
    )
    dense_rows = (
        strides[3] == 1
        and strides[2] == HEAD_SIZE_K
        and (heads == 1 or strides[1] == _STATE_ELEMENTS)
    )
    slot_stride = (
        wanted_rows <= 1
        or strides[0] % STATE_STRIDE_ELEMENTS == 0
        and strides[0] >= heads * _STATE_ELEMENTS
    )
    if dense_rows and slot_stride and tensor.data_ptr() % alignment == 0:
        return tensor
    if writable or rows is None:
        raise ValueError(
            f"{name} must hold dense [{HEAD_SIZE_V}, {HEAD_SIZE_K}] records of one head after another (strides "
            f"[slot, {_STATE_ELEMENTS}, {HEAD_SIZE_K}, 1]), a slot stride that is a multiple of "
            f"{STATE_STRIDE_ELEMENTS} elements and a {alignment}-byte aligned base; got strides {list(strides)} and a "
            f"base offset of {tensor.data_ptr() % alignment} bytes"
        )
    return tensor.contiguous() if not tensor.is_contiguous() else tensor.clone()


def _slot_stride(tensor: Optional[torch.Tensor], heads: int) -> int:
    """The slot stride the pool programs take for a state tensor (its ``stride(0)``; a one-slot tensor's is never
    used, so it takes the dense one; an absent tensor's is never read)."""

    if tensor is None:
        return 0
    if int(tensor.shape[0]) <= 1:
        return heads * _STATE_ELEMENTS
    return int(tensor.stride(0))


# The identity slot table of a dense BF16 / FP16 call (its state rows are slots 0 .. N - 1 of the per-sequence
# tensors), per device: one complete int32 table, grown on an eager call and sliced per call.  A table made during
# CUDA Graph capture holds its values only once the graph replays, so it is that call's own and never cached.  A
# graph captured after an eager call reads the cached table's memory at every replay, so a table that growth
# replaces is kept in _IDENTITY_SLOTS_SUPERSEDED for the life of the process (doubling bounds them below the
# current table's size).
_IDENTITY_SLOTS: dict[int, torch.Tensor] = {}
_IDENTITY_SLOTS_SUPERSEDED: list[torch.Tensor] = []


def _identity_slots(n: int, device: torch.device) -> torch.Tensor:
    index = _device_index(device)
    table = _IDENTITY_SLOTS.get(index)
    if table is not None and table.numel() >= n:
        return table[:n]
    if torch.cuda.is_current_stream_capturing():
        return torch.arange(n, dtype=torch.int32, device=device)
    size = max(n, 256, 2 * (table.numel() if table is not None else 0))
    table = torch.arange(size, dtype=torch.int32, device=device)
    torch.cuda.current_stream(
        device
    ).synchronize()  # complete before a call on any stream reads it
    previous = _IDENTITY_SLOTS.get(index)
    if previous is not None:
        _IDENTITY_SLOTS_SUPERSEDED.append(previous)
    _IDENTITY_SLOTS[index] = table
    return table[:n]


def _copy_pool_rows(
    initial_in: Optional[torch.Tensor],
    output_in: torch.Tensor,
    slots: torch.Tensor,
    oslots: torch.Tensor,
) -> None:
    """The final state of a batch of empty sequences on the pool programs, on the device: sequence ``n`` stores its
    initial record (slot ``slots[n]`` of ``initial_in``; zero when that slot is outside the pool or there is no
    initial state) at slot ``oslots[n]`` of ``output_in``, and stores nothing when that slot is outside the pool.
    The 16-bit records are copied as they are (rounding a representable value is the identity).  No slot is read on
    the host: a skipped store becomes a duplicate write of a value that is written anyway (another sequence's record
    to its own slot, or slot 0 to itself), so the stores stay deterministic and graph-capturable."""

    rows = int(output_in.shape[0])
    writes = oslots.long()
    write_ok = (writes >= 0) & (writes < rows)
    if initial_in is not None:
        reads = slots.long()
        read_ok = (reads >= 0) & (reads < int(initial_in.shape[0]))
        values = initial_in.index_select(0, torch.where(read_ok, reads, 0))
        values = torch.where(
            read_ok[:, None, None, None],
            values,
            torch.zeros((), dtype=values.dtype, device=values.device),
        )
    else:
        values = torch.zeros(
            (int(slots.numel()),) + tuple(output_in.shape[1:]),
            dtype=output_in.dtype,
            device=output_in.device,
        )
    any_ok = write_ok.any()
    first = torch.argmax(write_ok.to(torch.int32))
    fallback_slot = torch.where(any_ok, writes[first], 0)
    fallback_value = torch.where(any_ok, values[first], output_in[0])
    targets = torch.where(write_ok, writes, fallback_slot)
    values = torch.where(write_ok[:, None, None, None], values, fallback_value)
    output_in.index_copy_(0, targets, values)


class _V6State(NamedTuple):
    """The state of one Blackwell call as its program takes it: the form (``dtype`` as the records spell it,
    ``pool``: a pool program), the initial / output tensors (``None`` when not read / not stored), the read and
    write slot tables (int32; ``None`` for a dense FP32 program) and the rounding (the seed tensor and the round
    count of a 16-bit program; ``None`` / 0 otherwise)."""

    dtype: str
    pool: bool
    initial: Optional[torch.Tensor]
    output: Optional[torch.Tensor]
    slots: Optional[torch.Tensor]
    oslots: Optional[torch.Tensor]
    seed: Optional[torch.Tensor]
    rounds: int


def _blackwell_state(
    arch: str,
    device: torch.device,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    output_final_state: bool,
    state_indices: Optional[torch.Tensor],
    output_state_indices: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    philox_rounds: int,
    N: int,
    H: int,
) -> _V6State:
    """The state form of a Blackwell call and its operands.  One dtype serves the initial and the output state:
    with ``state_indices`` both pools are of the dtype of the ones given (FP32 when neither is); a dense call takes
    the dtype of ``output_state`` when it stores one, else of ``initial_state``.  PR 1's dense call keeps working:
    a BF16 ``initial_state`` with an FP32 (or internally allocated) final state is converted to FP32 once.  A
    dense BF16 / FP16 call runs the pool program on identity slots.  ``rand_seed`` (with ``philox_rounds``) rounds
    a BF16 / FP16 state's stores stochastically and is refused for an FP32 state."""

    pool_mode = state_indices is not None
    if output_final_state and output_state is None and pool_mode:
        raise ValueError(
            "output_final_state with state_indices needs an explicit output_state pool "
            "(the identical view of initial_state or disjoint memory)"
        )
    stored = output_state if output_final_state else None
    for tensor, name in ((initial_state, "initial_state"), (stored, "output_state")):
        if tensor is not None and (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype not in _PREFILL_STATE_DTYPES
        ):
            raise ValueError(
                f"{name} must be a float32, bfloat16 or float16 tensor, got "
                f"{tensor.dtype if isinstance(tensor, torch.Tensor) else type(tensor).__name__}"
            )
    compat = False
    if pool_mode:
        given = [t.dtype for t in (initial_state, stored) if t is not None]
        if len(set(given)) > 1:
            raise ValueError(
                f"initial_state and output_state pools must share one dtype, got {given[0]} and {given[1]}"
            )
        dtype = given[0] if given else torch.float32
    elif stored is not None:
        dtype = stored.dtype
        compat = (
            initial_state is not None
            and initial_state.dtype == torch.bfloat16
            and dtype == torch.float32
        )
        if initial_state is not None and initial_state.dtype != dtype and not compat:
            raise ValueError(
                f"initial_state and output_state must share one dtype, got {initial_state.dtype} and {dtype} "
                "(a bfloat16 initial_state with a float32 output_state is converted)"
            )
    elif initial_state is not None:
        if output_final_state and initial_state.dtype == torch.float16:
            raise ValueError(
                "a float16 initial_state stores a float16 final state: pass a float16 output_state (the final "
                "state allocated for you is float32)"
            )
        compat = initial_state.dtype == torch.bfloat16
        dtype = torch.float32 if compat else initial_state.dtype
    else:
        dtype = torch.float32
    key = _loader.PREFILL_STATE_DTYPES[dtype]
    pool = pool_mode or dtype != torch.float32
    if (key, pool) not in _loader.v6_state_forms(arch):
        raise ValueError(
            f"no exported Cake GDP prefill program on {arch} for a {dtype} "
            f"{'pool' if pool_mode else 'state'}; the route ships the state forms "
            f"{sorted(_loader.v6_state_forms(arch))} (state dtype, slot pool)"
        )
    if rand_seed is not None and dtype == torch.float32:
        raise ValueError(
            "rand_seed rounds the stores of a bfloat16 / float16 state stochastically; a float32 state is "
            "stored as it is"
        )
    if not pool:
        # the dense FP32 programs: one [N, H, 64, 128] record per sequence, as PR 1 ships them
        initial_in = (
            None
            if initial_state is None
            else _state_rows(
                initial_state,
                "initial_state",
                rows=N,
                heads=H,
                device=device,
                writable=False,
            )
        )
        if not output_final_state:
            output_in = None
        elif output_state is None:
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
        return _V6State(key, False, initial_in, output_in, None, None, None, 0)
    rows = None if pool_mode else N
    initial_in = (
        None
        if initial_state is None
        else _prefill_pool(
            initial_state,
            "initial_state",
            rows=rows,
            heads=H,
            device=device,
            dtype=dtype,
            writable=pool_mode,
        )
    )
    output_in = (
        None
        if stored is None
        else _prefill_pool(
            stored,
            "output_state",
            rows=rows,
            heads=H,
            device=device,
            dtype=dtype,
            writable=True,
        )
    )
    if pool_mode:
        slots = _index_input(state_indices, "state_indices", shape=(N,), device=device)
        oslots = (
            slots
            if output_state_indices is None
            else _index_input(
                output_state_indices, "output_state_indices", shape=(N,), device=device
            )
        )
    else:
        slots = oslots = _identity_slots(N, device)
    seed, rounds = None, 0
    if dtype != torch.float32:
        base = initial_in if initial_in is not None else output_in
        seed, rounds = _rounding(base, rand_seed, philox_rounds)
    return _V6State(key, True, initial_in, output_in, slots, oslots, seed, rounds)


def _int32_view(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """A contiguous index tensor as the kernel takes it: an int32 tensor as is, an int64 tensor as its int32 byte
    view (the kernel reads it as int64 under the metadata mask bit); no conversion copy. The fork-order (V6) programs
    read int32 metadata only: ``_run_single_pass`` casts an int64 ``cu_seqlens`` / slot table to int32 for them."""

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
    output_state_indices: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Chunked varlen Gated DeltaProduct prefill on the Cake kernels.

    The body of ``flashinfer.gdp_prefill.chunk_gated_delta_product(..., backend="cake")``; see that function for
    the full documentation of the arguments and semantics.  Every check here reads shapes, dtypes, strides and
    Python scalars only.  On SM100 / SM103 the call is one launch of the Blackwell body over the item table of
    ``prefill_v6_plan`` (the split rules read each sequence alone, short sequences' rows go to the least-loaded
    CTAs; ``num_householder=3``; ``g=None`` / ``beta=None`` filled with ones) on the program of the call's state
    form: the dense FP32 programs, or the pool programs of the state dtype (FP32, BF16 or FP16) for a
    ``state_indices`` pool (``output_state_indices`` its write slots) and for a dense BF16 / FP16 state (identity
    slots); the plan is built from the host lengths, so a captured graph is valid only for the batch it captured
    (other lengths need a new capture or an eager call; a replay over changed device offsets runs no item and writes
    NaN to ``o[0, 0, :2]`` and, with a dense FP32 final state, ``output_state[0, 0, 0, 0]``); the rest of this
    paragraph is SM90's.  Every call is one Cake kernel launch for the whole batch.  A gated call with a
    balanced sequence (a long sequence whose head-major chunks fill a slot per SM, ``prefill_v6_sm90_plan``) whose plan is the lighter
    launch runs the V6 split program of its flags on one persistent CTA per SM over the rows of that plan: each
    balanced sequence cut into slots at block heads (each piece past a cut starts from a zero state a warm-up before
    it and writes its output from the cut, a cut whose gates do not decay within the warm-up being run through),
    every other sequence one row per head, all in the same launch; that program compares the planned offsets with
    ``cu_seqlens`` on the device and, on a replay over changed offsets, computes nothing and writes NaN to
    ``o[0, 0, :2]`` (and to ``output_state[0, 0, 0, 0]`` when a final state is stored).  Any other call runs one
    body for the whole batch (``prefill_batch_body`` over the host sequence lengths): the V6 fork-order body for a
    unit-free call without a merged-class sequence (the body route of record: a sequence under
    ``ROUTE_MIN_CHUNKS_P3`` (384) chunks, or one whose split items are not the cheaper) and for a call reading no
    initial state and keeping no final state; the merged chunked body for
    a unit-gate / unit-beta call (its exact unit program); otherwise the body of the lesser modeled whole-batch
    kernel time (``prefill_batch_costs``: the most loaded persistent CTA under each body).  The V6 program of the
    call's flags runs over the ``N x H`` (sequence, head) tiles on ``min(SM count, N x H)`` persistent CTAs; the
    merged split program of the flags runs over the items of the split rule of record (``prefill_split_table``)
    over every sequence of the batch -- short and empty sequences one item per head, in the same launch -- with a
    workspace clear before it.  No token is gathered or scattered.  The item table is uploaded from pinned host
    memory into a device buffer cached by content (``_item_table``, shared with SM100 / SM103), so with
    ``cu_seqlens_cpu`` given the call captures into a CUDA graph; the body and the table are planned from the host
    lengths, so such a graph is valid only for the batch it captured (other lengths need a new capture or an eager
    call; outside the V6 split program a replay over changed offsets is not supported and its result is
    unspecified).  With ``cu_seqlens_cpu`` given, no device-to-host
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
    if output_state_indices is not None and not pool_mode:
        raise ValueError("output_state_indices needs state_indices (the read slots)")
    # SM100 / SM103: the Blackwell body (one launch for every flag set over a host item table), dense or on slot
    # pools, with an FP32, BF16 or FP16 state.  SM90 runs FP32 state programs alone.
    blackwell = arch in _loader.V6_ARCHES
    if not blackwell and (output_state_indices is not None or rand_seed is not None):
        raise ValueError(
            f"no exported Cake GDP prefill program on {arch} takes "
            f"{'output_state_indices' if output_state_indices is not None else 'rand_seed'}: the SM90 programs read "
            "and store a float32 state (bfloat16 / float16 state and its rounding run on SM100 / SM103)"
        )

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
    if blackwell and (unit_gates or unit_beta):
        # The Blackwell body has no unit-gate / unit-beta build: the unit values are materialized (the same
        # recurrence, one fill each).
        if unit_gates:
            gate = torch.ones((T, H), dtype=torch.float32, device=device)
        if unit_beta:
            beta_in = torch.ones((T * n, H), dtype=torch.float32, device=device)
    cu_in, cu_int64 = _int32_view(cu_seqlens)
    mask = _CU_SEQLENS_INT64 if cu_int64 else 0

    # -- the state
    v6_state = None
    if blackwell:
        v6_state = _blackwell_state(
            arch,
            device,
            initial_state,
            output_state,
            output_final_state,
            state_indices,
            output_state_indices,
            rand_seed,
            philox_rounds,
            N,
            H,
        )
        slots, initial_in, output_in = (
            v6_state.slots,
            v6_state.initial,
            v6_state.output,
        )
    elif pool_mode:
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
    if blackwell:
        _run_blackwell_v6(
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
            v6_state,
            scale_value if scale_value != 0.0 else HEAD_SIZE_K**-0.5,
            N,
            T,
            H,
            n,
            cu_cpu,
        )
    else:
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


class _Sm90Plan(NamedTuple):
    """The launch of one sm_90a call: its body, and for a split program the flat int32 item words, their bytes (the
    device-table cache key) and the item count; ``v6_grid`` is the persistent grid of a V6 split program (0 for the
    other programs, which run ``min(SM count, tiles)`` CTAs)."""

    body: str
    words: Optional[tuple[int, ...]]
    key: Optional[bytes]
    items: int
    v6_grid: int = 0


@functools.lru_cache(maxsize=256)
def _sm90_plan(
    arch: str,
    cu_cpu: tuple[int, ...],
    T: int,
    H: int,
    n: int,
    sm_count: int,
    stores: bool,
    unit_gates: bool,
    unit_beta: bool,
) -> _Sm90Plan:
    """The launch of the whole call.  A gated call whose ``prefill_v6_sm90_plan`` pays (``split``: a long sequence
    whose head-major chunks fill a slot per SM, the plan's most loaded CTA lighter than the V6 program's) runs the
    V6 split program over the rows of that plan, every sequence of the batch in the same launch.  Any other call
    runs the body of ``prefill_batch_body`` over the host offsets: the V6 program over the ``N x H`` tiles, or, for
    the merged gated body, its split program over the item table of the split rule of every sequence of the batch.
    A pure function of its arguments, cached: a model calls every layer with the same batch, and the plan is host
    arithmetic over the lengths."""

    if not (unit_gates or unit_beta):
        v6 = _loader.prefill_v6_sm90_plan(
            cu_cpu, num_heads=H, num_householder=n, sm_count=sm_count, total_tokens=T
        )
        if v6.split:
            words = _loader.v6_sm90_item_words(v6, cu_cpu)
            return _Sm90Plan(
                _loader.ROUTE_BODY_V6,
                words,
                array.array("i", words).tobytes(),
                len(v6.rows),
                v6.grid,
            )
    body = _loader.prefill_batch_body(
        cu_cpu,
        n,
        H,
        sm_count,
        store_final_state=stores,
        unit_gates=unit_gates,
        unit_beta=unit_beta,
        total_tokens=T,
    )
    if body == _loader.ROUTE_BODY_V6 or unit_gates or unit_beta:
        return _Sm90Plan(body, None, None, 0)
    table = _split_table(
        arch, cu_cpu, len(cu_cpu) - 1, T, H, n, sm_count, unit_gates, unit_beta
    )
    if table is None:
        raise ValueError(
            "Cake GDP prefill body route invariant broken: the batch route chose the merged body, so some sequence "
            f"is merged-class and the split rule of record cuts it, yet the rule left the batch whole or {arch} "
            "routes no split program"
        )
    words = tuple(word for item in table for word in item)
    return _Sm90Plan(body, words, array.array("i", words).tobytes(), len(table))


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
    grid: Optional[int] = None,
) -> None:
    """One launch of the prefill program ``name`` over the 23 logical arguments of the prefill records on ``grid``
    persistent CTAs (default ``min(sm_count, tiles)``), which also sizes the tensor-map staging."""

    launch = tiles if grid is None else grid
    grid = min(sm_count, tiles) if grid is None else grid
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
            tiles=launch,
            sm_count=sm_count,
        )


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
    """The one launch of an sm_90a call (``_sm90_plan`` over the host offsets ``cu_cpu``): the V6 split program of the
    call's flags over the rows of ``prefill_v6_sm90_plan`` on one CTA per SM when that plan pays; else the
    body ``prefill_batch_body`` picks for the whole batch -- the V6 program of the call's flags over the ``N x H``
    tiles on ``min(SM count, N x H)`` persistent CTAs, or, for the merged body, the exact unit program over those
    tiles or the split program of the flags over the items of the split rule of every sequence of the batch (tiles =
    the items; short and empty sequences one item per head).  Every V6 program stores a final state; a call keeping
    none gets the discard record.  No token is gathered or scattered.  The item table comes from pinned host memory
    into a cached device buffer (``_item_table``), so the call issues no synchronization and captures into a CUDA
    graph."""

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
    # wanting none gets a discard record); the plain call without a final state is V6 at every length, and the V6
    # plain program stores one too.
    stores = store_final or use_initial or unit_gates or unit_beta
    plan = _sm90_plan(arch, cu_cpu, T, H, n, sm_count, stores, unit_gates, unit_beta)
    body = plan.body
    pool_rows = N
    if use_indices:
        pool_rows = max(
            int(initial_in.shape[0]) if initial_in is not None else 0,
            int(output_in.shape[0]) if output_in is not None else 0,
        )
    flags = dict(
        use_initial_state=use_initial,
        use_state_indices=use_indices,
        store_final_state=stores,
        unit_gates=unit_gates,
        unit_beta=unit_beta,
    )
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
            **({_loader.PREFILL_SPLIT_FLAG: True} if plan.v6_grid else {}),
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
    if plan.v6_grid:
        # The V6 split program: formal 9 carries the item table, the tiles are its rows; the A-inverse workspace
        # holds the N x H seed-done flags, zero at launch (the kernel leaves them zero).
        table_in = _item_table(plan.words, device, key=plan.key)
        ainv_ws = torch.zeros((max(16, 4 * N * H),), dtype=torch.uint8, device=device)
        tiles = plan.items
    elif plan.words is not None:
        # The merged split program: formal 9 carries the item table, the tiles are the items; the A-inverse workspace
        # holds the N x H seed-done flags and two plan-echo words per item, zero at launch.
        table_in = _item_table(plan.words, device, key=plan.key)
        ainv_ws = torch.zeros(
            (4 * (N * H + 2 * plan.items),), dtype=torch.uint8, device=device
        )
        tiles = plan.items
    else:
        table_in = slot_in
        ainv_ws = torch.empty((16,), dtype=torch.uint8, device=device)
        tiles = N * H
    _launch_prefill(
        name,
        device=device,
        device_index=device_index,
        sm_count=sm_count,
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
        scale_value=scale_value,
        num_seqs=N,
        total_tokens=T,
        H=H,
        n=n,
        tiles=tiles,
        mask=mask,
        grid=plan.v6_grid or None,
    )


# ---- the item tables of the persistent programs (the SM90 split items, the Blackwell rows) ----

# Item buffers on the device, by (device, content).  An eager call uploads a new table on a private copy stream and
# waits for that copy alone (a few KB; the compute stream is not synchronized), then keeps the table in a small LRU;
# a CUDA graph captured after an eager call of the same batch (the usual warm-up) reuses that complete table, so the
# graph holds no copy, and the table is then kept for the life of the process (the graph reads it at every replay).
# A table first seen during capture is copied inside the graph from a pinned buffer; both are kept as long as the
# process, but that table is never served to another call: its memory holds the table only once the graph has
# replayed, so a later eager call or another capture of the same batch makes its own complete copy.
_ITEM_TABLES_EAGER: collections.OrderedDict[tuple[int, bytes], torch.Tensor] = (
    collections.OrderedDict()
)
_ITEM_TABLES_EAGER_MAX = 64
_ITEM_TABLES_CAPTURED: dict[tuple[int, bytes], torch.Tensor] = {}
_ITEM_TABLES_IN_GRAPHS: list[tuple[torch.Tensor, torch.Tensor]] = []
_COPY_STREAMS: dict[int, torch.cuda.Stream] = {}


def _item_table(
    words: tuple[int, ...], device: torch.device, *, key: Optional[bytes] = None
) -> torch.Tensor:
    """The int32 ``items`` buffer ``words`` (``v6_item_words`` on SM100 / SM103, the flat split items on SM90) on
    ``device`` (read only): a cached complete buffer, else an upload from pinned host memory (a pageable copy would
    synchronize the compute stream, which CUDA Graph capture refuses).  ``key`` is the bytes of ``words`` as int32
    when the caller has them already."""

    host = None
    if key is None:
        host = torch.tensor(words, dtype=torch.int32)
        key = host.numpy().tobytes()
    index = device.index or 0
    cache_key = (index, key)
    table = _ITEM_TABLES_CAPTURED.get(cache_key)
    if table is not None:
        return table
    if torch.cuda.is_current_stream_capturing():
        table = _ITEM_TABLES_EAGER.pop(cache_key, None)
        if table is not None:
            # a complete table: the graph reads it at every replay, so it is kept and served from now on
            _ITEM_TABLES_CAPTURED[cache_key] = table
            return table
        host = torch.tensor(words, dtype=torch.int32) if host is None else host
        pinned = host.pin_memory()
        table = torch.empty((len(words),), dtype=torch.int32, device=device)
        table.copy_(pinned, non_blocking=True)
        _ITEM_TABLES_IN_GRAPHS.append((pinned, table))
        return table
    table = _ITEM_TABLES_EAGER.get(cache_key)
    if table is not None:
        _ITEM_TABLES_EAGER.move_to_end(cache_key)
        return table
    host = torch.tensor(words, dtype=torch.int32) if host is None else host
    table = torch.empty((len(words),), dtype=torch.int32, device=device)
    copy = _COPY_STREAMS.setdefault(index, torch.cuda.Stream(device=device))
    with torch.cuda.stream(copy):
        table.copy_(host.pin_memory(), non_blocking=True)
    copy.synchronize()  # the table is complete before any stream reads it
    _ITEM_TABLES_EAGER[cache_key] = table
    if len(_ITEM_TABLES_EAGER) > _ITEM_TABLES_EAGER_MAX:
        _ITEM_TABLES_EAGER.popitem(last=False)
    return table


# ---- the Blackwell prefill body (SM100 / SM103) ----


def _run_blackwell_v6(
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
    state: _V6State,
    scale_value,
    N,
    T,
    H,
    n,
    cu_cpu,
) -> None:
    """One launch of the Blackwell prefill body over the item table of ``prefill_v6_plan``: the split build over the
    per-CTA lists of the ``split_table`` rows, or the block build (see the BLACKWELL V6 LAUNCH PLAN of the
    registry), of the program of the call's state form (``state``: the dense FP32 programs, or the pool programs of
    its dtype with the slot tables, the pools' slot strides and the rounding).  Every input is prepared on the
    device or uploaded from pinned memory: the call issues no synchronization and captures into a CUDA graph (the
    item table cached on the device after an eager call of the same batch, the split build's workspace cleared in
    the graph, then the kernel)."""

    initial_in, output_in = state.initial, state.output
    if T == 0:
        # Every sequence is empty: its final state is its initial state (the kernel wants a token).
        if output_in is None:
            return
        if state.pool:
            _copy_pool_rows(initial_in, output_in, state.slots, state.oslots)
        elif initial_in is not None:
            output_in.copy_(initial_in)
        else:
            output_in.zero_()
        return
    plan = _loader.prefill_v6_plan(
        cu_cpu,
        num_heads=H,
        num_householder=n,
        sm_count=sm_count,
        arch=arch,
        total_tokens=T,
    )
    name = _loader.selected_v6_module(
        arch, plan, state_dtype=state.dtype, pool=state.pool
    )
    if mask & _CU_SEQLENS_INT64:
        # the body reads int32 offsets: one small copy of the N + 1 entries
        cu_in = cu_in.view(torch.int64).to(torch.int32)
    # the rows, the CTAs' row offsets and the planned offsets the kernel checks against the device ``cu_seqlens``
    items = _item_table(_loader.v6_item_words(arch, plan, cu_cpu), device)
    if plan.split:
        # the N x H seed-done flags (the kernel leaves them zero) and two plan-echo words per row, zero at launch
        split_ws = torch.zeros(
            (4 * (N * H + 2 * len(plan.rows)),), dtype=torch.uint8, device=device
        )
    else:
        split_ws = torch.empty(
            (_loader.V6_SPLIT_WS_PLACEHOLDER,), dtype=torch.uint8, device=device
        )
    flags = (_loader.V6_FLAG_INITIAL_STATE if initial_in is not None else 0) | (
        _loader.V6_FLAG_FINAL_STATE if output_in is not None else 0
    )
    placeholder = None
    if initial_in is None or output_in is None:
        # an absent state, which the flags keep the kernel from reading or writing: one element of the program's
        # state dtype
        dtype = next(
            d for d, key in _loader.PREFILL_STATE_DTYPES.items() if key == state.dtype
        )
        placeholder = torch.empty((1,), dtype=dtype, device=device)
    values = {
        "q": q_in,
        "k": k_in,
        "v": v_in,
        "o": out,
        "gate": gate,
        "beta": beta_in,
        "cu_seqlens": cu_in,
        "items": items,
        "initial_state": initial_in if initial_in is not None else placeholder,
        "output_state": output_in if output_in is not None else placeholder,
        "tensormap_workspace": torch.empty(
            (plan.grid * _loader.V6_TENSORMAP_BYTES,), dtype=torch.uint8, device=device
        ),
        "split_workspace": split_ws,
        "state_stride_slot": H * _STATE_ELEMENTS,
        "scale": scale_value,
        "num_items": len(plan.rows),
        "total_tokens": T,
        "num_heads": H,
        "num_householder": n,
        "flags": flags,
        "num_seqs": N,
    }
    if state.pool:
        values.update(
            state_indices=state.slots,
            output_state_indices=state.oslots,
            initial_state_stride_slot=_slot_stride(initial_in, H),
            output_state_stride_slot=_slot_stride(output_in, H),
        )
    if state.dtype != "f32":
        values.update(rand_seed=state.seed, philox_rounds=state.rounds)
    with torch.cuda.device(device):
        _loader.run_module(
            name,
            values,
            batch=N,
            heads=H,
            device_index=device_index,
            # the plan's grid (min(SM count, grid) == grid)
            tiles=plan.grid,
            sm_count=sm_count,
        )


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
