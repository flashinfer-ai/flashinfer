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

Cake route of ``cutlass_fused_moe(backend="cake")`` for BF16 activations against
NVFP4 expert weights (the "W4A16 Split" form): BF16 ``x`` times E2M1 codes with
E4M3 block-16 scales, decoded to BF16 inside the kernels and accumulated in FP32
on ``tcgen05`` (sm_100a / sm_103a).

Launch chain, with programmatic dependent launch on every boundary (the FC1
program of the ``n = 8`` cell gathers the token rows itself and skips
``moe_permute``)::

    moe_sort (FlashInfer) -> moe_permute (FlashInfer) -> Cake FC1 (gated SwiGLU -> BF16)
      -> Cake FC2 (-> BF16) -> moe_unpermute (FlashInfer, FP32 FMA in top-k slot order)

Weights are the canonical ``PrequantizedMoEWeights`` tensors and are consumed in
place: ``w13`` uint8 ``[E_local, 2I, H/2]`` (two E2M1 codes per byte, low nibble
first; rows ``[0, I)`` are the gate projection, rows ``[I, 2I)`` the up
projection), ``w13_scale`` float8_e4m3fn ``[E_local, 2I, H/16]``, ``w2`` uint8
``[E_local, H, I/2]``, ``w2_scale`` float8_e4m3fn ``[E_local, H, I/16]``.  No
interleave, no scale re-layout and no BF16 expansion is applied or persisted.

Numerics: ``deq(w) = E2M1 x E4M3`` exactly in BF16; FC1 = ``BF16(x) x deq(w13)``
accumulated in FP32; per-expert ``fc1_alpha`` applied in FP32, optional
NaN-propagating clamp (``gate = min(gate, limit)``, ``up = clamp(up, -limit,
limit)``), ``out = up * (gate * sigmoid(gate))`` in FP32, one BF16 rounding;
FC2 = ``BF16(h) x deq(w2)`` in FP32, ``fc2_alpha``, one BF16 rounding; the
routing weight multiplies after FC2 inside ``moe_unpermute``, which sums the
BF16 route terms in FP32 in ascending slot order (``-1`` routes skipped, an
all-masked token writes zero).  No atomics touch a data sum.

Every size is a host function of the token count, the kernels exit by the
device-side ``num_non_exiting_tiles`` count and the route performs no host
synchronization, so a call is CUDA-graph capturable and bit-reproducible on
replay.  Scratch is the caller's ``workspace_buffer`` (sized by
:func:`cake_w4a16_workspace_size`); with ``output`` passed, this module
allocates nothing on the submit path.  The one allocation of the chain is
FlashInfer's: ``moe_sort`` exposes no ``out_`` buffer for its internal
``expert_counts`` (``2 * num_experts`` int32) and allocates it per eager call
when ``num_tokens > 1024``, in this route exactly as in FlashInfer's own W4A16
route; CUDA-graph replay performs no such allocation.

Program loading
---------------
The two kernels ship as the in-tree Cake family ``cake_fused_moe_w4a16``: the
generated loader ``flashinfer/jit/cake_fused_moe_w4a16.py`` publishes one
:class:`~flashinfer.jit.cake_program.CakeProgram` constant per cell, named
``FC1_N<n>[_CG2]`` / ``FC2_N<n>[_CG2]`` (:data:`PROGRAM_IDENTIFIERS`), each
shipped for ``sm_100a`` and ``sm_103a`` under one launch record (``use_pdl=True``,
not cooperative, 512 threads, cluster ``(cta_group, 1, 1)``).  This module
assembles the binding's positional call from the program's recorded argument
names (:data:`FC1_ARGS`, :data:`FC2_ARGS`), derives the grid
(:func:`cake_w4a16_grid`) and launches through ``program.entry(target)`` under
``tvm_ffi.use_torch_stream()``.  The FC1 program of the ``n = 8`` cell takes
the route map and gathers the token rows from ``x`` itself, so the route skips
``moe_permute`` for it.  :func:`cake_w4a16_available` tells
whether the loader imports and publishes every program, so tests skip cleanly
without it.  The programs need CUDA :data:`CAKE_W4A16_MIN_CUDA` or newer
(:func:`cake_w4a16_toolkit_unavailable` says why an older toolkit cannot run
them); the loader refuses to build them on an older toolkit.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import tvm_ffi

from ..tllm_enums import ActivationType
from ..utils import device_support_pdl, get_compute_capability, get_device_properties

__all__ = [
    "CAKE_W4A16_MIN_CUDA",
    "CELLS",
    "FC1_ARGS",
    "FC2_ARGS",
    "GRID_ARGS",
    "LOADER_MODULE",
    "PROGRAM_IDENTIFIERS",
    "Cell",
    "cake_w4a16_available",
    "cake_w4a16_fused_moe",
    "cake_w4a16_grid",
    "cake_w4a16_prepare_workspace",
    "cake_w4a16_toolkit_unavailable",
    "cake_w4a16_workspace_size",
    "is_cake_w4a16_form",
    "select_cell",
]

# Weight-stream K step of both kernels: one TMA stage decodes 256 K elements.
_K_STAGE = 256
# Output features one FC1 CTA produces per M tile (one 64-row gate box plus the
# matching 64-row up box) and output channels one FC2 CTA produces per M tile.
_FC1_M_TILE = 64
_FC2_M_TILE = 128
_ALIGN = 128
_STATE_ATTR = "_flashinfer_cake_w4a16_workspace"
#: The generated loader of the family (produced by the Cake export).
LOADER_MODULE = "flashinfer.jit.cake_fused_moe_w4a16"
#: The three trailing grid arguments of every generated binding.
GRID_ARGS: Tuple[str, str, str] = ("grid_x", "grid_y", "grid_z")
# Binding argument names of the two kernels, in the kernels' parameter order.
# ``W`` / ``S`` are the canonical packed codes and scale bytes (TMA descriptors
# encoded by the binding from the tensors; the scales viewed as ``uint8``, the
# descriptors' element type), ``X`` the BF16 B rows (``hidden_workspace`` for
# FC1, ``intermediate`` for FC2), ``C`` the BF16 output rows, the three
# ``moe_sort`` tables, the per-expert FP32 ``alpha``, FC1's runtime clamp, the
# reduction length ``K``, the per-CTA M-tile count ``num_m_tiles`` and the same
# count once more as ``m_tile_div``, the fast divisor of the kernel's work walk
# (the binding builds it from the plain integer).
FC1_ARGS: Tuple[str, ...] = (
    "W",
    "S",
    "X",
    "C",
    "tile_idx_to_expert_idx",
    "tile_idx_to_mn_limit",
    "num_non_exiting_tiles",
    "alpha",
    "swiglu_limit",
    "K",
    "num_m_tiles",
    "m_tile_div",
)
FC2_ARGS: Tuple[str, ...] = tuple(name for name in FC1_ARGS if name != "swiglu_limit")


@dataclass(frozen=True)
class Cell:
    """One shipped kernel configuration: ``n`` routed rows per route tile (the
    ``tcgen05`` N of the swap-AB MMA and ``moe_sort``'s ``tile_tokens_dim``) and
    the CTA group (1, or 2 for the 2-CTA ``M = 256`` instruction)."""

    n: int
    cta_group: int = 1

    def __post_init__(self) -> None:
        if self.cta_group not in (1, 2):
            raise ValueError(f"cell cta_group must be 1 or 2, got {self.cta_group}")
        if self.cta_group == 1 and self.n not in (8, 16, 32, 64):
            raise ValueError(
                f"the single-CTA cells ship for n = 8, 16, 32 and 64, got n = {self.n}"
            )
        if self.cta_group == 2 and self.n != 128:
            raise ValueError(f"the 2-CTA cell ships for n = 128 only, got n = {self.n}")

    @property
    def suffix(self) -> str:
        return f"N{self.n}" + ("_CG2" if self.cta_group == 2 else "")

    @property
    def cluster_ctas(self) -> int:
        return self.cta_group


#: The five cells the route ships (the ones :func:`select_cell` picks among),
#: in loader order.
CELLS: Tuple[Cell, ...] = (Cell(8), Cell(16), Cell(32), Cell(64), Cell(128, 2))
#: Loader identifier of every program: ``FC1_N8`` ... ``FC2_N128_CG2``.
PROGRAM_IDENTIFIERS: Tuple[str, ...] = tuple(
    f"{kernel}_{cell.suffix}" for kernel in ("FC1", "FC2") for cell in CELLS
)
# Deviation of the per-expert row count above which the one-tile-per-expert rule
# picks the next n: rows per touched expert are ~Binomial(R, 1/E_local), and
# n >= mean + 3.1 sd keeps P(rows > n) below 1e-3.
_TAIL_SIGMAS = 3.1


def select_cell(num_tokens: int, top_k: int, num_experts: int) -> Cell:
    """The cell for a call: a host function of the token count and the routing
    geometry only (never of the routing itself), so the choice is CUDA-graph
    stable.

    Rule (seed table of the design, confirmed per shape row against the
    FlashInfer tuner in the alignment round): with ``m = num_tokens * top_k /
    num_experts`` expected rows per expert, take the smallest ``n`` in 8, 16,
    32, 64, 128 with ``n >= m + 3.1 sqrt(m)`` (one route tile per expert with
    probability > 0.999); above that, the 2-CTA ``n = 128`` cell.  The ``n =
    128`` cell always runs as the 2-CTA pair.
    """
    if num_tokens < 0 or top_k <= 0 or num_experts <= 0:
        raise ValueError("num_tokens must be >= 0 and top_k, num_experts positive")
    mean_rows = num_tokens * top_k / num_experts
    needed = math.ceil(mean_rows + _TAIL_SIGMAS * math.sqrt(mean_rows))
    for n in (8, 16, 32, 64):
        if needed <= n:
            return Cell(n)
    return Cell(128, 2)


@dataclass(frozen=True)
class _ProgramHandle:
    """One loaded program: ``args`` are the binding's argument names in call
    order, ending in :data:`GRID_ARGS`; ``launch(*values)`` performs the launch
    with the values in that order."""

    args: Tuple[str, ...]
    entry: Callable[..., None]

    def launch(self, *values: Any) -> None:
        self.entry(*values)


class _Programs:
    """The shipped loader: ``CakeProgram`` constants resolved to the device's
    exact target, loaded once per (identifier, target)."""

    def __init__(self) -> None:
        self._handles: Dict[Tuple[str, str], _ProgramHandle] = {}

    @staticmethod
    @functools.cache
    def _loader() -> Any:
        import importlib

        return importlib.import_module(LOADER_MODULE)

    def handle(self, kernel: str, cell: Cell, device: torch.device) -> _ProgramHandle:
        identifier = f"{kernel}_{cell.suffix}"
        program = getattr(self._loader(), identifier, None)
        if program is None:
            raise RuntimeError(
                f"{LOADER_MODULE} publishes no program {identifier!r}; the Cake W4A16 "
                f"family ships {list(PROGRAM_IDENTIFIERS)}"
            )
        target = program.target_for(get_compute_capability(device))
        if target is None:
            major, minor = get_compute_capability(device)
            raise RuntimeError(
                "the Cake W4A16 route requires an exact sm_100a or sm_103a build; "
                f"{identifier} ships {list(program.targets)} and the device is {major}.{minor}"
            )
        key = (identifier, target)
        handle = self._handles.get(key)
        if handle is None:
            unavailable = cake_w4a16_toolkit_unavailable()
            if unavailable is not None:
                raise RuntimeError(f"{identifier}: {unavailable}")
            record = program.launch_record()
            if not record["use_pdl"] or record["cooperative"]:
                raise RuntimeError(
                    f"{identifier}: the Cake W4A16 programs launch with PDL and are not "
                    f"cooperative; the loader records {record}"
                )
            if tuple(program.cluster) != (cell.cta_group, 1, 1):
                raise RuntimeError(
                    f"{identifier}: expected cluster {(cell.cta_group, 1, 1)}, "
                    f"the loader records {tuple(program.cluster)}"
                )
            if tuple(program.args[-3:]) != GRID_ARGS:
                raise RuntimeError(
                    f"{identifier}: the binding's arguments end with "
                    f"{list(program.args[-3:])}, not {list(GRID_ARGS)}"
                )
            handle = _ProgramHandle(tuple(program.args), program.entry(target))
            self._handles[key] = handle
        return handle


_programs = _Programs()


def cake_w4a16_available() -> bool:
    """Whether the shipped loader imports and publishes every program."""
    try:
        loader = _Programs._loader()
    except ImportError:
        return False
    return all(hasattr(loader, identifier) for identifier in PROGRAM_IDENTIFIERS)


# Built with CUDA 12.9, the FC1 programs trap at their TMEM deallocation
# (``tcgen05.dealloc``: the ptxas 12.9 tcgen05 guardrail reports unallocated
# columns being deallocated after its bookkeeping words in the reserved shared
# memory window are overwritten during the kernel); CUDA 13.0 and newer
# toolkits build and run the same programs cleanly, so the route refuses an
# older toolkit instead of failing inside a launch.
CAKE_W4A16_MIN_CUDA = "13.0"


def cake_w4a16_toolkit_unavailable() -> Optional[str]:
    """Why this CUDA toolkit cannot run the Cake W4A16 programs, or ``None``."""

    from ..jit.cpp_ext import get_cuda_version, is_cuda_version_at_least

    if is_cuda_version_at_least(CAKE_W4A16_MIN_CUDA):
        return None
    return (
        f"the Cake W4A16 programs need CUDA {CAKE_W4A16_MIN_CUDA} or newer (built "
        "with CUDA 12.9 their FC1 kernels trap at the TMEM deallocation guardrail); "
        f"the CUDA toolkit found is {get_cuda_version()}"
    )


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(
        get_device_properties(torch.device("cuda", device_index)).multi_processor_count
    )


@functools.cache
def _max_active_clusters(device_index: int, cluster_ctas: int) -> int:
    """Co-resident clusters of ``cluster_ctas`` CTAs at one CTA per SM: the
    hardware's answer when the CuTe DSL probe is available (what the FlashInfer
    W4A16 route sizes its grid with), else the SM count divided by the cluster
    size (a static persistent grid larger than the co-resident capacity runs its
    excess clusters as a tail wave; correctness is unaffected)."""
    sm_count = _sm_count(device_index)
    if cluster_ctas == 1:
        return sm_count
    try:
        from ..cute_dsl.utils import get_max_active_clusters

        with torch.cuda.device(device_index):
            clusters = int(get_max_active_clusters(cluster_ctas))
    except Exception:  # noqa: BLE001 - the probe is optional; the bound below is safe
        clusters = 0
    if clusters <= 0 or clusters > sm_count // cluster_ctas:
        clusters = max(1, sm_count // cluster_ctas)
    return clusters


def cake_w4a16_grid(cell: Cell, device: torch.device) -> Tuple[int, int, int]:
    """The persistent grid of both kernels for ``cell`` on ``device``: one
    dimension of ``cluster CTAs x max active clusters`` CTAs (the kernels walk
    their work items by ``blockIdx.x``, or by the cluster index at
    ``cta_group=2``).  Never empty: at zero routed rows every CTA exits on the
    device-side tile count."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return (
        cell.cluster_ctas * _max_active_clusters(int(index), cell.cluster_ctas),
        1,
        1,
    )


# ----------------------------------------------------------------------------
# Workspace


def _fi_moe_utils() -> Any:
    from .cute_dsl import moe_utils

    return moe_utils


def _fi_moe_finalize() -> Any:
    from .cute_dsl.blackwell import moe_finalize

    return moe_finalize


def _route_geometry(
    num_tokens: int, top_k: int, num_local_experts: int, cell: Cell
) -> Tuple[int, int]:
    """``(max_tiles, route_slots)``: the tight tile bound of ``moe_sort`` for the
    worst routing of ``num_tokens * top_k`` routes over the local experts, and
    the padded row count both kernels address."""
    max_tiles = int(
        _fi_moe_utils().get_max_num_tiles(num_tokens, top_k, num_local_experts, cell.n)
    )
    return max_tiles, max_tiles * cell.n


def _layout(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    top_k: int,
    cell: Cell,
) -> Tuple[Dict[str, Tuple[int, int, torch.dtype, Tuple[int, ...]]], int]:
    """Byte layout of the workspace for one token count and cell: ``name ->
    (offset, nbytes, dtype, shape)`` and the total.  The two default-alpha
    regions come first at fixed offsets so :func:`cake_w4a16_prepare_workspace`
    fills them once for every token count."""
    max_tiles, route_slots = _route_geometry(num_tokens, top_k, num_local_experts, cell)
    regions: List[Tuple[str, torch.dtype, Tuple[int, ...]]] = [
        ("fc1_alpha_ones", torch.float32, (num_local_experts,)),
        ("fc2_alpha_ones", torch.float32, (num_local_experts,)),
        ("tile_idx_to_expert_idx", torch.int32, (max_tiles,)),
        ("tile_idx_to_mn_limit", torch.int32, (max_tiles,)),
        ("expanded_idx_to_permuted_idx", torch.int32, (num_tokens, top_k)),
        ("permuted_idx_to_expanded_idx", torch.int32, (route_slots,)),
        ("total_num_padded_tokens", torch.int32, (1,)),
        ("num_non_exiting_tiles", torch.int32, (1,)),
        # FC1's B rows (the permuted BF16 input); FC2 writes its expert output
        # over the same rows once FC1 has consumed them (its PDL wait).
        ("hidden_workspace", torch.bfloat16, (route_slots, hidden_size)),
        ("intermediate", torch.bfloat16, (route_slots, intermediate_size)),
    ]
    layout: Dict[str, Tuple[int, int, torch.dtype, Tuple[int, ...]]] = {}
    offset = 0
    for name, dtype, shape in regions:
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        layout[name] = (offset, nbytes, dtype, shape)
        offset = (offset + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    return layout, offset


def cake_w4a16_workspace_size(
    max_num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    top_k: int,
) -> int:
    """Bytes of the caller-owned workspace covering every token count up to
    ``max_num_tokens`` and every cell (the row padding differs per cell, so the
    bound is the maximum over cells at the maximum token count).  Host
    arithmetic only."""
    _check_geometry(hidden_size, intermediate_size, num_local_experts, top_k)
    if max_num_tokens <= 0:
        raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
    return max(
        _layout(
            max_num_tokens,
            hidden_size,
            intermediate_size,
            num_local_experts,
            top_k,
            cell,
        )[1]
        for cell in CELLS
    )


def _check_geometry(
    hidden_size: int, intermediate_size: int, num_local_experts: int, top_k: int
) -> None:
    if hidden_size <= 0 or hidden_size % _K_STAGE:
        raise ValueError(
            f"hidden_size must be a positive multiple of {_K_STAGE}, got {hidden_size}"
        )
    if intermediate_size <= 0 or intermediate_size % _K_STAGE:
        raise ValueError(
            f"intermediate_size must be a positive multiple of {_K_STAGE}, got {intermediate_size}"
        )
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
    if not 1 <= top_k <= 32:
        raise ValueError(f"top_k must be in [1, 32], got {top_k}")


def _views(
    workspace: torch.Tensor,
    layout: Mapping[str, Tuple[int, int, torch.dtype, Tuple[int, ...]]],
) -> Dict[str, torch.Tensor]:
    base = workspace.view(torch.uint8)
    views = {}
    for name, (offset, nbytes, dtype, shape) in layout.items():
        views[name] = base[offset : offset + nbytes].view(dtype).view(*shape)
    return views


def cake_w4a16_prepare_workspace(
    workspace_buffer: torch.Tensor,
    *,
    max_num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    top_k: int,
) -> int:
    """Bind ``workspace_buffer`` to one geometry and fill its default
    per-expert alphas (ones) once.  Returns the bytes the geometry needs.  The
    submit path prepares lazily when the caller skips this, which performs the
    one fill at that call; prepare before a CUDA-graph capture to keep the
    captured sequence to the five launches."""
    if not isinstance(workspace_buffer, torch.Tensor) or not workspace_buffer.is_cuda:
        raise TypeError("workspace_buffer must be a CUDA tensor")
    if not workspace_buffer.is_contiguous() or workspace_buffer.data_ptr() % _ALIGN:
        raise ValueError(
            f"workspace_buffer must be contiguous and {_ALIGN}-byte aligned"
        )
    needed = cake_w4a16_workspace_size(
        max_num_tokens, hidden_size, intermediate_size, num_local_experts, top_k
    )
    nbytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if nbytes < needed:
        raise ValueError(
            f"workspace_buffer holds {nbytes} bytes; the Cake W4A16 route needs {needed} for "
            f"max_num_tokens={max_num_tokens}, hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}, num_local_experts={num_local_experts}, top_k={top_k}"
        )
    key = (hidden_size, intermediate_size, num_local_experts, top_k)
    state = getattr(workspace_buffer, _STATE_ATTR, None)
    if state is None or state["key"] != key or state["max_tokens"] < max_num_tokens:
        layout, _ = _layout(
            max_num_tokens,
            hidden_size,
            intermediate_size,
            num_local_experts,
            top_k,
            CELLS[0],
        )
        views = _views(workspace_buffer, layout)
        views["fc1_alpha_ones"].fill_(1.0)
        views["fc2_alpha_ones"].fill_(1.0)
        state = {"key": key, "max_tokens": max_num_tokens, "calls": {}}
        setattr(workspace_buffer, _STATE_ATTR, state)
    return needed


def _call_views(
    workspace_buffer: torch.Tensor,
    num_tokens: int,
    cell: Cell,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    top_k: int,
) -> Dict[str, Any]:
    state = getattr(workspace_buffer, _STATE_ATTR, None)
    key = (hidden_size, intermediate_size, num_local_experts, top_k)
    if state is None or state["key"] != key or state["max_tokens"] < num_tokens:
        cake_w4a16_prepare_workspace(
            workspace_buffer,
            max_num_tokens=max(
                num_tokens, state["max_tokens"] if state and state["key"] == key else 0
            ),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )
        state = getattr(workspace_buffer, _STATE_ATTR)
    call = state["calls"].get((num_tokens, cell))
    if call is None:
        layout, _ = _layout(
            num_tokens, hidden_size, intermediate_size, num_local_experts, top_k, cell
        )
        views = _views(workspace_buffer, layout)
        max_tiles, route_slots = _route_geometry(
            num_tokens, top_k, num_local_experts, cell
        )
        call = {"views": views, "max_tiles": max_tiles, "route_slots": route_slots}
        state["calls"][(num_tokens, cell)] = call
    return call


# ----------------------------------------------------------------------------
# Contract


def _check_tensor(
    tensor: Any,
    name: str,
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")
    return tensor


def _resolve_limit(swiglu_limit: Any) -> float:
    """The FC1 clamp as the kernel's FP32 scalar: ``None`` is no clamp
    (``+inf`` is the identity under NaN-propagating min/max).  A tensor must be
    a one-element CPU tensor: reading a device value would synchronize."""
    if swiglu_limit is None:
        return math.inf
    if isinstance(swiglu_limit, torch.Tensor):
        if swiglu_limit.numel() != 1:
            raise ValueError(
                "swiglu_limit must hold one model-wide value for the Cake W4A16 route"
            )
        if swiglu_limit.is_cuda:
            raise ValueError(
                "swiglu_limit must be a Python float or a CPU tensor for the Cake W4A16 route "
                "(the kernel takes the clamp by value; a device tensor would need a host sync)"
            )
        swiglu_limit = swiglu_limit.item()
    limit = float(swiglu_limit)
    if math.isnan(limit) or limit <= 0.0:
        raise ValueError(f"swiglu_limit must be positive, got {limit}")
    return limit


def _resolve_alpha(
    alpha: Any,
    name: str,
    *,
    default: torch.Tensor,
    num_local_experts: int,
    device: torch.device,
) -> torch.Tensor:
    if alpha is None:
        return default
    return _check_tensor(
        alpha, name, shape=(num_local_experts,), dtype=torch.float32, device=device
    )


def is_cake_w4a16_form(options: Mapping[str, Any]) -> bool:
    """Whether a ``cutlass_fused_moe`` (or ``cutlass_fused_moe_workspace_size``)
    call is the BF16-activation x NVFP4-weight form this module serves: SwiGLU
    activation, BF16 input, uint8 packed weights with float8_e4m3fn block
    scales as the first two ``quant_scales``, and a geometry the kernels take
    (hidden and intermediate sizes multiples of 256, ``top_k`` at most 32).  A
    call outside this form belongs to the other ``backend="cake"`` route, whose
    own contract check reports it.  Host metadata only: no tensor is read."""
    if options.get("activation_type") != ActivationType.Swiglu:
        return False
    if "input" not in options:
        return (
            options.get("x_dtype") == torch.bfloat16
            and options.get("weight_dtype") == torch.uint8
            and not options.get("use_deepseek_fp8_block_scale")
            and not options.get("use_mxfp8_act_scaling")
            and _serves_geometry(
                options.get("hidden_size"),
                options.get("intermediate_size"),
                options.get("top_k"),
            )
        )
    x = options.get("input")
    w13 = options.get("fc1_expert_weights")
    w2 = options.get("fc2_expert_weights")
    routes = options.get("token_selected_experts")
    scales: Sequence[Any] = options.get("quant_scales") or ()
    return (
        isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and isinstance(w13, torch.Tensor)
        and w13.dtype == torch.uint8
        and w13.dim() == 3
        and isinstance(w2, torch.Tensor)
        and w2.dim() == 3
        and isinstance(routes, torch.Tensor)
        and routes.dim() == 2
        and len(scales) >= 2
        and isinstance(scales[0], torch.Tensor)
        and scales[0].dtype == torch.float8_e4m3fn
        and _serves_geometry(x.shape[1], w2.shape[2] * 2, routes.shape[1])
    )


def _serves_geometry(hidden_size: Any, intermediate_size: Any, top_k: Any) -> bool:
    """Whether the kernels take the geometry: positive multiples of the 256-wide
    K stage and ``1 <= top_k <= 32`` (``None`` or a non-integer is not served)."""
    try:
        hidden, inter, k = int(hidden_size), int(intermediate_size), int(top_k)
    except (TypeError, ValueError):
        return False
    return (
        hidden > 0
        and hidden % _K_STAGE == 0
        and inter > 0
        and inter % _K_STAGE == 0
        and 1 <= k <= 32
    )


_UNSUPPORTED_FLAGS = (
    "min_latency_mode",
    "use_deepseek_fp8_block_scale",
    "use_mxfp8_act_scaling",
    "use_packed_weights",
    "use_wfp4afp8_humming",
    "enable_alltoall",
)


def _cake_w4a16_workspace_size(options: Mapping[str, Any]) -> int:
    """``cutlass_fused_moe_workspace_size(backend="cake")`` for the W4A16 form."""
    if options["activation_type"] != ActivationType.Swiglu:
        raise ValueError(
            "the Cake W4A16 route requires activation_type=ActivationType.Swiglu"
        )
    if options["output_dtype"] != torch.bfloat16:
        raise ValueError(
            "the Cake W4A16 route writes BF16; output_dtype must be torch.bfloat16"
        )
    if options["x_dtype"] != torch.bfloat16 or options["weight_dtype"] != torch.uint8:
        raise ValueError(
            "the Cake W4A16 route takes BF16 input and uint8 packed NVFP4 weights"
        )
    for flag in _UNSUPPORTED_FLAGS:
        if options.get(flag):
            raise ValueError(f"the Cake W4A16 route does not take {flag}")
    ep_size = int(options["ep_size"])
    num_experts_total = int(options["num_experts_total"])
    if ep_size <= 0 or num_experts_total % ep_size:
        raise ValueError("num_experts_total must be a positive multiple of ep_size")
    return cake_w4a16_workspace_size(
        int(options["max_num_tokens"]),
        int(options["hidden_size"]),
        int(options["intermediate_size"]),
        num_experts_total // ep_size,
        int(options["top_k"]),
    )


def _cake_w4a16_fused_moe(options: Mapping[str, Any]) -> torch.Tensor:
    """``cutlass_fused_moe(backend="cake")`` for the W4A16 form: ``options`` is
    that function's ``locals()``.  ``quant_scales`` is ``[w13_scale, w2_scale]``
    optionally followed by ``fc1_alpha`` and ``fc2_alpha`` (FP32
    ``[num_local_experts]``); ``swiglu_limit`` is the model-wide clamp as a float
    or one-element CPU tensor; expert parallelism follows ``ep_size`` /
    ``ep_rank`` (the weights are the rank's contiguous expert slice)."""
    if options["activation_type"] != ActivationType.Swiglu:
        raise ValueError(
            "the Cake W4A16 route requires activation_type=ActivationType.Swiglu"
        )
    if options["output_dtype"] != torch.bfloat16:
        raise ValueError(
            "the Cake W4A16 route writes BF16; output_dtype must be torch.bfloat16"
        )
    for flag in _UNSUPPORTED_FLAGS:
        if options.get(flag):
            raise ValueError(f"the Cake W4A16 route does not take {flag}")
    if options["cluster_size"] != 1 or options["cluster_rank"] != 0:
        raise ValueError("the Cake W4A16 route performs no cross-rank collective")
    if options["profile_ids"] is not None:
        raise ValueError(
            "the Cake W4A16 route selects its cell from the token count; no profile ids"
        )
    for name in (
        "fc1_expert_biases",
        "fc2_expert_biases",
        "input_sf",
        "swiglu_alpha",
        "swiglu_beta",
    ):
        if options[name] is not None:
            raise ValueError(f"the Cake W4A16 route does not take {name}")
    if options["workspace_buffer"] is None:
        raise ValueError(
            "the Cake W4A16 route needs a caller-owned workspace_buffer "
            "(cutlass_fused_moe_workspace_size(..., backend='cake'))"
        )
    scales = list(options["quant_scales"])
    if len(scales) not in (2, 4):
        raise ValueError(
            "quant_scales must be [w13_scale, w2_scale] or [w13_scale, w2_scale, fc1_alpha, fc2_alpha]"
        )
    w13 = options["fc1_expert_weights"]
    num_local_experts = int(w13.shape[0])
    ep_size, ep_rank = int(options["ep_size"]), int(options["ep_rank"])
    if ep_size <= 0 or not 0 <= ep_rank < ep_size:
        raise ValueError("ep_rank must be in [0, ep_size)")
    return cake_w4a16_fused_moe(
        options["input"],
        options["token_selected_experts"],
        options["token_final_scales"],
        w13,
        scales[0],
        options["fc2_expert_weights"],
        scales[1],
        num_experts=num_local_experts * ep_size,
        local_expert_offset=ep_rank * num_local_experts,
        output=options["output"],
        workspace_buffer=options["workspace_buffer"],
        fc1_alpha=scales[2] if len(scales) == 4 else None,
        fc2_alpha=scales[3] if len(scales) == 4 else None,
        swiglu_limit=options["swiglu_limit"],
        enable_pdl=options["enable_pdl"],
    )


# ----------------------------------------------------------------------------
# Route


def cake_w4a16_fused_moe(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    num_experts: int,
    local_expert_offset: int = 0,
    output: Optional[torch.Tensor] = None,
    workspace_buffer: torch.Tensor,
    fc1_alpha: Optional[torch.Tensor] = None,
    fc2_alpha: Optional[torch.Tensor] = None,
    swiglu_limit: Any = None,
    enable_pdl: Optional[bool] = None,
    cell: Optional[Cell] = None,
) -> torch.Tensor:
    """BF16 ``[T, H]`` MoE output of BF16 ``x`` against the canonical NVFP4
    experts: ``moe_sort`` -> ``moe_permute`` -> FC1 (gated SwiGLU, BF16) -> FC2
    (BF16) -> ``moe_unpermute``.

    ``token_selected_experts`` int32 ``[T, top_k]`` holds global expert ids;
    ``-1`` and ids outside ``[local_expert_offset, local_expert_offset +
    E_local)`` are masked routes (their ``token_final_scales`` entries are never
    read).  ``token_final_scales`` float32 ``[T, top_k]``.  ``output`` BF16 ``[T,
    H]`` is written in place when given, else allocated.  ``workspace_buffer`` is
    caller-owned scratch (:func:`cake_w4a16_workspace_size`).  ``fc1_alpha`` /
    ``fc2_alpha`` are FP32 ``[E_local]`` (default ones).  ``swiglu_limit`` is the
    optional model-wide clamp (float or one-element CPU tensor).  ``enable_pdl``
    defaults to the device's support (the Cake programs always carry the launch
    attribute; it selects it on the FlashInfer launches).  ``cell`` overrides
    :func:`select_cell`.  A call with ``T = 0`` returns the empty output without
    launching.
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.bfloat16:
        raise TypeError("x must be a 2-D BF16 CUDA tensor")
    device = x.device
    if device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")
    num_tokens, hidden_size = int(x.shape[0]), int(x.shape[1])
    if not x.is_contiguous() or x.data_ptr() % 16:
        raise ValueError("x must be contiguous and 16-byte aligned")
    if (
        not isinstance(w13, torch.Tensor)
        or w13.dim() != 3
        or not isinstance(w2, torch.Tensor)
        or w2.dim() != 3
    ):
        raise TypeError("w13 and w2 must be 3-D uint8 tensors [E_local, rows, K/2]")
    if (
        not isinstance(token_selected_experts, torch.Tensor)
        or token_selected_experts.dim() != 2
    ):
        raise TypeError("token_selected_experts must be a 2-D int32 tensor [T, top_k]")
    num_local_experts = int(w13.shape[0])
    intermediate_size = int(w2.shape[2]) * 2
    top_k = int(token_selected_experts.shape[1])
    _check_geometry(hidden_size, intermediate_size, num_local_experts, top_k)
    if num_experts <= 0 or num_experts % num_local_experts:
        raise ValueError(
            "num_experts must be a positive multiple of the local expert count"
        )
    if not 0 <= local_expert_offset <= num_experts - num_local_experts:
        raise ValueError(
            "local_expert_offset must place the local experts inside [0, num_experts)"
        )
    _check_tensor(
        w13,
        "w13",
        shape=(num_local_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    _check_tensor(
        w13_scale,
        "w13_scale",
        shape=(num_local_experts, 2 * intermediate_size, hidden_size // 16),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _check_tensor(
        w2,
        "w2",
        shape=(num_local_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    _check_tensor(
        w2_scale,
        "w2_scale",
        shape=(num_local_experts, hidden_size, intermediate_size // 16),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _check_tensor(
        token_selected_experts,
        "token_selected_experts",
        shape=(num_tokens, top_k),
        dtype=torch.int32,
        device=device,
    )
    _check_tensor(
        token_final_scales,
        "token_final_scales",
        shape=(num_tokens, top_k),
        dtype=torch.float32,
        device=device,
    )
    if output is None:
        output = torch.empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
    else:
        _check_tensor(
            output,
            "output",
            shape=(num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
    limit = _resolve_limit(swiglu_limit)
    if enable_pdl is None:
        enable_pdl = device_support_pdl(device)
    if num_tokens == 0:
        return output
    if cell is None:
        cell = select_cell(num_tokens, top_k, num_experts)
    call = _call_views(
        workspace_buffer,
        num_tokens,
        cell,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        top_k=top_k,
    )
    views, route_slots = call["views"], call["route_slots"]
    alpha1 = _resolve_alpha(
        fc1_alpha,
        "fc1_alpha",
        default=views["fc1_alpha_ones"],
        num_local_experts=num_local_experts,
        device=device,
    )
    alpha2 = _resolve_alpha(
        fc2_alpha,
        "fc2_alpha",
        default=views["fc2_alpha_ones"],
        num_local_experts=num_local_experts,
        device=device,
    )

    fc1 = _programs.handle("FC1", cell, device)
    fc2 = _programs.handle("FC2", cell, device)
    grid = cake_w4a16_grid(cell, device)
    moe_utils = _fi_moe_utils()

    hidden_workspace = views["hidden_workspace"]
    intermediate = views["intermediate"]
    # 1. Routing metadata (FlashInfer): expert-major tiles of n rows, cumulative
    #    row limits, both index maps and the device-side active tile count.
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = moe_utils.moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_tokens_dim=cell.n,
        enable_pdl=enable_pdl,
        out_tile_idx_to_expert_idx=views["tile_idx_to_expert_idx"],
        out_tile_idx_to_mn_limit=views["tile_idx_to_mn_limit"],
        out_expanded_idx_to_permuted_idx=views["expanded_idx_to_permuted_idx"],
        out_permuted_idx_to_expanded_idx=views["permuted_idx_to_expanded_idx"],
        out_total_num_padded_tokens=views["total_num_padded_tokens"],
        out_num_non_exiting_tiles=views["num_non_exiting_tiles"],
    )
    # 2. Gather the routed BF16 rows (FlashInfer); padding rows keep stale bytes
    #    and are never read into a valid output. An FC1 program that gathers the
    #    rows itself (it takes the route map) reads x directly: no permute launch.
    fc1_gathers = "permuted_idx_to_expanded_idx" in fc1.args
    if not fc1_gathers:
        moe_utils.moe_permute(
            input=x,
            permuted_output=hidden_workspace,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            max_num_permuted_tokens=route_slots,
            top_k=top_k,
            tile_size=cell.n,
            enable_pdl=enable_pdl,
        )
    common = {
        "tile_idx_to_expert_idx": tile_idx_to_expert_idx,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "num_non_exiting_tiles": num_non_exiting_tiles,
        "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
        "x": x,
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
    }
    # The scale descriptors take their bytes: the kernels declare ``S`` as uint8.
    fc1_values = {
        **common,
        "W": w13,
        "S": w13_scale.view(torch.uint8),
        "X": x if fc1_gathers else hidden_workspace,
        "C": intermediate,
        "top_k_div": top_k,
        "alpha": alpha1,
        "swiglu_limit": limit,
        "K": hidden_size,
        "num_m_tiles": intermediate_size // _FC1_M_TILE,
        "m_tile_div": intermediate_size // _FC1_M_TILE,
    }
    fc2_values = {
        **common,
        "W": w2,
        "S": w2_scale.view(torch.uint8),
        "X": intermediate,
        "C": hidden_workspace,
        "alpha": alpha2,
        "K": intermediate_size,
        "num_m_tiles": hidden_size // _FC2_M_TILE,
        "m_tile_div": hidden_size // _FC2_M_TILE,
    }
    # 3 + 4. The Cake kernels, on torch's current stream.
    with tvm_ffi.use_torch_stream():
        fc1.launch(*_ordered(fc1, fc1_values, "FC1"))
        fc2.launch(*_ordered(fc2, fc2_values, "FC2"))
    # 5. Deterministic combine (FlashInfer): FP32 FMA in ascending slot order over
    #    the BF16 expert rows FC2 wrote into hidden_workspace.
    _fi_moe_finalize().moe_unpermute(
        permuted_input=hidden_workspace,
        output=output,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        topk_scales=token_final_scales,
        num_tokens=num_tokens,
        top_k=top_k,
        enable_pdl=enable_pdl,
    )
    return output


def _ordered(
    handle: _ProgramHandle, values: Mapping[str, Any], kernel: str
) -> List[Any]:
    """The binding's positional arguments from the values by name."""
    missing = [name for name in handle.args if name not in values]
    if missing:
        raise RuntimeError(
            f"the Cake W4A16 {kernel} program takes {missing}, which this host does not supply; "
            f"its arguments are {list(handle.args)}"
        )
    return [values[name] for name in handle.args]
