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

Host of the Cake experimental FP8 attention route (SM90): validation of the
caller's tensors, the workspace layout, the logical argument values of every
record of the chain and the runner that launches the chain.

The route is ``out = softmax(q k^T * softmax_scale) v`` over logical
``[B, H, S, 128]`` BF16 tensors with any strides: an ``[B, S, H, 128]``
buffer is passed as its ``[B, H, S, 128]`` view (``x.transpose(1, 2)``) and
read or written in place; strides alone cannot tell the two storage orders
apart when ``S == H``, so the view is the contract.  The kernels read ``q``,
``k`` and ``v`` and write ``out`` through explicit element strides from the
view's first element, so the host hands all four bindings the contiguous span
of storage the view covers.  Every workspace region is carved in the rank the
attention kernel maps or reads it with and viewed, per record, as the element
type the record's ``dtypes`` state (the quantizer writes 32-bit words of codes
that the attention maps as bytes).  Every launch quantizes the inputs again:
``stats_partials`` (per (batch, head, 128-row tile): the channel sums of K and
the channel maxima of |V|), ``stats_finalize`` (per (batch, head): the K
channel means and the V channel scales), ``quantize`` (the INT8 codes of Q with
one scale per 128-row block, the INT8 codes of the mean-centred K with one
scale per 64-key half, the transposed, per-channel scaled E4M3 V), then
``attention``; tensor values are therefore free between launches while
identities, shapes and strides are bound at :func:`prepare_fp8_attention`.
Nothing is allocated or synchronised at launch; the runner is CUDA-graph
capturable and its replays are bitwise repeatable (fixed reduction orders, no
device-side state across launches).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch
import tvm_ffi

from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    HEAD_DIM,
    LOGICAL_SCALARS,
    LOGICAL_TENSORS,
    MODULES,
    STAGES,
    arch_for_capability,
    launch_args,
    launch_grid,
    load,
    records_for,
)
from .cake_jit import sm_count as _sm_count

__all__ = [
    "ALIGN",
    "QK_FORMAT",
    "RECIPE_ID",
    "STRIDE_ALIGN",
    "FP8AttentionRunner",
    "WorkspaceLayout",
    "fp8_attention_workspace_bytes",
    "prepare_fp8_attention",
    "workspace_layout",
]

#: The recipe the route implements (the README states its numerics).  ``RECIPE_ID`` is a development placeholder
#: until the validation run fixes the recipe.  ``QK_FORMAT`` is the QK operand format of the shipped attention
#: kernel -- ``int8_blk``: INT8 Q with one scale per 128-row block and INT8 centred K with one scale per 64-key half
#: (no offset row); the P.V operands are E4M3 with one scale per V channel.  A record whose ``build`` spells a
#: ``qk_format`` keyword overrides it (``prepare_fp8_attention`` reports ``build.get("qk_format", QK_FORMAT)``).
RECIPE_ID = "fp8-sage-v1"
QK_FORMAT = "int8_blk"
#: Workspace regions start on 256-byte boundaries (every TMA-mapped region needs 16; 256 keeps L2 sectors whole).
ALIGN = 256
#: Element strides of the caller's tensors (batch, head, token) are multiples of 16 elements (32 bytes): the
#: kernels' 16-byte row loads and stores stay aligned.  The inner (channel) stride is 1.
STRIDE_ALIGN = 16
#: The kernels index every operand with 32-bit integers.
_INT32_LIMIT = 2**31
_LOG2E = 1.0 / math.log(2.0)
#: ``partials`` holds two 128-channel rows per (batch, head, 128-row tile): slot 0 the channel sums of K, slot 1 the
#: channel maxima of |V|.
_PARTIAL_ROWS = 2
#: K scales per 128-key tile: one per 64-key half (``k_scale[2 * tile + half]``).
_K_HALVES = 2


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _pad(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _device_index(device: Any) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"FP8 attention runs on CUDA devices, got {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _arch_of(device: Any) -> str:
    index = _device_index(device)
    capability = tuple(torch.cuda.get_device_capability(index))
    arch = arch_for_capability(capability)
    if arch is None:
        supported = ", ".join(
            f"{major}.{minor}" for major, minor in sorted(ARCH_BY_CAPABILITY)
        )
        raise ValueError(
            f"FP8 attention requires a device of compute capability {supported}, "
            f"got {capability[0]}.{capability[1]}"
        )
    return arch


# ----------------------------------------------------------------------------------------------------------------
# Workspace


@dataclass(frozen=True)
class Region:
    """One workspace region: its byte offset, dtype and logical shape (the view the host binds)."""

    name: str
    offset: int
    dtype: torch.dtype
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class WorkspaceLayout:
    """The carved workspace of one ``(arch, batch, heads, seqlen)``: regions by logical name and the total bytes."""

    arch: str
    batch: int
    heads: int
    seqlen: int
    regions: dict[str, Region]
    total_bytes: int

    def view(self, workspace: torch.Tensor, name: str) -> torch.Tensor:
        """Region ``name`` as a tensor over the caller's flat ``uint8`` workspace."""
        region = self.regions[name]
        flat = workspace[region.offset : region.offset + region.nbytes]
        return flat.view(region.dtype).view(region.shape)


def _regions_sm90(
    batch: int, heads: int, seqlen: int
) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    """The sm_90a workspace in the registry's order (the ``stats_partials`` record's ``workspace_rule``), each
    region in the rank and element type the attention kernel takes it: the statistics partials (per 128-row tile:
    slot 0 the channel sums of K, slot 1 the channel maxima of |V|), the K channel means and the V channel scales
    per (batch, head), the INT8 codes of Q and of the centred K as ``[B * H, S_pad, 128]`` byte rows and the
    transposed E4M3 V codes as ``[B * H, 128, S_pad]`` (the attention's three tensor maps), one FP32 Q scale per
    128-row block and one FP32 K scale per 64-key half (index ``2 * tile + half``).  ``S_pad = ceil(S / 128) *
    128``; rows and keys past ``S`` are written as zero codes with finite scales by the quantizer."""
    bh = batch * heads
    nqb = _ceil_div(seqlen, 128)
    s_pad = nqb * 128
    u8, f32 = torch.uint8, torch.float32
    return [
        ("partials", f32, (bh, nqb, _PARTIAL_ROWS, HEAD_DIM)),
        ("mean_k", f32, (bh, HEAD_DIM)),
        ("v_scale", f32, (bh, HEAD_DIM)),
        ("q8", u8, (bh, s_pad, HEAD_DIM)),
        ("k8", u8, (bh, s_pad, HEAD_DIM)),
        ("vt8", u8, (bh, HEAD_DIM, s_pad)),
        ("q_scale", f32, (bh, nqb)),
        ("k_scale", f32, (bh, _K_HALVES * nqb)),
    ]


_REGIONS = {"sm_90a": _regions_sm90}


@functools.cache
def workspace_layout(
    arch: str, *, batch: int, heads: int, seqlen: int
) -> WorkspaceLayout:
    """The workspace of one problem on ``arch``: every region the chain writes, carved on :data:`ALIGN`-byte
    boundaries in a fixed order, and the total byte count :func:`fp8_attention_workspace_bytes` reports."""
    if arch not in _REGIONS:
        raise ValueError(
            f"{arch!r} is not an exported FP8 attention architecture ({ARCHES})"
        )
    batch, heads, seqlen = int(batch), int(heads), int(seqlen)
    if min(batch, heads, seqlen) <= 0:
        raise ValueError(
            f"the workspace needs positive extents, got batch={batch}, heads={heads}, seqlen={seqlen}"
        )
    regions: dict[str, Region] = {}
    offset = 0
    for name, dtype, shape in _REGIONS[arch](batch, heads, seqlen):
        if name not in LOGICAL_TENSORS:
            raise RuntimeError(
                f"workspace region {name!r} is not in the package's tensor vocabulary"
            )
        region = Region(name, offset, dtype, tuple(int(x) for x in shape))
        regions[name] = region
        offset = _pad(offset + region.nbytes, ALIGN)
    return WorkspaceLayout(arch, batch, heads, seqlen, regions, offset)


def fp8_attention_workspace_bytes(
    batch: int, heads: int, seqlen: int, device: Any
) -> int:
    """Bytes of the ``uint8`` workspace :func:`prepare_fp8_attention` needs for ``[batch, heads, seqlen, 128]``
    tensors on ``device`` (a caller pooling workspaces sizes the pool with the largest problem's figure)."""
    return workspace_layout(
        _arch_of(device), batch=int(batch), heads=int(heads), seqlen=int(seqlen)
    ).total_bytes


# ----------------------------------------------------------------------------------------------------------------
# Validation and the logical values


def _check_tensor(x: Any, name: str, like: torch.Tensor | None) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be BF16, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.ndim != 4 or x.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"{name} must be a logical [B, H, S, {HEAD_DIM}] tensor, got shape {tuple(x.shape)}"
        )
    if like is not None:
        if x.shape != like.shape:
            raise ValueError(
                f"{name} has shape {tuple(x.shape)}; q has {tuple(like.shape)}"
            )
        if x.device != like.device:
            raise ValueError(f"{name} is on {x.device}; q is on {like.device}")
    if x.stride(-1) != 1:
        raise ValueError(
            f"{name} needs a contiguous head dimension (stride 1), got stride {x.stride(-1)}"
        )
    for axis, label in ((0, "batch"), (1, "head"), (2, "token")):
        if x.shape[axis] > 1 and x.stride(axis) % STRIDE_ALIGN:
            raise ValueError(
                f"{name}'s {label} stride {x.stride(axis)} is not a multiple of {STRIDE_ALIGN} elements "
                f"({STRIDE_ALIGN * 2} bytes)"
            )
    if x.shape[0] > 1 or x.shape[1] > 1 or x.shape[2] > 1:
        # No two logical rows may alias: the smallest non-unit extent's stride covers the row.
        if any(x.stride(axis) < HEAD_DIM for axis in range(3) if x.shape[axis] > 1):
            raise ValueError(
                f"{name}'s strides {x.stride()} overlap rows of the logical [B, H, S, {HEAD_DIM}] view"
            )


def _span(x: torch.Tensor) -> int:
    """Elements of the contiguous span of storage a logical view covers, from its first element."""
    return 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(x.shape, x.stride(), strict=True)
        if int(size) > 1
    )


def _check_index_range(
    batch: int, heads: int, seqlen: int, tensors: dict[str, torch.Tensor]
) -> None:
    """The kernels index every operand with 32-bit integers: the dense element count ``B * H * S * 128`` and the
    strided span of every caller tensor stay below ``2**31``."""
    elements = batch * heads * seqlen * HEAD_DIM
    if elements >= _INT32_LIMIT:
        raise ValueError(
            f"the route indexes its operands with 32-bit integers: B * H * S * {HEAD_DIM} must be below 2**31, "
            f"got {elements}"
        )
    for name, x in tensors.items():
        span = _span(x)
        if span >= _INT32_LIMIT:
            raise ValueError(
                f"{name}'s strides span {span} elements; the kernels' 32-bit element indices need fewer than 2**31"
            )


def _stride(x: torch.Tensor, axis: int) -> int:
    # A unit extent's stride is immaterial; spell it as the next stride so the kernels see a consistent layout.
    return (
        int(x.stride(axis))
        if x.shape[axis] > 1
        else int(max(x.stride(), default=HEAD_DIM))
    )


def _common_scalars(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> dict[str, Any]:
    """The scalars every record binds from: the extents, the 128-row block count of one (batch, head) (the
    kernels' ``num_tiles`` / ``num_blocks``), the softmax scale in the log2 domain and the element strides of the
    four ``[B, H, S, 128]`` views."""
    batch, heads, seqlen, _dim = (int(x) for x in q.shape)
    values: dict[str, Any] = {
        "batch": batch,
        "heads": heads,
        "seqlen": seqlen,
        "num_qblocks": _ceil_div(seqlen, 128),
        "softmax_scale_log2": float(softmax_scale) * _LOG2E,
    }
    for prefix, x in (("q", q), ("k", k), ("v", v), ("o", out)):
        values[f"{prefix}_stride_b"] = _stride(x, 0)
        values[f"{prefix}_stride_h"] = _stride(x, 1)
        values[f"{prefix}_stride_s"] = _stride(x, 2)
    return values


_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "u8": torch.uint8,
    "u32": torch.uint32,
    "i32": torch.int32,
}
_CALLER_TENSORS = ("q", "k", "v", "out")


def _flat(x: torch.Tensor) -> torch.Tensor:
    """The contiguous span of storage a logical view covers, from its first element: what a kernel reading or
    writing the view through its explicit element strides receives (the binding's pointer check wants a
    contiguous tensor)."""
    return x.as_strided((_span(x),), (1,), x.storage_offset())


def _as_dtype(x: torch.Tensor, token: str, *, record: str, key: str) -> torch.Tensor:
    dtype = _TORCH_DTYPES.get(token)
    if dtype is None:
        raise RuntimeError(
            f"record {record} states dtype {token!r} for {key!r}; the host views regions as {sorted(_TORCH_DTYPES)}"
        )
    return x if x.dtype == dtype else x.view(dtype)


def _values_for(
    record: dict[str, Any],
    *,
    tensors: dict[str, torch.Tensor],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Logical name -> value for every kernel parameter of ``record``'s argument plan: the caller's ``q`` / ``k`` /
    ``v`` / ``out`` as the span of storage their view covers (the kernels take plain pointers with explicit element
    strides), every region viewed as the element type the record's ``dtypes`` state, the scalars from ``common``.
    A logical name or dtype the host does not know is a registry/host mismatch and is refused here, before
    anything is loaded."""
    values: dict[str, Any] = {}
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            continue
        logical = record["logical"][key]
        if kind == "buffer":
            if logical not in tensors:
                raise RuntimeError(
                    f"record {record['library']} binds the tensor {logical!r} for {key!r}, which the host does not carve"
                )
            token = record["dtypes"].get(key)
            if token is None:
                raise RuntimeError(
                    f"record {record['library']} states no dtype for its buffer parameter {key!r}"
                )
            tensor = tensors[logical]
            if logical in _CALLER_TENSORS:
                if tensor.dtype != _TORCH_DTYPES.get(token):
                    raise RuntimeError(
                        f"record {record['library']} binds {logical!r} as {token}; the caller's tensor is {tensor.dtype}"
                    )
                values[logical] = _flat(tensor)
            else:
                values[logical] = _as_dtype(
                    tensor, token, record=record["library"], key=key
                )
        else:
            if logical not in common:
                raise RuntimeError(
                    f"record {record['library']} binds the scalar {logical!r} for {key!r}, which the host does not compute"
                )
            values[logical] = common[logical]
    unknown = sorted(set(values) - set(LOGICAL_TENSORS) - set(LOGICAL_SCALARS))
    if unknown:
        raise RuntimeError(
            f"record {record['library']} binds names outside the package vocabulary: {unknown}"
        )
    return values


# ----------------------------------------------------------------------------------------------------------------
# The runner


@dataclass(frozen=True)
class FP8AttentionRunner:
    """The prepared chain of one problem: ``launch()`` (or a call) runs ``stats_partials``, ``stats_finalize``,
    ``quantize`` and ``attention`` in order on the current torch stream of the device -- no allocation, no
    synchronisation -- and returns the caller-owned ``out``.  The runner is bound to the tensors' identities,
    shapes and strides; their values are re-read at every launch (quantization is part of the launch).  Capture
    it in a CUDA graph if you wish; replays are bitwise repeatable.  ``launch_stage(name)`` launches one stage
    alone, for diagnostics."""

    out: torch.Tensor
    workspace: torch.Tensor
    workspace_bytes: int
    route_metadata: dict[str, Any]
    stages: tuple[tuple[str, Any, tuple[Any, ...]], ...]

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            for _stage, entry, arguments in self.stages:
                entry(*arguments)
        return self.out

    __call__ = launch

    def launch_stage(self, name: str) -> None:
        """One stage (``"stats_partials"``, ``"stats_finalize"``, ``"quantize"`` or ``"attention"``) alone; the
        chain's data dependencies are the caller's to respect."""
        for stage, entry, arguments in self.stages:
            if stage == name:
                with tvm_ffi.use_torch_stream():
                    entry(*arguments)
                return
        raise KeyError(
            f"{name!r} is not a stage of the chain {[stage for stage, _, _ in self.stages]}"
        )


def _check_workspace(workspace: Any, device: torch.device, needed: int) -> torch.Tensor:
    if not isinstance(workspace, torch.Tensor):
        raise TypeError(
            f"workspace must be a torch.Tensor, got {type(workspace).__name__}"
        )
    if (
        workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be a contiguous one-dimensional uint8 tensor")
    if workspace.device != device:
        raise ValueError(
            f"workspace is on {workspace.device}; the inputs are on {device}"
        )
    if workspace.numel() < needed:
        raise ValueError(
            f"workspace holds {workspace.numel()} bytes; this problem needs {needed}"
        )
    if workspace.data_ptr() % ALIGN:
        raise ValueError(f"workspace must start on a {ALIGN}-byte boundary")
    return workspace


def prepare_fp8_attention(
    q, k, v, out, *, backend="cake", softmax_scale=None, workspace=None
):
    """Prepare the FP8 attention chain for BF16 logical ``[B, H, S, 128]`` ``q``, ``k``, ``v`` and ``out`` (any
    strides, inner stride 1, the others multiples of 16 elements; an ``[B, S, H, 128]`` buffer is passed as its
    ``[B, H, S, 128]`` view).  ``softmax_scale`` defaults to ``128 ** -0.5`` and must be positive.  ``workspace``
    is a caller-owned ``uint8`` tensor of at least :func:`fp8_attention_workspace_bytes` bytes on the inputs'
    device, else one is allocated here (never at launch).  Returns an :class:`FP8AttentionRunner`."""
    if backend != "cake":
        raise ValueError("FP8 attention supports backend='cake'")
    _check_tensor(q, "q", None)
    for name, x in (("k", k), ("v", v), ("out", out)):
        _check_tensor(x, name, q)
    batch, heads, seqlen, _dim = (int(x) for x in q.shape)
    if min(batch, heads, seqlen) <= 0:
        raise ValueError(
            f"FP8 attention needs positive extents, got shape {tuple(q.shape)}"
        )
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(
            f"softmax_scale must be a positive finite number, got {softmax_scale!r}"
        )
    arch = _arch_of(q.device)
    _check_index_range(batch, heads, seqlen, {"q": q, "k": k, "v": v, "out": out})
    layout = workspace_layout(arch, batch=batch, heads=heads, seqlen=seqlen)
    if workspace is None:
        workspace = torch.empty(layout.total_bytes, dtype=torch.uint8, device=q.device)
    else:
        workspace = _check_workspace(workspace, q.device, layout.total_bytes)
    tensors = {name: layout.view(workspace, name) for name in layout.regions}
    tensors.update(q=q, k=k, v=v, out=out)
    # Nothing to initialise: the chain has no counters and no tables; every launch rewrites every region it reads.
    device_index = _device_index(q.device)
    sm_count = _sm_count(device_index)
    common = _common_scalars(q, k, v, out, scale)
    chain = records_for(arch)
    stages: list[tuple[str, Any, tuple[Any, ...]]] = []
    grids: dict[str, tuple[int, int, int]] = {}
    for stage in STAGES:
        name = chain[stage]
        record = MODULES[name]
        grid = launch_grid(
            record, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
        )
        values = _values_for(record, tensors=tensors, common=common)
        arguments = launch_args(record, values, grid)
        entry = getattr(load(name, arch), record["ffi_entry"])
        stages.append((stage, entry, arguments))
        grids[stage] = grid
    build = MODULES[chain["attention"]]["build"]
    metadata = {
        "arch": arch,
        "recipe": RECIPE_ID,
        "qk_format": str(build.get("qk_format", QK_FORMAT)),
        "records": dict(chain),
        "grids": grids,
        "softmax_scale": scale,
        "workspace_bytes": layout.total_bytes,
        "shape": (batch, heads, seqlen, HEAD_DIM),
    }
    return FP8AttentionRunner(
        out, workspace, layout.total_bytes, metadata, tuple(stages)
    )
