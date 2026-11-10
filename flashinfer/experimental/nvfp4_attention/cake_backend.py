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

Host of the Cake experimental NVFP4 attention route: validation of the
caller's tensors, the workspace layout, the logical argument values of every
record of the chain and the runner that launches the chain.

The route is ``out = softmax(q k^T * softmax_scale) v`` over logical
``[B, H, S, 128]`` BF16 tensors with any strides: an ``[B, S, H, 128]``
buffer is passed as its ``[B, H, S, 128]`` view (``x.transpose(1, 2)``) and
read or written in place; strides alone cannot tell the two storage orders
apart when ``S == H``, so the view is the contract.  The kernels read ``q``,
``k`` and ``v`` through explicit element strides from the view's first
element, so the host hands their bindings the contiguous span of storage the
view covers; ``out`` is bound as the view itself.  Every workspace region is
carved in the rank the architecture's attention kernel maps or reads it with
and viewed, per record, as the element type the record's ``dtypes`` state
(the sm120 ``int8_blk`` quantizer writes 32-bit words of INT8 codes that its
attention maps as bytes; the SM100 / SM103 quantizer writes the FP32 rows
``qm32`` / ``kc32`` through byte pointers that the table program ``delta`` maps
as FP32).  Every launch quantizes the inputs
again (``stats``, ``quantize``, on SM100 / SM103 ``delta``, then
``attention``), so tensor values are free between launches; identities, shapes
and strides are bound at :func:`prepare_nvfp4_attention`.  Nothing is allocated or synchronised at
launch; the runner is CUDA-graph capturable and its replays are bitwise
repeatable (fixed reduction orders, self-resetting counters).  On SM100 / SM103
the attention launch splits the tiles of its partial last wave into key chunks
whose partial outputs live in the workspace (``part_o`` / ``part_ml``) until the
last-arriving chunk combines them; that split plan (``cake_jit.split_plan``),
and so the workspace size, depends on the SM count of the device the workspace
is carved for.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch
import tvm_ffi

from .cake_jit import (
    ARCHES,
    HEAD_DIM,
    LOGICAL_SCALARS,
    LOGICAL_TENSORS,
    MODULES,
    arch_for_capability,
    launch_args,
    launch_grid,
    load,
    records_for,
    split_plan,
)
from .cake_jit import sm_count as _sm_count

__all__ = [
    "ALIGN",
    "QK_FORMAT",
    "RECIPE_ID",
    "STRIDE_ALIGN",
    "NVFP4AttentionRunner",
    "WorkspaceLayout",
    "nvfp4_attention_workspace_bytes",
    "prepare_nvfp4_attention",
    "workspace_layout",
]

#: The recipe every architecture implements (the README states its numerics).  The QK operand format of a chain is
#: its attention record's ``build["qk_format"]`` (SM100 / SM103: ``nvfp4_res2d``, the two-level residual form with
#: the exact FP32 compensation table; SM120: ``int8_blk``, INT8 Q / K codes with per-block amax scales and the Q
#: blocks' INT8 offset rows); ``QK_FORMAT`` is the fallback of a record whose build spells none.
RECIPE_ID = "nvfp4-sage3-v1"
QK_FORMAT = "nvfp4"
#: Workspace regions start on 256-byte boundaries (every TMA-mapped region needs 16; 256 keeps L2 sectors whole).
ALIGN = 256
#: Element strides of the caller's tensors (batch, head, token) are multiples of 16 elements (32 bytes): the
#: kernels' 16-byte row loads and the TMA map of ``out`` stay aligned.  The inner (channel) stride is 1.
STRIDE_ALIGN = 16
_LOG2E = 1.0 / math.log(2.0)
_SM120_MAX_BH = (
    32768  # the sm_120a kernels pack (batch, head) into one i16 lane of the unit table
)
_TILE_ORDER = {"sm_100a": "desc", "sm_103a": "desc", "sm_120a": "asc"}
_KEY_GROUPS = {
    "sm_100a": "consecutive",
    "sm_103a": "consecutive",
    "sm_120a": "interleaved32",
}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _pad(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _device_index(device: Any) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"NVFP4 attention runs on CUDA devices, got {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _arch_of(device: Any) -> str:
    index = _device_index(device)
    capability = tuple(torch.cuda.get_device_capability(index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise ValueError(
            f"NVFP4 attention requires a device of compute capability {sorted(a[3:-1] for a in ARCHES)}, "
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
    """The carved workspace of one ``(arch, batch, heads, seqlen, sm_count)``: regions by logical name and the total
    bytes.  Only the SM100 / SM103 split regions (``part_o``, ``part_ml``, ``split_counters``) read ``sm_count``."""

    arch: str
    batch: int
    heads: int
    seqlen: int
    sm_count: int
    regions: dict[str, Region]
    total_bytes: int

    def view(self, workspace: torch.Tensor, name: str) -> torch.Tensor:
        """Region ``name`` as a tensor over the caller's flat ``uint8`` workspace."""
        region = self.regions[name]
        flat = workspace[region.offset : region.offset + region.nbytes]
        return flat.view(region.dtype).view(region.shape)


def _regions_sm100(
    arch: str, batch: int, heads: int, seqlen: int, sm_count: int
) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    """The sm_100a / sm_103a workspace: Q-side rows padded to 512 (the attention kernel's work tile), K/V-side rows
    padded to 128 (a key block); the two E2M1 levels of the centred Q and K (``q4`` / ``k4`` the first, ``q4l`` /
    ``k4l`` the residual second) with their UE4M3 scales as 1024-byte UTCCP tiles per 128 rows (``q_sf`` / ``k_sf``,
    ``q_sf_l`` / ``k_sf_l``); the transposed V codes and their scales as two 512-byte K-set tiles per 128 keys; the
    table program's operands -- the TF32-rounded FP32 block means ``qm32`` (one row per 128-row Q block of the padded
    Q side) and centred K rows ``kc32`` (one per padded key) -- and its output ``tab``, the attention's (hi, lo) BF16
    seed-operand image: 16 BF16 per row, 64 rows per (block group of 8 Q blocks, 64-key tile), ``G = ceil(Sq / 128
    / 8)`` groups and ``T = Skv / 64`` tiles per (batch, head); the statistics as f32 rows plus one u32 counter per
    (batch, head); the attention's split-tile partials -- one 512-row f32 tile of unnormalised output (``part_o``)
    and of (row maximum, row sum) (``part_ml``) per key-chunk item of the last wave, two self-resetting u32 counters
    per split tile -- sized by the arch's attention record's :func:`split_plan` for ``sm_count`` SMs (at least one
    row and one counter)."""
    bh = batch * heads
    q_rows, kv_rows = _pad(seqlen, 512), _pad(seqlen, 128)
    q_blocks = q_rows // 128
    groups, key_tiles = _ceil_div(q_blocks, 8), kv_rows // 64
    attention = MODULES[records_for(arch)["attention"]]
    plan = split_plan(
        attention, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
    )
    tile_rows = int(attention["grid_rule"]["tile_rows"])
    rows = max(1, plan["n_split"] * plan["n_chunks"])
    u8, f32, bf16 = torch.uint8, torch.float32, torch.bfloat16
    return [
        ("q4", u8, (bh * q_rows, HEAD_DIM // 2)),
        ("q_sf", u8, (bh * q_rows // 4, 32)),
        ("q4l", u8, (bh * q_rows, HEAD_DIM // 2)),
        ("q_sf_l", u8, (bh * q_rows // 4, 32)),
        ("k4", u8, (bh * kv_rows, HEAD_DIM // 2)),
        ("k_sf", u8, (bh * kv_rows // 4, 32)),
        ("k4l", u8, (bh * kv_rows, HEAD_DIM // 2)),
        ("k_sf_l", u8, (bh * kv_rows // 4, 32)),
        ("vt4", u8, (bh * HEAD_DIM, kv_rows // 2)),
        ("v_sf_lo", u8, (bh * kv_rows // 8, 32)),
        ("v_sf_hi", u8, (bh * kv_rows // 8, 32)),
        ("qm32", f32, (bh * q_blocks, HEAD_DIM)),
        ("kc32", f32, (bh * kv_rows, HEAD_DIM)),
        ("tab", bf16, (bh * groups * key_tiles * 64, 16)),
        ("mean_k", f32, (bh, HEAD_DIM)),
        ("partials", f32, (bh * _ceil_div(seqlen, 512), HEAD_DIM)),
        ("counters", torch.uint32, (bh,)),
        ("part_o", f32, (rows, tile_rows, HEAD_DIM)),
        ("part_ml", f32, (rows, tile_rows, 2)),
        ("split_counters", torch.uint32, (max(1, 2 * plan["n_split"]),)),
    ]


def _regions_sm120(
    arch: str, batch: int, heads: int, seqlen: int, sm_count: int
) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    """The sm_120a workspace of the shipped ``int8_blk`` form (``arch`` and ``sm_count`` are not read: it holds no
    split partials), each region in the rank and element type the attention kernel takes it: the statistics per
    (batch, Q block, head) -- ``partials`` and ``q_mean`` (FP32 rows), ``k_minmax`` (the per-tile K channel maxima
    and minima), the INT8 amax statistics ``q_amax`` / ``qm_amax`` and, per (head, 128-key block of the packed token
    axis), ``k_amax`` -- plus ``mean_k`` per (batch, head) and one u32 counter per (batch, head); the INT8 codes as
    token-major ``[B * S_pad, H, 128]`` byte rows of Q and K (the attention's tensor maps; the quantizer writes them
    as 32-bit words), the Q blocks' INT8 offset rows ``qm8`` per (batch, Q block, head), the transposed E2M1 V as
    ``[H, 128, B * S_pad / 2]`` bytes with its UE4M3 scales as ``[rows, 16]`` byte tiles (1 KiB per (head, 128-key
    block); the ``v_sf`` tensor map), and the unit table of the persistent attention grid.  Total bytes
    ``328 * T * H + 2204 * B * nqb * H + 516 * B * H`` with ``T = B * S_pad``."""
    bh = batch * heads
    s_pad = _pad(seqlen, 128)
    blocks, tokens = s_pad // 128, batch * s_pad
    u8, f32, u32 = torch.uint8, torch.float32, torch.uint32
    return [
        ("partials", f32, (batch * blocks, heads, HEAD_DIM)),
        ("counters", u32, (bh,)),
        ("mean_k", f32, (bh, HEAD_DIM)),
        ("q_mean", f32, (batch * blocks, heads, HEAD_DIM)),
        ("k_minmax", f32, (batch * blocks, heads, 2, HEAD_DIM)),
        ("q_amax", f32, (batch * blocks, heads)),
        ("qm_amax", f32, (batch * blocks, heads)),
        ("k_amax", f32, (heads, batch * blocks)),
        ("q8", u8, (tokens, heads, HEAD_DIM)),
        ("k8", u8, (tokens, heads, HEAD_DIM)),
        ("qm8", u8, (batch * blocks, heads, HEAD_DIM)),
        ("vt4", u8, (heads, HEAD_DIM, tokens // 2)),
        ("v_sf", u8, (heads * batch * blocks * 64, 16)),
        ("unit_table", torch.int32, (bh * blocks, 4)),
    ]


_REGIONS = {
    "sm_100a": _regions_sm100,
    "sm_103a": _regions_sm100,
    "sm_120a": _regions_sm120,
}


@functools.cache
def workspace_layout(
    arch: str, *, batch: int, heads: int, seqlen: int, sm_count: int
) -> WorkspaceLayout:
    """The workspace of one problem on ``arch`` for a device of ``sm_count`` SMs: every region the chain writes,
    carved on :data:`ALIGN`-byte boundaries in a fixed order, and the total byte count
    :func:`nvfp4_attention_workspace_bytes` reports.  Only the SM100 / SM103 split regions depend on ``sm_count``
    (:func:`split_plan`): the workspace is sized for the device it is carved for."""
    if arch not in _REGIONS:
        raise ValueError(
            f"{arch!r} is not an exported NVFP4 attention architecture ({ARCHES})"
        )
    batch, heads, seqlen, sm_count = int(batch), int(heads), int(seqlen), int(sm_count)
    if min(batch, heads, seqlen, sm_count) <= 0:
        raise ValueError(
            f"the workspace needs positive extents, got batch={batch}, heads={heads}, "
            f"seqlen={seqlen}, sm_count={sm_count}"
        )
    regions: dict[str, Region] = {}
    offset = 0
    for name, dtype, shape in _REGIONS[arch](arch, batch, heads, seqlen, sm_count):
        if name not in LOGICAL_TENSORS:
            raise RuntimeError(
                f"workspace region {name!r} is not in the package's tensor vocabulary"
            )
        region = Region(name, offset, dtype, tuple(int(x) for x in shape))
        regions[name] = region
        offset = _pad(offset + region.nbytes, ALIGN)
    return WorkspaceLayout(arch, batch, heads, seqlen, sm_count, regions, offset)


def nvfp4_attention_workspace_bytes(
    batch: int, heads: int, seqlen: int, device: Any
) -> int:
    """Bytes of the ``uint8`` workspace :func:`prepare_nvfp4_attention` needs for ``[batch, heads, seqlen, 128]``
    tensors on ``device``, sized for that device's SM count (a caller pooling workspaces sizes the pool with the
    largest problem's figure on the device the pool serves)."""
    return workspace_layout(
        _arch_of(device),
        batch=int(batch),
        heads=int(heads),
        seqlen=int(seqlen),
        sm_count=_sm_count(_device_index(device)),
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


def _stride(x: torch.Tensor, axis: int) -> int:
    # A unit extent's stride is immaterial; spell it as the next stride so the kernels see a consistent layout.
    return (
        int(x.stride(axis))
        if x.shape[axis] > 1
        else int(max(x.stride(), default=HEAD_DIM))
    )


def _common_scalars(
    arch: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> dict[str, Any]:
    batch, heads, seqlen, _dim = (int(x) for x in q.shape)
    blocks = _ceil_div(seqlen, 128)
    values: dict[str, Any] = {
        "batch": batch,
        "heads": heads,
        "seqlen": seqlen,
        "total_bh": batch * heads,
        "num_qblocks": blocks,
        "num_qblocks_pad": _pad(seqlen, 512) // 128 if arch != "sm_120a" else blocks,
        "num_kblocks": batch
        * blocks,  # 128-row blocks of the packed (batch x token) axis
        "num_units": batch * heads * blocks,
        "q_rows_pad": _pad(seqlen, 512) if arch != "sm_120a" else _pad(seqlen, 128),
        "kv_rows_pad": _pad(seqlen, 128),
        "softmax_scale_log2": float(softmax_scale) * _LOG2E,
    }
    for prefix, x in (("q", q), ("k", k), ("v", v), ("o", out)):
        values[f"{prefix}_stride_b"] = _stride(x, 0)
        values[f"{prefix}_stride_h"] = _stride(x, 1)
        values[f"{prefix}_stride_s"] = _stride(x, 2)
    return values


def _record_scalars(
    record: dict[str, Any], common: dict[str, Any], *, sm_count: int
) -> dict[str, Any]:
    """The scalars of one record: the common ones plus the record's own tile counts under its grid rule (a
    split-tile record's ``split_full_tiles`` / ``split_chunks`` from :func:`split_plan` for ``sm_count`` SMs)."""
    rule = record["grid_rule"]
    seqlen, bh = common["seqlen"], common["total_bh"]
    values = dict(common)
    if "rows" in rule:
        rows, pad = int(rule["rows"]), int(rule.get("pad", rule["rows"]))
        values["num_stats_tiles"] = _ceil_div(seqlen, rows)
        values["num_items"] = bh * _ceil_div(seqlen, pad) * (pad // rows)
    else:
        values["num_stats_tiles"] = _ceil_div(seqlen, 128)
        values["num_items"] = common["num_units"]
    if rule["kind"] == "split_persistent_tiles":
        plan = split_plan(
            record,
            batch=common["batch"],
            heads=common["heads"],
            seqlen=seqlen,
            sm_count=sm_count,
        )
        values["split_full_tiles"] = plan["n_full_tiles"]
        values["split_chunks"] = plan["n_chunks"]
    return values


def _unit_table(layout: WorkspaceLayout, workspace: torch.Tensor) -> None:
    """The sm_120a attention's static unit table ``[units, 4]``: one row ``(batch, head << 16 | q_tile, segment
    begin, segment end)`` per (batch, head, 128-row Q tile) in that enumeration order, the segment in the packed
    token axis (``begin = batch * S_pad``, ``end = begin + S``); filled once at prepare."""
    region = layout.regions.get("unit_table")
    if region is None:
        return
    batch, heads, seqlen = layout.batch, layout.heads, layout.seqlen
    s_pad, blocks = _pad(seqlen, 128), _ceil_div(seqlen, 128)
    b = torch.arange(batch).repeat_interleave(heads * blocks)
    h = torch.arange(heads).repeat_interleave(blocks).repeat(batch)
    qt = torch.arange(blocks).repeat(batch * heads)
    table = torch.stack([b, (h << 16) | qt, b * s_pad, b * s_pad + seqlen], dim=1).to(
        torch.int32
    )
    layout.view(workspace, "unit_table").copy_(
        table.to(workspace.device, non_blocking=True)
    )


_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "u8": torch.uint8,
    "u32": torch.uint32,
    "i32": torch.int32,
}
_CALLER_TENSORS = ("q", "k", "v", "out")


def _flat(x: torch.Tensor) -> torch.Tensor:
    """The contiguous span of storage a logical view covers, from its first element: what a kernel reading the
    view through its explicit element strides receives (the binding's pointer check wants a contiguous tensor)."""
    span = 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(x.shape, x.stride(), strict=True)
        if int(size) > 1
    )
    return x.as_strided((span,), (1,), x.storage_offset())


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
    sm_count: int,
) -> dict[str, Any]:
    """Logical name -> value for every kernel parameter of ``record``'s argument plan: the caller's ``q`` / ``k`` /
    ``v`` as the span of storage their view covers, ``out`` as the view, every region viewed as the element type the
    record's ``dtypes`` state, the scalars of :func:`_record_scalars` for ``sm_count`` SMs.  A logical name or dtype
    the host does not know is a registry/host mismatch and is refused here, before anything is loaded."""
    scalars = _record_scalars(record, common, sm_count=sm_count)
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
                values[logical] = tensor if logical == "out" else _flat(tensor)
            else:
                values[logical] = _as_dtype(
                    tensor, token, record=record["library"], key=key
                )
        else:
            if logical not in scalars:
                raise RuntimeError(
                    f"record {record['library']} binds the scalar {logical!r} for {key!r}, which the host does not compute"
                )
            values[logical] = scalars[logical]
    unknown = sorted(set(values) - set(LOGICAL_TENSORS) - set(LOGICAL_SCALARS))
    if unknown:
        raise RuntimeError(
            f"record {record['library']} binds names outside the package vocabulary: {unknown}"
        )
    return values


# ----------------------------------------------------------------------------------------------------------------
# The runner


@dataclass(frozen=True)
class NVFP4AttentionRunner:
    """The prepared chain of one problem: ``launch()`` (or a call) runs the architecture's chain (``stats``,
    ``quantize``, on SM100 / SM103 ``delta``, then ``attention``) in order on the current torch stream of the device
    -- no allocation, no synchronisation -- and returns the caller-owned ``out``.  The runner is bound to the tensors' identities, shapes and strides; their values are
    re-read at every launch (quantization is part of the launch).  Capture it in a CUDA graph if you wish; replays
    are bitwise repeatable.  ``launch_stage(name)`` launches one stage alone, for diagnostics."""

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
        """One stage (``"stats"``, ``"quantize"``, ``"delta"`` on SM100 / SM103, or ``"attention"``) alone; the
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
            f"workspace holds {workspace.numel()} bytes; this problem needs {needed} "
            f"on {device} (the layout is sized for the device's SM count)"
        )
    if workspace.data_ptr() % ALIGN:
        raise ValueError(f"workspace must start on a {ALIGN}-byte boundary")
    return workspace


def prepare_nvfp4_attention(
    q, k, v, out, *, backend="cake", softmax_scale=None, workspace=None
):
    """Prepare the NVFP4 attention chain for BF16 logical ``[B, H, S, 128]`` ``q``, ``k``, ``v`` and ``out`` (any
    strides, inner stride 1, the others multiples of 16 elements; an ``[B, S, H, 128]`` buffer is passed as its
    ``[B, H, S, 128]`` view).  ``softmax_scale`` defaults to ``128 ** -0.5`` and must be positive.  ``workspace``
    is a caller-owned ``uint8`` tensor of at least :func:`nvfp4_attention_workspace_bytes` bytes for the inputs'
    device (its figure: on SM100 / SM103 the layout follows the device's SM count), else one is allocated here
    (never at launch).  Returns an :class:`NVFP4AttentionRunner`."""
    if backend != "cake":
        raise ValueError("NVFP4 attention supports backend='cake'")
    _check_tensor(q, "q", None)
    for name, x in (("k", k), ("v", v), ("out", out)):
        _check_tensor(x, name, q)
    batch, heads, seqlen, _dim = (int(x) for x in q.shape)
    if min(batch, heads, seqlen) <= 0:
        raise ValueError(
            f"NVFP4 attention needs positive extents, got shape {tuple(q.shape)}"
        )
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(
            f"softmax_scale must be a positive finite number, got {softmax_scale!r}"
        )
    arch = _arch_of(q.device)
    if arch == "sm_120a" and batch * heads >= _SM120_MAX_BH:
        raise ValueError(
            f"the sm_120a route supports batch * heads < {_SM120_MAX_BH}, got {batch * heads}"
        )
    if batch * heads >= 2**31 or seqlen >= 2**31:
        raise ValueError("batch * heads and S must fit a 32-bit integer")
    device_index = _device_index(q.device)
    sm_count = _sm_count(device_index)
    chain = records_for(arch)
    # This device's layout: the SM100 / SM103 split regions follow its SM count, so a caller's workspace is checked
    # against the figure of the device the inputs are on.
    layout = workspace_layout(
        arch, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
    )
    if workspace is None:
        workspace = torch.empty(layout.total_bytes, dtype=torch.uint8, device=q.device)
    else:
        workspace = _check_workspace(workspace, q.device, layout.total_bytes)
    tensors = {name: layout.view(workspace, name) for name in layout.regions}
    tensors.update(q=q, k=k, v=v, out=out)
    for name in chain.values():
        rule = MODULES[name]["workspace_rule"]
        if rule is not None and rule["kind"] == "prepare_carved":
            # The kernels self-reset these counters; the first call starts from zero.  Nothing re-zeroes them at
            # launch, so a captured graph replays the self-reset alone.
            tensors[rule["counters"]].zero_()
    _unit_table(layout, workspace)
    common = _common_scalars(arch, q, k, v, out, scale)
    stages: list[tuple[str, Any, tuple[Any, ...]]] = []
    grids: dict[str, tuple[int, int, int]] = {}
    # The architecture's stages in launch order (the SM120 chain has no delta stage).
    for stage, name in chain.items():
        record = MODULES[name]
        grid = launch_grid(
            record, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
        )
        values = _values_for(record, tensors=tensors, common=common, sm_count=sm_count)
        arguments = launch_args(record, values, grid)
        entry = getattr(load(name, arch), record["ffi_entry"])
        stages.append((stage, entry, arguments))
        grids[stage] = grid
    attention = MODULES[chain["attention"]]
    build = attention["build"]
    metadata = {
        "arch": arch,
        "recipe": RECIPE_ID,
        "qk_format": str(build.get("qk_format", QK_FORMAT)),
        "tile_order": _TILE_ORDER[arch],
        "key_groups": _KEY_GROUPS[arch],
        "records": dict(chain),
        "grids": grids,
        "softmax_scale": scale,
        "workspace_bytes": layout.total_bytes,
        "shape": (batch, heads, seqlen, HEAD_DIM),
    }
    if attention["grid_rule"]["kind"] == "split_persistent_tiles":
        plan = split_plan(
            attention, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
        )
        metadata["split_plan"] = {
            "n_full_tiles": plan["n_full_tiles"],
            "n_chunks": plan["n_chunks"],
            "n_items": plan["n_items"],
            "grid": plan["grid"],
            "sm_count": sm_count,
        }
    return NVFP4AttentionRunner(
        out, workspace, layout.total_bytes, metadata, tuple(stages)
    )
