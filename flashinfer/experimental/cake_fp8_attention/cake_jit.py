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

Host plan and loader glue of the Cake experimental FP8 attention programs
(SM90), the ``backend="cake"`` of
:func:`flashinfer.prefill.prepare_fp8_attention`.

Contract of the route: ``out = bf16(softmax(q k^T * scale) v)`` over logical
``[B, H, S, 128]`` BF16 ``q`` / ``k`` / ``v`` / ``out`` with any strides (an
``[B, S, H, 128]`` buffer is passed as its ``[B, H, S, 128]`` view and read or
written in place), any ``S >= 1``, dense and noncausal, ``softmax_scale > 0``.
One call is a chain of four programs, launched in ``stage`` order under
programmatic dependent launch: ``stats_partials`` (per 128-row tile: channel
sums of K, channel maxima of |V|), ``stats_finalize`` (per-(batch, head) K
means and V channel scales), ``quantize`` (INT8 codes of Q with one scale per
128-row block, INT8 codes of the centred K with one scale per 64-key half, and
the transposed, per-channel scaled E4M3 V, written to the caller's workspace)
and ``attention`` (the quantized attention, BF16 output).  :mod:`.cake_backend` validates the tensors, carves the workspace,
binds the logical arguments and launches the chain; quantization happens
inside every launch, so tensor values are free between launches while
identities and strides are bound.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the source ``module`` and ``build`` keywords it was
rendered from, its ``role`` and ``stage`` in the chain, its launch record, the
argument plan of its ``run`` entry, the kernel-parameter -> logical-argument
map over the vocabulary ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``, the element
type (``dtypes``: ``bf16``, ``f32``, ``u8``, ``u32`` or ``i32``) the kernel
declares for every buffer parameter -- the host views the carved region as
that type before the launch -- the grid rule :func:`launch_grid` evaluates, the selection rule (one build per
architecture group) and the workspace rule (``prepare_carved``: the regions the
first program of the chain writes; ``counters`` is ``None``, the chain has no
self-resetting counters; ``None`` on every other record).  What to compile --
sources, digests, nvcc flags -- is recorded once in ``build.json`` next to
this file; :mod:`.cake_build` serves the library from a prebuilt ``aot/`` file
when a builder compiled it and otherwise through FlashInfer's JIT from the
same sources and flags.  Every ``build.json`` entry records the compile
options the program's kernel was validated with (none on any program); the loader passes
``use_fast_math=False`` so that the manifest, not the loader, states them.

Tensor maps travel by value in the kernel parameters: the binding encodes a
``CUtensorMap`` from each tensor view per call, so a launch is self-contained,
CUDA-graph capturable, and two layers never share descriptor storage.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_fp8_attention_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted,
every default spelled>, "module": <dotted source module>}`` of the kernel the
program was rendered from; a changed keyword is a new program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# The exported architectures and the device capabilities that select them.
ARCHES = ("sm_90a",)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
}

# The head dimension of every record.
HEAD_DIM = 128
# The chain, in launch order; every architecture delivers one record per stage,
# except the optional stages, which an architecture group may leave out.
STAGES = ("stats_partials", "stats_finalize", "quantize", "attention")
OPTIONAL_STAGES = ()

# The host vocabulary: the logical names ``cake_backend`` binds.  A record's
# ``logical`` map names, for every kernel parameter of its ``arg_plan``, the
# tensor (``buffer``) or scalar (``parameter``) it receives; ``num_qblocks``
# counts the 128-row blocks of one (batch, head) (the kernels' ``num_tiles`` /
# ``num_blocks``), the workspace rows of every operand are ``num_qblocks * 128``.
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "out",
    "q8",
    "k8",
    "vt8",
    "q_scale",
    "k_scale",
    "v_scale",
    "mean_k",
    "partials",
)
LOGICAL_SCALARS = (
    "batch",
    "heads",
    "seqlen",
    "num_qblocks",
    "softmax_scale_log2",
    "q_stride_b",
    "q_stride_h",
    "q_stride_s",
    "k_stride_b",
    "k_stride_h",
    "k_stride_s",
    "v_stride_b",
    "v_stride_h",
    "v_stride_s",
    "o_stride_b",
    "o_stride_h",
    "o_stride_s",
)

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "stats_partials_sm90": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "partials"],
            ["parameter", "seqlen"],
            ["parameter", "num_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "k_stride_b"],
            ["parameter", "k_stride_s"],
            ["parameter", "k_stride_h"],
            ["parameter", "v_stride_b"],
            ["parameter", "v_stride_s"],
            ["parameter", "v_stride_h"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "dtypes": {
            "K": "bf16",
            "V": "bf16",
            "partials": "f32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "batch_heads_tiles",
            "pad": 128,
            "rows": 128,
        },
        "kernel": "cake_fp8_attention_5402aaf711017814a0ce",
        "kernel_symbol": "kernel_cake_fp8_attention_5402aaf711017814a0ce",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_fp8_attention_5402aaf711017814a0ce",
        "logical": {
            "K": "k",
            "V": "v",
            "k_stride_b": "k_stride_b",
            "k_stride_h": "k_stride_h",
            "k_stride_s": "k_stride_s",
            "num_heads": "heads",
            "num_tiles": "num_qblocks",
            "partials": "partials",
            "seqlen": "seqlen",
            "v_stride_b": "v_stride_b",
            "v_stride_h": "v_stride_h",
            "v_stride_s": "v_stride_s",
        },
        "module": "tart.sage_fp8_attention_stats_partials.sm90.fp8_sage_attention_stats_partials",
        "role": "stats_partials",
        "select_rule": None,
        "stage": 0,
        "workspace_rule": {
            "counters": None,
            "kind": "prepare_carved",
            "regions": [
                "partials",
                "mean_k",
                "v_scale",
                "q8",
                "k8",
                "vt8",
                "q_scale",
                "k_scale",
            ],
        },
    },
    "stats_finalize_sm90": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "partials"],
            ["buffer", "mean_k"],
            ["buffer", "v_scale"],
            ["parameter", "seqlen"],
            ["parameter", "num_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"unroll": 8},
        "dtypes": {
            "mean_k": "f32",
            "partials": "f32",
            "v_scale": "f32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "batch_heads",
        },
        "kernel": "cake_fp8_attention_8f4172c0adf623a761c9",
        "kernel_symbol": "kernel_cake_fp8_attention_8f4172c0adf623a761c9",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fp8_attention_8f4172c0adf623a761c9",
        "logical": {
            "mean_k": "mean_k",
            "num_tiles": "num_qblocks",
            "partials": "partials",
            "seqlen": "seqlen",
            "v_scale": "v_scale",
        },
        "module": "tart.sage_fp8_attention_stats_finalize.sm90.fp8_sage_attention_stats_finalize",
        "role": "stats_finalize",
        "select_rule": None,
        "stage": 1,
        "workspace_rule": None,
    },
    "quantize_sm90": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "mean_k"],
            ["buffer", "v_scale"],
            ["buffer", "q8"],
            ["buffer", "k8"],
            ["buffer", "vt8"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["parameter", "seqlen"],
            ["parameter", "num_heads"],
            ["parameter", "num_blocks"],
            ["parameter", "q_stride_b"],
            ["parameter", "q_stride_s"],
            ["parameter", "q_stride_h"],
            ["parameter", "k_stride_b"],
            ["parameter", "k_stride_s"],
            ["parameter", "k_stride_h"],
            ["parameter", "v_stride_b"],
            ["parameter", "v_stride_s"],
            ["parameter", "v_stride_h"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "dtypes": {
            "K": "bf16",
            "Q": "bf16",
            "V": "bf16",
            "k8": "u32",
            "k_scale": "f32",
            "mean_k": "f32",
            "q8": "u32",
            "q_scale": "f32",
            "v_scale": "f32",
            "vt8": "u32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "batch_heads_tiles",
            "pad": 128,
            "rows": 128,
        },
        "kernel": "cake_fp8_attention_bf0833d889569b5406df",
        "kernel_symbol": "kernel_cake_fp8_attention_bf0833d889569b5406df",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
        "library": "cake_fp8_attention_bf0833d889569b5406df",
        "logical": {
            "K": "k",
            "Q": "q",
            "V": "v",
            "k8": "k8",
            "k_scale": "k_scale",
            "k_stride_b": "k_stride_b",
            "k_stride_h": "k_stride_h",
            "k_stride_s": "k_stride_s",
            "mean_k": "mean_k",
            "num_blocks": "num_qblocks",
            "num_heads": "heads",
            "q8": "q8",
            "q_scale": "q_scale",
            "q_stride_b": "q_stride_b",
            "q_stride_h": "q_stride_h",
            "q_stride_s": "q_stride_s",
            "seqlen": "seqlen",
            "v_scale": "v_scale",
            "v_stride_b": "v_stride_b",
            "v_stride_h": "v_stride_h",
            "v_stride_s": "v_stride_s",
            "vt8": "vt8",
        },
        "module": "tart.sage_fp8_attention_quantize.sm90.fp8_sage_attention_quantize",
        "role": "quantize",
        "select_rule": None,
        "stage": 2,
        "workspace_rule": None,
    },
    "attention_sm90": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q8"],
            ["buffer", "K8"],
            ["buffer", "VT8"],
            ["buffer", "O"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_scale"],
            ["parameter", "seqlen"],
            ["parameter", "num_heads"],
            ["parameter", "o_stride_b"],
            ["parameter", "o_stride_s"],
            ["parameter", "o_stride_h"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "num_stages": 2,
        },
        "dtypes": {
            "K8": "u8",
            "O": "bf16",
            "Q8": "u8",
            "VT8": "u8",
            "k_scale": "f32",
            "q_scale": "f32",
            "v_scale": "f32",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "batch_heads_tiles",
            "pad": 64,
            "rows": 64,
        },
        "kernel": "cake_fp8_attention_c0e2681a52a8b76b0256",
        "kernel_symbol": "kernel_cake_fp8_attention_c0e2681a52a8b76b0256",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 74752,
            "use_pdl": True,
        },
        "library": "cake_fp8_attention_c0e2681a52a8b76b0256",
        "logical": {
            "K8": "k8",
            "O": "out",
            "Q8": "q8",
            "VT8": "vt8",
            "k_scale": "k_scale",
            "num_heads": "heads",
            "o_stride_b": "o_stride_b",
            "o_stride_h": "o_stride_h",
            "o_stride_s": "o_stride_s",
            "q_scale": "q_scale",
            "seqlen": "seqlen",
            "softmax_scale_log2": "softmax_scale_log2",
            "v_scale": "v_scale",
        },
        "module": "tart.attention_fwd_sage_fp8_bf16o.sm90.fp8_sage_attention",
        "role": "attention",
        "select_rule": {
            "kind": "always",
        },
        "stage": 3,
        "workspace_rule": None,
    },
}


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target; the capability is immutable per process)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake FP8 attention target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def records_for(arch: str) -> dict[str, str]:
    """``stage -> record name`` of the chain delivered for ``arch``, in launch
    order (``RuntimeError`` unless exactly one record per stage ships for it; a
    stage of ``OPTIONAL_STAGES`` may have none and is then absent)."""
    found: dict[str, list[str]] = {stage: [] for stage in STAGES}
    for name, record in MODULES.items():
        if arch in record["arches"]:
            found.setdefault(record["role"], []).append(name)
    chain: dict[str, str] = {}
    for stage in STAGES:
        names = found.get(stage, [])
        if len(names) > 1 or (not names and stage not in OPTIONAL_STAGES):
            raise RuntimeError(
                f"Cake FP8 attention delivers {len(names)} {stage!r} programs for {arch}; "
                f"the exported architectures are {ARCHES}"
            )
        if names:
            chain[stage] = names[0]
    return chain


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake FP8 attention record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


def split_plan(
    record: dict[str, Any], *, batch: int, heads: int, seqlen: int, sm_count: int
) -> dict[str, int]:
    """The last-wave split plan of a ``split_persistent_tiles`` record for
    ``batch x heads`` heads of ``seqlen`` rows on ``sm_count`` SMs: ``n_clusters``
    persistent clusters of ``ctas_per_tile`` CTAs work through ``total_tiles``
    tiles of ``tile_rows`` rows; the ``n_split`` tiles of the partial last wave
    are split into ``n_chunks`` key chunks each (at most ``max_chunks``, at most
    ``key_blocks`` = ceil(S / key_block)); the work items are the
    ``n_full_tiles`` whole tiles, then ``n_split * n_chunks`` chunk items
    (``n_items``); ``grid`` is the launch grid.  The chunk items' partial
    outputs occupy ``n_split * n_chunks`` rows of the ``part_o`` / ``part_ml``
    workspace regions and ``2 * n_split`` self-resetting ``split_counters``
    (:func:`.cake_backend.workspace_layout` rounds both up to at least one), so
    the workspace is sized for the SM count of the device it was carved for."""
    rule = record["grid_rule"]
    if rule["kind"] != "split_persistent_tiles":
        raise ValueError(
            f"split_plan reads a 'split_persistent_tiles' rule, got {rule['kind']!r}"
        )
    batch, heads, seqlen, sm_count = int(batch), int(heads), int(seqlen), int(sm_count)
    if min(batch, heads, seqlen, sm_count) <= 0:
        raise ValueError(
            f"the grid rule needs positive extents, got batch={batch}, heads={heads}, "
            f"seqlen={seqlen}, sm_count={sm_count}"
        )
    ctas, tile_rows = int(rule["ctas_per_tile"]), int(rule["tile_rows"])
    key_block, max_chunks = int(rule["key_block"]), int(rule["max_chunks"])
    if sm_count < ctas:
        raise ValueError(
            f"the persistent cluster needs at least {ctas} SMs, got {sm_count}"
        )
    n_clusters = max(1, sm_count // ctas)
    total_tiles = batch * heads * -(-seqlen // tile_rows)
    key_blocks = -(-seqlen // key_block)
    n_split = total_tiles % n_clusters
    n_chunks = min(max_chunks, key_blocks, n_clusters // n_split) if n_split else 1
    if n_chunks < 2:
        n_split, n_chunks = 0, 1
    n_full_tiles = total_tiles - n_split
    n_items = n_full_tiles + n_split * n_chunks
    return {
        "n_clusters": n_clusters,
        "total_tiles": total_tiles,
        "key_blocks": key_blocks,
        "n_full_tiles": n_full_tiles,
        "n_split": n_split,
        "n_chunks": n_chunks,
        "n_items": n_items,
        "grid": ctas * min(n_clusters, n_items),
    }


def launch_grid(
    record: dict[str, Any], *, batch: int, heads: int, seqlen: int, sm_count: int
) -> tuple[int, int, int]:
    """Launch grid of one record for ``batch x heads`` heads of ``seqlen`` rows
    on a device of ``sm_count`` SMs under its grid rule (one-dimensional grids):
    ``persistent_tiles`` -- a persistent cluster of ``ctas_per_tile`` CTAs per
    ``tile_rows``-row tile, at most one cluster per ``ctas_per_tile`` SMs;
    ``split_persistent_tiles`` -- the same over the work items of
    :func:`split_plan` (whole tiles, then the key chunks of the last wave);
    ``capped_persistent`` -- one CTA per ``unit_rows``-row unit, at most one per
    SM; ``batch_heads`` -- one CTA per (batch, head); ``batch_heads_tiles`` --
    one CTA per ``rows``-row tile of every ``pad``-padded head; ``capped_tiles``
    -- the same, at most ``per_sm`` per SM; ``capped_table_tiles`` -- one CTA
    per (batch, head, ``blocks_per_tile``-block tile of the ``q_pad``-padded
    ``q_block``-row Q blocks, ``key_tile``-key tile), at most one per SM."""
    rule = record["grid_rule"]
    kind = rule["kind"]
    batch, heads, seqlen, sm_count = int(batch), int(heads), int(seqlen), int(sm_count)
    if min(batch, heads, seqlen, sm_count) <= 0:
        raise ValueError(
            f"the grid rule needs positive extents, got batch={batch}, heads={heads}, "
            f"seqlen={seqlen}, sm_count={sm_count}"
        )
    bh = batch * heads
    if kind == "persistent_tiles":
        ctas, tile_rows = int(rule["ctas_per_tile"]), int(rule["tile_rows"])
        if sm_count < ctas:
            raise ValueError(
                f"the persistent cluster needs at least {ctas} SMs, got {sm_count}"
            )
        return (ctas * min(sm_count // ctas, bh * -(-seqlen // tile_rows)), 1, 1)
    if kind == "split_persistent_tiles":
        plan = split_plan(
            record, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
        )
        return (plan["grid"], 1, 1)
    if kind == "capped_persistent":
        return (max(1, min(sm_count, bh * -(-seqlen // int(rule["unit_rows"])))), 1, 1)
    if kind == "batch_heads":
        return (bh, 1, 1)
    if kind == "capped_table_tiles":
        q_pad, q_block = int(rule["q_pad"]), int(rule["q_block"])
        blocks_per_tile, key_tile = int(rule["blocks_per_tile"]), int(rule["key_tile"])
        q_blocks = -(-seqlen // q_pad) * (q_pad // q_block)
        items = bh * -(-q_blocks // blocks_per_tile) * -(-seqlen // key_tile)
        return (max(1, min(items, sm_count)), 1, 1)
    if kind in ("batch_heads_tiles", "capped_tiles"):
        rows, pad = int(rule["rows"]), int(rule["pad"])
        tiles = bh * -(-seqlen // pad) * (pad // rows)
        if kind == "batch_heads_tiles":
            return (tiles, 1, 1)
        return (max(1, min(tiles, int(rule["per_sm"]) * sm_count)), 1, 1)
    raise ValueError(f"unknown Cake FP8 attention grid rule: {kind!r}")


def launch_args(
    record: dict[str, Any], values: dict[str, Any], grid: tuple[int, int, int]
) -> tuple[Any, ...]:
    """The positional arguments of the record's ``run`` entry: ``values`` by the
    logical name of each kernel parameter of ``arg_plan`` and the grid axes."""
    logical = record["logical"]
    args: list[Any] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(int(grid["xyz".index(key[-1])]))
            continue
        if key not in logical:
            raise ValueError(f"kernel parameter {key!r} has no logical name")
        try:
            args.append(values[logical[key]])
        except KeyError:
            raise ValueError(
                f"the launch needs logical argument {logical[key]!r}"
            ) from None
    return tuple(args)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid`` (the
    record's :func:`launch_grid`) on the current torch stream of the device."""
    record = MODULES[name]
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*launch_args(record, values, grid))


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HEAD_DIM",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "OPTIONAL_STAGES",
    "STAGES",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "records_for",
    "run_module",
    "sm_count",
    "specs_for_arch",
    "split_plan",
]
