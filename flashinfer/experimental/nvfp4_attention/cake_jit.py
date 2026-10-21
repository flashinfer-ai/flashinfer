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

Host plan and loader glue of the Cake experimental NVFP4 attention program
(SM103), the ``backend="cake"`` of
:func:`flashinfer.prefill.prepare_nvfp4_attention`.

Contract of the program: ``O = bf16(softmax(deq(Q) deq(K)^T / sqrt(128))
deq(V))`` over ``[B, H, S, 128]`` heads with E2M1 operands and UE4M3 block
scales of 16 values: ``Q`` / ``K`` ``[B*H*S, 64]`` packed code pairs along the
head dimension, ``Vt`` ``[B*H*128, S/2]`` the pairs of V transposed (quantized
along the sequence), ``SFQ`` / ``SFK`` ``[B*H*S/4, 32]`` and ``SFVtLo`` /
``SFVtHi`` ``[B*H*S/8, 32]`` the scale tiles in the MMA operand layout, ``O``
a BF16 ``[B*H, S, 128]`` view of the caller's output; ``seqlen_q`` /
``seqlen_kv`` the live rows, ``q_stride`` / ``kv_stride`` the padded row
stride of a head, ``softmax_scale_log2`` the base-2 softmax scale and
``total_bh`` the head count.  The route is noncausal with ``S`` a positive
multiple of :data:`TILE_ROWS`; :mod:`.cake_backend` quantizes, packs and binds
the caller's tensors.  Probabilities are re-quantized to E2M1 against each
score tile's maximum, the numerics the kernel was validated under.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the source ``module`` and ``build`` keywords it was
rendered from, its launch record, the argument plan of its ``run`` entry, the
kernel-parameter -> logical-argument map, the grid rule :func:`launch_grid`
evaluates and the workspace rule (``None``: the kernel owns no scratch).
What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.  Every ``build.json`` entry
records ``--use_fast_math`` (the flag the kernel was validated and measured
with); the loader passes ``use_fast_math=False`` so that the manifest, not
the loader, states it.

Tensor maps travel by value in the kernel parameters: the binding encodes a
``CUtensorMap`` from each tensor view per call, so a launch is self-contained,
CUDA-graph capturable, and two layers never share descriptor storage.  The
binding launches the program as a cluster of :data:`CLUSTER_CTAS` CTAs with
programmatic dependent launch (``launch.use_pdl``).

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_nvfp4_attention_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted,
every default spelled>, "module": <dotted source module>}`` of the kernel the
program was rendered from; a changed keyword is a new program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# One verified build: B300 / GB300 (``sm_103a``), compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``).
ARCHES = ("sm_103a",)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 3): "sm_103a",
}

# The head dimension every record's tensor maps are encoded for.
HEAD_DIM = 128
# Rows of one persistent work tile (four 128-row Q blocks per CTA pair) and the
# CTAs of the cluster that serves it: the grid rule of every record.
TILE_ROWS = 512
CLUSTER_CTAS = 2

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "cake_nvfp4_attention_aeef857fb82e9e29a633": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "SFQ"],
            ["buffer", "SFK"],
            ["buffer", "SFVtLo"],
            ["buffer", "SFVtHi"],
            ["buffer", "O"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_kv"],
            ["parameter", "q_stride"],
            ["parameter", "kv_stride"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "total_bh"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "groupwise": False,
            "has_kv_tail": False,
            "is_causal": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_tile": 2,
            "kind": "persistent_tiles",
            "tile_rows": 512,
        },
        "kernel": "cake_nvfp4_attention_aeef857fb82e9e29a633",
        "kernel_symbol": "kernel_cake_nvfp4_attention_aeef857fb82e9e29a633",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 151552,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_attention_aeef857fb82e9e29a633",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "SFK": "SFK",
            "SFQ": "SFQ",
            "SFVtHi": "SFVtHi",
            "SFVtLo": "SFVtLo",
            "Vt": "Vt",
            "kv_stride": "kv_stride",
            "q_stride": "q_stride",
            "seqlen_kv": "seqlen_kv",
            "seqlen_q": "seqlen_q",
            "softmax_scale_log2": "softmax_scale_log2",
            "total_bh": "total_bh",
        },
        "module": "tart.attention_fwd_nvfp4_ptilemax_bf16o.sm100_sm103.fp4_flash_attention",
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
            f"Cake NVFP4 attention target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def record_for(arch: str) -> str:
    """The name of the one record delivered for ``arch`` (``RuntimeError`` when
    none or more than one is: the route has one program per architecture)."""
    names = [name for name, record in MODULES.items() if arch in record["arches"]]
    if len(names) != 1:
        raise RuntimeError(
            f"Cake NVFP4 attention delivers {len(names)} programs for {arch}; "
            f"the exported architectures are {ARCHES}"
        )
    return names[0]


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake NVFP4 attention record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def load_module(arch: str) -> Any:
    """The loaded tvm-ffi module of the one record delivered for ``arch``."""
    return load(record_for(arch), arch)


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


def launch_grid(
    record: dict[str, Any], *, total_bh: int, seqlen_q: int, sm_count: int
) -> tuple[int, int, int]:
    """Launch grid of one record for ``total_bh`` heads of ``seqlen_q`` query rows
    on a device of ``sm_count`` SMs under its ``persistent_tiles`` rule: a
    persistent cluster of ``ctas_per_tile`` CTAs per ``tile_rows``-row tile, at
    most one cluster per ``ctas_per_tile`` SMs."""
    rule = record["grid_rule"]
    if rule["kind"] != "persistent_tiles":
        raise ValueError(f"unknown Cake NVFP4 attention grid rule: {rule['kind']!r}")
    ctas, tile_rows = int(rule["ctas_per_tile"]), int(rule["tile_rows"])
    total_bh, seqlen_q, sm_count = int(total_bh), int(seqlen_q), int(sm_count)
    if total_bh <= 0 or seqlen_q <= 0 or sm_count < ctas:
        raise ValueError(
            f"the grid rule needs positive heads and rows and at least {ctas} SMs, "
            f"got total_bh={total_bh}, seqlen_q={seqlen_q}, sm_count={sm_count}"
        )
    tiles = total_bh * -(-seqlen_q // tile_rows)
    return (ctas * min(sm_count // ctas, tiles), 1, 1)


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
    "CLUSTER_CTAS",
    "HEAD_DIM",
    "MODULES",
    "TILE_ROWS",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "load_module",
    "record_for",
    "run_module",
    "sm_count",
    "specs_for_arch",
]
