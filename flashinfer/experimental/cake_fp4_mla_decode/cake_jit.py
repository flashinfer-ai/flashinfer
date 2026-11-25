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

Host plan and loader glue of the Cake MLA decode over TensorRT-LLM's
``fp4_mla`` KV cache (SM100, SM103): the decode programs and the split-KV
reducer programs of their partials.

Contract of the decode: the cache is TensorRT-LLM's ``fp4_mla`` layout, 128
tokens per page, each token 320 bytes of E2M1 codes (512 NoPE channels, 64
RoPE-base and 64 RoPE-residual channels, the even channel in the low nibble)
with 40 E4M3 block scales per token in the 128x4 swizzle (5,120 bytes per
page), and TensorRT-LLM's MLA V-scale pool (4,096 bytes per page: per NoPE
channel, the K scale of the first token of each 16-token block); the query
is TensorRT-LLM's prequantized query, 320 bytes per ``token * H + head`` row
with its swizzled E4M3 scales; both carry an FP32 global scale on the
device.  ``S = sm_scale (qN.kN + (qRb + qRr).kRb + qRb.kRr) / (q_gs kv_gs)``
over the keys each query token sees (bottom-right causal), softmax, ``O = P
V`` with V the NoPE channels of the same cache bytes, scaled by the V-scale
pool; P is quantized to NVFP4 per 16 keys.  ``out`` is BF16 ``[rows, 512]``.

The reducer launches with programmatic dependent launch (its records'
``launch.use_pdl``): the decode releases it early and it waits for the
decode grid before it reads the partials.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the ``stage`` of the launch order it fills (``decode``,
then ``merge`` when the plan has more than one split), the ``build`` keywords
it was rendered from (the host plan picks a stage's record by them: the
decode by ``long_rows`` and ``persistent``, the reducer by
``warps_per_row``), its launch record, the argument
plan of its ``run`` entry, the kernel-parameter -> logical-argument map, the
grid rule :func:`launch_grid` evaluates and the workspace rule (the decode's
split partials, which the caller owns; ``None``: no scratch).  What to compile
-- sources, digests, flags -- is recorded once in ``build.json`` next to this
file.  The reducers' entries are nvcc builds, which :mod:`.cake_build` serves
from a prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.  The decodes' entries carry
an ``nvrtc`` record: NVRTC compiles their device source, with the recorded
options, to a cubin the library embeds, and nvcc compiles the binding that
launches it (:mod:`.cake_nvrtc`).  Every ``build.json`` entry records
``--use_fast_math`` (the flag the kernels were validated and measured with);
the loader passes ``use_fast_math=False`` so that the manifest, not the
loader, states it.

Tensor maps travel by value in the kernel parameters: the binding encodes a
``CUtensorMap`` from each tensor view per call, so a launch is self-contained
and CUDA-graph capturable.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_fp4_mla_decode_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON of the build keywords (sorted,
every default spelled) and the source kernel the program was rendered from;
a changed keyword is a new program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build, cake_nvrtc

# The verified builds: B200 / GB200 (``sm_100a``) and B300 / GB300
# (``sm_103a``), each compiled with its exact ``-gencode`` flag set
# (``build.json`` ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# The launch order of one decode call: the decode, then, when the plan has
# more than one split, the reducer of its partials.
STAGES = ("decode", "merge")
# Output dimensions per row (the NoPE latent) and rows per decode tile.
LATENT = 512
TILE_ROWS = 128

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_fp4_mla_decode_f8293113b184446b2322": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "kv_k"],
            ["buffer", "kv_v"],
            ["buffer", "kv_sf"],
            ["buffer", "v_pool"],
            ["buffer", "q"],
            ["buffer", "q_sf"],
            ["buffer", "page_table"],
            ["buffer", "kv_lens"],
            ["buffer", "q_gs"],
            ["buffer", "kv_gs"],
            ["buffer", "out"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["parameter", "sm_scale"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "num_split"],
            ["parameter", "page_stride"],
            ["parameter", "page_offset"],
            ["parameter", "batch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "long_rows": True,
            "persistent": False,
            "policy": "rnlsd",
            "tau": 4,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_pair": 2,
            "kind": "split_pairs",
            "tile_rows": 128,
        },
        "kernel": "cake_fp4_mla_decode_f8293113b184446b2322",
        "kernel_symbol": "kernel_cake_fp4_mla_decode_f8293113b184446b2322",
        "launch": {
            "block": (480, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 165888,
            "use_pdl": False,
        },
        "library": "cake_fp4_mla_decode_f8293113b184446b2322",
        "logical": {
            "batch": "batch",
            "kv_gs": "kv_gs",
            "kv_k": "kv_k",
            "kv_lens": "kv_lens",
            "kv_sf": "kv_sf",
            "kv_v": "kv_v",
            "num_heads": "num_heads",
            "num_split": "num_split",
            "out": "out",
            "page_offset": "page_offset",
            "page_stride": "page_stride",
            "page_table": "page_table",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "q": "q",
            "q_gs": "q_gs",
            "q_len": "q_len",
            "q_sf": "q_sf",
            "sm_scale": "sm_scale",
            "v_pool": "v_pool",
        },
        "stage": "decode",
        "workspace_rule": {
            "kind": "split_partials",
            "latent": 512,
        },
    },
    "cake_fp4_mla_decode_0180693993c8dcd16427": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "kv_k"],
            ["buffer", "kv_v"],
            ["buffer", "kv_sf"],
            ["buffer", "v_pool"],
            ["buffer", "q"],
            ["buffer", "q_sf"],
            ["buffer", "page_table"],
            ["buffer", "kv_lens"],
            ["buffer", "q_gs"],
            ["buffer", "kv_gs"],
            ["buffer", "out"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["parameter", "sm_scale"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "num_split"],
            ["parameter", "page_stride"],
            ["parameter", "page_offset"],
            ["parameter", "batch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "long_rows": True,
            "persistent": True,
            "policy": "rnlsd",
            "tau": 4,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_pair": 2,
            "kind": "persistent_pairs",
            "tile_rows": 128,
        },
        "kernel": "cake_fp4_mla_decode_0180693993c8dcd16427",
        "kernel_symbol": "kernel_cake_fp4_mla_decode_0180693993c8dcd16427",
        "launch": {
            "block": (480, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 165888,
            "use_pdl": False,
        },
        "library": "cake_fp4_mla_decode_0180693993c8dcd16427",
        "logical": {
            "batch": "batch",
            "kv_gs": "kv_gs",
            "kv_k": "kv_k",
            "kv_lens": "kv_lens",
            "kv_sf": "kv_sf",
            "kv_v": "kv_v",
            "num_heads": "num_heads",
            "num_split": "num_split",
            "out": "out",
            "page_offset": "page_offset",
            "page_stride": "page_stride",
            "page_table": "page_table",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "q": "q",
            "q_gs": "q_gs",
            "q_len": "q_len",
            "q_sf": "q_sf",
            "sm_scale": "sm_scale",
            "v_pool": "v_pool",
        },
        "stage": "decode",
        "workspace_rule": {
            "kind": "split_partials",
            "latent": 512,
        },
    },
    "cake_fp4_mla_decode_42a27c4a4f2cc133fe01": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "kv_k"],
            ["buffer", "kv_v"],
            ["buffer", "kv_sf"],
            ["buffer", "v_pool"],
            ["buffer", "q"],
            ["buffer", "q_sf"],
            ["buffer", "page_table"],
            ["buffer", "kv_lens"],
            ["buffer", "q_gs"],
            ["buffer", "kv_gs"],
            ["buffer", "out"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["parameter", "sm_scale"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "num_split"],
            ["parameter", "page_stride"],
            ["parameter", "page_offset"],
            ["parameter", "batch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "long_rows": False,
            "persistent": False,
            "policy": "rnlsd",
            "tau": 4,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_pair": 2,
            "kind": "split_pairs",
            "tile_rows": 128,
        },
        "kernel": "cake_fp4_mla_decode_42a27c4a4f2cc133fe01",
        "kernel_symbol": "kernel_cake_fp4_mla_decode_42a27c4a4f2cc133fe01",
        "launch": {
            "block": (480, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 165888,
            "use_pdl": False,
        },
        "library": "cake_fp4_mla_decode_42a27c4a4f2cc133fe01",
        "logical": {
            "batch": "batch",
            "kv_gs": "kv_gs",
            "kv_k": "kv_k",
            "kv_lens": "kv_lens",
            "kv_sf": "kv_sf",
            "kv_v": "kv_v",
            "num_heads": "num_heads",
            "num_split": "num_split",
            "out": "out",
            "page_offset": "page_offset",
            "page_stride": "page_stride",
            "page_table": "page_table",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "q": "q",
            "q_gs": "q_gs",
            "q_len": "q_len",
            "q_sf": "q_sf",
            "sm_scale": "sm_scale",
            "v_pool": "v_pool",
        },
        "stage": "decode",
        "workspace_rule": {
            "kind": "split_partials",
            "latent": 512,
        },
    },
    "cake_fp4_mla_decode_032f4dbae1b3e6c5522c": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "kv_k"],
            ["buffer", "kv_v"],
            ["buffer", "kv_sf"],
            ["buffer", "v_pool"],
            ["buffer", "q"],
            ["buffer", "q_sf"],
            ["buffer", "page_table"],
            ["buffer", "kv_lens"],
            ["buffer", "q_gs"],
            ["buffer", "kv_gs"],
            ["buffer", "out"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["parameter", "sm_scale"],
            ["parameter", "num_heads"],
            ["parameter", "q_len"],
            ["parameter", "num_split"],
            ["parameter", "page_stride"],
            ["parameter", "page_offset"],
            ["parameter", "batch"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "long_rows": False,
            "persistent": True,
            "policy": "rnlsd",
            "tau": 4,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_pair": 2,
            "kind": "persistent_pairs",
            "tile_rows": 128,
        },
        "kernel": "cake_fp4_mla_decode_032f4dbae1b3e6c5522c",
        "kernel_symbol": "kernel_cake_fp4_mla_decode_032f4dbae1b3e6c5522c",
        "launch": {
            "block": (480, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 165888,
            "use_pdl": False,
        },
        "library": "cake_fp4_mla_decode_032f4dbae1b3e6c5522c",
        "logical": {
            "batch": "batch",
            "kv_gs": "kv_gs",
            "kv_k": "kv_k",
            "kv_lens": "kv_lens",
            "kv_sf": "kv_sf",
            "kv_v": "kv_v",
            "num_heads": "num_heads",
            "num_split": "num_split",
            "out": "out",
            "page_offset": "page_offset",
            "page_stride": "page_stride",
            "page_table": "page_table",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "q": "q",
            "q_gs": "q_gs",
            "q_len": "q_len",
            "q_sf": "q_sf",
            "sm_scale": "sm_scale",
            "v_pool": "v_pool",
        },
        "stage": "decode",
        "workspace_rule": {
            "kind": "split_partials",
            "latent": 512,
        },
    },
    "cake_fp4_mla_decode_384210a7aa131bb687ed": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "ext": False,
            "pre_wait": True,
            "warps_per_row": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_warps",
            "warps_per_cta": 8,
            "warps_per_row": 1,
        },
        "kernel": "cake_fp4_mla_decode_384210a7aa131bb687ed",
        "kernel_symbol": "kernel_cake_fp4_mla_decode_384210a7aa131bb687ed",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fp4_mla_decode_384210a7aa131bb687ed",
        "logical": {
            "O": "out",
            "batch": "batch",
            "bmm2_scale": "bmm2_scale",
            "cum_seq_lens_q": "cum_seq_lens_q",
            "m_tiles": "merge_m_tiles",
            "n_full_items": "n_full_items",
            "num_heads": "num_heads",
            "num_split": "num_split",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "tile_rows": "merge_tile_rows",
        },
        "stage": "merge",
        "workspace_rule": None,
    },
    "cake_fp4_mla_decode_8ee8a5d7f516f133be58": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "ext": False,
            "pre_wait": True,
            "warps_per_row": 2,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_warps",
            "warps_per_cta": 8,
            "warps_per_row": 2,
        },
        "kernel": "cake_fp4_mla_decode_8ee8a5d7f516f133be58",
        "kernel_symbol": "kernel_cake_fp4_mla_decode_8ee8a5d7f516f133be58",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fp4_mla_decode_8ee8a5d7f516f133be58",
        "logical": {
            "O": "out",
            "batch": "batch",
            "bmm2_scale": "bmm2_scale",
            "cum_seq_lens_q": "cum_seq_lens_q",
            "m_tiles": "merge_m_tiles",
            "n_full_items": "n_full_items",
            "num_heads": "num_heads",
            "num_split": "num_split",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "tile_rows": "merge_tile_rows",
        },
        "stage": "merge",
        "workspace_rule": None,
    },
    "cake_fp4_mla_decode_03890e366badc8e5325e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "ext": False,
            "pre_wait": True,
            "warps_per_row": 4,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_warps",
            "warps_per_cta": 8,
            "warps_per_row": 4,
        },
        "kernel": "cake_fp4_mla_decode_03890e366badc8e5325e",
        "kernel_symbol": "kernel_cake_fp4_mla_decode_03890e366badc8e5325e",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fp4_mla_decode_03890e366badc8e5325e",
        "logical": {
            "O": "out",
            "batch": "batch",
            "bmm2_scale": "bmm2_scale",
            "cum_seq_lens_q": "cum_seq_lens_q",
            "m_tiles": "merge_m_tiles",
            "n_full_items": "n_full_items",
            "num_heads": "num_heads",
            "num_split": "num_split",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "tile_rows": "merge_tile_rows",
        },
        "stage": "merge",
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
            f"Cake fp4_mla decode target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def record_for(stage: str, arch: str, **build: Any) -> str:
    """The name of the one record of ``stage`` delivered for ``arch`` whose
    ``build`` keywords include ``build`` (``RuntimeError`` when none or more
    than one is): the decode by ``long_rows`` and ``persistent``, the reducer
    by ``warps_per_row``."""
    names = [
        name
        for name, record in MODULES.items()
        if record["stage"] == stage
        and arch in record["arches"]
        and all(record["build"].get(key) == value for key, value in build.items())
    ]
    if len(names) != 1:
        raise RuntimeError(
            f"Cake fp4_mla decode delivers {len(names)} {stage} programs for "
            f"{arch} with {build}; the exported architectures are {ARCHES}"
        )
    return names[0]


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake fp4_mla decode record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags, with NVRTC's cubin of the device
    source embedded for an entry with an ``nvrtc`` record
    (``cake_nvrtc.load_library``)."""
    return cake_nvrtc.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file that
    the local toolkit builds (for ahead-of-time warm-up through FlashInfer's
    ``build_jit_specs``): an entry with an ``nvrtc`` record through
    ``cake_nvrtc.jit_spec``, every other through ``cake_build.jit_spec``."""
    from flashinfer.jit.cpp_ext import get_cuda_version

    version = get_cuda_version()
    cuda = (int(version.major), int(version.minor))
    specs = []
    for library, entry in cake_build.libraries_by_name().items():
        if entry["arch"] != arch or cake_build.prebuilt_path(library).is_file():
            continue
        if cuda < tuple(int(part) for part in entry["min_cuda"].split(".")[:2]):
            continue
        if cake_nvrtc.nvrtc_record(library) is None:
            specs.append(cake_build.jit_spec(library))
        else:
            specs.append(cake_nvrtc.jit_spec(library))
    return specs


def launch_grid(
    record: dict[str, Any],
    *,
    batch: int,
    rows_per_request: int,
    num_split: int,
    pairs: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one record: under ``split_pairs`` (the decode) a cluster
    of ``ctas_per_pair`` CTAs per (split, ``tile_rows``-row tile, request);
    under ``persistent_pairs`` (the persistent decode) one cluster per work
    item up to ``pairs`` (the device's SM pairs, ``sm_count // 2``), the
    clusters looping over the items; under ``row_warps`` (the reducer)
    ``warps_per_row`` warps per output row in CTAs of ``warps_per_cta``
    warps."""
    rule = record["grid_rule"]
    batch, rows_per_request, num_split = (
        int(batch),
        int(rows_per_request),
        int(num_split),
    )
    if batch <= 0 or rows_per_request <= 0 or num_split <= 0:
        raise ValueError(
            "the grid rule needs positive batch, rows and splits, got "
            f"batch={batch}, rows_per_request={rows_per_request}, num_split={num_split}"
        )
    if rule["kind"] == "split_pairs":
        tiles = -(-rows_per_request // int(rule["tile_rows"]))
        return (int(rule["ctas_per_pair"]) * num_split, tiles, batch)
    if rule["kind"] == "persistent_pairs":
        if pairs is None or int(pairs) <= 0:
            raise ValueError(
                "the persistent decode's grid needs the device's SM pairs, got "
                f"pairs={pairs}"
            )
        tiles = -(-rows_per_request // int(rule["tile_rows"]))
        clusters = min(num_split * tiles * batch, int(pairs))
        return (int(rule["ctas_per_pair"]) * clusters, 1, 1)
    if rule["kind"] == "row_warps":
        warps = batch * rows_per_request * int(rule["warps_per_row"])
        return (-(-warps // int(rule["warps_per_cta"])), 1, 1)
    raise ValueError(f"unknown Cake fp4_mla decode grid rule: {rule['kind']!r}")


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
    "LATENT",
    "MODULES",
    "STAGES",
    "TILE_ROWS",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "record_for",
    "run_module",
    "sm_count",
    "specs_for_arch",
]
