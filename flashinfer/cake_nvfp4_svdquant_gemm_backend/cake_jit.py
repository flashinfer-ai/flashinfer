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

Host plan and loader glue of the Cake NVFP4 SVDQuant GEMM programs (SM100 /
SM103), the ``backend="cake"`` of :func:`flashinfer.gemm.mm_nvfp4_svdquant`.

Contract of every program: ``out = bf16(alpha * (deq(A) @ deq(B)^T + D @ L1^T)
+ bias)`` with ``A [M, K/2]`` and ``B [N, K/2]`` packed E2M1 pairs, ``SFA`` /
``SFB`` UE4M3 block-16 scales in the 128x4 swizzled layout, ``alpha`` one FP32
device scalar, ``D [M, rank]`` and ``L1 [N, rank]`` BF16 where the caller has
already divided ``L1`` by ``alpha`` (the kernels never rescale the low-rank
term), an optional BF16 ``bias [N]`` and a BF16 ``out [M, N]``.  Both products
accumulate in FP32 (tcgen05 ``kind::mxf4nvf4`` and ``kind::f16`` into one
TMEM accumulator); the epilogue applies ``alpha`` and ``bias`` in FP32 and
rounds once to BF16.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the tart ``module`` and ``build`` keywords it was rendered
from, its launch record, the argument plan of its ``run`` entry, the
kernel-parameter -> logical-argument map and the ``plan`` (tile sizes, the
scale-factor view the tensor maps expect, the grid rule and the variant).
What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.  ``build.json``'s
``requires.cuda_toolkit.min`` is the compiler floor of the entries; the public
``backend="cake"`` additionally requires CUDA 13.0 or later, checked by
:mod:`flashinfer.gemm.gemm_svdquant`.

Tensor maps travel by value in the kernel parameters: the binding encodes a
``CUtensorMap`` from each tensor view per call, so a launch is self-contained,
CUDA-graph capturable, and two layers never share descriptor storage.  Python
passes plain tensors and views; the scale factors are re-viewed per the
record's ``scale_elems`` (``uint8`` 5-D, ``uint16`` 4-D or ``uint32`` 3-D
boxes of 128 rows).  The register-direct store programs take the raw ``out``
pointer as a trailing ``out_ptr`` parameter beside the ``out`` tensor map; the
host passes ``out`` for both.

Routing is data: :data:`ROUTE_TABLE` is an ordered tuple of predicate rows
over the host scalars ``(M, N, K, rank, has_bias)``, the device's exported
architecture (a row's optional ``arches``; absent applies everywhere) and the
device's SM count (a ``wide_tail`` row admits only the problems whose
:func:`tailsplit_wide_units` schedule on that SM count is the wide one),
resolved once per call by :func:`select_record`; the last row is the
unconditional fallback.  The named
shape violations (``M >= 128``, ``N`` a multiple of 128, ``K`` a multiple of
the selected program's K tile, ``rank`` a multiple of 32 up to 128) raise
``ValueError`` after routing, so the K-tile message names the program that
would have run.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_nvfp4_svdquant_gemm_<20 hex>`` is the first twenty hex
digits of the SHA-256 of ``f"{module}\\n{json.dumps(build, sort_keys=True)}"``
over the record's tart ``module`` and ``build`` keywords.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# One verified build per architecture: B200 / GB200 (``sm_100a``) and B300 /
# GB300 (``sm_103a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); no family fallback.
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# Every tensor map of every program carries a 128-row box, so ``M >= 128``,
# ``N`` is a multiple of 128 and the scale-factor views are blocks of 128 rows.
BLOCK_M = 128
# The LoRA rank is a multiple of 32 (one ``rank_tile``) up to 128.
RANK_GRANULARITY = 32
MAX_RANK = 128
# One UE4M3 scale per 16 K elements; the swizzled layout stores them as
# ``[rows / 128, K / 64, 4, 8, 16]`` bytes per 128-row block.
SF_BLOCK = 16
# The kernels take every scalar as a 32-bit integer.
INT32_MAX = 2**31 - 1

# The host vocabulary of the plan: the variant names, the grid rules
# :func:`launch_grid` evaluates and the scale-factor views :func:`plan_values`
# builds.
VARIANTS = ("bias", "nobias")
GRID_RULES = (
    "linear",
    "cluster_n2",
    "pair_persistent",
    "pair_tailsplit",
    "pair_tailsplit_wide",
    "splitk_cluster",
)
SCALE_ELEMS = ("uint8", "uint16", "uint32")

# Logical argument names the host API binds; every record maps its kernel
# parameter names onto this vocabulary (``record["logical"]``).  ``a_sf`` /
# ``b_sf`` are the scale-factor views of the record's ``scale_elems``; ``bias``
# is the bias tensor, or ``out`` as a carrier when ``has_bias`` is 0 (the
# kernel never reads it then, but the binding needs a valid CUDA tensor).
# The ``D_tail`` / ``L1_tail`` descriptors of the rank-96 tail programs and
# the ``B_half`` descriptor of the tail-split pair program (and the ``B_wide``
# descriptor of its wide-tail build) are second tensor maps over the same
# ``d`` / ``l1`` / ``b`` tensors; the ``out_ptr`` parameter
# of the register-direct store programs (``out_store`` ``v8`` / ``v8hint``) is
# the raw pointer of the same ``out`` tensor the ``out`` tensor map views.
LOGICAL_TENSORS = (
    "a",
    "b",
    "a_sf",
    "b_sf",
    "d",
    "l1",
    "alpha",
    "bias",
    "out",
)
LOGICAL_SCALARS = (
    "M",
    "N",
    "K_tiles",
    "rank_tiles",
    "rank",
    "rank_chunks",
    "grid_n",
    "has_bias",
    "tok_tiles",
    "w_tiles",
    "num_tiles",
    "raster_group",
    "num_units",
    "full_units",
    "tail_split",
)
# Kernel parameter name -> logical name over the whole family; a record's
# ``logical`` is this map restricted to the parameters its plan keeps.
PARAMETER_LOGICAL: dict[str, str] = {
    "A": "a",
    "B": "b",
    "B_half": "b",
    "B_wide": "b",
    "SFA": "a_sf",
    "SFB": "b_sf",
    "D": "d",
    "L1": "l1",
    "D_tail": "d",
    "L1_tail": "l1",
    "alpha": "alpha",
    "bias": "bias",
    "out": "out",
    "out_ptr": "out",
    "M": "M",
    "N": "N",
    "K_tiles": "K_tiles",
    "rank_tiles": "rank_tiles",
    "rank": "rank",
    "rank_chunks": "rank_chunks",
    "grid_n": "grid_n",
    "has_bias": "has_bias",
    "tok_tiles": "tok_tiles",
    "w_tiles": "w_tiles",
    "num_tiles": "num_tiles",
    "raster_group": "raster_group",
    "num_units": "num_units",
    "full_units": "full_units",
    "tail_split": "tail_split",
}

# One record per program, keyed by the program name (``library`` == ``kernel``).
# ``plan``: ``block_m`` / ``block_n`` / ``block_k`` are the program's tile,
# ``rank_storage`` the rank the kernel is built for (``None``: the rank is a
# runtime parameter), ``scale_elems`` the element type of the scale-factor
# views, ``grid_rule`` one of :data:`GRID_RULES`, ``variant`` one of
# :data:`VARIANTS`, ``pdl`` whether the program's launch carries programmatic
# dependent launch (must agree with ``launch.use_pdl``), ``split_k`` (the
# ``splitk_cluster`` programs only) the cluster size the K dimension is
# split over.
MODULES: dict[str, dict[str, Any]] = {
    # tactic 0, 128x128x128, 11-stage ring, exact BF16 epilogue; the fallback
    # of every problem no other row claims.
    "cake_nvfp4_svdquant_gemm_db7820791092b67f10cd": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_db7820791092b67f10cd",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_db7820791092b67f10cd",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220160,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_db7820791092b67f10cd",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic0_exact_epilogue",
        "plan": {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "grid_rule": "linear",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint8",
            "variant": "bias",
        },
    },
    # tactic 0 with the rank-64 low-rank operands and the bias staged in shared
    # memory.
    "cake_nvfp4_svdquant_gemm_8463dfd07714ab7a0ca6": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_8463dfd07714ab7a0ca6",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_8463dfd07714ab7a0ca6",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220416,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_8463dfd07714ab7a0ca6",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic0_rank64_bias_smem",
        "plan": {
            "block_m": 128,
            "block_n": 128,
            "block_k": 128,
            "grid_rule": "linear",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint8",
            "variant": "bias",
        },
    },
    # tactic 25, 128x128x256 with a plain-pointer output store; runtime rank
    # in chunks of 64.
    "cake_nvfp4_svdquant_gemm_751651fd3ba4ef54ce53": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank"],
            ["parameter", "rank_chunks"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_751651fd3ba4ef54ce53",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_751651fd3ba4ef54ce53",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 185344,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_751651fd3ba4ef54ce53",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank": "rank",
            "rank_chunks": "rank_chunks",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic25",
        "plan": {
            "block_m": 128,
            "block_n": 128,
            "block_k": 256,
            "grid_rule": "linear",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint8",
            "variant": "bias",
        },
    },
    # tactic 25 for rank 96: a 64-wide low-rank chunk plus a compact 32-wide
    # tail read through second tensor maps (``D_tail`` / ``L1_tail``) over the
    # same ``d`` / ``l1`` tensors.
    "cake_nvfp4_svdquant_gemm_e15f358ed35018ca0015": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "D_tail"],
            ["buffer", "L1_tail"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank"],
            ["parameter", "rank_chunks"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_e15f358ed35018ca0015",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_e15f358ed35018ca0015",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 201984,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_e15f358ed35018ca0015",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "D_tail": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank": "rank",
            "rank_chunks": "rank_chunks",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic25_rank96_k32_tail",
        "plan": {
            "block_m": 128,
            "block_n": 128,
            "block_k": 256,
            "grid_rule": "linear",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint8",
            "variant": "bias",
        },
    },
    # Two-CTA persistent 256x256x256 (cta_group::2) with a static rank, one CTA
    # pair per scheduled tile claimed through cluster launch control; 32-bit
    # scale-factor view; programmatic dependent launch.  cake_r2: the output
    # leaves the epilogue registers through four 256-bit stores per thread and
    # subtile (``out_store="v8"``; the raw ``out`` pointer is the trailing
    # ``out_ptr`` parameter beside the ``out`` tensor map), on the launches
    # without an L2 store hint (``K_tiles < 24`` and ``w_tiles < 24`` in
    # 256-wide tiles: K < 6144 and N < 6016).  One build per rank.
    "cake_nvfp4_svdquant_gemm_63c7c8683df10c75591b": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_63c7c8683df10c75591b",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_63c7c8683df10c75591b",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_63c7c8683df10c75591b",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"clc_lead_kb": 8, "rank": 32}
    # (round-3 final of the exact-dispatch rows below; the plan of ``63c7c8683df10c75591b``).
    "cake_nvfp4_svdquant_gemm_1d78f2bd5722fb02b75a": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"clc_lead_kb": 8, "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_1d78f2bd5722fb02b75a",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_1d78f2bd5722fb02b75a",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_1d78f2bd5722fb02b75a",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_f57f1cbd289ee50c237e": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8", "rank": 64},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_f57f1cbd289ee50c237e",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_f57f1cbd289ee50c237e",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_f57f1cbd289ee50c237e",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 64,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_be138d4583b02c247633": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8", "rank": 96},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_be138d4583b02c247633",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_be138d4583b02c247633",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_be138d4583b02c247633",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 96,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_84b1b76b4ef917592df0": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8", "rank": 128},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_84b1b76b4ef917592df0",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_84b1b76b4ef917592df0",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_84b1b76b4ef917592df0",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 128,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The same program in its hinted store form (``out_store="v8hint"``: the
    # ``.L1::no_allocate.L2::evict_first`` 256-bit stores on sm_103a, the plain
    # ones on sm_100a) for the launches with ``K_tiles >= 24`` or ``w_tiles >=
    # 24`` (K >= 6144 or N >= 6016).  One build per rank.
    "cake_nvfp4_svdquant_gemm_3880e0aea2ecbe59106a": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_3880e0aea2ecbe59106a",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_3880e0aea2ecbe59106a",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_3880e0aea2ecbe59106a",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"out_hint": "on", "out_store": "v8hint", "rank": 32}
    # (round-3 final of the exact-dispatch rows below; the plan of ``3880e0aea2ecbe59106a``).
    "cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_hint": "on", "out_store": "v8hint", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_99be5794f0107a539d27": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 64},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_99be5794f0107a539d27",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_99be5794f0107a539d27",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_99be5794f0107a539d27",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 64,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"out_hint": "on", "out_store": "v8hint", "rank": 64}
    # (round-3 final of the exact-dispatch rows below; the plan of ``99be5794f0107a539d27``).
    "cake_nvfp4_svdquant_gemm_d16d801db12c7bdde4dc": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_hint": "on", "out_store": "v8hint", "rank": 64},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_d16d801db12c7bdde4dc",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_d16d801db12c7bdde4dc",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_d16d801db12c7bdde4dc",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 64,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_a474f70abaf1b70ca4fe": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 96},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_a474f70abaf1b70ca4fe",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_a474f70abaf1b70ca4fe",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_a474f70abaf1b70ca4fe",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 96,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_9f7d814ace4cd94fc4d6": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 128},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_9f7d814ace4cd94fc4d6",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_9f7d814ace4cd94fc4d6",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_9f7d814ace4cd94fc4d6",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 128,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # tactic 14 with the fused bias: 128x64x128 tiles on clusters of two
    # N-adjacent CTAs; 16-bit scale-factor view.
    "cake_nvfp4_svdquant_gemm_a72d5469d8a64ca32645": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_a72d5469d8a64ca32645",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_a72d5469d8a64ca32645",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 218240,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_a72d5469d8a64ca32645",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_bias",
        "plan": {
            "block_m": 128,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "bias",
        },
    },
    # tactic 14 on 64-row M tiles with the fused bias: 128x64x128 MMAs streaming
    # the live 64-row half of A, clusters of two N-adjacent CTAs, grid
    # ``(ceil(M / 64), N / 64)``; 16-bit scale-factor view.  cake_r2 form A: the
    # overlap prologue (``prologue="overlap"``: barriers published by one relaxed
    # cluster rendezvous, the TMEM allocation overlapped with the first load
    # stages) and the fused 64-column epilogue (``epi="fused64"``), for the
    # M 129 rows the K12288-class rows below do not take.  Built with the
    # shared-memory bias epilogue (``bias=True``).
    "cake_nvfp4_svdquant_gemm_e6789ee4eaed1dff09ae": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"bias": True, "epi": "fused64", "prologue": "overlap"},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_e6789ee4eaed1dff09ae",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_e6789ee4eaed1dff09ae",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 210048,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_e6789ee4eaed1dff09ae",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "bias",
        },
    },
    # The tactic14_m64_bias program built with {"bias": True, "epi": "fused64", "prologue": "overlap", "stages": 15}
    # (round-3 final of the exact-dispatch rows below; the plan of ``e6789ee4eaed1dff09ae``).
    "cake_nvfp4_svdquant_gemm_2270204930b79b8caef7": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"bias": True, "epi": "fused64", "prologue": "overlap", "stages": 15},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_2270204930b79b8caef7",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_2270204930b79b8caef7",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_2270204930b79b8caef7",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "bias",
        },
    },
    # Form A with the TMA L2 promotion of the weight descriptors (``l2_promotion
    # = "l2_256b"``: B and SFB are promoted to 256-byte L2 lines; the kernel
    # text is otherwise form A's), built with the bias epilogue: the sm_103a
    # build of the (129, 3072, 3072, {32, 64}, True) rows.  Ships for sm_103a
    # only (sm_100a keeps form A on those rows).
    "cake_nvfp4_svdquant_gemm_99bdf6e03d704178faf9": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": True,
            "epi": "fused64",
            "l2_promotion": "l2_256b",
            "prologue": "overlap",
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_99bdf6e03d704178faf9",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_99bdf6e03d704178faf9",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 210048,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_99bdf6e03d704178faf9",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "bias",
        },
    },
    # The 16-stage aliased ring of form A ({"stages": 16, "epi_alias": True})
    # with the TMA L2 promotion of the weight descriptors, built with the bias
    # epilogue: the sm_103a build of the (129, 3072, 3072, {96, 128}, True)
    # rows.  Ships for sm_103a only.
    "cake_nvfp4_svdquant_gemm_01405277c4bbc4031fbd": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": True,
            "epi": "fused64",
            "epi_alias": True,
            "l2_promotion": "l2_256b",
            "prologue": "overlap",
            "stages": 16,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_01405277c4bbc4031fbd",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_01405277c4bbc4031fbd",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 230528,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_01405277c4bbc4031fbd",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "bias",
        },
    },
    # Form A built without the bias epilogue (``bias=False``): the ``bias``
    # pointer and ``has_bias`` stay in the ABI and are never read, so only
    # bias-free problems are routed to it.
    "cake_nvfp4_svdquant_gemm_8dc2b79ff158918dd5c8": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"bias": False, "epi": "fused64", "prologue": "overlap"},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_8dc2b79ff158918dd5c8",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_8dc2b79ff158918dd5c8",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 210048,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_8dc2b79ff158918dd5c8",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "nobias",
        },
    },
    # Form A with the TMA L2 promotion of the weight descriptors, built without
    # the bias epilogue: the sm_103a build of the (129, 3072, 3072, 32, False)
    # row.  Ships for sm_103a only.
    "cake_nvfp4_svdquant_gemm_8319e1548886614ab273": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": False,
            "epi": "fused64",
            "l2_promotion": "l2_256b",
            "prologue": "overlap",
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_8319e1548886614ab273",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_8319e1548886614ab273",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 210048,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_8319e1548886614ab273",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "nobias",
        },
    },
    # The 16-stage aliased ring of form A with the TMA L2 promotion of the
    # weight descriptors, built without the bias epilogue: the sm_103a build of
    # the (129, 3072, 3072, 128, False) row.  Ships for sm_103a only.
    "cake_nvfp4_svdquant_gemm_23bfc212699bbb4e6214": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": False,
            "epi": "fused64",
            "epi_alias": True,
            "l2_promotion": "l2_256b",
            "prologue": "overlap",
            "stages": 16,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_23bfc212699bbb4e6214",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_23bfc212699bbb4e6214",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 230528,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_23bfc212699bbb4e6214",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "nobias",
        },
    },
    # cake_r2 form B for the M 129 rows with K >= 12288 and K % 256 == 0 (the
    # K12288-class rows): form A on 256-wide K boxes (``k_box=256``, four K64
    # MMA steps per stage), an 11-stage ring (``stages=11``) and the epilogue
    # staging aliased onto the drained ring (``epi_alias=True``).  Bias build.
    "cake_nvfp4_svdquant_gemm_233bc54a42d3ffd5567d": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": True,
            "epi": "fused64",
            "epi_alias": True,
            "k_box": 256,
            "prologue": "overlap",
            "stages": 11,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_233bc54a42d3ffd5567d",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_233bc54a42d3ffd5567d",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226432,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_233bc54a42d3ffd5567d",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "bias",
        },
    },
    # Form B without the bias epilogue (``bias=False``); bias-free problems only.
    "cake_nvfp4_svdquant_gemm_d615c1bd45783aae8506": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": False,
            "epi": "fused64",
            "epi_alias": True,
            "k_box": 256,
            "prologue": "overlap",
            "stages": 11,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_d615c1bd45783aae8506",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_d615c1bd45783aae8506",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226432,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_d615c1bd45783aae8506",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "nobias",
        },
    },
    # The two-CTA persistent program with the tail split: the tail tiles of the
    # last partial wave run as two 256x128 half units each (``B_half``: a 64-row
    # box over the same ``b`` tensor), scheduled by :func:`tailsplit_units`;
    # otherwise the ``2cta_persistent`` program's whole-tile path.  One build per
    # rank; routed only to the shapes where the split measured faster.  The rows
    # by rank are hinted launches (``K_tiles`` or ``w_tiles`` 48), so those builds
    # ship in the ``out_store="v8hint"`` form (the trailing ``out_ptr`` parameter
    # is the raw ``out`` pointer); the two sm_100a K 3072 rows (M 6912 rank 96,
    # M 16384 rank 32) take the TMA-store form with the register bias path (the
    # last two records).
    "cake_nvfp4_svdquant_gemm_e4f1cd01b65bed25c4f1": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_e4f1cd01b65bed25c4f1",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_e4f1cd01b65bed25c4f1",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_e4f1cd01b65bed25c4f1",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_tailsplit program built with {"load_order": "b_first", "out_store": "v8hint", "rank": 32}
    # (round-3 final of the exact-dispatch rows below; the plan of ``e4f1cd01b65bed25c4f1``).
    "cake_nvfp4_svdquant_gemm_2ca0cf40f7bbfca9df92": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"load_order": "b_first", "out_store": "v8hint", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_2ca0cf40f7bbfca9df92",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_2ca0cf40f7bbfca9df92",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_2ca0cf40f7bbfca9df92",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_0bd3a4870db1dc91bce4": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 64},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_0bd3a4870db1dc91bce4",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_0bd3a4870db1dc91bce4",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_0bd3a4870db1dc91bce4",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 64,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_a867c135c6da108b5f82": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 96},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_a867c135c6da108b5f82",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_a867c135c6da108b5f82",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_a867c135c6da108b5f82",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 96,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    "cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 128},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 128,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_tailsplit program built with {"bias_path": "regs", "rank": 96}: the
    # TMA-store form with the register bias path, the plan of ``e4f1cd01b65bed25c4f1``
    # without ``out_ptr``; the sm_100a route of (6912, 3072, 3072, 96, True), 324 pair
    # tiles whose 20 / 28 tail tiles split on 152 / 148 SMs.
    "cake_nvfp4_svdquant_gemm_0f7f27a483c337265bf9": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"bias_path": "regs", "rank": 96},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_0f7f27a483c337265bf9",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_0f7f27a483c337265bf9",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_0f7f27a483c337265bf9",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 96,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"k96": True, "k96_form": "panel",
    # "out_hint": "on", "out_store": "v8hint", "rank": 32}: the sm_103a K=96
    # block-scaled MMA (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K
    # stage) of ``fdcf777b3acd972f8841`` ({"out_hint": "on", "out_store": "v8hint",
    # "rank": 32}), whose host plan it shares; ships for sm_103a only (the
    # exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "k96": True,
            "k96_form": "panel",
            "out_hint": "on",
            "out_store": "v8hint",
            "rank": 32,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"k96": True, "k96_form": "panel",
    # "out_hint": "on", "out_store": "v8hint", "rank": 64}: the sm_103a K=96
    # block-scaled MMA (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K
    # stage) of ``d16d801db12c7bdde4dc`` ({"out_hint": "on", "out_store": "v8hint",
    # "rank": 64}), whose host plan it shares; ships for sm_103a only (the
    # exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_37c3da710589257a99f1": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "k96": True,
            "k96_form": "panel",
            "out_hint": "on",
            "out_store": "v8hint",
            "rank": 64,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_37c3da710589257a99f1",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_37c3da710589257a99f1",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_37c3da710589257a99f1",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 64,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"k96": True, "k96_form": "panel",
    # "out_hint": "on", "out_store": "v8hint", "rank": 96}: the sm_103a K=96
    # block-scaled MMA (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K
    # stage) of the {"out_hint": "on", "out_store": "v8hint", "rank": 96} build, whose
    # host plan it shares; ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_ff27b5bafbc8d2d15d5b": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "k96": True,
            "k96_form": "panel",
            "out_hint": "on",
            "out_store": "v8hint",
            "rank": 96,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_ff27b5bafbc8d2d15d5b",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_ff27b5bafbc8d2d15d5b",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_ff27b5bafbc8d2d15d5b",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 96,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"k96": True, "k96_form": "panel",
    # "out_hint": "on", "out_store": "v8hint", "rank": 128}: the sm_103a K=96
    # block-scaled MMA (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K
    # stage) of the {"out_hint": "on", "out_store": "v8hint", "rank": 128} build, whose
    # host plan it shares; ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_9d95357d28c063004b71": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "k96": True,
            "k96_form": "panel",
            "out_hint": "on",
            "out_store": "v8hint",
            "rank": 128,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_9d95357d28c063004b71",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_9d95357d28c063004b71",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_9d95357d28c063004b71",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 128,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"k96": True, "k96_form": "panel", "rank":
    # 32}: the sm_103a K=96 block-scaled MMA (kind::mxf4nvf4.block16) form (K=96 + K=96
    # + K=64 per 256-deep K stage) of the {"rank": 32} build, whose host plan it shares;
    # ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_5a1a17e73ef429cf8805": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"k96": True, "k96_form": "panel", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_5a1a17e73ef429cf8805",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_5a1a17e73ef429cf8805",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_5a1a17e73ef429cf8805",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"k96": True, "k96_form": "panel",
    # "out_store": "v8hint", "rank": 32}: the sm_103a K=96 block-scaled MMA
    # (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K stage) of
    # ``3880e0aea2ecbe59106a`` ({"out_store": "v8hint", "rank": 32}), whose host plan it
    # shares; ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"k96": True, "k96_form": "panel", "out_store": "v8hint", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"k96": True, "k96_form": "panel",
    # "out_store": "v8", "rank": 32}: the sm_103a K=96 block-scaled MMA
    # (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K stage) of
    # ``63c7c8683df10c75591b`` ({"out_store": "v8", "rank": 32}), whose host plan it
    # shares; ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_a234cd99c8b3ef6af762": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"k96": True, "k96_form": "panel", "out_store": "v8", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_a234cd99c8b3ef6af762",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_a234cd99c8b3ef6af762",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_a234cd99c8b3ef6af762",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_tailsplit program built with {"k96": True, "k96_form": "panel",
    # "out_store": "v8hint", "rank": 32}: the sm_103a K=96 block-scaled MMA
    # (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K stage) of
    # ``e4f1cd01b65bed25c4f1`` ({"out_store": "v8hint", "rank": 32}), whose host plan it
    # shares; ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_151d20658188ea0e9651": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"k96": True, "k96_form": "panel", "out_store": "v8hint", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_151d20658188ea0e9651",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_151d20658188ea0e9651",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_151d20658188ea0e9651",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_tailsplit program built with {"k96": True, "k96_form": "panel",
    # "out_store": "v8hint", "rank": 128}: the sm_103a K=96 block-scaled MMA
    # (kind::mxf4nvf4.block16) form (K=96 + K=96 + K=64 per 256-deep K stage) of
    # ``4f451914c2d6580eb2de`` ({"out_store": "v8hint", "rank": 128}), whose host plan
    # it shares; ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_34ae602d0db3ffa24b4b": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"k96": True, "k96_form": "panel", "out_store": "v8hint", "rank": 128},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_34ae602d0db3ffa24b4b",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_34ae602d0db3ffa24b4b",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_34ae602d0db3ffa24b4b",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 128,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_tailsplit program built with {"k96": True, "out_store": "v8hint", "rank":
    # 32}: the sm_103a K=96 block-scaled MMA (kind::mxf4nvf4.block16) form (K192 scale
    # units) of ``e4f1cd01b65bed25c4f1`` ({"out_store": "v8hint", "rank": 32}), whose
    # host plan it shares; ships for sm_103a only (the exact-dispatch rows below).
    "cake_nvfp4_svdquant_gemm_60d29f6b0abb512843b5": {
        "arches": ["sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"k96": True, "out_store": "v8hint", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_60d29f6b0abb512843b5",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_60d29f6b0abb512843b5",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_60d29f6b0abb512843b5",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The 2cta_persistent program built with {"bias_path": "regs", "rank": 32}: the
    # ``"bias_path": "regs"`` form of the {"rank": 32} build, whose host plan it shares,
    # for the sm_100a exact-dispatch row below that measured faster with it; it ships
    # for both architectures like the other exact-dispatch builds.
    "cake_nvfp4_svdquant_gemm_c832b08a912e4c58112b": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"bias_path": "regs", "rank": 32},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_c832b08a912e4c58112b",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_c832b08a912e4c58112b",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_c832b08a912e4c58112b",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "out": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_persistent",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_persistent",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
    # The tactic14_m64_bias program built with {"bias": True, "epi": "fused64",
    # "l2_promotion": "l2_256b", "prologue": "overlap", "stages": 15}: the
    # ``"l2_promotion": "l2_256b"`` form of ``2270204930b79b8caef7`` ({"bias": True,
    # "epi": "fused64", "prologue": "overlap", "stages": 15}), whose host plan it
    # shares, for the sm_100a exact-dispatch rows below that measured faster with it; it
    # ships for sm_100a only (its source builds it there only).
    "cake_nvfp4_svdquant_gemm_05bda082a73c0c1a68b0": {
        "arches": ["sm_100a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": True,
            "epi": "fused64",
            "l2_promotion": "l2_256b",
            "prologue": "overlap",
            "stages": 15,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_05bda082a73c0c1a68b0",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_05bda082a73c0c1a68b0",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_05bda082a73c0c1a68b0",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "bias",
        },
    },
    # The tactic14_m64_bias program built with {"bias": False, "epi": "fused64",
    # "l2_promotion": "l2_256b", "prologue": "overlap", "stages": 15}: the
    # ``"l2_promotion": "l2_256b"`` form of the {"bias": False, "epi": "fused64",
    # "prologue": "overlap", "stages": 15} build, whose host plan it shares, for the
    # sm_100a exact-dispatch rows below that measured faster with it; it ships for
    # sm_100a only (its source builds it there only).
    "cake_nvfp4_svdquant_gemm_5852ad0db045e502e10a": {
        "arches": ["sm_100a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": False,
            "epi": "fused64",
            "l2_promotion": "l2_256b",
            "prologue": "overlap",
            "stages": 15,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_5852ad0db045e502e10a",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_5852ad0db045e502e10a",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 224384,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_5852ad0db045e502e10a",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_tactic14_m64_bias",
        "plan": {
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "grid_rule": "cluster_n2",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint16",
            "variant": "nobias",
        },
    },
    # The swapab_splitk program built with {"bias": False, "inbox": "alias",
    # "l2_promotion": "l2_256b", "split_k": 4, "stages": 11}: the ``"inbox": "alias"``,
    # ``"l2_promotion": "l2_256b"``, ``"stages": 11`` form of the {"bias": False,
    # "split_k": 4} build, whose host plan it shares, for the sm_100a/sm_103a
    # exact-dispatch rows below that measured faster with it; it ships for both
    # architectures like the other exact-dispatch builds.
    "cake_nvfp4_svdquant_gemm_243af01860dcca0f4e2c": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": False,
            "inbox": "alias",
            "l2_promotion": "l2_256b",
            "split_k": 4,
            "stages": 11,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_243af01860dcca0f4e2c",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_243af01860dcca0f4e2c",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220672,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_243af01860dcca0f4e2c",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_swapab_splitk",
        "plan": {
            "block_m": 136,
            "block_n": 128,
            "block_k": 128,
            "grid_rule": "splitk_cluster",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint32",
            "split_k": 4,
            "variant": "nobias",
        },
    },
    # The swapab_splitk program built with {"bias": True, "inbox": "alias",
    # "l2_promotion": "l2_256b", "split_k": 4, "stages": 11}: the ``"inbox": "alias"``,
    # ``"l2_promotion": "l2_256b"``, ``"stages": 11`` form of the {"bias": True,
    # "split_k": 4} build, whose host plan it shares, for the sm_100a/sm_103a
    # exact-dispatch rows below that measured faster with it; it ships for both
    # architectures like the other exact-dispatch builds.
    "cake_nvfp4_svdquant_gemm_36c42441c9d625e03c54": {
        "arches": ["sm_100a", "sm_103a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "L1"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "rank_tiles"],
            ["parameter", "grid_n"],
            ["parameter", "has_bias"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "bias": True,
            "inbox": "alias",
            "l2_promotion": "l2_256b",
            "split_k": 4,
            "stages": 11,
        },
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_36c42441c9d625e03c54",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_36c42441c9d625e03c54",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220672,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_svdquant_gemm_36c42441c9d625e03c54",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "grid_n": "grid_n",
            "has_bias": "has_bias",
            "out": "out",
            "rank_tiles": "rank_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_swapab_splitk",
        "plan": {
            "block_m": 136,
            "block_n": 128,
            "block_k": 128,
            "grid_rule": "splitk_cluster",
            "pdl": False,
            "rank_storage": None,
            "scale_elems": "uint32",
            "split_k": 4,
            "variant": "bias",
        },
    },
    # The 2cta_tailsplit program built with {"out_store": "v8hint", "rank": 32,
    # "wide_tail": 192}: the build of ``e4f1cd01b65bed25c4f1`` whose last partial
    # wave, when it holds more than half and at most three quarters of the CTA
    # pairs, runs as one wave of 192-wide and 128-wide units (``B_wide``: a
    # 96-row box over the same ``b`` tensor), scheduled by
    # :func:`tailsplit_wide_units` (grid rule ``pair_tailsplit_wide``);
    # otherwise the schedule of :func:`tailsplit_units`.  Every output element
    # is still produced by exactly one unit in the whole tile's accumulation
    # order.  It ships for sm_100a only, for the exact-dispatch row below.
    "cake_nvfp4_svdquant_gemm_d758d9efa72998dbcf96": {
        "arches": ["sm_100a"],
        # derived; rewritten by the exporter
        "arg_plan": [
            ["buffer", "D"],
            ["buffer", "alpha"],
            ["buffer", "bias"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K_tiles"],
            ["parameter", "tok_tiles"],
            ["parameter", "w_tiles"],
            ["parameter", "num_tiles"],
            ["parameter", "raster_group"],
            ["parameter", "has_bias"],
            ["parameter", "rank"],
            ["parameter", "num_units"],
            ["parameter", "full_units"],
            ["parameter", "tail_split"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_half"],
            ["buffer", "B_wide"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "L1"],
            ["buffer", "L1_tail"],
            ["buffer", "out_ptr"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {"out_store": "v8hint", "rank": 32, "wide_tail": 192},
        "ffi_entry": "run",
        "kernel": "cake_nvfp4_svdquant_gemm_d758d9efa72998dbcf96",
        "kernel_symbol": "kernel_cake_nvfp4_svdquant_gemm_d758d9efa72998dbcf96",  # derived; rewritten by the exporter
        "launch": {  # derived; rewritten by the exporter
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228992,
            "use_pdl": True,
        },
        "library": "cake_nvfp4_svdquant_gemm_d758d9efa72998dbcf96",
        # derived; rewritten by the exporter
        "logical": {
            "A": "a",
            "B": "b",
            "B_half": "b",
            "B_wide": "b",
            "D": "d",
            "K_tiles": "K_tiles",
            "L1": "l1",
            "L1_tail": "l1",
            "M": "M",
            "N": "N",
            "SFA": "a_sf",
            "SFB": "b_sf",
            "alpha": "alpha",
            "bias": "bias",
            "full_units": "full_units",
            "has_bias": "has_bias",
            "num_tiles": "num_tiles",
            "num_units": "num_units",
            "out": "out",
            "out_ptr": "out",
            "rank": "rank",
            "raster_group": "raster_group",
            "tail_split": "tail_split",
            "tok_tiles": "tok_tiles",
            "w_tiles": "w_tiles",
        },
        "module": "tart.mm_nvfp4_svdquant.sm100_sm103.flashinfer_blackwell_nvfp4_svdquant_gemm_2cta_tailsplit",
        "plan": {
            "block_m": 256,
            "block_n": 256,
            "block_k": 256,
            "grid_rule": "pair_tailsplit_wide",
            "pdl": True,
            "rank_storage": 32,
            "scale_elems": "uint32",
            "variant": "bias",
        },
    },
}

# The routing table: ordered predicate rows over the host scalars, the first
# admitting row wins and the last row admits everything.  A row's optional
# keys: ``m_min`` / ``m_max``, ``n_min`` / ``n_max``, ``k_min`` / ``k_max``
# (inclusive bounds), ``k_multiple`` (``K`` divisible by it), ``ranks`` (the
# admitted ranks), ``bias`` (``True`` / ``False``; absent or ``None`` admits
# both), ``arches`` (the exported architectures the row applies to; absent
# applies to every architecture, so one problem may dispatch to different
# programs on ``sm_100a`` and ``sm_103a``) and ``wide_tail`` (``True``: the row
# admits a problem only where :func:`tailsplit_wide_units` on the device's SM
# count at the problem's :func:`raster_group` is the wide schedule, else the
# problem continues with the next rows); the target is ``record`` or
# ``records_by_rank`` (rank -> record).
# A row never widens the named shape violations: :func:`select_record`
# validates ``M``, ``N``, ``K`` against the selected program and ``rank`` after
# routing.  The head rows are the measured routes of the convergence rounds
# (the tail-split and whole-tile pair programs in the store form of the
# launch, the 64-row tactic-14 builds of the M 129 rows per K class); the
# tail rows reproduce the previous dispatcher for every other problem.
ROUTE_TABLE: tuple[dict[str, Any], ...] = (
    # Exact-dispatch rows: the sm_103a rows of the M 129 / N 3072 / K 3072
    # problems (the L2-promoted 64-row builds), the split-K rows, the
    # convergence finals' rows, the sm_103a rows of the K=96 builds of the
    # two-CTA programs and the sm_100a row of the wide-tail build of the
    # tail-split program, so that each names the
    # program of exactly one (M, N, K, rank, bias), per architecture where the two
    # architectures measured different winners (``arches``; absent = both):
    #   (129, 3072, 3072, {32, 64}, True) on sm_103a -> 99bdf6e03d704178faf9 (L2-promoted form A)
    #   (129, 3072, 3072, 32, False) on sm_103a -> 8319e1548886614ab273 (L2-promoted form A)
    #   (129, 3072, 3072, 32, True) on sm_100a -> 05bda082a73c0c1a68b0
    #   (129, 3072, 3072, 32, False) on sm_100a -> 5852ad0db045e502e10a
    #   (129, 3072, 3072, 64, True) on sm_100a -> 05bda082a73c0c1a68b0
    #   (129, 3072, 3072, {96, 128}, True) on sm_103a -> 01405277c4bbc4031fbd (L2-promoted 16-stage ring)
    #   (129, 3072, 3072, 96, True) on sm_100a -> 05bda082a73c0c1a68b0
    #   (129, 3072, 3072, 128, False) on sm_103a -> 23bfc212699bbb4e6214 (L2-promoted 16-stage ring)
    #   (129, 3072, 3072, 128, False) on sm_100a -> 5852ad0db045e502e10a
    #   (129, 3072, 3072, 128, True) on sm_100a -> 2270204930b79b8caef7
    #   (129, 3072, 12288, 32, False) on sm_100a -> 243af01860dcca0f4e2c
    #   (129, 3072, 12288, 32, True) on sm_100a -> 36c42441c9d625e03c54
    #   (129, 3072, 12288, 32, False) on sm_103a -> 243af01860dcca0f4e2c
    #   (129, 3072, 12288, 32, True) on sm_103a -> 36c42441c9d625e03c54
    #   (129, 3072, 12288, 128, False) on sm_100a -> 243af01860dcca0f4e2c
    #   (129, 3072, 12288, 128, True) on sm_100a -> 36c42441c9d625e03c54
    #   (129, 3072, 12288, 128, False) on sm_103a -> 243af01860dcca0f4e2c
    #   (129, 3072, 12288, 128, True) on sm_103a -> 36c42441c9d625e03c54
    #   (4096, 3072, 3072, 32, True) on sm_100a -> 1d78f2bd5722fb02b75a
    #   (4096, 3072, 3072, 32, True) on sm_103a -> 5a1a17e73ef429cf8805 (K=96 MMA)
    #   (4096, 3072, 12288, 32, True) on sm_100a -> d758d9efa72998dbcf96 (wide tail;
    #       ``wide_tail``: on an SM count without the wide schedule, e.g. 148,
    #       the rows below route it)
    #   (4096, 3072, 12288, 32, True) on sm_103a -> 3d47f26879e192ef655a (K=96 MMA)
    #   (4096, 12288, 3072, 32, True) on sm_100a -> 2ca0cf40f7bbfca9df92
    #   (4096, 12288, 3072, 32, True) on sm_103a -> 151d20658188ea0e9651 (K=96 MMA)
    #   (6889, 3072, 3072, 32, True) on sm_100a -> fdcf777b3acd972f8841
    #   (6889, 3072, 3072, 32, True) on sm_103a -> 9fe64b6a30bb7316d03c (K=96 MMA)
    #   (6889, 3072, 12288, 32, True) on sm_103a -> 151d20658188ea0e9651 (K=96 MMA)
    #   (6889, 12288, 3072, 32, True) on sm_103a -> 9fe64b6a30bb7316d03c (K=96 MMA)
    #   (6889, 12288, 3072, 32, True) on sm_100a -> 2ca0cf40f7bbfca9df92
    #   (6912, 3072, 3072, 32, True) on sm_100a -> fdcf777b3acd972f8841
    #   (6912, 3072, 3072, 32, True) on sm_103a -> 9fe64b6a30bb7316d03c (K=96 MMA)
    #   (6912, 3072, 3072, 64, True) on sm_100a -> d16d801db12c7bdde4dc
    #   (6912, 3072, 3072, 64, True) on sm_103a -> 37c3da710589257a99f1 (K=96 MMA)
    #   (6912, 3072, 3072, 96, True) on sm_100a -> 0f7f27a483c337265bf9
    #   (6912, 3072, 3072, 96, True) on sm_103a -> ff27b5bafbc8d2d15d5b (K=96 MMA)
    #   (6912, 3072, 3072, 128, True) on sm_100a -> 4f451914c2d6580eb2de
    #   (6912, 3072, 3072, 128, True) on sm_103a -> 9d95357d28c063004b71 (K=96 MMA)
    #   (6912, 3072, 12288, 32, True) on sm_103a -> 151d20658188ea0e9651 (K=96 MMA)
    #   (6912, 3072, 12288, 128, True) on sm_103a -> 34ae602d0db3ffa24b4b (K=96 MMA)
    #   (9216, 3072, 3072, 32, True) on sm_100a -> fdcf777b3acd972f8841
    #   (9216, 3072, 3072, 32, True) on sm_103a -> a234cd99c8b3ef6af762 (K=96 MMA)
    #   (9216, 3072, 12288, 32, True) on sm_103a -> 3d47f26879e192ef655a (K=96 MMA)
    #   (9216, 12288, 3072, 32, True) on sm_103a -> 3d47f26879e192ef655a (K=96 MMA)
    #   (16384, 3072, 3072, 32, True) on sm_100a -> fdcf777b3acd972f8841
    #   (16384, 3072, 3072, 32, True) on sm_103a -> 9fe64b6a30bb7316d03c (K=96 MMA)
    #   (16384, 3072, 12288, 32, True) on sm_103a -> 60d29f6b0abb512843b5 (K=96 MMA)
    #   (16384, 12288, 3072, 32, True) on sm_100a -> c832b08a912e4c58112b
    #   (16384, 12288, 3072, 32, True) on sm_103a -> 3d47f26879e192ef655a (K=96 MMA)
    #   (129, 3072, 12288, {32, 128}, True) on both -> 8958df827006875b2512 (split-K, S = 4)
    #   (129, 3072, 12288, {32, 128}, False) on both -> 0a3a28f6f508e7d75146 (split-K, S = 4)
    {
        "arches": ("sm_103a",),
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32, 64),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_99bdf6e03d704178faf9",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": False,
        "record": "cake_nvfp4_svdquant_gemm_8319e1548886614ab273",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (96, 128),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_01405277c4bbc4031fbd",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (128,),
        "bias": False,
        "record": "cake_nvfp4_svdquant_gemm_23bfc212699bbb4e6214",
    },
    {
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32, 128),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_36c42441c9d625e03c54",
    },
    {
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32, 128),
        "bias": False,
        "record": "cake_nvfp4_svdquant_gemm_243af01860dcca0f4e2c",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32, 128),
        "bias": False,
        "record": "cake_nvfp4_svdquant_gemm_5852ad0db045e502e10a",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (128,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_2270204930b79b8caef7",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32, 64, 96),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_05bda082a73c0c1a68b0",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 4096,
        "m_max": 4096,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_1d78f2bd5722fb02b75a",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 4096,
        "m_max": 4096,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_5a1a17e73ef429cf8805",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 4096,
        "m_max": 4096,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 4096,
        "m_max": 4096,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32,),
        "bias": True,
        "wide_tail": True,
        "record": "cake_nvfp4_svdquant_gemm_d758d9efa72998dbcf96",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 4096,
        "m_max": 4096,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_2ca0cf40f7bbfca9df92",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 4096,
        "m_max": 4096,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_151d20658188ea0e9651",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6889,
        "m_max": 6889,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6889,
        "m_max": 6889,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6889,
        "m_max": 6889,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_151d20658188ea0e9651",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6889,
        "m_max": 6889,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6889,
        "m_max": 6889,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_2ca0cf40f7bbfca9df92",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (64,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_d16d801db12c7bdde4dc",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (64,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_37c3da710589257a99f1",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (96,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_0f7f27a483c337265bf9",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (96,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_ff27b5bafbc8d2d15d5b",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (128,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (128,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_9d95357d28c063004b71",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_151d20658188ea0e9651",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (128,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_34ae602d0db3ffa24b4b",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 9216,
        "m_max": 9216,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 9216,
        "m_max": 9216,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_a234cd99c8b3ef6af762",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 9216,
        "m_max": 9216,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 9216,
        "m_max": 9216,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_fdcf777b3acd972f8841",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_9fe64b6a30bb7316d03c",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_60d29f6b0abb512843b5",
    },
    {
        "arches": ("sm_100a",),
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_c832b08a912e4c58112b",
    },
    {
        "arches": ("sm_103a",),
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_3d47f26879e192ef655a",
    },
    # The wave-quantized rows where the tail split measured faster than the
    # whole-tile pair program: 324 pair tiles of (M 6889..6912,
    # N 3072, K 12288), 768 of (M 16384, N 3072, K 12288) and (M 4096, N 12288,
    # K 3072); the tail-split program of the rank (hinted store form: every
    # row here has ``K_tiles`` or ``w_tiles`` 48).
    {
        "m_min": 6889,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32, 64, 96, 128),
        "bias": None,
        "records_by_rank": {
            32: "cake_nvfp4_svdquant_gemm_e4f1cd01b65bed25c4f1",
            64: "cake_nvfp4_svdquant_gemm_0bd3a4870db1dc91bce4",
            96: "cake_nvfp4_svdquant_gemm_a867c135c6da108b5f82",
            128: "cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de",
        },
    },
    {
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32, 64, 96, 128),
        "bias": None,
        "records_by_rank": {
            32: "cake_nvfp4_svdquant_gemm_e4f1cd01b65bed25c4f1",
            64: "cake_nvfp4_svdquant_gemm_0bd3a4870db1dc91bce4",
            96: "cake_nvfp4_svdquant_gemm_a867c135c6da108b5f82",
            128: "cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de",
        },
    },
    {
        "m_min": 4096,
        "m_max": 4096,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32, 64, 96, 128),
        "bias": None,
        "records_by_rank": {
            32: "cake_nvfp4_svdquant_gemm_e4f1cd01b65bed25c4f1",
            64: "cake_nvfp4_svdquant_gemm_0bd3a4870db1dc91bce4",
            96: "cake_nvfp4_svdquant_gemm_a867c135c6da108b5f82",
            128: "cake_nvfp4_svdquant_gemm_4f451914c2d6580eb2de",
        },
    },
    # Large M on 256-deep K tiles: the two-CTA persistent program of the rank in
    # the store form of the launch -- the hinted form where the kernel hints its
    # output stores (``K_tiles >= 24`` or ``w_tiles >= 24`` in 256-wide tiles:
    # K >= 6144, or N >= 6016 for a multiple of 128), the plain form otherwise.
    {
        "m_min": 4096,
        "k_min": 6144,
        "k_multiple": 256,
        "ranks": (32, 64, 96, 128),
        "bias": None,
        "records_by_rank": {
            32: "cake_nvfp4_svdquant_gemm_3880e0aea2ecbe59106a",
            64: "cake_nvfp4_svdquant_gemm_99be5794f0107a539d27",
            96: "cake_nvfp4_svdquant_gemm_a474f70abaf1b70ca4fe",
            128: "cake_nvfp4_svdquant_gemm_9f7d814ace4cd94fc4d6",
        },
    },
    {
        "m_min": 4096,
        "n_min": 6016,
        "k_multiple": 256,
        "ranks": (32, 64, 96, 128),
        "bias": None,
        "records_by_rank": {
            32: "cake_nvfp4_svdquant_gemm_3880e0aea2ecbe59106a",
            64: "cake_nvfp4_svdquant_gemm_99be5794f0107a539d27",
            96: "cake_nvfp4_svdquant_gemm_a474f70abaf1b70ca4fe",
            128: "cake_nvfp4_svdquant_gemm_9f7d814ace4cd94fc4d6",
        },
    },
    {
        "m_min": 4096,
        "k_multiple": 256,
        "ranks": (32, 64, 96, 128),
        "bias": None,
        "records_by_rank": {
            32: "cake_nvfp4_svdquant_gemm_63c7c8683df10c75591b",
            64: "cake_nvfp4_svdquant_gemm_f57f1cbd289ee50c237e",
            96: "cake_nvfp4_svdquant_gemm_be138d4583b02c247633",
            128: "cake_nvfp4_svdquant_gemm_84b1b76b4ef917592df0",
        },
    },
    # The M 129 / N 3072 rows: the 64-row tactic 14 program, 144 CTAs in one
    # wave, in the form of the K class -- form B for K >= 12288 with K % 256 == 0,
    # form A for every other K (a multiple of 128); the biased build for biased
    # problems, the bias-free build otherwise (the sm_103a K 3072 rows at the
    # head take the L2-promoted builds instead).
    {
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_multiple": 256,
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_233bc54a42d3ffd5567d",
    },
    {
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 12288,
        "k_multiple": 256,
        "bias": False,
        "record": "cake_nvfp4_svdquant_gemm_d615c1bd45783aae8506",
    },
    {
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_e6789ee4eaed1dff09ae",
    },
    {
        "m_min": 129,
        "m_max": 129,
        "n_min": 3072,
        "n_max": 3072,
        "bias": False,
        "record": "cake_nvfp4_svdquant_gemm_8dc2b79ff158918dd5c8",
    },
    # Small biased problems: the biased 128x64 cluster program.
    {
        "m_max": 255,
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_a72d5469d8a64ca32645",
    },
    # The previous dispatcher's general rows.
    {
        "ranks": (96,),
        "record": "cake_nvfp4_svdquant_gemm_e15f358ed35018ca0015",
    },
    {
        "ranks": (64,),
        "bias": True,
        "record": "cake_nvfp4_svdquant_gemm_8463dfd07714ab7a0ca6",
    },
    {
        "k_min": 12288,
        "k_max": 12288,
        "ranks": (32,),
        "record": "cake_nvfp4_svdquant_gemm_751651fd3ba4ef54ce53",
    },
    # The fallback: tactic 0 with the exact epilogue.
    {
        "record": "cake_nvfp4_svdquant_gemm_db7820791092b67f10cd",
    },
)

# The raster of the two-CTA persistent programs (``pair_persistent``): the CLC
# scheduler hands out 256 x 256 pair tiles token-first; ``raster_group`` > 0
# walks groups of that many token tiles per weight tile so the group's A rows
# stay L2-resident across the N walk, 0 walks every token tile of a weight
# tile.  Ordered predicate rows over ``(M, N, K, rank)`` in the ``ROUTE_TABLE``
# vocabulary (``m_min`` / ``m_max``, ``n_min`` / ``n_max``, ``k_min`` /
# ``k_max``, ``k_multiple``, ``ranks``, and ``arches``: a row naming
# architectures applies on those only), first admitting row wins, the last
# row unconditional; ``group`` is the value passed.  Measured: the deep-K rows
# (K 12288) and the wide-N rows (N 12288 at K 3072) run fastest with groups of
# two token tiles; the other rows keep groups of eight.
# The rows at the head of the table measured fastest with groups of 4 for
# (16384, 3072, 3072, 32) on both architectures; groups of 1 for (6889, 12288,
# 3072, 32), (16384, 12288, 3072, 32) on sm_100a; groups of 4 for (6912, 3072,
# 3072, 128) on sm_100a.
RASTER_GROUP_TABLE: tuple[dict[str, Any], ...] = (
    {
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "group": 4,
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6912,
        "m_max": 6912,
        "n_min": 3072,
        "n_max": 3072,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (128,),
        "group": 4,
    },
    {
        "arches": ("sm_100a",),
        "m_min": 6889,
        "m_max": 6889,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "group": 1,
    },
    {
        "arches": ("sm_100a",),
        "m_min": 16384,
        "m_max": 16384,
        "n_min": 12288,
        "n_max": 12288,
        "k_min": 3072,
        "k_max": 3072,
        "ranks": (32,),
        "group": 1,
    },
    {"k_min": 12288, "group": 2},
    {"n_min": 12288, "k_max": 4096, "group": 2},
    {"group": 8},
)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def _admits(
    row: dict[str, Any],
    *,
    m: int,
    n: int,
    k: int,
    rank: int,
    has_bias: bool | None = None,
) -> bool:
    for key, value in (("m", m), ("n", n), ("k", k)):
        low, high = row.get(f"{key}_min"), row.get(f"{key}_max")
        if low is not None and value < int(low):
            return False
        if high is not None and value > int(high):
            return False
    multiple = row.get("k_multiple")
    if multiple is not None and k % int(multiple):
        return False
    ranks = row.get("ranks")
    if ranks is not None and rank not in tuple(int(r) for r in ranks):
        return False
    bias = row.get("bias")
    if bias is not None and has_bias is not None and bool(has_bias) is not bool(bias):
        return False
    return True


def raster_group(*, m: int, n: int, k: int, rank: int, arch: str | None = None) -> int:
    """The ``raster_group`` scalar of the two-CTA persistent programs for one
    problem on ``arch``: the ``group`` of the first admitting
    :data:`RASTER_GROUP_TABLE` row.  A row that names ``arches`` applies on
    those architectures only; without ``arch`` such a row that admits the
    problem is a ``TypeError``, as in :func:`select_record`."""
    for row in RASTER_GROUP_TABLE:
        if not _admits(row, m=int(m), n=int(n), k=int(k), rank=int(rank)):
            continue
        arches = row.get("arches")
        if arches is not None:
            if arch is None:
                raise TypeError(
                    "the Cake NVFP4 SVDQuant raster-group table sets the group of "
                    f"(M={m}, N={n}, K={k}, rank={rank}) per architecture; "
                    "raster_group() needs arch=device_arch(index)"
                )
            if arch not in tuple(arches):
                continue
        return max(0, int(row["group"]))
    raise RuntimeError("the Cake NVFP4 SVDQuant raster-group table has no fallback row")


def _route(
    *,
    m: int,
    n: int,
    k: int,
    rank: int,
    has_bias: bool,
    arch: str | None = None,
    sm_count: int | None = None,
) -> str:
    for row in ROUTE_TABLE:
        if not _admits(row, m=m, n=n, k=k, rank=rank, has_bias=has_bias):
            continue
        arches = row.get("arches")
        if arches is not None:
            if arch is None:
                raise TypeError(
                    "the Cake NVFP4 SVDQuant route table routes "
                    f"(M={m}, N={n}, K={k}, rank={rank}, bias={has_bias}) per "
                    "architecture; select_record() needs arch=device_arch(index)"
                )
            if arch not in tuple(arches):
                continue
        if row.get("wide_tail"):
            if sm_count is None:
                raise TypeError(
                    "the Cake NVFP4 SVDQuant route table routes "
                    f"(M={m}, N={n}, K={k}, rank={rank}, bias={has_bias}) by the "
                    "device's SM count; select_record() needs sm_count=sm_count(index)"
                )
            group = raster_group(m=m, n=n, k=k, rank=rank, arch=arch)
            _, _, _, split = tailsplit_wide_units(
                m=m, n=n, sm_count=sm_count, raster_group=group
            )
            if split != WIDE_TAIL_SPLIT:
                continue
        by_rank = row.get("records_by_rank")
        if by_rank is not None:
            name = by_rank.get(rank)
            if name is None:
                continue
            return str(name)
        return str(row["record"])
    raise RuntimeError("the Cake NVFP4 SVDQuant route table has no fallback row")


def select_record(
    *,
    m: int,
    n: int,
    k: int,
    rank: int,
    has_bias: bool,
    arch: str | None = None,
    sm_count: int | None = None,
) -> str:
    """The record name of the program that serves ``(M, N, K, rank, has_bias)``
    on ``arch`` with ``sm_count`` SMs, or ``ValueError`` naming the violated
    shape rule.

    Routing is the first admitting row of :data:`ROUTE_TABLE` on ``arch`` (the
    device's exported architecture, :func:`device_arch`; a row that names
    ``arches`` applies on those architectures only) and ``sm_count`` (the
    device's SM count, :func:`sm_count`; a ``wide_tail`` row applies only where
    :func:`tailsplit_wide_units` on that SM count at the problem's
    :func:`raster_group` is the wide schedule); the shape rules are
    checked afterwards against the selected program (``M >= 128``, ``N`` a
    positive multiple of 128, ``K`` a positive multiple of the program's K
    tile, ``rank`` a multiple of 32 in ``[32, 128]``).  A problem with a bias
    is never routed to a bias-free program.  Without ``arch`` only the
    architecture-blind rows resolve: a per-architecture row that admits the
    problem is a ``TypeError``; without ``sm_count`` a ``wide_tail`` row that
    admits the problem on ``arch`` is a ``TypeError`` too, so no caller reads
    a route the device would not take.
    """
    m, n, k, rank, has_bias = int(m), int(n), int(k), int(rank), bool(has_bias)
    if arch is not None and arch not in ARCHES:
        raise RuntimeError(
            f"{arch!r} is not an exported Cake NVFP4 SVDQuant GEMM target; the "
            f"exported architectures are {ARCHES}"
        )
    name = _route(
        m=m,
        n=n,
        k=k,
        rank=rank,
        has_bias=has_bias,
        arch=arch,
        sm_count=None if sm_count is None else int(sm_count),
    )
    record = MODULES[name]
    if arch is not None and arch not in tuple(record["arches"]):
        raise RuntimeError(
            f"the Cake NVFP4 SVDQuant route table sends a problem on {arch} to "
            f"program {name}, which ships for {tuple(record['arches'])}"
        )
    plan = record["plan"]
    if m < BLOCK_M:
        raise ValueError(
            "Cake NVFP4 SVDQuant requires M >= 128 (every tensor map carries a "
            f"128-row box), got M={m}"
        )
    if n < BLOCK_M or n % BLOCK_M:
        raise ValueError(
            f"Cake NVFP4 SVDQuant requires N to be a positive multiple of {BLOCK_M}, got N={n}"
        )
    block_k = int(plan["block_k"])
    if k < block_k or k % block_k:
        raise ValueError(
            f"Cake NVFP4 SVDQuant requires K to be a positive multiple of {block_k} "
            f"for rank={rank}{' with bias' if has_bias else ''}, got K={k}"
        )
    if rank < RANK_GRANULARITY or rank > MAX_RANK or rank % RANK_GRANULARITY:
        raise ValueError(
            "Cake NVFP4 SVDQuant requires a LoRA rank that is a multiple of "
            f"{RANK_GRANULARITY} between {RANK_GRANULARITY} and {MAX_RANK}, got rank={rank}"
        )
    if has_bias and plan["variant"] != "bias":
        raise RuntimeError(
            f"the Cake NVFP4 SVDQuant route table sends a biased problem to the "
            f"bias-free program {name}"
        )
    storage = plan.get("rank_storage")
    if storage is not None and int(storage) != rank:
        raise RuntimeError(
            f"the Cake NVFP4 SVDQuant route table sends rank {rank} to program {name}, "
            f"built for rank {storage}"
        )
    return name


# The tail-split pair program (``pair_tailsplit``): the unit schedule of its
# launch.  ``PAIR_M`` x ``PAIR_N`` pair tiles over ``pairs = sm_count // 2`` CTA
# pairs; the ``tail_tiles`` of the last partial wave run as two half units each
# when they fit in half the pairs (``2 * tail_tiles <= pairs``), else every
# unit is a whole tile (the ``pair_persistent`` schedule).
PAIR_M = 256
PAIR_N = 256
TAIL_SPLITS = (1, 2)


def tailsplit_units(
    *, m: int, n: int, sm_count: int, tail_split: int | None = None
) -> tuple[int, int, int, int]:
    """``(num_tiles, num_units, full_units, tail_split)`` of the tail-split pair
    program for an ``(M, N)`` problem on ``sm_count`` SMs.

    ``num_tiles = ceil(M / 256) * ceil(N / 256)``; with ``pairs = sm_count //
    2``, ``full_waves = num_tiles // pairs`` and ``tail_tiles = num_tiles -
    full_waves * pairs``: when ``tail_tiles > 0`` and ``2 * tail_tiles <=
    pairs`` the tail tiles become ``2 * tail_tiles`` half units (``tail_split
    = 2``, ``full_units = full_waves * pairs``, ``num_units = full_units + 2 *
    tail_tiles``), otherwise every unit is a whole tile (``tail_split = 1``,
    ``num_units = full_units = num_tiles``).  ``tail_split`` given forces that
    schedule.  Units ``[0, full_units)`` are whole tiles in raster order, unit
    ``u >= full_units`` is N half ``(u - full_units) % 2`` of tail tile
    ``full_units + (u - full_units) // 2``.
    """
    m, n, sm_count = int(m), int(n), int(sm_count)
    num_tiles = _ceil_div(m, PAIR_M) * _ceil_div(n, PAIR_N)
    pairs = max(1, sm_count // 2)
    full_waves = num_tiles // pairs
    tail_tiles = num_tiles - full_waves * pairs
    if tail_split is None:
        tail_split = 2 if (tail_tiles > 0 and 2 * tail_tiles <= pairs) else 1
    if tail_split not in TAIL_SPLITS:
        raise ValueError(f"tail_split must be one of {TAIL_SPLITS}, got {tail_split!r}")
    if tail_split == 1 or tail_tiles == 0:
        return num_tiles, num_tiles, num_tiles, 1
    full_units = full_waves * pairs
    return num_tiles, full_units + 2 * tail_tiles, full_units, 2


# The wide-tail build of the tail-split pair program (``pair_tailsplit_wide``):
# a last partial wave of more than half and at most three quarters of the CTA
# pairs runs as one wave of ``pairs`` units under ``tail_split =
# WIDE_TAIL_SPLIT``: groups of ``WIDE_GROUP_TILES`` column-adjacent tail tiles
# of one token tile as four 192-wide units each, the other tail tiles as half
# units.  ``WIDE_TAIL_SPLIT`` (256 + 192) is outside :data:`TAIL_SPLITS`, so no
# other program ever receives it.
WIDE_TAIL_SPLIT = 448
WIDE_GROUP_TILES = 3


def tailsplit_wide_units(
    *,
    m: int,
    n: int,
    sm_count: int,
    raster_group: int,
    tail_split: int | None = None,
) -> tuple[int, int, int, int]:
    """``(num_tiles, num_units, full_units, tail_split)`` of the wide-tail
    build of the tail-split pair program for an ``(M, N)`` problem on
    ``sm_count`` SMs at the problem's ``raster_group``.

    The :func:`tailsplit_units` schedule, except that with ``pairs = sm_count
    // 2`` even, ``tail_tiles`` the tiles of the last partial wave,
    ``pairs < 2 * tail_tiles <= 3 * pairs / 2`` and ``N`` a multiple of 128,
    the tail runs as one wave of ``pairs`` units (``num_units = full_units +
    pairs``, ``tail_split = WIDE_TAIL_SPLIT``): ``groups = tail_tiles - pairs /
    2`` groups of three column-adjacent tail tiles as four 192-wide units each,
    the other ``tail_tiles - 3 * groups`` tail tiles as half units.  The groups
    fill blocks of three tile columns by the token tiles of one raster group,
    from the first column the tail starts or reaches whole in its raster group
    (else column 0 of the next raster group); when those blocks do not fit in
    the ``N // 256`` full 256-column tiles there, the tail keeps the
    :func:`tailsplit_units` schedule.  ``tail_split`` given forces a schedule:
    1 or 2 that of :func:`tailsplit_units`, ``WIDE_TAIL_SPLIT`` the wide one
    (``ValueError`` where it does not exist).
    """
    m, n, sm_count = int(m), int(n), int(sm_count)
    if tail_split in TAIL_SPLITS:
        return tailsplit_units(m=m, n=n, sm_count=sm_count, tail_split=tail_split)
    if tail_split not in (None, WIDE_TAIL_SPLIT):
        raise ValueError(
            f"tail_split must be one of {TAIL_SPLITS + (WIDE_TAIL_SPLIT,)}, "
            f"got {tail_split!r}"
        )
    tok_tiles = _ceil_div(m, PAIR_M)
    w_tiles = _ceil_div(n, PAIR_N)
    num_tiles = tok_tiles * w_tiles
    pairs = max(1, sm_count // 2)
    full_units = num_tiles // pairs * pairs
    tail_tiles = num_tiles - full_units
    wide = (
        pairs % 2 == 0
        and pairs < 2 * tail_tiles
        and 4 * tail_tiles <= 3 * pairs
        and n % 128 == 0
    )
    if wide:
        groups = tail_tiles - pairs // 2
        group_rows = int(raster_group) or tok_tiles
        size = group_rows * w_tiles
        first = full_units // size
        rows = min(group_rows, tok_tiles - first * group_rows)
        column, partial = divmod(full_units - first * size, rows)
        if partial != 0:
            if column + 1 < w_tiles:
                column += 1
            else:
                column = 0
                rows = min(group_rows, tok_tiles - (first + 1) * group_rows)
        blocks = _ceil_div(groups, rows) if rows >= 1 else 0
        wide = rows >= 1 and column + WIDE_GROUP_TILES * blocks <= n // PAIR_N
    if wide:
        return num_tiles, full_units + pairs, full_units, WIDE_TAIL_SPLIT
    if tail_split == WIDE_TAIL_SPLIT:
        raise ValueError(
            f"no wide tail schedule for (M={m}, N={n}) at raster group "
            f"{raster_group} on {sm_count} SMs"
        )
    return tailsplit_units(m=m, n=n, sm_count=sm_count)


def launch_grid(
    record: dict[str, Any],
    *,
    m: int,
    n: int,
    k: int,
    sm_count: int | None = None,
    raster_group: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one record for an ``(M, N, K)`` problem under its grid rule.

    With ``grid_m = ceil(M / block_m)`` (128-row tensor-map boxes, or the 64-row
    tiles of the ``block_m = 64`` programs) and ``grid_n = N / block_n``:
    ``linear`` launches ``(grid_m * grid_n, 1, 1)``; ``cluster_n2`` ``(grid_m,
    grid_n, 1)`` (clusters of two N-adjacent CTAs, ``grid_n`` even because
    ``N`` is a multiple of 128); ``pair_persistent`` ``(2 * tok_tiles * w_tiles, 1, 1)`` with
    ``tok_tiles = ceil(M / block_m)`` and ``w_tiles = ceil(N / block_n)``, two
    CTAs per scheduled tile; ``pair_tailsplit`` ``(2 * num_units, 1, 1)`` with
    the ``num_units`` of :func:`tailsplit_units` (``sm_count`` required);
    ``pair_tailsplit_wide`` the same with the ``num_units`` of
    :func:`tailsplit_wide_units` (``sm_count`` and the problem's
    ``raster_group`` required); ``splitk_cluster`` ``(split_k * grid_n, 1, 1)``: one cluster of the plan's
    ``split_k`` CTAs per 128-column weight tile, every CTA of a cluster walking
    ``K / split_k`` (the route rows pin ``K``, so ``K`` enters no grid
    arithmetic).
    """
    plan = record["plan"]
    rule = plan["grid_rule"]
    m, n = int(m), int(n)
    grid_m = _ceil_div(m, int(plan["block_m"]))
    grid_n = n // int(plan["block_n"])
    if rule == "linear":
        return (grid_m * grid_n, 1, 1)
    if rule == "cluster_n2":
        return (grid_m, grid_n, 1)
    if rule == "pair_persistent":
        tok_tiles = _ceil_div(m, int(plan["block_m"]))
        w_tiles = _ceil_div(n, int(plan["block_n"]))
        return (2 * tok_tiles * w_tiles, 1, 1)
    if rule == "pair_tailsplit":
        if sm_count is None:
            raise ValueError(
                "the pair_tailsplit grid rule needs the device's SM count (sm_count)"
            )
        _, num_units, _, _ = tailsplit_units(m=m, n=n, sm_count=sm_count)
        return (2 * num_units, 1, 1)
    if rule == "pair_tailsplit_wide":
        if sm_count is None or raster_group is None:
            raise ValueError(
                "the pair_tailsplit_wide grid rule needs the device's SM count "
                "(sm_count) and the problem's raster group (raster_group)"
            )
        _, num_units, _, _ = tailsplit_wide_units(
            m=m, n=n, sm_count=sm_count, raster_group=raster_group
        )
        return (2 * num_units, 1, 1)
    if rule == "splitk_cluster":
        return (int(plan["split_k"]) * grid_n, 1, 1)
    raise ValueError(f"unknown Cake NVFP4 SVDQuant grid rule: {rule!r}")


def plan_values(
    record: dict[str, Any],
    *,
    a: Any,
    b: Any,
    a_sf: Any,
    b_sf: Any,
    alpha: Any,
    d: Any,
    l1: Any,
    bias: Any,
    out: Any,
) -> dict[str, Any]:
    """The logical argument values of one record for the caller's tensors, plus
    the launch ``grid``.

    ``a [M, K/2]`` / ``b [N, K/2]`` uint8, ``a_sf`` / ``b_sf`` the swizzled
    scale factors (a larger contiguous backing buffer is permitted: only the
    leading ``padded_rows * K / 16`` bytes are viewed, rows padded to 128),
    ``alpha`` FP32 (its first element is used), ``d [M, rank]`` / ``l1 [N,
    rank]`` / ``out [M, N]`` BF16 and ``bias [N]`` BF16 or ``None``.  The scale
    factors are viewed per the record's ``scale_elems``: ``uint8`` as
    ``[rows / 128, K / 64, 4, 8, 16]``, ``uint16`` as ``[rows / 128, K / 64, 4,
    64]``, ``uint32`` as ``[rows / 128, K / 64, 128]``; every view keeps the
    caller's storage.  The scalars are ``M``, ``N``, ``K_tiles = K / block_k``,
    ``rank_tiles = rank / 32``, ``rank``, ``rank_chunks = ceil(rank / 64)``,
    ``grid_n = N / block_n``, ``has_bias``, and for the persistent pair programs ``tok_tiles = ceil(M /
    block_m)``, ``w_tiles = ceil(N / block_n)``, ``num_tiles`` and the
    :func:`raster_group` of the problem on the device's architecture, and for
    the tail-split pair program
    also ``num_units`` / ``full_units`` / ``tail_split`` of
    :func:`tailsplit_units` on the device's SM count (of
    :func:`tailsplit_wide_units` at that raster group for its wide-tail
    build).  ``bias`` is ``out``
    when the problem has no bias (a valid carrier the kernel never reads).
    """
    import torch

    plan = record["plan"]
    m, k_packed = (int(value) for value in a.shape)
    n = int(b.shape[0])
    k = 2 * k_packed
    rank = int(d.shape[1])
    has_bias = bias is not None
    block_m = int(plan["block_m"])
    block_n = int(plan["block_n"])
    block_k = int(plan["block_k"])

    padded_m = _ceil_div(m, BLOCK_M) * BLOCK_M
    padded_n = _ceil_div(n, BLOCK_M) * BLOCK_M
    sf_cols = k // SF_BLOCK
    sf_quads = sf_cols // 4
    a_sf_flat = a_sf.reshape(-1)[: padded_m * sf_cols]
    b_sf_flat = b_sf.reshape(-1)[: padded_n * sf_cols]
    scale_elems = plan["scale_elems"]
    if scale_elems == "uint8":
        a_sf_view = a_sf_flat.view(padded_m // BLOCK_M, sf_quads, 4, 8, 16)
        b_sf_view = b_sf_flat.view(padded_n // BLOCK_M, sf_quads, 4, 8, 16)
    elif scale_elems == "uint16":
        a_sf_view = a_sf_flat.view(torch.uint16).view(
            padded_m // BLOCK_M, sf_quads, 4, 64
        )
        b_sf_view = b_sf_flat.view(torch.uint16).view(
            padded_n // BLOCK_M, sf_quads, 4, 64
        )
    elif scale_elems == "uint32":
        a_sf_view = a_sf_flat.view(torch.uint32).view(
            padded_m // BLOCK_M, sf_quads, 128
        )
        b_sf_view = b_sf_flat.view(torch.uint32).view(
            padded_n // BLOCK_M, sf_quads, 128
        )
    else:
        raise ValueError(
            f"unknown Cake NVFP4 SVDQuant scale element kind: {scale_elems!r}"
        )

    grid_n = n // block_n
    tok_tiles = _ceil_div(m, block_m)
    w_tiles = _ceil_div(n, block_n)
    rule = plan["grid_rule"]
    tailsplit = rule in ("pair_tailsplit", "pair_tailsplit_wide")
    device_sms = sm_count(a.device.index) if tailsplit else None
    pair = tailsplit or rule == "pair_persistent"
    group = (
        raster_group(m=m, n=n, k=k, rank=rank, arch=device_arch(a.device.index))
        if pair
        else None
    )
    values: dict[str, Any] = {
        "a": a,
        "b": b,
        "a_sf": a_sf_view,
        "b_sf": b_sf_view,
        "d": d,
        "l1": l1,
        "alpha": alpha.reshape(-1)[:1],
        # The bias parameter needs a valid carrier; has_bias=0 keeps it unread.
        "bias": bias if has_bias else out,
        "out": out,
        "M": m,
        "N": n,
        "K_tiles": k // block_k,
        "rank_tiles": rank // RANK_GRANULARITY,
        "rank": rank,
        "rank_chunks": _ceil_div(rank, 64),
        "grid_n": grid_n,
        "has_bias": 1 if has_bias else 0,
        "tok_tiles": tok_tiles,
        "w_tiles": w_tiles,
        "num_tiles": tok_tiles * w_tiles,
        "grid": launch_grid(
            record, m=m, n=n, k=k, sm_count=device_sms, raster_group=group
        ),
    }
    if pair:
        values["raster_group"] = group
    if tailsplit:
        if rule == "pair_tailsplit_wide":
            _, num_units, full_units, tail_split = tailsplit_wide_units(
                m=m, n=n, sm_count=device_sms, raster_group=group
            )
        else:
            _, num_units, full_units, tail_split = tailsplit_units(
                m=m, n=n, sm_count=device_sms
            )
        values["num_units"] = num_units
        values["full_units"] = full_units
        values["tail_split"] = tail_split
    return values


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
            f"Cake NVFP4 SVDQuant GEMM target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake NVFP4 SVDQuant GEMM record {name} is not delivered for {arch}; "
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


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter and
    ``(None, axis)`` per grid argument, in ``run`` order; computed once."""
    record = MODULES[name]
    logical = record["logical"]
    plan: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append((None, "xyz".index(key[-1])))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], -1))
    return tuple(plan)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to
    tensors and Python scalars (:func:`plan_values` builds them); the record
    maps its kernel parameters onto that vocabulary and ``grid`` is the plan's
    launch grid (:func:`launch_grid`).  The kernel runs on the current torch
    stream of the device.
    """
    record = MODULES[name]
    args = []
    for logical_name, axis in _argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BLOCK_M",
    "GRID_RULES",
    "INT32_MAX",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_RANK",
    "MODULES",
    "PARAMETER_LOGICAL",
    "RANK_GRANULARITY",
    "RASTER_GROUP_TABLE",
    "ROUTE_TABLE",
    "SCALE_ELEMS",
    "SF_BLOCK",
    "VARIANTS",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "plan_values",
    "raster_group",
    "run_module",
    "select_record",
    "sm_count",
    "specs_for_arch",
    "tailsplit_units",
    "tailsplit_wide_units",
]
