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

Host plan and loader glue of the Cake single-GPU norm kernels of the Muse
Spark family: the gated residual with the next RMSNorm fused
(``flashinfer.norm.gated_residual_next_rmsnorm``) and the gated residual
RMSNorm (``flashinfer.norm.gated_residual_rmsnorm``), hidden size 8192, BF16
branch and FP32 residual stream, and the per-head RMSNorm times the sigmoid
gate (``flashinfer.norm.rmsnorm_sigmoid_gate``, head dimension 64, in place);
SM100 / SM103.

Every record of ``MODULES`` is one mechanically exported production build: a
CUDA device translation unit, a tvm-ffi binding and an occupancy query
compiled together for each architecture the record names.  The record carries
the host plan only: the ``library`` stem, the architectures it is delivered
for, the ``build`` keywords it was rendered with, its launch record, its grid
rule, its selection rule and the kernel-parameter -> logical-argument map.
What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route is the list of records one public call may launch:
``<op>_<variant>`` per architecture (``ROUTES``), the variant naming which of
the call's optional inputs the programs take (``gated_residual_next_rmsnorm``:
``alpha`` a tensor or a scalar, ``post_weight`` present or not) or the mode and
the PDL form of the call (``gated_residual_rmsnorm``: ``mode0`` / ``mode1``,
with ``_pdl`` for ``enable_pdl``).  Each record admits a window of row counts on
each of its architectures (``select_rule``: ``above < rows <= at_most``, the
last window unbounded); the windows of a route partition every positive row
count, so ``selected_module`` always answers.  The grid follows the record's
``grid_rule``: ``row_ctas`` CTAs per row (the cluster when the rule names none;
``gated_residual_rmsnorm`` groups the ``split`` CTAs of a row in placement
clusters of ``cluster``), or a persistent cluster loop of ``min(rows,
co-resident clusters)`` clusters, the co-residency read once per (record,
architecture, device) from the record's occupancy query (``occupancy_entry``).  No program owns a workspace (``workspace_rule`` is
``None`` on every record).  ``rmsnorm_sigmoid_gate`` routes on the head count and
the token count together (``select_rule`` ``tokens_heads_between``: ``[heads,
above, at_most]`` windows, a partition of the token counts per head count) and
launches the kernel's two-dimensional grid (``grid_rule`` ``tokens_heads``:
``rows_per_cta`` rows of the (tokens x heads) row space per CTA, the cluster
over one token's head blocks, the token-axis cluster, the CTA order and the
token interleave; :func:`tokens_heads_grid`).  Batch size is a runtime scalar,
so a changed row count never recompiles anything.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) and the ``ROUTES`` table are
# rewritten by the exporter from the rendered programs; every other field is
# host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_muse_spark_norm_004843ed6a555e0904da": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_004843ed6a555e0904da",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_004843ed6a555e0904da",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        32,
                        1024,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        32,
                        8192,
                        None,
                    ],
                ],
                "sm_107a": [
                    [
                        32,
                        257,
                        767,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_0782a35e531120ee0e27": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_0782a35e531120ee0e27",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_0782a35e531120ee0e27",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_103a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_107a": [
                    [
                        64,
                        128,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_102d29ef70510879670f": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 256,
            "split": 8,
            "exchange": False,
            "cluster": 2,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_102d29ef70510879670f",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                2,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 8576,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_102d29ef70510879670f",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_103a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_107a": [
                    [
                        1,
                        16,
                    ],
                    [
                        32,
                        64,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_1282e67c9f3fa0934fd6": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1282e67c9f3fa0934fd6",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_1282e67c9f3fa0934fd6",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        383,
                    ],
                    [
                        2048,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        224,
                        256,
                    ],
                    [
                        2048,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_13e0bf051991559b8b48": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "el",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_13e0bf051991559b8b48",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_13e0bf051991559b8b48",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        192,
                        256,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        255,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_1a810e28e0c14fe4e541": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "beta",
            "planes_path": "sts",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1a810e28e0c14fe4e541",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33152,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_1a810e28e0c14fe4e541",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        383,
                        511,
                    ],
                    [
                        1024,
                        2048,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_2413d64185f668557ed6": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 2,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "clamp",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 64,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_2413d64185f668557ed6",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_2413d64185f668557ed6",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        64,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_25235fb509f8cb1da0cf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_25235fb509f8cb1da0cf",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_25235fb509f8cb1da0cf",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "routes": {
                "rmsnorm_sigmoid_gate_bf16_pdl": {
                    "sm_103a": [
                        [
                            32,
                            1024,
                            8192,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_100a": [
                    [
                        32,
                        1024,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        32,
                        1024,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_2ea93a6e5dcb5daac83d": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 64,
            "lanes": 8,
            "rows_per_thread": 4,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": True,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_2ea93a6e5dcb5daac83d",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_2ea93a6e5dcb5daac83d",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        511,
                        512,
                    ],
                    [
                        32,
                        2047,
                        2048,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_2eb8f754997071825d1d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 4,
            "exchange": False,
            "cluster": 4,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_2eb8f754997071825d1d",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 16768,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_2eb8f754997071825d1d",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_107a": [
                    [
                        0,
                        1,
                    ],
                    [
                        16,
                        32,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_35e16e98dde0f51c82e1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_35e16e98dde0f51c82e1",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_35e16e98dde0f51c82e1",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        256,
                        512,
                    ],
                    [
                        768,
                        1024,
                    ],
                    [
                        8192,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        256,
                        512,
                    ],
                    [
                        768,
                        1024,
                    ],
                    [
                        2048,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_37adff50d4fef57403a1": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_37adff50d4fef57403a1",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_37adff50d4fef57403a1",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_103a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_107a": [
                    [
                        64,
                        128,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_38daeaf6d495ec29b417": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 1024,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": True,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": False,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_38daeaf6d495ec29b417",
        "launch": {
            "block": (
                1024,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_38daeaf6d495ec29b417",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_107a": [
                    [
                        256,
                        4096,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_3e42b08f1d3e52bc3829": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 32,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 16,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 8,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_3e42b08f1d3e52bc3829",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_3e42b08f1d3e52bc3829",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        1,
                        8,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        1,
                        8,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        1,
                        2,
                    ],
                    [
                        128,
                        2,
                        8,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_424f59177d4ed1568e7e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "beta",
            "planes_path": "sts",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_424f59177d4ed1568e7e",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33152,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_424f59177d4ed1568e7e",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        192,
                    ],
                ],
                "sm_103a": [
                    [
                        255,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_45fcc7c9bbfd36dabe9f": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 1024,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": True,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": False,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 3,
            "fill_first": False,
            "ring_content": "res",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_45fcc7c9bbfd36dabe9f",
        "launch": {
            "block": (
                1024,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 99712,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_45fcc7c9bbfd36dabe9f",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0",
        "select_rule": {
            "kind": "rows_between",
            "routes": {
                "gated_residual_rmsnorm_mode0_pdl": {
                    "sm_107a": [
                        [
                            4096,
                            None,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_107a": [
                    [
                        4096,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_519a7a9fdc31418e30cf": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "heads",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "heads",
            "rows_per_cta": 32,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_519a7a9fdc31418e30cf",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                16,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_519a7a9fdc31418e30cf",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        8,
                        64,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_55308ae83554784c040f": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 16,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_55308ae83554784c040f",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_55308ae83554784c040f",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        1,
                        8,
                    ],
                    [
                        16,
                        256,
                        1024,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        1,
                        8,
                    ],
                    [
                        16,
                        64,
                        1024,
                    ],
                ],
                "sm_107a": [
                    [
                        16,
                        1,
                        1024,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_57b7b5ad70819fd42c36": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_57b7b5ad70819fd42c36",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_57b7b5ad70819fd42c36",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        8,
                        256,
                    ],
                    [
                        32,
                        0,
                        1024,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        8,
                        256,
                    ],
                    [
                        32,
                        0,
                        1024,
                    ],
                ],
                "sm_107a": [
                    [
                        32,
                        0,
                        9,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_5f812bc0f781c0f98731": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": True,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_5f812bc0f781c0f98731",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_5f812bc0f781c0f98731",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        512,
                        768,
                    ],
                    [
                        1024,
                        8192,
                    ],
                ],
                "sm_103a": [
                    [
                        512,
                        768,
                    ],
                    [
                        1024,
                        2048,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_62b6802c3f845a5a02cb": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 2,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_62b6802c3f845a5a02cb",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_62b6802c3f845a5a02cb",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        32,
                        767,
                        2047,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_754e16217f4dd91d0be2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 4,
            "exchange": False,
            "cluster": 4,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "asm",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_754e16217f4dd91d0be2",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 16768,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_754e16217f4dd91d0be2",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_107a": [
                    [
                        0,
                        1,
                    ],
                    [
                        16,
                        32,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_77358559505c434c4bb7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 16,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_77358559505c434c4bb7",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_77358559505c434c4bb7",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        8,
                        24,
                    ],
                    [
                        128,
                        39,
                        40,
                    ],
                    [
                        128,
                        47,
                        64,
                    ],
                    [
                        32,
                        9,
                        13,
                    ],
                    [
                        32,
                        14,
                        24,
                    ],
                    [
                        32,
                        63,
                        64,
                    ],
                    [
                        32,
                        64,
                        65,
                    ],
                    [
                        32,
                        69,
                        70,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_778060a26127fd4ec76a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": True,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_778060a26127fd4ec76a",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_778060a26127fd4ec76a",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        4095,
                        8192,
                    ],
                ],
                "sm_103a": [
                    [
                        256,
                        384,
                    ],
                    [
                        512,
                        768,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_7b370733525fe3e25f75": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_7b370733525fe3e25f75",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_7b370733525fe3e25f75",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        256,
                        768,
                    ],
                    [
                        8192,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        384,
                        512,
                    ],
                    [
                        768,
                        1024,
                    ],
                    [
                        2048,
                        4096,
                    ],
                    [
                        8191,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_7b9c67698ca00f540a99": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 4,
            "rows_per_thread": 2,
            "width": 256,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 4,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 64,
            "tcluster": 1,
            "tpc": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_7b9c67698ca00f540a99",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_7b9c67698ca00f540a99",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "routes": {
                "rmsnorm_sigmoid_gate_bf16_pdl": {
                    "sm_107a": [
                        [
                            16,
                            12287,
                            32767,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_107a": [
                    [
                        16,
                        8191,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_7bfe75ee1890611cca58": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "heads",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "heads",
            "rows_per_cta": 32,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_7bfe75ee1890611cca58",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_7bfe75ee1890611cca58",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        256,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        2048,
                        None,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        256,
                        383,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_8418dc5f50dc570d66db": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 256,
            "split": 8,
            "exchange": False,
            "cluster": 2,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_8418dc5f50dc570d66db",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                2,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 8576,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_8418dc5f50dc570d66db",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_103a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_107a": [
                    [
                        1,
                        16,
                    ],
                    [
                        32,
                        64,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_87fc822d80b99514648a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": True,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_87fc822d80b99514648a",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_87fc822d80b99514648a",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        8,
                        16,
                    ],
                    [
                        128,
                        64,
                        256,
                    ],
                    [
                        32,
                        1,
                        64,
                    ],
                    [
                        32,
                        128,
                        192,
                    ],
                    [
                        32,
                        257,
                        1024,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        64,
                        256,
                    ],
                    [
                        32,
                        8,
                        64,
                    ],
                    [
                        32,
                        128,
                        192,
                    ],
                    [
                        32,
                        256,
                        1024,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_8d16ecfd030500bcd131": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 1024,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": True,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": False,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_8d16ecfd030500bcd131",
        "launch": {
            "block": (
                1024,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_8d16ecfd030500bcd131",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_107a": [
                    [
                        256,
                        4096,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_8e869908b1526226830a": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 1024,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": True,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": False,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_8e869908b1526226830a",
        "launch": {
            "block": (
                1024,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_8e869908b1526226830a",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_107a": [
                    [
                        256,
                        4096,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_8f55d029d0e862cc1e0b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 16,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_8f55d029d0e862cc1e0b",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_8f55d029d0e862cc1e0b",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "routes": {
                "rmsnorm_sigmoid_gate_bf16_pdl": {
                    "sm_100a": [
                        [
                            16,
                            1024,
                            None,
                        ],
                    ],
                    "sm_103a": [
                        [
                            16,
                            1024,
                            None,
                        ],
                    ],
                    "sm_107a": [
                        [
                            16,
                            1024,
                            2047,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_100a": [
                    [
                        16,
                        1024,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        1024,
                        None,
                    ],
                ],
                "sm_107a": [
                    [
                        16,
                        1024,
                        2047,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_90b7ec1d7920cc5c2545": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "beta",
            "planes_path": "sts_late",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_90b7ec1d7920cc5c2545",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33152,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_90b7ec1d7920cc5c2545",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        511,
                        1024,
                    ],
                    [
                        2048,
                        4096,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        2048,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_94ddf891ecc46283d87c": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 4,
            "rows_per_thread": 2,
            "width": 256,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 4,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 64,
            "tcluster": 1,
            "tpc": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_94ddf891ecc46283d87c",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_94ddf891ecc46283d87c",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        16,
                        6143,
                        8191,
                    ],
                    [
                        16,
                        8191,
                        12287,
                    ],
                    [
                        16,
                        32767,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_95651a753ac610e2b82d": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 2,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "clamp",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 64,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_95651a753ac610e2b82d",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_95651a753ac610e2b82d",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        64,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_9847484ea5457bb35538": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 4,
            "exchange": False,
            "cluster": 4,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_9847484ea5457bb35538",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 16768,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_9847484ea5457bb35538",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_107a": [
                    [
                        0,
                        1,
                    ],
                    [
                        16,
                        32,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_9ba32c9871c70ad8a05b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 16,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_9ba32c9871c70ad8a05b",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_9ba32c9871c70ad8a05b",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        0,
                        1024,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        0,
                        1024,
                    ],
                ],
                "sm_107a": [
                    [
                        16,
                        0,
                        1024,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_9fa7a50247299d3b4934": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_9fa7a50247299d3b4934",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_9fa7a50247299d3b4934",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_103a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_107a": [
                    [
                        32,
                        128,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_a44c84445d2eae1b90e1": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 64,
            "lanes": 8,
            "rows_per_thread": 4,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": True,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_a44c84445d2eae1b90e1",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_a44c84445d2eae1b90e1",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "routes": {
                "rmsnorm_sigmoid_gate_bf16_pdl": {
                    "sm_107a": [
                        [
                            128,
                            512,
                            2560,
                        ],
                        [
                            32,
                            2048,
                            None,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_107a": [
                    [
                        128,
                        511,
                        2560,
                    ],
                    [
                        32,
                        2047,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_aa9676a8f77dd150d76d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_aa9676a8f77dd150d76d",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_aa9676a8f77dd150d76d",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        383,
                    ],
                    [
                        4096,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        2048,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_abe82055212c1709a8f5": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "beta",
            "planes_path": "sts",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_abe82055212c1709a8f5",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33152,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_abe82055212c1709a8f5",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_af2f4d40b657be0bb34a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "el",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_af2f4d40b657be0bb34a",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_af2f4d40b657be0bb34a",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        256,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_b17781825a525cc005d3": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 8,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 16,
            "tcluster": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_b17781825a525cc005d3",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                8,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_b17781825a525cc005d3",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        32,
                        64,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_b45eb8d58f5d583f90de": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 4,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": True,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 128,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_b45eb8d58f5d583f90de",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_b45eb8d58f5d583f90de",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        383,
                        511,
                    ],
                    [
                        128,
                        2560,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_b9eed36611ad33234d8a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 256,
            "split": 8,
            "exchange": False,
            "cluster": 2,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "asm",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_b9eed36611ad33234d8a",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                2,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 8576,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_b9eed36611ad33234d8a",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_103a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_107a": [
                    [
                        1,
                        16,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_cfe0205d02e7d6bb3910": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 1024,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": True,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": False,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_cfe0205d02e7d6bb3910",
        "launch": {
            "block": (
                1024,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_cfe0205d02e7d6bb3910",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_107a": [
                    [
                        256,
                        4096,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_dd08e4a4868c172adcb0": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 1024,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": True,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": False,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 3,
            "fill_first": True,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_dd08e4a4868c172adcb0",
        "launch": {
            "block": (
                1024,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 148864,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_dd08e4a4868c172adcb0",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "routes": {
                "gated_residual_rmsnorm_mode1_pdl": {
                    "sm_107a": [
                        [
                            4096,
                            None,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_107a": [
                    [
                        4096,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_dfc7a1d004ac1a941e18": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": False,
            "fold": "chain",
            "packed": False,
            "hoist": False,
            "pfold": "chain",
            "branch_first": False,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "beta",
            "planes_path": "sts",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_dfc7a1d004ac1a941e18",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33152,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_dfc7a1d004ac1a941e18",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        383,
                        2048,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        224,
                    ],
                    [
                        256,
                        2048,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_e3b7f09c2efce010a4df": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 4,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": True,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 128,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_e3b7f09c2efce010a4df",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_e3b7f09c2efce010a4df",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        128,
                        383,
                        511,
                    ],
                    [
                        128,
                        2560,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_e9b450856a857e161abd": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 2,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_e9b450856a857e161abd",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_e9b450856a857e161abd",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_107a": [
                    [
                        32,
                        767,
                        2047,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_eaad142db89e17b223a5": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 64,
            "lanes": 4,
            "rows_per_thread": 1,
            "width": 256,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 16,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_eaad142db89e17b223a5",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_eaad142db89e17b223a5",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "routes": {
                "rmsnorm_sigmoid_gate_bf16_pdl": {
                    "sm_107a": [
                        [
                            16,
                            2047,
                            6143,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_107a": [
                    [
                        16,
                        2047,
                        8191,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_eaeebf48f0731067d197": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 0,
            "threads": 512,
            "split": 4,
            "exchange": False,
            "cluster": 4,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": True,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_eaeebf48f0731067d197",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 16768,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_eaeebf48f0731067d197",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode0_pdl",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        64,
                    ],
                ],
                "sm_107a": [
                    [
                        0,
                        1,
                    ],
                    [
                        16,
                        32,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_eb1723bf9c72eb1d1324": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "heads",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "heads",
            "rows_per_cta": 32,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_eb1723bf9c72eb1d1324",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_eb1723bf9c72eb1d1324",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "routes": {
                "rmsnorm_sigmoid_gate_bf16_pdl": {
                    "sm_103a": [
                        [
                            128,
                            256,
                            2048,
                        ],
                    ],
                },
            },
            "windows": {
                "sm_100a": [
                    [
                        128,
                        256,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        256,
                        None,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        256,
                        383,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_ebdeb2e781ad78b473b2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 256,
            "split": 8,
            "exchange": False,
            "cluster": 2,
            "vec": 8,
            "epilogue": "staged",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ebdeb2e781ad78b473b2",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                2,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 8576,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_ebdeb2e781ad78b473b2",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_103a": [
                    [
                        0,
                        16,
                    ],
                ],
                "sm_107a": [
                    [
                        1,
                        16,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_f08a4703a47deb7cd269": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 128,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": True,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 16,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_f08a4703a47deb7cd269",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_f08a4703a47deb7cd269",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        8,
                        256,
                    ],
                ],
                "sm_103a": [
                    [
                        16,
                        8,
                        64,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_fb3375381614d19f247c": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 32,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": False,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 16,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 1,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 8,
            "tcluster": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_fb3375381614d19f247c",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_fb3375381614d19f247c",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_100a": [
                    [
                        128,
                        0,
                        8,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        0,
                        8,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        1,
                        2,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_fbaed0bd87c768319f92": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "weight",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "mode": 1,
            "threads": 512,
            "split": 1,
            "exchange": False,
            "cluster": 1,
            "vec": 8,
            "epilogue": "direct",
            "persistent": False,
            "min_blocks": 0,
            "pdl": False,
            "rsqrt_approx": True,
            "fold": "chain",
            "packed": True,
            "hoist": True,
            "pfold": "tree",
            "branch_first": True,
            "pdl_late": False,
            "pdl_wait_top": False,
            "pdl_launch_top": False,
            "prologue_h": True,
            "grid_y": False,
            "dyn_cluster": False,
            "index32": False,
            "x_hint": "evict_first",
            "w_hint": "read_only",
            "s_hint": "evict_first",
            "ab_hint": "read_only",
            "o_hint": "none",
            "stage_buffers": 2,
            "carveout": None,
            "packed_widen": "shift_mask",
            "planes_smem": "none",
            "planes_path": "cp_async",
            "x_words": False,
            "rows_per_cta": 1,
            "plane_width": 256,
            "out_width": 256,
            "residual_first": False,
            "tma_stages": 0,
            "fill_first": False,
            "ring_content": "both",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_fbaed0bd87c768319f92",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 384,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_fbaed0bd87c768319f92",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "beta": "beta",
            "branch": "branch",
            "eps": "eps",
            "gain_center": "gain_center",
            "out": "out",
            "residual": "residual",
            "weight": "weight",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_rmsnorm",
        "route": "gated_residual_rmsnorm_mode1",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_103a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_107a": [
                    [
                        32,
                        128,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_fec226d785e1824e0276": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "x",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "parameter",
                "tokens",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "x_stride",
            ],
            [
                "parameter",
                "gate_stride",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "threads": 256,
            "lanes": 8,
            "rows_per_thread": 1,
            "width": 128,
            "pdl": True,
            "pdl_late": False,
            "pdl_launch_top": False,
            "pdl_tail": 0,
            "tail": "none",
            "clobber": True,
            "x_hint": "none",
            "g_hint": "none",
            "y_hint": "none",
            "packed": False,
            "cluster": 1,
            "addr": "i64",
            "sigmoid": "exp2",
            "static": None,
            "tcluster": 16,
            "g_nc": True,
            "order": "tokens",
            "interleave": 1,
            "tpc": 1,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "interleave": 1,
            "kind": "tokens_heads",
            "order": "tokens",
            "rows_per_cta": 32,
            "tcluster": 16,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_fec226d785e1824e0276",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_muse_spark_norm_fec226d785e1824e0276",
        "logical": {
            "eps": "eps",
            "gate": "gate",
            "gate_stride": "gate_stride",
            "heads": "heads",
            "tokens": "tokens",
            "x": "input",
            "x_stride": "input_stride",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "rmsnorm_sigmoid_gate",
        "route": "rmsnorm_sigmoid_gate_bf16_pdl",
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {
                "sm_103a": [
                    [
                        128,
                        8,
                        32,
                    ],
                    [
                        32,
                        1,
                        8,
                    ],
                ],
                "sm_107a": [
                    [
                        32,
                        1,
                        9,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_008d26db65c7b14d5fb5": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 16,
            "threads": 64,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_008d26db65c7b14d5fb5",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_008d26db65c7b14d5fb5",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        1,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_00f6bd3a06a615aef9f9": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 8,
            "threads": 128,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "row_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_00f6bd3a06a615aef9f9",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                8,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_00f6bd3a06a615aef9f9",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        4,
                        16,
                    ],
                ],
                "sm_107a": [
                    [
                        0,
                        512,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_16a1cb15c135e0a8464b": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 8,
            "threads": 128,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_16a1cb15c135e0a8464b",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                8,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_16a1cb15c135e0a8464b",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        1,
                        64,
                    ],
                ],
                "sm_107a": [
                    [
                        0,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_1784f3c908851626f7d3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 1,
            "threads": 512,
            "vec": 16,
            "persistent": False,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "acquire_cluster",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1784f3c908851626f7d3",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_1784f3c908851626f7d3",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        148,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_1aafa313e4b01bb7f5b7": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 1,
            "threads": 512,
            "vec": 16,
            "persistent": False,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "acquire_cluster",
            "issue_order": "source",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1aafa313e4b01bb7f5b7",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_1aafa313e4b01bb7f5b7",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        64,
                        148,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        256,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_28d1182647ea4b51bdac": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 1,
            "threads": 512,
            "vec": 16,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "acquire_cluster",
            "issue_order": "source",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_28d1182647ea4b51bdac",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_28d1182647ea4b51bdac",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_107a": [
                    [
                        4096,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_2c0fcd41ad79a0293829": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 1,
            "threads": 512,
            "vec": 16,
            "persistent": False,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "acquire_cluster",
            "issue_order": "source",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_2c0fcd41ad79a0293829",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_2c0fcd41ad79a0293829",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        0,
                        None,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_36c4ae432a5f4a53f938": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": True,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_36c4ae432a5f4a53f938",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_36c4ae432a5f4a53f938",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_103a": [
                    [
                        32,
                        128,
                    ],
                    [
                        320,
                        384,
                    ],
                    [
                        2560,
                        None,
                    ],
                ],
                "sm_107a": [
                    [
                        256,
                        4096,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_4302c6926f9ee940dffa": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 1,
            "threads": 512,
            "vec": 16,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "acquire_cluster",
            "issue_order": "source",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_4302c6926f9ee940dffa",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_4302c6926f9ee940dffa",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        4096,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        3584,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_4330a526938b917018e5": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": True,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_4330a526938b917018e5",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_4330a526938b917018e5",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_103a": [
                    [
                        32,
                        64,
                    ],
                    [
                        128,
                        512,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_4d8b2e18bb1c88e5ecd0": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": False,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "acquire_cluster",
            "issue_order": "source",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_4d8b2e18bb1c88e5ecd0",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_4d8b2e18bb1c88e5ecd0",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        64,
                        128,
                    ],
                ],
                "sm_103a": [
                    [
                        64,
                        128,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_6b9946cc9f011720311f": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 8,
            "threads": 128,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_6b9946cc9f011720311f",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                8,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_6b9946cc9f011720311f",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_107a": [
                    [
                        0,
                        128,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_84b31104ac86bd40d3fe": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 1,
            "threads": 512,
            "vec": 16,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "acquire_cluster",
            "issue_order": "source",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 3,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_84b31104ac86bd40d3fe",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 149504,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_84b31104ac86bd40d3fe",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        512,
                        1151,
                    ],
                ],
                "sm_103a": [
                    [
                        768,
                        1151,
                    ],
                ],
                "sm_107a": [
                    [
                        512,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_9fa40e26c305d1a9d015": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "acquire_cluster",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_9fa40e26c305d1a9d015",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_9fa40e26c305d1a9d015",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        256,
                        383,
                    ],
                    [
                        4096,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_a00e917ba1d1035dccf4": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 8,
            "threads": 128,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "row_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_a00e917ba1d1035dccf4",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                8,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_a00e917ba1d1035dccf4",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_103a": [
                    [
                        0,
                        32,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_ab7d2b1a4242b384709f": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ab7d2b1a4242b384709f",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_ab7d2b1a4242b384709f",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        16,
                        64,
                    ],
                    [
                        128,
                        512,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_c06a45c369bb267515f6": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 8,
            "threads": 128,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "row_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_c06a45c369bb267515f6",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                8,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_c06a45c369bb267515f6",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_103a": [
                    [
                        0,
                        32,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_cbd917375dce308daa21": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 16,
            "threads": 64,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": False,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_cbd917375dce308daa21",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                16,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_cbd917375dce308daa21",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        4,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_ccd3af7d6b21e7971425": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 2,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ccd3af7d6b21e7971425",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_ccd3af7d6b21e7971425",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        1151,
                        4096,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_feb60e6ba9fc9270be5d": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": "ld_ll",
            "packed": False,
            "wait_form": "acquire_cluster",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": False,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_feb60e6ba9fc9270be5d",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_feb60e6ba9fc9270be5d",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        383,
                        4096,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_6146ac6f8533b54a3137": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": False,
            "post_weight": False,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": "ld_ll",
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": True,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 0,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_6146ac6f8533b54a3137",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_6146ac6f8533b54a3137",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_scalar_nopw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_103a": [
                    [
                        256,
                        320,
                    ],
                    [
                        384,
                        2560,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_bdcc0ca1fef907537d3b": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "H": 8192,
            "cluster": 4,
            "threads": 256,
            "vec": 8,
            "persistent": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "resident": True,
            "exit_sync": False,
            "tensor_alpha": True,
            "post_weight": True,
            "pdl": False,
            "rsqrt_approx": False,
            "min_blocks": 0,
            "load_order": "vectors_first",
            "l2_hint": None,
            "packed": True,
            "wait_form": "cta",
            "issue_order": "early",
            "reload_fp32": False,
            "expect_hoist": True,
            "store_all_lanes": False,
            "tma_stages": 0,
            "pdl_launch": "statistic",
            "ring": None,
            "prefetch_l2": 2,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_bdcc0ca1fef907537d3b",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                4,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_bdcc0ca1fef907537d3b",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_103a": [
                    [
                        512,
                        768,
                    ],
                    [
                        1151,
                        3584,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "gated_residual_next_rmsnorm_alpha_scalar_nopw": [
            "cake_muse_spark_norm_008d26db65c7b14d5fb5",
            "cake_muse_spark_norm_16a1cb15c135e0a8464b",
            "cake_muse_spark_norm_1aafa313e4b01bb7f5b7",
            "cake_muse_spark_norm_1784f3c908851626f7d3",
            "cake_muse_spark_norm_9fa40e26c305d1a9d015",
            "cake_muse_spark_norm_feb60e6ba9fc9270be5d",
        ],
        "gated_residual_next_rmsnorm_alpha_scalar_pw": [
            "cake_muse_spark_norm_2c0fcd41ad79a0293829",
        ],
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_cbd917375dce308daa21",
            "cake_muse_spark_norm_00f6bd3a06a615aef9f9",
            "cake_muse_spark_norm_ab7d2b1a4242b384709f",
            "cake_muse_spark_norm_4d8b2e18bb1c88e5ecd0",
            "cake_muse_spark_norm_84b31104ac86bd40d3fe",
            "cake_muse_spark_norm_ccd3af7d6b21e7971425",
            "cake_muse_spark_norm_4302c6926f9ee940dffa",
        ],
        "gated_residual_rmsnorm_mode0": [
            "cake_muse_spark_norm_102d29ef70510879670f",
            "cake_muse_spark_norm_9847484ea5457bb35538",
            "cake_muse_spark_norm_37adff50d4fef57403a1",
            "cake_muse_spark_norm_1282e67c9f3fa0934fd6",
            "cake_muse_spark_norm_dfc7a1d004ac1a941e18",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_8418dc5f50dc570d66db",
            "cake_muse_spark_norm_eaeebf48f0731067d197",
            "cake_muse_spark_norm_0782a35e531120ee0e27",
            "cake_muse_spark_norm_424f59177d4ed1568e7e",
            "cake_muse_spark_norm_13e0bf051991559b8b48",
            "cake_muse_spark_norm_7b370733525fe3e25f75",
            "cake_muse_spark_norm_778060a26127fd4ec76a",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_ebdeb2e781ad78b473b2",
            "cake_muse_spark_norm_2eb8f754997071825d1d",
            "cake_muse_spark_norm_fbaed0bd87c768319f92",
            "cake_muse_spark_norm_aa9676a8f77dd150d76d",
            "cake_muse_spark_norm_1a810e28e0c14fe4e541",
            "cake_muse_spark_norm_90b7ec1d7920cc5c2545",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_b9eed36611ad33234d8a",
            "cake_muse_spark_norm_754e16217f4dd91d0be2",
            "cake_muse_spark_norm_9fa7a50247299d3b4934",
            "cake_muse_spark_norm_af2f4d40b657be0bb34a",
            "cake_muse_spark_norm_35e16e98dde0f51c82e1",
            "cake_muse_spark_norm_5f812bc0f781c0f98731",
        ],
        "rmsnorm_sigmoid_gate_bf16": [
            "cake_muse_spark_norm_fb3375381614d19f247c",
            "cake_muse_spark_norm_57b7b5ad70819fd42c36",
            "cake_muse_spark_norm_eb1723bf9c72eb1d1324",
            "cake_muse_spark_norm_25235fb509f8cb1da0cf",
            "cake_muse_spark_norm_9ba32c9871c70ad8a05b",
            "cake_muse_spark_norm_8f55d029d0e862cc1e0b",
        ],
        "rmsnorm_sigmoid_gate_bf16_pdl": [
            "cake_muse_spark_norm_3e42b08f1d3e52bc3829",
            "cake_muse_spark_norm_87fc822d80b99514648a",
            "cake_muse_spark_norm_7bfe75ee1890611cca58",
            "cake_muse_spark_norm_004843ed6a555e0904da",
            "cake_muse_spark_norm_55308ae83554784c040f",
            "cake_muse_spark_norm_f08a4703a47deb7cd269",
            "cake_muse_spark_norm_8f55d029d0e862cc1e0b",
        ],
    },
    "sm_103a": {
        "gated_residual_next_rmsnorm_alpha_scalar_nopw": [
            "cake_muse_spark_norm_c06a45c369bb267515f6",
            "cake_muse_spark_norm_36c4ae432a5f4a53f938",
            "cake_muse_spark_norm_1aafa313e4b01bb7f5b7",
            "cake_muse_spark_norm_6146ac6f8533b54a3137",
        ],
        "gated_residual_next_rmsnorm_alpha_scalar_pw": [
            "cake_muse_spark_norm_2c0fcd41ad79a0293829",
        ],
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_a00e917ba1d1035dccf4",
            "cake_muse_spark_norm_4330a526938b917018e5",
            "cake_muse_spark_norm_4d8b2e18bb1c88e5ecd0",
            "cake_muse_spark_norm_bdcc0ca1fef907537d3b",
            "cake_muse_spark_norm_84b31104ac86bd40d3fe",
            "cake_muse_spark_norm_4302c6926f9ee940dffa",
        ],
        "gated_residual_rmsnorm_mode0": [
            "cake_muse_spark_norm_102d29ef70510879670f",
            "cake_muse_spark_norm_9847484ea5457bb35538",
            "cake_muse_spark_norm_37adff50d4fef57403a1",
            "cake_muse_spark_norm_dfc7a1d004ac1a941e18",
            "cake_muse_spark_norm_1282e67c9f3fa0934fd6",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_8418dc5f50dc570d66db",
            "cake_muse_spark_norm_eaeebf48f0731067d197",
            "cake_muse_spark_norm_0782a35e531120ee0e27",
            "cake_muse_spark_norm_13e0bf051991559b8b48",
            "cake_muse_spark_norm_424f59177d4ed1568e7e",
            "cake_muse_spark_norm_778060a26127fd4ec76a",
            "cake_muse_spark_norm_7b370733525fe3e25f75",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_ebdeb2e781ad78b473b2",
            "cake_muse_spark_norm_2eb8f754997071825d1d",
            "cake_muse_spark_norm_fbaed0bd87c768319f92",
            "cake_muse_spark_norm_90b7ec1d7920cc5c2545",
            "cake_muse_spark_norm_aa9676a8f77dd150d76d",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_b9eed36611ad33234d8a",
            "cake_muse_spark_norm_754e16217f4dd91d0be2",
            "cake_muse_spark_norm_9fa7a50247299d3b4934",
            "cake_muse_spark_norm_af2f4d40b657be0bb34a",
            "cake_muse_spark_norm_35e16e98dde0f51c82e1",
            "cake_muse_spark_norm_5f812bc0f781c0f98731",
        ],
        "rmsnorm_sigmoid_gate_bf16": [
            "cake_muse_spark_norm_fb3375381614d19f247c",
            "cake_muse_spark_norm_57b7b5ad70819fd42c36",
            "cake_muse_spark_norm_eb1723bf9c72eb1d1324",
            "cake_muse_spark_norm_25235fb509f8cb1da0cf",
            "cake_muse_spark_norm_9ba32c9871c70ad8a05b",
            "cake_muse_spark_norm_8f55d029d0e862cc1e0b",
        ],
        "rmsnorm_sigmoid_gate_bf16_pdl": [
            "cake_muse_spark_norm_3e42b08f1d3e52bc3829",
            "cake_muse_spark_norm_fec226d785e1824e0276",
            "cake_muse_spark_norm_87fc822d80b99514648a",
            "cake_muse_spark_norm_eb1723bf9c72eb1d1324",
            "cake_muse_spark_norm_7bfe75ee1890611cca58",
            "cake_muse_spark_norm_25235fb509f8cb1da0cf",
            "cake_muse_spark_norm_004843ed6a555e0904da",
            "cake_muse_spark_norm_55308ae83554784c040f",
            "cake_muse_spark_norm_f08a4703a47deb7cd269",
            "cake_muse_spark_norm_8f55d029d0e862cc1e0b",
        ],
    },
    "sm_107a": {
        "gated_residual_next_rmsnorm_alpha_scalar_nopw": [
            "cake_muse_spark_norm_16a1cb15c135e0a8464b",
            "cake_muse_spark_norm_36c4ae432a5f4a53f938",
            "cake_muse_spark_norm_28d1182647ea4b51bdac",
        ],
        "gated_residual_next_rmsnorm_alpha_scalar_pw": [
            "cake_muse_spark_norm_6b9946cc9f011720311f",
            "cake_muse_spark_norm_2c0fcd41ad79a0293829",
        ],
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_00f6bd3a06a615aef9f9",
            "cake_muse_spark_norm_84b31104ac86bd40d3fe",
        ],
        "gated_residual_rmsnorm_mode0": [
            "cake_muse_spark_norm_9847484ea5457bb35538",
            "cake_muse_spark_norm_102d29ef70510879670f",
            "cake_muse_spark_norm_37adff50d4fef57403a1",
            "cake_muse_spark_norm_cfe0205d02e7d6bb3910",
            "cake_muse_spark_norm_45fcc7c9bbfd36dabe9f",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_eaeebf48f0731067d197",
            "cake_muse_spark_norm_8418dc5f50dc570d66db",
            "cake_muse_spark_norm_0782a35e531120ee0e27",
            "cake_muse_spark_norm_38daeaf6d495ec29b417",
            "cake_muse_spark_norm_45fcc7c9bbfd36dabe9f",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_2eb8f754997071825d1d",
            "cake_muse_spark_norm_ebdeb2e781ad78b473b2",
            "cake_muse_spark_norm_fbaed0bd87c768319f92",
            "cake_muse_spark_norm_1a810e28e0c14fe4e541",
            "cake_muse_spark_norm_8e869908b1526226830a",
            "cake_muse_spark_norm_dd08e4a4868c172adcb0",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_754e16217f4dd91d0be2",
            "cake_muse_spark_norm_b9eed36611ad33234d8a",
            "cake_muse_spark_norm_9fa7a50247299d3b4934",
            "cake_muse_spark_norm_abe82055212c1709a8f5",
            "cake_muse_spark_norm_8d16ecfd030500bcd131",
            "cake_muse_spark_norm_dd08e4a4868c172adcb0",
        ],
        "rmsnorm_sigmoid_gate_bf16": [
            "cake_muse_spark_norm_fb3375381614d19f247c",
            "cake_muse_spark_norm_519a7a9fdc31418e30cf",
            "cake_muse_spark_norm_95651a753ac610e2b82d",
            "cake_muse_spark_norm_eb1723bf9c72eb1d1324",
            "cake_muse_spark_norm_e3b7f09c2efce010a4df",
            "cake_muse_spark_norm_a44c84445d2eae1b90e1",
            "cake_muse_spark_norm_57b7b5ad70819fd42c36",
            "cake_muse_spark_norm_b17781825a525cc005d3",
            "cake_muse_spark_norm_e9b450856a857e161abd",
            "cake_muse_spark_norm_9ba32c9871c70ad8a05b",
            "cake_muse_spark_norm_8f55d029d0e862cc1e0b",
            "cake_muse_spark_norm_eaad142db89e17b223a5",
            "cake_muse_spark_norm_7b9c67698ca00f540a99",
        ],
        "rmsnorm_sigmoid_gate_bf16_pdl": [
            "cake_muse_spark_norm_3e42b08f1d3e52bc3829",
            "cake_muse_spark_norm_77358559505c434c4bb7",
            "cake_muse_spark_norm_2413d64185f668557ed6",
            "cake_muse_spark_norm_7bfe75ee1890611cca58",
            "cake_muse_spark_norm_b45eb8d58f5d583f90de",
            "cake_muse_spark_norm_2ea93a6e5dcb5daac83d",
            "cake_muse_spark_norm_a44c84445d2eae1b90e1",
            "cake_muse_spark_norm_fec226d785e1824e0276",
            "cake_muse_spark_norm_004843ed6a555e0904da",
            "cake_muse_spark_norm_62b6802c3f845a5a02cb",
            "cake_muse_spark_norm_55308ae83554784c040f",
            "cake_muse_spark_norm_8f55d029d0e862cc1e0b",
            "cake_muse_spark_norm_eaad142db89e17b223a5",
            "cake_muse_spark_norm_94ddf891ecc46283d87c",
            "cake_muse_spark_norm_7b9c67698ca00f540a99",
        ],
    },
}

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 /
# GB300 (``sm_103a``) and R200 (``sm_107a``).  Every architecture is compiled
# with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
HIDDEN_SIZE = 8192

#: The operators of the package and the variants of each (the route suffixes a
#: public call maps onto: which optional inputs the programs take, or the mode
#: and the PDL form).
OPS = ("gated_residual_next_rmsnorm", "gated_residual_rmsnorm", "rmsnorm_sigmoid_gate")
VARIANTS: dict[str, tuple[str, ...]] = {
    "gated_residual_next_rmsnorm": (
        "alpha_tensor_pw",
        "alpha_scalar_nopw",
        "alpha_scalar_pw",
        "alpha_tensor_nopw",
    ),
    "gated_residual_rmsnorm": ("mode0", "mode0_pdl", "mode1", "mode1_pdl"),
    "rmsnorm_sigmoid_gate": ("bf16", "bf16_pdl"),
}
SELECT_KIND = "rows_between"
SELECT_KINDS = (SELECT_KIND, "tokens_heads_between")
GRID_KINDS = ("rows", "persistent_rows", "tokens_heads")

#: The logical argument vocabulary of ``run_module`` ``values``.
LOGICAL_TENSORS = (
    "branch",
    "residual",
    "alpha",
    "beta",
    "next_weight",
    "post_weight",
    "residual_out",
    "ffn_input",
    "weight",
    "out",
    "input",
    "gate",
)
LOGICAL_SCALARS = (
    "rows",
    "post_eps",
    "next_eps",
    "post_gain_center",
    "next_gain_center",
    "alpha_value",
    "eps",
    "gain_center",
    "tokens",
    "heads",
    "input_stride",
    "gate_stride",
)


def route_key(op: str, variant: str) -> str:
    """The route one public call of ``op`` with ``variant`` launches from: ``<op>_<variant>``."""
    if op not in OPS:
        raise ValueError(f"unknown Cake norm operator {op!r}; the package ships {OPS}")
    if variant not in VARIANTS[op]:
        raise ValueError(
            f"unknown variant {variant!r} of {op}; its variants are {VARIANTS[op]}"
        )
    return f"{op}_{variant}"


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""
    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake norm target"
        )
    return arch


def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def route_available(arch: str, route: str) -> bool:
    """Whether ``arch`` ships a program of ``route``."""
    return bool(ROUTES.get(arch, {}).get(route))


def route_modules(arch: str, route: str) -> tuple[str, ...]:
    """The records of ``route`` on ``arch`` in window order (empty when the route is not exported there)."""
    return tuple(ROUTES.get(arch, {}).get(route, ()))


def admits(
    rule: dict[str, Any] | None,
    rows: int,
    arch: str,
    heads: int | None = None,
    route: str | None = None,
) -> bool:
    """The record's selection rule: ``None`` admits every call; ``rows_between`` admits ``rows`` on ``arch``
    when one of the architecture's windows holds ``above < rows <= at_most`` (``at_most`` ``None`` = unbounded);
    ``tokens_heads_between`` admits ``rows`` tokens of ``heads`` heads when one of the architecture's windows
    ``[heads, above, at_most]`` names the head count and holds the token count.  A rule's ``routes`` spell the
    windows another route gives the record (a PDL route launching the plain build): a call of that ``route``
    reads them in place of ``windows``."""
    if rule is None:
        return True
    kind = rule.get("kind")
    if kind not in SELECT_KINDS:
        raise ValueError(f"unknown Cake norm select rule {kind!r}")
    windows = rule.get("routes", {}).get(route, rule["windows"]).get(arch, ())
    if kind == SELECT_KIND:
        return any(
            rows > int(above) and (at_most is None or rows <= int(at_most))
            for above, at_most in windows
        )
    if heads is None:
        raise ValueError(f"a {kind} select rule needs the head count of the call")
    return any(
        int(count) == int(heads)
        and rows > int(above)
        and (at_most is None or rows <= int(at_most))
        for count, above, at_most in windows
    )


def selected_module(arch: str, route: str, rows: int, heads: int | None = None) -> str:
    """The record of ``route`` whose window admits ``rows`` on ``arch`` (``heads`` the head count of a route
    selecting on it)."""
    if rows <= 0:
        raise ValueError(f"Cake norm launches need a positive row count, got {rows}")
    for name in route_modules(arch, route):
        if admits(MODULES[name].get("select_rule"), rows, arch, heads, route):
            return name
    heads_text = "" if heads is None else f" of {heads} heads"
    raise NotImplementedError(
        f"no exported program of route {route!r} on {arch} admits {rows} rows{heads_text}"
    )


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake norm module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    ``cake_build.load_library`` serves a prebuilt ``aot/`` library when a
    builder compiled it and compiles the manifest's sources otherwise.
    """
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


@functools.cache
def cluster_capacity(name: str, arch: str, device_index: int) -> int:
    """How many clusters of record ``name``'s launch shape device ``device_index`` holds at once: the record's
    occupancy query over its launch record, asked once per (record, architecture, device).  Loads the library on
    first use: call it outside CUDA graph capture."""
    record = MODULES[name]
    launch = record["launch"]
    query = getattr(load(name, arch), record["occupancy_entry"])
    block = [int(value) for value in launch["block"]]
    cluster = [int(value) for value in launch["cluster"]]
    return int(
        query(
            int(device_index),
            *block,
            *cluster,
            int(launch["dynamic_smem_bytes"]),
            bool(launch["cooperative"]),
        )
    )


def tokens_heads_grid(
    rule: dict[str, Any], tokens: int, heads: int
) -> tuple[int, int, int]:
    """The grid of a ``tokens_heads`` rule for ``tokens`` tokens of ``heads`` heads (the kernel's own
    arithmetic): ``(ceil(heads / rows_per_cta), tokens rounded up to tcluster, 1)`` for a cluster over the head
    blocks or the head-major order, else ``(tokens rounded up to max(tcluster, interleave), ceil(heads /
    rows_per_cta), 1)``; for a multi-token program (the rule's ``tpc`` above one: a CTA holding ``tpc`` whole
    tokens) ``(ceil(tokens / tpc), ceil(heads / (rows_per_cta / tpc)), 1)``."""
    if tokens <= 0 or heads <= 0:
        raise ValueError(
            f"Cake norm tokens_heads launches need positive counts, got tokens={tokens}, heads={heads}"
        )
    tpc = int(rule.get("tpc", 1))
    if tpc > 1:
        return (
            -(-int(tokens) // tpc),
            -(-int(heads) // (int(rule["rows_per_cta"]) // tpc)),
            1,
        )
    blocks = -(-int(heads) // int(rule["rows_per_cta"]))
    tcluster, interleave = int(rule.get("tcluster", 1)), int(rule.get("interleave", 1))
    if int(rule["cluster"]) > 1 or rule.get("order") == "heads":
        return (blocks, -(-int(tokens) // tcluster) * tcluster, 1)
    step = max(tcluster, interleave)
    return (-(-int(tokens) // step) * step, blocks, 1)


def launch_grid(
    record: dict[str, Any],
    rows: int,
    *,
    device_index: int,
    values: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one record under its recorded grid rule: ``rows x row_ctas`` CTAs (``row_ctas`` the
    rule's CTAs per row, the cluster when it names none), ``min(rows, co-resident clusters) x cluster``
    CTAs for the persistent row loop, or the two-dimensional ``tokens_heads`` grid of ``rows`` tokens and the
    ``heads`` of ``values`` (:func:`tokens_heads_grid`)."""
    if rows <= 0:
        raise ValueError(f"Cake norm launches need a positive row count, got {rows}")
    rule = record["grid_rule"]
    cluster = int(rule["cluster"])
    kind = rule.get("kind")
    if kind == "rows":
        return (rows * int(rule.get("row_ctas", cluster)), 1, 1)
    if kind == "persistent_rows":
        capacity = cluster_capacity(
            record["library"], device_arch(device_index), device_index
        )
        return (min(rows, max(1, capacity)) * cluster, 1, 1)
    if kind == "tokens_heads":
        if values is None or "heads" not in values:
            raise ValueError(
                f"module {record['library']} launches a tokens_heads grid; its values must carry 'heads'"
            )
        return tokens_heads_grid(rule, rows, int(values["heads"]))
    raise ValueError(f"unknown Cake norm grid rule {kind!r}")


def launch_args(
    record: dict[str, Any], values: dict[str, Any], grid: Sequence[int]
) -> tuple[Any, ...]:
    """The positional arguments of the record's ``ffi_entry``: ``values`` by logical name in ``arg_plan`` order,
    the grid axes where the plan names them."""
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    logical = record["logical"]
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {record['library']} needs logical argument {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    return tuple(args)


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values`` (``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``)
    over ``rows`` rows on the current torch stream of device ``device_index``."""
    record = MODULES[name]
    grid = launch_grid(record, rows, device_index=device_index, values=values)
    args = launch_args(record, values, grid)
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GRID_KINDS",
    "HIDDEN_SIZE",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "OPS",
    "ROUTES",
    "SELECT_KIND",
    "SELECT_KINDS",
    "VARIANTS",
    "admits",
    "arch_for_capability",
    "cluster_capacity",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "route_available",
    "route_key",
    "route_modules",
    "run_module",
    "selected_module",
    "sm_count",
    "specs_for_arch",
    "tokens_heads_grid",
]
