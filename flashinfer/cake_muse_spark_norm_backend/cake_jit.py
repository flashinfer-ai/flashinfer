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
    "cake_muse_spark_norm_1aa28374ae397e05d09f": {
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
            "exit_sync": False,
            "expect_hoist": False,
            "issue_order": "source",
            "l2_hint": None,
            "load_order": "vectors_first",
            "min_blocks": 0,
            "packed": True,
            "pdl": False,
            "pdl_launch": "statistic",
            "persistent": True,
            "post_weight": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "reload_fp32": False,
            "resident": True,
            "rsqrt_approx": False,
            "store_all_lanes": False,
            "tensor_alpha": True,
            "threads": 512,
            "tma_stages": 0,
            "vec": 16,
            "wait_form": "acquire_cluster",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1aa28374ae397e05d09f",
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
        "library": "cake_muse_spark_norm_1aa28374ae397e05d09f",
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
                        1536,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        1536,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_1aab303bce601c44d1bf": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1aab303bce601c44d1bf",
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
        "library": "cake_muse_spark_norm_1aab303bce601c44d1bf",
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
    "cake_muse_spark_norm_514a5ab46110982beca0": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_514a5ab46110982beca0",
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
        "library": "cake_muse_spark_norm_514a5ab46110982beca0",
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
    "cake_muse_spark_norm_61c0e3739fd566fc4823": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_61c0e3739fd566fc4823",
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
        "library": "cake_muse_spark_norm_61c0e3739fd566fc4823",
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
                        1536,
                    ],
                ],
                "sm_103a": [
                    [
                        512,
                        1536,
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
    "cake_muse_spark_norm_649ee936d6fb2b6bebde": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_649ee936d6fb2b6bebde",
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
        "library": "cake_muse_spark_norm_649ee936d6fb2b6bebde",
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
    "cake_muse_spark_norm_674492f148b2f741c3fb": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_674492f148b2f741c3fb",
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
        "library": "cake_muse_spark_norm_674492f148b2f741c3fb",
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
    "cake_muse_spark_norm_77c158ed6d86d958e350": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_77c158ed6d86d958e350",
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
        "library": "cake_muse_spark_norm_77c158ed6d86d958e350",
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
    "cake_muse_spark_norm_93a116bbbd2f2ab6d8d8": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_93a116bbbd2f2ab6d8d8",
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
        "library": "cake_muse_spark_norm_93a116bbbd2f2ab6d8d8",
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
    "cake_muse_spark_norm_a412dc78e8a6e35982a1": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_a412dc78e8a6e35982a1",
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
        "library": "cake_muse_spark_norm_a412dc78e8a6e35982a1",
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
                        255,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_b1cd02a3e16d1ce6da24": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_b1cd02a3e16d1ce6da24",
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
        "library": "cake_muse_spark_norm_b1cd02a3e16d1ce6da24",
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
    "cake_muse_spark_norm_ba0bb7e00db23d01e03e": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ba0bb7e00db23d01e03e",
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
        "library": "cake_muse_spark_norm_ba0bb7e00db23d01e03e",
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
    "cake_muse_spark_norm_bf6e6987f25e4a2d184f": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_bf6e6987f25e4a2d184f",
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
        "library": "cake_muse_spark_norm_bf6e6987f25e4a2d184f",
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
    "cake_muse_spark_norm_bf9fcec05cbfa3d03bde": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_bf9fcec05cbfa3d03bde",
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
        "library": "cake_muse_spark_norm_bf9fcec05cbfa3d03bde",
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
                        256,
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
    "cake_muse_spark_norm_c4af0cbb5e8af3b727d4": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_c4af0cbb5e8af3b727d4",
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
        "library": "cake_muse_spark_norm_c4af0cbb5e8af3b727d4",
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
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_cd88d7ac28164403238d": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_cd88d7ac28164403238d",
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
        "library": "cake_muse_spark_norm_cd88d7ac28164403238d",
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
    "cake_muse_spark_norm_d12f249b80e1a32b7893": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 8,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_d12f249b80e1a32b7893",
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
        "library": "cake_muse_spark_norm_d12f249b80e1a32b7893",
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
    "cake_muse_spark_norm_01c5629ce7bb10cbedd5": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_01c5629ce7bb10cbedd5",
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
        "library": "cake_muse_spark_norm_01c5629ce7bb10cbedd5",
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
    "cake_muse_spark_norm_1cd221dc235a25c6ef14": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1cd221dc235a25c6ef14",
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
        "library": "cake_muse_spark_norm_1cd221dc235a25c6ef14",
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
    "cake_muse_spark_norm_35aa2c1130ddbd80b9f3": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_35aa2c1130ddbd80b9f3",
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
        "library": "cake_muse_spark_norm_35aa2c1130ddbd80b9f3",
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
                        4095,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        255,
                    ],
                    [
                        512,
                        1024,
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
    "cake_muse_spark_norm_3f8b7179529819e938b5": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_3f8b7179529819e938b5",
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
        "library": "cake_muse_spark_norm_3f8b7179529819e938b5",
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
    "cake_muse_spark_norm_4725afe7cdd93a93ff86": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_4725afe7cdd93a93ff86",
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
        "library": "cake_muse_spark_norm_4725afe7cdd93a93ff86",
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
    "cake_muse_spark_norm_584407b9521564b8200f": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_584407b9521564b8200f",
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
        "library": "cake_muse_spark_norm_584407b9521564b8200f",
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
                        2048,
                    ],
                ],
                "sm_103a": [
                    [
                        255,
                        2048,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        256,
                    ],
                    [
                        8192,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_5e9ca5bb1e4471cce3ff": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_5e9ca5bb1e4471cce3ff",
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
        "library": "cake_muse_spark_norm_5e9ca5bb1e4471cce3ff",
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
    "cake_muse_spark_norm_65d9966b792f76934fe2": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_65d9966b792f76934fe2",
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
        "library": "cake_muse_spark_norm_65d9966b792f76934fe2",
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
                        255,
                        2048,
                    ],
                ],
                "sm_107a": [
                    [
                        8192,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_688a50645a4e00669789": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_688a50645a4e00669789",
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
        "library": "cake_muse_spark_norm_688a50645a4e00669789",
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
    "cake_muse_spark_norm_6a387e4b7299123352f4": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_6a387e4b7299123352f4",
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
        "library": "cake_muse_spark_norm_6a387e4b7299123352f4",
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
    "cake_muse_spark_norm_8e2e17ebd97563f6a23a": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_8e2e17ebd97563f6a23a",
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
        "library": "cake_muse_spark_norm_8e2e17ebd97563f6a23a",
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
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        255,
                    ],
                    [
                        768,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_95298d67edf21a7f0c54": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_95298d67edf21a7f0c54",
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
        "library": "cake_muse_spark_norm_95298d67edf21a7f0c54",
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
    "cake_muse_spark_norm_96ba3336cf97f5be5841": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_96ba3336cf97f5be5841",
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
        "library": "cake_muse_spark_norm_96ba3336cf97f5be5841",
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
                        256,
                    ],
                ],
                "sm_103a": [
                    [
                        255,
                        512,
                    ],
                ],
                "sm_107a": [
                    [
                        8192,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_9ff8cc91c57ba9b2f7cc": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_9ff8cc91c57ba9b2f7cc",
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
        "library": "cake_muse_spark_norm_9ff8cc91c57ba9b2f7cc",
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
                        128,
                        255,
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
    "cake_muse_spark_norm_ad3d24641a423a260dea": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ad3d24641a423a260dea",
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
        "library": "cake_muse_spark_norm_ad3d24641a423a260dea",
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
    "cake_muse_spark_norm_b886dd6347cac2b58cf3": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_b886dd6347cac2b58cf3",
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
        "library": "cake_muse_spark_norm_b886dd6347cac2b58cf3",
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
                "sm_103a": [
                    [
                        255,
                        768,
                    ],
                ],
                "sm_107a": [
                    [
                        128,
                        256,
                    ],
                    [
                        8192,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_c6478d985959ea0426a6": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_c6478d985959ea0426a6",
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
        "library": "cake_muse_spark_norm_c6478d985959ea0426a6",
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
    "cake_muse_spark_norm_ce71d49220da1acf490f": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ce71d49220da1acf490f",
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
        "library": "cake_muse_spark_norm_ce71d49220da1acf490f",
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
                        2048,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        255,
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
    "cake_muse_spark_norm_e4067dd3b6842fa089a9": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_e4067dd3b6842fa089a9",
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
        "library": "cake_muse_spark_norm_e4067dd3b6842fa089a9",
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
    "cake_muse_spark_norm_ffbfde93a6d276795e1d": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ffbfde93a6d276795e1d",
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
        "library": "cake_muse_spark_norm_ffbfde93a6d276795e1d",
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
    "cake_muse_spark_norm_4e76b0cecea57bf56f1f": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_4e76b0cecea57bf56f1f",
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
        "library": "cake_muse_spark_norm_4e76b0cecea57bf56f1f",
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
                        8192,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_96fc60d0b38e67b4085a": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_96fc60d0b38e67b4085a",
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
        "library": "cake_muse_spark_norm_96fc60d0b38e67b4085a",
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
                        8192,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_b174718f7be21bb8a5d9": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_b174718f7be21bb8a5d9",
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
        "library": "cake_muse_spark_norm_b174718f7be21bb8a5d9",
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
    "cake_muse_spark_norm_bd0b88e124a14b9ebfad": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_bd0b88e124a14b9ebfad",
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
        "library": "cake_muse_spark_norm_bd0b88e124a14b9ebfad",
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
                        8192,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_c9753c6cab93c6f53ea2": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_c9753c6cab93c6f53ea2",
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
        "library": "cake_muse_spark_norm_c9753c6cab93c6f53ea2",
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
                        8192,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_0b979435de8b4c05214f": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_0b979435de8b4c05214f",
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
        "library": "cake_muse_spark_norm_0b979435de8b4c05214f",
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
    "cake_muse_spark_norm_0b9a1a6c1d1c3c614200": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
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
        "library": "cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
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
    "cake_muse_spark_norm_0e50091274e6013faa20": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_0e50091274e6013faa20",
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
        "library": "cake_muse_spark_norm_0e50091274e6013faa20",
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
    "cake_muse_spark_norm_1d29611fac2391f74182": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_1d29611fac2391f74182",
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
        "library": "cake_muse_spark_norm_1d29611fac2391f74182",
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
    "cake_muse_spark_norm_20723bd85c3bbf3e0ba1": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_20723bd85c3bbf3e0ba1",
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
        "library": "cake_muse_spark_norm_20723bd85c3bbf3e0ba1",
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
    "cake_muse_spark_norm_2206e897092d7ba2b069": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_2206e897092d7ba2b069",
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
        "library": "cake_muse_spark_norm_2206e897092d7ba2b069",
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
    "cake_muse_spark_norm_25a7f9c3f3bc6d00fda4": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_25a7f9c3f3bc6d00fda4",
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
        "library": "cake_muse_spark_norm_25a7f9c3f3bc6d00fda4",
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
    "cake_muse_spark_norm_37ea5413e1d3e395761f": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_37ea5413e1d3e395761f",
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
        "library": "cake_muse_spark_norm_37ea5413e1d3e395761f",
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
    "cake_muse_spark_norm_402f58bba74e553f32f7": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_402f58bba74e553f32f7",
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
        "library": "cake_muse_spark_norm_402f58bba74e553f32f7",
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
                        32,
                        9,
                        12,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_455dd018e9c2b6f6ffeb": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_455dd018e9c2b6f6ffeb",
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
        "library": "cake_muse_spark_norm_455dd018e9c2b6f6ffeb",
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
    "cake_muse_spark_norm_4dffe3ae22bb749cbf54": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_4dffe3ae22bb749cbf54",
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
        "library": "cake_muse_spark_norm_4dffe3ae22bb749cbf54",
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
    "cake_muse_spark_norm_4f35be1b624190afed66": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_4f35be1b624190afed66",
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
        "library": "cake_muse_spark_norm_4f35be1b624190afed66",
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
    "cake_muse_spark_norm_52fa7700c74897e3d56b": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_52fa7700c74897e3d56b",
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
        "library": "cake_muse_spark_norm_52fa7700c74897e3d56b",
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
    "cake_muse_spark_norm_5cf73e7885cf18f1c89a": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_5cf73e7885cf18f1c89a",
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
        "library": "cake_muse_spark_norm_5cf73e7885cf18f1c89a",
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
    "cake_muse_spark_norm_73cdd3ec3d059d8e95a9": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_73cdd3ec3d059d8e95a9",
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
        "library": "cake_muse_spark_norm_73cdd3ec3d059d8e95a9",
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
    "cake_muse_spark_norm_7c0cd927f0f3fea9f941": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_7c0cd927f0f3fea9f941",
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
        "library": "cake_muse_spark_norm_7c0cd927f0f3fea9f941",
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
    "cake_muse_spark_norm_8e6e989e8e2bab5d8163": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_8e6e989e8e2bab5d8163",
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
        "library": "cake_muse_spark_norm_8e6e989e8e2bab5d8163",
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
    "cake_muse_spark_norm_94dab509c0a287b9d339": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_94dab509c0a287b9d339",
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
        "library": "cake_muse_spark_norm_94dab509c0a287b9d339",
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
    "cake_muse_spark_norm_a34f5c08681c31eddf44": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_a34f5c08681c31eddf44",
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
        "library": "cake_muse_spark_norm_a34f5c08681c31eddf44",
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
    "cake_muse_spark_norm_a37e58e6a7d00cf8c622": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_a37e58e6a7d00cf8c622",
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
        "library": "cake_muse_spark_norm_a37e58e6a7d00cf8c622",
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
    "cake_muse_spark_norm_a9ffd668e885ed4c7e0b": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_a9ffd668e885ed4c7e0b",
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
        "library": "cake_muse_spark_norm_a9ffd668e885ed4c7e0b",
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
    "cake_muse_spark_norm_b246ba16efbd44a244f6": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_b246ba16efbd44a244f6",
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
        "library": "cake_muse_spark_norm_b246ba16efbd44a244f6",
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
    "cake_muse_spark_norm_b61c2310cd41e8e79bf8": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_b61c2310cd41e8e79bf8",
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
        "library": "cake_muse_spark_norm_b61c2310cd41e8e79bf8",
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
    "cake_muse_spark_norm_ba936288601d077849a2": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_ba936288601d077849a2",
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
        "library": "cake_muse_spark_norm_ba936288601d077849a2",
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
    "cake_muse_spark_norm_c7a321ceb4da94672252": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_c7a321ceb4da94672252",
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
        "library": "cake_muse_spark_norm_c7a321ceb4da94672252",
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
    "cake_muse_spark_norm_eb8b5c0195ced737054a": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_eb8b5c0195ced737054a",
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
        "library": "cake_muse_spark_norm_eb8b5c0195ced737054a",
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
    "cake_muse_spark_norm_f5cfa44f264f852f4db4": {
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
        "kernel_symbol": "kernel_cake_muse_spark_norm_f5cfa44f264f852f4db4",
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
        "library": "cake_muse_spark_norm_f5cfa44f264f852f4db4",
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
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "gated_residual_next_rmsnorm_alpha_scalar_nopw": [
            "cake_muse_spark_norm_649ee936d6fb2b6bebde",
            "cake_muse_spark_norm_1aab303bce601c44d1bf",
            "cake_muse_spark_norm_ba0bb7e00db23d01e03e",
            "cake_muse_spark_norm_93a116bbbd2f2ab6d8d8",
            "cake_muse_spark_norm_c4af0cbb5e8af3b727d4",
        ],
        "gated_residual_next_rmsnorm_alpha_scalar_pw": [
            "cake_muse_spark_norm_a412dc78e8a6e35982a1",
        ],
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_b1cd02a3e16d1ce6da24",
            "cake_muse_spark_norm_d12f249b80e1a32b7893",
            "cake_muse_spark_norm_674492f148b2f741c3fb",
            "cake_muse_spark_norm_514a5ab46110982beca0",
            "cake_muse_spark_norm_61c0e3739fd566fc4823",
            "cake_muse_spark_norm_1aa28374ae397e05d09f",
        ],
        "gated_residual_rmsnorm_mode0": [
            "cake_muse_spark_norm_1cd221dc235a25c6ef14",
            "cake_muse_spark_norm_e4067dd3b6842fa089a9",
            "cake_muse_spark_norm_95298d67edf21a7f0c54",
            "cake_muse_spark_norm_9ff8cc91c57ba9b2f7cc",
            "cake_muse_spark_norm_65d9966b792f76934fe2",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_688a50645a4e00669789",
            "cake_muse_spark_norm_5e9ca5bb1e4471cce3ff",
            "cake_muse_spark_norm_3f8b7179529819e938b5",
            "cake_muse_spark_norm_96ba3336cf97f5be5841",
            "cake_muse_spark_norm_35aa2c1130ddbd80b9f3",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_4725afe7cdd93a93ff86",
            "cake_muse_spark_norm_6a387e4b7299123352f4",
            "cake_muse_spark_norm_c6478d985959ea0426a6",
            "cake_muse_spark_norm_ce71d49220da1acf490f",
            "cake_muse_spark_norm_584407b9521564b8200f",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_01c5629ce7bb10cbedd5",
            "cake_muse_spark_norm_ad3d24641a423a260dea",
            "cake_muse_spark_norm_ffbfde93a6d276795e1d",
            "cake_muse_spark_norm_8e2e17ebd97563f6a23a",
        ],
        "rmsnorm_sigmoid_gate_bf16": [
            "cake_muse_spark_norm_ba936288601d077849a2",
            "cake_muse_spark_norm_2206e897092d7ba2b069",
            "cake_muse_spark_norm_a37e58e6a7d00cf8c622",
            "cake_muse_spark_norm_94dab509c0a287b9d339",
            "cake_muse_spark_norm_455dd018e9c2b6f6ffeb",
            "cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
        ],
        "rmsnorm_sigmoid_gate_bf16_pdl": [
            "cake_muse_spark_norm_0e50091274e6013faa20",
            "cake_muse_spark_norm_4f35be1b624190afed66",
            "cake_muse_spark_norm_8e6e989e8e2bab5d8163",
            "cake_muse_spark_norm_a9ffd668e885ed4c7e0b",
            "cake_muse_spark_norm_b61c2310cd41e8e79bf8",
            "cake_muse_spark_norm_7c0cd927f0f3fea9f941",
            "cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
        ],
    },
    "sm_103a": {
        "gated_residual_next_rmsnorm_alpha_scalar_nopw": [
            "cake_muse_spark_norm_cd88d7ac28164403238d",
            "cake_muse_spark_norm_bf9fcec05cbfa3d03bde",
            "cake_muse_spark_norm_ba0bb7e00db23d01e03e",
        ],
        "gated_residual_next_rmsnorm_alpha_scalar_pw": [
            "cake_muse_spark_norm_a412dc78e8a6e35982a1",
        ],
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_77c158ed6d86d958e350",
            "cake_muse_spark_norm_bf6e6987f25e4a2d184f",
            "cake_muse_spark_norm_514a5ab46110982beca0",
            "cake_muse_spark_norm_61c0e3739fd566fc4823",
            "cake_muse_spark_norm_1aa28374ae397e05d09f",
        ],
        "gated_residual_rmsnorm_mode0": [
            "cake_muse_spark_norm_1cd221dc235a25c6ef14",
            "cake_muse_spark_norm_e4067dd3b6842fa089a9",
            "cake_muse_spark_norm_95298d67edf21a7f0c54",
            "cake_muse_spark_norm_9ff8cc91c57ba9b2f7cc",
            "cake_muse_spark_norm_65d9966b792f76934fe2",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_688a50645a4e00669789",
            "cake_muse_spark_norm_5e9ca5bb1e4471cce3ff",
            "cake_muse_spark_norm_3f8b7179529819e938b5",
            "cake_muse_spark_norm_35aa2c1130ddbd80b9f3",
            "cake_muse_spark_norm_96ba3336cf97f5be5841",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_4725afe7cdd93a93ff86",
            "cake_muse_spark_norm_6a387e4b7299123352f4",
            "cake_muse_spark_norm_c6478d985959ea0426a6",
            "cake_muse_spark_norm_ce71d49220da1acf490f",
            "cake_muse_spark_norm_584407b9521564b8200f",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_01c5629ce7bb10cbedd5",
            "cake_muse_spark_norm_ad3d24641a423a260dea",
            "cake_muse_spark_norm_ffbfde93a6d276795e1d",
            "cake_muse_spark_norm_8e2e17ebd97563f6a23a",
            "cake_muse_spark_norm_b886dd6347cac2b58cf3",
        ],
        "rmsnorm_sigmoid_gate_bf16": [
            "cake_muse_spark_norm_ba936288601d077849a2",
            "cake_muse_spark_norm_2206e897092d7ba2b069",
            "cake_muse_spark_norm_a37e58e6a7d00cf8c622",
            "cake_muse_spark_norm_94dab509c0a287b9d339",
            "cake_muse_spark_norm_455dd018e9c2b6f6ffeb",
            "cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
        ],
        "rmsnorm_sigmoid_gate_bf16_pdl": [
            "cake_muse_spark_norm_0e50091274e6013faa20",
            "cake_muse_spark_norm_1d29611fac2391f74182",
            "cake_muse_spark_norm_4f35be1b624190afed66",
            "cake_muse_spark_norm_a37e58e6a7d00cf8c622",
            "cake_muse_spark_norm_8e6e989e8e2bab5d8163",
            "cake_muse_spark_norm_94dab509c0a287b9d339",
            "cake_muse_spark_norm_a9ffd668e885ed4c7e0b",
            "cake_muse_spark_norm_b61c2310cd41e8e79bf8",
            "cake_muse_spark_norm_7c0cd927f0f3fea9f941",
            "cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
        ],
    },
    "sm_107a": {
        "gated_residual_next_rmsnorm_alpha_scalar_nopw": [
            "cake_muse_spark_norm_1aab303bce601c44d1bf",
            "cake_muse_spark_norm_bf9fcec05cbfa3d03bde",
            "cake_muse_spark_norm_b174718f7be21bb8a5d9",
        ],
        "gated_residual_next_rmsnorm_alpha_scalar_pw": [
            "cake_muse_spark_norm_a412dc78e8a6e35982a1",
        ],
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_d12f249b80e1a32b7893",
            "cake_muse_spark_norm_61c0e3739fd566fc4823",
        ],
        "gated_residual_rmsnorm_mode0": [
            "cake_muse_spark_norm_e4067dd3b6842fa089a9",
            "cake_muse_spark_norm_1cd221dc235a25c6ef14",
            "cake_muse_spark_norm_95298d67edf21a7f0c54",
            "cake_muse_spark_norm_96fc60d0b38e67b4085a",
            "cake_muse_spark_norm_65d9966b792f76934fe2",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_5e9ca5bb1e4471cce3ff",
            "cake_muse_spark_norm_688a50645a4e00669789",
            "cake_muse_spark_norm_3f8b7179529819e938b5",
            "cake_muse_spark_norm_bd0b88e124a14b9ebfad",
            "cake_muse_spark_norm_96ba3336cf97f5be5841",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_6a387e4b7299123352f4",
            "cake_muse_spark_norm_4725afe7cdd93a93ff86",
            "cake_muse_spark_norm_c6478d985959ea0426a6",
            "cake_muse_spark_norm_584407b9521564b8200f",
            "cake_muse_spark_norm_c9753c6cab93c6f53ea2",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_ad3d24641a423a260dea",
            "cake_muse_spark_norm_01c5629ce7bb10cbedd5",
            "cake_muse_spark_norm_ffbfde93a6d276795e1d",
            "cake_muse_spark_norm_b886dd6347cac2b58cf3",
            "cake_muse_spark_norm_4e76b0cecea57bf56f1f",
        ],
        "rmsnorm_sigmoid_gate_bf16": [
            "cake_muse_spark_norm_ba936288601d077849a2",
            "cake_muse_spark_norm_73cdd3ec3d059d8e95a9",
            "cake_muse_spark_norm_20723bd85c3bbf3e0ba1",
            "cake_muse_spark_norm_a37e58e6a7d00cf8c622",
            "cake_muse_spark_norm_c7a321ceb4da94672252",
            "cake_muse_spark_norm_b246ba16efbd44a244f6",
            "cake_muse_spark_norm_2206e897092d7ba2b069",
            "cake_muse_spark_norm_5cf73e7885cf18f1c89a",
            "cake_muse_spark_norm_a34f5c08681c31eddf44",
            "cake_muse_spark_norm_455dd018e9c2b6f6ffeb",
            "cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
            "cake_muse_spark_norm_52fa7700c74897e3d56b",
            "cake_muse_spark_norm_25a7f9c3f3bc6d00fda4",
        ],
        "rmsnorm_sigmoid_gate_bf16_pdl": [
            "cake_muse_spark_norm_0e50091274e6013faa20",
            "cake_muse_spark_norm_402f58bba74e553f32f7",
            "cake_muse_spark_norm_37ea5413e1d3e395761f",
            "cake_muse_spark_norm_8e6e989e8e2bab5d8163",
            "cake_muse_spark_norm_f5cfa44f264f852f4db4",
            "cake_muse_spark_norm_0b979435de8b4c05214f",
            "cake_muse_spark_norm_b246ba16efbd44a244f6",
            "cake_muse_spark_norm_1d29611fac2391f74182",
            "cake_muse_spark_norm_a9ffd668e885ed4c7e0b",
            "cake_muse_spark_norm_4dffe3ae22bb749cbf54",
            "cake_muse_spark_norm_b61c2310cd41e8e79bf8",
            "cake_muse_spark_norm_0b9a1a6c1d1c3c614200",
            "cake_muse_spark_norm_52fa7700c74897e3d56b",
            "cake_muse_spark_norm_eb8b5c0195ced737054a",
            "cake_muse_spark_norm_25a7f9c3f3bc6d00fda4",
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
