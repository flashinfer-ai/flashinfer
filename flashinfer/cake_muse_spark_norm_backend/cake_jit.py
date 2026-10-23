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
branch and FP32 residual stream; SM100 / SM103.

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
``None`` on every record).  Batch size is a runtime scalar, so a changed row
count never recompiles anything.
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_100d7915374905e86ae7": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_100d7915374905e86ae7",
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
        "library": "cake_muse_spark_norm_100d7915374905e86ae7",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_2cf1350fe2bb3d8681d4": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_2cf1350fe2bb3d8681d4",
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
        "library": "cake_muse_spark_norm_2cf1350fe2bb3d8681d4",
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
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_57bd19b170e59eb603c5": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_57bd19b170e59eb603c5",
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
        "library": "cake_muse_spark_norm_57bd19b170e59eb603c5",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_5ce247436973b698ed3f": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_5ce247436973b698ed3f",
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
        "library": "cake_muse_spark_norm_5ce247436973b698ed3f",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_604aad8c803457de2cd4": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_604aad8c803457de2cd4",
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
        "library": "cake_muse_spark_norm_604aad8c803457de2cd4",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_66e57886d91745d0a431": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 4,
            "kind": "rows",
            "row_ctas": 4,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_66e57886d91745d0a431",
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
        "library": "cake_muse_spark_norm_66e57886d91745d0a431",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_6d2c8cb6101dd594d6e5": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_6d2c8cb6101dd594d6e5",
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
        "library": "cake_muse_spark_norm_6d2c8cb6101dd594d6e5",
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
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_a18848593d176466a0ca": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_a18848593d176466a0ca",
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
        "library": "cake_muse_spark_norm_a18848593d176466a0ca",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_b44f9cdb8275ad901549": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_b44f9cdb8275ad901549",
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
        "library": "cake_muse_spark_norm_b44f9cdb8275ad901549",
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
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        128,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_e460642efc266713e3d7": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_e460642efc266713e3d7",
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
        "library": "cake_muse_spark_norm_e460642efc266713e3d7",
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
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_fac9b431cbb084649adf": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_fac9b431cbb084649adf",
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
        "library": "cake_muse_spark_norm_fac9b431cbb084649adf",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_ff09b12ae44b206b07fa": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 2,
            "kind": "rows",
            "row_ctas": 8,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_ff09b12ae44b206b07fa",
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
        "library": "cake_muse_spark_norm_ff09b12ae44b206b07fa",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_9acc360b1e80436d7585": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_9acc360b1e80436d7585",
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
        "library": "cake_muse_spark_norm_9acc360b1e80436d7585",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_bdbfe170063c59ea80d2": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_bdbfe170063c59ea80d2",
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
        "library": "cake_muse_spark_norm_bdbfe170063c59ea80d2",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_e07f2a9ea7a5b5585a04": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_e07f2a9ea7a5b5585a04",
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
        "library": "cake_muse_spark_norm_e07f2a9ea7a5b5585a04",
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
            },
        },
        "workspace_rule": None,
    },
    "cake_muse_spark_norm_f8cdabb43c1cb144f2dd": {
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
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "rows",
            "row_ctas": 1,
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_f8cdabb43c1cb144f2dd",
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
        "library": "cake_muse_spark_norm_f8cdabb43c1cb144f2dd",
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
            "cake_muse_spark_norm_fac9b431cbb084649adf",
            "cake_muse_spark_norm_66e57886d91745d0a431",
            "cake_muse_spark_norm_e07f2a9ea7a5b5585a04",
            "cake_muse_spark_norm_2cf1350fe2bb3d8681d4",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_57bd19b170e59eb603c5",
            "cake_muse_spark_norm_5ce247436973b698ed3f",
            "cake_muse_spark_norm_bdbfe170063c59ea80d2",
            "cake_muse_spark_norm_6d2c8cb6101dd594d6e5",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_ff09b12ae44b206b07fa",
            "cake_muse_spark_norm_100d7915374905e86ae7",
            "cake_muse_spark_norm_f8cdabb43c1cb144f2dd",
            "cake_muse_spark_norm_b44f9cdb8275ad901549",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_a18848593d176466a0ca",
            "cake_muse_spark_norm_604aad8c803457de2cd4",
            "cake_muse_spark_norm_9acc360b1e80436d7585",
            "cake_muse_spark_norm_e460642efc266713e3d7",
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
            "cake_muse_spark_norm_fac9b431cbb084649adf",
            "cake_muse_spark_norm_66e57886d91745d0a431",
            "cake_muse_spark_norm_e07f2a9ea7a5b5585a04",
            "cake_muse_spark_norm_2cf1350fe2bb3d8681d4",
        ],
        "gated_residual_rmsnorm_mode0_pdl": [
            "cake_muse_spark_norm_57bd19b170e59eb603c5",
            "cake_muse_spark_norm_5ce247436973b698ed3f",
            "cake_muse_spark_norm_bdbfe170063c59ea80d2",
            "cake_muse_spark_norm_6d2c8cb6101dd594d6e5",
        ],
        "gated_residual_rmsnorm_mode1": [
            "cake_muse_spark_norm_ff09b12ae44b206b07fa",
            "cake_muse_spark_norm_100d7915374905e86ae7",
            "cake_muse_spark_norm_f8cdabb43c1cb144f2dd",
            "cake_muse_spark_norm_b44f9cdb8275ad901549",
        ],
        "gated_residual_rmsnorm_mode1_pdl": [
            "cake_muse_spark_norm_a18848593d176466a0ca",
            "cake_muse_spark_norm_604aad8c803457de2cd4",
            "cake_muse_spark_norm_9acc360b1e80436d7585",
            "cake_muse_spark_norm_e460642efc266713e3d7",
        ],
    },
}

# One verified build per architecture: B200 / GB200 (``sm_100a``) and B300 /
# GB300 (``sm_103a``); ``sm_107a`` is an exported target of the family with no
# routed program yet.  Every architecture is compiled with its exact
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
OPS = ("gated_residual_next_rmsnorm", "gated_residual_rmsnorm")
VARIANTS: dict[str, tuple[str, ...]] = {
    "gated_residual_next_rmsnorm": (
        "alpha_tensor_pw",
        "alpha_scalar_nopw",
        "alpha_scalar_pw",
        "alpha_tensor_nopw",
    ),
    "gated_residual_rmsnorm": ("mode0", "mode0_pdl", "mode1", "mode1_pdl"),
}
SELECT_KIND = "rows_between"
GRID_KINDS = ("rows", "persistent_rows")

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


def admits(rule: dict[str, Any] | None, rows: int, arch: str) -> bool:
    """The record's selection rule: ``None`` admits every call; ``rows_between`` admits ``rows`` on ``arch``
    when one of the architecture's windows holds ``above < rows <= at_most`` (``at_most`` ``None`` = unbounded)."""
    if rule is None:
        return True
    if rule.get("kind") != SELECT_KIND:
        raise ValueError(f"unknown Cake norm select rule {rule.get('kind')!r}")
    windows = rule["windows"].get(arch, ())
    return any(
        rows > int(above) and (at_most is None or rows <= int(at_most))
        for above, at_most in windows
    )


def selected_module(arch: str, route: str, rows: int) -> str:
    """The record of ``route`` whose window admits ``rows`` on ``arch``."""
    if rows <= 0:
        raise ValueError(f"Cake norm launches need a positive row count, got {rows}")
    for name in route_modules(arch, route):
        if admits(MODULES[name].get("select_rule"), rows, arch):
            return name
    raise NotImplementedError(
        f"no exported program of route {route!r} on {arch} admits {rows} rows"
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


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one record under its recorded grid rule: ``rows x row_ctas`` CTAs (``row_ctas`` the
    rule's CTAs per row, the cluster when it names none), or ``min(rows, co-resident clusters) x cluster``
    CTAs for the persistent row loop."""
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
    grid = launch_grid(record, rows, device_index=device_index)
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
]
