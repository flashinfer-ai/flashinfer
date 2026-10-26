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

Host plan and loader glue of the Cake experimental NVFP4 sparse MLA decode
programs (SM100 / SM103), the ``backend="cake"`` of
:func:`flashinfer.mla.nvfp4_sparse_mla_decode`.

Contract of a program: ``out[t, h] = out_scale * softmax_k(qk_scale *
q[t, h] . K[indices[t, k]]) @ V[indices[t, k]]`` over E4M3 ``[T, 16, 576]``
queries (``q``, handed to the kernel as int32 words), 352-byte NVFP4 cache
rows (``kv``, the row pool as int32 words, and ``kv_rows``, the same pool as
``[rows, 352]`` bytes the kernel's TMA gather map is encoded from), int32
``[T, topk]`` row ids with ``-1`` marking an empty slot (``indices``) and a
BF16 ``[T, 16, 512]`` output (``out``); ``qk_scale`` is the base-2 softmax
scale (``bmm1_scale * log2(e)``), ``out_scale`` the output scale and
``keys_per_cta`` the keys each CTA of the token's cluster owns
(``ceil(topk / C)``).  One thread-block cluster of ``C`` CTAs serves one query
token (grid ``(T, C, 1)``, cluster ``(1, C, 1)``, 512 threads): CTA ``c``
takes the keys ``[c * keys_per_cta, (c + 1) * keys_per_cta)`` in 128-key
tiles on the tensor core (tcgen05, TMEM) and either writes the normalized
output directly (``C = 1``) or merges its fp32 partials with the cluster
peers through distributed shared memory.  A program is built for one cluster
size ``C`` of :data:`CLUSTER_SIZES` and one split bound ``max_keys`` of
:data:`MAX_KEYS`: a launch whose ``keys_per_cta`` is at most 64, 128, 192 or
256 runs the program of that bound (:func:`program_max_keys`; 0 is the
general program, any split); :mod:`.cake_backend` plans ``C`` per launch.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit, a tvm-ffi binding and an occupancy query unit,
compiled once per architecture.  A record carries the host plan only: the
program's ``library`` stem (its ``build.json`` entries are
``<library>_<arch>``), the architectures it is delivered for, the source
``module`` and ``build`` keywords it was rendered from, its launch record,
the argument plan of its ``run`` entry, the kernel-parameter -> logical-
argument map, the cluster size and split bound of the host planner
(``cluster``, ``max_keys``), the grid rule :func:`launch_grid` evaluates,
the name of the occupancy entry (``occupancy_entry``:
``max_active_clusters(device, block_x, block_y, block_z, cluster_x,
cluster_y, cluster_z, dynamic_smem_bytes, cooperative)``, the one-wave
cluster capacity the planner asks for) and the workspace rule (``None``: the
kernel owns no scratch).  What to compile -- sources, digests, nvcc flags --
is recorded once in ``build.json`` next to this file; :mod:`.cake_build`
serves the library from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags.
Every ``build.json`` entry records ``--use_fast_math`` (the flag the kernel
was validated and measured with); the loader passes ``use_fast_math=False``
so that the manifest, not the loader, states it.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_nvfp4_sparse_mla_decode_<20 hex>`` is the first
twenty hex digits of the SHA-256 of the compact JSON ``{"build": <build
keywords, sorted, every default spelled>, "module": <dotted source module>}``
of the kernel the program was rendered from; a changed keyword is a new
program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# Two delivered architectures: B200 / GB200 (``sm_100a``) and B300 / GB300
# (``sm_103a``), each compiled with its exact ``-gencode`` flag set
# (``build.json`` ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# Thread-block cluster sizes (CTAs per query token) the programs are delivered
# for; 7 is not a variant of the kernel (see ``cake_backend.PLAN_CTAS``).
CLUSTER_SIZES = (1, 2, 3, 4, 5, 6, 8)
# Split bounds of the programs: a launch whose CTAs own at most 64, 128, 192 or
# 256 keys runs the program built for that bound (its tile count is a
# constant), a larger split the general program (0).  One record per (cluster
# size, split bound).
MAX_KEYS = (0, 64, 128, 192, 256)

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "cake_nvfp4_sparse_mla_decode_9627a6efb66a046f8cf2": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 1,
            "debug": 0,
            "max_keys": 0,
        },
        "cluster": 1,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_9627a6efb66a046f8cf2",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_9627a6efb66a046f8cf2",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 218880,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_9627a6efb66a046f8cf2",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 0,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_614e9562bde9684ffb9a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 1,
            "debug": 0,
            "max_keys": 64,
        },
        "cluster": 1,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_614e9562bde9684ffb9a",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_614e9562bde9684ffb9a",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 96000,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_614e9562bde9684ffb9a",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 64,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_803702f746170d774fb5": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 1,
            "debug": 0,
            "max_keys": 128,
        },
        "cluster": 1,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_803702f746170d774fb5",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_803702f746170d774fb5",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 96000,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_803702f746170d774fb5",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 128,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_f20b02d097659bdc47bd": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 1,
            "debug": 0,
            "max_keys": 192,
        },
        "cluster": 1,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_f20b02d097659bdc47bd",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_f20b02d097659bdc47bd",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 218880,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_f20b02d097659bdc47bd",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 192,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_aa370e07497081bfeea1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 1,
            "debug": 0,
            "max_keys": 256,
        },
        "cluster": 1,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_aa370e07497081bfeea1",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_aa370e07497081bfeea1",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 218880,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_aa370e07497081bfeea1",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 256,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_02da9b5a54d5a66aafb9": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 2,
            "debug": 0,
            "max_keys": 0,
        },
        "cluster": 2,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_02da9b5a54d5a66aafb9",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_02da9b5a54d5a66aafb9",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 219264,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_02da9b5a54d5a66aafb9",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 0,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_1fd3c893bae4ad90f276": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 2,
            "debug": 0,
            "max_keys": 64,
        },
        "cluster": 2,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_1fd3c893bae4ad90f276",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_1fd3c893bae4ad90f276",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 129152,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_1fd3c893bae4ad90f276",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 64,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_0425e3543bb9c12ba2fc": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 2,
            "debug": 0,
            "max_keys": 128,
        },
        "cluster": 2,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_0425e3543bb9c12ba2fc",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_0425e3543bb9c12ba2fc",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 129152,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_0425e3543bb9c12ba2fc",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 128,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_44f4999c79cea6a046fc": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 2,
            "debug": 0,
            "max_keys": 192,
        },
        "cluster": 2,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_44f4999c79cea6a046fc",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_44f4999c79cea6a046fc",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 219264,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_44f4999c79cea6a046fc",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 192,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_cfb063781561be8fdba1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 2,
            "debug": 0,
            "max_keys": 256,
        },
        "cluster": 2,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_cfb063781561be8fdba1",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_cfb063781561be8fdba1",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 219264,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_cfb063781561be8fdba1",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 256,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_0631dd655f25a545f52f": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 3,
            "debug": 0,
            "max_keys": 0,
        },
        "cluster": 3,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_0631dd655f25a545f52f",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_0631dd655f25a545f52f",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 3, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220416,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_0631dd655f25a545f52f",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 0,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_b58aec4c312bfac8960e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 3,
            "debug": 0,
            "max_keys": 64,
        },
        "cluster": 3,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_b58aec4c312bfac8960e",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_b58aec4c312bfac8960e",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 3, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 130304,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_b58aec4c312bfac8960e",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 64,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_fb0887ac7c6ddde37463": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 3,
            "debug": 0,
            "max_keys": 128,
        },
        "cluster": 3,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_fb0887ac7c6ddde37463",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_fb0887ac7c6ddde37463",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 3, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 130304,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_fb0887ac7c6ddde37463",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 128,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_30edcc16ece68f64276b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 3,
            "debug": 0,
            "max_keys": 192,
        },
        "cluster": 3,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_30edcc16ece68f64276b",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_30edcc16ece68f64276b",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 3, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220416,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_30edcc16ece68f64276b",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 192,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_d46057a4092ef860ca92": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 3,
            "debug": 0,
            "max_keys": 256,
        },
        "cluster": 3,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_d46057a4092ef860ca92",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_d46057a4092ef860ca92",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 3, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220416,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_d46057a4092ef860ca92",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 256,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_2beb3ac65e8202fb7fef": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 4,
            "debug": 0,
            "max_keys": 0,
        },
        "cluster": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_2beb3ac65e8202fb7fef",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_2beb3ac65e8202fb7fef",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 4, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 219520,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_2beb3ac65e8202fb7fef",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 0,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_49b00dcecc9e9934529f": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 4,
            "debug": 0,
            "max_keys": 64,
        },
        "cluster": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_49b00dcecc9e9934529f",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_49b00dcecc9e9934529f",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 4, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 129408,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_49b00dcecc9e9934529f",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 64,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_d542b3efeb634430ee60": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 4,
            "debug": 0,
            "max_keys": 128,
        },
        "cluster": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_d542b3efeb634430ee60",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_d542b3efeb634430ee60",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 4, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 129408,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_d542b3efeb634430ee60",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 128,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_7a101562cfca6ea9996c": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 4,
            "debug": 0,
            "max_keys": 192,
        },
        "cluster": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_7a101562cfca6ea9996c",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_7a101562cfca6ea9996c",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 4, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 219520,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_7a101562cfca6ea9996c",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 192,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_1f1a7edd888d8a0da211": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 4,
            "debug": 0,
            "max_keys": 256,
        },
        "cluster": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_1f1a7edd888d8a0da211",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_1f1a7edd888d8a0da211",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 4, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 219520,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_1f1a7edd888d8a0da211",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 256,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_7ca1b6bf3a40d094efa6": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 5,
            "debug": 0,
            "max_keys": 0,
        },
        "cluster": 5,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_7ca1b6bf3a40d094efa6",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_7ca1b6bf3a40d094efa6",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 5, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220160,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_7ca1b6bf3a40d094efa6",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 0,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_fa8ce8d9ceb0162c5ffa": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 5,
            "debug": 0,
            "max_keys": 64,
        },
        "cluster": 5,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_fa8ce8d9ceb0162c5ffa",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_fa8ce8d9ceb0162c5ffa",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 5, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 130048,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_fa8ce8d9ceb0162c5ffa",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 64,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_b0504562acde0b04a18d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 5,
            "debug": 0,
            "max_keys": 128,
        },
        "cluster": 5,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_b0504562acde0b04a18d",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_b0504562acde0b04a18d",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 5, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 130048,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_b0504562acde0b04a18d",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 128,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_3a530e125689f2e26d2f": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 5,
            "debug": 0,
            "max_keys": 192,
        },
        "cluster": 5,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_3a530e125689f2e26d2f",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_3a530e125689f2e26d2f",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 5, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220160,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_3a530e125689f2e26d2f",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 192,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_031e5e115fe4406c1991": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 5,
            "debug": 0,
            "max_keys": 256,
        },
        "cluster": 5,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_031e5e115fe4406c1991",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_031e5e115fe4406c1991",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 5, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220160,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_031e5e115fe4406c1991",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 256,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_06aeadf2d8ac7424c83e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 6,
            "debug": 0,
            "max_keys": 0,
        },
        "cluster": 6,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_06aeadf2d8ac7424c83e",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_06aeadf2d8ac7424c83e",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 6, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220800,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_06aeadf2d8ac7424c83e",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 0,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_eb71b60ae1631b05e289": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 6,
            "debug": 0,
            "max_keys": 64,
        },
        "cluster": 6,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_eb71b60ae1631b05e289",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_eb71b60ae1631b05e289",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 6, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 130688,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_eb71b60ae1631b05e289",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 64,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_043cd6a507373b4c1379": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 6,
            "debug": 0,
            "max_keys": 128,
        },
        "cluster": 6,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_043cd6a507373b4c1379",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_043cd6a507373b4c1379",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 6, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 130688,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_043cd6a507373b4c1379",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 128,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_952b02dc16cdc94d4c96": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 6,
            "debug": 0,
            "max_keys": 192,
        },
        "cluster": 6,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_952b02dc16cdc94d4c96",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_952b02dc16cdc94d4c96",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 6, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220800,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_952b02dc16cdc94d4c96",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 192,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_165f65c46b5d41cdab69": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 6,
            "debug": 0,
            "max_keys": 256,
        },
        "cluster": 6,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_165f65c46b5d41cdab69",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_165f65c46b5d41cdab69",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 6, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220800,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_165f65c46b5d41cdab69",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 256,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_2d84c2577cbebd591cd9": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 8,
            "debug": 0,
            "max_keys": 0,
        },
        "cluster": 8,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_2d84c2577cbebd591cd9",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_2d84c2577cbebd591cd9",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 8, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220032,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_2d84c2577cbebd591cd9",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 0,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_b65bf21f01d35c10e07e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 8,
            "debug": 0,
            "max_keys": 64,
        },
        "cluster": 8,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_b65bf21f01d35c10e07e",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_b65bf21f01d35c10e07e",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 8, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 129920,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_b65bf21f01d35c10e07e",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 64,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_801249552261fb7e5d8c": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 8,
            "debug": 0,
            "max_keys": 128,
        },
        "cluster": 8,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_801249552261fb7e5d8c",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_801249552261fb7e5d8c",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 8, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 129920,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_801249552261fb7e5d8c",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 128,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_f139efb8395e9b8eb339": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 8,
            "debug": 0,
            "max_keys": 192,
        },
        "cluster": 8,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_f139efb8395e9b8eb339",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_f139efb8395e9b8eb339",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 8, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220032,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_f139efb8395e9b8eb339",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 192,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_abfc3b744b2207189ce2": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "kv_nope"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 8,
            "debug": 0,
            "max_keys": 256,
        },
        "cluster": 8,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_abfc3b744b2207189ce2",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_abfc3b744b2207189ce2",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 8, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 220032,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_abfc3b744b2207189ce2",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "kv_nope": "kv_rows",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys": 256,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode_tc",
        "occupancy_entry": "max_active_clusters",
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
            f"Cake NVFP4 sparse MLA decode target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def program_max_keys(keys_per_cta: int) -> int:
    """The split bound of the program a launch of ``keys_per_cta`` keys per CTA
    runs: the smallest of :data:`MAX_KEYS` holding it, 0 (the general program)
    past 256."""
    kpc = int(keys_per_cta)
    return next((bound for bound in MAX_KEYS[1:] if kpc <= bound), 0)


def record_for(cluster: int, max_keys: int) -> str:
    """The name of the one record of ``cluster`` CTAs per token built for splits
    of at most ``max_keys`` keys (``RuntimeError`` when none or more than one
    is: the route has one program per cluster size and split bound)."""
    names = [
        name
        for name, record in MODULES.items()
        if int(record["cluster"]) == cluster and int(record["max_keys"]) == max_keys
    ]
    if len(names) != 1:
        raise RuntimeError(
            f"Cake NVFP4 sparse MLA decode delivers {len(names)} programs of cluster "
            f"size {cluster} and split bound {max_keys}; the delivered cluster sizes "
            f"are {CLUSTER_SIZES}, the split bounds {MAX_KEYS}"
        )
    return names[0]


def registered_programs(arch: str) -> set[tuple[int, int]]:
    """``(cluster size, split bound)`` of every program delivered for ``arch``."""
    return {
        (int(record["cluster"]), int(record["max_keys"]))
        for record in MODULES.values()
        if arch in record["arches"]
    }


def registered_clusters(arch: str) -> set[int]:
    """Cluster sizes whose programs are delivered for ``arch``, every split bound
    of the size included."""
    programs = registered_programs(arch)
    return {
        cluster
        for cluster in CLUSTER_SIZES
        if all((cluster, bound) in programs for bound in MAX_KEYS)
    }


def select_module(arch: str, cluster: int, keys_per_cta: int) -> str:
    """The record name of the program of ``cluster`` CTAs per token that serves
    ``keys_per_cta`` keys per CTA on ``arch`` (``NotImplementedError`` when this
    checkout delivers none)."""
    key = (int(cluster), program_max_keys(keys_per_cta))
    if key not in registered_programs(arch):
        raise NotImplementedError(
            f"The generated NVFP4 sparse MLA decode program (cluster size {key[0]}, "
            f"split bound {key[1]}) is not delivered for {arch} in this checkout"
        )
    return record_for(*key)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake NVFP4 sparse MLA decode record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch`` (its ``run``
    and ``max_active_clusters`` entries): the prebuilt ``aot/`` library when a
    builder compiled it, else FlashInfer's JIT build of the same sources and
    flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


def launch_grid(record: dict[str, Any], *, num_tokens: int) -> tuple[int, int, int]:
    """Launch grid of one record for ``num_tokens`` query tokens under its
    ``tokens_by_cluster`` rule: one cluster of the record's ``cluster`` CTAs per
    token, the cluster along the grid's y axis."""
    rule = record["grid_rule"]
    if rule["kind"] != "tokens_by_cluster":
        raise ValueError(
            f"unknown Cake NVFP4 sparse MLA decode grid rule: {rule['kind']!r}"
        )
    num_tokens, cluster = int(num_tokens), int(record["cluster"])
    if num_tokens <= 0:
        raise ValueError(
            f"the grid rule needs a positive token count, got num_tokens={num_tokens}"
        )
    return (num_tokens, cluster, 1)


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
    "CLUSTER_SIZES",
    "MAX_KEYS",
    "MODULES",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "program_max_keys",
    "record_for",
    "registered_clusters",
    "registered_programs",
    "run_module",
    "select_module",
    "sm_count",
    "specs_for_arch",
]
