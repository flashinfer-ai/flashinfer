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

JIT loader of the Cake fused row-wise FP8 GEMM + all-reduce.

One generated program per (architecture, carrier, capacity variant): a
persistent GEMM whose epilogue stores each owned bf16 partial tile into this
rank's output and publishes every other partial tile into the chunk owner's
symmetric receive slot, whose reducer warps sum the partials in fixed rank
order and broadcast the result (unicast ``p2p`` carrier, or the NVLink
multicast ``nvls`` carrier), and whose drainer warps pull the non-owned chunks.
``PROGRAMS`` records each program's device and binding translation units,
launch block, cluster shape, dynamic shared memory and the positional argument
names of its ``run`` launcher; ``ROUTES[arch]`` maps the stage key
``main_ws{world_size}_{comm_mode}{variant}`` to it. A program is compiled once
for its exact architecture together with its thin tvm-ffi launcher.
``COMM_MODES`` / ``FROZEN`` / ``GEOMETRY`` / ``CAPACITY_VARIANTS`` record the
carrier per world size, the trace-time choices frozen into the generated
programs (never knobs) and the per-architecture geometry the backend sizes its
workspace and launch with. ``LEVER_DISPATCH`` records, per architecture, the
per-call choice among the programs of one route: a call whose shape meets a
lever's condition runs the program of the route key extended by the lever's
suffix (``call_program``); every such program computes the same result as the
route's program and only schedules it differently. The tables are filled by the
exporter from the program bundle; the loader never hashes sources or interprets
argument plans.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import torch

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm90a_nvcc_flags,
)

# Filled mechanically from the program bundle.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_fp8_rowwise_gemm_allreduce_0882885198687141cd4a": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_0882885198687141cd4a_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_0882885198687141cd4a_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 213088,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_14343fabdff465e3e64a": {
        "family": "blackwell",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_14343fabdff465e3e64a_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_14343fabdff465e3e64a_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [384, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 197632,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "pair_tiles",
            "K_tiles",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_32edd0405014d238b067": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_32edd0405014d238b067_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_32edd0405014d238b067_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 213088,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_39c6c11b20cb6d7d4baa": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_39c6c11b20cb6d7d4baa_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_39c6c11b20cb6d7d4baa_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 213088,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_43c80d0e40bf111ee438": {
        "family": "blackwell",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_43c80d0e40bf111ee438_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_43c80d0e40bf111ee438_binding.cu",
        ],
        "arches": ["sm_100a"],
        "block": [384, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 222208,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "pair_tiles",
            "K_tiles",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_4e73c78f58b60636f408": {
        "family": "blackwell",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_4e73c78f58b60636f408_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_4e73c78f58b60636f408_binding.cu",
        ],
        "arches": ["sm_103a"],
        "block": [384, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 222208,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "pair_tiles",
            "K_tiles",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_82fa3df922ed493ea553": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_82fa3df922ed493ea553_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_82fa3df922ed493ea553_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 225376,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_92002e9c22c1d8ea9829": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_92002e9c22c1d8ea9829_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_92002e9c22c1d8ea9829_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 225376,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_a2589cb399561c4c5a54": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_a2589cb399561c4c5a54_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_a2589cb399561c4c5a54_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 225376,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_cdaec3a4bab3b1b1d6d6": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_cdaec3a4bab3b1b1d6d6_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_cdaec3a4bab3b1b1d6d6_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 225376,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_d2557ff50c504747e37d": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_d2557ff50c504747e37d_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_d2557ff50c504747e37d_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 213088,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
    "cake_fp8_rowwise_gemm_allreduce_d94d1f7305feac3269dc": {
        "family": "hopper",
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_d94d1f7305feac3269dc_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/hopper/cake_fp8_rowwise_gemm_allreduce_d94d1f7305feac3269dc_binding.cu",
        ],
        "arches": ["sm_90a"],
        "block": [512, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 213088,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "tiles",
            "K_tiles",
            "group_m",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
}
ROUTES: dict[str, dict[str, str]] = {
    "sm_90a": {
        "main_ws2_p2p": "cake_fp8_rowwise_gemm_allreduce_32edd0405014d238b067",
        "main_ws2_p2p_ls2": "cake_fp8_rowwise_gemm_allreduce_d94d1f7305feac3269dc",
        "main_ws2_p2p_r6": "cake_fp8_rowwise_gemm_allreduce_82fa3df922ed493ea553",
        "main_ws2_p2p_r6_ls2": "cake_fp8_rowwise_gemm_allreduce_92002e9c22c1d8ea9829",
        "main_ws4_p2p": "cake_fp8_rowwise_gemm_allreduce_32edd0405014d238b067",
        "main_ws4_p2p_dp": "cake_fp8_rowwise_gemm_allreduce_d2557ff50c504747e37d",
        "main_ws4_p2p_ls2": "cake_fp8_rowwise_gemm_allreduce_d94d1f7305feac3269dc",
        "main_ws4_p2p_ls2_dp": "cake_fp8_rowwise_gemm_allreduce_0882885198687141cd4a",
        "main_ws4_p2p_r6": "cake_fp8_rowwise_gemm_allreduce_82fa3df922ed493ea553",
        "main_ws4_p2p_r6_dp": "cake_fp8_rowwise_gemm_allreduce_a2589cb399561c4c5a54",
        "main_ws8_p2p": "cake_fp8_rowwise_gemm_allreduce_32edd0405014d238b067",
        "main_ws8_p2p_dp": "cake_fp8_rowwise_gemm_allreduce_d2557ff50c504747e37d",
        "main_ws8_p2p_g4": "cake_fp8_rowwise_gemm_allreduce_39c6c11b20cb6d7d4baa",
        "main_ws8_p2p_g4_r6": "cake_fp8_rowwise_gemm_allreduce_cdaec3a4bab3b1b1d6d6",
        "main_ws8_p2p_ls2": "cake_fp8_rowwise_gemm_allreduce_d94d1f7305feac3269dc",
        "main_ws8_p2p_ls2_dp": "cake_fp8_rowwise_gemm_allreduce_0882885198687141cd4a",
        "main_ws8_p2p_r6": "cake_fp8_rowwise_gemm_allreduce_82fa3df922ed493ea553",
        "main_ws8_p2p_r6_dp": "cake_fp8_rowwise_gemm_allreduce_a2589cb399561c4c5a54",
    },
    "sm_100a": {
        "main_ws2_p2p": "cake_fp8_rowwise_gemm_allreduce_43c80d0e40bf111ee438",
        "main_ws4_p2p": "cake_fp8_rowwise_gemm_allreduce_43c80d0e40bf111ee438",
        "main_ws8_nvls": "cake_fp8_rowwise_gemm_allreduce_14343fabdff465e3e64a",
    },
    "sm_103a": {
        "main_ws2_p2p": "cake_fp8_rowwise_gemm_allreduce_4e73c78f58b60636f408",
        "main_ws4_p2p": "cake_fp8_rowwise_gemm_allreduce_4e73c78f58b60636f408",
        "main_ws8_nvls": "cake_fp8_rowwise_gemm_allreduce_14343fabdff465e3e64a",
    },
}
COMPILE_FLAGS: list[str] = []
# Per architecture: the exchange carrier of each delivered world size.
COMM_MODES: dict[str, dict[int, str]] = {
    "sm_90a": {2: "p2p", 4: "p2p", 8: "p2p"},
    "sm_100a": {2: "p2p", 4: "p2p", 8: "nvls"},
    "sm_103a": {2: "p2p", 4: "p2p", 8: "nvls"},
}
# Frozen trace-time choices per architecture and carrier (documentation, never knobs).
FROZEN: dict[str, dict[str, dict[str, Any]]] = {
    "sm_90a": {
        "p2p": {
            "bc_release": "release",
            "chunk_groups": 8,
            "layout": "wg4",
            "num_stages": 5,
            "promote_k32_steps": 4,
            "timeout_cycles_log2": 34,
            "transport": "pull",
            "wgmma_inflight_groups": 2,
        },
    },
    "sm_100a": {
        "nvls": {"g1_sched": "static", "num_stages": 4, "timeout_cycles_log2": 34},
        "p2p": {
            "chunk_groups": 8,
            "num_stages": 4,
            "p2p_reduce_wave": 2,
            "timeout_cycles_log2": 34,
            "transport": "pull",
        },
    },
    "sm_103a": {
        "nvls": {"g1_sched": "static", "num_stages": 4, "timeout_cycles_log2": 34},
        "p2p": {
            "chunk_groups": 8,
            "num_stages": 4,
            "p2p_reduce_wave": 4,
            "timeout_cycles_log2": 34,
            "transport": "pull",
        },
    },
}
# Tile geometry, control-word layout and grid rule per architecture (mirrors the Cake export adapter).
GEOMETRY: dict[str, dict[str, Any]] = {
    "sm_90a": {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "CHUNK_ROWS": 128,
        "CLUSTER_DIMS": [1, 1, 1],
        "OUT_BYTES": 2,
        "TILE_BYTES": 32768,
        "FLAG_WORD_BYTES_PER_CHUNK": 8,
        "MAX_CTAS": 1024,
        "CTL_EXTRA_WORDS": 2,
        "CTL_STEAL_WORDS": False,
        "CTL_SCHED_WORDS": False,
        "S_ROWS_OWNED_ONLY_UNDER_PULL": True,
        "GRID_RULE": "persistent_sm_count",
        "TILE_ARG": "tiles",
        "MAX_DYNAMIC_SMEM_BYTES": 232448,
        "THREADS": {"p2p": 512, "nvls": 384},
        "RASTER_SM_COUNT": 132,
        "RASTER_L2_BYTES": 52428800,
        "RASTER_GROUP_CHOICES": [1, 2, 4, 8, 16, 32],
        "RASTER_MIN_GRID_N": 16,
        "RASTER_MIN_K": 4096,
        "RASTER_MIN_CHUNKS": 4,
    },
    "sm_100a": {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 128,
        "CHUNK_ROWS": 128,
        "CLUSTER_DIMS": [2, 1, 1],
        "OUT_BYTES": 2,
        "TILE_BYTES": 65536,
        "FLAG_WORD_BYTES_PER_CHUNK": 8,
        "MAX_CTAS": 1024,
        "CTL_EXTRA_WORDS": 2,
        "CTL_STEAL_WORDS": True,
        "CTL_SCHED_WORDS": True,
        "S_ROWS_OWNED_ONLY_UNDER_PULL": False,
        "GRID_RULE": "persistent_cluster_pairs",
        "TILE_ARG": "pair_tiles",
        "MAX_DYNAMIC_SMEM_BYTES": 232448,
        "THREADS": {"p2p": 384, "nvls": 384},
        "PAIR_TILE_M": 256,
    },
    "sm_103a": {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 128,
        "CHUNK_ROWS": 128,
        "CLUSTER_DIMS": [2, 1, 1],
        "OUT_BYTES": 2,
        "TILE_BYTES": 65536,
        "FLAG_WORD_BYTES_PER_CHUNK": 8,
        "MAX_CTAS": 1024,
        "CTL_EXTRA_WORDS": 2,
        "CTL_STEAL_WORDS": True,
        "CTL_SCHED_WORDS": True,
        "S_ROWS_OWNED_ONLY_UNDER_PULL": False,
        "GRID_RULE": "persistent_cluster_pairs",
        "TILE_ARG": "pair_tiles",
        "MAX_DYNAMIC_SMEM_BYTES": 232448,
        "THREADS": {"p2p": 384, "nvls": 384},
        "PAIR_TILE_M": 256,
    },
}
# Per architecture: ``[world_size, rows_capacity, suffix]`` rows naming the program variant a
# prepared capacity of exactly ``rows_capacity`` rows selects (empty when every capacity shares one program).
CAPACITY_VARIANTS: dict[str, list[list[Any]]] = {
    "sm_90a": [[8, 4096, "_g4"]],
    "sm_100a": [],
    "sm_103a": [],
}
# Per architecture: the levers a call may switch on (empty when every route runs one program).
# ``REDUCERS``: route suffix and the largest ``K_l`` per world size of the program with more
# reducer warps (it also needs ``K_l * world_size <= N`` and, for a world size in
# ``mn_limit``, ``M * N`` below that limit). ``LOCKSTEP``: route suffix, the
# smallest raster group and the smallest tile count of the program that throttles its
# producers to the grid's progress, and the program variants (``""``: the default
# capacity) whose calls may take it. ``DRAIN``: route suffix, the world sizes, the largest
# ``M`` and the program variants of the program whose drainer pulls from every peer at
# once (it also needs a grid of at least ``world_size - 1`` CTAs, one per peer). The
# suffixes apply in that order; an absent lever is never selected.
LEVER_DISPATCH: dict[str, dict[str, Any]] = {
    "sm_90a": {
        "REDUCERS": {
            "suffix": "_r6",
            "max_k_l": {2: 4096, 4: 1536, 8: 1536},
            "mn_limit": {2: 1073741824},
        },
        "LOCKSTEP": {
            "suffix": "_ls2",
            "min_raster_group": 2,
            "min_tiles": 12288,
            "variants": [""],
        },
        "DRAIN": {"suffix": "_dp", "worlds": [4, 8], "max_m": 32768, "variants": [""]},
    },
    "sm_100a": {},
    "sm_103a": {},
}
# Exact-architecture compile flags (``flashinfer.jit.core``) and the capability -> architecture table.
ARCH_FLAGS: dict[str, list[str]] = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
CAPABILITY_ARCH: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (2, 4, 8)

SOURCE_PACKAGE = "cake_fp8_rowwise_gemm_allreduce"
# Family directories of the package: their programs and the headers they include.
SOURCE_DIRECTORIES: tuple[str, ...] = ("hopper",)
EXCHANGE_MODE = "X16"
# One chunk is 128 rows on every architecture; the column tile is per architecture (``GEOMETRY[arch]["BLOCK_N"]``).
BLOCK_M = 128
BLOCK_K = 128
OUT_BYTES = 2
MAX_ROWS_TIMES_COLS = 1 << 30
# The column multiple of the operand contract on every architecture (the
# sm_90a tile is 128 wide, but only multiples of 256 are delivered).
N_MULTIPLE = 256
# Reducer/drainer vector indices are 32-bit: 16 * N plus the drainer's
# 8 * 64 * MAX_CTAS stride overshoot must stay below 2**31.
MAX_COLS = (1 << 27) - (1 << 15)
GRID_RULE_CLUSTER_PAIRS = "persistent_cluster_pairs"
GRID_RULE_SM_COUNT = "persistent_sm_count"


class DeviceFacts(NamedTuple):
    capability: tuple[int, int]
    arch: str | None
    sm_count: int


@functools.cache
def device_facts(device_index: int) -> DeviceFacts:
    """Compute capability, exported architecture and SM count, queried once per device."""

    properties = torch.cuda.get_device_properties(device_index)
    capability = (int(properties.major), int(properties.minor))
    return DeviceFacts(
        capability=capability,
        arch=CAPABILITY_ARCH.get(capability),
        sm_count=int(properties.multi_processor_count),
    )


def geometry(arch: str) -> dict[str, Any]:
    """The geometry record of a delivered architecture."""

    if arch not in GEOMETRY:
        raise ValueError(
            f"no fused FP8 GEMM + all-reduce program is delivered for architecture {arch!r} "
            f"(delivered: {sorted(GEOMETRY)})"
        )
    return GEOMETRY[arch]


def block_n(arch: str) -> int:
    return int(geometry(arch)["BLOCK_N"])


def tile_bytes(arch: str) -> int:
    return int(geometry(arch)["TILE_BYTES"])


def ceil128(rows: int) -> int:
    """``rows`` rounded up to the 128-row chunk."""

    rows = int(rows)
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    return (rows + BLOCK_M - 1) // BLOCK_M * BLOCK_M


def chunk_count(m: int) -> int:
    """``C = ceil128(M) / 128`` chunks of 128 rows x N."""

    return ceil128(m) // BLOCK_M


def chunk_capacity(max_rows: int) -> int:
    """``C_cap = ceil128(max_rows) / 128``: sizes the receive workspace and the flag words."""

    return ceil128(max_rows) // BLOCK_M


def grid_n(n: int, arch: str) -> int:
    """Column tiles ``N / BLOCK_N`` of the architecture."""

    n = int(n)
    width = block_n(arch)
    if n <= 0 or n % width:
        raise ValueError(f"N must be a positive multiple of {width}, got {n}")
    return n // width


def tiles(m: int, n: int, arch: str) -> int:
    """Persistent tiles of one call: two-CTA pair tiles ``ceil(M / 256) * grid_n`` on the
    cluster-pair grid, chunk tiles ``C * grid_n`` on the SM-count grid."""

    m = int(m)
    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    record = geometry(arch)
    if record["GRID_RULE"] == GRID_RULE_CLUSTER_PAIRS:
        return -(-m // int(record["PAIR_TILE_M"])) * grid_n(n, arch)
    return chunk_count(m) * grid_n(n, arch)


def comm_mode(world_size: int, arch: str) -> str:
    """The exchange carrier of ``world_size`` on ``arch`` (``p2p`` or ``nvls``)."""

    table = COMM_MODES.get(arch)
    if table is None:
        geometry(arch)
        raise ValueError(f"no carrier table is delivered for {arch!r}")
    world_size = int(world_size)
    if world_size not in table:
        raise ValueError(
            f"no fused FP8 GEMM + all-reduce program is delivered for world size {world_size} "
            f"on {arch} (delivered: {sorted(table)})"
        )
    return str(table[world_size])


def transport(world_size: int, arch: str) -> str | None:
    """The Y-broadcast transport frozen into the p2p program (``None`` on the nvls carrier)."""

    mode = comm_mode(world_size, arch)
    value = FROZEN[arch][mode].get("transport")
    return None if value is None else str(value)


def slots_per_chunk(world_size: int, mode: str) -> int:
    """Slots per chunk in the symmetric workspace.

    ``nvls``: one replica slot. ``p2p``: ``world_size - 1`` sender slots plus the owner's result
    slot at three or more ranks (one slot at two ranks); the result slot is never written by a
    sender, so a peer's next-call partial cannot overwrite a result another peer is still pulling.
    """

    world_size = int(world_size)
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if mode == "nvls":
        return 1
    return 1 if world_size <= 2 else world_size


def s_rows(c_cap: int, world_size: int, arch: str) -> int:
    """Chunk rows of the receive image: owned ordinals only (``ceil(C_cap / world_size)``) when the
    architecture compacts the image under the pull transport, every ordinal otherwise."""

    c_cap = int(c_cap)
    if c_cap <= 0:
        raise ValueError(f"C_cap must be positive, got {c_cap}")
    world_size = int(world_size)
    if (
        geometry(arch)["S_ROWS_OWNED_ONLY_UNDER_PULL"]
        and comm_mode(world_size, arch) == "p2p"
        and transport(world_size, arch) == "pull"
    ):
        return (c_cap + world_size - 1) // world_size
    return c_cap


def workspace_bytes(max_rows: int, n: int, world_size: int, arch: str) -> int:
    """Symmetric receive workspace per rank: ``s_rows x slots x grid_n x TILE_BYTES``."""

    c_cap = chunk_capacity(max_rows)
    return (
        s_rows(c_cap, world_size, arch)
        * slots_per_chunk(world_size, comm_mode(world_size, arch))
        * grid_n(n, arch)
        * tile_bytes(arch)
    )


def flag_words(max_rows: int) -> int:
    """Readiness words of one rank: ``2 * C_cap`` uint32."""

    return 2 * chunk_capacity(max_rows)


def flag_bytes(max_rows: int, arch: str) -> int:
    return int(geometry(arch)["FLAG_WORD_BYTES_PER_CHUNK"]) * chunk_capacity(max_rows)


def ctl_words(c_cap: int, arch: str) -> int:
    """Local control words for a chunk capacity ``c_cap``: ``arr[C_cap] | red_done[C_cap] |
    cta_arrivals[MAX_CTAS] | exit ticket | EP`` plus ``steal[C_cap]`` where the program steals and the
    ``sched`` pair-tile claim word where the program claims tiles dynamically."""

    c_cap = int(c_cap)
    if c_cap <= 0:
        raise ValueError(f"C_cap must be positive, got {c_cap}")
    record = geometry(arch)
    words = 2 * c_cap + int(record["MAX_CTAS"]) + int(record["CTL_EXTRA_WORDS"])
    if record["CTL_STEAL_WORDS"]:
        words += c_cap
    if record["CTL_SCHED_WORDS"]:
        words += 1
    return words


def validate_shape(m: int, n: int, k_l: int, arch: str) -> tuple[int, int, int]:
    """``M >= 1``, ``N % 256 == 0``, ``N <= MAX_COLS``, ``K_l % 128 == 0``, ``M * N <= 2**30``."""

    m, n, k_l = int(m), int(n), int(k_l)
    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    grid_n(n, arch)
    if n % N_MULTIPLE:
        raise ValueError(f"N must be a positive multiple of {N_MULTIPLE}, got {n}")
    if n > MAX_COLS:
        raise ValueError(
            f"N = {n} exceeds {MAX_COLS} (32-bit reducer/drainer vector index)"
        )
    if k_l <= 0 or k_l % BLOCK_K:
        raise ValueError(f"K_l must be a positive multiple of {BLOCK_K}, got {k_l}")
    if m * n > MAX_ROWS_TIMES_COLS:
        raise ValueError(f"M * N = {m * n} exceeds 2**30")
    return m, n, k_l


def program_variant(world_size: int, max_rows: int, arch: str) -> str:
    """The program suffix a capacity of ``max_rows`` rows selects on ``arch`` (empty by default)."""

    rows_capacity = ceil128(max_rows)
    for ws, capacity, suffix in CAPACITY_VARIANTS.get(arch, []):
        if int(ws) == int(world_size) and int(capacity) == rows_capacity:
            return str(suffix)
    return ""


def route_key(world_size: int, arch: str, max_rows: int) -> str:
    """``main_ws{world_size}_{comm_mode}{variant}`` of a capacity on ``arch``."""

    return (
        f"main_ws{int(world_size)}_{comm_mode(world_size, arch)}"
        f"{program_variant(world_size, max_rows, arch)}"
    )


def main_program(world_size: int, arch: str, max_rows: int) -> str:
    """The fused program of ``world_size`` on ``arch`` for a workspace of ``max_rows`` rows."""

    key = route_key(world_size, arch, max_rows)
    table = ROUTES.get(arch, {})
    if key not in table:
        raise ValueError(
            f"no fused FP8 GEMM + all-reduce program is delivered for {key!r} on {arch} "
            f"(world_size={int(world_size)}; delivered: {sorted(table)})"
        )
    return table[key]


def raster_group_m(m: int, n: int, k_l: int, arch: str) -> int:
    """Raster group of the SM-count grid: the smallest group whose live A / W tile set fits half
    the L2 (1 where the lever does not apply: short ``K_l``, few column tiles, few chunks). 1 on
    the cluster-pair grid, whose program carries no raster argument."""

    record = geometry(arch)
    if record["GRID_RULE"] != GRID_RULE_SM_COUNT:
        return 1
    c = chunk_count(m)
    gn = grid_n(n, arch)
    k_l = int(k_l)
    if (
        gn < int(record["RASTER_MIN_GRID_N"])
        or k_l < int(record["RASTER_MIN_K"])
        or c < int(record["RASTER_MIN_CHUNKS"])
    ):
        return 1
    tile = BLOCK_M * k_l
    sm_count = int(record["RASTER_SM_COUNT"])
    candidates = [int(g) for g in record["RASTER_GROUP_CHOICES"] if int(g) <= c]

    def live(g: int) -> float:
        return (sm_count / g + g) * tile

    fits = [g for g in candidates if live(g) <= 0.5 * int(record["RASTER_L2_BYTES"])]
    return fits[0] if fits else min(candidates, key=live)


def lever_suffixes(arch: str) -> tuple[str, ...]:
    """Every lever suffix a call on ``arch`` can select, ``""`` (no lever) first."""

    table = LEVER_DISPATCH.get(arch, {})
    suffixes = [""]
    for lever in ("REDUCERS", "LOCKSTEP", "DRAIN"):
        if lever in table:
            suffix = str(table[lever]["suffix"])
            suffixes += [item + suffix for item in suffixes]
    return tuple(suffixes)


def lever_variant(
    world_size: int,
    m: int,
    n: int,
    k_l: int,
    arch: str,
    *,
    sm_count: int,
    variant: str = "",
) -> str:
    """The lever suffix of one call on ``arch`` (empty: the route's program).

    Reducers: ``K_l`` at most the world size's bound, ``K_l * world_size <= N`` and
    ``M * N`` below the world size's ``mn_limit`` (when it has one).
    Lockstep: a program ``variant`` of the capacity (``program_variant``) among the
    lever's ``variants``, a raster group of at least ``min_raster_group``, at least
    ``min_tiles`` tiles and a grid below ``MAX_CTAS`` (the throttle counts in the last
    arrival word).
    Drain: a world size among the lever's ``worlds``, a program ``variant`` among its
    ``variants``, ``M`` at most ``max_m`` and a grid (``min(sm_count, tiles)``) of at
    least ``world_size - 1`` CTAs (one per peer).
    """

    table = LEVER_DISPATCH.get(arch, {})
    world_size, m, n, k_l = int(world_size), int(m), int(n), int(k_l)
    suffix = ""
    reducers = table.get("REDUCERS")
    bound = None if reducers is None else reducers["max_k_l"].get(world_size)
    limit = None if reducers is None else reducers.get("mn_limit", {}).get(world_size)
    if (
        bound is not None
        and k_l <= int(bound)
        and k_l * world_size <= n
        and (limit is None or m * n < int(limit))
    ):
        suffix += str(reducers["suffix"])
    lockstep = table.get("LOCKSTEP")
    if (
        lockstep is not None
        and str(variant) in lockstep["variants"]
        and 0 < int(sm_count) < int(geometry(arch)["MAX_CTAS"])
        and raster_group_m(m, n, k_l, arch) >= int(lockstep["min_raster_group"])
        and tiles(m, n, arch) >= int(lockstep["min_tiles"])
    ):
        suffix += str(lockstep["suffix"])
    drain = table.get("DRAIN")
    if (
        drain is not None
        and str(variant) in drain["variants"]
        and world_size in drain["worlds"]
        and m <= int(drain["max_m"])
        and min(int(sm_count), tiles(m, n, arch)) >= world_size - 1
    ):
        suffix += str(drain["suffix"])
    return suffix


def call_program(
    world_size: int,
    arch: str,
    max_rows: int,
    m: int,
    n: int,
    k_l: int,
    *,
    sm_count: int,
) -> str:
    """The program one call of ``(m, n, k_l)`` runs in a workspace of ``max_rows`` rows."""

    key = route_key(world_size, arch, max_rows) + lever_variant(
        world_size,
        m,
        n,
        k_l,
        arch,
        sm_count=sm_count,
        variant=program_variant(world_size, max_rows, arch),
    )
    table = ROUTES.get(arch, {})
    if key not in table:
        raise ValueError(
            f"no fused FP8 GEMM + all-reduce program is delivered for {key!r} on {arch} "
            f"(world_size={int(world_size)}; delivered: {sorted(table)})"
        )
    return table[key]


def dispatch_programs(
    world_size: int,
    arch: str,
    max_rows: int,
    n: int,
    k_l: int | None = None,
    *,
    sm_count: int,
) -> tuple[str, ...]:
    """Every program a workspace of ``max_rows`` rows can dispatch, its main program first.

    With ``k_l`` (a prepared launcher binds ``N`` and ``K_l``): the programs of the calls of
    1 to ``chunk_capacity(max_rows)`` chunks, each at its first and its last row count (a
    call's lever depends on ``M`` through its chunk count, the reducers' ``M * N``
    limit and the drainer's ``max_m``, which a chunk's two ends bracket). Without ``k_l`` (the one-shot workspace
    serves any ``K_l``): every program of the route's lever keys. On an architecture
    without levers every call runs the main program, so no chunk is enumerated.
    """

    main = main_program(world_size, arch, max_rows)
    if not LEVER_DISPATCH.get(arch):
        return (main,)
    base = route_key(world_size, arch, max_rows)
    table = ROUTES.get(arch, {})
    if k_l is None:
        found = [table[base + s] for s in lever_suffixes(arch) if base + s in table]
    else:
        found = [
            call_program(
                world_size,
                arch,
                max_rows,
                m,
                n,
                k_l,
                sm_count=sm_count,
            )
            for chunks in range(1, chunk_capacity(max_rows) + 1)
            for m in ((chunks - 1) * BLOCK_M + 1, chunks * BLOCK_M)
        ]
    return tuple(dict.fromkeys([main, *found]))


def launch_grid(
    m: int, n: int, arch: str, *, max_clusters: int = 0, sm_count: int = 0
) -> tuple[int, int, int]:
    """Persistent grid of one call.

    Cluster-pair grid: ``2 * min(co-resident clusters, pair tiles)`` CTAs (every cluster must be
    resident, the exchange protocol waits on peers' tiles). SM-count grid: ``min(SMs, tiles)``
    CTAs, one per SM.
    """

    record = geometry(arch)
    count = tiles(m, n, arch)
    if record["GRID_RULE"] == GRID_RULE_CLUSTER_PAIRS:
        max_clusters = int(max_clusters)
        if max_clusters <= 0:
            raise ValueError(f"max_clusters must be positive, got {max_clusters}")
        return (2 * min(max_clusters, count), 1, 1)
    sm_count = int(sm_count)
    if sm_count <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")
    return (min(sm_count, count), 1, 1)


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake fused FP8 GEMM + all-reduce CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    nested = len(parts) == 4 and parts[2] in SOURCE_DIRECTORIES
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or not (len(parts) == 3 or nested):
        raise ValueError(
            f"exported source path is outside the fused FP8 GEMM + all-reduce package: {relative!r}"
        )
    return _source_dir().joinpath(*parts[2:])


@functools.cache
def spec(program: str, arch: str) -> JitSpec:
    """Build specification of one program for one exact architecture."""

    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(
            f"program {program!r} is delivered for {record['arches']}, not {arch!r}"
        )
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_{arch}_{program}",
        sources=[_source_path(relative) for relative in record["sources"]],
        extra_cuda_cflags=[*ARCH_FLAGS[arch], *COMPILE_FLAGS],
        extra_include_paths=[_source_dir().parent],
        # The source build carries every math flag in ``COMPILE_FLAGS``; the
        # default ``-use_fast_math`` would otherwise be added or dropped.
        use_fast_math="--use_fast_math" in COMPILE_FLAGS,
    )


@functools.cache
def load(program: str, arch: str):
    return spec(program, arch).build_and_load()


def launch_block(program: str) -> tuple[int, int, int]:
    block = PROGRAMS[program]["block"]
    return (int(block[0]), int(block[1]), int(block[2]))


def cluster_dims(program: str) -> tuple[int, int, int]:
    cluster = PROGRAMS[program]["cluster"]
    return (int(cluster[0]), int(cluster[1]), int(cluster[2]))


def dynamic_smem_bytes(program: str) -> int:
    return int(PROGRAMS[program]["dynamic_smem_bytes"])


def launcher_args(program: str) -> tuple[str, ...]:
    """Positional argument names of the program's ``run`` launcher (grid triple last)."""

    return tuple(PROGRAMS[program]["args"])


_PROBE_KERNEL = "cake_fp8_rowwise_gemm_allreduce_cluster_capacity_probe"
_PROBE_SOURCE = f'extern "C" __global__ void {_PROBE_KERNEL}() {{}}\n'


@functools.cache
def cluster_capacity(device_index: int, program: str) -> int:
    """Co-resident clusters of ``program`` on ``device_index`` (the cluster-pair grid).

    ``cuOccupancyMaxActiveClusters`` on an empty probe kernel launched with the
    program's block, cluster shape and dynamic shared memory. The shared memory
    pins one CTA per SM (as the program itself does), so the answer follows the
    part's GPC topology after floorsweeping, not ``SMs // 2``: it is the number
    the persistent grid must not exceed for every cluster to be resident.
    """

    from ..cuda_utils import checkCudaErrors, driver, nvrtc

    block = launch_block(program)
    cluster = cluster_dims(program)
    smem = dynamic_smem_bytes(program)
    with torch.cuda.device(device_index):
        torch.empty(1, device="cuda")  # primary context
        major, minor = torch.cuda.get_device_capability(device_index)
        prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(_PROBE_SOURCE.encode(), b"probe.cu", 0, [], [])
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        size = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, cubin))
        checkCudaErrors(nvrtc.nvrtcDestroyProgram(prog))
        ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(device_index))
        try:
            checkCudaErrors(driver.cuCtxSetCurrent(ctx))
            module = checkCudaErrors(driver.cuModuleLoadData(cubin))
            try:
                func = checkCudaErrors(
                    driver.cuModuleGetFunction(module, _PROBE_KERNEL.encode())
                )
                checkCudaErrors(
                    driver.cuFuncSetAttribute(
                        func,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem,
                    )
                )
                attr = driver.CUlaunchAttribute()
                attr.id = (
                    driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                )
                attr.value.clusterDim.x = cluster[0]
                attr.value.clusterDim.y = cluster[1]
                attr.value.clusterDim.z = cluster[2]
                config = driver.CUlaunchConfig()
                config.gridDimX = cluster[0]
                config.gridDimY = cluster[1]
                config.gridDimZ = cluster[2]
                config.blockDimX = block[0]
                config.blockDimY = block[1]
                config.blockDimZ = block[2]
                config.sharedMemBytes = smem
                config.attrs = [attr]
                config.numAttrs = 1
                capacity = int(
                    checkCudaErrors(driver.cuOccupancyMaxActiveClusters(func, config))
                )
            finally:
                checkCudaErrors(driver.cuModuleUnload(module))
        finally:
            checkCudaErrors(driver.cuDevicePrimaryCtxRelease(device_index))
    if capacity <= 0:
        raise RuntimeError(
            f"device {device_index} holds no co-resident {cluster} cluster of program {program!r}"
        )
    return capacity


__all__ = [
    "ARCH_FLAGS",
    "BLOCK_K",
    "BLOCK_M",
    "CAPABILITY_ARCH",
    "CAPACITY_VARIANTS",
    "COMM_MODES",
    "COMPILE_FLAGS",
    "EXCHANGE_MODE",
    "FROZEN",
    "GEOMETRY",
    "GRID_RULE_CLUSTER_PAIRS",
    "GRID_RULE_SM_COUNT",
    "LEVER_DISPATCH",
    "MAX_COLS",
    "MAX_ROWS_TIMES_COLS",
    "N_MULTIPLE",
    "OUT_BYTES",
    "PROGRAMS",
    "ROUTES",
    "SOURCE_PACKAGE",
    "SUPPORTED_WORLD_SIZES",
    "DeviceFacts",
    "block_n",
    "call_program",
    "ceil128",
    "chunk_capacity",
    "chunk_count",
    "cluster_capacity",
    "cluster_dims",
    "comm_mode",
    "ctl_words",
    "device_facts",
    "dispatch_programs",
    "dynamic_smem_bytes",
    "flag_bytes",
    "flag_words",
    "geometry",
    "grid_n",
    "launch_block",
    "launch_grid",
    "launcher_args",
    "lever_suffixes",
    "lever_variant",
    "load",
    "main_program",
    "program_variant",
    "raster_group_m",
    "route_key",
    "s_rows",
    "slots_per_chunk",
    "spec",
    "tile_bytes",
    "tiles",
    "transport",
    "validate_shape",
    "workspace_bytes",
]
