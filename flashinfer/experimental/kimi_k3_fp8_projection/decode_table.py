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
"""

# Measured dispatch table of the Kimi-K3 FP8 projection, keyed
# "<n_tiles128>,<num_k_iters>,<m_bucket>": the fastest route over every decode
# configuration and the quantization + GEMM chain (CUDA-graph replay with a
# cold L2) measured on B200 (sm_100a) and B300 (sm_103a).  ``DECODE_TABLE``
# holds the cells both architectures measured alike; ``DECODE_TABLE_OVERRIDES``
# the cells whose fastest route differs per architecture; ``decode_table(arch)``
# merges them.  ``route`` is "decode" (swap-AB split-K decode kernel: ``tok``
# token-tile width, ``split`` K splits, ``fused`` in-CTA quantization,
# ``persist`` persistent grid, optional ``grid`` balanced persistent CTA count,
# ``resident`` resident token tiles, optional ``csplit`` cluster split-K width,
# ``xb_stages`` "auto" | N decoupled BF16 token ring and ``qlanes`` 4 / 8
# narrow quantization units of the fused variant, optional ``acc_bufs`` TMEM
# accumulator sets and ``epi_groups`` epilogue warp groups of the instance) or
# "gemm" (quantization launch + persistent 2-CTA GEMM); a two-launch cell may
# pin the width of its quantization launch (``quant_units``).
# ``ROUTE_TWINS`` holds, per architecture and cell, the twin programs a cell's
# launches take instead of the program the table selects: ``{selected kernel
# key: twin kernel key}``, the twin an instance of the same program with build
# time changes that keep every output bit (the L2 eviction-priority hints
# ``_wef`` / ``_xef`` of a decode program's weight / token stream, ``_bef`` of a
# GEMM program's weight stream, ``_xef_qel`` of the quantization launch's
# activation reads and hand-off stores; ``_k128`` an unfused decode program's
# 256-K TMA stages loaded as two 128-K halves, ``_sko<N>`` a 16-token cluster
# split-K decode program's partial-sum exchange bounded to the first N rows, in
# cells of at most N rows; ``_cv100`` the quantization launch with a preferred
# shared-memory carveout of 100 percent, which the decode launch starting beside
# it under PDL uses; ``_csrt`` a 4-way cluster split-K decode program's closing
# cluster barrier arriving relaxed after a tcgen05 fence; ``_gsb`` the gs-branch
# form of a 128-K grid-stride GEMM program; ``_ts3_tde`` a 192-wide 128-K
# TMA-store GEMM program with three staging slots per epilogue group and the
# pair's tensor memory freed by the MMA warps, ``_tde`` alone a GEMM program
# with the pair's tensor memory freed by the MMA warps; ``_prg`` a TMA-store GEMM
# program whose MMA warp allocates the pair's tensor memory, the load warp left
# out of the allocation rendezvous), measured faster on that
# architecture's rows of the cell; ``route_twins(arch)`` reads it.
# ``DECODE_TABLE_M_RANGES`` holds, per architecture and cell, in-bucket row
# ranges ``(lo, hi, entry)`` whose rows take ``entry`` instead of the cell's
# entry, where the fastest route changes inside one bucket; ``decode_table_m_ranges(arch)``
# reads it.  A row may name twins that only its launches into a TMA-store
# output view take (``tstore_twins``, ``{selected kernel key: twin kernel
# key}`` as in ``ROUTE_TWINS``), and a GEMM row whose ``gemm_bn`` ships the
# TMA-store epilogue only the width of its staged-register view
# (``gemm_rstaged_bn``); its other views keep the selected programs.
# Generated from the Cake measurement sweeps recorded in the source
# repository; do not edit by hand.

from __future__ import annotations

import functools
from typing import Any

ARCHES = ("sm_100a", "sm_103a")

DECODE_TABLE: dict[str, dict[str, Any]] = {
    "1,28,1": {
        "route": "decode",
        "tok": 16,
        "split": 14,
        "fused": True,
        "persist": True,
        "csplit": 14,
        "pfx": 1,
        "xp": 1,
    },
    "1,28,8": {
        "route": "decode",
        "tok": 16,
        "split": 14,
        "fused": True,
        "persist": True,
        "csplit": 14,
        "pfx": 1,
        "xp": 1,
    },
    "1,28,2048": {
        "route": "decode",
        "tok": 64,
        "split": 4,
        "fused": True,
        "persist": False,
        "csplit": 4,
        "xb_stages": 2,
        "qlanes": 8,
        "stages": 2,
        "qwarps": 16,
        "xbh": 1,
        "xp": 1,
    },
    "1,28,4096": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": True,
        "persist": True,
        "xb_stages": "auto",
        "qlanes": 4,
        "tstore": True,
    },
    "5,28,1": {
        "route": "decode",
        "tok": 16,
        "split": 14,
        "fused": True,
        "persist": True,
        "csplit": 14,
        "xp": 1,
    },
    "5,28,8": {
        "route": "decode",
        "tok": 16,
        "split": 14,
        "fused": True,
        "persist": True,
        "csplit": 14,
        "pfx": 1,
    },
    "5,28,64": {
        "route": "decode",
        "tok": 32,
        "split": 8,
        "fused": False,
        "persist": True,
        "csplit": 8,
        "xp": 1,
    },
    "5,28,256": {
        "route": "decode",
        "tok": 32,
        "split": 2,
        "fused": False,
        "persist": True,
        "csplit": 3,
    },
    "5,28,1024": {
        "route": "decode",
        "tok": 64,
        "split": 1,
        "fused": False,
        "persist": False,
        "qlanes": 16,
        "stages": 4,
        "qwarps": 8,
        "tstore": True,
    },
    "5,28,4096": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
    "5,28,16384": {"route": "gemm", "gemm_bn": 192, "gemm_skf": 1, "gemm_bk": 128},
    "12,1,1": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "12,1,8": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "12,1,64": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
        "pfx": 1,
    },
    "12,1,256": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": True,
        "persist": True,
        "resident": True,
        "tstore": True,
    },
    "12,1,1024": {
        "route": "decode",
        "tok": 128,
        "split": 1,
        "fused": False,
        "persist": False,
        "qlanes": 16,
        "stages": 3,
        "qwarps": 8,
        "tstore": 1,
    },
    "12,1,4096": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1},
    "12,1,16384": {
        "route": "gemm",
        "gemm_bn": 192,
        "gemm_stages": 2,
        "gemm_ts_slots": 4,
    },
    "12,28,1": {
        "route": "decode",
        "tok": 16,
        "split": 7,
        "fused": True,
        "persist": True,
        "csplit": 7,
        "xb_stages": 3,
    },
    "12,28,8": {
        "route": "decode",
        "tok": 16,
        "split": 7,
        "fused": True,
        "persist": True,
        "csplit": 7,
        "xb_stages": 3,
    },
    "12,28,64": {
        "route": "decode",
        "tok": 32,
        "split": 4,
        "fused": False,
        "persist": True,
        "csplit": 4,
        "pf": 1,
    },
    "12,28,256": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": False,
        "persist": True,
        "stages": 5,
        "epi_chunk": 16,
        "tstore": True,
    },
    "12,28,1024": {
        "route": "decode",
        "tok": 128,
        "split": 1,
        "fused": False,
        "persist": False,
        "qlanes": 16,
        "stages": 3,
        "qwarps": 8,
        "tstore": True,
    },
    "12,28,16384": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
    "17,28,1": {
        "route": "decode",
        "tok": 16,
        "split": 4,
        "fused": True,
        "persist": True,
        "csplit": 4,
    },
    "17,28,64": {
        "route": "decode",
        "tok": 64,
        "split": 8,
        "fused": False,
        "persist": True,
    },
    "17,28,256": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": False,
        "persist": True,
        "stages": 5,
        "epi_chunk": 16,
        "tstore": True,
    },
    "18,6,1": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "18,6,256": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": False,
        "persist": True,
        "tstore": True,
    },
    "18,6,1024": {
        "route": "decode",
        "tok": 128,
        "split": 1,
        "fused": False,
        "persist": False,
        "qlanes": 16,
        "stages": 3,
        "qwarps": 8,
        "tstore": True,
    },
    "18,6,4096": {"route": "gemm", "gemm_pf": 1, "gemm_bk": 128},
    "24,2,8": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
        "pfx": 1,
    },
    "24,2,64": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
        "qlanes": 8,
        "pfx": 1,
    },
    "24,2,256": {
        "route": "decode",
        "tok": 64,
        "split": 1,
        "fused": False,
        "persist": True,
        "tstore": True,
    },
    "24,2,1024": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1},
    "24,2,2048": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1},
    "24,2,16384": {
        "route": "gemm",
        "gemm_bk": 128,
        "gemm_stages": 2,
        "gemm_ts_slots": 4,
    },
    "48,28,1": {
        "route": "decode",
        "tok": 16,
        "split": 2,
        "fused": True,
        "persist": True,
        "csplit": 2,
    },
    "48,28,8": {
        "route": "decode",
        "tok": 16,
        "split": 2,
        "fused": True,
        "persist": True,
        "csplit": 2,
    },
    "48,28,64": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": False,
        "persist": True,
        "stages": 5,
        "epi_chunk": 16,
        "pf": 2,
        "tstore": True,
    },
    "48,28,256": {
        "route": "decode",
        "tok": 128,
        "split": 1,
        "fused": False,
        "persist": True,
        "pf": 3,
        "mc": 2,
        "tstore": True,
    },
    "48,28,2048": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
    "48,28,4096": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
    "50,28,1": {
        "route": "decode",
        "tok": 16,
        "split": 2,
        "fused": True,
        "persist": True,
        "csplit": 2,
    },
    "50,28,8": {
        "route": "decode",
        "tok": 16,
        "split": 2,
        "fused": True,
        "persist": True,
        "csplit": 2,
    },
    "50,28,64": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": False,
        "persist": True,
        "stages": 5,
        "epi_chunk": 16,
        "pf": 2,
    },
    "50,28,256": {
        "route": "decode",
        "tok": 128,
        "split": 1,
        "fused": False,
        "persist": True,
        "pf": 3,
        "mc": 2,
    },
    "50,28,2048": {"route": "gemm", "gemm_bk": 128},
    "56,6,1": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "56,6,8": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "56,6,64": {
        "route": "decode",
        "tok": 32,
        "split": 1,
        "fused": False,
        "persist": True,
        "tstore": True,
    },
    "56,6,16384": {"route": "gemm", "gemm_bk": 128},
    "56,48,1": {
        "route": "decode",
        "tok": 16,
        "split": 2,
        "fused": True,
        "persist": True,
        "csplit": 2,
    },
    "56,48,8": {
        "route": "decode",
        "tok": 16,
        "split": 2,
        "fused": True,
        "persist": True,
        "csplit": 2,
    },
    "96,1,1": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "96,1,8": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "96,1,2048": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1},
    "96,1,4096": {
        "route": "gemm",
        "gemm_bk": 128,
        "gemm_stages": 2,
        "gemm_ts_slots": 4,
    },
    "96,1,16384": {"route": "gemm", "gemm_stages": 2, "gemm_ts_slots": 4},
    "96,28,256": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 2, "gemm_bk": 128},
    "96,28,1024": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
    "96,28,4096": {"route": "gemm", "gemm_skf": 1},
    "144,6,8": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "144,6,64": {
        "route": "decode",
        "tok": 64,
        "split": 1,
        "fused": False,
        "persist": True,
        "tstore": True,
    },
    "144,6,256": {"route": "gemm", "gemm_bk": 128},
    "144,6,512": {"route": "gemm", "gemm_pf": 2},
    "144,6,1024": {"route": "gemm", "gemm_pf": 2},
    "144,6,4096": {"route": "gemm", "gemm_pf": 1},
    "192,2,1": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "192,2,2048": {"route": "gemm", "gemm_bk": 128},
    "384,28,8": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
        "pf": 1,
    },
    "384,28,64": {
        "route": "decode",
        "tok": 64,
        "split": 1,
        "fused": False,
        "persist": True,
        "tstore": True,
    },
    "384,28,256": {"route": "gemm", "gemm_skf": 1, "gemm_pf": 1},
    "386,28,1": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
    },
    "386,28,8": {
        "route": "decode",
        "tok": 16,
        "split": 1,
        "fused": True,
        "persist": True,
        "pf": 1,
    },
    "386,28,64": {
        "route": "decode",
        "tok": 64,
        "split": 1,
        "fused": False,
        "persist": True,
    },
    "386,28,256": {"route": "gemm", "gemm_skf": 1, "gemm_pf": 1},
}

DECODE_TABLE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "sm_100a": {
        "1,28,64": {
            "route": "decode",
            "tok": 16,
            "split": 14,
            "fused": True,
            "persist": True,
            "csplit": 14,
            "xb_stages": 2,
            "qlanes": 8,
            "xbh": 1,
            "qer": 1,
            "xp": 1,
        },
        "1,28,256": {
            "route": "decode",
            "tok": 16,
            "split": 4,
            "fused": True,
            "persist": True,
            "csplit": 6,
            "xb_stages": 4,
            "qlanes": 8,
            "xq_stages": 4,
        },
        "1,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 4,
            "fused": False,
            "persist": False,
            "csplit": 8,
            "qlanes": 16,
            "stages": 3,
            "qwarps": 8,
            "xp": 1,
            "quant_units": 2,
        },
        "1,28,1024": {
            "route": "decode",
            "tok": 64,
            "split": 4,
            "fused": True,
            "persist": False,
            "csplit": 6,
            "xb_stages": 3,
            "qlanes": 4,
            "stages": 2,
            "pf": 1,
            "qwarps": 16,
            "xbh": 1,
            "qer": 1,
            "xp": 1,
            "epi_groups": 2,
        },
        "1,28,16384": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": True,
            "persist": True,
            "grid": 128,
            "xb_stages": "auto",
            "qlanes": 4,
            "pf": 4,
            "pfx": 1,
            "pfi": 2,
            "qwarps": 16,
            "xbh": 1,
            "qer": 1,
            "tstore": 1,
        },
        "5,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "csplit": 3,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
            "epi_groups": 2,
            "quant_units": 2,
        },
        "5,28,2048": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 3,
            "qwarps": 8,
            "tstore": 1,
        },
        "12,1,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": True,
            "persist": False,
            "xb_stages": 2,
            "qlanes": 8,
            "stages": 2,
            "qwarps": 8,
            "xbh": 1,
            "tstore": 1,
        },
        "12,1,2048": {
            "route": "gemm",
            "gemm_bn": 192,
            "gemm_bk": 128,
            "gemm_static": 1,
        },
        "12,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
        },
        "12,28,2048": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 2},
        "12,28,4096": {"route": "gemm", "gemm_skp": 4},
        "17,28,8": {
            "route": "decode",
            "tok": 16,
            "split": 4,
            "fused": True,
            "persist": True,
            "csplit": 4,
            "pfx": 1,
        },
        "17,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
        },
        "17,28,1024": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 3,
            "qwarps": 8,
            "tstore": 1,
        },
        "17,28,2048": {"route": "gemm", "gemm_bk": 128, "gemm_gs": 1},
        "17,28,4096": {"route": "gemm"},
        "17,28,16384": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "18,6,8": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "pfx": 1,
        },
        "18,6,64": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": False,
            "persist": True,
            "pw": 2,
        },
        "18,6,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
            "tstore": True,
        },
        "18,6,2048": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": True,
            "qlanes": 16,
            "stages": 3,
            "qwarps": 8,
            "tstore": True,
        },
        "18,6,16384": {"route": "gemm"},
        "24,2,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "qlanes": 8,
        },
        "24,2,512": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 3,
            "qwarps": 8,
            "tstore": 1,
        },
        "24,2,4096": {"route": "gemm", "gemm_bk": 128, "gemm_gs": 1},
        "48,28,512": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 2},
        "48,28,1024": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 1},
        "50,28,512": {
            "route": "gemm",
            "gemm_bn": 192,
            "gemm_bk": 128,
            "gemm_pf": 2,
            "gemm_rstaged_pf": 1,
        },
        "50,28,1024": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "50,28,4096": {
            "route": "gemm",
            "gemm_bk": 128,
            "gemm_skf": 1,
            "gemm_rstaged_skf": 1,
        },
        "50,28,16384": {"route": "gemm"},
        "56,6,256": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": True,
            "pf": 1,
            "tstore": True,
        },
        "56,6,512": {"route": "gemm", "gemm_bn": 224, "gemm_bk": 128, "gemm_gs": 1},
        "56,6,1024": {"route": "gemm", "gemm_pf": 2, "gemm_bk": 128},
        "56,6,2048": {"route": "gemm", "gemm_bk": 128, "gemm_gs": 1},
        "56,6,4096": {"route": "gemm", "gemm_bk": 128},
        "56,48,64": {
            "route": "decode",
            "tok": 64,
            "split": 2,
            "fused": False,
            "persist": True,
            "csplit": 2,
        },
        "56,48,256": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": True,
            "pf": 3,
            "mc": 2,
            "tstore": True,
            "quant_units": 4,
        },
        "56,48,512": {"route": "gemm", "gemm_bn": 224, "gemm_pf": 2},
        "56,48,1024": {
            "route": "gemm",
            "gemm_bn": 192,
            "gemm_skf": 1,
            "gemm_pf": 1,
            "gemm_bk": 128,
        },
        "56,48,2048": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
        "56,48,4096": {"route": "gemm", "gemm_skf": 1},
        "56,48,16384": {"route": "gemm"},
        "96,1,64": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": True,
            "persist": True,
            "resident": True,
            "tstore": True,
        },
        "96,1,256": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": True,
            "persist": True,
            "resident": True,
            "tstore": True,
        },
        "96,1,512": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "96,1,1024": {
            "route": "gemm",
            "gemm_pf": 2,
            "gemm_bk": 128,
            "gemm_stages": 2,
            "gemm_ts_slots": 4,
        },
        "96,28,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
        },
        "96,28,8": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
        },
        "96,28,64": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": True,
            "tstore": True,
        },
        "96,28,512": {"route": "gemm", "gemm_bn": 352},
        "96,28,2048": {"route": "gemm", "gemm_skf": 1},
        "96,28,16384": {"route": "gemm"},
        "144,6,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "stages": 3,
        },
        "144,6,2048": {"route": "gemm", "gemm_pf": 1},
        "144,6,16384": {"route": "gemm", "gemm_bk": 128},
        "192,2,8": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "xb_stages": 2,
            "qlanes": 8,
            "xbh": 1,
            "qer": 1,
        },
        "192,2,64": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": True,
            "tstore": True,
        },
        "192,2,256": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "192,2,512": {"route": "gemm", "gemm_bk": 128, "gemm_gs": 1},
        "192,2,1024": {
            "route": "gemm",
            "gemm_bk": 128,
            "gemm_gs": 1,
            "gemm_ars": 1,
        },
        "192,2,4096": {
            "route": "gemm",
            "gemm_bk": 128,
            "gemm_gs": 1,
            "gemm_ars": 1,
        },
        "192,2,16384": {
            "route": "gemm",
            "gemm_bk": 128,
            "gemm_gs": 1,
            "gemm_ars": 1,
        },
        "384,28,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": False,
        },
        "384,28,512": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
        "384,28,1024": {"route": "gemm", "gemm_bk": 128},
        "384,28,2048": {"route": "gemm"},
        "384,28,16384": {"route": "gemm", "gemm_bn": 384},
        "386,28,512": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
        "386,28,1024": {"route": "gemm", "gemm_bk": 128},
        "386,28,2048": {"route": "gemm"},
        "386,28,16384": {"route": "gemm", "gemm_bn": 384},
    },
    "sm_103a": {
        "1,28,64": {
            "route": "decode",
            "tok": 16,
            "split": 14,
            "fused": True,
            "persist": True,
            "csplit": 14,
            "xb_stages": 2,
            "qlanes": 8,
            "pfx": 1,
            "xbh": 1,
            "xp": 1,
        },
        "1,28,256": {
            "route": "decode",
            "tok": 16,
            "split": 4,
            "fused": True,
            "persist": True,
            "csplit": 6,
            "xb_stages": 2,
            "qlanes": 8,
            "xq_stages": 4,
            "pfx": 1,
            "xbh": 1,
            "qer": 1,
        },
        "1,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 4,
            "fused": False,
            "persist": False,
            "csplit": 8,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
            "quant_units": 2,
        },
        "1,28,1024": {
            "route": "decode",
            "tok": 64,
            "split": 4,
            "fused": True,
            "persist": False,
            "csplit": 6,
            "xb_stages": 3,
            "qlanes": 4,
            "stages": 2,
            "qwarps": 16,
            "xbh": 1,
            "qer": 1,
        },
        "1,28,16384": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": True,
            "persist": True,
            "grid": 128,
            "xb_stages": "auto",
            "qlanes": 4,
            "pf": 4,
            "pfx": 1,
            "pfi": 2,
            "qwarps": 16,
            "xbh": 1,
            "qer": 1,
        },
        "5,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "csplit": 3,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
            "acc_bufs": 2,
            "epi_groups": 2,
            "quant_units": 2,
        },
        "5,28,2048": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_cs": 2},
        "12,1,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 2,
            "qwarps": 8,
        },
        "12,1,2048": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1},
        "12,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
            "tstore": 1,
        },
        "12,28,2048": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 1},
        "12,28,4096": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "17,28,8": {
            "route": "decode",
            "tok": 16,
            "split": 4,
            "fused": True,
            "persist": True,
            "csplit": 4,
            "xb_stages": 3,
            "pf": 1,
            "pfx": 1,
        },
        "17,28,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
            "tstore": 1,
        },
        "17,28,1024": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 3,
            "pf": 1,
            "mc": 2,
            "qwarps": 8,
            "tstore": 1,
        },
        "17,28,2048": {"route": "gemm", "gemm_pf": 2},
        "17,28,4096": {"route": "gemm", "gemm_bk": 128},
        "17,28,16384": {"route": "gemm", "gemm_bn": 192, "gemm_skf": 1, "gemm_bk": 128},
        "18,6,8": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "qlanes": 8,
            "pfx": 1,
        },
        "18,6,64": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": False,
            "persist": True,
        },
        "18,6,512": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 4,
            "qwarps": 8,
            "tstore": 1,
            "quant_units": 2,
        },
        "18,6,2048": {"route": "gemm", "quant_units": 2, "gemm_bk": 128, "gemm_gs": 1},
        "18,6,16384": {"route": "gemm", "gemm_bk": 128},
        "24,2,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
        },
        "24,2,512": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": False,
            "qlanes": 16,
            "stages": 3,
            "qwarps": 8,
            "tstore": 1,
            "quant_units": 2,
        },
        "24,2,4096": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "48,28,512": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 2, "gemm_bk": 128},
        "48,28,1024": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 1, "gemm_bk": 128},
        "50,28,512": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "50,28,1024": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_pdlb": 1},
        "50,28,4096": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "50,28,16384": {"route": "gemm", "gemm_bk": 128},
        "56,6,256": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": True,
            "tstore": True,
            "quant_units": 4,
        },
        "56,6,512": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 1},
        "56,6,1024": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": True,
            "qlanes": 16,
            "stages": 3,
            "qwarps": 8,
            "tstore": 1,
        },
        "56,6,2048": {"route": "gemm", "gemm_pf": 2, "quant_units": 2},
        "56,6,4096": {"route": "gemm", "gemm_pf": 2},
        "56,48,64": {
            "route": "decode",
            "tok": 64,
            "split": 2,
            "fused": False,
            "persist": True,
            "csplit": 2,
            "acc_bufs": 2,
        },
        "56,48,256": {
            "route": "decode",
            "tok": 128,
            "split": 1,
            "fused": False,
            "persist": True,
            "pf": 3,
            "mc": 2,
            "tstore": True,
            "quant_units": 2,
        },
        "56,48,512": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 2, "gemm_bk": 128},
        "56,48,1024": {"route": "gemm", "gemm_bn": 192, "gemm_pf": 1, "gemm_bk": 128},
        "56,48,2048": {"route": "gemm", "gemm_bk": 128},
        "56,48,4096": {"route": "gemm"},
        "56,48,16384": {"route": "gemm", "gemm_bn": 384},
        "96,1,64": {
            "route": "decode",
            "tok": 32,
            "split": 1,
            "fused": True,
            "persist": True,
            "resident": True,
            "tstore": True,
        },
        "96,1,256": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": True,
            "persist": True,
            "resident": True,
            "tstore": True,
            "dpf": 1,
        },
        "96,1,512": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1},
        "96,1,1024": {
            "route": "gemm",
            "gemm_pf": 4,
            "gemm_stages": 2,
            "gemm_ts_slots": 4,
        },
        "96,28,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "dpf": 1,
        },
        "96,28,8": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "dpf": 1,
        },
        "96,28,64": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": True,
            "pf": 1,
            "tstore": True,
        },
        "96,28,512": {"route": "gemm", "gemm_bn": 384},
        "96,28,2048": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
        "96,28,16384": {"route": "gemm", "gemm_bn": 384},
        "144,6,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
        },
        "144,6,2048": {"route": "gemm", "gemm_pf": 2},
        "144,6,16384": {"route": "gemm", "gemm_bk": 128, "gemm_gm": 64},
        "192,2,8": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
        },
        "192,2,64": {
            "route": "decode",
            "tok": 64,
            "split": 1,
            "fused": False,
            "persist": True,
            "tstore": True,
            "pw": 2,
        },
        "192,2,256": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1},
        "192,2,512": {"route": "gemm", "gemm_bk": 128},
        "192,2,1024": {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128},
        "192,2,4096": {"route": "gemm", "gemm_bk": 128},
        "192,2,16384": {"route": "gemm", "gemm_bk": 128},
        "384,28,1": {
            "route": "decode",
            "tok": 16,
            "split": 1,
            "fused": True,
            "persist": True,
            "grid": 128,
        },
        "384,28,512": {"route": "gemm", "gemm_skf": 1, "gemm_pf": 1},
        "384,28,1024": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
        "384,28,2048": {"route": "gemm", "gemm_bk": 128},
        "384,28,16384": {"route": "gemm"},
        "386,28,512": {"route": "gemm", "gemm_skf": 1, "gemm_pf": 2},
        "386,28,1024": {"route": "gemm", "gemm_skf": 1, "gemm_bk": 128},
        "386,28,2048": {"route": "gemm", "gemm_bk": 128},
        "386,28,16384": {"route": "gemm"},
    },
}


DECODE_TABLE_M_RANGES: dict[
    str, dict[str, tuple[tuple[int, int, dict[str, Any]], ...]]
] = {
    "sm_100a": {
        "50,28,2048": (
            (
                1281,
                2048,
                {
                    "route": "gemm",
                    "gemm_bk": 128,
                    "gemm_skf": 1,
                    "gemm_rstaged_skf": 1,
                },
            ),
        ),
        "50,28,4096": ((3329, 3584, {"route": "gemm", "gemm_bk": 128}),),
        "192,2,2048": (
            (
                1281,
                2048,
                {"route": "gemm", "gemm_bk": 128, "gemm_gs": 1, "gemm_ars": 1},
            ),
        ),
    },
    "sm_103a": {
        "96,28,512": (
            (
                497,
                512,
                {
                    "route": "gemm",
                    "gemm_bn": 352,
                    "gemm_rstaged_bn": 384,
                    "tstore_twins": {"quant:u4": "quant:u4_xef_qel"},
                },
            ),
        ),
        "50,28,2048": (
            (
                1537,
                2048,
                {
                    "route": "gemm",
                    "gemm_bk": 128,
                    "gemm_skf": 1,
                    "gemm_rstaged_skf": 1,
                },
            ),
        ),
        "192,2,16384": (
            (
                4097,
                12288,
                {"route": "gemm", "gemm_bk": 128, "gemm_gs": 1, "gemm_ars": 1},
            ),
            (
                16384,
                16384,
                {"route": "gemm", "gemm_bk": 128, "gemm_gs": 1, "gemm_ars": 1},
            ),
        ),
    },
}

ROUTE_TWINS: dict[str, dict[str, dict[str, str]]] = {
    "sm_100a": {
        "1,28,1": {
            "decode:t16_p3_fused_cs14_px1_xp": "decode:t16_p3_fused_cs14_px1_xp_sko1",
        },
        "1,28,8": {
            "decode:t16_p3_fused_cs14_px1_xp": "decode:t16_p3_fused_cs14_px1_xp_sko8",
        },
        "1,28,512": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "1,28,2048": {
            "decode:t64_p2_fused_w16_r2_xh_q8_cs4_xp": "decode:t64_p2_fused_w16_r2_xh_q8_cs4_xp_xef",
        },
        "1,28,4096": {
            "decode:t32_p3_fused_r5_q4_tso": "decode:t32_p3_fused_r5_q4_tso_xef",
            "decode:t32_p3_fused_r5_q4": "decode:t32_p3_fused_r5_q4_xef",
        },
        "1,28,16384": {
            "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_tso_pi2": "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_tso_pi2_xef",
            "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_pi2": "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_pi2_xef",
        },
        "5,28,1": {
            "decode:t16_p3_fused_cs14_xp": "decode:t16_p3_fused_cs14_xp_sko1",
        },
        "5,28,8": {
            "decode:t16_p3_fused_cs14_px1": "decode:t16_p3_fused_cs14_px1_sko8",
        },
        "5,28,64": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "5,28,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "5,28,512": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "5,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "5,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "5,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,1,4096": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_n192_k128_gs": "gemm_tstore_n192_k128_gs_prg",
        },
        "12,1,16384": {
            "gemm_tstore_n192_s2_ts4": "gemm_tstore_n192_s2_ts4_prg",
        },
        "12,28,64": {
            "decode:t32_p4_cs4_pf1": "decode:t32_p4_cs4_pf1_csrt",
        },
        "12,28,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "12,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,64": {
            "quant:u1": "quant:u1_cv100",
        },
        "17,28,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "17,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "18,6,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "18,6,512": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_tso": "decode:t64_p4_tso_k128",
        },
        "18,6,1024": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "18,6,2048": {
            "decode:t128_p3_tso": "decode:t128_p3_tso_k128",
        },
        "18,6,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "24,2,1024": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "24,2,2048": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "24,2,4096": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "48,28,1": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "48,28,8": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "48,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t128_p3_mc2_pf3_tso": "decode:t128_p3_mc2_pf3_tso_k128_wef",
        },
        "48,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "48,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "48,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "48,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,1": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "50,28,8": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "50,28,64": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "50,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t128_p3_mc2_pf3": "decode:t128_p3_mc2_pf3_k128_wef",
        },
        "50,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,6,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "56,6,512": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "56,6,1024": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "56,6,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,1": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "56,48,8": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "56,48,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_cs2a": "decode:t64_p4_cs2a_wef",
        },
        "56,48,256": {
            "quant:u4": "quant:u4_xef_qel",
            "decode:t128_p3_mc2_pf3_tso": "decode:t128_p3_mc2_pf3_tso_wef",
        },
        "56,48,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "96,1,512": {
            "gemm_tstore_n192_k128": "gemm_tstore_n192_k128_ts3_tde",
        },
        "96,1,1024": {
            "gemm_tstore_k128_s2_ts4_pf2": "gemm_tstore_k128_s2_ts4_pf2_prg",
        },
        "96,1,2048": {
            "gemm_tstore_n192_k128_gs": "gemm_tstore_n192_k128_gs_gsb",
        },
        "96,1,4096": {
            "gemm_tstore_k128_s2_ts4": "gemm_tstore_k128_s2_ts4_tde_prg",
        },
        "96,28,1": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "96,28,8": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "96,28,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_tso": "decode:t64_p4_tso_wef",
        },
        "96,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_n192_k128_pf2": "gemm_tstore_n192_k128_pf2_bef",
        },
        "96,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "96,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "96,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "144,6,1": {
            "decode:t16_p3_fused": "decode:t16_p3_fused_wef",
        },
        "144,6,8": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "144,6,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_tso": "decode:t64_p4_tso_wef",
        },
        "144,6,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_k128": "gemm_tstore_k128_bef",
        },
        "144,6,512": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "144,6,1024": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "192,2,256": {
            "gemm_tstore_n192_k128": "gemm_tstore_n192_k128_ts3_tde",
        },
        "192,2,2048": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "384,28,1": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "384,28,8": {
            "decode:t16_p4_fused_pf1": "decode:t16_p4_fused_pf1_wef",
        },
        "384,28,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_tso": "decode:t64_p4_tso_wef",
        },
        "384,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_skf_pf1": "gemm_tstore_skf_pf1_bef",
        },
        "386,28,1": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "386,28,8": {
            "decode:t16_p4_fused_pf1": "decode:t16_p4_fused_pf1_wef",
        },
        "386,28,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4": "decode:t64_p4_wef",
        },
        "386,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_skf_pf1": "gemm_tstore_skf_pf1_bef",
        },
    },
    "sm_103a": {
        "1,28,1": {
            "decode:t16_p3_fused_cs14_px1_xp": "decode:t16_p3_fused_cs14_px1_xp_sko1",
        },
        "1,28,8": {
            "decode:t16_p3_fused_cs14_px1_xp": "decode:t16_p3_fused_cs14_px1_xp_sko8",
        },
        "1,28,512": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "1,28,2048": {
            "decode:t64_p2_fused_w16_r2_xh_q8_cs4_xp": "decode:t64_p2_fused_w16_r2_xh_q8_cs4_xp_xef",
        },
        "1,28,4096": {
            "decode:t32_p3_fused_r5_q4_tso": "decode:t32_p3_fused_r5_q4_tso_xef",
            "decode:t32_p3_fused_r5_q4": "decode:t32_p3_fused_r5_q4_xef",
        },
        "1,28,16384": {
            "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_pi2": "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_pi2_xef",
        },
        "5,28,1": {
            "decode:t16_p3_fused_cs14_xp": "decode:t16_p3_fused_cs14_xp_sko1",
        },
        "5,28,8": {
            "decode:t16_p3_fused_cs14_px1": "decode:t16_p3_fused_cs14_px1_sko8",
        },
        "5,28,64": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "5,28,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "5,28,512": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "5,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "5,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "5,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,1,4096": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_n192_k128_gs": "gemm_tstore_n192_k128_gs_ts3_tde_prg",
        },
        "12,28,64": {
            "decode:t32_p4_cs4_pf1": "decode:t32_p4_cs4_pf1_csrt",
        },
        "12,28,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "12,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "12,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4": "decode:t64_p4_wef",
        },
        "17,28,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "17,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "17,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "18,6,256": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "18,6,512": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "18,6,1024": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "18,6,2048": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "18,6,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "24,2,512": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "24,2,1024": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "24,2,2048": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "24,2,4096": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "48,28,1": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "48,28,8": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "48,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t128_p3_mc2_pf3_tso": "decode:t128_p3_mc2_pf3_tso_wef",
        },
        "48,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "48,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "48,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "48,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,1": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "50,28,8": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "50,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t128_p3_mc2_pf3": "decode:t128_p3_mc2_pf3_wef",
        },
        "50,28,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "50,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,6,512": {
            "quant:u1": "quant:u1_xef_qel",
        },
        "56,6,1024": {
            "quant:u2": "quant:u2_xef_qel",
        },
        "56,6,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,1": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "56,48,8": {
            "decode:t16_p4_fused_cs2": "decode:t16_p4_fused_cs2_wef",
        },
        "56,48,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_a2_cs2a": "decode:t64_p4_a2_cs2a_wef",
        },
        "56,48,256": {
            "quant:u2": "quant:u2_xef_qel",
            "decode:t128_p3_mc2_pf3_tso": "decode:t128_p3_mc2_pf3_tso_wef",
        },
        "56,48,512": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "56,48,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "96,1,512": {
            "gemm_tstore_n192_k128_gs": "gemm_tstore_n192_k128_gs_ts3_tde_prg",
        },
        "96,1,1024": {
            "gemm_tstore_s2_ts4_pf4": "gemm_tstore_s2_ts4_pf4_tde",
        },
        "96,1,2048": {
            "gemm_tstore_n192_k128_gs": "gemm_tstore_n192_k128_gs_ts3_tde",
        },
        "96,1,4096": {
            "gemm_tstore_k128_s2_ts4": "gemm_tstore_k128_s2_ts4_tde_prg",
        },
        "96,28,1": {
            "decode:t16_p4_fused_dpf": "decode:t16_p4_fused_dpf_wef",
        },
        "96,28,8": {
            "decode:t16_p4_fused_dpf": "decode:t16_p4_fused_dpf_wef",
        },
        "96,28,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_pf1_tso": "decode:t64_p4_pf1_tso_wef",
        },
        "96,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_n192_k128_pf2": "gemm_tstore_n192_k128_pf2_bef",
        },
        "96,28,1024": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "96,28,2048": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "96,28,4096": {
            "quant:u4": "quant:u4_xef_qel",
        },
        "144,6,1": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "144,6,8": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "144,6,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_tso": "decode:t64_p4_tso_wef",
        },
        "144,6,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_k128": "gemm_tstore_k128_bef",
        },
        "192,2,256": {
            "gemm_tstore_n192_k128_gs": "gemm_tstore_n192_k128_gs_ts3_tde_prg",
        },
        "384,28,1": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "384,28,8": {
            "decode:t16_p4_fused_pf1": "decode:t16_p4_fused_pf1_wef",
        },
        "384,28,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4_tso": "decode:t64_p4_tso_wef",
        },
        "384,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_skf_pf1": "gemm_tstore_skf_pf1_bef",
        },
        "386,28,1": {
            "decode:t16_p4_fused": "decode:t16_p4_fused_wef",
        },
        "386,28,8": {
            "decode:t16_p4_fused_pf1": "decode:t16_p4_fused_pf1_wef",
        },
        "386,28,64": {
            "quant:u1": "quant:u1_xef_qel",
            "decode:t64_p4": "decode:t64_p4_wef",
        },
        "386,28,256": {
            "quant:u1": "quant:u1_xef_qel",
            "gemm_tstore_skf_pf1": "gemm_tstore_skf_pf1_bef",
        },
    },
}


@functools.cache
def decode_table(arch: str) -> dict[str, dict[str, Any]]:
    """The measured table of ``arch``: the shared cells with the overrides of that architecture."""
    overrides = DECODE_TABLE_OVERRIDES.get(arch)
    if overrides is None:
        raise ValueError(
            f"no measured dispatch table for {arch!r} (tabulated: {ARCHES})"
        )
    return {**DECODE_TABLE, **overrides}


def decode_table_m_ranges(
    arch: str,
) -> dict[str, tuple[tuple[int, int, dict[str, Any]], ...]]:
    """The in-bucket row ranges of ``arch``: ``{cell: ((lo, hi, entry), ...)}``, rows ``lo..hi`` of the cell's bucket
    taking ``entry`` instead of the cell's entry (none for an untabulated architecture)."""
    return DECODE_TABLE_M_RANGES.get(arch, {})


def route_twins(arch: str) -> dict[str, dict[str, str]]:
    """The twin programs of ``arch``: ``{cell: {selected kernel key: twin kernel key}}`` (none for an untabulated
    architecture)."""
    return ROUTE_TWINS.get(arch, {})
