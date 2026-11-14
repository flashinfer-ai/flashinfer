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

Host plan and loader glue of the Cake mHC training head, non-deterministic
mode: the stream mean of the four residual streams and its gradient (hidden
4096, 5120 and 7168; SM100 / SM103 / SM107).

The programs are the deterministic head's, exported from the same kernel
sources under the library stem ``cake_mhc_train_head_*`` of the
non-deterministic mHC training family: the same launch records, grid rules,
argument plans and routes as the deterministic head's registry, so the same
arithmetic in the same fixed order.  This package keeps its own copy of every
record and imports nothing outside its directory.

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its role,
its launch and grid rule and the kernel-parameter -> logical-argument map.
What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A role is the kernel of one public call: ``head_fwd`` (the forward),
``head_bwd`` (the backward that writes ``dX``) and ``head_bwd_acc`` (the
backward that adds into ``dX``).  Each architecture group has its own
libraries, and ``ROUTES[arch][role][hidden]`` names the record a role runs
on an architecture at a hidden size (:func:`route_module`).  On sm_100a and
sm_103a every kernel moves 64-bit vectors (4 BF16 values) per thread and is
built for one hidden size (the ``_h<hidden>`` libraries; the record's
``hidden``): the forward in 256-thread CTAs on sm_100a and 128-thread CTAs on
sm_103a, the backward in 512-thread CTAs (on sm_100a 384 at hidden 4096 and
256 at hidden 5120 and 7168) and the accumulating backward in 768-thread
CTAs.  On sm_107a the hidden-4096 forward and writing backward move 256-bit
vectors (16 values) in 128-thread CTAs, and the accumulating backward has L2
do its additions (bulk reductions from a shared-memory segment of 128-bit
vectors, 256-thread CTAs).
The unsuffixed libraries take any hidden size (``hidden`` 0), with 128-bit
forward vectors (8 values) and 256-bit backward vectors (16 values) in
256-thread CTAs; they serve a small problem (:data:`SMALL_ROUTES`): every role
on sm_107a, the backward that writes ``dX`` on sm_100a and sm_103a.  Each
output element is one fixed chain of operations, the same chain in every
build (the one addition of an accumulating element is L2's in the bulk-reduce
library and the thread's in the others, both a BF16 addition rounded to
nearest), so the outputs are the same bits whichever library and grid compute
them.

Each kernel moves one vector of ``vector_elems`` BF16 values per thread on a
plain (non-persistent) grid of ``ceil(T * hv / threads)`` CTAs, where ``hv =
hidden / vector_elems`` (the kernel parameter ``hv``, a fast divisor) is the
vector count of one row.  A library built for one hidden size finds a
thread's token and column by divisions by constants and requires ``hv ==
hidden / vector_elems``; an unsuffixed library divides by ``hv`` at run time.
The token count ``T`` is a runtime scalar (``T == 0`` launches nothing).  The
kernels require ``T <= MAX_TOKENS`` and ``hv <= MAX_VECTORS`` (2048 vectors:
hidden 8192 at 4 values per vector).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

MODULES: dict[str, dict[str, Any]] = {
    "cake_mhc_train_head_fwd": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 8},
        "hidden": 0,
        "kernel": "cake_mhc_train_head_fwd",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_fwd__lw64_t128_h4096": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 128, "vector_elems": 4},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_fwd__lw64_t128_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd__lw64_t128_h4096",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd__lw64_t128_h4096",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_fwd__lw64_t128_h5120": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 128, "vector_elems": 4},
        "hidden": 5120,
        "kernel": "cake_mhc_train_head_fwd__lw64_t128_h5120",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd__lw64_t128_h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd__lw64_t128_h5120",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_fwd__lw64_t128_h7168": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 128, "vector_elems": 4},
        "hidden": 7168,
        "kernel": "cake_mhc_train_head_fwd__lw64_t128_h7168",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd__lw64_t128_h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd__lw64_t128_h7168",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_fwd__lw64_t256_h4096": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 4},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_fwd__lw64_t256_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd__lw64_t256_h4096",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd__lw64_t256_h4096",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_fwd__lw64_t256_h5120": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 4},
        "hidden": 5120,
        "kernel": "cake_mhc_train_head_fwd__lw64_t256_h5120",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd__lw64_t256_h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd__lw64_t256_h5120",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_fwd__lw64_t256_h7168": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 4},
        "hidden": 7168,
        "kernel": "cake_mhc_train_head_fwd__lw64_t256_h7168",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd__lw64_t256_h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd__lw64_t256_h7168",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_fwd__lw256_t128_h4096": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 128, "vector_elems": 16},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_fwd__lw256_t128_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_fwd__lw256_t128_h4096",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_fwd__lw256_t128_h4096",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_train_head_bwd": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 16},
        "hidden": 0,
        "kernel": "cake_mhc_train_head_bwd",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__lw64_t384_h4096": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 384, "vector_elems": 4},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_bwd__lw64_t384_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__lw64_t384_h4096",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__lw64_t384_h4096",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__lw64_t512_h4096": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 512, "vector_elems": 4},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_bwd__lw64_t512_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__lw64_t512_h4096",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__lw64_t512_h4096",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__lw64_t512_h5120": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 512, "vector_elems": 4},
        "hidden": 5120,
        "kernel": "cake_mhc_train_head_bwd__lw64_t512_h5120",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__lw64_t512_h5120",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__lw64_t512_h5120",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__lw64_t512_h7168": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 512, "vector_elems": 4},
        "hidden": 7168,
        "kernel": "cake_mhc_train_head_bwd__lw64_t512_h7168",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__lw64_t512_h7168",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__lw64_t512_h7168",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__lw64_t256_h5120": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 4},
        "hidden": 5120,
        "kernel": "cake_mhc_train_head_bwd__lw64_t256_h5120",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__lw64_t256_h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__lw64_t256_h5120",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__lw64_t256_h7168": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 4},
        "hidden": 7168,
        "kernel": "cake_mhc_train_head_bwd__lw64_t256_h7168",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__lw64_t256_h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__lw64_t256_h7168",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__lw256_t128_h4096": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 128, "vector_elems": 16},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_bwd__lw256_t128_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__lw256_t128_h4096",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__lw256_t128_h4096",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_train_head_bwd__acc": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 16},
        "hidden": 0,
        "kernel": "cake_mhc_train_head_bwd__acc",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__acc",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__acc",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd_acc",
    },
    "cake_mhc_train_head_bwd__acc_lw64_t768_h4096": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 768, "vector_elems": 4},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_bwd__acc_lw64_t768_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__acc_lw64_t768_h4096",
        "launch": {
            "block": (768, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__acc_lw64_t768_h4096",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd_acc",
    },
    "cake_mhc_train_head_bwd__acc_lw64_t768_h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 768, "vector_elems": 4},
        "hidden": 5120,
        "kernel": "cake_mhc_train_head_bwd__acc_lw64_t768_h5120",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__acc_lw64_t768_h5120",
        "launch": {
            "block": (768, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__acc_lw64_t768_h5120",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd_acc",
    },
    "cake_mhc_train_head_bwd__acc_lw64_t768_h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 768, "vector_elems": 4},
        "hidden": 7168,
        "kernel": "cake_mhc_train_head_bwd__acc_lw64_t768_h7168",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__acc_lw64_t768_h7168",
        "launch": {
            "block": (768, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__acc_lw64_t768_h7168",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd_acc",
    },
    "cake_mhc_train_head_bwd__acc_l2red_lw128_t256_h4096": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 8},
        "hidden": 4096,
        "kernel": "cake_mhc_train_head_bwd__acc_l2red_lw128_t256_h4096",
        "kernel_symbol": "kernel_cake_mhc_train_head_bwd__acc_l2red_lw128_t256_h4096",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_head_bwd__acc_l2red_lw128_t256_h4096",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd_acc",
    },
}
# The record a public call launches on each architecture, by role and hidden
# size: the forward (``head_fwd``), the backward that writes ``dX``
# (``head_bwd``) and the backward that adds into it (``head_bwd_acc``,
# ``accumulate=True``).  :data:`SMALL_ROUTES` takes precedence for a small
# problem.
ROUTES: dict[str, dict[str, dict[int, str]]] = {
    "sm_100a": {
        "head_fwd": {
            4096: "cake_mhc_train_head_fwd__lw64_t256_h4096",
            5120: "cake_mhc_train_head_fwd__lw64_t256_h5120",
            7168: "cake_mhc_train_head_fwd__lw64_t256_h7168",
        },
        "head_bwd": {
            4096: "cake_mhc_train_head_bwd__lw64_t384_h4096",
            5120: "cake_mhc_train_head_bwd__lw64_t256_h5120",
            7168: "cake_mhc_train_head_bwd__lw64_t256_h7168",
        },
        "head_bwd_acc": {
            4096: "cake_mhc_train_head_bwd__acc_lw64_t768_h4096",
            5120: "cake_mhc_train_head_bwd__acc_lw64_t768_h5120",
            7168: "cake_mhc_train_head_bwd__acc_lw64_t768_h7168",
        },
    },
    "sm_103a": {
        "head_fwd": {
            4096: "cake_mhc_train_head_fwd__lw64_t128_h4096",
            5120: "cake_mhc_train_head_fwd__lw64_t128_h5120",
            7168: "cake_mhc_train_head_fwd__lw64_t128_h7168",
        },
        "head_bwd": {
            4096: "cake_mhc_train_head_bwd__lw64_t512_h4096",
            5120: "cake_mhc_train_head_bwd__lw64_t512_h5120",
            7168: "cake_mhc_train_head_bwd__lw64_t512_h7168",
        },
        "head_bwd_acc": {
            4096: "cake_mhc_train_head_bwd__acc_lw64_t768_h4096",
            5120: "cake_mhc_train_head_bwd__acc_lw64_t768_h5120",
            7168: "cake_mhc_train_head_bwd__acc_lw64_t768_h7168",
        },
    },
    "sm_107a": {
        "head_fwd": {
            4096: "cake_mhc_train_head_fwd__lw256_t128_h4096",
        },
        "head_bwd": {
            4096: "cake_mhc_train_head_bwd__lw256_t128_h4096",
        },
        "head_bwd_acc": {
            4096: "cake_mhc_train_head_bwd__acc_l2red_lw128_t256_h4096",
        },
    },
}
# A call of a served hidden size with at most :func:`small_elements` values per
# stream runs the unsuffixed library of its role where this names one: on
# sm_100a and sm_103a the backward that writes ``dX`` (its 256-bit vectors keep
# more bytes in flight per thread when the grid is small), and on sm_107a every
# role, whose hidden-4096 libraries are timed on the large problems.  The limit
# is ``SMALL_ELEMENTS`` (T 2048 at hidden 4096) but where
# ``SMALL_ELEMENTS_BY_ROUTE`` names another: the sm_107a forward's 128-thread
# CTAs of 256-bit vectors overtake the unsuffixed forward only past T 7168 at
# hidden 4096.
SMALL_ELEMENTS = 2048 * 4096
SMALL_ELEMENTS_BY_ROUTE: dict[str, dict[str, int]] = {
    "sm_107a": {"head_fwd": 7168 * 4096},
}
SMALL_ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {"head_bwd": "cake_mhc_train_head_bwd"},
    "sm_103a": {"head_bwd": "cake_mhc_train_head_bwd"},
    "sm_107a": {
        "head_fwd": "cake_mhc_train_head_fwd",
        "head_bwd": "cake_mhc_train_head_bwd",
        "head_bwd_acc": "cake_mhc_train_head_bwd__acc",
    },
}

# The architectures the head ships for: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``); a library is compiled for every architecture of its
# record with that architecture's exact ``-gencode`` flag set (``build.json``
# ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
STREAMS = 4
# The hidden sizes each architecture serves: the keys of each of its routes.
# SM107 serves 4096, as the rest of the package does, until its 5120 / 7168
# correctness and timing are recorded on R200.
HIDDEN_SIZES: dict[str, tuple[int, ...]] = {
    "sm_100a": (4096, 5120, 7168),
    "sm_103a": (4096, 5120, 7168),
    "sm_107a": (4096,),
}
# The kernels' launch preconditions ``T <= MAX_TOKENS`` and ``1 <= hv <=
# MAX_VECTORS``.  A kernel never forms the flat vector index ``T * hv``: within
# these bounds the grid, every dividend of its fast divisions and every token
# index stay below 2^31.
MAX_TOKENS = (1 << 22) - 1
MAX_VECTORS = 1 << 11


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_module(arch: str, role: str, hidden: int, rows: int) -> str:
    """The module of ``role`` on ``arch`` for ``rows`` tokens of ``hidden``
    values (``ValueError`` when none is exported)."""

    name = ROUTES.get(arch, {}).get(role, {}).get(int(hidden))
    if name is None:
        raise ValueError(
            f"no exported Cake mHC training head {role!r} module on {arch!r} for "
            f"hidden {hidden}; exported architectures: {ARCHES}, hidden sizes: "
            f"{HIDDEN_SIZES}"
        )
    small = SMALL_ROUTES.get(arch, {}).get(role)
    if small is not None and int(rows) * int(hidden) <= small_elements(arch, role):
        return small
    return name


def small_elements(arch: str, role: str) -> int:
    """The most values per stream a call of ``role`` on ``arch`` has for
    :data:`SMALL_ROUTES` to take it."""

    return SMALL_ELEMENTS_BY_ROUTE.get(arch, {}).get(role, SMALL_ELEMENTS)


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """Hidden sizes the head serves on ``arch`` (empty for an architecture it does not ship)."""

    return HIDDEN_SIZES.get(arch, ())


def vectors_per_token(record: dict[str, Any], hidden: int) -> int:
    """``hv``: the vectors of one token row of ``hidden`` values (the kernels' fast divisor)."""

    if record["hidden"] not in (0, hidden):
        raise ValueError(
            f"module {record['library']} is built for hidden {record['hidden']}, got {hidden}"
        )
    elems = int(record["grid_rule"]["vector_elems"])
    if hidden % elems:
        raise ValueError(
            f"hidden {hidden} is not a multiple of the {elems}-value vector"
        )
    return hidden // elems


def launch_grid(record: dict[str, Any], rows: int, hidden: int) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens of ``hidden`` values: one
    vector per thread, ``ceil(rows * hv / threads)`` CTAs.  ``rows`` must be
    positive; the public API returns empty outputs for zero rows without
    launching."""

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    rule = record["grid_rule"]
    if rule["kind"] != "vectors":
        raise ValueError(f"unknown Cake mHC training head grid rule: {rule['kind']!r}")
    return (
        _ceil_div(rows * vectors_per_token(record, hidden), int(rule["threads"])),
        1,
        1,
    )


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake mHC training head module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    from . import cake_build

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    from . import cake_build

    return cake_build.jit_specs(arch)
