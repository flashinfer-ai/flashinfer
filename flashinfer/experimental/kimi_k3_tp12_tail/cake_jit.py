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

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated programs of the
# Kimi-K3 TP12 fused LatentMoE communication tail (SM100 / SM103, twelve
# ranks in one multi-node NVLink domain).
#
# ``MODULES`` holds one record per generated program (a kernel plus its host
# binding): translation units, compile flags, FFI entry, argument plan, launch
# contract, closure identity and the architectures it is built for.
# ``KERNELS`` maps the logical kernel key to the program the host runtime
# resolves at preparation; both architectures run the same programs.  Every
# route is two generated kernels, K1 and the tail, with cuBLAS between them
# for ``M > 8``:
#
# * ``k1_oneshot_ess``  the one-shot Lamport all-reduce of the routed partial
#   fused with KimiRMSNorm for ``M <= 16`` tokens with the early shared
#   scatter (ESS): the same CTAs also scatter this rank's columns of the
#   shared partial into the K3 workspace slots of their owner ranks (the K3
#   workspace pointers and flags are extra arguments); the rank is a launch
#   argument that places the packets and selects the retained local packet in
#   the poll, the twelve-way sum stays in ascending rank order;
# * ``k1_twoshot_ess:grouped``  the token-sliced two-shot form (owner reduce +
#   norm, multicast broadcast) with the early shared scatter for
#   ``16 < M < 256``; ``grouped`` issues all twelve remote loads of the owner
#   retry body before the first test;
# * ``k1_twoshot:<grouped|pinned>``  the two-shot form without the shared
#   scatter for ``M >= 256`` (the persistent tail scatters the shared partial
#   itself); ``grouped`` at ``M = 256``, ``pinned`` (the per-rank pinned retry
#   body) above;
# * ``k23:c<4|8>``  the fused up-projection + tail for ``M <= 8``: the fp32
#   slice GEMM of the normalised latent with this rank's weight slice in the
#   K2-stream summation order, fused with the owner reduce of the
#   ESS-scattered shared columns, the add, one BF16 rounding and the multicast
#   all-gather of the output (the numerics of the round-4 K2-stream + fp32-add
#   K3 pair); one CTA per eight output columns, one program per accumulator
#   capacity (four tokens for ``M <= 4``, eight for ``5 <= M <= 8``); the
#   rank's column width only sizes the grid;
# * ``k3_ess:grouped``  the tail for ``4 < M < 256`` after an ESS K1: owner
#   reduce of the scattered shared columns, fused add of this rank's cuBLAS
#   up-projection slice, one BF16 rounding and the multicast all-gather of
#   the output row; one CTA per token and column half, no shared operand;
# * ``k3_persist:grouped``  the full K3 (own scatter of the shared partial,
#   owner reduce + add + rounding, all-gather) on a persistent grid of
#   ``min(M, SM count)`` CTAs per column half, each walking its tokens as a
#   three-stage pipeline (scatter ``t``, owner reduce + multicast ``t - P``,
#   gather ``t - 2P``) so the fabric hops of consecutive tokens overlap
#   (``M = 256``);
# * ``k3_persist_bulk:pinned``  the persistent K3 whose reduce-scatter stage
#   pushes the shared columns with ``cp.async.bulk``, pinned poll schedule
#   (``M > 256``).
#
# Every program is one source compiled for each architecture it lists (the
# SM100-family lowering preferences are ``__CUDA_ARCH__`` guards inside the
# source) and is launched with programmatic dependent launch.  Both literals
# are populated verbatim by the generated-program export; do not edit them by
# hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_tp12_tail_45df1981c8c6ea60f8e7": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_45df1981c8c6ea60f8e7_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_45df1981c8c6ea60f8e7_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "7147fd938d016ed5402187e8bf7569ced3087fddb05c2aa11ace9e1021a6bc2c",
    },
    "cake_kimi_k3_tp12_tail_4dafdc41b373cd2ba5a5": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_4dafdc41b373cd2ba5a5_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_4dafdc41b373cd2ba5a5_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "y"],
            ["buffer", "w_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 3840,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "b381dae992f05f5c2717579df742547ac66d02b448e72caa3dab1be923ecc819",
    },
    "cake_kimi_k3_tp12_tail_7919fc819bb5f04d4df0": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_7919fc819bb5f04d4df0_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_7919fc819bb5f04d4df0_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "y"],
            ["buffer", "w_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 1920,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "5ffd8ec71ad1e5cd33c64a3122616bb0dfcbb1b5f90d132033df0a3ce865d568",
    },
    "cake_kimi_k3_tp12_tail_8a4ce18ab9cd01e004d5": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_8a4ce18ab9cd01e004d5_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_8a4ce18ab9cd01e004d5_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "shared"],
            ["buffer", "gemm_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["parameter", "my_cols"],
            ["parameter", "gemm_plane_stride"],
            ["parameter", "num_gemm_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 14336,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "ecbde1127ad330bd73b92ea5934834ac8fafa86215862dcabfa53f6fbd38efdc",
    },
    "cake_kimi_k3_tp12_tail_dc6b04abf0dbc4001019": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_dc6b04abf0dbc4001019_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_dc6b04abf0dbc4001019_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "ada71f65e9943bfca679d2a23d8b16720258228044cce8450ea92ce76aff7f45",
    },
    "cake_kimi_k3_tp12_tail_dcff3505bf890a147662": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_dcff3505bf890a147662_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_dcff3505bf890a147662_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "shared"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["buffer", "k3_peer_ptrs"],
            ["raw_pointer", "k3_mcast_ptr"],
            ["buffer", "k3_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "f947815b792d2d3a3e1c9e3c4051ff4321b7462a4f1ef4e4e96aedafa7a3938b",
    },
    "cake_kimi_k3_tp12_tail_e604573d36e1bb117b68": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_e604573d36e1bb117b68_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_e604573d36e1bb117b68_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "routed"],
            ["buffer", "shared"],
            ["buffer", "y_out"],
            ["buffer", "gamma"],
            ["raw_pointer", "mcast_ptr"],
            ["raw_pointer", "local_unicast_ptr"],
            ["buffer", "buffer_flags"],
            ["buffer", "k3_peer_ptrs"],
            ["raw_pointer", "k3_mcast_ptr"],
            ["buffer", "k3_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "epsilon"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "a527ddb456b2926ffe7ecbccb2ee3bb719db5023577eb18b1c3c4ce4d5ea5fff",
    },
    "cake_kimi_k3_tp12_tail_f1ce43c00dd2fc5bc34e": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_f1ce43c00dd2fc5bc34e_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_f1ce43c00dd2fc5bc34e_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "gemm_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["parameter", "my_cols"],
            ["parameter", "gemm_plane_stride"],
            ["parameter", "num_gemm_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "5a82ab4ac6c5a42c3d88fa23095684400c96037850f8010c95322df9b4595f2d",
    },
    "cake_kimi_k3_tp12_tail_f8e15786d6ad190fa245": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_f8e15786d6ad190fa245_kernel.cu",
            "cake_kimi_k3_tp12_tail/cake_kimi_k3_tp12_tail_f8e15786d6ad190fa245_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "shared"],
            ["buffer", "gemm_slice"],
            ["buffer", "out"],
            ["buffer", "peer_ptrs"],
            ["raw_pointer", "mcast_ptr"],
            ["buffer", "buffer_flags"],
            ["parameter", "num_tokens"],
            ["parameter", "rank"],
            ["parameter", "my_col_begin"],
            ["parameter", "my_cols"],
            ["parameter", "gemm_plane_stride"],
            ["parameter", "num_gemm_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [448, 1, 1],
            "cluster": [1, 1, 1],
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": "1a1f7251f922bd803931ebe1d3666450720ed0443317003912deebd42b16ea17",
    },
}

KERNELS: dict[str, str] = {
    "k1_oneshot_ess": "cake_kimi_k3_tp12_tail_e604573d36e1bb117b68",
    "k1_twoshot:grouped": "cake_kimi_k3_tp12_tail_45df1981c8c6ea60f8e7",
    "k1_twoshot:pinned": "cake_kimi_k3_tp12_tail_dc6b04abf0dbc4001019",
    "k1_twoshot_ess:grouped": "cake_kimi_k3_tp12_tail_dcff3505bf890a147662",
    "k23:c4": "cake_kimi_k3_tp12_tail_7919fc819bb5f04d4df0",
    "k23:c8": "cake_kimi_k3_tp12_tail_4dafdc41b373cd2ba5a5",
    "k3_ess:grouped": "cake_kimi_k3_tp12_tail_f1ce43c00dd2fc5bc34e",
    "k3_persist:grouped": "cake_kimi_k3_tp12_tail_f8e15786d6ad190fa245",
    "k3_persist_bulk:pinned": "cake_kimi_k3_tp12_tail_8a4ce18ab9cd01e004d5",
}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
WORLD_SIZE = 12
POLL_SCHEDULES = ("grouped", "pinned")
#: K23 accumulator capacities (one program each): four tokens for ``M <= 4``, eight for ``5 <= M <= 8``.
K23_CAPACITIES = (4, 8)


def _nvcc_flags(arches: list[str]) -> list[str]:
    """The code-generation flags of every architecture a program lists, followed by the common flags, each once."""
    flags: list[str] = []
    for arch in arches:
        for flag in ARCH_NVCC_FLAGS[arch]:
            if flag not in flags:
                flags.append(flag)
    return flags


def required_kernel_keys() -> tuple[str, ...]:
    """Every logical kernel the runtime can select (nine keys, the same programs on both architectures)."""
    # Unreachable, hence not registered: ``k1_twoshot_ess:pinned`` and ``k3_ess:pinned``
    # (the ESS range ``M < 256`` is grouped-only), ``k3_persist:pinned`` and
    # ``k3_persist_bulk:grouped`` (the plain persistent K3 serves exactly 256 tokens,
    # the cp.async.bulk form the whole pinned range ``M > 256``).
    return (
        "k1_oneshot_ess",
        "k1_twoshot_ess:grouped",
        *(f"k1_twoshot:{schedule}" for schedule in POLL_SCHEDULES),
        *(f"k23:c{capacity}" for capacity in K23_CAPACITIES),
        "k3_ess:grouped",
        "k3_persist:grouped",
        "k3_persist_bulk:pinned",
    )


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when every key in ``required_keys`` is registered with a program built for ``arch``."""
    if arch not in ARCHES or not KERNELS:
        return False
    return all(
        key in KERNELS and arch in MODULES[KERNELS[key]]["arches"]
        for key in required_keys
    )


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered program of logical kernel ``key``, checked to be built for ``arch``."""
    if arch not in ARCHES or not KERNELS:
        raise NotImplementedError(
            f"The generated Kimi-K3 TP12 tail programs for {arch} are not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#4542)"
        )
    name = KERNELS.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 TP12 tail kernel {key!r} is not "
            "registered in this checkout (see flashinfer-ai/flashinfer#4542)"
        )
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"registered program {name!r} is not built for {arch} (arches: {record['arches']})"
        )
    return name


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_kimi_k3_tp12_tail_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[*_nvcc_flags(record["arches"]), *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_kimi_k3_tp12_tail_module(name: str):
    return gen_cake_kimi_k3_tp12_tail_module(name).build_and_load()
