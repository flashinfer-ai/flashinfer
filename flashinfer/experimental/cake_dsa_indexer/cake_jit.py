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
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Explicit target-owned registration of the generated indexer programs.  One
# record per architecture (``sm_100a``, ``sm_103a``, ``sm_107a``).  A record
# carries ``arch``, the host binding profile ``abi`` (the keyword set its
# kernels expect, see ``cake_backend.CONTRACT_TENSORS`` / ``CONTRACT_SCALARS``),
# the list of kernel ``stages`` it registers, the host-evaluated candidate-gate
# policy ``gate_policy`` (see ``cake_backend.GatePolicy``), the documented
# numerics of the program (``numerics``: the zero-sign policy of the head
# reduction), and one physical entry per stage (translation units, compile
# flags, FFI entry, argument plan, grid rule, launch geometry and closure
# identity).
#
# PLACEHOLDER: the registry is empty until the generated-program export lands.
# ``select_module`` raises ``NotImplementedError`` for every architecture,
# ``cake_backend.generated_program_available`` returns ``False`` and the GPU
# tests skip.  Populated verbatim by the export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_indexer_topk_sm_100a": {
        "arch": "sm_100a",
        "abi": "dsa_indexer_v1",
        "stages": ["scan", "finalize", "finalize_small"],
        "gate_policy": {
            "queries_per_cta": 4,
            "candidate_entry_bytes": 8,
            "candidate_multiplier": 4,
            "candidate_slack": 128,
            "tile_keys": 128,
            "check_period_max": 32,
            "check_period_cap_divisor": 512,
            "sample_tiles_max": 32,
            "sample_shift_permille": 250,
            "finalize_small_max_top_k": 2048,
        },
        "numerics": {"zero_sign_policy": "positive_accumulator"},
        "scan": {
            "module": "cake_dsa_indexer_topk_ced38ebf705fb756eaf8",
            "sources": [
                "cake_dsa_indexer_topk/sm_100a/cake_dsa_indexer_topk_ced38ebf705fb756eaf8_kernel.cu",
                "cake_dsa_indexer_topk/sm_100a/cake_dsa_indexer_topk_ced38ebf705fb756eaf8_binding.cu",
            ],
            "compile_flags": ["--ptxas-options=--register-usage-level=10"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "W"],
                ["buffer", "cu_seqlens_q"],
                ["buffer", "cu_seqlens_k"],
                ["buffer", "q_offsets"],
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["buffer", "Cand"],
                ["parameter", "num_segments"],
                ["parameter", "top_k"],
                ["parameter", "ratio"],
                ["parameter", "has_offsets"],
                ["parameter", "cand_cap"],
                ["parameter", "first_cap"],
                ["parameter", "sample_tiles_max"],
                ["parameter", "sample_shift_permille"],
                ["parameter", "check_period"],
                ["parameter", "grid_ctas"],
                ["parameter", "softmax_scale"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "0509ec60f7be7b5bfbcdfe1d7fc4866fdc9420314e26ae47bb21ad7e30756ddd",
            "workspace_bytes": 0,
            "grid": ["sms", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "finalize": {
            "module": "cake_dsa_indexer_topk_8349f8ea2dea76b31e67",
            "sources": [
                "cake_dsa_indexer_topk/sm_100a/cake_dsa_indexer_topk_8349f8ea2dea76b31e67_kernel.cu",
                "cake_dsa_indexer_topk/sm_100a/cake_dsa_indexer_topk_8349f8ea2dea76b31e67_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["parameter", "top_k"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "1c548782c61335afbc92c71afa5a544559ad97e26a9a83f43ca3f6acdad3d77e",
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
        },
        "finalize_small": {
            "module": "cake_dsa_indexer_topk_5236b280c317fcb283a7",
            "sources": [
                "cake_dsa_indexer_topk/sm_100a/cake_dsa_indexer_topk_5236b280c317fcb283a7_kernel.cu",
                "cake_dsa_indexer_topk/sm_100a/cake_dsa_indexer_topk_5236b280c317fcb283a7_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["parameter", "top_k"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "26ea7537f84d2292cbba8deb58b1aff8267aabfd47e005ceed6156ff7f3ce374",
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "2f49a1d36098705e2eb6f862ae9b9c4c67608dc48334b417dc675a58e9dcd95a",
    },
    "cake_dsa_indexer_topk_sm_103a": {
        "arch": "sm_103a",
        "abi": "dsa_indexer_v1",
        "stages": ["scan", "finalize", "finalize_small"],
        "gate_policy": {
            "queries_per_cta": 4,
            "candidate_entry_bytes": 8,
            "candidate_multiplier": 4,
            "candidate_slack": 128,
            "tile_keys": 128,
            "check_period_max": 32,
            "check_period_cap_divisor": 512,
            "sample_tiles_max": 32,
            "sample_shift_permille": 250,
            "finalize_small_max_top_k": 2048,
        },
        "numerics": {"zero_sign_policy": "positive_accumulator"},
        "scan": {
            "module": "cake_dsa_indexer_topk_c3b23e89f972021a2a22",
            "sources": [
                "cake_dsa_indexer_topk/sm_103a/cake_dsa_indexer_topk_c3b23e89f972021a2a22_kernel.cu",
                "cake_dsa_indexer_topk/sm_103a/cake_dsa_indexer_topk_c3b23e89f972021a2a22_binding.cu",
            ],
            "compile_flags": ["--ptxas-options=--register-usage-level=10"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "W"],
                ["buffer", "cu_seqlens_q"],
                ["buffer", "cu_seqlens_k"],
                ["buffer", "q_offsets"],
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["buffer", "Cand"],
                ["parameter", "num_segments"],
                ["parameter", "top_k"],
                ["parameter", "ratio"],
                ["parameter", "has_offsets"],
                ["parameter", "cand_cap"],
                ["parameter", "first_cap"],
                ["parameter", "sample_tiles_max"],
                ["parameter", "sample_shift_permille"],
                ["parameter", "check_period"],
                ["parameter", "grid_ctas"],
                ["parameter", "softmax_scale"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "99bbb5df72ae298caf0c4b229298441ee5ab6757b66de8a2b9bb22c2baa45961",
            "workspace_bytes": 0,
            "grid": ["sms", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "finalize": {
            "module": "cake_dsa_indexer_topk_41ae90bb45be5f611933",
            "sources": [
                "cake_dsa_indexer_topk/sm_103a/cake_dsa_indexer_topk_41ae90bb45be5f611933_kernel.cu",
                "cake_dsa_indexer_topk/sm_103a/cake_dsa_indexer_topk_41ae90bb45be5f611933_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["parameter", "top_k"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "917d46c3af00c6c212ec26fd0ba90b3535e4d70e66bf1a380f7511b481b87be3",
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
        },
        "finalize_small": {
            "module": "cake_dsa_indexer_topk_219065b3049d18cbe817",
            "sources": [
                "cake_dsa_indexer_topk/sm_103a/cake_dsa_indexer_topk_219065b3049d18cbe817_kernel.cu",
                "cake_dsa_indexer_topk/sm_103a/cake_dsa_indexer_topk_219065b3049d18cbe817_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["parameter", "top_k"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "fdd84aa06e2facccd13d4e6096941180d70cf34f7c91819d991e9695c97336c4",
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "24febc5447e7528c782ea27b21962fc2a60c7c27f35538447dd2ddd87078fabc",
    },
    "cake_dsa_indexer_topk_sm_107a": {
        "arch": "sm_107a",
        "abi": "dsa_indexer_v1",
        "stages": ["scan", "finalize", "finalize_small"],
        "gate_policy": {
            "queries_per_cta": 4,
            "candidate_entry_bytes": 8,
            "candidate_multiplier": 4,
            "candidate_slack": 128,
            "tile_keys": 128,
            "check_period_max": 32,
            "check_period_cap_divisor": 1024,
            "sample_tiles_max": 32,
            "sample_shift_permille": 250,
            "finalize_small_max_top_k": 2048,
        },
        "numerics": {"zero_sign_policy": "positive_accumulator"},
        "scan": {
            "module": "cake_dsa_indexer_topk_d28c37b28e080b362652",
            "sources": [
                "cake_dsa_indexer_topk/sm_107a/cake_dsa_indexer_topk_d28c37b28e080b362652_kernel.cu",
                "cake_dsa_indexer_topk/sm_107a/cake_dsa_indexer_topk_d28c37b28e080b362652_binding.cu",
            ],
            "compile_flags": ["--ptxas-options=--register-usage-level=10"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "W"],
                ["buffer", "cu_seqlens_q"],
                ["buffer", "cu_seqlens_k"],
                ["buffer", "q_offsets"],
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["buffer", "Cand"],
                ["parameter", "num_segments"],
                ["parameter", "top_k"],
                ["parameter", "ratio"],
                ["parameter", "has_offsets"],
                ["parameter", "cand_cap"],
                ["parameter", "first_cap"],
                ["parameter", "sample_tiles_max"],
                ["parameter", "sample_shift_permille"],
                ["parameter", "check_period"],
                ["parameter", "grid_ctas"],
                ["parameter", "softmax_scale"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "7cf44c8efc0499bc099f1de042858caa3c7ae338e64ca5eb068bd11a6882f22e",
            "workspace_bytes": 0,
            "grid": ["sms", 1, 1],
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "finalize": {
            "module": "cake_dsa_indexer_topk_509a92e74c3031dae9f4",
            "sources": [
                "cake_dsa_indexer_topk/sm_107a/cake_dsa_indexer_topk_509a92e74c3031dae9f4_kernel.cu",
                "cake_dsa_indexer_topk/sm_107a/cake_dsa_indexer_topk_509a92e74c3031dae9f4_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["parameter", "top_k"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "19b9cda36b04bb23d8a2720e1be6ca948686558aa76556d377462a5d69b8be40",
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [512, 1, 1], "cluster": [1, 1, 1]},
        },
        "finalize_small": {
            "module": "cake_dsa_indexer_topk_0d0c2fbabc742a52364a",
            "sources": [
                "cake_dsa_indexer_topk/sm_107a/cake_dsa_indexer_topk_0d0c2fbabc742a52364a_kernel.cu",
                "cake_dsa_indexer_topk/sm_107a/cake_dsa_indexer_topk_0d0c2fbabc742a52364a_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Indices"],
                ["buffer", "Scores"],
                ["parameter", "top_k"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "adab6043eb074f02708b80910cbf2aa6f5f96d44dea63269a8499fcc5f7d3987",
            "workspace_bytes": 0,
            "grid": ["num_queries", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "85bbe2b9e62b8e4bb84c91504c4ff65a67ff325855e282387494dc005133e580",
    },
}

# Kernel stages of one indexer call, in launch order.  ``scan`` is the
# persistent fused kernel (scoring, exact candidate gate, per-row selection; it
# writes the unordered selected (id, score) pairs and the padding straight into
# the outputs); ``finalize`` / ``finalize_small`` sort every row by ascending
# key id in place (one CTA per row).  A record registers ``scan`` and at least
# one finalize stage; ``finalize_small`` serves ``top_k <=
# gate_policy["finalize_small_max_top_k"]`` and ``finalize`` every ``top_k``.
STAGES = ("scan", "finalize", "finalize_small")
FINALIZE_STAGES = ("finalize", "finalize_small")
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 / SM107.)"""
    return arch in ARCH_NVCC_FLAGS


def registered_archs() -> tuple[str, ...]:
    """Architectures with a registered program, in ``ARCHES`` order."""
    present = {record["arch"] for record in MODULES.values()}
    return tuple(arch for arch in ARCHES if arch in present)


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch``."""
    names = [name for name, record in MODULES.items() if record["arch"] == arch]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one DSA indexer program: {names}"
        )
    if not names:
        raise NotImplementedError(
            f"The generated DSA indexer top-k program for {arch} is not registered "
            "in this checkout yet (see flashinfer-ai/flashinfer#5676)"
        )
    return names[0]


def registered_stages(name: str) -> tuple[str, ...]:
    """Stages a record registers, in launch order."""
    present = tuple(stage for stage in STAGES if stage in MODULES[name])
    declared = tuple(MODULES[name].get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record {name!r} declares stages {declared} but carries {present}"
        )
    return present


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
def gen_cake_dsa_indexer_module(name: str, stage: str):
    record = MODULES[name]
    if not toolchain_supports(record["arch"]):
        raise RuntimeError(
            f"generated DSA indexer program {name!r} targets {record['arch']}, "
            "which this checkout cannot compile"
        )
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_dsa_indexer_module(name: str, stage: str):
    return gen_cake_dsa_indexer_module(name, stage).build_and_load()
