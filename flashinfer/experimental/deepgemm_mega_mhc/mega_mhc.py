"""Prepared fused projection, Sinkhorn, residual mixing and RMSNorm on SM100a/SM103a/SM107a.

The FP8-output programs (hidden 5120, SM100a / SM103a) are one generated source
per schedule class (shifted x shared scale layout x the unroll and partition
choices of a split-count class); the loader compiles each with the exact flag
set of the device it runs on and passes the split count, its FP8 norm partition
count and the device SM count on the compile line. The hidden-4096 programs
(normal mode, SM107a) take the split count and the launch grid as kernel
arguments, so one JIT module serves every SM count: the three BF16-output programs
(``sf_layout="bf16"``: the tailpipe program at the split counts up to 22, where
the tail's round-1 loads overlap the fold, its weight-staging twin at the counts
up to 16 when the tasks run in five or more waves of the grid, where the tail
reads the RMSNorm weight staged once per CTA in shared memory, and the BF16
program at 32 and 64 splits, chosen by ``mhc_infer_tail_program`` from the token
count and the SM count; all three write bitwise the same output) and the
quantized-output variants,
which store the same BF16 output and beside it its quantization: ``"fp8_group128"`` the FP8 E4M3
values with one FP32 scale per 128-element group of a token (vLLM's
``per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=fp8_sf_ue8m0)``), ``"nvfp4"``
the packed NVFP4 values with one E4M3 scale per 16-element group under a global
scale (vLLM's ``scaled_fp4_quant(y_bf16, fp4_global_scale)`` in the linear scale
layout). The token count is a run-time argument everywhere: the split count and
the launch grid are chosen here from the same rules the producer uses.
"""

from __future__ import annotations

import functools
import threading
import weakref
from typing import Any

GENERATED_ROOT = "csrc/experimental/deepgemm_mega_mhc/generated"
CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
MAX_TOKENS = 1 << 20
SPLIT_BARRIER_WORDS = 2 * (MAX_TOKENS // 64) * 16
# The BF16-output route: hidden size, the pieces and padded rows of its split projection operand.
BF16_HIDDEN = 4096
FN_ROWS = 24
FNP_PIECES = 3
FNP_ROWS = 32
# The output layouts a route serves: the FP8 layouts ("col" and "extra") share the FP8 programs; a quantized
# layout of the hidden-4096 route is a program of its own, storing the BF16 output and its quantization.
FP8_LAYOUTS = ("col", "extra")
QUANT_LAYOUTS = ("fp8_group128", "nvfp4")
# The architectures whose exported programs include the quantized variants' tail forms (mhc_infer_variant_program).
VARIANT_TAIL_ARCHES = ("sm_107a",)
FP8_GROUP = 128
FP4_GROUP = 16

# Registry written by the exporter: one record per delivered source, and the
# schedule keys (hidden, split count, shifted, shared scale layout, output
# layout) each source serves with its compile-line defines. A record may spell
# its own ``arches``, ``compile_flags``, ``compile_options`` (the source build's
# numerics, passed after ``--fmad=false``), ``arg_plan``, ``launch`` and
# ``compile_time_sm_count``; a record without them takes the module-level
# ARCHES, COMPILE_FLAGS and ARG_PLAN and the device SM count on the compile
# line. A route with ``num_splits: "runtime"`` passes the split count to the
# kernel instead of compiling it in; its ``sf_layout`` is "bf16" for the
# BF16-output programs and "fp8" (the default) for the FP8-output programs.
# A layout served by several programs keys each of its routes on ``regime``,
# the name the host rule ``mhc_infer_tail_program`` gives the split regime of
# a (token count, SM count): "bf16" (32 and 64 splits), "tailpipe" (the counts
# up to 22) and "tailpipe_wsmem" (the counts up to 16 over five or more task
# waves); a route without ``regime`` serves every count of its key.
# The second source of every record is the one shared host binding
# (cake_mega_mhc_134570ae7cc112e9d84e_binding.cu): the exported bindings differed only in the
# kernel symbol they launch and their host-shim namespace id, so load_program passes
# the record's kernel_symbol as the CAKE_PROGRAM_KERNEL compile definition.
# The four 40-split records build from the 27-split kernel source of their class:
# the two sources differed only at sites now behind `#if NUM_SPLITS == 40` arms, which
# the route's NUM_SPLITS define selects; each record keeps its own kernel symbol, so
# the kept sources spell their kernel name through CAKE_PROGRAM_KERNEL as well.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_mega_mhc_134570ae7cc112e9d84e": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_134570ae7cc112e9d84e",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [16],
    },
    "cake_mega_mhc_1576b7789968f04c8419": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_1576b7789968f04c8419_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_1576b7789968f04c8419",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [16, 20],
    },
    "cake_mega_mhc_2b99261aab173a941cb4": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_2b99261aab173a941cb4_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_2b99261aab173a941cb4",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [16, 20],
    },
    "cake_mega_mhc_407321791585dc9fc44b": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_407321791585dc9fc44b_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_407321791585dc9fc44b",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [27],
    },
    "cake_mega_mhc_41df2a8c4d3c4849adbf": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_b8c6c9ebebc80eb063b8_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_41df2a8c4d3c4849adbf",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [40],
    },
    "cake_mega_mhc_43b829c4b071416a4595": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_43b829c4b071416a4595_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_43b829c4b071416a4595",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [27],
    },
    "cake_mega_mhc_8fd99969edc5072d3477": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_8fd99969edc5072d3477_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_8fd99969edc5072d3477",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [16],
    },
    "cake_mega_mhc_a5521c9dfaf81d2c0f1f": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_43b829c4b071416a4595_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_a5521c9dfaf81d2c0f1f",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [40],
    },
    "cake_mega_mhc_b52ed5c3afc2fcbd2d0f": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_b52ed5c3afc2fcbd2d0f_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_b52ed5c3afc2fcbd2d0f",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [20],
    },
    "cake_mega_mhc_b73ed09be74eaa063166": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_b73ed09be74eaa063166_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_b73ed09be74eaa063166",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [20],
    },
    "cake_mega_mhc_b8c6c9ebebc80eb063b8": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_b8c6c9ebebc80eb063b8_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_b8c6c9ebebc80eb063b8",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [27],
    },
    "cake_mega_mhc_c7892aa3a7b985dcac4a": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_407321791585dc9fc44b_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_c7892aa3a7b985dcac4a",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [40],
    },
    "cake_mega_mhc_dc85bbfab3b20cf209fc": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_dc85bbfab3b20cf209fc_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_dc85bbfab3b20cf209fc",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [27],
    },
    "cake_mega_mhc_f7107723c112a3b117a9": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_dc85bbfab3b20cf209fc_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_134570ae7cc112e9d84e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_f7107723c112a3b117a9",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [40],
    },
    "cake_mega_mhc_f24c2f3f98bfe3356398": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f24c2f3f98bfe3356398_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f24c2f3f98bfe3356398_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_f24c2f3f98bfe3356398",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a", "sm_100a", "sm_103a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_c549b064530c3765f9ba": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_c549b064530c3765f9ba_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_c549b064530c3765f9ba_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_c549b064530c3765f9ba",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a", "sm_100a", "sm_103a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_41024b807c02040677b3": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_41024b807c02040677b3_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_41024b807c02040677b3_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_41024b807c02040677b3",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a", "sm_100a", "sm_103a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 217088,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_a940c94296afad2b2497": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_a940c94296afad2b2497_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_a940c94296afad2b2497_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_a940c94296afad2b2497",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a", "sm_100a", "sm_103a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp8"],
            ["buffer", "y_fp8_sf"],
            ["parameter", "fp8_sf_ue8m0"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_1c61fed11caf93d1b762": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_1c61fed11caf93d1b762_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_1c61fed11caf93d1b762_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_1c61fed11caf93d1b762",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a", "sm_100a", "sm_103a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp4"],
            ["buffer", "y_fp4_sf"],
            ["buffer", "fp4_global_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_ed3f407e5a68eb65478e": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_ed3f407e5a68eb65478e_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_ed3f407e5a68eb65478e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_ed3f407e5a68eb65478e",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp8"],
            ["buffer", "y_fp8_sf"],
            ["parameter", "fp8_sf_ue8m0"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_892d774130f06423fd7c": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_892d774130f06423fd7c_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_892d774130f06423fd7c_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_892d774130f06423fd7c",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp8"],
            ["buffer", "y_fp8_sf"],
            ["parameter", "fp8_sf_ue8m0"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 217088,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_91c9979888321fdbf94f": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_91c9979888321fdbf94f_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_91c9979888321fdbf94f_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_91c9979888321fdbf94f",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp8"],
            ["buffer", "y_fp8_sf"],
            ["parameter", "fp8_sf_ue8m0"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_9823742e18a9ffa6515b": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_9823742e18a9ffa6515b_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_9823742e18a9ffa6515b_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_9823742e18a9ffa6515b",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp8"],
            ["buffer", "y_fp8_sf"],
            ["parameter", "fp8_sf_ue8m0"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_5c5eaa3072b0d4f69ccd": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_5c5eaa3072b0d4f69ccd_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_5c5eaa3072b0d4f69ccd_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_5c5eaa3072b0d4f69ccd",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp8"],
            ["buffer", "y_fp8_sf"],
            ["parameter", "fp8_sf_ue8m0"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 217088,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_5018692c75652529e6fd": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_5018692c75652529e6fd_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_5018692c75652529e6fd_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_5018692c75652529e6fd",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp4"],
            ["buffer", "y_fp4_sf"],
            ["buffer", "fp4_global_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 208896,
            "cooperative": True,
            "use_pdl": False,
        },
    },
    "cake_mega_mhc_12e5bcc41e7940e7d511": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_12e5bcc41e7940e7d511_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_12e5bcc41e7940e7d511_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_12e5bcc41e7940e7d511",
        "hidden": 4096,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": "runtime",
        "arches": ["sm_107a"],
        "compile_flags": [],
        "compile_options": ["--fmad=true"],
        "compile_time_sm_count": False,
        "arg_plan": [
            ["tma_buffer", "residual_map"],
            ["tma_buffer", "x_map"],
            ["tma_buffer", "fnp_map"],
            ["tma_buffer", "post_map"],
            ["tma_buffer", "comb_map"],
            ["tma_buffer", "new_residual_map"],
            ["buffer", "mix_scales"],
            ["buffer", "mix_bases"],
            ["buffer", "new_post_mix"],
            ["buffer", "new_comb_res_mix"],
            ["buffer", "rmsnorm_weight"],
            ["buffer", "new_residual"],
            ["buffer", "y_bf16"],
            ["buffer", "scratch"],
            ["buffer", "split_barriers"],
            ["buffer", "launch_epochs"],
            ["parameter", "num_tokens"],
            ["parameter", "num_splits"],
            ["parameter", "split_div"],
            ["parameter", "num_ctas"],
            ["parameter", "hc_norm_eps"],
            ["parameter", "hc_pre_eps"],
            ["parameter", "hc_post_scale"],
            ["parameter", "sinkhorn_eps"],
            ["parameter", "num_sinkhorn_iters"],
            ["parameter", "rmsnorm_eps"],
            ["parameter", "rmsnorm_scale"],
            ["buffer", "y_fp4"],
            ["buffer", "y_fp4_sf"],
            ["buffer", "fp4_global_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {
            "block": [768, 1, 1],
            "dynamic_smem_bytes": 217088,
            "cooperative": True,
            "use_pdl": False,
        },
    },
}
ROUTES: list[dict[str, Any]] = [
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_2b99261aab173a941cb4",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_1576b7789968f04c8419",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_134570ae7cc112e9d84e",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_8fd99969edc5072d3477",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_2b99261aab173a941cb4",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_1576b7789968f04c8419",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_b52ed5c3afc2fcbd2d0f",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_b73ed09be74eaa063166",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_dc85bbfab3b20cf209fc",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_407321791585dc9fc44b",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_b8c6c9ebebc80eb063b8",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_43b829c4b071416a4595",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_f7107723c112a3b117a9",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_c7892aa3a7b985dcac4a",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_41df2a8c4d3c4849adbf",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_a5521c9dfaf81d2c0f1f",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "bf16",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "bf16",
        "program": "cake_mega_mhc_f24c2f3f98bfe3356398",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "bf16",
        "program": "cake_mega_mhc_c549b064530c3765f9ba",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe_wsmem",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "bf16",
        "program": "cake_mega_mhc_41024b807c02040677b3",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "bf16",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "fp8_group128",
        "fp8_sf_ue8m0": True,
        "program": "cake_mega_mhc_91c9979888321fdbf94f",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "fp8_group128",
        "fp8_sf_ue8m0": True,
        "program": "cake_mega_mhc_9823742e18a9ffa6515b",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe_wsmem",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "fp8_group128",
        "fp8_sf_ue8m0": True,
        "program": "cake_mega_mhc_5c5eaa3072b0d4f69ccd",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "bf16",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "fp8_group128",
        "program": "cake_mega_mhc_a940c94296afad2b2497",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "bf16",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "nvfp4",
        "program": "cake_mega_mhc_1c61fed11caf93d1b762",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "fp8_group128",
        "program": "cake_mega_mhc_ed3f407e5a68eb65478e",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe_wsmem",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "fp8_group128",
        "program": "cake_mega_mhc_892d774130f06423fd7c",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "nvfp4",
        "program": "cake_mega_mhc_5018692c75652529e6fd",
        "defines": {},
    },
    {
        "hidden": 4096,
        "num_splits": "runtime",
        "regime": "tailpipe_wsmem",
        "shifted": False,
        "shared_sf": False,
        "sf_layout": "nvfp4",
        "program": "cake_mega_mhc_12e5bcc41e7940e7d511",
        "defines": {},
    },
]
ARG_PLAN: list[list[str]] = [
    ["tma_buffer", "residual_map"],
    ["tma_buffer", "x_map"],
    ["tma_buffer", "fn_map"],
    ["tma_buffer", "post_map"],
    ["tma_buffer", "comb_map"],
    ["tma_buffer", "prev_map"],
    ["tma_buffer", "new_residual_map"],
    ["tma_buffer", "y_map"],
    ["buffer", "mix_scales"],
    ["buffer", "mix_bases"],
    ["buffer", "new_prev_mix"],
    ["buffer", "new_post_mix"],
    ["buffer", "new_comb_res_mix"],
    ["buffer", "rmsnorm_weight"],
    ["buffer", "new_residual"],
    ["buffer", "y_bf16"],
    ["buffer", "y_fp8"],
    ["buffer", "y_primary_sf"],
    ["buffer", "y_shared_sf"],
    ["buffer", "scratch"],
    ["buffer", "split_barriers"],
    ["buffer", "launch_epochs"],
    ["parameter", "num_tokens"],
    ["parameter", "hc_norm_eps"],
    ["parameter", "hc_pre_eps"],
    ["parameter", "hc_post_scale"],
    ["parameter", "sinkhorn_eps"],
    ["parameter", "num_sinkhorn_iters"],
    ["parameter", "rmsnorm_eps"],
    ["parameter", "rmsnorm_scale"],
    ["parameter", "primary_sf_stride_token"],
    ["parameter", "primary_sf_stride_word"],
    ["parameter", "shared_sf_stride_word"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
COMPILE_FLAGS: list[str] = ["--use_fast_math"]
ARCHES: list[str] = ["sm_100a", "sm_103a"]


def num_splits(num_tokens, hidden, num_sms, deterministic=False):
    """Fewest splits reaching the shortest longest task (deterministic mode fixes 16)."""
    if deterministic:
        return 16
    num_m_blocks = (num_tokens + 63) // 64
    blocks = hidden // 64
    max_splits = min(64, max(16, num_sms // num_m_blocks))
    longest = (blocks + max_splits - 1) // max_splits
    return (blocks + longest - 1) // longest


def num_launch_sms(
    num_tokens, hidden, num_sms, splits, shifted, store_fp8, has_gemm_sf, deterministic
):
    """Active CTAs of the launch; the kernel keeps its NUM_SMS scheduler stride."""
    if (
        num_sms in (148, 152)
        and hidden == 5120
        and not deterministic
        and shifted
        and store_fp8
        and num_tokens in (1, 64, 65)
    ):
        # Shifted FP8 T1/T64/T65 launch only the CTAs some role can reach.
        projection_ctas = ((num_tokens + 63) // 64) * splits
        mix_ctas = (num_tokens + 3) // 4
        norm_partitions = hidden // 512 if splits > 16 and hidden // 256 > 8 else 2
        if norm_partitions > 2:
            norm_partitions = min(
                norm_partitions, max(2, num_sms * 4 // ((num_tokens + 1) // 2))
            )
        norm_ctas = (((num_tokens + 1) // 2) * norm_partitions + 3) // 4
        return min(num_sms, max(projection_ctas, mix_ctas, norm_ctas))
    if (
        num_sms in (148, 152)
        and hidden == 5120
        and not deterministic
        and not shifted
        and num_tokens == 65
        and store_fp8
        and has_gemm_sf
    ):
        return 80
    return num_sms


# Host rules of the BF16-output programs (hidden 4096, 4 routes, 64 K blocks of 64
# hidden x 4 routes): one text shared with the producer's own host rules and its
# kernel tests, which check the three copies against each other.

TAIL_PIPE_MAX_SPLITS = 22  # the largest split count the tailpipe program is launched at (mhc_infer_tail_program)
# The tailpipe_wsmem program is launched at the split counts up to WEIGHT_SMEM_MAX_SPLITS
# when the tasks run in at least WEIGHT_SMEM_MIN_WAVES waves of the grid (mhc_infer_tail_program).
WEIGHT_SMEM_MAX_SPLITS = 16
WEIGHT_SMEM_MIN_WAVES = 5
VARIANT_WSMEM_REGIME_SPLITS = 8  # the quantized variants' count in that regime
VARIANT_PLAIN_MIN_WAVES = 7  # ... from this many task waves of the production count on


def mhc_infer_num_splits(
    num_tokens: int, num_sms: int, deterministic: bool = False
) -> int:
    """Splits of the 64 K blocks (H4096, 4 routes), 16 when deterministic. The deepgemm_mega_mhc count fills one wave
    of the SMs with the K blocks divided evenly, never below 16; when the tokens fit two rounds of the tail (twelve
    tokens per CTA of the grid), a smaller count of at least ten whose tasks fit two waves of the grid and whose
    waves x K blocks per task is smaller takes its place (the smallest such count), because the run then ends with
    the coefficient chains of the last wave and not with the tail, which the idle CTAs of a partial last wave serve."""
    if deterministic:
        return 16
    m_blocks = (num_tokens + 63) // 64
    max_splits = min(64, max(16, num_sms // m_blocks))
    longest = (64 + max_splits - 1) // max_splits
    splits = (64 + longest - 1) // longest
    grid = mhc_infer_num_ctas(num_tokens, num_sms, splits)
    best = ((m_blocks * splits + grid - 1) // grid) * longest
    if m_blocks * 64 <= 12 * grid:
        for candidate in range(10, splits):
            waves = (m_blocks * candidate + grid - 1) // grid
            blocks = (64 + candidate - 1) // candidate
            if waves <= 2 and waves * blocks < best:
                splits, best = candidate, waves * blocks
    return splits


def mhc_infer_num_ctas(num_tokens: int, num_sms: int, num_splits: int) -> int:
    """CTAs to launch: every projection task and every mix warp in one round, never more than the SMs; the kernel
    strides tasks, tokens and m-block ownership by the grid, so any grid >= m_blocks is correct and this one is fast."""
    m_blocks = (num_tokens + 63) // 64
    return min(num_sms, max(m_blocks * num_splits, (num_tokens + 3) // 4))


def mhc_infer_scratch_floats(num_tokens: int, num_splits: int) -> int:
    """64 tokens x 24 mapping partials + 64 sum-of-squares words + 64 spare per task (the deepgemm_mega_mhc layout)."""
    return ((num_tokens + 63) // 64) * num_splits * (1536 + 128)


def mhc_infer_switch_points(num_sms: int, max_tokens: int = 1 << 20) -> tuple[int, ...]:
    """Every T at which num_splits changes (the correctness tests cover both sides of each): the rule changes only at
    m-block boundaries, so the scan steps by 64, and it stops once the tokens exceed two tail rounds of the SMs and the
    count has returned to the deepgemm count (never below 16 from there on)."""
    points, previous, t = [], mhc_infer_num_splits(1, num_sms), 65
    while t <= max_tokens and (previous != 16 or t <= 12 * num_sms):
        current = mhc_infer_num_splits(t, num_sms)
        if current != previous:
            points.append(t)
            previous = current
        t += 64
    return tuple(points)


def mhc_infer_tail_program(
    num_tokens: int, num_sms: int, deterministic: bool = False
) -> str:
    """The program of the launch: ``"tailpipe_wsmem"`` (``mhc_infer_step``'s ``tail_pipe`` + ``weight_smem`` build: the
    tailpipe tail reading the RMSNorm weight staged once per CTA in shared memory) when :func:`mhc_infer_num_splits`
    yields at most WEIGHT_SMEM_MAX_SPLITS splits and the tasks run in at least WEIGHT_SMEM_MIN_WAVES waves of the grid
    (``ceil(m_blocks * splits / grid)``: 16 splits over five or more waves, T >= 3329 on 208 SMs, where the tail is the
    teardown bound and the staging pays off), ``"tailpipe"`` (the ``tail_pipe`` build: the tail's round-1 loads overlap
    the fold) at the other counts up to TAIL_PIPE_MAX_SPLITS (22: the counts 22, 16 and the two-wave counts below them),
    else ``"bf16"`` (32 and 64 splits: the fold over many split partials is the token's critical path and the early
    loads would compete with it)."""
    splits = mhc_infer_num_splits(num_tokens, num_sms, deterministic)
    if splits <= WEIGHT_SMEM_MAX_SPLITS:
        m_blocks = (num_tokens + 63) // 64
        grid = mhc_infer_num_ctas(num_tokens, num_sms, splits)
        if (m_blocks * splits + grid - 1) // grid >= WEIGHT_SMEM_MIN_WAVES:
            return "tailpipe_wsmem"
    return "tailpipe" if splits <= TAIL_PIPE_MAX_SPLITS else "bf16"


def mhc_infer_variant_num_splits(
    num_tokens: int, num_sms: int, deterministic: bool = False
) -> int:
    """Splits of the quantized-output variants (``sf_layout`` ``"fp8_group128"`` / ``"nvfp4"``): the production rule's
    count (:func:`mhc_infer_num_splits`), except VARIANT_WSMEM_REGIME_SPLITS = 8 where that rule launches the production
    program as ``"tailpipe_wsmem"`` (:func:`mhc_infer_tail_program`) and its tasks run in at least
    VARIANT_PLAIN_MIN_WAVES = 7 waves of the grid (``ceil(m_blocks * splits / grid)``: T >= 3521 on 148 SMs, 3649 on
    152, 4993 on 208). The variants are the plain program: over that many rounds the two-warp fold of every token
    costs the helper warps their concurrent tail (what the weight_smem program avoids by keeping the one-warp fold),
    and 8 splits halve the fold records per token and the tasks per round. Measured on three boards (16 -> 8 splits):
    8 wins at 7 waves (T = 4096 on 148 and 152 SMs, 5-6 %: nvfp4 99.857 -> 94.241 us and 92.407 -> 86.344 us,
    fp8_group128 98.935 -> 92.792 us), at 10 waves (T = 8192 on 208 SMs, 1-1.5 %: fp8_group128 125.032 -> 123.119 us,
    nvfp4 115.792 -> 114.640 us) and at 14 waves (T = 8192 on 148 and 152 SMs, 4-5 %: nvfp4 195.091 -> 186.994 us
    and 183.975 -> 174.863 us, fp8_group128 194.872 -> 186.456 us); 16 wins at 5 waves (T = 4096 on 208 SMs:
    fp8_group128 62.168 -> 63.384 us, nvfp4 57.175 -> 57.912 us) and the two counts are mixed at 4 waves (T = 2048 on
    152 SMs, below the regime), so a T between the tailpipe_wsmem switch and the 7-wave switch keeps the production
    count, and below that regime (the tailpipe counts) fewer splits are neutral or slower. The outputs of the two
    counts differ in their last bits (a split's record is the Kahan sum of its own K blocks' chunk partials, so the
    fold's operands regroup with the count): the count is part of a launch's identity. In deterministic mode the count
    stays the production rule's 16 unconditionally, because the wave count depends on the SM count and deterministic
    outputs must stay host-independent: deterministic quantized launches at T >= the 7-wave switch keep 16 splits and
    the plain program's multi-round cost."""
    splits = mhc_infer_num_splits(num_tokens, num_sms, deterministic)
    if deterministic:
        return splits
    if mhc_infer_tail_program(num_tokens, num_sms, deterministic) == "tailpipe_wsmem":
        m_blocks = (num_tokens + 63) // 64
        grid = mhc_infer_num_ctas(num_tokens, num_sms, splits)
        if (m_blocks * splits + grid - 1) // grid >= VARIANT_PLAIN_MIN_WAVES:
            return VARIANT_WSMEM_REGIME_SPLITS
    return splits


def mhc_infer_variant_program(
    num_tokens: int,
    num_sms: int,
    sf_layout: str,
    deterministic: bool = False,
    tail_variants: bool = False,
) -> str:
    """The program of a quantized-output variant's launch (``sf_layout`` ``"fp8_group128"`` / ``"nvfp4"``) at its split
    count (:func:`mhc_infer_variant_num_splits`). ``"bf16"`` (the plain variant program) unless the arch ships the
    variants' tail forms: ``tail_variants`` is the caller's statement of the programs it has (the sm_107a
    ``mhc_infer_step`` entries with ``sf_layout`` and ``tail_pipe`` [+ ``weight_smem``]), never inferred from the SM
    count. Then the production program's form (:func:`mhc_infer_tail_program`: ``"tailpipe"`` / ``"tailpipe_wsmem"``,
    ``"bf16"`` at 32 and 64 splits: T < 385 on 208 SMs), except where the plain program stays: where the production
    program is ``"tailpipe"`` and all tasks fit one wave of the grid, ``"fp8_group128"`` at fewer than 16 splits
    (T = 833..1280 on 208 SMs) and ``"nvfp4"`` at 16 splits or more (T = 385..832 on 208 SMs), and both layouts in
    deterministic mode below the tailpipe_wsmem regime; in those ranges the tail form measured slower than the plain
    program at some counts. A tail form reorders loads, not arithmetic: at the same split count it writes bitwise the
    plain variant's outputs, so the choice never changes a launch's outputs."""
    if not tail_variants:
        return "bf16"
    program = mhc_infer_tail_program(num_tokens, num_sms, deterministic)
    if program != "tailpipe":
        return program
    if deterministic:
        return "bf16"
    splits = mhc_infer_num_splits(num_tokens, num_sms, deterministic)
    m_blocks = (num_tokens + 63) // 64
    grid = mhc_infer_num_ctas(num_tokens, num_sms, splits)
    if m_blocks * splits <= grid and (
        splits < 16 if sf_layout == "fp8_group128" else splits >= 16
    ):
        return "bf16"
    return program


def split_fn(fn, out=None):
    """The projection operand of the BF16-output programs: ``fn[24, 4H]`` FP32 as
    three BF16 pieces ``fnp[3, 32, 4H]`` (rows 24..31 zero) with ``b1 = BF16(fn)``,
    ``b2 = BF16(fn - b1)`` and ``b3 = BF16(fn - b1 - b2)``, each residual exact in
    FP32, so ``b1 + b2 + b3`` carries ``fn`` to within 2^-27 relative. Torch
    operations on the current stream only (CUDA Graph capturable); ``out`` is
    the caller's ``[3, 32, 4H]`` BF16 buffer to overwrite, else a new one."""
    import torch

    rows, width = fn.shape
    if out is None:
        out = torch.empty(
            (FNP_PIECES, FNP_ROWS, width), dtype=torch.bfloat16, device=fn.device
        )
    b1 = fn.to(torch.bfloat16)
    r1 = fn - b1.float()
    b2 = r1.to(torch.bfloat16)
    b3 = (r1 - b2.float()).to(torch.bfloat16)
    out[0, :rows].copy_(b1)
    out[1, :rows].copy_(b2)
    out[2, :rows].copy_(b3)
    out[:, rows:].zero_()
    return out


@functools.cache
def device_facts(device_index):
    """(architecture, SM count) of ``cuda:<device_index>``, queried once per process."""
    import torch

    from ...utils import get_compute_capability, get_device_sm_count

    device = torch.device("cuda", device_index)
    capability = get_compute_capability(device)
    arch = CAPABILITIES.get(capability)
    if arch is None or arch not in shipped_arches():
        raise RuntimeError(
            f"Mega mHC has no exported programs for compute capability {capability}; "
            f"supported: {shipped_arches()}"
        )
    return arch, get_device_sm_count(device)


def record_arches(record):
    """The architectures a program ships for: its own ``arches`` leaf, else ARCHES."""
    return list(record.get("arches", ARCHES))


def shipped_arches():
    """Every architecture some exported program ships for: the records' own
    ``arches`` leaves, else ARCHES."""
    return sorted(
        {arch for record in PROGRAMS.values() for arch in record_arches(record)}
    )


def record_arg_plan(record):
    """The (kind, name) pairs of a program's ``run`` arguments: its own ``arg_plan``, else ARG_PLAN."""
    return [tuple(pair) for pair in record.get("arg_plan", ARG_PLAN)]


def record_compile_flags(record):
    """The flags of a program's compile line after the architecture flags: its own
    ``compile_flags`` (else COMPILE_FLAGS), then, for a record carrying the source
    build's ``compile_options``, ``--fmad=false`` followed by those options (the
    later flag wins, as the producer compiles)."""
    flags = list(record.get("compile_flags", COMPILE_FLAGS))
    options = record.get("compile_options")
    if options is not None:
        flags += ["--fmad=false", *options]
    return flags


def route_output(sf_layout):
    """The output a route serves for ``sf_layout``: the FP8 layouts share the FP8
    programs (``"fp8"``); ``"bf16"`` and the quantized layouts are their own."""
    return "fp8" if sf_layout in FP8_LAYOUTS else sf_layout


def supported(arch, hidden, sf_layout):
    """Whether an exported program serves ``hidden`` with the output layout
    ``sf_layout`` on ``arch``, without loading anything."""
    output = route_output(sf_layout)
    return any(
        route["hidden"] == hidden
        and route.get("sf_layout", "fp8") == output
        and arch in record_arches(PROGRAMS[route["program"]])
        for route in ROUTES
    )


def route_regime(route):
    """The regime a route serves: the name ``mhc_infer_tail_program`` gives the split
    regime of a layout with several programs, ``None`` for a route serving every
    count of its key."""
    return route.get("regime")


def _support_matrix():
    rows = sorted(
        {
            (
                route["hidden"],
                route.get("sf_layout", "fp8"),
                tuple(record_arches(PROGRAMS[route["program"]])),
                route_regime(route) or "",
                route["program"] if route_regime(route) else "",
            )
            for route in ROUTES
        }
    )
    return "; ".join(
        f"hidden={hidden} output={output} on {list(arches)}"
        + (f" ({regime}: {program})" if regime else "")
        for hidden, output, arches, regime, program in rows
    )


def select_program(
    hidden, splits, shifted, shared_sf, sf_layout, arch, regime=None, fp8_sf_ue8m0=None
):
    """(source name, compile-line defines) serving the schedule key on ``arch``, or raise.
    ``regime`` is the host rule's name for the split regime of the launch
    (``mhc_infer_tail_program`` over the token count and the SM count); a route
    keyed on a regime serves that regime alone, a route without one every count of
    its key, so a layout with several programs needs the regime to select.
    ``fp8_sf_ue8m0`` is the FP8 group-128 scale mode of the launch: a route keyed on
    it (the UE8M0-only programs on sm_107a) serves that mode alone, a route without
    the key both modes and a caller that passes no mode."""
    output = route_output(sf_layout)
    for route in ROUTES:
        if (
            route["hidden"] == hidden
            and route["num_splits"] in (splits, "runtime")
            and route_regime(route) in (None, regime)
            and route.get("fp8_sf_ue8m0") in (None, fp8_sf_ue8m0)
            and route["shifted"] == shifted
            and route["shared_sf"] == shared_sf
            and route.get("sf_layout", "fp8") == output
            and arch in record_arches(PROGRAMS[route["program"]])
        ):
            return route["program"], dict(route["defines"])
    raise NotImplementedError(
        f"No exported mHC program for hidden={hidden}, num_splits={splits}, "
        f"regime={regime!r}, shifted={shifted}, shared_sf={shared_sf}, "
        f"sf_layout={sf_layout!r} on {arch}; exported: {_support_matrix()}"
    )


@functools.cache
def load_program(name, arch, num_sms, defines):
    """Build ``name`` for ``arch`` with the route's defines, the program's kernel symbol
    (``CAKE_PROGRAM_KERNEL``, read by the shared binding and by the kernel sources shared
    between a 27-split and a 40-split program) and, for a record compiled per SM count,
    ``NUM_SMS`` on the compile line."""
    from flashinfer.jit import env
    from flashinfer.jit.core import (
        gen_jit_spec,
        sm100a_nvcc_flags,
        sm103a_nvcc_flags,
        sm107a_nvcc_flags,
    )

    record = PROGRAMS[name]
    arch_flags = {
        "sm_100a": sm100a_nvcc_flags,
        "sm_103a": sm103a_nvcc_flags,
        "sm_107a": sm107a_nvcc_flags,
    }[arch]
    values = tuple(defines)
    if record.get("compile_time_sm_count", True):
        values = (*values, ("NUM_SMS", num_sms))
    spec = gen_jit_spec(
        name="_".join(
            [name, arch, *(f"{key.lower()}{value}" for key, value in values)]
        ),
        sources=[
            env.FLASHINFER_CSRC_DIR / path.removeprefix("csrc/")
            for path in record["sources"]
        ],
        extra_cuda_cflags=[
            *arch_flags,
            *record_compile_flags(record),
            *(f"-D{key}={value}" for key, value in values),
            f"-DCAKE_PROGRAM_KERNEL={PROGRAMS[name]['kernel_symbol']}",
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            env.FLASHINFER_CSRC_DIR / GENERATED_ROOT.removeprefix("csrc/"),
            env.FLASHINFER_CSRC_DIR,
            env.FLASHINFER_INCLUDE_DIR,
        ],
        use_fast_math=False,
    )
    return spec.build_and_load(), str(spec.get_library_path())


def _new_outputs(
    x, residual, post_mix, comb_res_mix, shifted_prev_mix, sf_layout, block_m
):
    import torch

    tokens, hidden = x.shape
    outputs = dict(
        new_residual=torch.empty_like(residual),
        new_post_mix=torch.empty_like(post_mix),
        new_comb_res_mix=torch.empty_like(comb_res_mix),
        y_bf16=torch.empty_like(x),
    )
    if shifted_prev_mix is not None:
        outputs["new_prev_mix"] = torch.empty_like(shifted_prev_mix)
    if sf_layout == "fp8_group128":
        outputs["y_fp8"] = torch.empty(
            (tokens, hidden), dtype=torch.float8_e4m3fn, device=x.device
        )
        outputs["y_fp8_sf"] = torch.empty(
            (tokens, hidden // FP8_GROUP), dtype=torch.float32, device=x.device
        )
    elif sf_layout == "nvfp4":
        outputs["y_fp4"] = torch.empty(
            (tokens, hidden // 2), dtype=torch.uint8, device=x.device
        )
        outputs["y_fp4_sf"] = torch.empty(
            (tokens, hidden // FP4_GROUP), dtype=torch.float8_e4m3fn, device=x.device
        )
    elif sf_layout in FP8_LAYOUTS:
        outputs["y_fp8"] = torch.empty(
            (tokens, hidden), dtype=torch.float8_e4m3fn, device=x.device
        )
        if sf_layout == "col":
            outputs["y_gemm_sf"] = torch.empty_strided(
                (tokens, hidden // 128),
                (1, (tokens + 3) // 4 * 4),
                dtype=torch.int32,
                device=x.device,
            )
        else:
            outputs["y_routed_sf"] = torch.empty(
                (tokens, hidden // 128), dtype=torch.int32, device=x.device
            )
            rows = (tokens + block_m - 1) // block_m * ((block_m + 127) // 128 * 128)
            storage = torch.empty_strided(
                (rows, hidden // 128), (1, rows), dtype=torch.int32, device=x.device
            )
            outputs.update(
                y_shared_sf_storage=storage,
                y_shared_sf=storage[:tokens],
                shared_sf_block_m=block_m,
            )
    return outputs


#: The launch grid of every live plan's ``launch_epochs`` (caller-owned or the
#: plan's own), by the storage's (data pointer, bytes, device): the kernel
#: advances one epoch slot per active CTA, so plans sharing the storage must
#: launch the same grid. Read and updated under ``_EPOCH_GRIDS_LOCK``.
_EPOCH_GRIDS: dict = {}
_EPOCH_GRIDS_LOCK = threading.Lock()


def _claim_launch_epochs(plan, launch_epochs, grid):
    """Record ``plan`` as a user of ``launch_epochs`` launching ``grid`` CTAs;
    refuse the storage while another live plan launches a different grid. The
    tensor must own its whole storage (offset 0, spanning the storage), so two
    views of overlapping memory cannot carry different keys."""
    storage = launch_epochs.untyped_storage()
    nbytes = launch_epochs.numel() * launch_epochs.element_size()
    if launch_epochs.storage_offset() != 0 or storage.nbytes() != nbytes:
        raise ValueError(
            "launch_epochs must own its whole storage (storage offset 0 and "
            f"{launch_epochs.numel()} elements spanning the storage); a sliced or "
            f"offset view (offset {launch_epochs.storage_offset()}, storage of "
            f"{storage.nbytes()} bytes) is refused"
        )
    key = (storage.data_ptr(), nbytes, launch_epochs.device.index)
    with _EPOCH_GRIDS_LOCK:
        for stale in [
            k for k, users in _EPOCH_GRIDS.items() if not any(ref() for ref, _ in users)
        ]:
            del _EPOCH_GRIDS[stale]
        users = [
            (ref, other)
            for ref, other in _EPOCH_GRIDS.get(key, ())
            if ref() is not None
        ]
        for _ref, other in users:
            if other != grid:
                raise ValueError(
                    f"launch_epochs is in use by a plan launching {other} CTAs; this "
                    f"plan launches {grid}: plans sharing launch_epochs must launch "
                    "the same grid (one epoch slot per active CTA)"
                )
        users.append((weakref.ref(plan), grid))
        _EPOCH_GRIDS[key] = users


def _raw_storage(tensor):
    import torch

    words = (
        tensor.untyped_storage().nbytes() // tensor.element_size()
        - tensor.storage_offset()
    )
    return tensor.as_strided((words,), (1,)).view(torch.uint32)


class MegaMHCPlan:
    """Bind a fused mHC operation and reuse its private invocation state.

    x/residual/rmsnorm_weight are BF16; projection and mixing coefficients are
    FP32. The four residual routes produce BF16 normalized output and, with
    ``sf_layout='col'`` or ``'extra'`` (hidden 5120), FP8 E4M3 output with packed
    per-32 UE8M0 scales: ``'col'`` stores column-major scale words, ``'extra'``
    also stores the shared-expert padded row permutation. ``sf_layout='bf16'``
    (hidden 4096, normal mode) stores the BF16 output alone: no FP8 values, no
    scale words. The quantized layouts of hidden 4096 store the BF16 output and
    its quantization beside it, bitwise what vLLM computes from that BF16
    output: ``'fp8_group128'`` writes ``y_fp8`` (float8_e4m3fn [T, H]) and
    ``y_fp8_sf`` (float32 [T, H // 128], one scale per 128-element group of a
    token; ``fp8_sf_ue8m0`` keeps each scale a power of two, as
    ``per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=True)`` does, ``False``
    the raw ``amax / 448`` scale); ``'nvfp4'`` writes ``y_fp4`` (uint8 [T, H // 2],
    element 2i in the low nibble of byte i) and ``y_fp4_sf`` (float8_e4m3fn
    [T, H // 16] in the linear layout) under the caller's ``fp4_global_scale``
    (float32 [1] on the device), as ``scaled_fp4_quant(y_bf16, fp4_global_scale,
    is_sf_swizzled_layout=False)`` does, a zero scale guarded as there. Any token
    count in [1, 2**20] is accepted; the split count follows the token count and
    the SM count (or ``deterministic``): ``mhc_infer_num_splits`` for the BF16
    output, ``mhc_infer_variant_num_splits`` for the quantized layouts (8 splits
    from seven task waves of the production count on), and the program for that
    split count must be exported for ``hidden`` with that layout on the device's
    architecture. The BF16-output
    route runs one program per split regime, named by ``mhc_infer_tail_program``
    from the token count and the SM count (``"tailpipe_wsmem"`` up to
    ``WEIGHT_SMEM_MAX_SPLITS`` (16) splits over ``WEIGHT_SMEM_MIN_WAVES`` (5) task
    waves of the grid or more, ``"tailpipe"`` at the other counts up to
    ``TAIL_PIPE_MAX_SPLITS`` (22), ``"bf16"`` at 32 and 64); the programs write
    bitwise the same output, and ``plan.program`` / ``plan.config["regime"]``
    name the one bound.

    The BF16-output programs read the projection ``fn`` through its three BF16
    pieces (``split_fn``), built once here from ``fn`` as given to the plan, or
    taken from the caller's ``fnp`` buffer. ``fn`` is captured at plan time: a
    later change of ``fn`` reaches the kernel through ``update_fn()``, which
    re-splits into the retained buffer on the current stream.

    Scratch, split_barriers, launch_epochs and fnp may be caller-owned. A
    caller-owned ``scratch`` holds ``mhc_infer_scratch_floats(tokens, splits)``
    floats of the plan's own split count (``plan.config["num_splits"]``): for a
    quantized layout at the token counts where that count is 8, a buffer sized
    from the production count is refused. Fresh
    barriers/epochs must be zeroed once before first use and retained across
    ordinary calls and CUDA Graph replay. Epoch storage always has num_sms
    entries, including CTAs omitted by a selected launch grid. One plan must
    not execute concurrently on multiple streams. run() uses the current
    PyTorch stream and performs no device allocation or workspace reset. Plans
    sharing a ``launch_epochs`` -- caller-owned or another plan's own -- must
    launch the same grid: the kernel advances one epoch slot per active CTA, so
    a plan refuses a ``launch_epochs`` another live plan uses with a different
    grid. A caller-owned ``launch_epochs`` must own its whole storage (offset 0,
    spanning the storage); a sliced or offset view is refused. A caller-owned
    ``launch_epochs`` must not be reused across plans of different grids: use a
    fresh zeroed tensor per grid, since the slots an earlier smaller grid
    advanced leave the storage non-uniform and the next full-grid launch waits
    forever.
    """

    def __init__(
        self,
        *,
        x,
        residual,
        post_mix,
        comb_res_mix,
        shifted_prev_mix,
        fn,
        mix_scales,
        mix_bases,
        rmsnorm_weight,
        hc_mult=4,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        num_sinkhorn_iters=20,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=1.25,
        sf_layout="col",
        shared_sf_block_m=224,
        out=None,
        deterministic=False,
        scratch=None,
        split_barriers=None,
        launch_epochs=None,
        fnp=None,
        fp8_sf_ue8m0=True,
        fp4_global_scale=None,
    ):
        import torch

        if x.device.type != "cuda":
            raise RuntimeError("Mega mHC requires a CUDA device")
        layouts = ("bf16", *FP8_LAYOUTS, *QUANT_LAYOUTS)
        if x.ndim != 2 or hc_mult != 4 or sf_layout not in layouts:
            raise ValueError(
                f"Expected x[T,H], four residual routes and a scale layout of {layouts}"
            )
        if sf_layout != "fp8_group128" and fp8_sf_ue8m0 is not True:
            raise ValueError(
                "fp8_sf_ue8m0 selects the scale format of sf_layout='fp8_group128'"
            )
        if sf_layout != "nvfp4" and fp4_global_scale is not None:
            raise ValueError("fp4_global_scale is the operand of sf_layout='nvfp4'")
        tokens, hidden = x.shape
        if hidden % 1024 or not 1 <= tokens <= MAX_TOKENS or num_sinkhorn_iters < 1:
            raise ValueError(
                "Expected H divisible by 1024, 1 <= T <= 2**20 and positive Sinkhorn iterations"
            )
        if sf_layout == "extra" and shared_sf_block_m <= 0:
            raise ValueError("shared_sf_block_m must be positive")
        arch, sms = device_facts(x.device.index)
        shifted = shifted_prev_mix is not None
        store_fp8 = sf_layout in FP8_LAYOUTS
        if not store_fp8 and (hidden != BF16_HIDDEN or shifted):
            raise NotImplementedError(
                f"The hidden-{BF16_HIDDEN} mHC programs (sf_layout {sf_layout!r}) serve "
                f"normal mode (shifted_prev_mix=None); got hidden={hidden}, "
                f"shifted={shifted}. Exported: {_support_matrix()}"
            )
        shared_sf = sf_layout == "extra"
        if store_fp8:
            splits = num_splits(tokens, hidden, sms, deterministic)
            regime = None
        else:
            # The quantized variants take their own count: 8 from seven task waves of
            # the production count on (inside the weight-staging regime), the
            # production count elsewhere; the regime names the BF16-output program.
            if sf_layout in QUANT_LAYOUTS:
                splits = mhc_infer_variant_num_splits(tokens, sms, deterministic)
                regime = mhc_infer_variant_program(
                    tokens, sms, sf_layout, deterministic, arch in VARIANT_TAIL_ARCHES
                )
            else:
                splits = mhc_infer_num_splits(tokens, sms, deterministic)
                regime = mhc_infer_tail_program(tokens, sms, deterministic)
        program, defines = select_program(
            hidden,
            splits,
            shifted,
            shared_sf,
            sf_layout,
            arch,
            regime=regime,
            fp8_sf_ue8m0=(bool(fp8_sf_ue8m0) if sf_layout == "fp8_group128" else None),
        )
        record = PROGRAMS[program]
        block_m = shared_sf_block_m if shared_sf else 0
        if shared_sf and block_m != record["shared_sf_block_m"]:
            raise NotImplementedError(
                f"Exported mHC programs use shared_sf_block_m={record['shared_sf_block_m']}"
            )
        if store_fp8:
            launch_sms = num_launch_sms(
                tokens,
                hidden,
                sms,
                splits,
                shifted,
                True,
                sf_layout == "col",
                deterministic,
            )
            scratch_shape = ((tokens + 63) // 64 * splits * (1536 + 128),)
        else:
            launch_sms = mhc_infer_num_ctas(tokens, sms, splits)
            scratch_shape = (mhc_infer_scratch_floats(tokens, splits),)
        self.config = dict(
            num_splits=splits,
            num_sms=sms,
            num_launch_sms=launch_sms,
            program=program,
            regime=regime,
        )
        self.program = program
        self.defines = defines
        self.sf_layout = sf_layout

        def require(name, tensor, dtype, shape, *, contiguous=True):
            if (
                tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or tensor.device != x.device
            ):
                raise ValueError(
                    f"{name} must be {dtype}{shape} on the input CUDA device"
                )
            if contiguous and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        for name, tensor, dtype, shape in (
            ("x", x, torch.bfloat16, (tokens, hidden)),
            ("residual", residual, torch.bfloat16, (tokens, 4, hidden)),
            ("post_mix", post_mix, torch.float32, (tokens, 4, 1)),
            ("comb_res_mix", comb_res_mix, torch.float32, (tokens, 4, 4)),
            ("fn", fn, torch.float32, (FN_ROWS, 4 * hidden)),
            ("mix_scales", mix_scales, torch.float32, (3,)),
            ("mix_bases", mix_bases, torch.float32, (FN_ROWS,)),
            ("rmsnorm_weight", rmsnorm_weight, torch.bfloat16, (hidden,)),
        ):
            require(name, tensor, dtype, shape)
        if shifted:
            require("shifted_prev_mix", shifted_prev_mix, torch.float32, (tokens, 4, 1))
        if out is None:
            out = _new_outputs(
                x,
                residual,
                post_mix,
                comb_res_mix,
                shifted_prev_mix,
                sf_layout,
                block_m,
            )
        for name, dtype, shape in (
            ("new_residual", torch.bfloat16, (tokens, 4, hidden)),
            ("new_post_mix", torch.float32, (tokens, 4, 1)),
            ("new_comb_res_mix", torch.float32, (tokens, 4, 4)),
            ("y_bf16", torch.bfloat16, (tokens, hidden)),
        ):
            require(name, out[name], dtype, shape)
        if shifted:
            require("new_prev_mix", out["new_prev_mix"], torch.float32, (tokens, 4, 1))
        if sf_layout == "fp8_group128":
            require("y_fp8", out["y_fp8"], torch.float8_e4m3fn, (tokens, hidden))
            require(
                "y_fp8_sf",
                out["y_fp8_sf"],
                torch.float32,
                (tokens, hidden // FP8_GROUP),
            )
        if sf_layout == "nvfp4":
            if fp4_global_scale is None:
                raise ValueError("sf_layout='nvfp4' requires fp4_global_scale")
            require("fp4_global_scale", fp4_global_scale, torch.float32, (1,))
            require("y_fp4", out["y_fp4"], torch.uint8, (tokens, hidden // 2))
            require(
                "y_fp4_sf",
                out["y_fp4_sf"],
                torch.float8_e4m3fn,
                (tokens, hidden // FP4_GROUP),
            )
        if store_fp8:
            require("y_fp8", out["y_fp8"], torch.float8_e4m3fn, (tokens, hidden))
            primary = out["y_gemm_sf" if sf_layout == "col" else "y_routed_sf"]
            require(
                "primary scale words",
                primary,
                torch.int32,
                (tokens, hidden // 128),
                contiguous=False,
            )
            expected_stride = (
                (1, (tokens + 3) // 4 * 4) if sf_layout == "col" else (hidden // 128, 1)
            )
            if primary.stride() != expected_stride:
                raise ValueError(f"Primary scale strides must be {expected_stride}")
        if shared_sf:
            rows = (tokens + block_m - 1) // block_m * ((block_m + 127) // 128 * 128)
            storage, shared = out["y_shared_sf_storage"], out["y_shared_sf"]
            require(
                "shared scale storage",
                storage,
                torch.int32,
                (rows, hidden // 128),
                contiguous=False,
            )
            require(
                "shared scale view",
                shared,
                torch.int32,
                (tokens, hidden // 128),
                contiguous=False,
            )
            if (
                storage.stride() != (1, rows)
                or shared.stride() != (1, rows)
                or storage.data_ptr() != shared.data_ptr()
                or out["shared_sf_block_m"] != block_m
            ):
                raise ValueError(
                    "Shared scales require the padded column-major storage and its first-T-row view"
                )
        if scratch is None:
            scratch = torch.empty(scratch_shape, dtype=torch.float32, device=x.device)
        if split_barriers is None:
            split_barriers = torch.zeros(
                (SPLIT_BARRIER_WORDS,), dtype=torch.uint64, device=x.device
            )
        if launch_epochs is None:
            launch_epochs = torch.zeros((sms,), dtype=torch.uint64, device=x.device)
        require("scratch", scratch, torch.float32, scratch_shape)
        require("split_barriers", split_barriers, torch.uint64, (SPLIT_BARRIER_WORDS,))
        require("launch_epochs", launch_epochs, torch.uint64, (sms,))
        # every plan registers its epoch storage, caller-owned or its own, so a
        # plan's own tensor handed to another plan is checked the same way
        _claim_launch_epochs(self, launch_epochs, launch_sms)
        if store_fp8:
            if fnp is not None:
                raise ValueError(
                    "fnp is the BF16-output program's operand; the FP8 programs read fn"
                )
            dummy_u32 = torch.empty(1, dtype=torch.uint32, device=x.device)
            dummy_float = torch.empty(1, dtype=torch.float32, device=x.device)
            shared = out["y_shared_sf"] if shared_sf else dummy_u32
            self.bindings = dict(
                residual_map=residual,
                x_map=x,
                fn_map=fn,
                post_map=post_mix.view(tokens, 4),
                comb_map=comb_res_mix.view(tokens, 16),
                prev_map=(shifted_prev_mix if shifted else post_mix).view(tokens, 4),
                new_residual_map=out["new_residual"],
                y_map=out["y_bf16"] if shifted else x,
                mix_scales=mix_scales,
                mix_bases=mix_bases,
                new_prev_mix=out["new_prev_mix"] if shifted else dummy_float,
                new_post_mix=out["new_post_mix"],
                new_comb_res_mix=out["new_comb_res_mix"],
                rmsnorm_weight=rmsnorm_weight,
                new_residual=out["new_residual"],
                y_bf16=out["y_bf16"],
                y_fp8=out["y_fp8"].view(torch.uint8),
                y_primary_sf=_raw_storage(primary),
                y_shared_sf=_raw_storage(shared),
                scratch=scratch,
                split_barriers=split_barriers,
                launch_epochs=launch_epochs,
                num_tokens=tokens,
                hc_norm_eps=hc_norm_eps,
                hc_pre_eps=hc_pre_eps,
                hc_post_scale=hc_post_scale,
                sinkhorn_eps=sinkhorn_eps,
                num_sinkhorn_iters=num_sinkhorn_iters,
                rmsnorm_eps=rmsnorm_eps,
                rmsnorm_scale=rmsnorm_scale,
                primary_sf_stride_token=primary.stride(0),
                primary_sf_stride_word=primary.stride(1),
                shared_sf_stride_word=shared.stride(1) if shared_sf else 0,
                grid_x=launch_sms,
                grid_y=1,
                grid_z=1,
            )
            retained = (dummy_u32, dummy_float)
        else:
            fnp_shape = (FNP_PIECES, FNP_ROWS, 4 * hidden)
            if fnp is None:
                fnp = split_fn(fn)
            require("fnp", fnp, torch.bfloat16, fnp_shape)
            self.bindings = dict(
                residual_map=residual,
                x_map=x,
                fnp_map=fnp,
                post_map=post_mix.view(tokens, 4),
                comb_map=comb_res_mix.view(tokens, 16),
                new_residual_map=out["new_residual"],
                mix_scales=mix_scales,
                mix_bases=mix_bases,
                new_post_mix=out["new_post_mix"],
                new_comb_res_mix=out["new_comb_res_mix"],
                rmsnorm_weight=rmsnorm_weight,
                new_residual=out["new_residual"],
                y_bf16=out["y_bf16"],
                scratch=scratch,
                split_barriers=split_barriers,
                launch_epochs=launch_epochs,
                num_tokens=tokens,
                num_splits=splits,
                split_div=splits,
                num_ctas=launch_sms,
                hc_norm_eps=hc_norm_eps,
                hc_pre_eps=hc_pre_eps,
                hc_post_scale=hc_post_scale,
                sinkhorn_eps=sinkhorn_eps,
                num_sinkhorn_iters=num_sinkhorn_iters,
                rmsnorm_eps=rmsnorm_eps,
                rmsnorm_scale=rmsnorm_scale,
                grid_x=launch_sms,
                grid_y=1,
                grid_z=1,
            )
            if sf_layout == "fp8_group128":
                self.bindings.update(
                    y_fp8=out["y_fp8"],
                    y_fp8_sf=out["y_fp8_sf"],
                    fp8_sf_ue8m0=int(bool(fp8_sf_ue8m0)),
                )
            elif sf_layout == "nvfp4":
                self.bindings.update(
                    y_fp4=out["y_fp4"],
                    y_fp4_sf=out["y_fp4_sf"],
                    fp4_global_scale=fp4_global_scale,
                )
            retained = (fnp, fp4_global_scale)
        module, self.library_path = load_program(
            program, arch, sms, tuple(sorted(defines.items()))
        )
        unbound = [
            name for _kind, name in record_arg_plan(record) if name not in self.bindings
        ]
        if unbound:
            raise RuntimeError(
                f"The argument plan of {program} names {unbound}, which this plan does not bind"
            )
        args = tuple(self.bindings[name] for _kind, name in record_arg_plan(record))
        self._submission = (module["run"], args)
        self._retained = (
            module,
            x,
            residual,
            post_mix,
            comb_res_mix,
            shifted_prev_mix,
            fn,
            mix_scales,
            mix_bases,
            rmsnorm_weight,
            *retained,
        )
        self.fn, self.fnp = fn, fnp
        self.outputs, self.scratch, self.split_barriers, self.launch_epochs = (
            out,
            scratch,
            split_barriers,
            launch_epochs,
        )

    def update_fn(self, fn=None):
        """Re-split the projection into the plan's ``fnp`` buffer on the current
        stream: the plan's ``fn`` as it reads now, or ``fn`` (same shape and
        dtype), which the plan retains from then on. Torch operations only, so a
        CUDA Graph may capture it. The FP8 programs read the ``fn`` bound at
        plan time in place: there this call is a no-op, and a replacement
        tensor is refused rather than ignored."""
        if self.fnp is None:
            if fn is not None:
                raise ValueError(
                    "the FP8 programs read the fn bound at plan time; "
                    "prepare a new plan to replace it"
                )
            return
        if fn is not None:
            if (
                fn.shape != self.fn.shape
                or fn.dtype != self.fn.dtype
                or fn.device != self.fn.device
            ):
                raise ValueError("fn must keep the plan's shape, dtype and device")
            self.fn = fn
        split_fn(self.fn, out=self.fnp)

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.outputs
