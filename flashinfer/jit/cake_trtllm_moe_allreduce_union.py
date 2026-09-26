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

Verified source-only JIT loading for the Cake SM100 world-size-4 MoE all-reduce union.

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding that FlashInfer's JIT compiles
together.  ``MODULES`` and ``ROUTES`` are filled by the exporter from the
complete verified program bundle.  The loader checks every source file's
SHA-256 before its first build, so a modified or partially delivered source
tree fails closed instead of silently running a different kernel.
"""

from __future__ import annotations

import functools
import hashlib
import weakref
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags

# Filled mechanically from the complete verified program bundle.
MODULES: dict[str, dict[str, Any]] = {
    "cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe_binding.cu": "7128a348719ee43860a9bc7d1f252fa7cd9d637189a19285d211c71fc01c01ad",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe_kernel.cu": "6f6219bea3b13077eb8ff42fad65e2847490255155fef513c44482dd0975ddb7",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e_binding.cu": "55a021375c74ec78b2aeecf5241d7080492d359dd1f0c6f99ecf2b1fa11b153f",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e_kernel.cu": "699833c8ae4388472da0ba9efa8c87b6c8a74ad6ec44fa7ece203383cbc134ea",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9_binding.cu": "8db8e169a04333c6e70e93601677d24dc5341d8a5c5970e63f5b0163f9e2e266",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9_kernel.cu": "7d6a8e7bce0dfba3770a9919cea7584ad07e58c89a04f63924b88b1db4943f56",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_438d651723864ef58d3b": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_438d651723864ef58d3b_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_438d651723864ef58d3b",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_438d651723864ef58d3b_binding.cu": "95ca695ea002d80838dc0939c001b080b4782b7d62cce91192939a5619b2edc8",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_438d651723864ef58d3b_kernel.cu": "7fe5e9acbfd67d66303e047f72553ae0bbafb533a7a680ce03670362aa56df8e",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_438d651723864ef58d3b_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_438d651723864ef58d3b_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6_binding.cu": "85c9f5e7ea2ab164bd28a19a3bb37f1c3184940b9bbaac4a55798cefb92873ab",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6_kernel.cu": "b04159b2c0840aae7032966f059b56fa5c9cdb52195cea8c4345abcad868ba25",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8_binding.cu": "a18245ef00ae05ce7d6d9441ad70e85e508e66a6cbedf8f258c592879479928d",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8_kernel.cu": "f9c6f64198378e6976ce5884a663218614b5606c71afe983d230831fbd60766b",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98_binding.cu": "7dc42ea3f78cf471b361961e57a34974521c595e5552b013d3325941e417f5d2",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98_kernel.cu": "b73544aaa49918826a00ed13f74b5ce9de7d06cb670ab53d2861a17e5252b212",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_7a5993556607a1facdce": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_7a5993556607a1facdce_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_7a5993556607a1facdce",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7a5993556607a1facdce_binding.cu": "1ab8c45e79df12a402b67de4902e1e3f2d2fc102ea2069c8008b7d2bb1846414",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7a5993556607a1facdce_kernel.cu": "880431f080e541f7aea7b7def5e59186b8c72461dfcf1843f7432af48c904f46",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7a5993556607a1facdce_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_7a5993556607a1facdce_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7_binding.cu": "8b6d397d0012dd39cb8772fa830f08e5d033bdf356d298b55ca2cc79ef0153bd",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7_kernel.cu": "0c12e6fdbb2f446439b2ff43d7511133f5cf4269e0007bc9aba64eb066133910",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_86937d819491be970051": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_86937d819491be970051_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_86937d819491be970051",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_86937d819491be970051_binding.cu": "6ced398073211d7590b9c3aaab91ca59a71954625e092d2f81651530a8917448",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_86937d819491be970051_kernel.cu": "1ab913416d8108ff898f5276de4c917482623a32977439dcbf030a497e01e209",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_86937d819491be970051_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_86937d819491be970051_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_87de697e8471d30da720": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_87de697e8471d30da720_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_87de697e8471d30da720",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_87de697e8471d30da720_binding.cu": "448708034bec33ec14b2af8107d1e65c832c495f01630d3ba1fce8f1ec65764d",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_87de697e8471d30da720_kernel.cu": "cb49fe15ca42eb015a3033ec509b0dbef322b59211a4b321504018ec8d5feea1",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_87de697e8471d30da720_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_87de697e8471d30da720_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97_binding.cu": "03d0fc96372c6e26bfa9df6a00cddb49ae678d428221a15e5c2699eaad77eb01",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97_kernel.cu": "56bb27d3d2dec1203467c79df306fa3f9bb89ffdad2bc267be2ebc9aa239227e",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65_binding.cu": "0a1e15b7ab7f555c482d81be0d77312d14aa4951180ab2f6b2bf50f976bd6a52",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65_kernel.cu": "423ec225e753292c76497e88db9d0406b80cddb14a461745f32c4bd913aa36dc",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481_binding.cu": "fe678daf9a6a0b2e997d26fa5ccc3c4537272d0c37468a8c0230d20d41b0847b",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481_kernel.cu": "5b939ac22757227e938c5d1ddb6ed3c717b5bb1061a9997b708be689c289ddbd",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_9d436078fa133b24495b": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_9d436078fa133b24495b_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_9d436078fa133b24495b",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_9d436078fa133b24495b_binding.cu": "eca7de77f051575548cdab9238d4bb2b0935b05209e2641a7172bbbe3cc88485",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_9d436078fa133b24495b_kernel.cu": "9a8792fab723cc46ce2876b5f2cf89652c79584136ed0562d64ce5720d15d9f0",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_9d436078fa133b24495b_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_9d436078fa133b24495b_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de_binding.cu": "5bc8244526ea433471fe0ed84db98cf42a460a8b5b162cf89de1130ff174aa7f",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de_kernel.cu": "a2ed7aac9c66652310f168ce37a5c0ca9639bb28b16d198e4b0652633f458297",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a_binding.cu": "cf3a04101d68f17506bfcc4fa2396727380a65c6577aa721a9c7683e1bf2f442",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a_kernel.cu": "ea6ce8b0db8ecc0a96e360b57469e4e6a6c38b716ed63e7bb3a26e14157fc725",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551_binding.cu": "cb6e9991f8cefb66abde3bd3be85fd8b9f6d9490ee06bd4acc7a95a0cd4f74d9",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551_kernel.cu": "b186a83f7298f6a92215d2f47ac00497b7439e5eaca17acba6beb34b8c8be19b",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2_binding.cu": "db80f4e682c4ea96cfdc6b4a3c025054e9586e20b69410f460b869140ea725c2",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2_kernel.cu": "010b128512e71fe373e8d5bb4e12a302ca4c51bfbaab4050f0e4a2010b1580db",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30_binding.cu": "2c981a6911ad754d42f3f012900143721fc3ee5f801d08aeef2893c418bb3b37",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30_kernel.cu": "c5658f22aa02149bd8bc9a9aa7cc8dd3c1aa64362e7e933551713336b0eda819",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75_binding.cu": "b6cd6a856bdbd705207979544634175da061872db54b07f80d81243ddc3482af",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75_kernel.cu": "04b1e8cff5930181157a11094b416696f88af2224780e48c17532882f62e1483",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_cc574506bff228edc548": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_cc574506bff228edc548_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_cc574506bff228edc548",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cc574506bff228edc548_binding.cu": "6d6c30f58039d309851ccc393ddfbaab6fc5c836bc41ed56267bbde984ff5df6",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cc574506bff228edc548_kernel.cu": "35b5d30bc604ad78acdeb085a88caf07b603a13066704fe4e8b4419e99447a84",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cc574506bff228edc548_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cc574506bff228edc548_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e_binding.cu": "64df38992705ff0782c884405213cfea4e9bae0d2a35455f6189128ee905704b",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e_kernel.cu": "19735060848392e29206c985893749768354e8a942c0e6172c23518246a3f21a",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813_binding.cu": "e851ce07e9d929fac40f729ec1cc1beb5670a6fd00c2f87f78cee072b68a015e",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813_kernel.cu": "0a70939b2c54ca71925bd7878f51fb688ad9995de15d3af8cbd293d32bc7cc29",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d_binding.cu": "79468a7232681ae8ab4652f17838877970dcb995b7066721f0d8b4dca3017172",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d_kernel.cu": "e1c3904ecd7eb76bba59b8d3b449a807525d6c28dab8505d95ed13208a47ead6",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d_binding.cu": "acf8a6ce79517f54903fb93accea8e85c3906ba52bad50e49722f675f07f5e89",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d_kernel.cu": "0d18592cfd51d35bc98f2868b0f276d9ba3981f984f237fb822eea3e78549e67",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1_binding.cu": "5838dccc854107f7d0e2bae3238580400c23362b5673a85c0d0236712fb01c55",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1_kernel.cu": "c2b5afeadbb757793a3cf60f2edb12cfd994cfcad3af599d62883385aba219e9",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1_binding.cu",
        ],
    },
    "cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432": {
        "arg_plan": [
            ("buffer", "active_expert_tokens"),
            ("buffer", "expert_scales"),
            ("buffer", "token_input"),
            ("buffer", "residual"),
            ("buffer", "gamma"),
            ("buffer", "moe_allreduce_out"),
            ("buffer", "residual_out"),
            ("buffer", "norm_out"),
            ("buffer", "quant_out"),
            ("buffer", "scale_out"),
            ("buffer", "workspace_tensor"),
            ("raw_pointer", "workspace_control"),
            ("raw_pointer", "workspace_payload_0"),
            ("raw_pointer", "workspace_payload_1"),
            ("raw_pointer", "workspace_payload_2"),
            ("raw_pointer", "workspace_payload_3"),
            ("parameter", "world_rank"),
            ("parameter", "tokens"),
            ("parameter", "active_experts"),
            ("parameter", "epsilon"),
            ("parameter", "weight_bias"),
            ("parameter", "scale_factor"),
            ("parameter", "layout_code"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432_sm_100a",
        "compile_flags": [
            "--use_fast_math",
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 256,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432_binding.cu": "c9bb013c52a22f3680d15e29fe37e9ae8dd25c48ad519a2fd4652c97c2d186a1",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432_kernel.cu": "93098f75f75b5a2c1548c3f1a3e59f044189064e8fd820e567e9726242db0ba5",
        },
        "sources": [
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432_kernel.cu",
            "csrc/cake_trtllm_moe_allreduce_union/sm_100a/cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432_binding.cu",
        ],
    },
}
ROUTES: dict[tuple[str, bool, str], tuple[str, ...]] = {
    ("bfloat16", False, "generic"): (
        "cake_trtllm_moe_allreduce_union_1ea3d610e4c5f2fb9dfe",
        "cake_trtllm_moe_allreduce_union_f38c5bdae2850e919bd1",
        "cake_trtllm_moe_allreduce_union_1edae1e51d4c5344cb1e",
        "cake_trtllm_moe_allreduce_union_bc25afe050a9efe89d30",
    ),
    ("bfloat16", False, "t1_e8_serial_clear"): (
        "cake_trtllm_moe_allreduce_union_9d436078fa133b24495b",
        "cake_trtllm_moe_allreduce_union_cc574506bff228edc548",
        "cake_trtllm_moe_allreduce_union_a2f2997e9848d5ccfa8a",
        "cake_trtllm_moe_allreduce_union_87de697e8471d30da720",
    ),
    ("bfloat16", True, "generic"): (
        "cake_trtllm_moe_allreduce_union_4ad8b949226ea7cb6fa6",
        "cake_trtllm_moe_allreduce_union_ab7b6f0eca9debeaf551",
        "cake_trtllm_moe_allreduce_union_8e7170e56161075cfb65",
        "cake_trtllm_moe_allreduce_union_6925cdd7ab9abf9afdc8",
    ),
    ("float16", False, "generic"): (
        "cake_trtllm_moe_allreduce_union_7a5993556607a1facdce",
        "cake_trtllm_moe_allreduce_union_89e75b4e955f7e661d97",
        "cake_trtllm_moe_allreduce_union_adb939faf49f17c5def2",
        "cake_trtllm_moe_allreduce_union_dcd9410689074148ce9d",
    ),
    ("float16", False, "t64_e12_resident"): (
        "cake_trtllm_moe_allreduce_union_cf144a0dcaf634c0bc7e",
        "cake_trtllm_moe_allreduce_union_2c5b3050aea9079e2fb9",
        "cake_trtllm_moe_allreduce_union_c36f9d3b7f7cffc42b75",
        "cake_trtllm_moe_allreduce_union_438d651723864ef58d3b",
    ),
    ("float16", True, "generic"): (
        "cake_trtllm_moe_allreduce_union_7831765c69cf8ba63e98",
        "cake_trtllm_moe_allreduce_union_84e52e0b171f1fa1ecc7",
        "cake_trtllm_moe_allreduce_union_8e8f98718525df2c1481",
        "cake_trtllm_moe_allreduce_union_d8af8c82addb531f1813",
    ),
    ("float16", True, "t128_e16_owner_forward"): (
        "cake_trtllm_moe_allreduce_union_e438abd27c8c7145b93d",
        "cake_trtllm_moe_allreduce_union_f679285ae0a374b0d432",
        "cake_trtllm_moe_allreduce_union_86937d819491be970051",
        "cake_trtllm_moe_allreduce_union_a20c9fc47879979f14de",
    ),
}

SOURCE_PACKAGE = "cake_trtllm_moe_allreduce_union"
ARCH = "sm_100a"
DEVICE_CAPABILITY = (10, 0)
WORLD_SIZE = 4
HIDDEN_DIM = 7168
CLUSTER_CTAS = 4
# Public pointer-table layout of ``trtllm_create_ipc_workspace_for_all_reduce_fusion``:
# three world-sized pointer regions (buffer, flags, Lamport payload) followed by
# the local control pointer.  The kernel derives every device address from this
# table; the raw ``workspace_control`` / ``workspace_payload_<rank>`` bindings
# name the same allocations for the binding's ownership contract.
WORKSPACE_PAYLOAD_BASE_FACTOR = 2
WORKSPACE_CONTROL_INDEX_FACTOR = 3
LAYOUT_CODE_SWIZZLED_128X4 = 0

SPECIALIZATION_GENERIC = "generic"
SPECIALIZATION_T1_E8_SERIAL_CLEAR = "t1_e8_serial_clear"
SPECIALIZATION_T64_E12_RESIDENT = "t64_e12_resident"
SPECIALIZATION_T128_E16_OWNER_FORWARD = "t128_e16_owner_forward"
# Reviewed shape specializations of the union: (dtype, launch_with_pdl, tokens,
# active experts).  Every other configuration runs the generic schedule of its
# (dtype, launch_with_pdl) pair.
_REVIEWED_SPECIALIZATIONS = {
    ("bfloat16", False, 1, 8): SPECIALIZATION_T1_E8_SERIAL_CLEAR,
    ("float16", False, 64, 12): SPECIALIZATION_T64_E12_RESIDENT,
    ("float16", True, 128, 16): SPECIALIZATION_T128_E16_OWNER_FORWARD,
}
_DTYPE_NAME = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
_RAW_POINTER_KEYS = frozenset(
    ("workspace_control", *(f"workspace_payload_{peer}" for peer in range(WORLD_SIZE)))
)


def select_specialization(
    dtype_name: str, launch_with_pdl: bool, token_num: int, active_experts: int
) -> str:
    """Return the reviewed specialization name for one launch configuration."""

    return _REVIEWED_SPECIALIZATIONS.get(
        (dtype_name, bool(launch_with_pdl), int(token_num), int(active_experts)),
        SPECIALIZATION_GENERIC,
    )


def route_applies(
    *, world_size: int, device_capability: Sequence[int], emit_moe_allreduce: bool
) -> bool:
    """Whether the verified union export owns this Cake MoE all-reduce call."""

    return (
        bool(ROUTES)
        and int(world_size) == WORLD_SIZE
        and tuple(int(value) for value in device_capability) == DEVICE_CAPABILITY
        and bool(emit_moe_allreduce)
    )


def route_module_names(
    *, dtype_name: str, launch_with_pdl: bool, token_num: int, active_experts: int
) -> tuple[str, ...]:
    """Return the rank-ordered module names of one exported route."""

    key = (
        str(dtype_name),
        bool(launch_with_pdl),
        select_specialization(dtype_name, launch_with_pdl, token_num, active_experts),
    )
    names = ROUTES.get(key)
    if names is None:
        raise ValueError(f"unsupported Cake MoE all-reduce union route: {key}")
    return tuple(names)


def route_module_name(
    *,
    dtype_name: str,
    launch_with_pdl: bool,
    token_num: int,
    active_experts: int,
    world_rank: int,
) -> str:
    names = route_module_names(
        dtype_name=dtype_name,
        launch_with_pdl=launch_with_pdl,
        token_num=token_num,
        active_experts=active_experts,
    )
    if not 0 <= int(world_rank) < len(names):
        raise ValueError(f"world_rank must be in [0, {len(names)}), got {world_rank}")
    return names[int(world_rank)]


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake MoE all-reduce union CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 4:
        raise ValueError(
            f"exported source path is outside the union package: {relative!r}"
        )
    return _source_dir().joinpath(*parts[2:])


@functools.cache
def verified_sources(name: str) -> tuple[Path, ...]:
    """Resolve one module's sources and verify their recorded SHA-256 digests."""

    record = MODULES[name]
    paths = []
    for relative in record["sources"]:
        path = _source_path(relative)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"Cake MoE all-reduce union source is missing: {path}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = record["source_sha256"][relative]
        if digest != expected:
            raise RuntimeError(
                f"Cake MoE all-reduce union source {relative} does not match its "
                f"exported digest (expected {expected}, found {digest})"
            )
        paths.append(path)
    return tuple(paths)


@functools.cache
def spec(name: str) -> JitSpec:
    record = MODULES[name]
    return gen_jit_spec(
        name=record["cache_name"],
        sources=list(verified_sources(name)),
        extra_cuda_cflags=[*sm100a_nvcc_flags, *record["compile_flags"]],
        extra_include_paths=[_source_dir().parent],
    )


@functools.cache
def load(name: str):
    return spec(name).build_and_load()


_WORKSPACE_POINTERS: dict[int, tuple[weakref.ReferenceType, tuple[int, ...]]] = {}


def register_workspace_pointers(
    workspace_tensor: torch.Tensor, pointers: Sequence[int]
) -> None:
    """Retain the host-known pointer table behind an all-reduce workspace tensor."""

    if (
        not isinstance(workspace_tensor, torch.Tensor)
        or workspace_tensor.dtype != torch.int64
    ):
        raise TypeError("workspace_tensor must be an int64 torch.Tensor")
    values = tuple(int(value) for value in pointers)
    if workspace_tensor.numel() != len(values):
        raise ValueError("workspace pointer count does not match the workspace tensor")
    _WORKSPACE_POINTERS[int(workspace_tensor.data_ptr())] = (
        weakref.ref(workspace_tensor),
        values,
    )


def workspace_pointers(
    workspace_ptrs: torch.Tensor, world_size: int
) -> tuple[int, ...]:
    """Return the first ``3 * world_size + 1`` table entries without a device readback.

    Tables created by ``trtllm_create_ipc_workspace_for_all_reduce_fusion`` are
    registered at creation.  Any other table is read back exactly once outside
    CUDA Graph capture and then retained for the lifetime of that tensor.
    """

    needed = WORKSPACE_CONTROL_INDEX_FACTOR * int(world_size) + 1
    if workspace_ptrs.numel() < needed:
        raise ValueError(f"workspace_ptrs must contain at least {needed} pointers")
    address = int(workspace_ptrs.data_ptr())
    entry = _WORKSPACE_POINTERS.get(address)
    if entry is not None:
        owner, values = entry
        alive = owner()
        if (
            alive is not None
            and int(alive.data_ptr()) == address
            and len(values) >= needed
        ):
            return values[:needed]
        del _WORKSPACE_POINTERS[address]
    if (
        workspace_ptrs.device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "the Cake MoE all-reduce union needs the host-known workspace pointers before "
            "CUDA Graph capture; run one eager call first or create the workspace with "
            "trtllm_create_ipc_workspace_for_all_reduce_fusion"
        )
    values = tuple(int(value) for value in workspace_ptrs[:needed].tolist())
    _WORKSPACE_POINTERS[address] = (weakref.ref(workspace_ptrs), values)
    return values


def launch_grid_x(token_num: int, cooperative: bool, sm_count: int) -> int:
    """Resident-grid builds launch one cooperative cluster per token; every
    other build uses the SM-bounded persistent cluster grid."""

    if cooperative:
        grid = int(token_num) * CLUSTER_CTAS
    else:
        grid = (
            min(int(sm_count), int(token_num) * CLUSTER_CTAS) // CLUSTER_CTAS
        ) * CLUSTER_CTAS
    if grid <= 0:
        raise ValueError("the launch requires at least one complete four-CTA cluster")
    return grid


@functools.cache
def _dummy(device_index: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(1, dtype=dtype, device=torch.device("cuda", device_index))


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def run_cake_moe_allreduce_union(
    *,
    backend: Literal["cake"] = "cake",
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: torch.Tensor,
    moe_reduction_active_experts_token_input: torch.Tensor,
    moe_reduction_token_input: torch.Tensor,
    moe_allreduce_out: Optional[torch.Tensor],
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    weight_bias: Optional[float],
) -> None:
    """Select and launch one exported union module for the caller's validated tensors."""

    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    if int(world_size) != WORLD_SIZE:
        raise ValueError(
            f"the Cake MoE all-reduce union export covers world_size={WORLD_SIZE}"
        )
    if int(hidden_dim) != HIDDEN_DIM:
        raise ValueError(
            f"the Cake MoE all-reduce union export requires hidden_dim={HIDDEN_DIM}"
        )
    if moe_allreduce_out is None:
        raise ValueError("the Cake MoE all-reduce union export emits moe_allreduce_out")
    dtype = moe_reduction_active_experts_token_input.dtype
    dtype_name = _DTYPE_NAME.get(dtype)
    if dtype_name is None:
        raise ValueError(
            "the Cake MoE all-reduce union supports float16 and bfloat16 only"
        )
    device_index = moe_reduction_active_experts_token_input.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if tuple(torch.cuda.get_device_capability(device_index)) != DEVICE_CAPABILITY:
        raise ValueError("the Cake MoE all-reduce union export targets SM100 only")
    name = route_module_name(
        dtype_name=dtype_name,
        launch_with_pdl=bool(launch_with_pdl),
        token_num=int(token_num),
        active_experts=int(moe_reduction_device_num_experts),
        world_rank=int(world_rank),
    )
    record = MODULES[name]
    pointers = workspace_pointers(workspace_ptrs, WORLD_SIZE)
    dummy = _dummy(device_index, dtype)
    values: dict[str, object] = {
        "active_expert_tokens": moe_reduction_active_experts_token_input,
        "expert_scales": moe_reduction_scale_input,
        "token_input": moe_reduction_token_input,
        "residual": residual_in,
        "gamma": rms_gamma,
        "moe_allreduce_out": moe_allreduce_out,
        "residual_out": residual_out,
        "norm_out": norm_out,
        "quant_out": dummy,
        "scale_out": dummy,
        "workspace_tensor": workspace_ptrs,
        "workspace_control": pointers[WORKSPACE_CONTROL_INDEX_FACTOR * WORLD_SIZE],
        "world_rank": int(world_rank),
        "tokens": int(token_num),
        "active_experts": int(moe_reduction_device_num_experts),
        "epsilon": float(rms_eps),
        "weight_bias": float(0.0 if weight_bias is None else weight_bias),
        "scale_factor": float(scale_factor),
        "layout_code": LAYOUT_CODE_SWIZZLED_128X4,
        "grid_x": launch_grid_x(
            int(token_num), record["launch"]["cooperative"], _sm_count(device_index)
        ),
        "grid_y": 1,
        "grid_z": 1,
    }
    for peer in range(WORLD_SIZE):
        values[f"workspace_payload_{peer}"] = pointers[
            WORKSPACE_PAYLOAD_BASE_FACTOR * WORLD_SIZE + peer
        ]
    module = load(name)
    getattr(module, record["ffi_entry"])(
        *(values[key] for _kind, key in record["arg_plan"])
    )


__all__ = [
    "ARCH",
    "DEVICE_CAPABILITY",
    "HIDDEN_DIM",
    "MODULES",
    "ROUTES",
    "WORLD_SIZE",
    "launch_grid_x",
    "load",
    "register_workspace_pointers",
    "route_applies",
    "route_module_name",
    "route_module_names",
    "run_cake_moe_allreduce_union",
    "select_specialization",
    "spec",
    "verified_sources",
    "workspace_pointers",
]
