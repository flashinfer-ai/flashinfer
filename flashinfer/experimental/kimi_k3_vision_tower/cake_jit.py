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
# Kimi-K3 vision tower (MoonViT-3D encoder + PatchMergerV2; SM100 / SM103).
#
# ``MODULES`` holds one record per generated program (a kernel plus its host
# binding): the two translation units under ``csrc/``, the architectures the
# source is built for (``arches``; the loader compiles it with the exact flag
# set of the device it runs on), the extra compile flags, the FFI entry, the
# name of its argument plan in ``ARG_PLANS`` and the sealed closure identity
# per architecture.  One source serves every architecture it lists; a program
# listed for one architecture only is a form the host plan selects on that
# architecture alone (the plain one-tile and two-tile-split attention forms
# exist on SM103 only) or a kernel whose device code genuinely differs
# between the two (the attention kernels drain their scores with
# ``tcgen05.ld.red`` on SM103).
#
# ``ARG_PLANS`` lists each distinct argument plan once (``[kind, name]`` in
# launch order; ``kind`` is ``tma_buffer`` / ``buffer`` / ``parameter`` /
# ``grid``), keyed by the program kind that uses it.
#
# ``KERNELS`` maps ``"<arch>"`` to the logical kernel key -> program
# assignment the host launcher resolves at preparation:
#
# * ``gemm:<variant>:<tile>``   the fused tcgen05 BF16 GEMM family
#   (``pos``, ``norm_qkv_rope``, ``residual_wo``, ``norm_gelu``,
#   ``residual_fc1``, ``gelu_erf``, ``rmsnorm``) on the production tile
#   configuration ``select_tile_config`` picks for the token count in its
#   production PDL form;
# * ``gemm:<variant>:<tile>:pdle`` the same tile's ``PDL_EARLY`` binary (the
#   grid-dependency wait moved into the load / epilogue roles), which the host
#   launches only inside the form's census window (``cake_backend.pdl_early_on``)
#   and never on the stream-K / tail / multicast / ``pos`` tiles
#   (``cake_backend.pdl_early_selected``); same kernel parameters, same
#   numerics;
# * ``gemm:<variant>:<tile>`` with a ``*_h`` tile: the half-N tail twin of a
#   plain persistent tile, routed where the tile census leaves a tail that
#   fits one half-round (``cake_backend.half_tail_split``);
# * ``attention:<layout>[:wide][:split]``   the packed-varlen BF16 attention
#   kernel: ``<layout>`` = ``tiles2`` / ``tiles1`` / ``ring3`` (two-tile and
#   SPLIT_KV unit layouts, the host plan rule chooses per ``grid_thws`` batch;
#   ``ring3`` is the SPLIT_KV form with the shared-O three-deep score ring,
#   ``cake_backend.ring3_selected``); ``:wide`` = the build reading the
#   four-word unit table with the next-unit prefetch, selected on short rows
#   (``cake_backend.unit_prefetch_selected``); ``:split`` = the partial-output
#   build of the rows whose tail-round units the plan splits into K/V ranges
#   (``cake_backend.select_layout_and_split``), followed by
# * ``attention_merge``           the exact fixed-order merge of those split
#   units (one launch after every attention launch of a split row);
# * ``merge``                     final RMSNorm + 2x2 spatial / temporal-mean merge;
# * ``rmsnorm_apply``             the post-projector RMSNorm apply pass.
#
# The three literals are populated verbatim by the generated-program export;
# do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_vision_tower_05528121bdb20ac63f4a": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_05528121bdb20ac63f4a_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_05528121bdb20ac63f4a_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "6d908a3a9c5a6030c9d6fb3b829a8323210309febf89ce152b8080f9af783343",
        },
    },
    "cake_kimi_k3_vision_tower_08d980e26f7723727971": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_08d980e26f7723727971_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_08d980e26f7723727971_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "c16026bb897ae01908638e3ace94072348a671a2f5b3cc1bd7fac8ea6c30fcbb",
            "sm_103a": "0fc4ba3a43c212e5e1ded2013ece6ec0f0d0c901b6e8b37f37263c15e1a0e89c",
        },
    },
    "cake_kimi_k3_vision_tower_0edb981e0644cf0bc327": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_0edb981e0644cf0bc327_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_0edb981e0644cf0bc327_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6c633085f35a7deab8ff7dc69e9cdc139c660ca4949bf5c81252e3ff7977f62b",
            "sm_103a": "8ef9dc1745b24756839f7329c390e3133a3bafb49ea25120664d8ea716ac9637",
        },
    },
    "cake_kimi_k3_vision_tower_1111425111141a4c76aa": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_1111425111141a4c76aa_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_1111425111141a4c76aa_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "1276238f1e18754f41fa5fd2bdd7ec34017f9dab9370db923362eb3679afa0d0",
            "sm_103a": "3558b8c177066df7292fde2858cf9a7903102d2b03441bd391d51f7a3f17804c",
        },
    },
    "cake_kimi_k3_vision_tower_153c43aa8701296c68b2": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_153c43aa8701296c68b2_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_153c43aa8701296c68b2_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "92b303c4c73ebf96d436a20d7de2423fac59193e60c442fb26ea63200dadf1b8",
        },
    },
    "cake_kimi_k3_vision_tower_1d473f131d0bf9fe2508": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_1d473f131d0bf9fe2508_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_1d473f131d0bf9fe2508_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "288c224bd7161eb70a21edf2df8e37ed8fd735461a185c6408aa85417cec4d5c",
            "sm_103a": "68f41480088e6573287ba751373d52663035b54492664dff5f107161d861d93c",
        },
    },
    "cake_kimi_k3_vision_tower_22ce0ac3a66ecbb5f1de": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_22ce0ac3a66ecbb5f1de_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_22ce0ac3a66ecbb5f1de_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "06eb697b007e2805e38bcdb1504e5538812cf56a09bb29f922dcc377811aae91",
            "sm_103a": "eeddaec357f255ea6d3dff8640ac1d5d2d6a493d7e51134c177b43b156e9f92d",
        },
    },
    "cake_kimi_k3_vision_tower_23237a8f58b5d594d271": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_23237a8f58b5d594d271_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_23237a8f58b5d594d271_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "57daaa0c39b88d3831a280ff04059ca15387ccc50da0948ae2c210d8c8f31527",
        },
    },
    "cake_kimi_k3_vision_tower_233a1176df61a9a0c965": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_233a1176df61a9a0c965_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_233a1176df61a9a0c965_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "43b707ef24903b6bbcff12be95bb423d337325dfbaf6217445d8ba5344e3b87f",
            "sm_103a": "16b6cc575ad52b6709dfdb005c35d8ebab939a858cb76453a8b49cf2364b702c",
        },
    },
    "cake_kimi_k3_vision_tower_271dc4ac70c5d911e926": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_271dc4ac70c5d911e926_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_271dc4ac70c5d911e926_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "2b3fc7a334d2e5be141227af0b22944339420088233d6b000bb12580a77e99af",
            "sm_103a": "338203b695ea98545b01e0dfba04aeac479cb37acb84b95e9f6383047279355e",
        },
    },
    "cake_kimi_k3_vision_tower_27e643774380ec1f4d0d": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_27e643774380ec1f4d0d_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_27e643774380ec1f4d0d_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "74d7857a5f480e8473c5634a12e2d7105583ef409d400221e6c2c7cfc38f4505",
            "sm_103a": "0fa6babd9ac8874f9a239260edebde1ce3a23855a35e51f7012b4d9d0225358b",
        },
    },
    "cake_kimi_k3_vision_tower_2b1f73ebc160206c4ba9": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_2b1f73ebc160206c4ba9_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_2b1f73ebc160206c4ba9_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_100a": "6c7b1788111fc67b9913742b15cda9028da89e7215490679ef03482dab16e729",
        },
    },
    "cake_kimi_k3_vision_tower_2c6f70951276ee4ad16a": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_2c6f70951276ee4ad16a_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_2c6f70951276ee4ad16a_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "16857acb1db94751b3d8af82d16559e790aa6de9cff0d75d5ae996ccd26e733a",
            "sm_103a": "6ad78691739cd5c97e903a7b0fe1d7dad7172f415189cc15de255dc5580a50c9",
        },
    },
    "cake_kimi_k3_vision_tower_2d6aef50287c307414cf": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_2d6aef50287c307414cf_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_2d6aef50287c307414cf_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "9eda1cbd193f155b9fdacca6fe4b22ee77054efb3fdf800d392c9ebf3a1bbd98",
            "sm_103a": "701b82acd8001101f86a9b32a87f9700169e7b11a98f19375eac6f4e8df48311",
        },
    },
    "cake_kimi_k3_vision_tower_305b1af2834f5df4a215": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_305b1af2834f5df4a215_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_305b1af2834f5df4a215_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "0b0b38e9e937438a5f4d198b1e55342e7aabd56032d57475f3922f863f488735",
            "sm_103a": "1653ec136d4401beecf117db59c15908bea913ea4a232567ea21df1505d73644",
        },
    },
    "cake_kimi_k3_vision_tower_3192e923fe894b0e2b76": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3192e923fe894b0e2b76_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3192e923fe894b0e2b76_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "9778a6f86b10cef6e5e2c707edc47e7f37929bb469c8260de60e7a62ef3a30c7",
            "sm_103a": "3f54f42bbfff9ddcf790ce3ba4ed38b8a03b1134186463c410d82cbb656b8911",
        },
    },
    "cake_kimi_k3_vision_tower_34fb28355074f71ead09": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_34fb28355074f71ead09_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_34fb28355074f71ead09_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "1ebe7f2c669ce515c34be2f06fe9a1e7dbf11fc565537f8db2e7f774df60216f",
            "sm_103a": "c67576883ebd8b4109acad8306cf28667a08aa63afb40c0540261068ae1467ad",
        },
    },
    "cake_kimi_k3_vision_tower_37082353724ecfc06b4b": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_37082353724ecfc06b4b_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_37082353724ecfc06b4b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "9f3f5ff68131eeab7349ea094cac4864d13ec4b107c7f361c7aefa07a1524526",
            "sm_103a": "e0d92ad603a104a3d6f64676e693a12abb324b3f5dbf1c82f5b56789031441bb",
        },
    },
    "cake_kimi_k3_vision_tower_383aff1079a05f46ee76": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_383aff1079a05f46ee76_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_383aff1079a05f46ee76_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "036f3a3dd14afb0cf9ec16bea9769cd75663831f952be7f309a9d233a9c6f1de",
            "sm_103a": "06780908a58b89a2edacb326cfd27a60bc7ff8756f715e2a399240be9d3b0fc8",
        },
    },
    "cake_kimi_k3_vision_tower_3bce5c6a1f5bf7453e9c": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3bce5c6a1f5bf7453e9c_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3bce5c6a1f5bf7453e9c_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "109db36436d2b9145254eb7223a0f319dfe14ab538f55e9cdfd36b5b97087ff9",
            "sm_103a": "a8f30b162cd52ef5e91d8807998acca694c6a08b41fc1b70e2e8dc74e4581649",
        },
    },
    "cake_kimi_k3_vision_tower_3c66a23ea8d562de0cca": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3c66a23ea8d562de0cca_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3c66a23ea8d562de0cca_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "2ed055f91e40cdd2091f9cddade53ecd59e6adedd725d083c273569b7824b7a7",
            "sm_103a": "b441c53e9a86063beabfb7559c7354a2ee3759ed9e9ab7f65dabc1dcaba84cf3",
        },
    },
    "cake_kimi_k3_vision_tower_3fa4eee3bd95517c6334": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3fa4eee3bd95517c6334_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_3fa4eee3bd95517c6334_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "5f2f846faf36b9d75c20a96153101efb1573550a73022c05b37812f1c29f5e98",
            "sm_103a": "675811ff515d2a67b612d8ed915b9ceb46d6142129274e151af8c6d100fb55ca",
        },
    },
    "cake_kimi_k3_vision_tower_444f24ac65b233f0870e": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_444f24ac65b233f0870e_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_444f24ac65b233f0870e_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "eb54f966958cfe785603d7943963bcc9f0b716c10d5674fae4f8d5c0f6195679",
            "sm_103a": "6048e79d34189a0c6a728de8c8f1d5abf87d0ef7cf4d6d0f9bb150b2977c7c8d",
        },
    },
    "cake_kimi_k3_vision_tower_500195ea53ac5c8b3237": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_500195ea53ac5c8b3237_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_500195ea53ac5c8b3237_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "d65fe22c85226614dd2abae08c42bfe1856729bb4aeaff1c137575858923b8c7",
            "sm_103a": "fd1f152777ba3c49fe20c3424e4b5193c3367c2cd91a98bea051ff8d4cee4952",
        },
    },
    "cake_kimi_k3_vision_tower_500a20583d1bf09bca00": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_500a20583d1bf09bca00_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_500a20583d1bf09bca00_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "f3af527f14d5795c6940f74ff9b4cd57e0f465129e13162c7332ff6a4a2c3f92",
            "sm_103a": "c2ed6d09b9904ddfeea106155664751796846a27eebd5368e50140b835278e70",
        },
    },
    "cake_kimi_k3_vision_tower_549562cf9e312aa6b3cd": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_549562cf9e312aa6b3cd_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_549562cf9e312aa6b3cd_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "210c904437e2e5b27305e8100dc88f525a5d80a88e25c69dd51c1771589368c3",
            "sm_103a": "f0a13890e8a8c6d273b2ff54bceab260bdda5ebb97d85a37c5e606c8bdb90fb7",
        },
    },
    "cake_kimi_k3_vision_tower_578a6bdda6a300f3ad5b": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_578a6bdda6a300f3ad5b_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_578a6bdda6a300f3ad5b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "8ee89f2a095c6f8191ee41c8a22cf6f6879ada4e79a40542c8e0ac3d8b0af478",
            "sm_103a": "59673bdd34514692464941a4d7361be595fc83cb08a08bb023a9b8f84d6787f6",
        },
    },
    "cake_kimi_k3_vision_tower_588033a85f785e5ab47e": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_588033a85f785e5ab47e_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_588033a85f785e5ab47e_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "abc91da4e436b3239a36ea075188e45c1ce12b63077c93d989ef4b5ce6c2ef5d",
            "sm_103a": "ab7ab5c9800c798405c9b2453391da1574d442958d24e5612d6f5b7a42de4ef3",
        },
    },
    "cake_kimi_k3_vision_tower_5932b4a4ba51246a1b70": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_5932b4a4ba51246a1b70_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_5932b4a4ba51246a1b70_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6c7e5237aacbea4001756053cd3432a3dfec7bae180ebb4aea63935c1cb11231",
            "sm_103a": "5d4d191e37ef34c9b8ee24dbd7f7068402ee3108e911e0d4f4e6f7560ddf9e95",
        },
    },
    "cake_kimi_k3_vision_tower_5d0c411d41b3acc90942": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_5d0c411d41b3acc90942_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_5d0c411d41b3acc90942_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "c0a9deeb8e5f857dafdc791b8f605344f2497def6dbe2e824de744fe55d422ac",
            "sm_103a": "43e06b60ba05c5ecdc49c1aae0037d03e76ce22582e72bf55d6b98664f76eecb",
        },
    },
    "cake_kimi_k3_vision_tower_5f015168b8e10231ef00": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_5f015168b8e10231ef00_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_5f015168b8e10231ef00_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "ec73291d7e1d3c6eafe01950a2ed44aa2374f2a1d011f675b8e45b1327527720",
            "sm_103a": "15741d8360c293b8af086a77ff84149be99fb117779b21f66f4d9ff05cf867d8",
        },
    },
    "cake_kimi_k3_vision_tower_6493b5e12e1dff606c6d": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_6493b5e12e1dff606c6d_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_6493b5e12e1dff606c6d_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "259282dd58349f44242887558585daddf1b806b7715f256c5828abe4911b66dd",
            "sm_103a": "8a7efdc0e169f687975f823ef1b059825b5cb1cd940e91ac0ca185151ba6e591",
        },
    },
    "cake_kimi_k3_vision_tower_6a51ed9304f30c8815cc": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_6a51ed9304f30c8815cc_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_6a51ed9304f30c8815cc_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "261449cbf1e735a0cafc163f7bb60a677b40af5cb49999567a4be22232ba32b9",
        },
    },
    "cake_kimi_k3_vision_tower_71ff4f073d1a66f79bd1": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_71ff4f073d1a66f79bd1_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_71ff4f073d1a66f79bd1_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "23161760e5b35d0880d0863ef510fe1ce98192a0d771a3e48a091bcdc142257b",
            "sm_103a": "a00d3ea84dcde276d5b9d7f4c5ea9f1a5b0e2f86538c1683cc70a98cd3368187",
        },
    },
    "cake_kimi_k3_vision_tower_7315eace6540f14b3ad9": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7315eace6540f14b3ad9_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7315eace6540f14b3ad9_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "a0b3b95363212b0d6225a63d03a277bb14fc1d4675568b734cd84d4a2e4fd9ec",
            "sm_103a": "93b7f6f0171eb32cec5e979cb3af7b43bff6b652770d2b031835fbe7f66e2575",
        },
    },
    "cake_kimi_k3_vision_tower_756759066a3ea781e5d0": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_756759066a3ea781e5d0_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_756759066a3ea781e5d0_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "238168dcad2ab84f3519a75f58f6593e4e92a71cf542a7dbc005f14b46097803",
            "sm_103a": "a036657420822eff0c61e97eb8219039d752627055b641e3bb0c4251742f87d5",
        },
    },
    "cake_kimi_k3_vision_tower_77bc888412b990ccbb9b": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_77bc888412b990ccbb9b_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_77bc888412b990ccbb9b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "5da6317f914af12d99fc76aab700774e3a83484469ab0fecb4dbd007e0b1bf0b",
            "sm_103a": "9e901f25620a4e502b9b23dcf6fd8d2bbaf4f5c20061faec86023e3dc6cd4c85",
        },
    },
    "cake_kimi_k3_vision_tower_7994f90134f7dadf4bd4": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7994f90134f7dadf4bd4_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7994f90134f7dadf4bd4_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "3d7720128d2cedaa978a9b32bf18c43efe0e2c44ec8da36239984813373f35c7",
            "sm_103a": "b0ac64b76420f7ddc32dd0547dda8902e64f2784281993fc9cc0ee9d7b1ce3fa",
        },
    },
    "cake_kimi_k3_vision_tower_7bd7dea39285bd36a559": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7bd7dea39285bd36a559_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7bd7dea39285bd36a559_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "bd649389f90437085d9ea8b6d69b675d26fdd7eb584440845a001ed18ff84e0a",
            "sm_103a": "5a586444b9d1566478819ca7c3969dbbdf1cd0ce88ca1d0c864c0fa7f100f186",
        },
    },
    "cake_kimi_k3_vision_tower_7d1fd2b79d6a381fc4ac": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7d1fd2b79d6a381fc4ac_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7d1fd2b79d6a381fc4ac_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "rmsnorm_apply",
        "closure_sha256": {
            "sm_100a": "d41c75bb675b68b3c376673843dc7e42fdd9af68a525c515f71f2af06e529caf",
            "sm_103a": "8de61a6bed68230d16e0091b3a7ef4ef065c1a860fba7af84c965bc0932f198d",
        },
    },
    "cake_kimi_k3_vision_tower_7d601111390a7f44c612": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7d601111390a7f44c612_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_7d601111390a7f44c612_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6525dbcd7d621f10f63b7ab18083cd91765577e6b9225d4fece955b6de340b11",
            "sm_103a": "31e83a6d4ad17da161a5c7c4c1670b61806ca54817a92b04cfef63bdf3d9b726",
        },
    },
    "cake_kimi_k3_vision_tower_872157e810f7de49c890": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_872157e810f7de49c890_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_872157e810f7de49c890_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "33d07d8fbe0925188fe9eb4b8e1666740b7811dcd3d8252ed90627eb64728c74",
            "sm_103a": "57619cf502b7bfc01ce499d3cbf57492369e31798d065f4aeea1360a6d7d26f6",
        },
    },
    "cake_kimi_k3_vision_tower_8b3fb47bc4fd2f09dde3": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_8b3fb47bc4fd2f09dde3_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_8b3fb47bc4fd2f09dde3_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "b8aeb5f9411b36b12dc390b5f522517bb224b6fdfd16026d9700afaac0c7380e",
        },
    },
    "cake_kimi_k3_vision_tower_901428864fefe2c56b14": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_901428864fefe2c56b14_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_901428864fefe2c56b14_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "a53b46654bf866be1768c3d42b58bf3994f1e6bed2061a7ff61a1372938c1bc8",
        },
    },
    "cake_kimi_k3_vision_tower_91336598d9b2fdc0508a": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_91336598d9b2fdc0508a_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_91336598d9b2fdc0508a_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "b222f3590f1e45c655c1abe4db6f811eba5a22efb4c9b7fce1ced6f3ace28f71",
            "sm_103a": "dce2d22e935eaba45b5df9edb95fb6d77c54652a8e051f22eac3ee9fb1226c7f",
        },
    },
    "cake_kimi_k3_vision_tower_938f1e0c1b33d616fea7": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_938f1e0c1b33d616fea7_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_938f1e0c1b33d616fea7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "merge",
        "closure_sha256": {
            "sm_100a": "cde546df4209281b75f3105354f07ecab2d24dd74af32b8c8aabd753304a2fe9",
            "sm_103a": "cc61702ed0ac3c91757503baae0762d21db0406cf3ec59211014ec6501fbafab",
        },
    },
    "cake_kimi_k3_vision_tower_989472145506999c2ebb": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_989472145506999c2ebb_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_989472145506999c2ebb_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "42dfdb1a19c3477f886eff3795fdd0d340455f086c52482bf3c3590825aa0e4b",
            "sm_103a": "a683540a14422962e103fe796422e3d6e7b3022c570e9f06c13dd5641de1d97b",
        },
    },
    "cake_kimi_k3_vision_tower_99833ffa0cdf69c0acc7": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_99833ffa0cdf69c0acc7_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_99833ffa0cdf69c0acc7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "85a0605881b4aeb28d0b84a391a6849a41632b512a748374460b17237ac86bc8",
            "sm_103a": "97b70ae78d9819333ae3fe4a6aba39fdbdced65ffb262707aced1692aaa34b4c",
        },
    },
    "cake_kimi_k3_vision_tower_a4adf8330da8ca78e7a3": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_a4adf8330da8ca78e7a3_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_a4adf8330da8ca78e7a3_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "c9b34b977e7ce1d04d6333af6fd230b0866779b84c77ed0af3260fa2500b35ed",
            "sm_103a": "6b12548386f005a7894af0763e0370cf7250517cc36f8ab5fbf93058a56b24d4",
        },
    },
    "cake_kimi_k3_vision_tower_a7c57693ffc21df68ada": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_a7c57693ffc21df68ada_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_a7c57693ffc21df68ada_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "01855535beeb342e651de42763e7593eb0bac9a27563d3af80dca9bbf632a5f8",
            "sm_103a": "b43663f90fb66cc39a7de4fbf694fa1c80582fc3be4740705ca515e090eb351b",
        },
    },
    "cake_kimi_k3_vision_tower_b217c8895787ddb2a4de": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b217c8895787ddb2a4de_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b217c8895787ddb2a4de_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_100a": "061da2d9c6707ab2c49699b18540a2d965e4bb85ace68381d5bd35189fbaf1bb",
        },
    },
    "cake_kimi_k3_vision_tower_b2eeaf27a8eaffecafa2": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b2eeaf27a8eaffecafa2_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b2eeaf27a8eaffecafa2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6c396845da9e74e267149b50fae908137a41375c011de8e8ade084c6dbde3726",
            "sm_103a": "26b67af3d2643f5e6bf34655d9a4432096013c5b9275eaefbaa7afdf12c4ffff",
        },
    },
    "cake_kimi_k3_vision_tower_b7eb9c9816240252c48c": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b7eb9c9816240252c48c_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b7eb9c9816240252c48c_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_100a": "5bcd31ba18cb55f80b6a15538f39d359fa30d7ce80dd8baa6a8b8eab98d92cf2",
        },
    },
    "cake_kimi_k3_vision_tower_b8e7e62f4c62c1cc15de": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b8e7e62f4c62c1cc15de_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b8e7e62f4c62c1cc15de_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "4f584cb4a0d0eb2bd40e125b13d5921811cdcf4e6cefc0b74d1dfd9b98ba4587",
            "sm_103a": "305e6fdab22693e6fc69946faddfd666ea3f4b499c85bea2fab3911fabbaacab",
        },
    },
    "cake_kimi_k3_vision_tower_b9768b899c32133e295e": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b9768b899c32133e295e_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_b9768b899c32133e295e_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "4118ceee33d02939fc74dce80cd65e5e96f8c66b4e8a1cbaaaa353073abbeeac",
            "sm_103a": "ebe4ae53da36f751092180f6d20fe8e9540649fe4430c9e4f915bf7602d78bf4",
        },
    },
    "cake_kimi_k3_vision_tower_baca276f87b855ba86ee": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_baca276f87b855ba86ee_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_baca276f87b855ba86ee_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "13020ab80684c00bf1c924fb85baadc1fe5861a26de6f9c5ab6acb9a5e3eb174",
        },
    },
    "cake_kimi_k3_vision_tower_c13cb49b638bcff4e175": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c13cb49b638bcff4e175_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c13cb49b638bcff4e175_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "59315006e04c12d502d1a1f6459651a77c8b8ceafd156f2183a8b647c48475cc",
            "sm_103a": "7826e1ee9509a6cc0ba9da1bef311868d1f8e2e33320ab5ebac907c8f3c7db48",
        },
    },
    "cake_kimi_k3_vision_tower_c55f4fa4ba71d511955e": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c55f4fa4ba71d511955e_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c55f4fa4ba71d511955e_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "d5bff823318a82b86c34e6686b748ea3e1dd25cc403cc207cf6f62a4a5b02b9f",
            "sm_103a": "ac698e30da8946d9fcf9f2a58a9d66ee2509b63651f5bc21ab981c10a12262a8",
        },
    },
    "cake_kimi_k3_vision_tower_c69ccc304ef4503e90fe": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c69ccc304ef4503e90fe_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c69ccc304ef4503e90fe_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "cc2851276fb8df87ee0a9f574186b6bce3625be559a19259c5f88b87a922c6df",
            "sm_103a": "14c2cd7ebaf94cb6f7d66cf4af3dbad6323b9899c10b5f5b381dfeb5af3d14b8",
        },
    },
    "cake_kimi_k3_vision_tower_c907c55838c4fa0f4292": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c907c55838c4fa0f4292_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_c907c55838c4fa0f4292_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "72fca7687e0e9710c505340995154bc2a96bd80b077b3584efe7e6cea4aeaf57",
        },
    },
    "cake_kimi_k3_vision_tower_cdba842167bb848c365a": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_cdba842167bb848c365a_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_cdba842167bb848c365a_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "e95e7dc5e4f8692d5fbcacbbe5f5e6e06161e2d631e6fc71a49d678eefdc7a08",
            "sm_103a": "ee9ed162e477900886ede3784eda6b78e2cf2c69f1a86e62872ca6fb4e0fcd72",
        },
    },
    "cake_kimi_k3_vision_tower_ceb72031b5bae82ea07e": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_ceb72031b5bae82ea07e_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_ceb72031b5bae82ea07e_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "3eef8456fd70383c1f36834a2907b3f2fe3a8c6107ad31857e4a7471cf38c351",
            "sm_103a": "a96521b6cae9641515cd2af66a5e25f6969f2a63ab14d0cc132b3c04c2519495",
        },
    },
    "cake_kimi_k3_vision_tower_d027933e3830a35dbcaa": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_d027933e3830a35dbcaa_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_d027933e3830a35dbcaa_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "e85e34e2308ee861783031bb5b41642efcec77bc6edc37f370bf72d876d96a72",
            "sm_103a": "e8d2ac60c1ed0668efece294b47cd56cb98ca58e5d30fc66fc3d5104b042628a",
        },
    },
    "cake_kimi_k3_vision_tower_d3641cd21e427b2ed442": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_d3641cd21e427b2ed442_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_d3641cd21e427b2ed442_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_100a": "98860aec8b813eb76d37004acb13ef4339ed11ade93e086273c263bcf5aeb1ed",
        },
    },
    "cake_kimi_k3_vision_tower_d8e69a7d4433d7fe81a5": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_d8e69a7d4433d7fe81a5_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_d8e69a7d4433d7fe81a5_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "4171869b9f842326a684a840e93cf9386539e9693af136c1b9e782f81229bf6a",
            "sm_103a": "a6470505120958f68fda2d2bc4634a16cfb8cec1a3f8365482e0b3c0505e88ab",
        },
    },
    "cake_kimi_k3_vision_tower_db2e4e6986f3081ba04c": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_db2e4e6986f3081ba04c_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_db2e4e6986f3081ba04c_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "041304898aab93b2853e305cbbed9bd4f0a91658ee14c073174b5fb0b2722dba",
            "sm_103a": "05f7c6c49a1cd3041ffa6593d20aa9209a45a2e7c60d6422e01b11848a427569",
        },
    },
    "cake_kimi_k3_vision_tower_dfe112d51921ab20f27d": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_dfe112d51921ab20f27d_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_dfe112d51921ab20f27d_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "374bbdfffa4b4fe04f252be8b5cfa0c5a955d4bba27b1146e7bfbf2d8a8550c6",
            "sm_103a": "204be41fb5a7c462f0ecb1c35c2ed996e8eac86d99f142e765d70751a624b4e1",
        },
    },
    "cake_kimi_k3_vision_tower_e0b196f00aea91ddbf42": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e0b196f00aea91ddbf42_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e0b196f00aea91ddbf42_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "attention_merge",
        "closure_sha256": {
            "sm_100a": "8e97c7900f02de9f93e700b7dac5bccff4975a6eb1031cdeb5c1c8ccdc03a588",
            "sm_103a": "480ece8e2705f6e59b462d5c5ffb00402da622a4d29882cde3d4026794e251a4",
        },
    },
    "cake_kimi_k3_vision_tower_e0ffb5db037008a9ae4f": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e0ffb5db037008a9ae4f_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e0ffb5db037008a9ae4f_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "2e91c7727cbf0a356c629e180cf4edc85d8a28c11ccdadf91cad8a18383b4348",
            "sm_103a": "b3d57d6def751851c2bdf8721c54ddc480ddd4fb02222cf5c5c8b42d10ef10a6",
        },
    },
    "cake_kimi_k3_vision_tower_e23e007914f3e62dc498": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e23e007914f3e62dc498_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e23e007914f3e62dc498_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "58834f6aaf288ae8507f32c751e0130fb92f354ea8d200814d49b661f52daa2b",
            "sm_103a": "b087cfbd0f7ae1ebdf3a59a548b6569364ec59cdb3030fe5b56c64d076ffca9c",
        },
    },
    "cake_kimi_k3_vision_tower_e29b6e2b82f24c223c3c": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e29b6e2b82f24c223c3c_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e29b6e2b82f24c223c3c_binding.cu",
        ],
        "arches": ["sm_103a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_103a": "dda3d077268a9ccedb0b0a1eec944398430f82a426c3d02aac3d487d248906c8",
        },
    },
    "cake_kimi_k3_vision_tower_e4470b785885e4cf8e34": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e4470b785885e4cf8e34_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_e4470b785885e4cf8e34_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "f34bac196c8cc5d1ce6780d62ff953fd1e6832bcea692ec04b07c516e3d34f23",
            "sm_103a": "8aea0412507770bfc59716f7ac8a0b8b8e27750849da190951c33dc07e33f050",
        },
    },
    "cake_kimi_k3_vision_tower_ec6b4fe4671efd0cf5b4": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_ec6b4fe4671efd0cf5b4_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_ec6b4fe4671efd0cf5b4_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "e5e5614e3e7308706dcef739f8a804a4da15aae39b6dcd0f3a3814979a0910d1",
            "sm_103a": "596b6bae8a6612c3df3f5f5d4f84fd38b9097619d45d2d397ba47aeb3ece9b95",
        },
    },
    "cake_kimi_k3_vision_tower_ee92e52df4965a9d956c": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_ee92e52df4965a9d956c_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_ee92e52df4965a9d956c_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "e06112c160da79cda044a1b4ce9d5acad7b0181939817ffd4e58965556f18bfc",
            "sm_103a": "8c452d5f04073726d6d9a4392ab855d93cbf16827969663c3f1be5d48eeb17be",
        },
    },
    "cake_kimi_k3_vision_tower_eff7e7abe0f9d89c58d7": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_eff7e7abe0f9d89c58d7_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_eff7e7abe0f9d89c58d7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6006e3d2a2b34da3168868609d8a5f8defb04b4a2ff01d299b87bf2768efbfc5",
            "sm_103a": "14006cb7ed138432591be82aacc304662684f9287548598b7cd29ed1c832597b",
        },
    },
    "cake_kimi_k3_vision_tower_f3a513bb9c3f5376a578": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f3a513bb9c3f5376a578_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f3a513bb9c3f5376a578_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "4d317a2b8f561226e52d0dbcf22543753c877488481383c9f844f0e69e0e5240",
            "sm_103a": "44ad38ebfc7b87c45f8e06221bc9fe55bea22e51adb41ce4e4a5ca148932a733",
        },
    },
    "cake_kimi_k3_vision_tower_f404bc4408d53752e702": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f404bc4408d53752e702_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f404bc4408d53752e702_binding.cu",
        ],
        "arches": ["sm_100a"],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": "attention",
        "closure_sha256": {
            "sm_100a": "7e6c53e0e14077974e291ff6347bcca746a3c11200fa1d9d60bff9b94a56fbf3",
        },
    },
    "cake_kimi_k3_vision_tower_f4b3b1373781bf4578ad": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f4b3b1373781bf4578ad_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f4b3b1373781bf4578ad_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "9989ec0863b3f6dec9d17276278708367649a92951664dc3a592c0f20acb3be0",
            "sm_103a": "89cde02e18db6213674f095b8c08d5f34727c809ec3a62c3f36ce43d7ffba32d",
        },
    },
    "cake_kimi_k3_vision_tower_f6147e75f57721f530db": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f6147e75f57721f530db_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f6147e75f57721f530db_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "fdb513e99dac79a5642b226fa0588ae786f36e1034c8e05b2ca6591b659bad2d",
            "sm_103a": "caf6533f265c220c5e0a39f659fd47aa69944d5e02fbc2805c890a154200b1de",
        },
    },
    "cake_kimi_k3_vision_tower_f6e7c197edcd927c1ff4": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f6e7c197edcd927c1ff4_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_f6e7c197edcd927c1ff4_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "bd9118ad6c27ece7c197bdfc2b26232f5a141e69a55af9df7ea760c0d1cc5eb3",
            "sm_103a": "09a369e760d07475acad9b24dcb87b3e6dcae501fa86334e3662c20af55e357f",
        },
    },
    "cake_kimi_k3_vision_tower_fb3646f34c27b5490c69": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_fb3646f34c27b5490c69_kernel.cu",
            "cake_kimi_k3_vision_tower/cake_kimi_k3_vision_tower_fb3646f34c27b5490c69_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "59599843ecca702ea02af5387e8a047726655129d00507e6b65d6bd9976cdb83",
            "sm_103a": "15b96f338e38fd99c45942e33bf76949a43147696a3607840af7efdbe6931f06",
        },
    },
}

ARG_PLANS: dict[str, list[list[str]]] = {
    "attention": [
        ["tma_buffer", "Q"],
        ["buffer", "Q_raw"],
        ["tma_buffer", "K"],
        ["tma_buffer", "V"],
        ["buffer", "O"],
        ["tma_buffer", "O_tma"],
        ["buffer", "seg_begin"],
        ["buffer", "seg_len"],
        ["buffer", "unit_table"],
        ["buffer", "probe"],
        ["parameter", "total_tiles"],
        ["parameter", "num_heads"],
        ["parameter", "softmax_scale_log2"],
        ["buffer", "partial_O"],
        ["buffer", "partial_ML"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "attention_merge": [
        ["buffer", "partial_O"],
        ["buffer", "partial_ML"],
        ["buffer", "merge_table"],
        ["buffer", "seg_begin"],
        ["buffer", "seg_len"],
        ["buffer", "O"],
        ["parameter", "num_heads"],
        ["parameter", "part_rows"],
        ["parameter", "softmax_scale_log2"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm": [
        ["tma_buffer", "A"],
        ["tma_buffer", "A2"],
        ["tma_buffer", "B"],
        ["buffer", "C"],
        ["buffer", "C2"],
        ["buffer", "C3"],
        ["buffer", "R"],
        ["buffer", "COS"],
        ["buffer", "SIN"],
        ["buffer", "CS"],
        ["buffer", "SQ"],
        ["buffer", "XW"],
        ["buffer", "WN"],
        ["buffer", "WS"],
        ["buffer", "FLAGS"],
        ["tma_buffer", "RT"],
        ["tma_buffer", "CT"],
        ["tma_buffer", "XWT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "full_tiles"],
        ["parameter", "tail_split"],
        ["parameter", "pf_l2"],
        ["parameter", "eps"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "merge": [
        ["buffer", "x"],
        ["buffer", "norm_weight"],
        ["buffer", "merge_table"],
        ["buffer", "m_out"],
        ["parameter", "eps"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "rmsnorm_apply": [
        ["buffer", "y"],
        ["buffer", "weight"],
        ["buffer", "rowsumsq"],
        ["parameter", "M"],
        ["parameter", "eps"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

KERNELS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "attention:ring3": "cake_kimi_k3_vision_tower_b217c8895787ddb2a4de",
        "attention:ring3:split": "cake_kimi_k3_vision_tower_b7eb9c9816240252c48c",
        "attention:ring3:wide": "cake_kimi_k3_vision_tower_d3641cd21e427b2ed442",
        "attention:tiles2": "cake_kimi_k3_vision_tower_f404bc4408d53752e702",
        "attention:tiles2:wide": "cake_kimi_k3_vision_tower_2b1f73ebc160206c4ba9",
        "attention_merge": "cake_kimi_k3_vision_tower_e0b196f00aea91ddbf42",
        "gemm:gelu_erf:l": "cake_kimi_k3_vision_tower_f4b3b1373781bf4578ad",
        "gemm:gelu_erf:l_h": "cake_kimi_k3_vision_tower_db2e4e6986f3081ba04c",
        "gemm:gelu_erf:l_sk": "cake_kimi_k3_vision_tower_fb3646f34c27b5490c69",
        "gemm:gelu_erf:s": "cake_kimi_k3_vision_tower_71ff4f073d1a66f79bd1",
        "gemm:gelu_erf:s:pdle": "cake_kimi_k3_vision_tower_d027933e3830a35dbcaa",
        "gemm:gelu_erf:s_h": "cake_kimi_k3_vision_tower_3192e923fe894b0e2b76",
        "gemm:gelu_erf:s_h:pdle": "cake_kimi_k3_vision_tower_ceb72031b5bae82ea07e",
        "gemm:gelu_erf:xs": "cake_kimi_k3_vision_tower_3bce5c6a1f5bf7453e9c",
        "gemm:gelu_erf:xs:pdle": "cake_kimi_k3_vision_tower_ec6b4fe4671efd0cf5b4",
        "gemm:norm_gelu:l_e8": "cake_kimi_k3_vision_tower_7bd7dea39285bd36a559",
        "gemm:norm_gelu:l_e8:pdle": "cake_kimi_k3_vision_tower_500a20583d1bf09bca00",
        "gemm:norm_gelu:l_e8_h": "cake_kimi_k3_vision_tower_d8e69a7d4433d7fe81a5",
        "gemm:norm_gelu:l_e8_h:pdle": "cake_kimi_k3_vision_tower_1111425111141a4c76aa",
        "gemm:norm_gelu:m_e8:pdle": "cake_kimi_k3_vision_tower_34fb28355074f71ead09",
        "gemm:norm_gelu:m_e8_h:pdle": "cake_kimi_k3_vision_tower_383aff1079a05f46ee76",
        "gemm:norm_gelu:s:pdle": "cake_kimi_k3_vision_tower_5f015168b8e10231ef00",
        "gemm:norm_gelu:s_h:pdle": "cake_kimi_k3_vision_tower_578a6bdda6a300f3ad5b",
        "gemm:norm_gelu:xs:pdle": "cake_kimi_k3_vision_tower_b2eeaf27a8eaffecafa2",
        "gemm:norm_qkv_rope:l_e8_cs": "cake_kimi_k3_vision_tower_271dc4ac70c5d911e926",
        "gemm:norm_qkv_rope:l_e8_cs:pdle": "cake_kimi_k3_vision_tower_77bc888412b990ccbb9b",
        "gemm:norm_qkv_rope:l_e8_cs_h": "cake_kimi_k3_vision_tower_e0ffb5db037008a9ae4f",
        "gemm:norm_qkv_rope:l_e8_cs_h:pdle": "cake_kimi_k3_vision_tower_f3a513bb9c3f5376a578",
        "gemm:norm_qkv_rope:m_e8_cs:pdle": "cake_kimi_k3_vision_tower_7994f90134f7dadf4bd4",
        "gemm:norm_qkv_rope:m_e8_cs_h:pdle": "cake_kimi_k3_vision_tower_756759066a3ea781e5d0",
        "gemm:norm_qkv_rope:s_cs:pdle": "cake_kimi_k3_vision_tower_37082353724ecfc06b4b",
        "gemm:norm_qkv_rope:s_cs_h:pdle": "cake_kimi_k3_vision_tower_c55f4fa4ba71d511955e",
        "gemm:norm_qkv_rope:xs_cs_pf:pdle": "cake_kimi_k3_vision_tower_99833ffa0cdf69c0acc7",
        "gemm:pos_sqxw:s_e8_pf": "cake_kimi_k3_vision_tower_1d473f131d0bf9fe2508",
        "gemm:pos_sqxw:xs_pf": "cake_kimi_k3_vision_tower_7d601111390a7f44c612",
        "gemm:residual_fc1:l_e8_pf:pdle": "cake_kimi_k3_vision_tower_989472145506999c2ebb",
        "gemm:residual_fc1:l_e8_pf_h:pdle": "cake_kimi_k3_vision_tower_27e643774380ec1f4d0d",
        "gemm:residual_fc1:l_sk": "cake_kimi_k3_vision_tower_ee92e52df4965a9d956c",
        "gemm:residual_fc1:m_p": "cake_kimi_k3_vision_tower_6493b5e12e1dff606c6d",
        "gemm:residual_fc1:m_p:pdle": "cake_kimi_k3_vision_tower_7315eace6540f14b3ad9",
        "gemm:residual_fc1:m_p_h:pdle": "cake_kimi_k3_vision_tower_305b1af2834f5df4a215",
        "gemm:residual_fc1:m_sk": "cake_kimi_k3_vision_tower_08d980e26f7723727971",
        "gemm:residual_fc1:s_e8_pf:pdle": "cake_kimi_k3_vision_tower_444f24ac65b233f0870e",
        "gemm:residual_fc1:xs_k4_pf:pdle": "cake_kimi_k3_vision_tower_2d6aef50287c307414cf",
        "gemm:residual_fc1:xs_pf:pdle": "cake_kimi_k3_vision_tower_549562cf9e312aa6b3cd",
        "gemm:residual_fc1_sqxw:l_e8_pf:pdle": "cake_kimi_k3_vision_tower_500195ea53ac5c8b3237",
        "gemm:residual_fc1_sqxw:l_e8_pf_h:pdle": "cake_kimi_k3_vision_tower_e4470b785885e4cf8e34",
        "gemm:residual_fc1_sqxw:l_sk": "cake_kimi_k3_vision_tower_c69ccc304ef4503e90fe",
        "gemm:residual_fc1_sqxw:m_p": "cake_kimi_k3_vision_tower_f6147e75f57721f530db",
        "gemm:residual_fc1_sqxw:m_p:pdle": "cake_kimi_k3_vision_tower_22ce0ac3a66ecbb5f1de",
        "gemm:residual_fc1_sqxw:m_p_h:pdle": "cake_kimi_k3_vision_tower_5932b4a4ba51246a1b70",
        "gemm:residual_fc1_sqxw:m_sk": "cake_kimi_k3_vision_tower_5d0c411d41b3acc90942",
        "gemm:residual_fc1_sqxw:s_e8_pf:pdle": "cake_kimi_k3_vision_tower_dfe112d51921ab20f27d",
        "gemm:residual_fc1_sqxw:xs_k4_pf:pdle": "cake_kimi_k3_vision_tower_91336598d9b2fdc0508a",
        "gemm:residual_fc1_sqxw:xs_pf:pdle": "cake_kimi_k3_vision_tower_0edb981e0644cf0bc327",
        "gemm:residual_wo_sqxw:m_tma1": "cake_kimi_k3_vision_tower_e23e007914f3e62dc498",
        "gemm:residual_wo_sqxw:m_tma1:pdle": "cake_kimi_k3_vision_tower_b9768b899c32133e295e",
        "gemm:residual_wo_sqxw:m_tma1_h": "cake_kimi_k3_vision_tower_b8e7e62f4c62c1cc15de",
        "gemm:residual_wo_sqxw:m_tma1_h:pdle": "cake_kimi_k3_vision_tower_c13cb49b638bcff4e175",
        "gemm:residual_wo_sqxw:s_e8_pf:pdle": "cake_kimi_k3_vision_tower_eff7e7abe0f9d89c58d7",
        "gemm:residual_wo_sqxw:s_e8_pf_h:pdle": "cake_kimi_k3_vision_tower_a7c57693ffc21df68ada",
        "gemm:residual_wo_sqxw:xs_pf:pdle": "cake_kimi_k3_vision_tower_3c66a23ea8d562de0cca",
        "gemm:rmsnorm:l": "cake_kimi_k3_vision_tower_2c6f70951276ee4ad16a",
        "gemm:rmsnorm:l:pdle": "cake_kimi_k3_vision_tower_588033a85f785e5ab47e",
        "gemm:rmsnorm:l_h": "cake_kimi_k3_vision_tower_872157e810f7de49c890",
        "gemm:rmsnorm:l_sk": "cake_kimi_k3_vision_tower_a4adf8330da8ca78e7a3",
        "gemm:rmsnorm:s": "cake_kimi_k3_vision_tower_cdba842167bb848c365a",
        "gemm:rmsnorm:s:pdle": "cake_kimi_k3_vision_tower_f6e7c197edcd927c1ff4",
        "gemm:rmsnorm:s_h": "cake_kimi_k3_vision_tower_233a1176df61a9a0c965",
        "gemm:rmsnorm:s_h:pdle": "cake_kimi_k3_vision_tower_3fa4eee3bd95517c6334",
        "merge": "cake_kimi_k3_vision_tower_938f1e0c1b33d616fea7",
        "rmsnorm_apply": "cake_kimi_k3_vision_tower_7d1fd2b79d6a381fc4ac",
    },
    "sm_103a": {
        "attention:ring3": "cake_kimi_k3_vision_tower_baca276f87b855ba86ee",
        "attention:ring3:split": "cake_kimi_k3_vision_tower_6a51ed9304f30c8815cc",
        "attention:ring3:wide": "cake_kimi_k3_vision_tower_23237a8f58b5d594d271",
        "attention:tiles1": "cake_kimi_k3_vision_tower_c907c55838c4fa0f4292",
        "attention:tiles1:split": "cake_kimi_k3_vision_tower_901428864fefe2c56b14",
        "attention:tiles1:wide": "cake_kimi_k3_vision_tower_8b3fb47bc4fd2f09dde3",
        "attention:tiles2": "cake_kimi_k3_vision_tower_05528121bdb20ac63f4a",
        "attention:tiles2:split": "cake_kimi_k3_vision_tower_e29b6e2b82f24c223c3c",
        "attention:tiles2:wide": "cake_kimi_k3_vision_tower_153c43aa8701296c68b2",
        "attention_merge": "cake_kimi_k3_vision_tower_e0b196f00aea91ddbf42",
        "gemm:gelu_erf:l": "cake_kimi_k3_vision_tower_f4b3b1373781bf4578ad",
        "gemm:gelu_erf:l_h": "cake_kimi_k3_vision_tower_db2e4e6986f3081ba04c",
        "gemm:gelu_erf:l_sk": "cake_kimi_k3_vision_tower_fb3646f34c27b5490c69",
        "gemm:gelu_erf:s": "cake_kimi_k3_vision_tower_71ff4f073d1a66f79bd1",
        "gemm:gelu_erf:s:pdle": "cake_kimi_k3_vision_tower_d027933e3830a35dbcaa",
        "gemm:gelu_erf:s_h": "cake_kimi_k3_vision_tower_3192e923fe894b0e2b76",
        "gemm:gelu_erf:s_h:pdle": "cake_kimi_k3_vision_tower_ceb72031b5bae82ea07e",
        "gemm:gelu_erf:xs": "cake_kimi_k3_vision_tower_3bce5c6a1f5bf7453e9c",
        "gemm:gelu_erf:xs:pdle": "cake_kimi_k3_vision_tower_ec6b4fe4671efd0cf5b4",
        "gemm:norm_gelu:l_e8": "cake_kimi_k3_vision_tower_7bd7dea39285bd36a559",
        "gemm:norm_gelu:l_e8:pdle": "cake_kimi_k3_vision_tower_500a20583d1bf09bca00",
        "gemm:norm_gelu:l_e8_h": "cake_kimi_k3_vision_tower_d8e69a7d4433d7fe81a5",
        "gemm:norm_gelu:l_e8_h:pdle": "cake_kimi_k3_vision_tower_1111425111141a4c76aa",
        "gemm:norm_gelu:m_e8:pdle": "cake_kimi_k3_vision_tower_34fb28355074f71ead09",
        "gemm:norm_gelu:m_e8_h:pdle": "cake_kimi_k3_vision_tower_383aff1079a05f46ee76",
        "gemm:norm_gelu:s:pdle": "cake_kimi_k3_vision_tower_5f015168b8e10231ef00",
        "gemm:norm_gelu:s_h:pdle": "cake_kimi_k3_vision_tower_578a6bdda6a300f3ad5b",
        "gemm:norm_gelu:xs:pdle": "cake_kimi_k3_vision_tower_b2eeaf27a8eaffecafa2",
        "gemm:norm_qkv_rope:l_e8_cs": "cake_kimi_k3_vision_tower_271dc4ac70c5d911e926",
        "gemm:norm_qkv_rope:l_e8_cs:pdle": "cake_kimi_k3_vision_tower_77bc888412b990ccbb9b",
        "gemm:norm_qkv_rope:l_e8_cs_h": "cake_kimi_k3_vision_tower_e0ffb5db037008a9ae4f",
        "gemm:norm_qkv_rope:l_e8_cs_h:pdle": "cake_kimi_k3_vision_tower_f3a513bb9c3f5376a578",
        "gemm:norm_qkv_rope:m_e8_cs:pdle": "cake_kimi_k3_vision_tower_7994f90134f7dadf4bd4",
        "gemm:norm_qkv_rope:m_e8_cs_h:pdle": "cake_kimi_k3_vision_tower_756759066a3ea781e5d0",
        "gemm:norm_qkv_rope:s_cs:pdle": "cake_kimi_k3_vision_tower_37082353724ecfc06b4b",
        "gemm:norm_qkv_rope:s_cs_h:pdle": "cake_kimi_k3_vision_tower_c55f4fa4ba71d511955e",
        "gemm:norm_qkv_rope:xs_cs_pf:pdle": "cake_kimi_k3_vision_tower_99833ffa0cdf69c0acc7",
        "gemm:pos_sqxw:s_e8_pf": "cake_kimi_k3_vision_tower_1d473f131d0bf9fe2508",
        "gemm:pos_sqxw:xs_pf": "cake_kimi_k3_vision_tower_7d601111390a7f44c612",
        "gemm:residual_fc1:l_e8_pf:pdle": "cake_kimi_k3_vision_tower_989472145506999c2ebb",
        "gemm:residual_fc1:l_e8_pf_h:pdle": "cake_kimi_k3_vision_tower_27e643774380ec1f4d0d",
        "gemm:residual_fc1:l_sk": "cake_kimi_k3_vision_tower_ee92e52df4965a9d956c",
        "gemm:residual_fc1:m_p": "cake_kimi_k3_vision_tower_6493b5e12e1dff606c6d",
        "gemm:residual_fc1:m_p:pdle": "cake_kimi_k3_vision_tower_7315eace6540f14b3ad9",
        "gemm:residual_fc1:m_p_h:pdle": "cake_kimi_k3_vision_tower_305b1af2834f5df4a215",
        "gemm:residual_fc1:m_sk": "cake_kimi_k3_vision_tower_08d980e26f7723727971",
        "gemm:residual_fc1:s_e8_pf:pdle": "cake_kimi_k3_vision_tower_444f24ac65b233f0870e",
        "gemm:residual_fc1:xs_k4_pf:pdle": "cake_kimi_k3_vision_tower_2d6aef50287c307414cf",
        "gemm:residual_fc1:xs_pf:pdle": "cake_kimi_k3_vision_tower_549562cf9e312aa6b3cd",
        "gemm:residual_fc1_sqxw:l_e8_pf:pdle": "cake_kimi_k3_vision_tower_500195ea53ac5c8b3237",
        "gemm:residual_fc1_sqxw:l_e8_pf_h:pdle": "cake_kimi_k3_vision_tower_e4470b785885e4cf8e34",
        "gemm:residual_fc1_sqxw:l_sk": "cake_kimi_k3_vision_tower_c69ccc304ef4503e90fe",
        "gemm:residual_fc1_sqxw:m_p": "cake_kimi_k3_vision_tower_f6147e75f57721f530db",
        "gemm:residual_fc1_sqxw:m_p:pdle": "cake_kimi_k3_vision_tower_22ce0ac3a66ecbb5f1de",
        "gemm:residual_fc1_sqxw:m_p_h:pdle": "cake_kimi_k3_vision_tower_5932b4a4ba51246a1b70",
        "gemm:residual_fc1_sqxw:m_sk": "cake_kimi_k3_vision_tower_5d0c411d41b3acc90942",
        "gemm:residual_fc1_sqxw:s_e8_pf:pdle": "cake_kimi_k3_vision_tower_dfe112d51921ab20f27d",
        "gemm:residual_fc1_sqxw:xs_k4_pf:pdle": "cake_kimi_k3_vision_tower_91336598d9b2fdc0508a",
        "gemm:residual_fc1_sqxw:xs_pf:pdle": "cake_kimi_k3_vision_tower_0edb981e0644cf0bc327",
        "gemm:residual_wo_sqxw:m_tma1": "cake_kimi_k3_vision_tower_e23e007914f3e62dc498",
        "gemm:residual_wo_sqxw:m_tma1:pdle": "cake_kimi_k3_vision_tower_b9768b899c32133e295e",
        "gemm:residual_wo_sqxw:m_tma1_h": "cake_kimi_k3_vision_tower_b8e7e62f4c62c1cc15de",
        "gemm:residual_wo_sqxw:m_tma1_h:pdle": "cake_kimi_k3_vision_tower_c13cb49b638bcff4e175",
        "gemm:residual_wo_sqxw:s_e8_pf:pdle": "cake_kimi_k3_vision_tower_eff7e7abe0f9d89c58d7",
        "gemm:residual_wo_sqxw:s_e8_pf_h:pdle": "cake_kimi_k3_vision_tower_a7c57693ffc21df68ada",
        "gemm:residual_wo_sqxw:xs_pf:pdle": "cake_kimi_k3_vision_tower_3c66a23ea8d562de0cca",
        "gemm:rmsnorm:l": "cake_kimi_k3_vision_tower_2c6f70951276ee4ad16a",
        "gemm:rmsnorm:l:pdle": "cake_kimi_k3_vision_tower_588033a85f785e5ab47e",
        "gemm:rmsnorm:l_h": "cake_kimi_k3_vision_tower_872157e810f7de49c890",
        "gemm:rmsnorm:l_sk": "cake_kimi_k3_vision_tower_a4adf8330da8ca78e7a3",
        "gemm:rmsnorm:s": "cake_kimi_k3_vision_tower_cdba842167bb848c365a",
        "gemm:rmsnorm:s:pdle": "cake_kimi_k3_vision_tower_f6e7c197edcd927c1ff4",
        "gemm:rmsnorm:s_h": "cake_kimi_k3_vision_tower_233a1176df61a9a0c965",
        "gemm:rmsnorm:s_h:pdle": "cake_kimi_k3_vision_tower_3fa4eee3bd95517c6334",
        "merge": "cake_kimi_k3_vision_tower_938f1e0c1b33d616fea7",
        "rmsnorm_apply": "cake_kimi_k3_vision_tower_7d1fd2b79d6a381fc4ac",
    },
}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


PDL_EARLY_KEY_SUFFIX = "pdle"


def gemm_kernel_key(variant: str, tile: str, pdl_early: bool = False) -> str:
    """``gemm:<variant>:<tile>`` for the production PDL binary of the tile,
    ``gemm:<variant>:<tile>:pdle`` for its PDL_EARLY binary."""
    key = f"gemm:{variant}:{tile}"
    return f"{key}:{PDL_EARLY_KEY_SUFFIX}" if pdl_early else key


ATTENTION_WIDE_KEY_SUFFIX = (
    "wide"  # UNIT_PREFETCH build (wide unit table, next-unit prefetch)
)
ATTENTION_SPLIT_KEY_SUFFIX = (
    "split"  # KV_PARTIAL build (partial O / (m, l) workspace, merge launch)
)


def attention_kernel_key(
    tiles_per_cta: int, ring3: bool = False, *, wide: bool = False, split: bool = False
) -> str:
    """``attention:<layout>[:wide][:split]``: ``tiles<n>`` for the plain layouts, ``ring3`` for the
    production SPLIT_KV form with the shared-O score ring (selected per arch / longest segment by the
    plan), ``:wide`` for the prefetching wide-table build of the short rows, ``:split`` for the
    partial-output build of the rows with split units."""
    if ring3:
        if int(tiles_per_cta) != 1:
            raise ValueError(
                "the ring3 attention form is the SPLIT_KV (one-tile) layout"
            )
        key = "attention:ring3"
    else:
        key = f"attention:tiles{int(tiles_per_cta)}"
    if wide:
        key += f":{ATTENTION_WIDE_KEY_SUFFIX}"
    if split:
        key += f":{ATTENTION_SPLIT_KEY_SUFFIX}"
    return key


def parse_attention_kernel_key(key: str) -> dict[str, Any]:
    """``dict(tiles_per_cta, ring3, wide, split)`` of an ``attention:`` key."""
    parts = key.split(":")
    if (
        parts[0] != "attention"
        or len(parts) < 2
        or parts[1] not in ("tiles2", "tiles1", "ring3")
    ):
        raise ValueError(f"not an attention kernel key: {key!r}")
    flags = parts[2:]
    allowed = [ATTENTION_WIDE_KEY_SUFFIX, ATTENTION_SPLIT_KEY_SUFFIX]
    if flags != [f for f in allowed if f in flags] or len(set(flags)) != len(flags):
        raise ValueError(f"malformed attention kernel key: {key!r}")
    return dict(
        tiles_per_cta=1 if parts[1] in ("tiles1", "ring3") else 2,
        ring3=parts[1] == "ring3",
        wide=ATTENTION_WIDE_KEY_SUFFIX in flags,
        split=ATTENTION_SPLIT_KEY_SUFFIX in flags,
    )


ATTENTION_MERGE_KERNEL_KEY = "attention_merge"
MERGE_KERNEL_KEY = "merge"
RMSNORM_APPLY_KERNEL_KEY = "rmsnorm_apply"


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when ``arch`` is registered and carries every key in ``required_keys``."""
    table = KERNELS.get(arch)
    return table is not None and all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered program for ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if table is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 vision tower programs for {arch} are not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#4568)"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 vision tower kernel {key!r} for {arch} is not "
            "registered in this checkout (see flashinfer-ai/flashinfer#4568)"
        )
    if arch not in MODULES[name]["arches"]:
        raise RuntimeError(
            f"registered program {name!r} is not built for {arch} "
            f"(arches {MODULES[name]['arches']})"
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
def gen_cake_kimi_k3_vision_tower_module(name: str, arch: str):
    """JIT spec of program ``name`` built for ``arch``: the shared source compiled with the
    exact flag set of that architecture; the architecture and the sealed closure identity are
    part of the spec name, so two architectures never share one cached library."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(f"program {name!r} is not built for {arch}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_{arch}_" + record["closure_sha256"][arch][:12],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_kimi_k3_vision_tower_module(name: str, arch: str):
    return gen_cake_kimi_k3_vision_tower_module(name, arch).build_and_load()
