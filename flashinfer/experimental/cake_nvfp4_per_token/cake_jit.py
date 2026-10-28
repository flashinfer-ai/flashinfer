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

# Registry of the generated programs of the per-token NVFP4 path (SM100 /
# SM103).
#
# ``ARG_PLANS`` holds the five argument orders of the generated host
# launchers as ``[kind, name]`` pairs (``tma_buffer`` / ``buffer`` tensors,
# ``parameter`` scalars and the ``grid`` dims; ``cake_backend._entry`` orders
# a call by name): ``quant`` (the quantizer), ``gemm`` (the 1-CTA GEMM
# programs of both orientations), ``gemm_2cta`` (the 2-CTA cluster programs,
# which also take the raw block-scale buffers), ``gemm_2cta_skt`` (their
# stream-K form, which adds the reduction workspace) and ``gemm_2cta_ptr``
# (the 2-CTA programs that store the output from registers, which take it as
# a plain buffer instead of a TMA descriptor).  ``_PROGRAMS`` holds
# one row per generated program, keyed by the 20-hex digest of the module
# name ``cake_nvfp4_per_token_<digest>``: the ``ARG_PLANS`` key, the
# exporter's closure digest of each architecture the program is built for
# (SHA-256 over the exported translation units and the shared headers; the
# JIT library of that architecture is named by its first 20 hex digits) and,
# where a record differs from the family default, the digests of the programs
# whose kernel and host binding translation units it builds (``kernel`` /
# ``binding``; default: its own ``<digest>_kernel.cu`` and
# ``<digest>_binding.cu``) and its extra nvcc flags (``compile_flags``;
# default: none).  ``MODULES`` expands the rows into one record per module
# name: ``sources`` (the kernel and host binding translation units under
# ``csrc/``; one source pair serves every architecture in ``arches``, the
# few lowering lines that differ between SM100 and SM103 sit under
# ``__CUDA_ARCH__`` guards), ``arches`` (the architectures whose ``KERNELS``
# table assigns the program), ``compile_flags``, ``arg_plan`` and
# ``closure_sha256``; every host binding exports the ``run`` entry.
# ``KERNELS`` maps ``"<arch>"`` to the logical kernel key -> program
# assignment the host dispatcher resolves at preparation; it is built from
# the assignments both architectures share (``_KERNELS_SHARED``) and the ones
# only one of them registers (``_KERNELS_ONLY``), each table sorted by key:
#
# * ``quant:...`` the per-token quantizer (one CTA per token row,
#   register-resident row) for one CTA width, launch-bounds occupancy, input
#   dtype and folded output scale (``cake_backend.quant_plan``); the
#   ``_k<K>`` keys of the small-row tier share one program per CTA shape whose
#   row width is the compile-line definition ``K_STATIC`` (``DEFINITIONS``),
#   and the ``_cs<C>`` keys split each token row over a cluster of ``C`` CTAs;
# * ``gemm:<orientation><tile>[_k512][_2cta][_hm][_sk<split>]_<bf16|f16>
#   [_a<hint>][_b<hint>][_l2<promo>][_g<group>][_clc][_s<stages>][_o<ctas>]``
#   the persistent block-scaled tcgen05 GEMM with the per-token alpha epilogue
#   for one tactic (``cake_backend.default_tactic`` / ``gemm_kernel_key``):
#   ``m<tile_n>`` is the 128-token (``_2cta``: 256-token cluster pair,
#   ``_hm``: 128-token half-M pair) tile with ``tile_n`` weight columns,
#   ``n<tile_n>`` the swapped orientation of ``M <= 32`` with ``tile_n``
#   tokens per tile; ``_sk`` is the cluster split-K variant of either
#   orientation.
#
# ``DEFINITIONS`` maps a logical kernel key to the compile-line definitions
# (``-D<name>=<value>``) its program is built with; keys absent from it build
# with none.  ``ARG_PLANS``, the rows, the ``KERNELS`` tables and
# ``DEFINITIONS`` are populated by the generated-program export.  The rows
# and the sources they name are exporter output hand-folded here pending
# exporter support: a row whose kernel or host binding source is another
# program's translation unit (a program that differs from it only by
# output dtype, TMA cache-hint policies, raster group, quantizer CTA shape,
# folded output scale or store policy, or only by its kernel symbol and
# host-shim namespace) names that program in ``kernel`` / ``binding`` and
# carries in ``compile_flags`` the knob values (``-DCAKE_<knob>=<value>``)
# and ``-DCAKE_PROGRAM_KERNEL`` / ``-DCAKE_PROGRAM_SHIM`` naming its own
# kernel symbol and host-shim namespace; every shared source carries
# ``#ifndef`` defaults for the program it was exported for and ``#if``
# arms for the knobs.  The row form itself -- the named argument plans,
# the sparse rows, the shared ``KERNELS`` table and the two builders -- is
# the same hand fold of the exported per-record literal.  The shared
# headers ``csrc/cake_nvfp4_per_token/cake_nvfp4_per_token_device_common.cuh``
# (device helpers) and ``cake_nvfp4_per_token_host_common.cuh`` (launch
# helpers and, included again from inside each GEMM binding's namespace,
# the TMA descriptor encoders every GEMM binding carried) are included by
# every program.
ARG_PLANS: dict[str, list[list[str]]] = {
    "quant": [
        ["buffer", "x"],
        ["buffer", "out_fp4"],
        ["buffer", "out_sf"],
        ["buffer", "per_token_scale"],
        ["buffer", "global_scale_inv"],
        ["buffer", "out_scale"],
        ["parameter", "M"],
        ["parameter", "K"],
        ["parameter", "x_row_stride"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "SFA"],
        ["tma_buffer", "SFB"],
        ["buffer", "alpha"],
        ["tma_buffer", "out"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K_tiles"],
        ["parameter", "tok_tiles"],
        ["parameter", "num_tiles"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_2cta": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "SFA"],
        ["tma_buffer", "SFB"],
        ["buffer", "SFA_RAW"],
        ["buffer", "SFB_RAW"],
        ["buffer", "alpha"],
        ["tma_buffer", "out"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K_tiles"],
        ["parameter", "tok_tiles"],
        ["parameter", "num_tiles"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_2cta_skt": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "SFA"],
        ["tma_buffer", "SFB"],
        ["buffer", "SFA_RAW"],
        ["buffer", "SFB_RAW"],
        ["buffer", "alpha"],
        ["tma_buffer", "out"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K_tiles"],
        ["parameter", "tok_tiles"],
        ["parameter", "num_tiles"],
        ["buffer", "red_ws"],
        ["buffer", "red_flags"],
        ["buffer", "red_gen"],
        ["parameter", "sk_tiles"],
        ["parameter", "sk_pairs"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_2cta_ptr": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "SFA"],
        ["tma_buffer", "SFB"],
        ["buffer", "SFA_RAW"],
        ["buffer", "SFB_RAW"],
        ["buffer", "alpha"],
        ["buffer", "out"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K_tiles"],
        ["parameter", "tok_tiles"],
        ["parameter", "num_tiles"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

# Row vocabulary: the path stem of the generated sources under ``csrc/``.
_SOURCE_PREFIX = "cake_nvfp4_per_token/cake_nvfp4_per_token_"
_PROGRAMS: dict[str, dict[str, Any]] = {
    "045ed4b8200220c8d8cd": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "5156399d3116fd4dad1a137dcbec7b22fc9b9551043fc466cdebaaadef9910ef",
            "sm_103a": "6f5f46a97d07eabfa12fd14b1a6961d022afd65e49525520d64e6db80348059b",
        },
    },
    "05727367662197581d3c": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "2d66ac73a27ab95f062447b907c166d4b3a9a8f1b96dc16006614128e7b48045",
            "sm_103a": "9cbc854271a2a747171d2980639968308718018e5e16f4e87e04ed92973a75a4",
        },
    },
    "0588831a72930146a0e8": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "77ce100a9abc61b1d14f5c698984cc1e99c4af3967a775c200dcd007879dac87",
            "sm_103a": "890b7e20915425f65acb664f581c701f2a41a84add1312a1f7cc1aa088e2c0bf",
        },
    },
    "05cdc205d9b471f666a1": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "583e63c82275f33f0355e4e1a585405c7f92488309a0ab857fbea3bf910647ac",
        },
    },
    "062311ac440fbaf37629": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=3",
            "-DCAKE_Q_WORDS=24",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_062311ac440fbaf37629",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_ae9aa916a579ea14",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "0a1725933de27f2135bbe32fe416e0a59ad5ce5b61ec28e0cede15ab0749967a",
            "sm_103a": "cad03285e892b41eb9eb324971cf5696cb17530b6e81716303e0d43c18459bfc",
        },
    },
    "07b0189c8953c02acff3": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "cb5dcafb12863e123b7bb5ce50e7dfffd6f1bdae33f5f76af5e9704eff64b18f",
            "sm_103a": "8da91139d12c43ad6a6e5de5ed6f6a15cd9543a5a1cd057dd2afad7b5ce83c5a",
        },
    },
    "0b7a9083d24d01fe3aec": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "de19e718a1ac880c46420b5e060055c9484c483ee627372e050792d7909d4203",
        },
    },
    "0d13de6f1682e55c1373": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_0d13de6f1682e55c1373",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_48801c681b24ba45",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "290e9680cb806e06b82c06cea77f801acf9a83a6cdd79126df149bffa79514e9",
            "sm_103a": "669982b77295f313419364eb2ec4993a931beae132cb206483022fbec90efae8",
        },
    },
    "0ee9b811105da3e1e757": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "97a0fb4240ac01069994665ccbe60bf45b90a48753fe397a913b94843364247a",
        },
    },
    "1005a13dfc670fd5de26": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "29be6eecb79930e958c7a2a2f91e6ee9259181f4d19d5b2f289c01f27261c925",
            "sm_103a": "ee12ca78387fde67748ea5e986a9cdc50d500db79639c3189e38e027e7a9d6ec",
        },
    },
    "102a58d983738fb4ba20": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "230e64f8a56c6ff88f0481be81e7786956138b87b250d08b8e5d9fcffbc30790",
        },
    },
    "12c633607bd8e310fc7c": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "e013544058ee2372d15a4fbd56db59890c3518c664c84610268b83de364ee6bb",
            "sm_103a": "561947a47e7a2fe80ecbf8e53e38387c4eb69d78b2fd3850f354f3521e97b59b",
        },
    },
    "132e33a6f1dd393dc000": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "09e5181ad4f5eea2e7e16d10dec88e110e8d1b02c9415b7ce04df6c299624e46",
        },
    },
    "139c3fab2a0249e02c3e": {
        "kernel": "a6fc65c9fcae34137fec",
        "binding": "a6fc65c9fcae34137fec",
        "compile_flags": [
            "-DCAKE_RASTER_GROUP=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_139c3fab2a0249e02c3e",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_5cead2195728a5a6",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "32bcd35d3959634f146efceb7c7a79c31021a30aebbb5ca65ef509dd53cd82e7",
            "sm_103a": "a6862b292f9727d30bdb66c7837461be420f4c0eed61b3b5404bd2d270e5836e",
        },
    },
    "13cec0563138e6e72423": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "a525b33c27d116955f0313f684fc6c540693d49f24a909e03afd7279d6da7396",
            "sm_103a": "ad745f524278031769c787d46e13e57d576de3684c6c9c04d9ccb79dfc33ead0",
        },
    },
    "14b9336da0e3b4fbfade": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "ba0d1e162b217cf249645bfb7801148750c742727c185ba87695ac9d616a5206",
        },
    },
    "17270a6f1bb78bfd8f92": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "886fb8fd724006b7277fdfdf1b88f78b6b71ee5b4c1ac27a87737378c371e90a",
            "sm_103a": "56c7ca9b6657546b7e70c1473ac8105ded3530701f35b2b9d43e845c6c65b795",
        },
    },
    "183e28bdf124c263ecfb": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "17deec37539b4b2cd65cd567ee1b63e466f79c86949e78db7807b42964a74e94",
        },
    },
    "18fd61a4a764c2d1ab3e": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "3c87bc92057a08e728f4e3ca035ca691c68b5b190598bb313eefe0abed1cf209",
            "sm_103a": "f59107660f0794fa5322ca7cc44ddfe904b3a3c5bea7a7ff8a2a7a3dcca2614a",
        },
    },
    "1932a5484f6bbe6d70c3": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "45b9346df17c0dbcb4e8ed861b0c507a63590d7f1d6b20e15bc7e078e399a9fa",
        },
    },
    "1957cf0620e9be66eee8": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "490e6740f512ea51994314e9826670fd8477af6c705970709bbe13c042bd08f8",
        },
    },
    "1bc4fe20d5d1fa219ea4": {
        "kernel": "288adcab31da85191a26",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_1bc4fe20d5d1fa219ea4",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_98d44e229d4048b2",
        ],
        "arg_plan": "gemm_2cta_skt",
        "closure_sha256": {
            "sm_100a": "b58720bfd2ffdba48e9b3e7d3ffc3ebafa09419c9408544f98341ce56f4f70c9",
        },
    },
    "1d6630a145ff559d1fa3": {
        "kernel": "78fcd21adb756e3fddc9",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_1d6630a145ff559d1fa3",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2348e5a05c5e6302",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "903b184290f30a9a991dc0d698b33faff29e328f5632bbb1204a742cb70b479b",
            "sm_103a": "a9e06e53fdb6c82c8c7ad0577910b80e07b01e77f56d2a108ecc657a906f21df",
        },
    },
    "1de51aa8fd0339bfc55c": {
        "kernel": "a6fc65c9fcae34137fec",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_1de51aa8fd0339bfc55c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_45d5f61eead32f1d",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "fac93ca816c8718d15439008c2b99df8d44e9eaba1ec85f033db753ebd751431",
            "sm_103a": "101b472a66502cdcafe147e0c5cf32ab1e80f71213dee3ce6d1dd3e608c90bfa",
        },
    },
    "1fa4a36589c5c72b1efd": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "fe424a59ccf6a3766b385a918010c58ab43471d149453ea80429101d4c38302a",
        },
    },
    "1fa625dd627403d62d62": {
        "kernel": "a1674da021d7fdb5e3d0",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_1fa625dd627403d62d62",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3f6f2cd66ddb6cd1",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "7814b2916ebc512941dc4566426f921159ba50981f96edfbef1aab2c37207918",
            "sm_103a": "361eb309e6b67f214b18d1eae256966bd4dff4171f64c98ff647b7bdc3014ed8",
        },
    },
    "2126ed4021a5df3e8461": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_2126ed4021a5df3e8461",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_99d8a68a104de03e",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "f0dc4bc2fb154e0c59bf5c10f36054ad2e4730d2b6210721370367b3976f38bf",
            "sm_103a": "614aac5cc43375395f4e80b6838c192c8f7048b186bd0b381edbaa39864100af",
        },
    },
    "234a3b72a787d48e951e": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "591096b739329a0e0379ce79c7e96addacf11c76848c420ec3ac9a921d740521",
            "sm_103a": "d17f5d98876caf194d13416f2fee4f24508700da8d92cd5c0e89283ff6607049",
        },
    },
    "2363f86763371089670f": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "c8df11902d160adffffc18bb7756d2f6f3e5916913d77e2ec881860af71a685a",
            "sm_103a": "e0bff08e5524e541ab5f984ad5a7ddf38f353a5dc6d4f0836de22bd9ff13d04e",
        },
    },
    "257fb8ea914211995ad6": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "5ee729805970a9e4eac437bd6eb46a55cd95124ff4454ebc35ddfeffcfcf41eb",
            "sm_103a": "1e5b10794c1a3d8ef6e2a80108829d5671fab3c916fa6c6ac3f74b72e84e2524",
        },
    },
    "266051f22af995b10bc8": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "ed689d00419d6e019123c9864622539758a58d2d7c1fdfe1d1d24ca779835046",
            "sm_103a": "65cf5ade2e3f72f67d884aec4cda3d34465d460a70c957d45b6ede4ec8ac6799",
        },
    },
    "267ea66edacba2a365e7": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "47ed3a0870c94a6899e38503bd4e814a5b3b732da50f520e25ed5672f3e50c03",
            "sm_103a": "59c68094d20ec8140b7212433031a04032efe396f155e3bcc6a7b175888eb6fe",
        },
    },
    "27d1205b4f5b3183d9de": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "b72099a2fcad859ae29e7f59a4b740125d3efe150b7995d5debde7f1f7bf42f6",
            "sm_103a": "8c53d3580366fc37a3ed62ff63a9f9d3866427d67764716cfe703abee54c8141",
        },
    },
    "288adcab31da85191a26": {
        "arg_plan": "gemm_2cta_skt",
        "closure_sha256": {
            "sm_100a": "c07fb7bfdeebcba23999e54128513adc8ca0b4debf95ffdf4b83d191e7c6ee77",
        },
    },
    "2c9d5ff42bc5ac1b60f6": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "6fca46f9f239595c7734aa2fe9450e9a6553b72700584dca469ab5f7f45b77f7",
            "sm_103a": "1fa8fbd3477541af9308cf89c293ae83b00c27791b861dd210d84000526be085",
        },
    },
    "2cfd03d9bf011a475da4": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "c638f2b95ae8ee408b56250f0444fd7eeec1cbfc702247b9c60e5de8aef47981",
            "sm_103a": "95a0559391b764128d44a4211a96999d63112dddd8bebce00723b83a45d2a3b2",
        },
    },
    "2e26ffc7ce8d8fa6f128": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_2e26ffc7ce8d8fa6f128",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_e651fcce480d6f44",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "fd12facae6267d925e668deb464743d518cee351d8ffe22853d001ead90d0490",
            "sm_103a": "26993c89cf9a8a9213021ba28e913c67229bb409485ff6adc1d83cd8440f735c",
        },
    },
    "2f8110591ec3a80e3f68": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6e812b448046096320ffd34f40c3f0d087643e06e17f65e7eea84bf2ff9709ac",
            "sm_103a": "55559ea75bb97468dd6063196ed60b573f4c132086f62332d73ee0f13218a211",
        },
    },
    "30cc10800fc253a813ab": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "24165ed32a2c7a510fdafe18c611311b6f77aa7028c1d84c3b48ba4b1c754297",
            "sm_103a": "8af47477ea2aea820113d4140dd2e8eaefb4ee61fccf8bd846b27176c9213a34",
        },
    },
    "310dce04e62dc4479f2f": {
        "kernel": "925e8165fab2f00266d9",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_310dce04e62dc4479f2f",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_bc3efea40712df4e",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6375db98b3b4b50329e9349afd1e61aac994238f78a7951da842bb65522e0a2f",
            "sm_103a": "935a1e2ea20d76b33c20488a91ac943b5eb6dd5aa09d000d5d02194e8e03dc8e",
        },
    },
    "318ddeeecf4e9addca1e": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "6970cbfc0685f9f847042cfda98731caadc2972a79071b35ed8f4abc487d6962",
            "sm_103a": "85c68987e141528ec440d74a3bd5bb1f4261915ec581554a417f96f43cd8ed8e",
        },
    },
    "326a7ad2e4cc24b95493": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "c31057b6fad0d18b50326e77e281ca097a82610745dbe97c9223c74701d08c68",
            "sm_103a": "118254aa37d03be90f57f3e96eb85e40c9bc2503d28cad91a2b67af00d205ab7",
        },
    },
    "32c8706387a82055788f": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "5f24e31ab79a7ef20fd2f780c856d08581f8125f6008ff79a511fda163495739",
        },
    },
    "33b08b7620357f6c712c": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "7d64bc50b81b54759f87995eff31af32b49e228e00e11d8a34ca4605c9f9e25b",
            "sm_103a": "4f79fca801f6410e8f3eb71aac88405ff12331c6755d49cc74f6d02912c8ff68",
        },
    },
    "34d44d8d6d4e632aea8c": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "935c7dd0944a0f93b4a185e0db6cebdb800456f1b2d25a9be3d63b74062dff6a",
        },
    },
    "34e066f626b4be154822": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_34e066f626b4be154822",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_bb0b1dffe43bdf04",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "516bdd6f4300f5ec5805ef380a6d01b2d535c46066c1086d817dd8a16cea6e01",
            "sm_103a": "ea39078648f81ed5049a0a61546cb710af7e0f7b9f033a5f91405d65fe69695e",
        },
    },
    "38827385d2d509e25ed9": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "24ee80090cd379fc16517511516b4a0921870e75ec33937a6d56e2fdb6e1eec6",
            "sm_103a": "2e610e08a4e6d5711e90be2e849097767f603beb7007cff522500391ed9d7073",
        },
    },
    "3b069158e5efd68b3d56": {
        "kernel": "dd3a532430c9f881f75d",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_3b069158e5efd68b3d56",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_90fff4b2be488c40",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "197b9a5e211428a4e59c03c013ecbf34a42df33107576abf50fd494b7750d1b5",
            "sm_103a": "e8a3325e528e9276a7e667d07882e713b86a5271ba09255ff35c7dfe577d1faf",
        },
    },
    "3bebdf16f8b8ec303083": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=3",
            "-DCAKE_Q_WORDS=24",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_3bebdf16f8b8ec303083",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_e0e26e1509b6ae48",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "5bfdb72a3229a852abae36a6c6b967aa3e8995e23ab36009e64b21e6fe24afa8",
            "sm_103a": "8d1064bc119aaae0b443c157cbd97bb6c1ab8fe9b92642d344c134cb400846cd",
        },
    },
    "3c0a0335f02050ebb60e": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_3c0a0335f02050ebb60e",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_b1963fada7f4f475",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "e12f7236880cdefc0a888bca2d64e21e2968b3e07b2bd9c0e18ef3706d7211f3",
            "sm_103a": "e6123f293b8e24f578110c783f92238fea02f1e72d11a8ff66d1f94f8d8c5bb4",
        },
    },
    "3cd58718ee867217608b": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "67f2352f97fb87a756f65dbc5a57de215e1250d4599674b675905e4e55a397e6",
        },
    },
    "3deb92b0859de2875bd4": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "89d3e825b4bb9eeaf9e58603ab14a5f164c3c8b2dc8535ace594706bd4fb2aa5",
        },
    },
    "3df66e1abbe9dc0f2230": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "b803c48d49eed98b9e92d6c24ce136ab93c499a1f032d2a0f09a00c74d86ed91",
            "sm_103a": "1f7e4bd154f9c88569784d7e15c69d1fd3bda5ebb6b19affbda4cfdba14f24f7",
        },
    },
    "3eae14c9d3c4c24cc180": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "b3014000bd72842c5e522926e3254eed5a5b0b741babdd07e6b6dbf621f080a0",
            "sm_103a": "5411391eff3284eb3e9a8d0db3a06993f8d3ee17c212a9e4ac6edf265262878d",
        },
    },
    "3f4db4aa72227118688f": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_103a": "0a8338782374d2845485deddbd2064e8f2a9b1d1b5455504db34e1cef94b9e6e",
        },
    },
    "4083e913d0b68dac7255": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "24c3139d0d4038d53dd266706bf6846673273a621249bbbf9259b5cdd6043423",
        },
    },
    "455c60a5c10aa2bcece8": {
        "kernel": "18fd61a4a764c2d1ab3e",
        "binding": "18fd61a4a764c2d1ab3e",
        "compile_flags": [
            "-DCAKE_RASTER_GROUP=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_455c60a5c10aa2bcece8",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_ffaf036d5f85c138",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "aba71acc8c139472c87d5e5fa4b78cd54ffee1f95dea3188415fc832f3a6a10d",
            "sm_103a": "1e830a30ab671cef7c354eb61e9b581d6909c219c8f76aab0bda941cdb1a7c82",
        },
    },
    "45ce0d9fdd8d0936a613": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "05d94c26f337e2e03f271a477f477b02eb3452ca8601524f52937ee49faf87a2",
            "sm_103a": "c5a56adee4c66a10b7febc9ea2f8b1992e46998b76ca04ef65574eddd4ef3e22",
        },
    },
    "45fc8d20d916671c5f36": {
        "kernel": "18fd61a4a764c2d1ab3e",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_RASTER_GROUP=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_45fc8d20d916671c5f36",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_ba4c49ed42f27127",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "2c22ad093e706c9acc0b445d25b682d60696d5be26508ae3998b890f3eb1c4a7",
            "sm_103a": "a1ab25508157f1298b7b9165a44247f8aa5323db2189dcabb2620a4a2c70cbc5",
        },
    },
    "48485b687a58cef62559": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_BPT=3",
            "-DCAKE_Q_WORDS=24",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_48485b687a58cef62559",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2d2fcafaed6cfa25",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "cc7cee2f28a19dbf6d7e212c5a14e05c585ce6d773daef0f8c63a2de6e383407",
            "sm_103a": "2af00116cb16f128e22931939bc4134c4aee048b9496f05a5328e8dc3e01ed9f",
        },
    },
    "4a4ae65a3551bedbc110": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_BPT=3",
            "-DCAKE_Q_WORDS=24",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_4a4ae65a3551bedbc110",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a3f40889094d4f03",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "855a935cfa22e2d02ccccee7c4ed2628d48ba157a73cac234a68f9561b2d63d0",
            "sm_103a": "d64799f52e1a375fce97585b4480e0bb5da4bc86244f3780b87a0c09ed6fc612",
        },
    },
    "4a8ea29b299cdfc83a99": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "0bbde0219a4aca1f440b346baca3d3324b0a9f0c0d31fce42a58bc128a8c3327",
            "sm_103a": "311ae82ea0526283333ca7c072ec4e60f1b04bf3a1fb70b5e15954c5c886e29d",
        },
    },
    "4d85976d6c18ce148ca4": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_4d85976d6c18ce148ca4",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a1a905c2e5f86ddb",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "e380f239300cd75e9832f5057cf9c2d3e18a9995b56c65be6ff2a66d6d92fe60",
            "sm_103a": "1cc92ae3db4e8eabd1efe6b35f1935cbf858a0b8f001426d7b1bcc878b13b9e0",
        },
    },
    "4dfd2b6475da344471f9": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "86f8518cc7b0575389c9dd96842ddb835b53bfb8ffc1580e4caf37bc7babbcc7",
            "sm_103a": "41f221a1db95f1d117dc496c5e74ffdf6cc47deb204f9a8e4571e246e451fe10",
        },
    },
    "4f250853fb07cca4f4f2": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "3d56014bac182d1ef36e0aa4bfff8c513f3aae24b904fe346c33665b904e0e7a",
            "sm_103a": "b27166b58146bb5a8606f0880455be2e20da47588e1d87de7134334d0222caf3",
        },
    },
    "4f4d6711fee3151ee9ac": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "7539c589b765df09f99aed86bb99111ef4a37da5f9c94bf04bacfd3497776541",
            "sm_103a": "c715a0ce3e5d0ee73b5647a5433de508ebf36028155e97079f0be0ce606b3ee5",
        },
    },
    "4fad98101b1166c3ac8f": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "ea0dc6e6a45cb44c28ca7b8f7e89d2a3d60ee25702bcf10b2f14d400d35be189",
        },
    },
    "502fc8dafecb312370d7": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "45b4b7f91e050f49d5f15bb81c500a8f0ad58be4a0d8a8b2d42aa545e5efc59e",
        },
    },
    "52f7cc1603bc0a530734": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "32dbd7e4479114782a938c7cc211be0dcef0008d2a354dbf4a7460e785a2c6b9",
            "sm_103a": "54401a6a0a4f03d160cb85e5f06c3be325b7c61cb7105fbb59b5abdc52babffb",
        },
    },
    "53e1454ac0d533cefcda": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_53e1454ac0d533cefcda",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_fd09e1b86969caa1",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "6e0c4f1b4924d33b593a486bb88d385dbb2dfd0d430d7f7fb14403ca90070574",
            "sm_103a": "d1b101b2807486eaa394074f2b77f9061dbc81395872ed046629aef79687c761",
        },
    },
    "55dc9450f28f61f389fa": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "ead3038a9cc11c4ed5b3a68b807fedad4f929c2adf7881b065465eb3f6ff21ca",
            "sm_103a": "6ff825bbea7a1cddbcdcfd5c7c654a26ce3ab69852a0e6174e0e8e329b3a2841",
        },
    },
    "5600b6d759d61fbe3ae0": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "ca4a2248501cdf4de907a478781bfbc3d3d91e090a2b9d55bef02e1c1228a697",
            "sm_103a": "938ca0fe00bdfe38f28502430b7aef355ad20d1d44d99f1497d40f1b9dfbf3d0",
        },
    },
    "5606b2d9571f0787e563": {
        "kernel": "9ca737b8f50f70eb4b5d",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_5606b2d9571f0787e563",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_dcd764c5d858dcea",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "84b92983b41856e1ae56fefea252001090a8359f35d8a95de17a8b359850f08f",
            "sm_103a": "6bece1c4be39820596d8d05fd96f82862de91894c701ed006cd0359a52a9c682",
        },
    },
    "57acea79765928447145": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "f734dede83502f1908d79b2e5c636f3da1da996cfd654efa7db25f75432d2530",
            "sm_103a": "78eef6791414207dea6a35f0b0ef6a0365c4393bdd860f4ecb793924c27f4a8f",
        },
    },
    "580e9edbb66ac9359718": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "0817202b6068d9fbe5ab4ad43e3bd47b5f467042b5af7c27f11fef896f1adb7c",
            "sm_103a": "b1e615c6b78275a294e46cff41aeeb76ba69e74cc92c7c2476b19f64cd8e4dfc",
        },
    },
    "5872000a0bd8bd391059": {
        "kernel": "288adcab31da85191a26",
        "binding": "288adcab31da85191a26",
        "compile_flags": [
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_5872000a0bd8bd391059",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_80990f8b9094a706",
        ],
        "arg_plan": "gemm_2cta_skt",
        "closure_sha256": {
            "sm_100a": "d533d39deb253b8f7caa3864839029a1a3c627b374077d1118ae87e14ca67aa2",
        },
    },
    "58eeece815f48fd1d43a": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_58eeece815f48fd1d43a",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_611e7697a0ce40bf",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "b0d3bade0b86c712ea527e2c3c218f421bbabd0b583a44ec6113220d8eea2742",
            "sm_103a": "790457f84372490c4d21399cedc5db8e85fa553c2bef5b5488e360caa456e8cd",
        },
    },
    "59777655982de0ec51e2": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "ba3d99d99d632bd58ef38dff122d39ad78b1a196342fa78d8701e56011e89cf6",
            "sm_103a": "3b553206784cccf4e92cf66b45c93ffa2ba6bd7068317af8a77a625940d37541",
        },
    },
    "5d8bea1558154053d314": {
        "kernel": "18fd61a4a764c2d1ab3e",
        "binding": "18fd61a4a764c2d1ab3e",
        "compile_flags": [
            "-DCAKE_RASTER_GROUP=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_5d8bea1558154053d314",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_871475b1bda7a9a5",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "6c7bee81cc7d06593f6c9076c30331b84f3faf5799da958eebb0654d2421eec8",
            "sm_103a": "cdf067254284d95839de11c3bf1a88cbc3f4c93b313f800e5b4a038a93b66e3a",
        },
    },
    "60a084e22436164149fc": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "cd18569048a540bce77229c8635e6ea65a22ef1d3f0457eecdf8e0633614459c",
        },
    },
    "640c33b699a8478f60bb": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "ffb94cd40727ed5410f3f460c6d0afb416134280ac804c4f504eda06608afe75",
            "sm_103a": "4ce8202ce4f80af3955aefa7ac62d0ad00d1ac01836705bcadadb457ac5c0a88",
        },
    },
    "64313df7982791417a2b": {
        "kernel": "18fd61a4a764c2d1ab3e",
        "binding": "45fc8d20d916671c5f36",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_64313df7982791417a2b",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_f4c260890861e49b",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "428afa6bdccb2bc332041d7c7b6c3fda17719cfd99393aec72a042e94f8ba3e4",
            "sm_103a": "129fce37a61476039053aae58f9e86c8d097eabe83d6fe5c3cf3c7a49baa5e13",
        },
    },
    "6659f7757bcb7bae97b1": {
        "kernel": "a1674da021d7fdb5e3d0",
        "binding": "1fa625dd627403d62d62",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_6659f7757bcb7bae97b1",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3011f0106a0d7e88",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "6335d32acaf90e04ae5ed7c945670b5f279a65bb7f091a08d9e5aa9fb5c468c5",
        },
    },
    "66c17b2c6c8645cc5a76": {
        "kernel": "becd24cbb8333dc48dcb",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_66c17b2c6c8645cc5a76",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_8888a7aa83344549",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "76091cf38bf43c5df615a7f4120573d807c55847b9ca4d9f72af4f9829477831",
            "sm_103a": "1fa53371cc7b61e8083ba74bd766ce20fcc44127d2979f712c656b9b7ff161ad",
        },
    },
    "6717875e66a6fcec8341": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_BPT=7",
            "-DCAKE_Q_WORDS=56",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_6717875e66a6fcec8341",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_ac469cd799b76098",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "1aa0541afcca4d780841cb9f05d39c7bab4b54473b86063f988f8b9b6048323f",
            "sm_103a": "605472a09514e37ad46d72888e25752d0337d65066f0a44f3e9dc0419df08068",
        },
    },
    "6791cd4bf5cc9b5e088b": {
        "kernel": "17270a6f1bb78bfd8f92",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_6791cd4bf5cc9b5e088b",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_b51fdc97c4df0b62",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "3742baaad32dbfdafbe359b3a78cb1015be7d6c7e29a00b2432589ad09a1e641",
            "sm_103a": "62350c3ee5aab2cb4e6d502ea2561ddbfef8f39eb58cdc9650dd136a3c7d1192",
        },
    },
    "684d5b4c9c173753e179": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_684d5b4c9c173753e179",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_d8935a6c9e0fcf06",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "1525672c90a4ca07ff0c1d6df66bc25d5a00e17c6f55c366f257c2516fb275f1",
            "sm_103a": "8c32008068b7d92f3f927c7121b597dd0a6a0331de5c292ff8e6284f9fad227c",
        },
    },
    "6864d70ab56de3ba8bfd": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_Q_BPT=5",
            "-DCAKE_Q_WORDS=40",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_6864d70ab56de3ba8bfd",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_1bbbe57555a27f52",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "92e0820818e84bd6f7261afb1cf1d9fc4e7d2cb169136b1d20fa637558ebaf5e",
            "sm_103a": "468fca29217f4e4cfe4963543ec9bda4ca9791a2d2a656c883dae94e7f76e12e",
        },
    },
    "68ee524b89f5e289133c": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=128",
            "-DCAKE_Q_BLOCK=128u",
            "-DCAKE_Q_MIN_BLOCKS=5",
            "-DCAKE_Q_BPT=8",
            "-DCAKE_Q_WORDS=64",
            "-DCAKE_Q_WARPS=4",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_68ee524b89f5e289133c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_f8cee2deea678a22",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "b6635bc8fde147fd39c7a2d46ded339fa8470c94b2911aecae7d610e692f741d",
            "sm_103a": "b75415f251543d16299e566bf3323b013835d19d2f596c4c6603066250e926c6",
        },
    },
    "69f8274b12662366ecb8": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_69f8274b12662366ecb8",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_490dd6ddc757b332",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "9b05312c6580f405a872dbdb70bf0fd8be38b587615a45670a8649ff2168ca58",
            "sm_103a": "853ec0e5a148cca11c29ddb439862a711c1302d7e02b9af770f6136a8423255d",
        },
    },
    "6a8f0bd74f7f517d1b11": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "97f92d4a67e4af06b75a05ee6f01a0817b787bc87a338c9d8203be9a02c4e83d",
            "sm_103a": "9bc0e7c33db47ca69d05af0c28b97da012affb47457a622763d3aa479ad541a4",
        },
    },
    "6c716dab52bbcbf07637": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "6a64b591c02c8a971288e0ec052920951332a8f3c106981725fedf8adb57dabd",
            "sm_103a": "071d313ffa28826f59fce81d4091804888be23e93eae4ca9d6a391f4047c0b64",
        },
    },
    "6d96bb93d7840b5b9b42": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "a27bcfeca0a8b0a14782908f4847344a74932177229516c606a38a268e1b23ec",
        },
    },
    "6ed9e5d8627577028e1d": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "41e428dfde61eb0c332923c252a76607851a6f3753b290db10ddf952e93b7799",
            "sm_103a": "25d390df68fbf41b7a5a7ba7da7f84c4d28923aba9c00ed0d33e170b9ab21282",
        },
    },
    "6fed99e520aa239db6c9": {
        "kernel": "a1674da021d7fdb5e3d0",
        "binding": "a1674da021d7fdb5e3d0",
        "compile_flags": [
            "-DCAKE_RASTER_GROUP=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_6fed99e520aa239db6c9",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_f9d5f5b51f57de9c",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "011c854b96f13940d0f05f0698f7af774f7e085163be7e72f37904f0b2aa5d2e",
            "sm_103a": "1911c55fb452eb176655030fae00377a67d0c765f2f68edc7d4bec69a2e762da",
        },
    },
    "7143db00594241971565": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "eca40fa35e0fef136602c3cac0a7923590c9fc331d495df290728222fc5526c5",
        },
    },
    "72a49b93eb629736017a": {
        "kernel": "a6fc65c9fcae34137fec",
        "binding": "1de51aa8fd0339bfc55c",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_RASTER_GROUP=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_72a49b93eb629736017a",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_017fda2ef28e92d7",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "71db072b8c90927aef6697d09bd37a73de8ab7a4de6e4badd73f151b7431ca39",
            "sm_103a": "30c5ae2537916ddac06a38424bd884bac7e0f0d551152c05f9213a63d65cfbb6",
        },
    },
    "74bf657c544a53c216c6": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "48848b4072029cc4aa7cbe0fb0997106483adcd4e0596f3d0a4deb03757f7042",
            "sm_103a": "e23ade247c8e2dfab7ed732147f23f0d4db2e650cc33183ac7bccc237308f47f",
        },
    },
    "75d30269a025a195653c": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_Q_BPT=5",
            "-DCAKE_Q_WORDS=40",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_75d30269a025a195653c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_45c198a2409476f6",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "98d6a76d3766edf1d16143327c793d2fb9233a55590303f19dcefc881f6af310",
            "sm_103a": "718bd7d795d33cd2231d1f31a03cb279edc8a54ca108dcde4c0f7d6e991aab76",
        },
    },
    "7652be76af016751909c": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_7652be76af016751909c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_fb4acc149191df75",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "72e08b56819e3be08c4638c1588a716c3eea9a440861d90fe5ebda01d6706581",
            "sm_103a": "8423c0688ac8394dd2a92a5c630642685408483de71de573d8a3749eb2bd6b85",
        },
    },
    "76b9d6595f0d3be566ce": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "efd79a599037c4563cbbc91988e630354af020975a5b76e09c5f942b710d5278",
        },
    },
    "772a0fce61655a1df842": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_103a": "c6241e699adacd9bba77c4dc4000aa907581f0014ebc4152315011fa731910ff",
        },
    },
    "78452434e3ec2d8f5e13": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_78452434e3ec2d8f5e13",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_f920f4a18b27d2be",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "477ffd5a9f8c6eab42612acad7af263633489def9125c1bcf5bc15fecf4b0ae4",
            "sm_103a": "81bee7d67bd83182ae8d8f383a885030d43479a79766661b140edb3cb1d5a19c",
        },
    },
    "784b1664a803959166c5": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "ae4812cbca4c535904963010349fd6c0f5827b94e06a4f75821b7c5d4349b04c",
        },
    },
    "78fcd21adb756e3fddc9": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "02456e1d27453f4eca1ee07cddc44232ab306c7926f5c8bdced91293af07f3b9",
            "sm_103a": "14db454ffe1e870c0f09072922a62dfec5645ccf7c6d98b15a638e1345b34bbe",
        },
    },
    "795f9ce57ce7e6812b6c": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "5899171efb790f2ea2dec878db3de7558cebe2da5e61656cfae7374466a9370b",
            "sm_103a": "aced366251467c1b378cf931bf3dc4e0813303e52c0ae03537f10ef936ddc30b",
        },
    },
    "797cb820f08d9d66a3dd": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "1c35f5a70c32d078c16c228e13724590c48ef4904be429d44314abe4ebf34427",
            "sm_103a": "f229d0dbf8585c8c6931a5e265264359e62a08b02730057b4e2c9b648e212519",
        },
    },
    "7a07511bc4c0c0f06089": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_7a07511bc4c0c0f06089",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_e6ff2547bdfa33c6",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "4adfd1c086382725d930b605076d6580027a92e7d899dc480711b57281fc4fde",
            "sm_103a": "417e322d1be84531670c455755fc7cb296a332423763b6f630777c6a7df334ed",
        },
    },
    "7a7786a40f1adf944110": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "d3e44425734091052a9a061a52d326bb8bb088d23bf62884148c68e24ac23bcb",
        },
    },
    "7abdeaf1ad1dc7c64ef4": {
        "kernel": "dd3a532430c9f881f75d",
        "binding": "dd3a532430c9f881f75d",
        "compile_flags": [
            "-DCAKE_RASTER_GROUP=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_7abdeaf1ad1dc7c64ef4",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_cca0178f2311dd45",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "4c6b7c3c4ed6efaae336062c4897f4694781b445932003f920a24e715121c7d2",
            "sm_103a": "9c52b0189173d91bb0c7d0d1c6b76bc9a0f72d5d25e80d9a2234b8bc2385ba55",
        },
    },
    "7ca723b13756dd926bfd": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "fac55129a00a57f59e8e3cb32693e9153148e54b9457b2ac7d8e35e0b7ccbbec",
            "sm_103a": "701903f20eeff87d45e0bf64547704f91b638f7e3827c1307a6acf301e76a1ed",
        },
    },
    "7d87cdf4f15f1439b3e0": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "8e61da7d4dfaa4d59176abbc6716ca485d97f70dd72db0e6fe6ae3a009348fb3",
        },
    },
    "7e68f46d2c17ead412fc": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "6ee07a908b3cdf352d8927f635cb7ecac7e3a1a6e0573164878a566b8e0fb8c1",
        },
    },
    "800dc1b69b288d010f61": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "511f742ab4fe4985248c41666b282c357b67effdd20f593ce5a5f4954847e501",
            "sm_103a": "7688ca3fa1a7c0beac63202691229708d6bcbb05aa5ec6498a197946f18b9284",
        },
    },
    "83fec50b7af0311f8244": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_BPT=7",
            "-DCAKE_Q_WORDS=56",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_83fec50b7af0311f8244",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_9e76cbd15f10cadd",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "a6b7044b7c320c984601a18180a1dd4494c028f15a833c63dc4ac0a1203eb612",
            "sm_103a": "91fae3998c862af1c83bc2806e99fdf1e19228d48224041de1947ada0acc8b5e",
        },
    },
    "8499e042c80c6cf6a00f": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=128",
            "-DCAKE_Q_BLOCK=128u",
            "-DCAKE_Q_MIN_BLOCKS=8",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_Q_WARPS=4",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_8499e042c80c6cf6a00f",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_08b9c3e927c3d209",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "27771219f2edc8b2a939e22ce095e346f2da0d6b16450eba2fba4fa01e7bd733",
            "sm_103a": "8b20aeeb254f14c8eea60d2c3436f1fdc2b071b6b655e2276bd64765c1d4e160",
        },
    },
    "8befa090dcb3f0e7a7d9": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "bffc80c07f9c3a5d0fa778cdfcf5de9b62367b6c41be21366851149731e1d647",
            "sm_103a": "318ac9f3d10a16957892d4b3e3c0ac413e191c22ac0b162dacda9b92c36eb681",
        },
    },
    "8c27d1a9b8f95d84ea63": {
        "kernel": "a6fc65c9fcae34137fec",
        "binding": "a6fc65c9fcae34137fec",
        "compile_flags": [
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_8c27d1a9b8f95d84ea63",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_d7cc7ee6952e1c5d",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "2687b2e0ba57caf4b5979a927fd4b62ae937715b28e0fe61f925fd49c014dee1",
            "sm_103a": "1eed68707b90a474ff82307d8fdd5cf92dbd5082e368faa74ce49bc1cf23c4dd",
        },
    },
    "8c70d72aaecb580a5783": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_8c70d72aaecb580a5783",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_0d43859601e4c551",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "6b3081e283dcd35705f897c41cbfab16c52f54898996db45dbc78f0d0be27d8c",
            "sm_103a": "a034ae107fb29c0ab5d158a7d7fb0e513960a5587274479fff22d330b6301878",
        },
    },
    "8cf7856e23c713904d31": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_8cf7856e23c713904d31",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2643c9b8ea78d678",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "f7a1a296f3e9e5331d2d9c2aeb06f89bbab2c2a66f83b6345dcaa69fd87aeba7",
            "sm_103a": "061bd2f80d5feb3809afaf9740fa36b2873265a0857069397f348845e681768b",
        },
    },
    "9069465975d6f1ca6311": {
        "kernel": "a6fc65c9fcae34137fec",
        "binding": "1de51aa8fd0339bfc55c",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_9069465975d6f1ca6311",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2de55b625e7f53eb",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "0fbe8c83b0497611227b931e2dedfb60bc1d8d8e4839d02c2811938ff177efc2",
            "sm_103a": "61fd984ec6a1132a036f88f660ad1119520e9082bbd817eec80838dbebcc1598",
        },
    },
    "90858258ec070501f7f1": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "c9dcaa3b2ce10cc8744fd9fd68ab20aed742c2c01a1f67f5b6ebffe41f108dcd",
            "sm_103a": "9e8b37a4e06b8374e4bb667d05b85614fa0934279e750460d41967a416c763a5",
        },
    },
    "90f233fd2ad0ca2fdd38": {
        "kernel": "a1674da021d7fdb5e3d0",
        "binding": "a1674da021d7fdb5e3d0",
        "compile_flags": [
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_90f233fd2ad0ca2fdd38",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_2497d0446f18cff6",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "cb3d8656850d3d17bf23b1187b87f421e91e8ae78a2ab35d698e40ff6840fe12",
        },
    },
    "91cba126dc2e7278c729": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "58e0723fa4dc019187908d47b7faaa20d24dbe3f3fc346dd967fc459823662be",
            "sm_103a": "56db6ace2b84e07297229b29cf368569e06bfc7dfb95090e5ae47eb99a8b18d2",
        },
    },
    "925e8165fab2f00266d9": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "a44589c97e2420293bfae94d5adde3f7a1ee987ec4140464aab90ddb798b893d",
            "sm_103a": "d626bb279215eb6c5f46e10f8702b2186efc8c7935f9b790c3cc2c23a1f689e0",
        },
    },
    "93b60320bbfa3449a921": {
        "kernel": "07b0189c8953c02acff3",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_A_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_93b60320bbfa3449a921",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_fd9a2ba20cd553b7",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "1f9019b823765df2aa0a939ca0e0376c8be65d37bbb20dabadbf7ccdccaaba55",
            "sm_103a": "aa26170bdedbe558823ec9b654ea69e669a13c7bbd0cfbd05c6ffcbeeaf8c8c5",
        },
    },
    "942b39aa9ca66fb980e8": {
        "kernel": "07b0189c8953c02acff3",
        "binding": "93b60320bbfa3449a921",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_942b39aa9ca66fb980e8",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_8cf67c1b20d9eaf2",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "f7b7e02bc47f7258542f2decb7f326a2be4371ac5b96dc7c868e5e763dd04770",
            "sm_103a": "a5f3551c138fc7157fffe8d53f3a248dfd5fad03e7047952a52f23237a125bab",
        },
    },
    "94e5e5aea8a1a29c27ec": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "66c6f1adb9942f03f98d4ca1a3ef604e21af33204ab94ace2db72ab8e688cffe",
        },
    },
    "95369b42a903d39fca2c": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "e4e48a5247cb77240c7e8f086dc8538d2ed3685a8baf9fd3edb521f13c84b7ed",
        },
    },
    "95a134a511d7eeb8a6b8": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "904928f860a2e7496f45ba715948b1e3253406c2e6c68431b67f5c410e8c5988",
        },
    },
    "96e142849dcfaaf7c1a4": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_96e142849dcfaaf7c1a4",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_c61cd0333af6286c",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "f84c65d3debe76aca964127e6f738836b427f65ed65039f97b88fdc9bc9b4944",
            "sm_103a": "be8646a78e70ac2b6fb0ff5878c2a5fe1b4deccd67abfe32aea75292630ec409",
        },
    },
    "98778736f957e3baf9cd": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_98778736f957e3baf9cd",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_b3eb77835d0e86b3",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "32becdbafb9552da0e957e97aec5ff6a50432d3824ae7505e86b50470c51133e",
            "sm_103a": "8851315b1e1f24cd81026327a8b7c2862913abff5ea5055fc3a36496b0f3eb72",
        },
    },
    "98c1d7945ad0dcb98963": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=128",
            "-DCAKE_Q_BLOCK=128u",
            "-DCAKE_Q_MIN_BLOCKS=8",
            "-DCAKE_Q_WARPS=4",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_98c1d7945ad0dcb98963",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_b2713e87d0d1525d",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "4e7d97f9631466bb0f4d755851f98ae091ccb018a6f386dc09044d1873f39593",
            "sm_103a": "cc9485eb4aef5dd8153bc986b87348e08f5cb5f3517c005a7fb1fecd2fe0d237",
        },
    },
    "99374a4603d9e9ec58a2": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "185867964413e5d02aa764d434e28317c718355403afac40adce7149cb1d37bb",
            "sm_103a": "5ab9a36c6e243f3edfd0c46dbbced1f14e105c800ac55602ccf931bfe4795097",
        },
    },
    "996c7fd21642f288f697": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=128",
            "-DCAKE_Q_BLOCK=128u",
            "-DCAKE_Q_MIN_BLOCKS=5",
            "-DCAKE_Q_BPT=8",
            "-DCAKE_Q_WORDS=64",
            "-DCAKE_Q_WARPS=4",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_996c7fd21642f288f697",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3138970ca2bfcfab",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "f283c525472f314f1ee2e57823fbe63f2a7bd36c6ef6db3782e345e8d91bd879",
            "sm_103a": "dcf25ae25ce75b7572bf067b18cb9b2986dda14fe1e3a8d33c31c5eb3c8b59aa",
        },
    },
    "9ca737b8f50f70eb4b5d": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "0cd9d5856ad923a8b3a8e299c2fc081ccc51649a8e45147635fdadf538c7f913",
            "sm_103a": "7be8344dfe56d89470d8d67f84f01add78fcdb7f8cf115182c8d3381ad83e666",
        },
    },
    "9db9fb4fa9cbe2095158": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "ccacc9634e547137a01569ee81146540dbef854164d19dd0437d016a1022f44e",
        },
    },
    "a0ba477a1d576d523533": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "d3d396e8eb812838e8de088ae57c9d5f5257b23106a45145b35b9bc0f4bf8cd5",
        },
    },
    "a1674da021d7fdb5e3d0": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "9bf3c50cf90ae566146d89ed16bd496c18e5171ccea5e731cf87a17f5b142df3",
            "sm_103a": "c5e5efbd1bb2c6e6615e6218192eb036c27bbdd928e78a78ff36275f5baa6bf4",
        },
    },
    "a1b33fc9ef3db1732eb7": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "568a5ebcf9bc048e6e02331c300a17c12004ed2a818db3dc0d235953e6a64721",
            "sm_103a": "e535fedc962bd2edc9d903575d313b08e3e1c2de118a13e036a0348ea4a40985",
        },
    },
    "a2b26a244ffb71a5141a": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "26f67ff01d0eb7abe37e5fae4d0413149b7ba15834664408e0ecbfc25eb45c1f",
        },
    },
    "a3b720172d1a7538d508": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "ba595ecce420b50635b3fa6e064538696a2d65b02720f872b4ff2837d36240eb",
            "sm_103a": "4c4979cf8c9e92bb97662c50d7b03d6759b1ed9ec025f4019de98e5587fd0bed",
        },
    },
    "a4857feefa4431fea347": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "b6b96c618710049845dd77d0878731fc71eda2dc9c33239c8113a1da9a3e1753",
        },
    },
    "a4c4b7bb3ae7574dd507": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "ec9094ba2c354d934710fb601f19e02a521cb8a6837429b57ca028fcec9ea45d",
        },
    },
    "a6cf54d2fb100c00ad63": {
        "kernel": "18fd61a4a764c2d1ab3e",
        "binding": "45fc8d20d916671c5f36",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_RASTER_GROUP=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_a6cf54d2fb100c00ad63",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_278ad15d72dab5d0",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "4522e2fb4aab1d40dd5178e5310be92fc542bb5f14777c27237ac63bd840acf3",
            "sm_103a": "1050a60b7f6301a6b9fae296889f15e4177d618c9dfa164d640d5079521b2658",
        },
    },
    "a6fc65c9fcae34137fec": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "f66c5868adec926d36751ffcd741813ab631f64cbd22b304c02dd25ba2d0aa28",
            "sm_103a": "fc932b547fa68ada83869e0b0314a78a8029f68f638d40adddcedcb5ef578301",
        },
    },
    "aa069aa70914114fa1c5": {
        "kernel": "288adcab31da85191a26",
        "binding": "1bc4fe20d5d1fa219ea4",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_aa069aa70914114fa1c5",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_620082c0371a42cf",
        ],
        "arg_plan": "gemm_2cta_skt",
        "closure_sha256": {
            "sm_100a": "478175d46b5c7b15d717d97466c3cdea407295173656d21d41f80e2d9c226300",
        },
    },
    "aa08d05b7a61e9ea3498": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_aa08d05b7a61e9ea3498",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_4d762063ff52037d",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "84a4f847d9bce1500a2fd0f4932a5fee6358e1a98fb111f2d22be275b5508441",
            "sm_103a": "66b6387b565a5803531e61264dbea76481868fb3f79bdfa0d1d056fdfe06ccb1",
        },
    },
    "aa2933408365d2cd4269": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_aa2933408365d2cd4269",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_01e8db3e023f58ac",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "0fcb9bc4532f33622201125ec3b03e964270328c0a4709c120a799b4610bba66",
            "sm_103a": "8ababe93d7047770df40feae1635247942082270bafd6773e691366724afedf7",
        },
    },
    "ab2e761d5b22bebc40e7": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_ab2e761d5b22bebc40e7",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a9d83e8f8cb8e13f",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "3caa7c3ec69132d2d715ee8847d7f8905ba267b2c5d319b35dfdc3857fe99fdc",
            "sm_103a": "d84a57f19077ac51751077e17feac928be4c3026db56796db7eee85f6a7cc554",
        },
    },
    "ad54bb8bc23776e32f04": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "7ee177212c0ba29fd90d747b45ac6aa3c8586fc93059b0e3ceba792a3dc30058",
            "sm_103a": "04c796d5e34493e184bce5aeb39c5ae6890da2962735e40eea5e90ff7bca1a55",
        },
    },
    "adf05dbe9f624ff487d0": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "61580c90477bc6e3d57f625ed97313d2f528d2782800cd87ca2149d06b4a726f",
            "sm_103a": "2b7cf314026180ae8270fa182a34227e52e0f489fb63e9504db7cf0dbd1fdecb",
        },
    },
    "ae447cd8c8a488cbe10c": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "d5b0896f45b2a35983e7cd5aefed236d550a68bb6d2e693b39581bbcbd2b8cda",
        },
    },
    "afb177dad29f9232b073": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "e4084067b2f7928954f50a292b875c9a96168048930f4c63da1b5de756e7ab6e",
        },
    },
    "b0b95c4118991384623d": {
        "kernel": "12c633607bd8e310fc7c",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_b0b95c4118991384623d",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_8c2d9e568c0b8969",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "a834848c1b5040c60d27bfacd337b30ceef5ca36af27cc83de56147f35289690",
            "sm_103a": "cc5f06d8158bff008518c62ce7ce8de94af3248f37ede1fcc1a56228e79428b7",
        },
    },
    "b4effb8c3185b65b4d84": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "ad98d48a32c0e1f87b2968a9dc090059435b681c177ca5f53bfb37d58f9a075e",
            "sm_103a": "528849cb6139a9ccb2948865a4fcd5972b211636b3c3ea485421ab52433978be",
        },
    },
    "b649171f6e9e74551522": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "182126921fed0a4875f78db9f89f8b9cb6bd6f800862e21668cea562e7216187",
            "sm_103a": "25629ffad21b250002155c021b4252f6885a1468b431645e58e6413988aa472e",
        },
    },
    "b6793e01c3abcc81980c": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "72a7942b482624d8c46db70c93c9160fad2470ee15d1de4290cf53ab9a3ae887",
            "sm_103a": "68ffa0e71ee218c4dc4d30cb4d3116dba18ff5c63ce42eb89911d500f6d639e6",
        },
    },
    "b6b2ef866ed5f8f91335": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=128",
            "-DCAKE_Q_BLOCK=128u",
            "-DCAKE_Q_MIN_BLOCKS=8",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_Q_WARPS=4",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_b6b2ef866ed5f8f91335",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3d58e71dc87f6070",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "07d4943d5fe97ec96931c336bc763e204207ffd76e9a22929bb71162b961831d",
            "sm_103a": "4a4ef73cc8c5aa746d51b8aa4850327321bbf748a3b1955ea8d4a0cd3d4164f5",
        },
    },
    "b8a8b7c06a7533a489cd": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "c3e5e111fa9f4c5d4396db2a0f5e1a879e96661f5ed78260c3521fd3af8c6ea7",
            "sm_103a": "fe6a453e79ca9e668ac740f148de07ef695ca39ee5257f834a50f9b7ca550480",
        },
    },
    "becd24cbb8333dc48dcb": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "efba8b56aef887daafce9d35c5291413cbcac732e802d6090d394a8a0846d4df",
            "sm_103a": "06b41ce47a4dbb8f27eb59ce912591b6d87d1130814393016d05b4004fa650de",
        },
    },
    "bfa0c3e4e60a86104b16": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "bef7c44d3779f129ec30daebde1fc05f215d427bfde80ee78b93a63f188602fd",
            "sm_103a": "9b03fa6a68b9ee207721a6ff1d8c2235ea627e095e5c763fdcb7dfd1c493beed",
        },
    },
    "c002f42b63a48a7bd2b9": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "f901759873cfd520a54dd70d4861daadde4dd3c3b1f678c822ec7fbd5dfcaedd",
            "sm_103a": "ba712a54d52900d679a65f1b395aafaa859cb560bc6c62036a4b0166411bb888",
        },
    },
    "c048f1b68848e4fc102d": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "065be11851a1b9010a77f5bd7738acddf4830874e7ee32218b4860904f5f01dd",
            "sm_103a": "396c04ce6f71cd6746713b74ae4ee8789237afbf5fc848e5df937104cd8c79e6",
        },
    },
    "c253402ceb2f05cb709c": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_c253402ceb2f05cb709c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_8b64ece63c7cf6b9",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "2d0fa36a3a9b35880b0e4f1a706a4d5b6bd6d34ea2cf09dd6868904321f3c657",
            "sm_103a": "9009dd73267a04e1f4f6f60d29a391365e88d7e07b6d1b3cb0ba0ba4ff193984",
        },
    },
    "c5d78751c560c3d11f7c": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_c5d78751c560c3d11f7c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_faf2675ff2c118db",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "d77ebc41d9dff1a98fc7f9ca0f9065aa4e2522e50d11efaaf107e5576ae5ad4a",
            "sm_103a": "fc99f82a24b39f84ba15be626badbd8b93811ed6cfa6c26623b175f10c048507",
        },
    },
    "c6923d9ce04d6f662377": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "75123f12e227cd9163e576e242d5f89698f7bd11fd085a3c8aeb3e78d0f4a3be",
            "sm_103a": "49ee0d21a0808874f9f6b1d23a957e8e6bb330618dcf91345b0b4a651eb29c9d",
        },
    },
    "c87c8c30e66509e69143": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "f1243a03ff83aaa4d5a9111344a51ca7a8653eaeb72d5e63abeb6fdff5203b05",
        },
    },
    "cce6da2b9b938e36636e": {
        "kernel": "e51e19d19bcf42ae3ee0",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_cce6da2b9b938e36636e",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_c1b5ef50690a4dd3",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "b44ca8ce5bf856e1b661b2a8364982774dca4794d72331246ac22e865bac4efd",
        },
    },
    "ce3ff83a062648fc8cf3": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_103a": "c42e078eac56f174361b1f9a3335fafe72b21ebf588cdfe1dd46d035e5e5ca66",
        },
    },
    "cf1cb7fc1bc0e3d2e586": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "eff8cc905c52661b30d22b55bf89457a8b532780c6e0d8fc4f42a8e33dd08845",
        },
    },
    "cf59fa0f9ceb0c7f38f8": {
        "kernel": "6a8f0bd74f7f517d1b11",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_cf59fa0f9ceb0c7f38f8",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a780a1eff74bee17",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "7a6356a8f5da2772a6b07abbe092e601eeef15a3f0f0a18f1a5690be89a091fd",
            "sm_103a": "cf362ae693066f42b3586e795ac347cabfbdf95d571f61f3f145e221689a2fb7",
        },
    },
    "d0fe858e838021c309d5": {
        "binding": "318ddeeecf4e9addca1e",
        "compile_flags": [
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_d0fe858e838021c309d5",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3aebf8f11c710b7c",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "0d54304ad6ea34ffd42e224316171c379e364a5e474946dd6babdccd45c639d8",
            "sm_103a": "0d2504ab0beef8bb0e6b59106a039d3b5eeab3a7c1e87a967197a1f2b91b25de",
        },
    },
    "d1466677b74827b7a614": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "35ad62b6f8806dfae71f5a6198b6ec9dd13d632437aa813a655db74bc4ca4474",
            "sm_103a": "fe1b65ce36a45c16a41330970c64998804aba76c85432e56bebaf4857e399697",
        },
    },
    "d37bd37aefbd1c2663ce": {
        "kernel": "795f9ce57ce7e6812b6c",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_d37bd37aefbd1c2663ce",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_e8df8f724688efa0",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "311e57b6a9d6496dd71426563a4d21062d2da118a69401f556e98e681c101eae",
            "sm_103a": "f6d0206646856053b8cb04f5de569f904487780c1399332e76fafc386d50dbb0",
        },
    },
    "d4316f7bb385939a95f9": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_d4316f7bb385939a95f9",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_095cb0007084350e",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "623e4d4b86e0f04a69f87124a5c5d69beda32e597fead3bc958d61c8ffd8e1b3",
            "sm_103a": "1ea7a29c9461641622cd4e6f243d81418dcdcec2853749ecdd3e41491a0e7181",
        },
    },
    "d7bad35e46134c38d973": {
        "kernel": "267ea66edacba2a365e7",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_d7bad35e46134c38d973",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_9694422168006a9d",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "1a50a0f20eac0adca75beca569475e2b1171d8767a9a178928c5f4643d8e5e27",
            "sm_103a": "4cb0fa00f0200b0625f7ba22864a3c09f0bb8080b44404c9bb5a04d99b838ac3",
        },
    },
    "da7c611282ca76f06547": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "075d224a0c0b7db511c9ccd504ac15d003aba6abc6d20a6c91277eaa7d0d9a15",
            "sm_103a": "598a80395fa29d35e8456d385b53133b98bb5461e010b6fd572fae2d3d89e62f",
        },
    },
    "dc74162695ffedf79430": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "66a4a591dc62d749070bae642fba55eb17b1172f0e997074c41c3f79cc2db2c4",
        },
    },
    "dcd89b2e097fbee53eb7": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=1",
            "-DCAKE_Q_WORDS=8",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_dcd89b2e097fbee53eb7",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_e58e97981fbf7913",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "edd0de2ff6ee0820afec7ebfcdb9bc3a85a098241fdd199e9237719bc7cb6ee3",
            "sm_103a": "13ec53226961fbdfb61d6fd9f9b2b262d7b30e29a690b22fa9fde9ee020b7311",
        },
    },
    "dd3a532430c9f881f75d": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "63d7df8eeceb080a0c7bd874055c79a34d4ea336f280b99f2752e943295f90b2",
            "sm_103a": "90993e42d0bf0a95c57cac6ffc780cebd734cf3bf6f4c4037843ba039c59bfcf",
        },
    },
    "df3fe3e881a30df9d6c8": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_MIN_BLOCKS=4",
            "-DCAKE_FOLD=1",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_df3fe3e881a30df9d6c8",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_4e9353661205c671",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "98a17be6b6f04ceb9b99a6a596bc21c92c0f195b105f45e40a67d745e9433d99",
            "sm_103a": "099643cf57c2162a7b69aaf0be2754560f792a986eacdfc5d019a7d8ed84e795",
        },
    },
    "e08870a4eb2ac432bddb": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "854a6e13ece8c5e30234aac765eafe38900e71dcc9fc59b635ab9aff770afba4",
        },
    },
    "e101e3ffad576f12a678": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=1",
            "-DCAKE_Q_BPT=4",
            "-DCAKE_Q_WORDS=32",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_e101e3ffad576f12a678",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_d18b3ddea00fecfa",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "3ea2b6933b73bbdc5c6360413d50933560168c3ea598b5aa23fdea0eb2207d14",
            "sm_103a": "2f3fbe56792cc40f85a57014ce31867e070834987f519aee2ee089e0ba7f83a2",
        },
    },
    "e3cc580e3d59b4530e6c": {
        "kernel": "07b0189c8953c02acff3",
        "binding": "07b0189c8953c02acff3",
        "compile_flags": [
            "-DCAKE_A_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_e3cc580e3d59b4530e6c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_583f7938bf79b95d",
        ],
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "f52081490e4f020378f9c90f03819757163e7995a62d106ad305432ae556df8f",
            "sm_103a": "e08bba939c461f662e07fc51dd833426024c743491a9e06a309b17eae6a098b4",
        },
    },
    "e3d47605ebbbddca6c95": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "d0819d34a7ae1a2294de356f16f73c84a516c0f9f4eca6182526ac0973a3d8db",
        },
    },
    "e44e8d8454778c38ae99": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "6dc0340a8a8172f6619b3bf7a1b63fd737e83455b771375b2d21ac0edbe0de3f",
        },
    },
    "e51e19d19bcf42ae3ee0": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "ae567b113a6cf52f096ec93edc193502f541ce14833f124f34e34ad428cfad30",
        },
    },
    "e883e0ef0aaab55e1d12": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "6ec6272f7ef987ed9794e31b59a11e599c27c991049c15fff40563806052540f",
            "sm_103a": "97cd6523ad33ab32f6f11bfc47e6b626f23b8d32b2846913208ff36731286327",
        },
    },
    "e98c42f779ed53e68b30": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_e98c42f779ed53e68b30",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_94d7519b3ec8760e",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "8008bcfe130b853d793e1d92083409fce45e5298b0d5d95c24ca010159c8f2e9",
            "sm_103a": "23b49df119ccb84c977fbc316745f132993b43d90656eb1becd20f92e243fdb6",
        },
    },
    "eac184768fafd7b7da36": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "28e8ac514ae280d824311a2bcd2eb2bdb9d5bcb38743f4f4ffd5df8fcb36d663",
            "sm_103a": "b4fbcbff04023bce2013ec40f33c73b7da820b566f7ae572bb84c2a8efa93f3a",
        },
    },
    "eb92c19102b3704c9256": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "2822dc08f74ebcf3e060bc6da10bd6ad8c2de5bd6b172f7b9bc849133cc17398",
            "sm_103a": "43bf22f6d82afd7a3adf9313a72bcd4c1256fc2275610eed56dc21db3969f60f",
        },
    },
    "ecef0c7e6fb2c77bd726": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "c4b6d209b23c210ff9c33e38f886eb6b966410a0eac3b005cc96f930e31bae37",
        },
    },
    "ed1f7ddf3b4e31cb9644": {
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_103a": "f482139c4673208ddda04a7b53ea990db83720e4d7b7c86b6ac7d208d93554df",
        },
    },
    "ed87ebc67c2bc540d361": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "c6cfe5e649409f451f2d009cb35776d221057c96a03909129e5b61c745a6c71d",
            "sm_103a": "a415e68d1bad80bd5011af0913c82ff94add4ade3583c99eff4120e1561a631c",
        },
    },
    "f2568ff35a8d59e839a4": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_103a": "928e0623d434e3e8d0c051be91ad7f1ef395e5e54e62d94f97aaf3dc11f2ddf5",
        },
    },
    "f53911591c205d67bcce": {
        "kernel": "a1674da021d7fdb5e3d0",
        "binding": "1fa625dd627403d62d62",
        "compile_flags": [
            "-DCAKE_OUT_F16=1",
            "-DCAKE_RASTER_GROUP=8",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_f53911591c205d67bcce",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_35f508f05a4142e9",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "95be93a203fe8763f2d24185e1f485e6470ee1d9329153772ffe83f1850d099c",
            "sm_103a": "91025a22375b349c6b2eca100c67fe0659d85bd47e0d3cbefe9c9bce90e80488",
        },
    },
    "f60f72c632bbe3a388e8": {
        "arg_plan": "gemm",
        "closure_sha256": {
            "sm_100a": "418dd2e4e16b63bde4c1bc888ce43e54499785748332aa6ea6bc1e88ac5c6c6c",
            "sm_103a": "83ccc209f3fe9fab995a4008af41137fecc02dbc251838bb7e821ce467fa51d0",
        },
    },
    "f7ff78ec5855b9b22ea7": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_f7ff78ec5855b9b22ea7",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a39c7e44083b7975",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "8f74f39c9d16e0027ce9eba3bfaf2bb54d3c4577dc22f9e9712325c1823a6620",
            "sm_103a": "bab83a64802ae59870560ec24056011926b8f222810db5a874df654d430768e6",
        },
    },
    "f876d047b54150469a71": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=512",
            "-DCAKE_Q_BLOCK=512u",
            "-DCAKE_Q_MIN_BLOCKS=2",
            "-DCAKE_Q_WARPS=16",
            "-DCAKE_FOLD=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_f876d047b54150469a71",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_f3f75c866bf65a53",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "ce1cea442ce17a97a6aa76a5424620a69bfc931b39430ec7bb6f9a609f8cb77c",
            "sm_103a": "c860ad346e0ebc0d5f58b60cb1eca76db62f61a8059c91482479ea2572392359",
        },
    },
    "fa93c77677bea4179924": {
        "kernel": "266051f22af995b10bc8",
        "binding": "266051f22af995b10bc8",
        "compile_flags": [
            "-DCAKE_Q_THREADS=128",
            "-DCAKE_Q_BLOCK=128u",
            "-DCAKE_Q_MIN_BLOCKS=8",
            "-DCAKE_Q_WARPS=4",
            "-DCAKE_STORE_EVICT_LAST=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_fa93c77677bea4179924",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_815589414a88ccc3",
        ],
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "60c652c49c8e6bcd6104c4df9cfd37aaf72a2e3f5910c67049aa7228a4de9cda",
            "sm_103a": "17bd5b01aa5be0d19e1c9b662d1838b1f22711283eaf9f02fcba1b9263ede13a",
        },
    },
    "fb4cabc9f04a36b804d3": {
        "arg_plan": "gemm_2cta_ptr",
        "closure_sha256": {
            "sm_103a": "2384d5825822d9043752deb7ecfdb8e48642f81091ad58bbe4a6cec378eda3c2",
        },
    },
    "fbb5be3d584b9a5e954d": {
        "arg_plan": "quant",
        "closure_sha256": {
            "sm_100a": "e9948bfd1c997c3e1a24874f20a672c1fe14de45b9e19e52e9edc841c6bdade6",
            "sm_103a": "06706d49e3251f6d2df05b36c2dff39f0fd352e36ed154061725427a3dde22af",
        },
    },
    "fbbc792eeaea807c03ac": {
        "kernel": "dd3a532430c9f881f75d",
        "binding": "dd3a532430c9f881f75d",
        "compile_flags": [
            "-DCAKE_A_POLICY=0",
            "-DCAKE_B_POLICY=0x12F0000000000000ULL",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_nvfp4_per_token_fbbc792eeaea807c03ac",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_4ff099b96a5eb045",
        ],
        "arg_plan": "gemm_2cta",
        "closure_sha256": {
            "sm_100a": "c5030c545763ffd50e594e514293ae0d06c2115370c62cdf20b90358dfff3c85",
            "sm_103a": "ace3617d84c448fa0fa3557ffdd2b2afedee8398187b6ce8b51d51a55c7b2a76",
        },
    },
}

# Logical kernel key -> program: the assignments both architectures share,
# then the ones only one of them registers (the SM100-only stream-K ``skt6``
# programs; the SM103-only ``m192`` ``bF`` programs and two-CTAs-per-SM
# ``_o2`` programs).
_KERNELS_SHARED: dict[str, str] = {
    "gemm:m128_2cta_bf16_aL_bL_l2256b": "cake_nvfp4_per_token_dd3a532430c9f881f75d",
    "gemm:m128_2cta_bf16_aL_bL_l2256b_g8": "cake_nvfp4_per_token_7abdeaf1ad1dc7c64ef4",
    "gemm:m128_2cta_bf16_bF_l2256b": "cake_nvfp4_per_token_fbbc792eeaea807c03ac",
    "gemm:m128_2cta_f16_bF_l2256b": "cake_nvfp4_per_token_3b069158e5efd68b3d56",
    "gemm:m128_bf16_bF": "cake_nvfp4_per_token_795f9ce57ce7e6812b6c",
    "gemm:m128_bf16_bF_ep4": "cake_nvfp4_per_token_d1466677b74827b7a614",
    "gemm:m128_bf16_bF_ep4_fold": "cake_nvfp4_per_token_045ed4b8200220c8d8cd",
    "gemm:m128_bf16_bF_fold": "cake_nvfp4_per_token_b8a8b7c06a7533a489cd",
    "gemm:m128_f16_bF": "cake_nvfp4_per_token_d37bd37aefbd1c2663ce",
    "gemm:m128_f16_bF_ep4": "cake_nvfp4_per_token_8befa090dcb3f0e7a7d9",
    "gemm:m128_f16_bF_ep4_fold": "cake_nvfp4_per_token_2f8110591ec3a80e3f68",
    "gemm:m128_f16_bF_fold": "cake_nvfp4_per_token_234a3b72a787d48e951e",
    "gemm:m192_2cta_bf16_aL_bL_l2256b_g8": "cake_nvfp4_per_token_6fed99e520aa239db6c9",
    "gemm:m192_2cta_f16_aL_bL_l2256b_g8": "cake_nvfp4_per_token_f53911591c205d67bcce",
    "gemm:m256_2cta_bf16_bF_l2256b": "cake_nvfp4_per_token_8c27d1a9b8f95d84ea63",
    "gemm:m256_2cta_bf16_bF_l2256b_b2fold": "cake_nvfp4_per_token_4f4d6711fee3151ee9ac",
    "gemm:m256_2cta_f16_bF_l2256b": "cake_nvfp4_per_token_9069465975d6f1ca6311",
    "gemm:m256_2cta_f16_bF_l2256b_b2fold": "cake_nvfp4_per_token_27d1205b4f5b3183d9de",
    "gemm:m64_2cta_bf16_bF_l2256b": "cake_nvfp4_per_token_78fcd21adb756e3fddc9",
    "gemm:m64_2cta_f16_bF_l2256b": "cake_nvfp4_per_token_1d6630a145ff559d1fa3",
    "gemm:m64_2cta_hm_bf16_bF_l2256b": "cake_nvfp4_per_token_becd24cbb8333dc48dcb",
    "gemm:m64_2cta_hm_f16_bF_l2256b": "cake_nvfp4_per_token_66c17b2c6c8645cc5a76",
    "gemm:m64_bf16_bF_l2256b": "cake_nvfp4_per_token_17270a6f1bb78bfd8f92",
    "gemm:m64_bf16_bF_l2256b_ro": "cake_nvfp4_per_token_1005a13dfc670fd5de26",
    "gemm:m64_f16_bF_l2256b": "cake_nvfp4_per_token_6791cd4bf5cc9b5e088b",
    "gemm:m64_f16_bF_l2256b_ro": "cake_nvfp4_per_token_c048f1b68848e4fc102d",
    "gemm:n16_bf16_aF_l2256b": "cake_nvfp4_per_token_3eae14c9d3c4c24cc180",
    "gemm:n16_k512_bf16_aF_l2256b_s3": "cake_nvfp4_per_token_eac184768fafd7b7da36",
    "gemm:n16_sk2_bf16_aF_l2256b": "cake_nvfp4_per_token_da7c611282ca76f06547",
    "gemm:n16_sk3_bf16_l2256b": "cake_nvfp4_per_token_e883e0ef0aaab55e1d12",
    "gemm:n16_sk3_bf16_l2256b_ep4": "cake_nvfp4_per_token_74bf657c544a53c216c6",
    "gemm:n16_sk3_f16_l2256b": "cake_nvfp4_per_token_55dc9450f28f61f389fa",
    "gemm:n16_sk3_f16_l2256b_ep4": "cake_nvfp4_per_token_2363f86763371089670f",
    "gemm:n16_sk4_bf16_l2256b": "cake_nvfp4_per_token_adf05dbe9f624ff487d0",
    "gemm:n32_bf16_aF_l2256b": "cake_nvfp4_per_token_6a8f0bd74f7f517d1b11",
    "gemm:n32_bf16_aF_l2256b_ep4": "cake_nvfp4_per_token_bfa0c3e4e60a86104b16",
    "gemm:n32_f16_aF_l2256b": "cake_nvfp4_per_token_cf59fa0f9ceb0c7f38f8",
    "gemm:n32_f16_aF_l2256b_ep4": "cake_nvfp4_per_token_4a8ea29b299cdfc83a99",
    "gemm:n32_k512_bf16_aF_l2256b_s3": "cake_nvfp4_per_token_267ea66edacba2a365e7",
    "gemm:n32_k512_bf16_aF_l2256b_s3_ep4": "cake_nvfp4_per_token_326a7ad2e4cc24b95493",
    "gemm:n32_k512_f16_aF_l2256b_s3": "cake_nvfp4_per_token_d7bad35e46134c38d973",
    "gemm:n32_k512_f16_aF_l2256b_s3_ep4": "cake_nvfp4_per_token_05727367662197581d3c",
    "gemm:n32_sk2_bf16_aF_l2256b": "cake_nvfp4_per_token_257fb8ea914211995ad6",
    "gemm:n32_sk2_bf16_aF_l2256b_ep4": "cake_nvfp4_per_token_f60f72c632bbe3a388e8",
    "gemm:n32_sk2_f16_aF_l2256b": "cake_nvfp4_per_token_99374a4603d9e9ec58a2",
    "gemm:n32_sk2_f16_aF_l2256b_ep4": "cake_nvfp4_per_token_797cb820f08d9d66a3dd",
    "gemm:n8_bf16_aF_l2256b": "cake_nvfp4_per_token_925e8165fab2f00266d9",
    "gemm:n8_bf16_aF_l2256b_ep4": "cake_nvfp4_per_token_a1b33fc9ef3db1732eb7",
    "gemm:n8_bf16_aF_l2256b_o2": "cake_nvfp4_per_token_12c633607bd8e310fc7c",
    "gemm:n8_f16_aF_l2256b": "cake_nvfp4_per_token_310dce04e62dc4479f2f",
    "gemm:n8_f16_aF_l2256b_ep4": "cake_nvfp4_per_token_5600b6d759d61fbe3ae0",
    "gemm:n8_f16_aF_l2256b_o2": "cake_nvfp4_per_token_b0b95c4118991384623d",
    "gemm:n8_k512_bf16_aF_l2256b_s3": "cake_nvfp4_per_token_9ca737b8f50f70eb4b5d",
    "gemm:n8_k512_bf16_aF_l2256b_s3_ep4": "cake_nvfp4_per_token_4f250853fb07cca4f4f2",
    "gemm:n8_k512_f16_aF_l2256b_s3": "cake_nvfp4_per_token_5606b2d9571f0787e563",
    "gemm:n8_k512_f16_aF_l2256b_s3_ep4": "cake_nvfp4_per_token_59777655982de0ec51e2",
    "gemm:n8_sk2_bf16_aF_l2256b": "cake_nvfp4_per_token_e3cc580e3d59b4530e6c",
    "gemm:n8_sk2_bf16_aF_l2256b_ep4": "cake_nvfp4_per_token_ad54bb8bc23776e32f04",
    "gemm:n8_sk2_bf16_l2256b": "cake_nvfp4_per_token_07b0189c8953c02acff3",
    "gemm:n8_sk2_f16_aF_l2256b": "cake_nvfp4_per_token_93b60320bbfa3449a921",
    "gemm:n8_sk2_f16_aF_l2256b_ep4": "cake_nvfp4_per_token_800dc1b69b288d010f61",
    "gemm:n8_sk2_f16_l2256b": "cake_nvfp4_per_token_942b39aa9ca66fb980e8",
    "gemm:n8_sk4_bf16_l2256b": "cake_nvfp4_per_token_3df66e1abbe9dc0f2230",
    "gemm:n8_sk4_bf16_l2256b_ep4": "cake_nvfp4_per_token_57acea79765928447145",
    "gemm:n8_sk4_f16_l2256b": "cake_nvfp4_per_token_7ca723b13756dd926bfd",
    "gemm:n8_sk4_f16_l2256b_ep4": "cake_nvfp4_per_token_6ed9e5d8627577028e1d",
    "quant:t128_b1_mb1_bf16_cs4": "cake_nvfp4_per_token_33b08b7620357f6c712c",
    "quant:t128_b1_mb1_bf16_cs4_k7168": "cake_nvfp4_per_token_580e9edbb66ac9359718",
    "quant:t128_b1_mb1_bf16_cs8": "cake_nvfp4_per_token_13cec0563138e6e72423",
    "quant:t128_b1_mb1_bf16_fold_cs4": "cake_nvfp4_per_token_0588831a72930146a0e8",
    "quant:t128_b1_mb1_bf16_fold_cs4_k7168": "cake_nvfp4_per_token_6c716dab52bbcbf07637",
    "quant:t128_b1_mb1_bf16_fold_cs8": "cake_nvfp4_per_token_2c9d5ff42bc5ac1b60f6",
    "quant:t128_b2_mb8_bf16_fold_sl": "cake_nvfp4_per_token_98c1d7945ad0dcb98963",
    "quant:t128_b2_mb8_bf16_sl": "cake_nvfp4_per_token_fa93c77677bea4179924",
    "quant:t128_b4_mb8_bf16_fold_sl": "cake_nvfp4_per_token_8499e042c80c6cf6a00f",
    "quant:t128_b4_mb8_bf16_sl": "cake_nvfp4_per_token_b6b2ef866ed5f8f91335",
    "quant:t128_b8_mb5_bf16_fold_sl": "cake_nvfp4_per_token_996c7fd21642f288f697",
    "quant:t128_b8_mb5_bf16_sl": "cake_nvfp4_per_token_68ee524b89f5e289133c",
    "quant:t256_b1_mb1_bf16_cs4": "cake_nvfp4_per_token_30cc10800fc253a813ab",
    "quant:t256_b1_mb1_bf16_cs8": "cake_nvfp4_per_token_b4effb8c3185b65b4d84",
    "quant:t256_b1_mb1_bf16_fold_cs4": "cake_nvfp4_per_token_c002f42b63a48a7bd2b9",
    "quant:t256_b1_mb1_bf16_fold_cs8": "cake_nvfp4_per_token_91cba126dc2e7278c729",
    "quant:t256_b1_mb1_bf16_fold_k2688": "cake_nvfp4_per_token_dcd89b2e097fbee53eb7",
    "quant:t256_b1_mb1_bf16_fold_k4096": "cake_nvfp4_per_token_dcd89b2e097fbee53eb7",
    "quant:t256_b1_mb1_bf16_k2688": "cake_nvfp4_per_token_53e1454ac0d533cefcda",
    "quant:t256_b1_mb1_bf16_k4096": "cake_nvfp4_per_token_53e1454ac0d533cefcda",
    "quant:t256_b1_mb3_bf16_cs4": "cake_nvfp4_per_token_640c33b699a8478f60bb",
    "quant:t256_b1_mb3_bf16_cs8": "cake_nvfp4_per_token_fbb5be3d584b9a5e954d",
    "quant:t256_b1_mb3_bf16_fold_cs4": "cake_nvfp4_per_token_52f7cc1603bc0a530734",
    "quant:t256_b1_mb3_bf16_fold_cs8": "cake_nvfp4_per_token_b6793e01c3abcc81980c",
    "quant:t256_b1_mb3_bf16_fold_k2688": "cake_nvfp4_per_token_0d13de6f1682e55c1373",
    "quant:t256_b1_mb3_bf16_fold_k4096": "cake_nvfp4_per_token_0d13de6f1682e55c1373",
    "quant:t256_b1_mb3_bf16_k2688": "cake_nvfp4_per_token_4d85976d6c18ce148ca4",
    "quant:t256_b1_mb3_bf16_k4096": "cake_nvfp4_per_token_4d85976d6c18ce148ca4",
    "quant:t256_b1_mb4_bf16_fold_sl_k2688": "cake_nvfp4_per_token_34e066f626b4be154822",
    "quant:t256_b1_mb4_bf16_fold_sl_k4096": "cake_nvfp4_per_token_34e066f626b4be154822",
    "quant:t256_b1_mb4_bf16_sl_k2688": "cake_nvfp4_per_token_96e142849dcfaaf7c1a4",
    "quant:t256_b1_mb4_bf16_sl_k4096": "cake_nvfp4_per_token_96e142849dcfaaf7c1a4",
    "quant:t256_b2_mb3_bf16": "cake_nvfp4_per_token_266051f22af995b10bc8",
    "quant:t256_b2_mb3_bf16_cs4": "cake_nvfp4_per_token_4dfd2b6475da344471f9",
    "quant:t256_b2_mb3_bf16_fold": "cake_nvfp4_per_token_2e26ffc7ce8d8fa6f128",
    "quant:t256_b2_mb3_bf16_fold_cs4": "cake_nvfp4_per_token_eb92c19102b3704c9256",
    "quant:t256_b2_mb3_bf16_fold_k7168": "cake_nvfp4_per_token_e98c42f779ed53e68b30",
    "quant:t256_b2_mb3_bf16_k7168": "cake_nvfp4_per_token_f7ff78ec5855b9b22ea7",
    "quant:t256_b2_mb3_f16_k7168": "cake_nvfp4_per_token_318ddeeecf4e9addca1e",
    "quant:t256_b2_mb4_bf16_fold_sl": "cake_nvfp4_per_token_aa08d05b7a61e9ea3498",
    "quant:t256_b2_mb4_bf16_fold_sl_k7168": "cake_nvfp4_per_token_df3fe3e881a30df9d6c8",
    "quant:t256_b2_mb4_bf16_sl": "cake_nvfp4_per_token_8cf7856e23c713904d31",
    "quant:t256_b2_mb4_bf16_sl_k7168": "cake_nvfp4_per_token_8c70d72aaecb580a5783",
    "quant:t256_b2_mb4_f16_sl_k7168": "cake_nvfp4_per_token_d0fe858e838021c309d5",
    "quant:t256_b4_mb2_bf16": "cake_nvfp4_per_token_78452434e3ec2d8f5e13",
    "quant:t256_b4_mb2_bf16_fold": "cake_nvfp4_per_token_c253402ceb2f05cb709c",
    "quant:t256_b4_mb4_bf16_fold_sl": "cake_nvfp4_per_token_aa2933408365d2cd4269",
    "quant:t256_b4_mb4_bf16_sl": "cake_nvfp4_per_token_c5d78751c560c3d11f7c",
    "quant:t256_b5_mb4_bf16_fold_sl": "cake_nvfp4_per_token_6864d70ab56de3ba8bfd",
    "quant:t256_b5_mb4_bf16_sl": "cake_nvfp4_per_token_75d30269a025a195653c",
    "quant:t256_b7_mb3_bf16_fold_sl": "cake_nvfp4_per_token_6717875e66a6fcec8341",
    "quant:t256_b7_mb3_bf16_sl": "cake_nvfp4_per_token_83fec50b7af0311f8244",
    "quant:t512_b1_mb1_bf16": "cake_nvfp4_per_token_58eeece815f48fd1d43a",
    "quant:t512_b1_mb1_bf16_cs4": "cake_nvfp4_per_token_c6923d9ce04d6f662377",
    "quant:t512_b1_mb1_bf16_fold": "cake_nvfp4_per_token_69f8274b12662366ecb8",
    "quant:t512_b1_mb1_bf16_fold_cs4": "cake_nvfp4_per_token_38827385d2d509e25ed9",
    "quant:t512_b1_mb1_bf16_fold_k7168": "cake_nvfp4_per_token_2126ed4021a5df3e8461",
    "quant:t512_b1_mb1_bf16_k7168": "cake_nvfp4_per_token_3c0a0335f02050ebb60e",
    "quant:t512_b2_mb1_bf16": "cake_nvfp4_per_token_7652be76af016751909c",
    "quant:t512_b2_mb1_bf16_fold": "cake_nvfp4_per_token_d4316f7bb385939a95f9",
    "quant:t512_b2_mb2_bf16": "cake_nvfp4_per_token_7a07511bc4c0c0f06089",
    "quant:t512_b2_mb2_bf16_fold": "cake_nvfp4_per_token_f876d047b54150469a71",
    "quant:t512_b3_mb1_bf16": "cake_nvfp4_per_token_062311ac440fbaf37629",
    "quant:t512_b3_mb1_bf16_fold": "cake_nvfp4_per_token_3bebdf16f8b8ec303083",
    "quant:t512_b3_mb1_bf16_fold_pc": "cake_nvfp4_per_token_90858258ec070501f7f1",
    "quant:t512_b3_mb1_bf16_pc": "cake_nvfp4_per_token_2cfd03d9bf011a475da4",
    "quant:t512_b3_mb2_bf16": "cake_nvfp4_per_token_4a4ae65a3551bedbc110",
    "quant:t512_b3_mb2_bf16_fold": "cake_nvfp4_per_token_48485b687a58cef62559",
    "quant:t512_b4_mb1_bf16": "cake_nvfp4_per_token_684d5b4c9c173753e179",
    "quant:t512_b4_mb1_bf16_fold": "cake_nvfp4_per_token_e101e3ffad576f12a678",
    "quant:t512_b4_mb2_bf16": "cake_nvfp4_per_token_ab2e761d5b22bebc40e7",
    "quant:t512_b4_mb2_bf16_fold": "cake_nvfp4_per_token_98778736f957e3baf9cd",
}
_KERNELS_ONLY: dict[str, dict[str, str]] = {
    "sm_100a": {
        "gemm:m128_bf16_bF_ro": "cake_nvfp4_per_token_4083e913d0b68dac7255",
        "gemm:m128_bf16_bF_ro_ep4": "cake_nvfp4_per_token_7143db00594241971565",
        "gemm:m128_f16_bF_ro": "cake_nvfp4_per_token_0b7a9083d24d01fe3aec",
        "gemm:m128_f16_bF_ro_ep4": "cake_nvfp4_per_token_4fad98101b1166c3ac8f",
        "gemm:m192_2cta_bf16_aL_bL_l2256b": "cake_nvfp4_per_token_a1674da021d7fdb5e3d0",
        "gemm:m192_2cta_f16_aL_bL_l2256b": "cake_nvfp4_per_token_1fa625dd627403d62d62",
        "gemm:m192_2cta_skt6_bf16_aL_bL_l2256b": "cake_nvfp4_per_token_288adcab31da85191a26",
        "gemm:m192_2cta_skt6_bf16_bF_l2256b": "cake_nvfp4_per_token_5872000a0bd8bd391059",
        "gemm:m192_2cta_skt6_f16_aL_bL_l2256b": "cake_nvfp4_per_token_aa069aa70914114fa1c5",
        "gemm:m192_2cta_skt6_f16_bF_l2256b": "cake_nvfp4_per_token_1bc4fe20d5d1fa219ea4",
        "gemm:m256_2cta_bf16_aL_bL_l2256b": "cake_nvfp4_per_token_a6fc65c9fcae34137fec",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_clc": "cake_nvfp4_per_token_18fd61a4a764c2d1ab3e",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g16_clc": "cake_nvfp4_per_token_455c60a5c10aa2bcece8",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g8": "cake_nvfp4_per_token_139c3fab2a0249e02c3e",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g8_clc": "cake_nvfp4_per_token_5d8bea1558154053d314",
        "gemm:m256_2cta_f16_aL_bL_l2256b": "cake_nvfp4_per_token_1de51aa8fd0339bfc55c",
        "gemm:m256_2cta_f16_aL_bL_l2256b_clc": "cake_nvfp4_per_token_64313df7982791417a2b",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g16_clc": "cake_nvfp4_per_token_45fc8d20d916671c5f36",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g8": "cake_nvfp4_per_token_72a49b93eb629736017a",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g8_clc": "cake_nvfp4_per_token_a6cf54d2fb100c00ad63",
        "gemm:m64_bf16_bF_l2256b_ro_ep4": "cake_nvfp4_per_token_45ce0d9fdd8d0936a613",
        "gemm:m64_f16_bF_l2256b_ro_ep4": "cake_nvfp4_per_token_a3b720172d1a7538d508",
        "gemm:n8_bf16_aF_l2256b_o2_ep4": "cake_nvfp4_per_token_502fc8dafecb312370d7",
        "gemm:n8_f16_aF_l2256b_o2_ep4": "cake_nvfp4_per_token_a2b26a244ffb71a5141a",
        "gemm:n8_sk2_bf16_l2256b_ep4": "cake_nvfp4_per_token_dc74162695ffedf79430",
        "gemm:n8_sk2_f16_l2256b_ep4": "cake_nvfp4_per_token_a0ba477a1d576d523533",
        "gemm:n8_sk3_bf16_l2256b": "cake_nvfp4_per_token_b649171f6e9e74551522",
        "gemm:n8_sk3_bf16_l2256b_ep4": "cake_nvfp4_per_token_94e5e5aea8a1a29c27ec",
        "gemm:n8_sk3_f16_l2256b": "cake_nvfp4_per_token_ed87ebc67c2bc540d361",
        "gemm:n8_sk3_f16_l2256b_ep4": "cake_nvfp4_per_token_e08870a4eb2ac432bddb",
        "quant:t256_b2_mb3_bf16_cs2": "cake_nvfp4_per_token_32c8706387a82055788f",
        "quant:t256_b2_mb3_bf16_fold_cs2": "cake_nvfp4_per_token_e44e8d8454778c38ae99",
    },
    "sm_103a": {
        "gemm:m192_2cta_bf16_aL_bL_l2256b": "cake_nvfp4_per_token_1fa4a36589c5c72b1efd",
        "gemm:m192_2cta_bf16_bF_l2256b": "cake_nvfp4_per_token_90f233fd2ad0ca2fdd38",
        "gemm:m192_2cta_f16_aL_bL_l2256b": "cake_nvfp4_per_token_ed1f7ddf3b4e31cb9644",
        "gemm:m192_2cta_f16_bF_l2256b": "cake_nvfp4_per_token_6659f7757bcb7bae97b1",
        "gemm:m192_bf16_bF_l2256b": "cake_nvfp4_per_token_e51e19d19bcf42ae3ee0",
        "gemm:m192_f16_bF_l2256b": "cake_nvfp4_per_token_cce6da2b9b938e36636e",
        "gemm:m256_2cta_bf16_aL_bL_l2256b": "cake_nvfp4_per_token_ae447cd8c8a488cbe10c",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_clc": "cake_nvfp4_per_token_05cdc205d9b471f666a1",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_clc_sfr": "cake_nvfp4_per_token_6d96bb93d7840b5b9b42",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_clc_sfr_mc42": "cake_nvfp4_per_token_a4857feefa4431fea347",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g16_clc": "cake_nvfp4_per_token_1957cf0620e9be66eee8",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g16_clc_sfr": "cake_nvfp4_per_token_ecef0c7e6fb2c77bd726",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g8": "cake_nvfp4_per_token_102a58d983738fb4ba20",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g8_clc": "cake_nvfp4_per_token_0ee9b811105da3e1e757",
        "gemm:m256_2cta_bf16_aL_bL_l2256b_g8_sfr": "cake_nvfp4_per_token_e3d47605ebbbddca6c95",
        "gemm:m256_2cta_f16_aL_bL_l2256b": "cake_nvfp4_per_token_60a084e22436164149fc",
        "gemm:m256_2cta_f16_aL_bL_l2256b_clc": "cake_nvfp4_per_token_34d44d8d6d4e632aea8c",
        "gemm:m256_2cta_f16_aL_bL_l2256b_clc_sfr": "cake_nvfp4_per_token_fb4cabc9f04a36b804d3",
        "gemm:m256_2cta_f16_aL_bL_l2256b_clc_sfr_mc42": "cake_nvfp4_per_token_7e68f46d2c17ead412fc",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g16_clc": "cake_nvfp4_per_token_3deb92b0859de2875bd4",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g16_clc_sfr": "cake_nvfp4_per_token_afb177dad29f9232b073",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g8": "cake_nvfp4_per_token_9db9fb4fa9cbe2095158",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g8_clc": "cake_nvfp4_per_token_cf1cb7fc1bc0e3d2e586",
        "gemm:m256_2cta_f16_aL_bL_l2256b_g8_sfr": "cake_nvfp4_per_token_132e33a6f1dd393dc000",
        "gemm:m64_bf16_bF_l2256b_ro_ep4": "cake_nvfp4_per_token_45ce0d9fdd8d0936a613",
        "gemm:m64_f16_bF_l2256b_ro_ep4": "cake_nvfp4_per_token_a3b720172d1a7538d508",
        "gemm:n16_bf16_aF_l2256b_o2": "cake_nvfp4_per_token_183e28bdf124c263ecfb",
        "gemm:n32_bf16_aF_l2256b_o2": "cake_nvfp4_per_token_7d87cdf4f15f1439b3e0",
        "gemm:n32_bf16_aF_l2256b_o2_ep4": "cake_nvfp4_per_token_95369b42a903d39fca2c",
        "gemm:n32_f16_aF_l2256b_o2": "cake_nvfp4_per_token_3cd58718ee867217608b",
        "gemm:n32_f16_aF_l2256b_o2_ep4": "cake_nvfp4_per_token_1932a5484f6bbe6d70c3",
        "gemm:n8_bf16_aF_l2256b_w64": "cake_nvfp4_per_token_f2568ff35a8d59e839a4",
        "gemm:n8_bf16_aF_l2256b_w64_ep4": "cake_nvfp4_per_token_784b1664a803959166c5",
        "gemm:n8_f16_aF_l2256b_w64": "cake_nvfp4_per_token_95a134a511d7eeb8a6b8",
        "gemm:n8_f16_aF_l2256b_w64_ep4": "cake_nvfp4_per_token_7a7786a40f1adf944110",
        "gemm:n8_sk3_bf16_l2256b": "cake_nvfp4_per_token_76b9d6595f0d3be566ce",
        "gemm:n8_sk3_bf16_l2256b_ep4": "cake_nvfp4_per_token_14b9336da0e3b4fbfade",
        "gemm:n8_sk3_f16_l2256b": "cake_nvfp4_per_token_c87c8c30e66509e69143",
        "gemm:n8_sk3_f16_l2256b_ep4": "cake_nvfp4_per_token_a4c4b7bb3ae7574dd507",
        "quant:t256_b7_mb3_bf16_fold_sl_ru": "cake_nvfp4_per_token_ce3ff83a062648fc8cf3",
        "quant:t512_b2_mb1_bf16_fold_pc": "cake_nvfp4_per_token_3f4db4aa72227118688f",
        "quant:t512_b2_mb1_bf16_pc": "cake_nvfp4_per_token_772a0fce61655a1df842",
    },
}
KERNELS: dict[str, dict[str, str]] = {
    arch: dict(sorted({**_KERNELS_SHARED, **only}.items()))
    for arch, only in _KERNELS_ONLY.items()
}

DEFINITIONS: dict[str, dict[str, int]] = {
    "quant:t128_b1_mb1_bf16_cs4_k7168": {"K_STATIC": 7168},
    "quant:t128_b1_mb1_bf16_fold_cs4_k7168": {"K_STATIC": 7168},
    "quant:t256_b1_mb1_bf16_fold_k2688": {"K_STATIC": 2688},
    "quant:t256_b1_mb1_bf16_fold_k4096": {"K_STATIC": 4096},
    "quant:t256_b1_mb1_bf16_k2688": {"K_STATIC": 2688},
    "quant:t256_b1_mb1_bf16_k4096": {"K_STATIC": 4096},
    "quant:t256_b1_mb3_bf16_fold_k2688": {"K_STATIC": 2688},
    "quant:t256_b1_mb3_bf16_fold_k4096": {"K_STATIC": 4096},
    "quant:t256_b1_mb3_bf16_k2688": {"K_STATIC": 2688},
    "quant:t256_b1_mb3_bf16_k4096": {"K_STATIC": 4096},
    "quant:t256_b1_mb4_bf16_fold_sl_k2688": {"K_STATIC": 2688},
    "quant:t256_b1_mb4_bf16_fold_sl_k4096": {"K_STATIC": 4096},
    "quant:t256_b1_mb4_bf16_sl_k2688": {"K_STATIC": 2688},
    "quant:t256_b1_mb4_bf16_sl_k4096": {"K_STATIC": 4096},
    "quant:t256_b2_mb3_bf16_fold_k7168": {"K_STATIC": 7168},
    "quant:t256_b2_mb3_bf16_k7168": {"K_STATIC": 7168},
    "quant:t256_b2_mb3_f16_k7168": {"K_STATIC": 7168},
    "quant:t256_b2_mb4_bf16_fold_sl_k7168": {"K_STATIC": 7168},
    "quant:t256_b2_mb4_bf16_sl_k7168": {"K_STATIC": 7168},
    "quant:t256_b2_mb4_f16_sl_k7168": {"K_STATIC": 7168},
    "quant:t512_b1_mb1_bf16_fold_k7168": {"K_STATIC": 7168},
    "quant:t512_b1_mb1_bf16_k7168": {"K_STATIC": 7168},
}


def _module(digest: str, row: dict[str, Any]) -> dict[str, Any]:
    """The ``MODULES`` record of program ``digest`` from its ``_PROGRAMS`` row."""
    name = f"cake_nvfp4_per_token_{digest}"
    return {
        "sources": [
            f"{_SOURCE_PREFIX}{row.get('kernel', digest)}_kernel.cu",
            f"{_SOURCE_PREFIX}{row.get('binding', digest)}_binding.cu",
        ],
        "arches": [arch for arch, table in KERNELS.items() if name in table.values()],
        "compile_flags": list(row.get("compile_flags", ())),
        "arg_plan": row["arg_plan"],
        "closure_sha256": row["closure_sha256"],
    }


MODULES: dict[str, dict[str, Any]] = {
    f"cake_nvfp4_per_token_{digest}": _module(digest, row)
    for digest, row in _PROGRAMS.items()
}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def route_available(arch: str) -> bool:
    """True when ``arch`` is registered."""
    return arch in KERNELS


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered program for ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if table is None:
        raise NotImplementedError(
            f"The generated per-token NVFP4 programs for {arch} are not "
            "registered in this checkout"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated per-token NVFP4 kernel {key!r} for {arch} is not "
            "registered in this checkout"
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


def kernel_definitions(key: str) -> tuple[tuple[str, int], ...]:
    """Compile-line definitions of the program behind ``key`` (sorted ``(name, value)`` pairs)."""
    return tuple(sorted(DEFINITIONS.get(key, {}).items()))


@functools.cache
def gen_cake_nvfp4_per_token_module(
    name: str, arch: str, definitions: tuple[tuple[str, int], ...] = ()
):
    """JIT spec of program ``name`` compiled for ``arch`` with the compile-line
    ``definitions`` (one build per architecture and definition set; the spec
    name carries every supplied value)."""
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    suffix = "".join(f"_{key.lower()}{value}" for key, value in definitions)
    return gen_jit_spec(
        name=f"{name}_{arch}_" + record["closure_sha256"][arch][:20] + suffix,
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
            *(f"-D{key}={value}" for key, value in definitions),
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_nvfp4_per_token_module(
    name: str, arch: str, definitions: tuple[tuple[str, int], ...] = ()
):
    return gen_cake_nvfp4_per_token_module(name, arch, definitions).build_and_load()
