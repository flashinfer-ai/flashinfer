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
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, Sequence

from . import env as jit_env
from ._kda_jit_common import get_flashinfer_include_dir, get_kda_csrc_dir
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags

CakeFusedKDADecodeTarget = Literal["sm100a", "sm103a"]
CakeFusedKDADecodeStateIndicesMode = Literal[
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
]

_TARGETS: tuple[CakeFusedKDADecodeTarget, ...] = ("sm100a", "sm103a")
_STATE_INDICES_MODES: tuple[CakeFusedKDADecodeStateIndicesMode, ...] = (
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
)
_TARGET_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}

# Generated programs, shared by every target architecture (the exporter
# writes this registry).  The 64-bit slot-offset twin of a schedule and the
# LAUNCH_MIN_BLOCKS=1 instantiation of the wide512 positive-FP32 schedule
# compile the same ``_kernel.cu`` body as their base program; ``defines``
# lists the ``-D`` knobs (``CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS``,
# ``CAKE_FUSED_KDA_LAUNCH_MIN_BLOCKS``) that select one member's text, so
# the body's ``#if`` blocks expand to exactly the former per-program source.
# The tvm-ffi binding of a program is rendered from the template of its launch
# ABI (``render_cake_fused_kda_decode_binding``); every program exports the same
# ``run`` entry point and is built with the same ``--use_fast_math`` flag.
MODULES: dict[str, dict[str, Any]] = {
    "cake_fused_kda_decode_0743aa5d22f746909b3c": {
        "sources": ["cake_fused_kda_decode_0743aa5d22f746909b3c_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_0743aa5d22f746909b3c",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 36480,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "b6ccb815c6e67337b2763e20ac3400b4a0a4a9be9a1ab9532d20e516e06cc2a1",
            "sm_103a": "4e3ac89dcf270dc3ccb4055f1533bb01bf44b03f0167820cc482d0f933d42bdf",
        },
    },
    "cake_fused_kda_decode_1105a251fd33b55719ae": {
        "sources": ["cake_fused_kda_decode_7e841caca37f5c48f0a3_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_1105a251fd33b55719ae",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 9344,
            "cluster": [2, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "2f090a8d65ea80dd384a731f5611e76c04e2e23291240dd25a0bd21b67540e62",
            "sm_103a": "39384e8ce7cd0bd4b0ebd898e8daa3ed8507d9bcfe57a14f7b8d15f36f3d1e62",
        },
    },
    "cake_fused_kda_decode_23498e31ce2702fee03c": {
        "sources": ["cake_fused_kda_decode_f415905e21bbc11b4581_kernel.cu"],
        "defines": [
            "CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1",
            "CAKE_FUSED_KDA_LAUNCH_MIN_BLOCKS=1",
        ],
        "kernel_symbol": "kernel_cake_fused_kda_decode_23498e31ce2702fee03c",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "8d2a722baadd2e243376a847c5c3211b86b0b2f3fd81d064dfa81b303a79d6a4",
            "sm_103a": "58fd4e63326f0f4acf3d0fe59bb0b0e7d6161a8f2833169612d8b370043dc57e",
        },
    },
    "cake_fused_kda_decode_26bd2da2218a44305e35": {
        "sources": ["cake_fused_kda_decode_26bd2da2218a44305e35_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_26bd2da2218a44305e35",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "3559f735e57b6690b1e6ebf3aaa9cedb02d2a07ea29867ff678e65e6ccb1a253",
            "sm_103a": "e82bad2cffb796b637a0e77c1e78d0525d4d5d2b0ed9ea7346c7655538bed6d4",
        },
    },
    "cake_fused_kda_decode_3109b5a8356a11c3dabb": {
        "sources": ["cake_fused_kda_decode_97a7e6ab6064b0fa1bc8_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_3109b5a8356a11c3dabb",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "52a02103350c7b74d53cefbb4eaf263ce33216b5ba7222c8264f095365aa3622",
            "sm_103a": "3fcec670d4d35449ac5143e720daaffdc8324eae19d68461537ee2fa874bb018",
        },
    },
    "cake_fused_kda_decode_3d014ff47814a933a847": {
        "sources": ["cake_fused_kda_decode_3d014ff47814a933a847_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_3d014ff47814a933a847",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 70272,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "ce27e9ef2460dbcf4630c51741ae603c48ac6effef7e0720fabd2a009fe99400",
            "sm_103a": "ced39d189683873ac724e4ab937083bf49e882cf37e7464f5cdf7fc8fd305f10",
        },
    },
    "cake_fused_kda_decode_3fcc8696ad27c9a56a84": {
        "sources": ["cake_fused_kda_decode_3fcc8696ad27c9a56a84_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_3fcc8696ad27c9a56a84",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "ae2f7f66d714b33525d4cb43d533dac3e7b0e537d04ae9481b362ee3c1dad2fc",
            "sm_103a": "fc7d63b192f7bd0776860e9ec0366aeba2c562036c6ff842d8b0148d0a7ab5ed",
        },
    },
    "cake_fused_kda_decode_47e6858f50f60799991b": {
        "sources": ["cake_fused_kda_decode_9fb82e69235a7fdd5f96_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_47e6858f50f60799991b",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 28288,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "f26bc67cbb17202221666e61ae698587a8fbcd443efaac83c1e568a65ca1c6bb",
            "sm_103a": "8b0bad2fa314944d852fcfcac35339997586b623e8e5c1ada58bd58442a67fb3",
        },
    },
    "cake_fused_kda_decode_4d1216db0b67e580d6f7": {
        "sources": ["cake_fused_kda_decode_4d1216db0b67e580d6f7_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_4d1216db0b67e580d6f7",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 36480,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "4d1d4294a92b2d441d4a01e91dc50d8777924bf8e0ea7d16b3a3b6cf937a7aba",
            "sm_103a": "5a6f0cdfd784c526e9ab0edf686c9c23eb533b5a970981242f9ff6445a8bea1a",
        },
    },
    "cake_fused_kda_decode_4dc09b8c1d1bab526c64": {
        "sources": ["cake_fused_kda_decode_0743aa5d22f746909b3c_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_4dc09b8c1d1bab526c64",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 36480,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "c54c4903790f27e46785191e76c895c7a4938dabb54001bf89e15be668e348b0",
            "sm_103a": "2db6cb410f718d20caa515d0985fa14a75c198823b243653ce8b334d776eda02",
        },
    },
    "cake_fused_kda_decode_583b3404f8230ee830a8": {
        "sources": ["cake_fused_kda_decode_59e2055a73df5c6ec7fa_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_583b3404f8230ee830a8",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 36480,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "95ce5d231a8b59149d4aa356759ef10e3a32f3cc467b1b2f40017a060ee2a2dc",
            "sm_103a": "2fb83a39f2e902e3e6cb3a07bd24a05be74770b23ab0b7195df6df87392264e9",
        },
    },
    "cake_fused_kda_decode_59e2055a73df5c6ec7fa": {
        "sources": ["cake_fused_kda_decode_59e2055a73df5c6ec7fa_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_59e2055a73df5c6ec7fa",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 36480,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "3fdebd37ae963396d006ffffc5afeee5847127271761d08763dbebb082f2879d",
            "sm_103a": "5c674b65b77777dd9953803d4c134604c28d31f10cf306311d41deb3f4352644",
        },
    },
    "cake_fused_kda_decode_5b7fd31f6c4783cbe2c6": {
        "sources": ["cake_fused_kda_decode_903332461e85b6c8df36_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_5b7fd31f6c4783cbe2c6",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "24a6b0ac37bc2839624ade793b62b0aa3b868a8955034cf78d4e8b3be260842a",
            "sm_103a": "d10ae39bb55ddde9d94d9e24ec20b65fd789c7b99adba6af19282517d8648e1b",
        },
    },
    "cake_fused_kda_decode_753293b369d11fb0880f": {
        "sources": ["cake_fused_kda_decode_753293b369d11fb0880f_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_753293b369d11fb0880f",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "b87fd5ff1bce588af8094d2dfdd24c6e0083c0de88ba41c17670d1947ad36e22",
            "sm_103a": "48b58f7d589849d47646d41b1472ebcd04c9240dfbbf589fb38bf67748a5cbfe",
        },
    },
    "cake_fused_kda_decode_7c97fb1858b20c965ee0": {
        "sources": ["cake_fused_kda_decode_7e5629eb5984e1177160_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_7c97fb1858b20c965ee0",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "b63991695d6987c5caaef8b9234b3805cde656c38f88967d36f5d78f7a4b520c",
            "sm_103a": "0a6433e6c76c5a12b6c0d30f29ffc730febf41a228b1e9a6c1fa9ff220a6945f",
        },
    },
    "cake_fused_kda_decode_7caf25be4378d6f0f0ed": {
        "sources": ["cake_fused_kda_decode_7caf25be4378d6f0f0ed_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_7caf25be4378d6f0f0ed",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 28288,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "414b11bdb03ef6c69a89089f7aefbc4e7e436af9a8efe24a02a69d5d406c420e",
            "sm_103a": "2e2a5b43f8d70793c26555b192dcdd7ca4e3c83cca9b4c4662c0437d31a82621",
        },
    },
    "cake_fused_kda_decode_7e5629eb5984e1177160": {
        "sources": ["cake_fused_kda_decode_7e5629eb5984e1177160_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_7e5629eb5984e1177160",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "c1e2e7f3f6b6603bad05fac3dd2aac1e8d1f50a42c8834edcd2a9b826e3d4855",
            "sm_103a": "8b5ca26bbd6e805775ca07a47eefe8fcc21d869200a9371945e27234fba0a79a",
        },
    },
    "cake_fused_kda_decode_7e841caca37f5c48f0a3": {
        "sources": ["cake_fused_kda_decode_7e841caca37f5c48f0a3_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_7e841caca37f5c48f0a3",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 9344,
            "cluster": [2, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "69c7e1a21aa46907f1bef5c450df9871960d9330a6f915d140eaab18dc4d1979",
            "sm_103a": "b4d35b3f4e625a3e9233ebc4a4ef30316ad3f1a81a51ee11bc954822c11ff05b",
        },
    },
    "cake_fused_kda_decode_82fc564c161daa575154": {
        "sources": ["cake_fused_kda_decode_3d014ff47814a933a847_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_82fc564c161daa575154",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 70272,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "2602257b1ec3ef1c077b6ed481af937d150ad57af2e09016c48d386526fb37a9",
            "sm_103a": "ae82a1461620a1df9f6712697beb9ae71a838bf10afe5039704bc57dda47d53b",
        },
    },
    "cake_fused_kda_decode_836d666a9d0b6decc492": {
        "sources": ["cake_fused_kda_decode_a07c16f264b1fc295f84_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_836d666a9d0b6decc492",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 20096,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "1581a8aaa9270d1827a41100cdfb0afc42723d1eebe6942d5ce0ca707529aa30",
            "sm_103a": "0f895d0ab32407ff0d86372e6650c1d939f6d7e2d96a0012963c069c1822f4fa",
        },
    },
    "cake_fused_kda_decode_8b697d61d1b15a6fac4d": {
        "sources": ["cake_fused_kda_decode_3fcc8696ad27c9a56a84_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_8b697d61d1b15a6fac4d",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "596c857ef2d889a10e4ba1c4e058cc17261d990e0da3532229fce6feaf294c46",
            "sm_103a": "be81f3a605f2c5a395ba2f2820cec85a23c418402cca9f68caaeec1ba68d8c33",
        },
    },
    "cake_fused_kda_decode_903332461e85b6c8df36": {
        "sources": ["cake_fused_kda_decode_903332461e85b6c8df36_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_903332461e85b6c8df36",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "3c8a37525cd5ec15ba314e13604ca329b370531eded37b83ec9ebf3a50d207c5",
            "sm_103a": "fbac3fe121b1dcbe6fa3cf004c18c6574bdd275ac292b43198adfc6a0a48f92d",
        },
    },
    "cake_fused_kda_decode_97a7e6ab6064b0fa1bc8": {
        "sources": ["cake_fused_kda_decode_97a7e6ab6064b0fa1bc8_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_97a7e6ab6064b0fa1bc8",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "63330d200a80cda1049274a77505b233b503a8b8e6fc36eb7ae9c39abda454c9",
            "sm_103a": "061993608e849e1cee2bec3c562209c3300bad907a5ea5969d369b392b1f924e",
        },
    },
    "cake_fused_kda_decode_9823df19f9a25b5b670e": {
        "sources": ["cake_fused_kda_decode_dd24f0fe2c61f6730898_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_9823df19f9a25b5b670e",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "a31bf3eacc2a07fe270f7e09483fe4e956f2874ac7bb50d2af70d669917cb51f",
            "sm_103a": "f1f73bdbb78c4261a1a5348fa0618bf28ab8287bed1718b8a2a608f7e2dc0948",
        },
    },
    "cake_fused_kda_decode_9dc0bb23b72067eaee2f": {
        "sources": ["cake_fused_kda_decode_7caf25be4378d6f0f0ed_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_9dc0bb23b72067eaee2f",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 28288,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "a5e2a1be2f597b74b93d46873d7339042619aa16db94b6a90bd549bfac35e700",
            "sm_103a": "f79f5d24df250f2b497f2077c0d5d064817fb87b00a23ced31f568a450e6df9b",
        },
    },
    "cake_fused_kda_decode_9fb82e69235a7fdd5f96": {
        "sources": ["cake_fused_kda_decode_9fb82e69235a7fdd5f96_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_9fb82e69235a7fdd5f96",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 28288,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "fb80aa8d366df4e73dfecbbe1f040454f374fca795e070271defe71139596f9c",
            "sm_103a": "8371254bfdd00c78b395a78bd3a5ce36660f3c14f1708cff3d98cefab6745554",
        },
    },
    "cake_fused_kda_decode_a07c16f264b1fc295f84": {
        "sources": ["cake_fused_kda_decode_a07c16f264b1fc295f84_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_a07c16f264b1fc295f84",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 20096,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "bd26229606257ecd4e849cba84d875e53ce1b284621a6713345921b0b984f418",
            "sm_103a": "e3e19a6277faeb3f703df8d69c1fed2d773acadeb565c8614921bcfb54c6a12e",
        },
    },
    "cake_fused_kda_decode_c988ce55655115a24ea5": {
        "sources": ["cake_fused_kda_decode_f415905e21bbc11b4581_kernel.cu"],
        "defines": [
            "CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1",
            "CAKE_FUSED_KDA_LAUNCH_MIN_BLOCKS=2",
        ],
        "kernel_symbol": "kernel_cake_fused_kda_decode_c988ce55655115a24ea5",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "531364a86d9dfee8f7e7058797b279a64cd9874b421e3ed14c2bbfb727178367",
            "sm_103a": "a09bc2a949a722449bff8f4d0def9b5054bfe8e1e3b86b499cbf7dd872a01df4",
        },
    },
    "cake_fused_kda_decode_ceb0985ab93695f02817": {
        "sources": ["cake_fused_kda_decode_753293b369d11fb0880f_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_ceb0985ab93695f02817",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "38de31ef9ec8fdeb10bda411d0c07e18c84fddb5b54e0c82e32b0df9625d20b8",
            "sm_103a": "bce7d3e30be483d0db91b3ddd4e029bb5de72bb13194e450fdd7f0b216488b95",
        },
    },
    "cake_fused_kda_decode_dd24f0fe2c61f6730898": {
        "sources": ["cake_fused_kda_decode_dd24f0fe2c61f6730898_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_dd24f0fe2c61f6730898",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "562dceac882b91bb008e2d7d706bb56be5adf3c47a328410804c5f76fa6587c8",
            "sm_103a": "c073bd565e4811603e74e11497a9398334102a09eb77a01a1eb80b51e570cc67",
        },
    },
    "cake_fused_kda_decode_e71bdc5861ac91ffb4bc": {
        "sources": ["cake_fused_kda_decode_f415905e21bbc11b4581_kernel.cu"],
        "defines": [
            "CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0",
            "CAKE_FUSED_KDA_LAUNCH_MIN_BLOCKS=1",
        ],
        "kernel_symbol": "kernel_cake_fused_kda_decode_e71bdc5861ac91ffb4bc",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "6dbaf3003cef0668ec9fef82041d125ca61eba76d899fe9bbbb7ec0ad78112fc",
            "sm_103a": "677be823a99cdf9dd1026640df6c0acafae476fd776beaafb6d7c78e4c550994",
        },
    },
    "cake_fused_kda_decode_f2aada6c39fd5dce4bee": {
        "sources": ["cake_fused_kda_decode_4d1216db0b67e580d6f7_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_f2aada6c39fd5dce4bee",
        "launch": {
            "block": [256, 1, 1],
            "dynamic_smem_bytes": 36480,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "c9ed1d546b59559483cd3dfac739cfa295c8ec43df05aabd6e02ef0875c062cb",
            "sm_103a": "21d30b5d6ac2b3e376f7de3c0567762ef7c1f3ef13b40afda9e10125b656b1ca",
        },
    },
    "cake_fused_kda_decode_f415905e21bbc11b4581": {
        "sources": ["cake_fused_kda_decode_f415905e21bbc11b4581_kernel.cu"],
        "defines": [
            "CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=0",
            "CAKE_FUSED_KDA_LAUNCH_MIN_BLOCKS=2",
        ],
        "kernel_symbol": "kernel_cake_fused_kda_decode_f415905e21bbc11b4581",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "4763ee0205472fe15d4ff06f61ef1a786683ac992be4e7fd27a19cdc30bf64c3",
            "sm_103a": "07626becd74faa005488986682263b35032d30afd255f0e248cb6eb608c898cc",
        },
    },
    "cake_fused_kda_decode_fd601115ded1c0f96ee1": {
        "sources": ["cake_fused_kda_decode_26bd2da2218a44305e35_kernel.cu"],
        "defines": ["CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS=1"],
        "kernel_symbol": "kernel_cake_fused_kda_decode_fd601115ded1c0f96ee1",
        "launch": {
            "block": [512, 1, 1],
            "dynamic_smem_bytes": 3712,
            "cluster": [1, 1, 1],
        },
        "closure_sha256": {
            "sm_100a": "9ef865553cecc42426c86e334d97325ee401978f58e46b998e1828862052b8eb",
            "sm_103a": "c1046c6cadcba9c70edc2bb1067ee080e71a03780fdc71d1e3e7421eef5fbb35",
        },
    },
}
# Logical kernel key (variant name) -> generated program.
ROUTES: dict[str, str] = {
    "cluster2_wide_positive_f32": "cake_fused_kda_decode_7e841caca37f5c48f0a3",
    "cluster2_wide_positive_f32_wide_slot_offsets": "cake_fused_kda_decode_1105a251fd33b55719ae",
    "compact_async_bf16": "cake_fused_kda_decode_59e2055a73df5c6ec7fa",
    "compact_async_bf16_wide_slot_offsets": "cake_fused_kda_decode_583b3404f8230ee830a8",
    "compact_async_f32": "cake_fused_kda_decode_4d1216db0b67e580d6f7",
    "compact_async_f32_wide_slot_offsets": "cake_fused_kda_decode_f2aada6c39fd5dce4bee",
    "compact_async_positive_f32": "cake_fused_kda_decode_0743aa5d22f746909b3c",
    "compact_async_positive_f32_wide_slot_offsets": "cake_fused_kda_decode_4dc09b8c1d1bab526c64",
    "direct_bf16": "cake_fused_kda_decode_3fcc8696ad27c9a56a84",
    "direct_bf16_wide_slot_offsets": "cake_fused_kda_decode_8b697d61d1b15a6fac4d",
    "direct_f32": "cake_fused_kda_decode_7e5629eb5984e1177160",
    "direct_f32_wide_slot_offsets": "cake_fused_kda_decode_7c97fb1858b20c965ee0",
    "high_work_bf16": "cake_fused_kda_decode_a07c16f264b1fc295f84",
    "high_work_bf16_wide_slot_offsets": "cake_fused_kda_decode_836d666a9d0b6decc492",
    "high_work_f32": "cake_fused_kda_decode_9fb82e69235a7fdd5f96",
    "high_work_f32_wide_slot_offsets": "cake_fused_kda_decode_47e6858f50f60799991b",
    "high_work_positive_f32": "cake_fused_kda_decode_7caf25be4378d6f0f0ed",
    "high_work_positive_f32_wide_slot_offsets": "cake_fused_kda_decode_9dc0bb23b72067eaee2f",
    "repeated_safe_bf16": "cake_fused_kda_decode_dd24f0fe2c61f6730898",
    "repeated_safe_bf16_wide_slot_offsets": "cake_fused_kda_decode_9823df19f9a25b5b670e",
    "repeated_safe_f32": "cake_fused_kda_decode_97a7e6ab6064b0fa1bc8",
    "repeated_safe_f32_wide_slot_offsets": "cake_fused_kda_decode_3109b5a8356a11c3dabb",
    "stream_bf16": "cake_fused_kda_decode_3d014ff47814a933a847",
    "stream_bf16_wide_slot_offsets": "cake_fused_kda_decode_82fc564c161daa575154",
    "wide512_bf16": "cake_fused_kda_decode_753293b369d11fb0880f",
    "wide512_bf16_wide_slot_offsets": "cake_fused_kda_decode_ceb0985ab93695f02817",
    "wide512_f32": "cake_fused_kda_decode_26bd2da2218a44305e35",
    "wide512_f32_wide_slot_offsets": "cake_fused_kda_decode_fd601115ded1c0f96ee1",
    "wide512_positive_f32": "cake_fused_kda_decode_f415905e21bbc11b4581",
    "wide512_positive_f32_wide_slot_offsets": "cake_fused_kda_decode_c988ce55655115a24ea5",
    "wide512_regcap128_positive_f32": "cake_fused_kda_decode_e71bdc5861ac91ffb4bc",
    "wide512_regcap128_positive_f32_wide_slot_offsets": "cake_fused_kda_decode_23498e31ce2702fee03c",
    "wide512_vector4_positive_f32": "cake_fused_kda_decode_903332461e85b6c8df36",
    "wide512_vector4_positive_f32_wide_slot_offsets": "cake_fused_kda_decode_5b7fd31f6c4783cbe2c6",
}

_COMMON_BUFFER_ABI: tuple[tuple[str, str, str], ...] = (
    ("buffer", "x", "bfloat16"),
    ("buffer", "weight", "float32"),
    ("buffer", "conv_state", "bfloat16"),
    ("buffer", "raw_gate", "bfloat16"),
    ("buffer", "raw_beta", "bfloat16"),
    ("buffer", "A_log", "float32"),
    ("buffer", "dt_bias", "float32"),
    ("buffer", "state_indices", "int32"),
    ("buffer", "state", "state_dtype"),
    ("buffer", "output_gate", "bfloat16"),
    ("buffer", "norm_weight", "float32"),
    ("buffer", "output", "bfloat16"),
)
_COMMON_SCALAR_ABI: tuple[tuple[str, str, str], ...] = (
    ("parameter", "x_row_stride", "int32"),
    ("parameter", "conv_slot_stride", "int32"),
    ("parameter", "beta_row_stride", "int32"),
    ("parameter", "state_slot_stride", "int32"),
    ("parameter", "output_gate_row_stride", "int32"),
    ("parameter", "H", "int32"),
)
_RUNTIME_CONFIG_ABI: tuple[tuple[str, str, str], ...] = (
    ("parameter", "use_lower_bound", "int32"),
    ("parameter", "lower_bound_log2", "float32_scalar"),
    ("parameter", "norm_eps", "float32_scalar"),
)
_ROWS_ABI: tuple[tuple[str, str, str], ...] = (("parameter", "rows", "int32"),)
CAKE_FUSED_KDA_DECODE_ABIS: dict[str, tuple[tuple[str, str, str], ...]] = {
    # One CTA per (head, row) grid cell.
    "standard": _COMMON_BUFFER_ABI + _COMMON_SCALAR_ABI + _RUNTIME_CONFIG_ABI,
    # The owner-chain kernel receives one row per launch; the program's binding
    # walks the rows sequentially (one same-stream launch each) so repeated
    # slots observe every mutation.
    "repeated_safe": (
        _COMMON_BUFFER_ABI + _COMMON_SCALAR_ABI + _ROWS_ABI + _RUNTIME_CONFIG_ABI
    ),
    # One persistent launch receives every row and walks the (row, head) work
    # items with a grid-stride loop; the host sizes the grid from the device's
    # resident-CTA capacity instead of the item count.
    "persistent_rows": (
        _COMMON_BUFFER_ABI + _COMMON_SCALAR_ABI + _ROWS_ABI + _RUNTIME_CONFIG_ABI
    ),
}
_HEADS: tuple[int, ...] = (8, 12, 24, 32, 48, 96)
# H=8 positive-unique FP32 rows served by the two-CTA cluster split (measured
# on B200 and B300).
_H8_CLUSTER2_MAX_ROWS = 9
# One wide512 wave at one CTA per SM: the LAUNCH_MIN_BLOCKS=1 instantiation of the
# wide512 positive-FP32 schedule serves rows 10..18.
_H8_WIDE512_REGCAP128_MAX_ROWS = 18
# The persistent BF16 streaming family keeps three 256-thread CTAs resident per
# SM (register and 70 KiB dynamic shared memory budget).
_PERSISTENT_RESIDENT_CTAS_PER_SM = 3
_LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class CakeFusedKDADecodeEligibility:
    """One conjunction of host-known facts selecting an exported Cake kernel."""

    heads: tuple[int, ...]
    minimum_rows: int
    maximum_rows: int | None
    state_indices_modes: tuple[CakeFusedKDADecodeStateIndicesMode, ...]
    lower_bound_values: tuple[float | None, ...] | None
    norm_eps_values: tuple[float, ...] | None
    strides: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True)
class CakeFusedKDADecodeVariant:
    """One registered logical kernel: a generated program plus its launch description."""

    name: str
    target: CakeFusedKDADecodeTarget
    module: str
    body_path: Path
    binding_template: Path
    source_sha256: str
    kernel_symbol: str
    ffi_entry: str
    arg_plan: tuple[tuple[str, str], ...]
    abi_kind: str
    state_dtype: str
    slot_offset_bits: int
    extra_cuda_cflags: tuple[str, ...]
    threads: int
    dynamic_smem_bytes: int
    eligibility: tuple[CakeFusedKDADecodeEligibility, ...]
    # CTAs per (row, head) work item along grid.x: 1 for every single-CTA
    # schedule, 2 for the two-CTA cluster split (compile-time
    # ``__cluster_dims__(2,1,1)`` in the program; the generated binding launches
    # with the matching cluster attribute).
    cluster_x: int = 1


# Eligibility vocabulary.  A band is an inclusive ``(minimum_rows, maximum_rows)``
# interval (``None`` = unbounded); ``_rules`` expands ``{heads: bands}`` into one
# rule per (head, band) in table order.
_POSITIVE = ["positive_unique"]
_NULLABLE = ["unique_or_null"]
_UNIQUE = ["positive_unique", "unique_or_null"]
_REPEATED = ["repeated_positive"]
_RUNTIME_STRIDES: dict[str, int | None] = {
    "x_row_stride": None,
    "conv_slot_stride": None,
    "beta_row_stride": None,
    "state_slot_stride": None,
    "output_gate_row_stride": None,
}
_OPEN = ((1, None),)
_ALL_HEADS_OPEN = {heads: _OPEN for heads in _HEADS}
# Two-CTA-wave bands of the 512-thread family and the staged compact family,
# measured per head count (B200/B300 route matrices, 2026-09).
_WIDE512_BANDS = {
    8: ((1, 18),),
    12: ((1, 12),),
    24: ((1, 6),),
    32: ((1, 4),),
    48: ((1, 3),),
    96: ((1, 1),),
}
_COMPACT_BANDS = {
    12: ((25, 37),),
    24: ((13, 18),),
    32: ((10, 13),),
    48: ((7, 9),),
    96: ((4, 4),),
}
_HIGH_WORK_BANDS = {
    12: ((99, None),),
    24: ((50, None),),
    32: ((32, None),),
    48: ((25, None),),
    96: ((13, None),),
}
_DIRECT_NULLABLE_BANDS = {
    12: ((13, 24), (38, 98)),
    24: ((7, 12), (19, 49)),
    32: ((5, 9), (14, 31)),
    48: ((4, 6), (10, 24)),
    96: ((2, 3), (5, 12)),
}


def _rules(
    modes: list[str],
    bands: Mapping[int, tuple[tuple[int, int | None], ...]],
) -> list[dict[str, Any]]:
    return [
        {
            "heads": [heads],
            "minimum_rows": minimum_rows,
            "maximum_rows": maximum_rows,
            "state_indices_modes": list(modes),
            "lower_bound_values": "any",
            "norm_eps_values": "any",
            "strides": dict(_RUNTIME_STRIDES),
        }
        for heads, head_bands in bands.items()
        for minimum_rows, maximum_rows in head_bands
    ]


def _variant(
    name: str,
    *,
    abi_kind: str,
    state_dtype: str,
    eligibility: list[dict[str, Any]],
    slot_offset_bits: int,
    cluster_x: int = 1,
) -> dict[str, Any]:
    return {
        "name": name,
        "abi_kind": abi_kind,
        "state_dtype": state_dtype,
        "slot_offset_bits": slot_offset_bits,
        "eligibility": tuple(eligibility),
        "cluster_x": cluster_x,
    }


def _twins(name: str, **spec: Any) -> tuple[dict[str, Any], ...]:
    """The 64-bit slot-offset program and its 32-bit twin share one launch contract.

    Both compile the base schedule's kernel body; the program's ``defines``
    (``CAKE_FUSED_KDA_WIDE_SLOT_OFFSETS``) select the member's text.
    """

    return (
        _variant(f"{name}_wide_slot_offsets", slot_offset_bits=64, **spec),
        _variant(name, slot_offset_bits=32, **spec),
    )


_F32 = dict(abi_kind="standard", state_dtype="float32")
_BF16 = dict(abi_kind="standard", state_dtype="bfloat16")

# Logical kernel registrations in selection order: the first variant whose
# eligibility matches wins.  Every entry maps to a generated program through
# ``ROUTES``; evaluation-layout or head-count clones do not exist, the generic
# stride-taking programs serve every layout.
_VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    # Repeated slots: the owner-chain body walks one row per launch; the binding loops the rows.
    *_twins(
        "repeated_safe_f32",
        abi_kind="repeated_safe",
        state_dtype="float32",
        eligibility=_rules(_REPEATED, _ALL_HEADS_OPEN),
    ),
    *_twins(
        "repeated_safe_bf16",
        abi_kind="repeated_safe",
        state_dtype="bfloat16",
        eligibility=_rules(_REPEATED, _ALL_HEADS_OPEN),
    ),
    # H=8 BF16 persistent stream (three resident CTAs per SM) from rows 64.
    *_twins(
        "stream_bf16",
        abi_kind="persistent_rows",
        state_dtype="bfloat16",
        eligibility=_rules(_POSITIVE, {8: ((64, None),)}),
    ),
    # H=8 positive-unique FP32 cluster band below one CTA per SM.
    *_twins(
        "cluster2_wide_positive_f32",
        **_F32,
        cluster_x=2,
        eligibility=_rules(_POSITIVE, {8: ((1, _H8_CLUSTER2_MAX_ROWS),)}),
    ),
    # H=8 positive-unique FP32, one wide512 wave at one CTA per SM (rows 10..18):
    # the same schedule as wide512_positive_f32 with LAUNCH_MIN_BLOCKS 1 (a declared
    # schedule constant), i.e. one template, two instantiations.
    *_twins(
        "wide512_regcap128_positive_f32",
        **_F32,
        eligibility=_rules(
            _POSITIVE,
            {8: ((_H8_CLUSTER2_MAX_ROWS + 1, _H8_WIDE512_REGCAP128_MAX_ROWS),)},
        ),
    ),
    # 512-thread family within one two-CTA wave (positive FP32 also covers the
    # H=8 unstaged band 56..76).
    *_twins(
        "wide512_positive_f32",
        **_F32,
        eligibility=_rules(
            _POSITIVE,
            {
                8: ((_H8_WIDE512_REGCAP128_MAX_ROWS + 1, 37), (56, 76)),
                **{h: _WIDE512_BANDS[h] for h in (12, 24, 32, 48, 96)},
            },
        ),
    ),
    *_twins("wide512_f32", **_F32, eligibility=_rules(_NULLABLE, _WIDE512_BANDS)),
    *_twins("wide512_bf16", **_BF16, eligibility=_rules(_UNIQUE, _WIDE512_BANDS)),
    # Staged compact family (two to three waves).
    *_twins(
        "compact_async_positive_f32",
        **_F32,
        eligibility=_rules(_POSITIVE, _COMPACT_BANDS),
    ),
    *_twins(
        "compact_async_f32",
        **_F32,
        eligibility=_rules(_NULLABLE, {8: ((38, 55),), **_COMPACT_BANDS}),
    ),
    *_twins(
        "compact_async_bf16",
        **_BF16,
        eligibility=_rules(_UNIQUE, {8: ((19, 63),), **_COMPACT_BANDS}),
    ),
    # Rotating high-work pipeline (many waves).
    *_twins(
        "high_work_positive_f32",
        **_F32,
        eligibility=_rules(_POSITIVE, {8: ((38, 55), (77, None)), **_HIGH_WORK_BANDS}),
    ),
    *_twins(
        "high_work_f32",
        **_F32,
        eligibility=_rules(_NULLABLE, {8: ((148, None),), **_HIGH_WORK_BANDS}),
    ),
    *_twins(
        "high_work_bf16",
        **_BF16,
        eligibility=_rules(_NULLABLE, {8: ((148, None),)})
        + _rules(_UNIQUE, _HIGH_WORK_BANDS),
    ),
    # Direct 256-thread fallback between the staged bands.  No "direct_positive_f32"
    # registration: positive-unique FP32 dispatch is gated by _positive_f32_variants
    # below, whose wave arithmetic always returns one of cluster2/wide512*/compact_async/
    # high_work/wide512_vector4 (``wide512_positive_f32`` is its unconditional fallback,
    # since work_items < 2*sm_count never clears the compact/high-work/vector4 bands) --
    # the name "direct_positive_f32" is never produced, so an eligibility band here would
    # be dead by construction.  The shipped GPU test confirmed this: it exercised
    # this variant only through ``factory_only=True`` (direct factory call bypassing the
    # dispatcher), never through the public API.
    *_twins(
        "direct_f32",
        **_F32,
        eligibility=_rules(
            _NULLABLE, {8: ((19, 37), (56, 147)), **_DIRECT_NULLABLE_BANDS}
        ),
    ),
    *_twins(
        "direct_bf16",
        **_BF16,
        eligibility=_rules(_NULLABLE, {8: ((64, 147),)})
        + _rules(_UNIQUE, _DIRECT_NULLABLE_BANDS),
    ),
    # Vector-four producers near a full two-CTA wave (H=12/24 only).
    *_twins(
        "wide512_vector4_positive_f32",
        **_F32,
        eligibility=_rules(_POSITIVE, {12: _OPEN, 24: _OPEN}),
    ),
)
# Every generated program exports one tvm-ffi entry point and is built with the
# exporter's module-wide fast-math flag (the registry used to repeat both).
_FFI_ENTRY = "run"
_EXPORTED_COMPILE_FLAGS: tuple[str, ...] = ("--use_fast_math",)
_GRID_PLAN: tuple[tuple[str, str], ...] = (
    ("grid", "grid_x"),
    ("grid", "grid_y"),
    ("grid", "grid_z"),
)
# The tvm-ffi binding of a program is rendered from the template of its launch
# ABI (``csrc/kda/cake_fused_kda_decode_templates/``, keyed by ABI kind and
# cluster launch); the rendered text is byte-identical to the former per-program
# binding file.  ``@NAME@`` placeholders take the kernel symbol, the entry point,
# the host-shim namespace, the state dtype tokens and the launch constants.
_BINDING_TEMPLATE_DIR = "cake_fused_kda_decode_templates"
_BINDING_TEMPLATES: dict[tuple[str, bool], str] = {
    ("standard", False): "standard_binding.cu.in",
    ("standard", True): "standard_cluster_binding.cu.in",
    ("persistent_rows", False): "persistent_rows_binding.cu.in",
    ("repeated_safe", False): "repeated_safe_binding.cu.in",
}
_BINDING_PLACEHOLDER = re.compile(r"@([A-Z][A-Z0-9_]*)@")
_STATE_DTYPE_TOKENS: dict[str, tuple[str, str]] = {
    "float32": ("float", "DLDataType{kDLFloat, 32, 1}"),
    "bfloat16": ("__nv_bfloat16", "DLDataType{kDLBfloat, 16, 1}"),
}
# Preserve three resident CTAs when NVCC allocates more registers than the
# source compiler for this wide-offset staged schedule.
_OCCUPANCY_FLAGS: dict[str, tuple[str, ...]] = {
    "compact_async_f32_wide_slot_offsets": ("-Xptxas=--minnctapersm=3",),
}
# Build the two-CTA cluster schedule with precise math instead of the module-wide
# fast-math set: under fast math NVCC allocates 62 instead of 64 registers for this
# program and the kernel runs about 2 % slower; the outputs are bitwise identical
# because the program spells its flush-to-zero arithmetic per instruction.
_PRECISE_MATH_VARIANTS: frozenset[str] = frozenset(
    {"cluster2_wide_positive_f32", "cluster2_wide_positive_f32_wide_slot_offsets"}
)
_FAST_MATH_FLAGS: frozenset[str] = frozenset({"-use_fast_math", "--use_fast_math"})


def _values_or_any(value: object) -> tuple[Any, ...] | None:
    if value == "any":
        return None
    if not isinstance(value, list):
        raise ValueError("Cake fused KDA eligibility values must be a list or 'any'")
    return tuple(value)


def _source_path(relative: str) -> Path:
    """Resolve a registry source (relative to ``csrc/kda``) in installed and source checkouts."""

    return (get_kda_csrc_dir() / Path(relative).name).resolve()


def _binding_template_path(abi_kind: str, cluster_x: int) -> Path:
    """Resolve the binding template of one launch ABI under ``csrc/kda``."""

    template = _BINDING_TEMPLATES[(abi_kind, cluster_x != 1)]
    return (get_kda_csrc_dir() / _BINDING_TEMPLATE_DIR / template).resolve()


@functools.cache
def get_cake_fused_kda_decode_variants(
    target: CakeFusedKDADecodeTarget = "sm100a",
) -> tuple[CakeFusedKDADecodeVariant, ...]:
    """Resolve the shared program registry for one explicit compilation target."""

    if target not in _TARGETS:
        raise ValueError(f"unsupported Cake fused KDA target: {target}")
    if not MODULES:
        return ()
    result: list[CakeFusedKDADecodeVariant] = []
    for item in _VARIANT_SPECS:
        name = item["name"]
        if name not in ROUTES:
            raise ValueError(f"Cake fused KDA variant {name} has no generated program")
        record = MODULES[ROUTES[name]]
        (device_source,) = record["sources"]
        device = _source_path(device_source)
        if not device.name.startswith("cake_fused_kda_decode_"):
            raise ValueError(f"invalid Cake fused KDA source path: {device}")
        if not device.is_file():
            raise FileNotFoundError(f"Cake fused KDA source not found: {device}")
        # One closure digest per exported architecture.
        if _TARGET_ARCH[target] not in record["closure_sha256"]:
            raise ValueError(
                f"Cake fused KDA program {ROUTES[name]} does not support {target}"
            )
        eligibility = []
        for rule in item["eligibility"]:
            eligibility.append(
                CakeFusedKDADecodeEligibility(
                    heads=tuple(rule["heads"]),
                    minimum_rows=rule["minimum_rows"],
                    maximum_rows=rule["maximum_rows"],
                    state_indices_modes=tuple(rule["state_indices_modes"]),
                    lower_bound_values=_values_or_any(rule["lower_bound_values"]),
                    norm_eps_values=_values_or_any(rule["norm_eps_values"]),
                    strides=tuple(rule["strides"].items()),
                )
            )
        cluster_x = int(item["cluster_x"])
        if cluster_x != 1 and item["abi_kind"] != "standard":
            raise ValueError(
                f"Cake fused KDA cluster variant {name} must use the standard launch ABI"
            )
        binding_template = _binding_template_path(item["abi_kind"], cluster_x)
        if not binding_template.is_file():
            raise FileNotFoundError(
                f"Cake fused KDA binding template not found: {binding_template}"
            )
        abi = CAKE_FUSED_KDA_DECODE_ABIS[item["abi_kind"]]
        arg_plan = tuple((kind, argument) for kind, argument, _dtype in abi)
        result.append(
            CakeFusedKDADecodeVariant(
                name=name,
                target=target,
                module=ROUTES[name],
                body_path=device,
                binding_template=binding_template,
                # record["closure_sha256"] is {arch: digest} (one build per
                # exported architecture); CakeFusedKDADecodeVariant.source_sha256
                # is this *target's* single digest, matching
                # get_cake_fused_kda_decode_program_identity()'s docstring ("one
                # closure digest per generated program").
                source_sha256=record["closure_sha256"][_TARGET_ARCH[target]],
                kernel_symbol=record["kernel_symbol"],
                ffi_entry=_FFI_ENTRY,
                arg_plan=arg_plan + _GRID_PLAN,
                abi_kind=item["abi_kind"],
                state_dtype=item["state_dtype"],
                slot_offset_bits=item["slot_offset_bits"],
                extra_cuda_cflags=tuple(
                    flag
                    for flag in _EXPORTED_COMPILE_FLAGS
                    if name not in _PRECISE_MATH_VARIANTS
                    or flag not in _FAST_MATH_FLAGS
                )
                + tuple(f"-D{define}" for define in record["defines"])
                + _OCCUPANCY_FLAGS.get(name, ()),
                threads=int(record["launch"]["block"][0]),
                dynamic_smem_bytes=int(record["launch"]["dynamic_smem_bytes"]),
                eligibility=tuple(eligibility),
                cluster_x=cluster_x,
            )
        )
    return tuple(result)


def get_cake_fused_kda_decode_program_identity(
    target: CakeFusedKDADecodeTarget = "sm100a",
) -> str:
    """Return a stable identity for one target's registered executable closure.

    The exporter records one closure digest per generated program (device
    source, binding source and compile flags); the identity lists every route's
    digest so any regenerated program or re-routed variant changes it.
    """

    return ";".join(
        f"{variant.name}={variant.source_sha256}:{variant.target}"
        for variant in get_cake_fused_kda_decode_variants(target)
    )


def _eligibility_matches(
    rule: CakeFusedKDADecodeEligibility,
    *,
    num_heads: int,
    num_rows: int,
    state_indices_mode: CakeFusedKDADecodeStateIndicesMode,
    lower_bound: float | None,
    norm_eps: float,
    strides: dict[str, int],
    check_row_bounds: bool = True,
) -> bool:
    if num_heads not in rule.heads:
        return False
    if check_row_bounds and (
        num_rows < rule.minimum_rows
        or (rule.maximum_rows is not None and num_rows > rule.maximum_rows)
    ):
        return False
    if state_indices_mode not in rule.state_indices_modes:
        return False
    if (
        rule.lower_bound_values is not None
        and lower_bound not in rule.lower_bound_values
    ):
        return False
    if rule.norm_eps_values is not None and norm_eps not in rule.norm_eps_values:
        return False
    return all(
        expected is None or strides[name] == expected for name, expected in rule.strides
    )


# The producer's positive-unique FP32 wave arithmetic is a fixed constant on
# every device and architecture: the generator uses the B200 SM count (148)
# directly (never a live device query) in its route-selection wave arithmetic.
# The shipped flashinfer/jit/cake_fused_kda_decode.py matched this exactly:
# `_positive_f32_variants(num_heads, num_rows)` took no sm_count parameter,
# hardcoding `sm_count = 148` internally. Route selection must stay this way
# regardless of the launching device's real SM count -- GB300/sm_103a reports
# 152, not 148 (an earlier regeneration had wrongly threaded a *live* device
# query into this function; the export's device-count guard refused the
# resulting routes). Grid SIZING for the persistent
# "stream_bf16"/"repeated_safe" families is a separate, genuinely
# device-dependent concern (the producer's own `_stream_grid_size` takes the
# live device SM count, matching `cake_fused_kda_decode_grid` below, which
# still takes a live `sm_count` parameter) and is unaffected by this fix.
_ROUTE_SM_COUNT = 148


def _positive_f32_variants(num_heads: int, num_rows: int) -> tuple[str, ...]:
    """Choose positive-unique FP32 schedules by resident-CTA wave capacity.

    The wave arithmetic below uses the producer's fixed, measured constant
    (``_ROUTE_SM_COUNT``, 148) on every device/architecture, never a live
    device query; the H=8 row bands are likewise absolute.
    """

    sm_count = _ROUTE_SM_COUNT

    if num_heads == 8:
        # Measured H=8 bands (B200 and B300 route matrices, 2026-09-25): while
        # SMs are idle the two-CTA cluster split halves each item's recurrence
        # chain (rows <= _H8_CLUSTER2_MAX_ROWS); the staged compact family
        # loses 5-16 % to high-work for rows 38..55, the vector-four producer
        # loses to the spread producer, and the nearly empty third high-work
        # wave makes wide512 the better route for rows 56..76.  Beyond that
        # the rotating high-work pipeline is at least as fast everywhere.
        # Rows 10..18 (one CTA per SM) run the LAUNCH_MIN_BLOCKS=1 instantiation of
        # the wide512 schedule ("wide512_regcap128_positive_f32"), measured
        # 3-7 % faster there (B200 2026-09-25; paired A/B 2026-10-02).
        if num_rows <= _H8_CLUSTER2_MAX_ROWS:
            return ("cluster2_wide_positive_f32",)
        if num_rows <= _H8_WIDE512_REGCAP128_MAX_ROWS:
            return ("wide512_regcap128_positive_f32",)
        if num_rows <= 37 or 56 <= num_rows <= 76:
            return ("wide512_positive_f32",)
        return ("high_work_positive_f32",)
    work_items = num_heads * num_rows
    if 2 * sm_count < work_items <= 3 * sm_count:
        return ("compact_async_positive_f32",)
    partial_wave = work_items % (2 * sm_count)
    high_work = (
        (num_heads == 32 and num_rows >= 32)
        or (work_items >= 8 * sm_count)
        or (
            work_items > 4 * sm_count
            and 0 < partial_wave <= sm_count // 2
            and (num_heads != 12 or partial_wave >= num_heads)
        )
    )
    if high_work:
        return ("high_work_positive_f32",)
    # Pair-channel producers win the measured H12/22 and H24/10 cases.
    if (
        num_heads <= 24
        and (num_heads, num_rows) not in ((12, 22), (24, 10))
        and 3 * sm_count < 2 * work_items <= 4 * sm_count
    ):
        return ("wide512_vector4_positive_f32",)
    return ("wide512_positive_f32",)


def select_cake_fused_kda_decode_variant(
    *,
    target: CakeFusedKDADecodeTarget,
    num_heads: int,
    num_rows: int,
    num_slots: int,
    state_dtype: str,
    state_indices_mode: CakeFusedKDADecodeStateIndicesMode,
    lower_bound: float | None,
    norm_eps: float,
    x_row_stride: int,
    conv_slot_stride: int,
    beta_row_stride: int,
    state_slot_stride: int,
    output_gate_row_stride: int,
    variants: Sequence[CakeFusedKDADecodeVariant] | None = None,
) -> CakeFusedKDADecodeVariant | None:
    """Select one Cake route using host scalars and tensor metadata only.

    Never queries or takes the launching device's multiprocessor count: the
    positive-unique FP32 wave bands are the producer's fixed, measured
    constant (``_ROUTE_SM_COUNT``), matching the shipped selector,
    which took no ``sm_count`` parameter either.
    """

    if target not in _TARGETS:
        raise ValueError(f"unsupported Cake fused KDA target: {target}")
    if num_heads not in _HEADS:
        raise ValueError(f"unsupported Cake fused KDA head count: {num_heads}")
    if num_rows <= 0 or num_slots <= 0:
        raise ValueError("Cake fused KDA rows and slots must be positive")
    if state_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"unsupported Cake fused KDA state dtype: {state_dtype}")
    if state_indices_mode not in _STATE_INDICES_MODES:
        raise ValueError(
            f"unsupported Cake fused KDA state_indices_mode: {state_indices_mode}"
        )
    if lower_bound is not None and (
        not math.isfinite(float(lower_bound)) or float(lower_bound) >= 0.0
    ):
        raise ValueError("lower_bound must be finite, negative, or None")
    if not math.isfinite(float(norm_eps)) or float(norm_eps) < 0.0:
        raise ValueError("norm_eps must be finite and non-negative")
    strides = {
        "x_row_stride": x_row_stride,
        "conv_slot_stride": conv_slot_stride,
        "beta_row_stride": beta_row_stride,
        "state_slot_stride": state_slot_stride,
        "output_gate_row_stride": output_gate_row_stride,
    }
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in strides.values()
    ):
        raise ValueError("Cake fused KDA strides must be non-negative integers")
    if num_rows > 2**31 - 1 or any(value > 2**31 - 1 for value in strides.values()):
        return None
    qkv_size = 3 * num_heads * 128
    required_slot_offset_bits = (
        64
        if (
            (num_slots - 1) * conv_slot_stride + 3 * qkv_size - 1 > 2**31 - 1
            or (num_slots - 1) * state_slot_stride + num_heads * 128 * 128 - 1
            > 2**31 - 1
        )
        else 32
    )
    available = (
        get_cake_fused_kda_decode_variants(target)
        if variants is None
        else tuple(variants)
    )
    positive_f32_variants = (
        _positive_f32_variants(num_heads, num_rows)
        if state_dtype == "float32" and state_indices_mode == "positive_unique"
        else None
    )
    for variant in available:
        if (
            variant.target != target
            or variant.state_dtype != state_dtype
            or variant.slot_offset_bits != required_slot_offset_bits
        ):
            continue
        if (
            positive_f32_variants is not None
            and variant.name.removesuffix("_wide_slot_offsets")
            not in positive_f32_variants
        ):
            continue
        if any(
            _eligibility_matches(
                rule,
                num_heads=num_heads,
                num_rows=num_rows,
                state_indices_mode=state_indices_mode,
                lower_bound=lower_bound,
                norm_eps=norm_eps,
                strides=strides,
                check_row_bounds=positive_f32_variants is None,
            )
            for rule in variant.eligibility
        ):
            return variant
    return None


def cake_fused_kda_decode_grid(
    variant: CakeFusedKDADecodeVariant, *, num_heads: int, num_rows: int, sm_count: int
) -> tuple[int, int, int]:
    """Launch grid of one Cake fused KDA decode call ((H, 1, 1) for repeated slots: the binding
    launches that grid once per row)."""

    if variant.abi_kind == "persistent_rows":
        resident = _PERSISTENT_RESIDENT_CTAS_PER_SM * sm_count
        return (max(1, min(num_heads * num_rows, resident)), 1, 1)
    if variant.abi_kind == "repeated_safe":
        return (num_heads, 1, 1)
    return (num_heads * variant.cluster_x, num_rows, 1)


def cake_fused_kda_decode_lower_bound_log2(lower_bound: float | None) -> float:
    """The kernel consumes the gate lower bound in log2 units (0 when the softplus mode is active)."""

    return 0.0 if lower_bound is None else float(lower_bound) * _LOG2E


def cake_fused_kda_decode_is_available() -> bool:
    """Return whether the generated Cake program registry is installed."""

    return bool(get_cake_fused_kda_decode_variants())


def get_cake_fused_kda_decode_variant(
    name: str, target: CakeFusedKDADecodeTarget
) -> CakeFusedKDADecodeVariant:
    for variant in get_cake_fused_kda_decode_variants(target):
        if variant.name == name and variant.target == target:
            return variant
    raise RuntimeError(f"Cake fused KDA source is unavailable for {name}/{target}")


def get_cake_fused_kda_decode_uri(name: str, target: CakeFusedKDADecodeTarget) -> str:
    variant = get_cake_fused_kda_decode_variant(name, target)
    return f"cake_fused_kda_decode_{name}_{variant.source_sha256[:16]}_{target}"


def render_cake_fused_kda_decode_binding(variant: CakeFusedKDADecodeVariant) -> str:
    """Render the tvm-ffi binding translation unit of one generated program.

    The template of the variant's launch ABI takes the kernel symbol, the entry
    point, the host-shim namespace (``cake_host_shim_`` plus the first 16 hex
    digits of the SHA-256 of the program name), the state dtype tokens and the
    launch constants.  The result is byte-identical to the former per-program
    binding file (``tests/kda/cake_fused_kda_decode_binding_sha256.json`` pins it).
    """

    state_ctype, state_dltype = _STATE_DTYPE_TOKENS[variant.state_dtype]
    program_digest = hashlib.sha256(variant.module.encode("utf-8")).hexdigest()
    values = {
        "KERNEL_SYMBOL": variant.kernel_symbol,
        "FFI_ENTRY": variant.ffi_entry,
        "SHIM_NAMESPACE": f"cake_host_shim_{program_digest[:16]}",
        "STATE_CTYPE": state_ctype,
        "STATE_DLTYPE": state_dltype,
        "THREADS": str(variant.threads),
        "DYNAMIC_SMEM_BYTES": str(variant.dynamic_smem_bytes),
    }
    if variant.cluster_x != 1:
        values["CLUSTER_X"] = str(variant.cluster_x)
    template = variant.binding_template.read_text(encoding="utf-8")
    unused = set(values)

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise RuntimeError(
                f"{variant.binding_template.name} needs {key!r} for {variant.name}"
            )
        unused.discard(key)
        return values[key]

    text = _BINDING_PLACEHOLDER.sub(substitute, template)
    if unused:
        raise RuntimeError(
            f"unused binding values for {variant.name}: {sorted(unused)}"
        )
    return text


def _write_rendered_binding(path: Path, text: str) -> None:
    """Write the rendered binding when its text changed.

    Explicit UTF-8: the generated header line contains an em dash.
    """

    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@functools.cache
def gen_cake_fused_kda_decode_module(
    name: str, target: CakeFusedKDADecodeTarget
) -> JitSpec:
    """Build a Cake module from its device source and its rendered binding."""

    variant = get_cake_fused_kda_decode_variant(name, target)
    csrc_dir = get_kda_csrc_dir()
    uri = get_cake_fused_kda_decode_uri(name, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_fused_kda_decode_binding.cu"
    _write_rendered_binding(binding, render_cake_fused_kda_decode_binding(variant))
    spec = gen_jit_spec(
        name=uri,
        sources=[variant.body_path, binding],
        extra_cuda_cflags=[*_TARGET_NVCC_FLAGS[target], *variant.extra_cuda_cflags],
        extra_include_paths=[csrc_dir, csrc_dir.parent, get_flashinfer_include_dir()],
        use_fast_math=name not in _PRECISE_MATH_VARIANTS,
    )
    logger.info(
        f"Generated Cake fused KDA decode {name} {target} JIT spec: {spec.name}"
    )
    return spec


@functools.cache
def load_cake_fused_kda_decode_module(name: str, target: CakeFusedKDADecodeTarget):
    module = gen_cake_fused_kda_decode_module(name, target).build_and_load()
    logger.info(f"Loaded Cake fused KDA decode {name} {target} module")
    return module


__all__ = [
    "CAKE_FUSED_KDA_DECODE_ABIS",
    "MODULES",
    "ROUTES",
    "CakeFusedKDADecodeEligibility",
    "CakeFusedKDADecodeStateIndicesMode",
    "CakeFusedKDADecodeTarget",
    "CakeFusedKDADecodeVariant",
    "cake_fused_kda_decode_grid",
    "cake_fused_kda_decode_is_available",
    "cake_fused_kda_decode_lower_bound_log2",
    "gen_cake_fused_kda_decode_module",
    "get_cake_fused_kda_decode_program_identity",
    "get_cake_fused_kda_decode_uri",
    "get_cake_fused_kda_decode_variant",
    "get_cake_fused_kda_decode_variants",
    "load_cake_fused_kda_decode_module",
    "render_cake_fused_kda_decode_binding",
    "select_cake_fused_kda_decode_variant",
]
