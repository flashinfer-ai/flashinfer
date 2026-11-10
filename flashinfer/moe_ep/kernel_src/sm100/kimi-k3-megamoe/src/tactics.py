# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

_TAC_C4_PI_E21: Tuple = (  # NVFP4, FP8 combine: C4 + C2 fallback, phase_interleave, transfer-warp token-back
    [256, 256, 256], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 1, (2, 1),
)

_TAC_N224_C4_PI_E21: Tuple = (
    [256, 224, 256], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 1, (2, 1),
)

_TAC_N240_C4_PI_E21: Tuple = (
    [256, 240, 256], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 1, (2, 1),
)

_TAC_MX_N240_C4PI_E21: Tuple = (
    [256, 240, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 4, (2, 1),
)

_TAC_BEST_E21_S1: Tuple = (  # NVFP4, BF16 combine: bulk FC2 (1 TMA stage), epi-warp token-back
    [256, 256, 256], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 1, 4, (2, 1),
)

_TAC_MX_C4PI_E21: Tuple = (  # MXFP4 x MXFP8, FP8 combine (K=128: unpacked fp4 weights, 6 AB stages with mx+sR)
    [256, 256, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 4, (2, 1),
)

_TAC_MX_BULK_C4PI_E21_S2: Tuple = (  # MXFP4 x MXFP8, BF16 combine: bulk FC2 (2 TMA stages), epi-warp token-back
    [256, 256, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 2, 4, (2, 1),
)

_TAC_MX_N224_C4PI_E21: Tuple = (
    [256, 224, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 4, (2, 1),
)

_TAC_MX_BULK_N224_E21_S2: Tuple = (
    [256, 224, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 2, 4, (2, 1),
)

_TAC_MX_BULK_N240_E21_S2: Tuple = (
    [256, 240, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 2, 4, (2, 1),
)

_TAC_BULK_N224_E21_S1: Tuple = (
    [256, 224, 256], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 1, 4, (2, 1),
)

_TAC_BULK_N240_E21_S1: Tuple = (
    [256, 240, 256], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 1, 4, (2, 1),
)

_TAC_MX_C4G512_E21: Tuple = (  # MX BF16 runner-up (non-bulk, grouped 512)
    [256, 256, 128], [4, 1, 1], [2, 1, 1], ("grouped", 512), "atomic_counter",
    "reuse_dispatch_warps", False, None, 4, (2, 1),
)

_TAC_MX_N128_SMALL: Tuple = (
    [256, 128, 128], [2, 1, 1], None, ("grouped", 512), "atomic_counter",
    "epi_warps", True, 1, 4, (1, 1),
)

_TAC_NV_BF16_N128_SMALL: Tuple = (
    [256, 128, 256], [2, 1, 1], None, ("grouped", 512), "grid_stride",
    "epi_warps", True, 1, 1, (1, 1),
)

_TAC_MX_C4PI_E21_F1: Tuple = (  # MX FP8 512: N = 256 (N = 224 / 240 mixed at u512), flag batch 1
    [256, 256, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 1, (2, 1),
)

_TAC_MX_N224_C4PI_E21_F1: Tuple = (  # MX FP8 1024: N = 224, flag batch 1
    [256, 224, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "reuse_dispatch_warps", False, None, 1, (2, 1),
)

_TAC_MX_BULK_C4PI_E21_S2_F1: Tuple = (  # MX BF16 512: bulk N = 256, flag batch 1
    [256, 256, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 2, 1, (2, 1),
)

_TAC_MX_BULK_N224_E21_S2_F1: Tuple = (  # MX BF16 1024: bulk N = 224, flag batch 1
    [256, 224, 128], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 2, 1, (2, 1),
)

_TAC_BULK_N240_E21_S1_F1: Tuple = (  # NVFP4 BF16 512 / 1024: bulk N = 240, flag batch 1
    [256, 240, 256], [4, 1, 1], [2, 1, 1], ("phase_interleave", None), "atomic_counter",
    "epi_warps", True, 1, 1, (2, 1),
)

PROD1_TOKEN_BUCKETS: Tuple[int, ...] = (256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)  # [build 7] + 384

PROD1_DEFAULT_TACTICS: dict = {
    ("nvfp4", "32e4m3xe8m0", 256): _TAC_C4_PI_E21,
    ("nvfp4", "32e4m3xe8m0", 384): _TAC_C4_PI_E21,  # [build 7] rung added for the other configs; unchanged here
    ("nvfp4", "32e4m3xe8m0", 512): _TAC_N240_C4_PI_E21,  # [build 9] N = 240 (was N = 256)
    ("nvfp4", "32e4m3xe8m0", 1024): _TAC_N240_C4_PI_E21,  # [build 9] N = 240 (was N = 256)
    ("nvfp4", "32e4m3xe8m0", 2048): _TAC_N240_C4_PI_E21,  # s24: N240 + TBX/DB + async FC2 flags at 2k too
    ("nvfp4", "32e4m3xe8m0", 4096): _TAC_N240_C4_PI_E21,  # + TBX / DB (enabled by the kernel at NVFP4 N = 240)
    ("nvfp4", "32e4m3xe8m0", 8192): _TAC_N240_C4_PI_E21,
    ("nvfp4", "32e4m3xe8m0", 16384): _TAC_N240_C4_PI_E21,
    ("nvfp4", "32e4m3xe8m0", 32768): _TAC_C4_PI_E21,
    ("nvfp4", "32e4m3xe8m0", 65536): _TAC_C4_PI_E21,
    ("nvfp4", "bf16", 256): _TAC_NV_BF16_N128_SMALL,  # [build 7]
    ("nvfp4", "bf16", 384): _TAC_NV_BF16_N128_SMALL,  # [build 7]
    ("nvfp4", "bf16", 512): _TAC_BULK_N240_E21_S1_F1,  # [build 9]
    ("nvfp4", "bf16", 1024): _TAC_BULK_N240_E21_S1_F1,  # [build 9]
    ("nvfp4", "bf16", 2048): _TAC_BULK_N224_E21_S1,
    ("nvfp4", "bf16", 4096): _TAC_BEST_E21_S1,
    ("nvfp4", "bf16", 8192): _TAC_BULK_N240_E21_S1,
    ("nvfp4", "bf16", 16384): _TAC_BEST_E21_S1,
    ("nvfp4", "bf16", 32768): _TAC_BEST_E21_S1,
    ("nvfp4", "bf16", 65536): _TAC_BEST_E21_S1,
    ("mxfp4_mxfp8", "32e4m3xe8m0", 256): _TAC_MX_N128_SMALL,  # [build 7]
    ("mxfp4_mxfp8", "32e4m3xe8m0", 384): _TAC_MX_N128_SMALL,  # [build 7]
    ("mxfp4_mxfp8", "32e4m3xe8m0", 512): _TAC_MX_C4PI_E21_F1,  # [build 9]
    ("mxfp4_mxfp8", "32e4m3xe8m0", 1024): _TAC_MX_N224_C4PI_E21_F1,  # [build 9]
    ("mxfp4_mxfp8", "32e4m3xe8m0", 2048): _TAC_MX_N224_C4PI_E21,
    ("mxfp4_mxfp8", "32e4m3xe8m0", 4096): _TAC_MX_N240_C4PI_E21,
    ("mxfp4_mxfp8", "32e4m3xe8m0", 8192): _TAC_MX_N240_C4PI_E21,
    ("mxfp4_mxfp8", "32e4m3xe8m0", 16384): _TAC_MX_N240_C4PI_E21,
    ("mxfp4_mxfp8", "32e4m3xe8m0", 32768): _TAC_MX_C4PI_E21,
    ("mxfp4_mxfp8", "32e4m3xe8m0", 65536): _TAC_MX_C4PI_E21,
    ("mxfp4_mxfp8", "bf16", 256): _TAC_MX_N128_SMALL,  # [build 7]
    ("mxfp4_mxfp8", "bf16", 384): _TAC_MX_N128_SMALL,  # [build 7]
    ("mxfp4_mxfp8", "bf16", 512): _TAC_MX_BULK_C4PI_E21_S2_F1,  # [build 9]
    ("mxfp4_mxfp8", "bf16", 1024): _TAC_MX_BULK_N224_E21_S2_F1,  # [build 9]
    ("mxfp4_mxfp8", "bf16", 2048): _TAC_MX_BULK_N224_E21_S2,
    ("mxfp4_mxfp8", "bf16", 4096): _TAC_MX_BULK_N240_E21_S2,
    ("mxfp4_mxfp8", "bf16", 8192): _TAC_MX_BULK_N240_E21_S2,
    ("mxfp4_mxfp8", "bf16", 16384): _TAC_MX_BULK_N240_E21_S2,
    ("mxfp4_mxfp8", "bf16", 32768): _TAC_MX_BULK_C4PI_E21_S2,
    ("mxfp4_mxfp8", "bf16", 65536): _TAC_MX_BULK_C4PI_E21_S2,
}
