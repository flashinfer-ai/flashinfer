# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration for the FMHA decode TS kernel.

The class FmhaDecodeConfig encapsulates static configuration parameters
that should be set before kernel compilation.
"""

from functools import wraps

from cutlass import BFloat16, Float16, Float8E4M3FN


_D192_GROUPED_KEEPS_PROFILES = {
    (dtype, dtype, dtype, output, 192, stage, 1, 1)
    for dtype, output in (
        (Float16, Float16),
        (BFloat16, BFloat16),
        (Float8E4M3FN, Float16),
        (Float8E4M3FN, BFloat16),
        (Float8E4M3FN, Float8E4M3FN),
    )
    for stage in ((128, 192) if dtype in (Float16, BFloat16) else (128,))
}


# Sparse route unions reuse the dense Keeps staging: D64/D128 use two
# unstaged K/V instances; D256 uses one instance with D128 bands or a full
# D256 FP8 stage assembled from 128-byte swizzled chunks.
_Q_TOKEN_KV_BLOCK_SPARSE_GROUPED_KEEPS_PROFILES = {
    (
        dtype,
        dtype,
        dtype,
        output,
        dim,
        128 if dim == 256 else 0,
        1 if dim == 256 else 2,
        1 if dim == 256 else 2,
    )
    for dim in (64, 128, 256)
    for dtype, output in (
        (Float16, Float16),
        (BFloat16, BFloat16),
        (Float8E4M3FN, Float16),
        (Float8E4M3FN, BFloat16),
    )
}
_Q_TOKEN_KV_BLOCK_SPARSE_GROUPED_KEEPS_PROFILES.update(
    (Float8E4M3FN, Float8E4M3FN, Float8E4M3FN, output, 256, 256, 1, 1)
    for output in (Float16, BFloat16)
)
# Full-D256 BF16 stages exist only as the NVFP4 Keeps transform target (see
# FmhaDecodeConfig.uses_full_d256_keeps_stage).
_Q_TOKEN_KV_BLOCK_SPARSE_GROUPED_KEEPS_PROFILES.add(
    (BFloat16, BFloat16, BFloat16, BFloat16, 256, 256, 1, 1)
)


def _accept_legacy_dtype_aliases(cls):
    """Keep legacy constructor keywords outside the canonical dataclass fields."""
    init = cls.__init__

    @wraps(init)
    def init_with_dtype_aliases(self, *args, qkv_dtype=None, kv_dtype=None, **kwargs):
        if qkv_dtype is not None:
            kwargs.setdefault("q_dtype", qkv_dtype)
            kwargs.setdefault("k_dtype", kv_dtype or qkv_dtype)
            kwargs.setdefault("v_dtype", kv_dtype or qkv_dtype)
        elif kv_dtype is not None:
            kwargs.setdefault("k_dtype", kv_dtype)
            kwargs.setdefault("v_dtype", kv_dtype)
        init(self, *args, **kwargs)

    cls.__init__ = init_with_dtype_aliases
    return cls
