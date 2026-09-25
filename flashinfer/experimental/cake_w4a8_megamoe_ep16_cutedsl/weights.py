# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bit-preserving static FP4 pair layout for sm_103a W4A8 compute.

This host preparation accepts canonical packed rows. Scales are unchanged.
Repack whenever weight contents change; input and routing are not arguments.
"""


def pack_pair_major_weights(packed):
    import torch

    if packed.dtype != torch.uint8 or packed.ndim != 2 or not packed.is_contiguous():
        raise ValueError(
            "Expected a contiguous two-dimensional packed uint8 weight tensor"
        )
    rows, row_bytes = packed.shape
    if rows % 256 or row_bytes % 128:
        raise ValueError("Pair layout requires complete 256-feature and 256-K tiles")
    return (
        packed.reshape(rows // 256, 2, 128, row_bytes // 128, 2, 64)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .reshape(-1, 64)
    )


def unpack_pair_major_weights(packed, canonical_shape):
    rows, row_bytes = canonical_shape
    return (
        packed.reshape(rows // 256, row_bytes // 128, 2, 2, 128, 64)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .reshape(rows, row_bytes)
    )
