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

"""Byte layout for the fixed EP16 distributed workspace."""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class WorkspaceRegion:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


class WorkspaceLayout:
    def __init__(self, token_capacity: int = 384, ring_tokens: int = 41088):
        if type(token_capacity) is not int or type(ring_tokens) is not int:
            raise TypeError("workspace dimensions must be integers")
        if token_capacity <= 0 or token_capacity % 384:
            raise ValueError("token_capacity must be a positive multiple of 384")
        if ring_tokens <= 0 or ring_tokens % 384:
            raise ValueError("ring_tokens must be a positive multiple of 384")
        self.token_capacity = token_capacity
        self.ring_tokens = ring_tokens
        self.ring_blocks = ring_tokens // 8
        self.sf_ring_tokens = max(
            (ring_tokens // m) * ((m + 127) // 128) * 128
            for m in (8, 16, 32, 64, 96, 128, 192)
        )
        self.index_stride = 16 * token_capacity
        self.pool_tokens = ((16 * token_capacity * 8 + 32 * 191 + 383) // 384) * 384
        self.regions: dict[str, WorkspaceRegion] = {}
        sizes = {
            "uint8": 1,
            "bfloat16": 2,
            "uint32": 4,
            "float32": 4,
            "uint64": 8,
            "int64": 8,
        }

        def add(name, offset, shape, dtype):
            size = prod(shape) * sizes[dtype]
            self.regions[name] = WorkspaceRegion(offset, size, shape, dtype)
            return offset + size

        add("grid", 0, (4,), "uint32")
        add("status", 16, (1,), "uint32")
        add("signals", 20, (2,), "uint32")
        add("claims", 28, (2,), "uint32")
        add("shared_claims", 36, (2,), "uint32")
        cursor = 128
        cursor = add("send", cursor, (512,), "uint64")
        cursor = add("recv", cursor, (16, 32), "uint64")
        cursor = add("totals", cursor, (32,), "uint64")
        for name in ("l1_full", "l1_empty", "l2_full", "l2_empty"):
            cursor = add(name, cursor, (self.ring_blocks,), "uint32")
        cursor = add("shared_l2_full", cursor, ((token_capacity + 7) // 8,), "uint32")
        cursor = add("indices", cursor, (32, 16, self.index_stride), "uint32")
        cursor = add("metadata", cursor, (self.pool_tokens, 3), "uint32")
        self.workspace_bytes = ((cursor + 15) // 16) * 16
        cursor = self.workspace_bytes
        for name, shape, dtype in (
            ("input", (token_capacity, 3072), "uint8"),
            ("input_sf", (token_capacity, 24), "uint32"),
            ("ids", (token_capacity, 8), "int64"),
            ("route_weights", (token_capacity, 8), "float32"),
            ("l1", (ring_tokens, 3072), "uint8"),
            ("l1_sf", (24, self.sf_ring_tokens), "uint32"),
            ("l1_weights", (ring_tokens,), "float32"),
            ("l2", (ring_tokens, 5120), "uint8"),
            ("l2_sf", (40, self.sf_ring_tokens), "uint32"),
            ("combine", (8, token_capacity, 3072), "bfloat16"),
        ):
            cursor = add(name, cursor, shape, dtype)
        self.nbytes = cursor

    def view(self, buffer, name):
        """Create a zero-copy typed view of a caller-owned symmetric buffer."""
        import torch

        if (
            buffer.dtype not in (torch.uint8, torch.int8)
            or buffer.ndim != 1
            or not buffer.is_contiguous()
            or buffer.numel() != self.nbytes
        ):
            raise ValueError(
                "expected one contiguous byte buffer of exact workspace size"
            )
        region = self.regions[name]
        return (
            buffer.narrow(0, region.offset, region.nbytes)
            .view(getattr(torch, region.dtype))
            .reshape(region.shape)
        )
