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

import pytest
import torch

from flashinfer.comm import (
    merge_ulysses_output_head_chunk,
    pack_ulysses_qkv_head_chunk,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _reference_pack(q, k, v, world_size, head_offset, head_count):
    local_heads = q.shape[2] // world_size
    planes = []
    for tensor in (q, k, v):
        bands = [
            tensor[
                :,
                :,
                rank * local_heads + head_offset : rank * local_heads
                + head_offset
                + head_count,
                :,
            ]
            for rank in range(world_size)
        ]
        planes.append(torch.cat(bands, dim=2))
    return torch.cat(planes, dim=-1).contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("schedule", [(3, 8, 3), (5, 5, 4)])
def test_pack_qkv_uneven_schedule_noncontiguous_views(dtype, schedule):
    B, S, W, H_LOCAL, D = 2, 11, 4, 14, 32
    projection = torch.randn(
        B,
        S,
        W * H_LOCAL,
        3,
        D,
        device="cuda",
        dtype=dtype,
    )
    query, key, value = projection.unbind(dim=3)
    assert not query.is_contiguous()

    head_offset = 0
    for head_count in schedule:
        out = torch.empty(
            B,
            S,
            W * head_count,
            3 * D,
            device=query.device,
            dtype=query.dtype,
        )
        returned = pack_ulysses_qkv_head_chunk(
            query,
            key,
            value,
            world_size=W,
            head_offset=head_offset,
            head_count=head_count,
            out=out,
        )
        expected = _reference_pack(query, key, value, W, head_offset, head_count)
        assert returned is out
        assert torch.equal(out, expected)
        head_offset += head_count


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("schedule", [(3, 8, 3), (5, 5, 4)])
def test_merge_output_reconstructs_uneven_schedule_on_nondefault_stream(
    dtype, schedule
):
    B, S, W, H_LOCAL, D = 2, 13, 4, 14, 32
    full = torch.randn(B, S, W * H_LOCAL, D, device="cuda", dtype=dtype)
    reconstructed = torch.full_like(full, float("nan"))
    stream = torch.cuda.Stream()
    offset = 0
    with torch.cuda.stream(stream):
        for head_count in schedule:
            compact = torch.cat(
                [
                    full[
                        :,
                        :,
                        rank * H_LOCAL + offset : rank * H_LOCAL + offset + head_count,
                        :,
                    ]
                    for rank in range(W)
                ],
                dim=2,
            ).contiguous()
            returned = merge_ulysses_output_head_chunk(
                compact,
                world_size=W,
                local_heads=H_LOCAL,
                head_offset=offset,
                out=reconstructed,
            )
            assert returned is reconstructed
            offset += head_count
    stream.synchronize()
    assert torch.equal(reconstructed, full)


def test_head_chunk_validation():
    qkv = torch.randn(1, 4, 8, 3, 16, device="cuda", dtype=torch.float16)
    q, k, v = qkv.unbind(dim=3)
    with pytest.raises(ValueError, match="divisible"):
        pack_ulysses_qkv_head_chunk(
            q,
            k,
            v,
            world_size=3,
            head_offset=0,
            head_count=1,
        )
    with pytest.raises(ValueError, match="exceeds local_heads"):
        pack_ulysses_qkv_head_chunk(
            q,
            k,
            v,
            world_size=2,
            head_offset=3,
            head_count=2,
        )
    with pytest.raises(ValueError, match="does not match query"):
        pack_ulysses_qkv_head_chunk(
            q,
            k[:, :, :-1],
            v,
            world_size=2,
            head_offset=0,
            head_count=1,
        )
    with pytest.raises(ValueError, match="must not alias"):
        pack_ulysses_qkv_head_chunk(
            q,
            k,
            v,
            world_size=2,
            head_offset=0,
            head_count=4,
            out=qkv.view(1, 4, 8, 3 * 16),
        )

    compact = torch.randn(1, 4, 4, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="expected"):
        merge_ulysses_output_head_chunk(
            compact,
            world_size=2,
            local_heads=4,
            head_offset=0,
            out=torch.empty(1, 4, 7, 16, device="cuda", dtype=torch.float16),
        )
