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

import math

import pytest
import torch

from flashinfer import sm110_gqa_decode
from flashinfer.experimental.sm110_gqa_decode.jit import _read_manifest


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    q_scale: float,
) -> torch.Tensor:
    batch, _, head_dim = q.shape
    capacity = int(kv.shape[-2])
    grouped_q = q.float().view(batch, 8, 4, head_dim)
    k = kv[:, 0].float()
    v = kv[:, 1].float()
    scores = torch.einsum("bhgd,bhkd->bhgk", grouped_q, k)
    positions = torch.arange(capacity, device=q.device)
    valid = positions.view(1, 1, 1, capacity) < sequence_lengths.view(batch, 1, 1, 1)
    scores.masked_fill_(~valid, -torch.inf)
    probabilities = torch.softmax(scores * q_scale / math.sqrt(head_dim), dim=-1)
    return torch.einsum("bhgk,bhkd->bhgd", probabilities, v).reshape_as(q).half()


def _has_sm110() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0)


def test_public_entry_point_is_experimental() -> None:
    assert sm110_gqa_decode.is_experimental


def test_generated_source_closure() -> None:
    _, manifest = _read_manifest()
    assert manifest["architecture"] == "sm_110a"
    assert manifest["contract"]["gqa_group_size"] == 4
    assert {route["ffi_entry"] for route in manifest["routes"]} == {
        "run_short",
        "run_long",
    }


@pytest.mark.skipif(not _has_sm110(), reason="requires an exact SM110 GPU")
@pytest.mark.parametrize(
    ("batch", "capacity", "lengths", "seed"),
    [
        (1, 64, [1], 95601),
        (4, 256, [64, 127, 191, 256], 95602),
        (1, 1024, [1024], 95603),
        (1, 4096, [3968], 95604),
    ],
)
def test_sm110_gqa_decode_matches_reference(
    batch: int,
    capacity: int,
    lengths: list[int],
    seed: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch,
        32,
        128,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    kv = torch.randn(
        batch,
        2,
        8,
        capacity,
        128,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    sequence_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    expected = _reference(q, kv, sequence_lengths, q_scale=1.0)
    q_before = q.clone()
    kv_before = kv.clone()
    lengths_before = sequence_lengths.clone()
    out = torch.empty_like(q)

    actual = sm110_gqa_decode(
        q,
        kv,
        sequence_lengths,
        out=out,
    )
    torch.cuda.synchronize()

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    assert torch.isfinite(actual).all()
    assert torch.equal(q, q_before)
    assert torch.equal(kv, kv_before)
    assert torch.equal(sequence_lengths, lengths_before)
