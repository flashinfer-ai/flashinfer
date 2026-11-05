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

V-split of the SM100/SM103/SM107 chunked GDN prefill kernel.

``v_split=2`` runs each (sequence, head) pair as two CTAs that own half of the
value rows of the state and of the output. The rows evolve independently, so
the output, the final states and the checkpoints must be bitwise identical to
``v_split=1`` for every state layout the kernel serves.
"""

from __future__ import annotations

import math

import pytest
import torch

import flashinfer.gdn_prefill as gdn_prefill
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import get_compute_capability

from .conftest import gen_qkv
from .reference_delta_rule import blockwise_delta_rule, exclusive_cumsum

HEAD_SIZE = 128
SCALE = 1.0 / math.sqrt(HEAD_SIZE)
HEADS = [
    pytest.param((4, 4, 8), id="gva"),
    pytest.param((8, 8, 8), id="mha"),
    pytest.param((8, 4, 4), id="gqa"),
]


def _skip_if_not_sm10x():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, _ = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("v_split runs on the SM100/SM103/SM107 prefill kernel")
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    if cuda_major < 13:
        pytest.skip(f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}")


def _inputs(seq_lens, heads, dtype, seed=0):
    hq, hk, hv = heads
    torch.manual_seed(seed)
    total = sum(seq_lens)
    ho = max(hq, hv)
    with torch.device("cuda"):
        q, k, v = gen_qkv(seq_lens, hq, hk, hv, HEAD_SIZE, dtype)
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        g = torch.rand(total, ho)
        beta = torch.rand(total, ho)
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32)
    return q, k, v, g, beta, cu_seqlens


def _states(rows, ho, dtype, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(rows, ho, HEAD_SIZE, HEAD_SIZE, generator=gen, device="cuda")
    return (0.1 * s).to(dtype)


def _nan_output(q, ho):
    return torch.full(
        (q.size(0), ho, HEAD_SIZE), float("nan"), dtype=q.dtype, device=q.device
    )


def _assert_bitwise(actual, expected, what):
    torch.testing.assert_close(
        actual, expected, atol=0, rtol=0, equal_nan=True, msg=lambda m: f"{what}: {m}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("state_dtype", [None, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_lens", [[64], [1, 93, 300], [130, 0, 257]])
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_v_split_matches_stock_packed_states(dtype, heads, seq_lens, state_dtype):
    """Packed initial/final states (and no initial state) match bitwise."""
    _skip_if_not_sm10x()
    q, k, v, g, beta, cu_seqlens = _inputs(seq_lens, heads, dtype)
    ho = max(heads[0], heads[2])
    init = None if state_dtype is None else _states(len(seq_lens), ho, state_dtype)
    results = []
    for v_split in (1, 2):
        out = _nan_output(q, ho)
        final = torch.full(
            (len(seq_lens), ho, HEAD_SIZE, HEAD_SIZE),
            float("nan"),
            dtype=state_dtype or torch.float32,
            device="cuda",
        )
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            SCALE,
            init,
            True,
            cu_seqlens,
            output=out,
            output_state=final,
            use_cp=False,
            v_split=v_split,
        )
        results.append((out, final))
    torch.cuda.synchronize()
    _assert_bitwise(results[1][0], results[0][0], "output")
    _assert_bitwise(results[1][1], results[0][1], "final state")


@torch.inference_mode()
@pytest.mark.parametrize("padded_rows", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads", HEADS)
def test_v_split_matches_stock_in_place_pool(heads, state_dtype, padded_rows):
    """An indexed state pool updated in place (one tensor read and written)."""
    _skip_if_not_sm10x()
    seq_lens = [200, 64, 0, 1]
    q, k, v, g, beta, cu_seqlens = _inputs(seq_lens, heads, torch.bfloat16)
    ho = max(heads[0], heads[2])
    pool_rows = 7
    row_elems = ho * HEAD_SIZE * HEAD_SIZE
    base = _states(pool_rows, ho, state_dtype)
    indices = torch.tensor([5, 0, 3, 6], dtype=torch.int32, device="cuda")
    results = []
    for v_split in (1, 2):
        if padded_rows:
            storage = torch.zeros(
                pool_rows, row_elems + 64, dtype=state_dtype, device="cuda"
            )
            pool = storage[:, :row_elems].view(pool_rows, ho, HEAD_SIZE, HEAD_SIZE)
            pool.copy_(base)
        else:
            pool = base.clone()
        out = _nan_output(q, ho)
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            SCALE,
            pool,
            True,
            cu_seqlens,
            output=out,
            output_state=pool,
            state_indices=indices,
            use_cp=False,
            v_split=v_split,
        )
        results.append((out, pool))
    torch.cuda.synchronize()
    _assert_bitwise(results[1][0], results[0][0], "output")
    _assert_bitwise(results[1][1], results[0][1], "state pool")


@torch.inference_mode()
@pytest.mark.parametrize("checkpoint_every_n_tokens", [64, 128])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_v_split_matches_stock_checkpoints(state_dtype, checkpoint_every_n_tokens):
    """Packed checkpoints, including one at the end of a sequence."""
    _skip_if_not_sm10x()
    seq_lens = [256, 128, 64 * 5 + 3]
    heads = (4, 4, 8)
    q, k, v, g, beta, cu_seqlens = _inputs(seq_lens, heads, torch.bfloat16)
    ho = max(heads[0], heads[2])
    counts = [n // checkpoint_every_n_tokens for n in seq_lens]
    cu_checkpoints = torch.tensor(
        exclusive_cumsum(counts), dtype=torch.int32, device="cuda"
    )
    init = _states(len(seq_lens), ho, state_dtype)
    results = []
    for v_split in (1, 2):
        out = _nan_output(q, ho)
        final = torch.full_like(init, float("nan"))
        checkpoints = torch.full(
            (sum(counts), ho, HEAD_SIZE, HEAD_SIZE),
            float("nan"),
            dtype=state_dtype,
            device="cuda",
        )
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            SCALE,
            init,
            True,
            cu_seqlens,
            output=out,
            output_state=final,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=cu_checkpoints,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            use_cp=False,
            v_split=v_split,
        )
        results.append((out, final, checkpoints))
    torch.cuda.synchronize()
    for name, got, want in zip(
        ("output", "final state", "checkpoints"), results[1], results[0], strict=True
    ):
        _assert_bitwise(got, want, name)


@torch.inference_mode()
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_v_split_matches_stock_slot_mode(state_dtype):
    """Paged slots: separate read/write rows with negative entries, checkpoint
    pool rows and unaligned prefixes (leading identity rows)."""
    _skip_if_not_sm10x()
    seq_lens = [200, 130, 192]
    prefix = [100, 0, 64]
    n = 128
    heads = (4, 4, 8)
    q, k, v, g, beta, cu_seqlens = _inputs(seq_lens, heads, torch.bfloat16)
    ho = max(heads[0], heads[2])
    counts = [(p + s) // n - p // n for p, s in zip(prefix, seq_lens, strict=True)]

    def idx(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    pool_in = _states(6, ho, state_dtype)
    checkpoint_rows = [7, -1, 0, 5, 2][: sum(counts)]
    results = []
    for v_split in (1, 2):
        out = _nan_output(q, ho)
        pool_out = torch.full_like(pool_in, float("nan"))
        checkpoints = torch.full(
            (8, ho, HEAD_SIZE, HEAD_SIZE),
            float("nan"),
            dtype=state_dtype,
            device="cuda",
        )
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            SCALE,
            pool_in,
            True,
            cu_seqlens,
            output=out,
            output_state=pool_out,
            state_indices=idx([4, -1, 2]),
            output_state_indices=idx([1, 3, -1]),
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=idx(exclusive_cumsum(counts)),
            checkpoint_every_n_tokens=n,
            checkpoint_state_indices=idx(checkpoint_rows),
            prefix_lens=idx(prefix),
            v_split=v_split,
        )
        results.append((out, pool_out, checkpoints))
    torch.cuda.synchronize()
    for name, got, want in zip(
        ("output", "final-state pool", "checkpoint pool"),
        results[1],
        results[0],
        strict=True,
    ):
        _assert_bitwise(got, want, name)


@torch.inference_mode()
@pytest.mark.parametrize("regime", ["under_filled", "between", "over_one_wave"])
def test_v_split_auto_selection(monkeypatch, regime):
    """``v_split="auto"`` passes the heuristic's factor to the kernel and
    stays bitwise identical to the stock kernel."""
    _skip_if_not_sm10x()
    heads = (8, 8, 8)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_seqs, expected = {
        "under_filled": (1, 2),
        "between": (num_sms // heads[2], 1),
        "over_one_wave": (num_sms // heads[2] + 1, 1),
    }[regime]
    seq_lens = [64 + 7 * (i % 5) for i in range(num_seqs)]
    q, k, v, g, beta, cu_seqlens = _inputs(seq_lens, heads, torch.bfloat16)
    init = _states(num_seqs, heads[2], torch.float32)

    seen = []
    kernel = gdn_prefill.chunk_gated_delta_rule_sm100

    def spy(*args, **kwargs):
        seen.append(kwargs["v_split"])
        return kernel(*args, **kwargs)

    monkeypatch.setattr(gdn_prefill, "chunk_gated_delta_rule_sm100", spy)
    results = []
    for v_split in ("auto", 1):
        out, final = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            SCALE,
            init,
            True,
            cu_seqlens,
            use_cp=False,
            v_split=v_split,
        )
        results.append((out, final))
    torch.cuda.synchronize()
    assert seen == [expected, 1]
    _assert_bitwise(results[0][0], results[1][0], "output")
    _assert_bitwise(results[0][1], results[1][1], "final state")


@torch.inference_mode()
@pytest.mark.parametrize("heads", HEADS)
def test_v_split_matches_reference(heads):
    _skip_if_not_sm10x()
    seq_lens = [64, 128, 93]
    q, k, v, g, beta, cu_seqlens = _inputs(seq_lens, heads, torch.bfloat16)
    out, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        SCALE,
        None,
        True,
        cu_seqlens,
        use_cp=False,
        v_split=2,
    )
    torch.cuda.synchronize()
    ref_o, ref_state = blockwise_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        seq_lens,
        scale_factor=SCALE,
        alpha=g,
        beta=beta,
        state_dtype=torch.float32,
    )
    torch.testing.assert_close(out, ref_o.to(q.dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final.transpose(-1, -2), ref_state, atol=5e-3, rtol=1e-3)
