# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import numpy as np
import pytest
import torch

from flashinfer.swa_indices import compute_swa_indices_and_lens


def _ref_compute_swa_indices_and_lens(
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    window_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    token_offset: int,
    num_tokens: int,
) -> None:
    qsl = query_start_loc.cpu().numpy()
    sl = seq_lens.cpu().numpy()
    t2r = token_to_req_indices.cpu().numpy()
    iv = is_valid_token.cpu().numpy()
    bt = block_table.cpu().numpy()
    W = window_size

    # Snapshot the buffers so invalid rows + tail columns keep the caller's fill.
    last_dim = swa_indices.size(-1)
    flat_idx = swa_indices.view(num_tokens, last_dim)
    work_idx = flat_idx.cpu().numpy().copy()
    work_len = swa_lens.cpu().numpy().copy()

    for pid in range(num_tokens):
        token_idx = pid + token_offset
        if not bool(iv[token_idx]):
            work_len[pid] = 0
            continue
        r = int(t2r[token_idx])
        query_start = int(qsl[r])
        query_end = int(qsl[r + 1])
        query_len = query_end - query_start
        seq_len = int(sl[r])
        prefix_len = seq_len - query_len
        pos = prefix_len + token_idx - query_start
        start_pos = max(pos - W + 1, 0)
        swa_len = (pos + 1) - start_pos
        work_len[pid] = swa_len
        for offset in range(W):
            if offset < swa_len:
                p = start_pos + offset
                block_number = int(bt[r, p // block_size])
                work_idx[pid, offset] = block_number * block_size + (p % block_size)
            else:
                work_idx[pid, offset] = -1

    flat_idx.copy_(torch.from_numpy(work_idx).to(swa_indices.device))
    swa_lens.copy_(torch.from_numpy(work_len).to(swa_lens.device))


def _build_inputs(
    num_reqs: int,
    query_lens: list[int],
    prefix_lens: list[int],
    window_size: int,
    block_size: int,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    assert len(query_lens) == num_reqs and len(prefix_lens) == num_reqs

    query_start_loc = np.concatenate(([0], np.cumsum(query_lens))).astype(np.int32)
    seq_lens = (np.array(prefix_lens) + np.array(query_lens)).astype(np.int32)
    T = int(query_start_loc[-1])
    token_to_req_indices = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    is_valid_token = rng.random(T) > 0.1  # ~10% invalid exercises the early-exit path
    max_blocks = int(np.ceil(seq_lens.max() / block_size)) + 4
    # Non-zero block numbers so slot-id bugs surface as drift, not lucky zeros.
    block_table = rng.integers(100, 10_000, size=(num_reqs, max_blocks), dtype=np.int32)

    device = torch.device("cuda")
    return {
        "window_size": window_size,
        "block_size": block_size,
        "num_tokens": T,
        "query_start_loc": torch.from_numpy(query_start_loc).to(device),
        "seq_lens": torch.from_numpy(seq_lens).to(device),
        "token_to_req_indices": torch.from_numpy(token_to_req_indices).to(device),
        "is_valid_token": torch.from_numpy(is_valid_token).to(device),
        "block_table": torch.from_numpy(block_table).to(device),
    }


@pytest.mark.parametrize(
    "num_reqs,query_lens,prefix_lens,window_size,block_size",
    [
        # Prefill batch: a few short requests with mixed prefix lengths.
        (3, [4, 7, 5], [0, 11, 30], 64, 16),
        # Decode-like step: one query per request.
        (8, [1] * 8, [10, 250, 17, 500, 64, 0, 129, 88], 128, 16),
        # Window larger than any prefix-aware position (forces start_pos = 0).
        (2, [3, 3], [0, 1], 256, 32),
        # block_size == 64 (DSv4-Flash production).
        (4, [2, 1, 5, 3], [13, 7, 200, 1023], 128, 64),
        # Mid-sized batch.
        (16, [1] * 16, list(range(0, 16_000, 1000)), 256, 64),
    ],
)
def test_compute_swa_indices_and_lens_2d(
    num_reqs, query_lens, prefix_lens, window_size, block_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    inputs = _build_inputs(
        num_reqs, query_lens, prefix_lens, window_size, block_size, seed=42
    )
    T = inputs["num_tokens"]
    device = torch.device("cuda")

    pad = 8  # oversize last dim so we can assert nothing is written past W
    swa_indices = torch.full(
        (T, window_size + pad), -99, dtype=torch.int32, device=device
    )
    swa_lens = torch.full((T,), -99, dtype=torch.int32, device=device)

    compute_swa_indices_and_lens(
        swa_indices,
        swa_lens,
        window_size,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req_indices"],
        inputs["is_valid_token"],
        inputs["block_table"],
        block_size,
        token_offset=0,
        num_tokens=T,
    )

    ref_indices = torch.full_like(swa_indices, -99)
    ref_lens = torch.full_like(swa_lens, -99)
    _ref_compute_swa_indices_and_lens(
        ref_indices,
        ref_lens,
        window_size,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req_indices"],
        inputs["is_valid_token"],
        inputs["block_table"],
        block_size,
        token_offset=0,
        num_tokens=T,
    )

    assert (swa_lens != -99).all(), "swa_lens has un-written entries"

    torch.testing.assert_close(swa_lens, ref_lens)
    torch.testing.assert_close(
        swa_indices[:, :window_size], ref_indices[:, :window_size]
    )
    assert (swa_indices[:, window_size:] == -99).all(), (
        "kernel wrote beyond window_size"
    )

    valid_mask = inputs["is_valid_token"].cpu().numpy()
    for pid in range(T):
        if not bool(valid_mask[pid]):
            assert swa_lens[pid].item() == 0
            assert (swa_indices[pid, :window_size] == -99).all(), (
                f"invalid row pid={pid} had its window slice written"
            )


def test_compute_swa_indices_and_lens_3d_buffer():
    # Callers carrying a singleton s_q dim pass swa_indices as [N, 1, W].
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    inputs = _build_inputs(
        num_reqs=4,
        query_lens=[3, 1, 2, 4],
        prefix_lens=[50, 7, 200, 0],
        window_size=64,
        block_size=16,
        seed=7,
    )
    T = inputs["num_tokens"]
    W = inputs["window_size"]
    device = torch.device("cuda")

    swa_indices_3d = torch.full((T, 1, W), -99, dtype=torch.int32, device=device)
    swa_lens = torch.zeros((T,), dtype=torch.int32, device=device)

    compute_swa_indices_and_lens(
        swa_indices_3d,
        swa_lens,
        W,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req_indices"],
        inputs["is_valid_token"],
        inputs["block_table"],
        inputs["block_size"],
        token_offset=0,
        num_tokens=T,
    )

    ref_indices = torch.full((T, W), -99, dtype=torch.int32, device=device)
    ref_lens = torch.zeros_like(swa_lens)
    _ref_compute_swa_indices_and_lens(
        ref_indices.unsqueeze(1),
        ref_lens,
        W,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req_indices"],
        inputs["is_valid_token"],
        inputs["block_table"],
        inputs["block_size"],
        token_offset=0,
        num_tokens=T,
    )
    torch.testing.assert_close(swa_lens, ref_lens)
    torch.testing.assert_close(swa_indices_3d.squeeze(1), ref_indices)


def test_compute_swa_indices_and_lens_token_offset():
    # Two chunked calls (offset=0/n=12 then offset=12/n=20) must reproduce a
    # single offset=0/n=32 call.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    inputs = _build_inputs(
        num_reqs=6,
        query_lens=[5, 7, 4, 6, 3, 7],
        prefix_lens=[0, 64, 128, 11, 47, 200],
        window_size=128,
        block_size=16,
        seed=11,
    )
    T = inputs["num_tokens"]
    assert T == 32
    W = inputs["window_size"]
    device = torch.device("cuda")

    full_idx = torch.full((T, W), -99, dtype=torch.int32, device=device)
    full_len = torch.zeros((T,), dtype=torch.int32, device=device)
    compute_swa_indices_and_lens(
        full_idx,
        full_len,
        W,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req_indices"],
        inputs["is_valid_token"],
        inputs["block_table"],
        inputs["block_size"],
        token_offset=0,
        num_tokens=T,
    )

    # Match the one-shot fill so invalid (untouched) rows compare equal.
    chunk_idx = torch.full((T, W), -99, dtype=torch.int32, device=device)
    chunk_len = torch.zeros((T,), dtype=torch.int32, device=device)
    for offset, n in [(0, 12), (12, 20)]:
        compute_swa_indices_and_lens(
            chunk_idx[offset : offset + n],
            chunk_len[offset : offset + n],
            W,
            inputs["query_start_loc"],
            inputs["seq_lens"],
            inputs["token_to_req_indices"],
            inputs["is_valid_token"],
            inputs["block_table"],
            inputs["block_size"],
            token_offset=offset,
            num_tokens=n,
        )

    torch.testing.assert_close(chunk_len, full_len)
    torch.testing.assert_close(chunk_idx, full_idx)


def test_compute_swa_indices_and_lens_zero_tokens():
    # num_tokens == 0 must be a no-op (no kernel launch, no writes).
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    W = 32
    swa_indices = torch.full((1, W), 7, dtype=torch.int32, device=device)
    swa_lens = torch.full((1,), 13, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, 1], dtype=torch.int32, device=device)
    sl = torch.tensor([1], dtype=torch.int32, device=device)
    t2r = torch.tensor([0], dtype=torch.int32, device=device)
    iv = torch.tensor([True], dtype=torch.bool, device=device)
    bt = torch.tensor([[42]], dtype=torch.int32, device=device)

    compute_swa_indices_and_lens(
        swa_indices,
        swa_lens,
        W,
        qsl,
        sl,
        t2r,
        iv,
        bt,
        block_size=16,
        token_offset=0,
        num_tokens=0,
    )
    # Buffers untouched.
    assert (swa_indices == 7).all()
    assert (swa_lens == 13).all()
