# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper-level tests for the cuTile ragged-prefill backend
# (BatchPrefillWithRaggedKVCacheWrapper(backend="cutile")), checked against fa2.
# Unlike test_fmha_prefill_bsr_cutile.py (which loads the kernel directly), these
# exercise the full plan()/run() wiring in flashinfer/prefill.py.

import math

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")


def _cutile_available():
    try:
        from flashinfer.cutile.cutile_common import is_cuda_tile_available

        return is_cuda_tile_available()
    except Exception:
        return False


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


pytestmark = pytest.mark.skipif(
    not (_cutile_available() and _is_blackwell()),
    reason="cuTile ragged prefill requires cuda.tile and a Blackwell (SM100+) GPU",
)


def _run(backend, qo_indptr, kv_indptr, q, k, v, nqo, nkv, hd, causal):
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend=backend)
    w.plan(qo_indptr, kv_indptr, nqo, nkv, hd, causal=causal, q_data_type=q.dtype)
    return w.run(q, k, v, return_lse=True)


@pytest.mark.parametrize(
    "seq_lens,nqo,nkv",
    [
        ([128, 128], 4, 4),  # MHA, multi-batch
        ([256], 8, 2),  # GQA 4:1
        ([128, 256, 384], 4, 4),  # variable length
        ([512, 512], 16, 1),  # GQA 16:1
    ],
)
@pytest.mark.parametrize("causal", [True, False])
def test_ragged_prefill_cutile_matches_fa2(seq_lens, nqo, nkv, causal):
    dev, dt, hd = "cuda", torch.bfloat16, 128
    torch.manual_seed(0)
    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=dev
    )
    kv_indptr = qo_indptr.clone()  # cuTile ragged requires qo_indptr == kv_indptr
    total = int(qo_indptr[-1])
    q = torch.randn(total, nqo, hd, dtype=dt, device=dev)
    k = torch.randn(total, nkv, hd, dtype=dt, device=dev)
    v = torch.randn(total, nkv, hd, dtype=dt, device=dev)

    o_ct, lse_ct = _run("cutile", qo_indptr, kv_indptr, q, k, v, nqo, nkv, hd, causal)
    o_fa, lse_fa = _run("fa2", qo_indptr, kv_indptr, q, k, v, nqo, nkv, hd, causal)

    assert o_ct.shape == (total, nqo, hd)
    assert not torch.isnan(o_ct).any()
    rel = (o_ct.float() - o_fa.float()).abs().max() / (o_fa.float().abs().max() + 1e-6)
    assert rel < 2e-2, f"output rel error {rel:.4f} too large"
    # cuTile and fa2 must agree on FlashInfer's base-2 LSE convention.
    lse_diff = (lse_ct.float() - lse_fa.float()).abs().max()
    assert lse_diff < 5e-2, f"lse diff {lse_diff:.4f} too large (base convention?)"


def test_ragged_prefill_cutile_rejects_unequal_indptr():
    """qo_indptr != kv_indptr (append/chunked prefill) must raise, not miscompute."""
    dev, dt, hd = "cuda", torch.bfloat16, 128
    qo_indptr = torch.tensor([0, 64], dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor([0, 128], dtype=torch.int32, device=dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="cutile")
    with pytest.raises(NotImplementedError):
        w.plan(qo_indptr, kv_indptr, 4, 4, hd, causal=True, q_data_type=dt)
