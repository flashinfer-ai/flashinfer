# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""FP16 follows the same paged auto placement rules as BF16."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import flashinfer.prefill as fp

pytestmark = pytest.mark.cpu_only


def kwargs(dim, qs, ks, group=4, page=16):
    return dict(
        device=torch.device("cuda"),
        layout="HND",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        head_dim_qk=dim,
        head_dim_vo=dim,
        batch=len(qs),
        hq=32,
        hk=32 // group,
        max_q=max(qs),
        max_kv=max(ks),
        page_size=page,
        total_q=sum(qs),
        qo_indptr_host=torch.tensor([0, *accumulate(qs)], dtype=torch.int32),
        kv_lens=torch.tensor(ks, dtype=torch.int32),
        seq_lens_q=None,
    )


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (9, 0), (10, 7), (12, 0)])
@pytest.mark.parametrize(
    "dim,qs,ks,group,page",
    [
        (128, [128], [8192], 4, 16),
        (128, [768] * 6, [4096] * 6, 2, 16),
        (128, [2048], [8192], 16, 16),
        (256, [128] * 4, [8192] * 4, 4, 16),
        (256, [2048, 1024], [2048, 1024], 1, 64),
        (128, [2048, 1024], [2048, 1024], 8, 64),
        (64, [128] * 4, [8192] * 4, 4, 16),
        (256, [32], [4096], 4, 16),
        (128, [64, 2048], [8192] * 2, 4, 64),
        (256, [2048], [8192], 4, 16),
    ],
)
def test_fp16_domain_transfer_and_mixed_dtype_rejection(
    monkeypatch, cc, dim, qs, ks, group, page
):
    monkeypatch.setattr(fp, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(fp, "_cudnn_supports_paged_auto", lambda: True)
    monkeypatch.setattr(fp, "_cudnn_supports_paged_gqa_auto", lambda dim: True)
    monkeypatch.setattr(fp, "_cudnn_supports_paged_full_prefill_auto", lambda dim: True)
    args = kwargs(dim, qs, ks, group, page)
    bf16 = fp._blackwell_paged_auto_cudnn(**args)
    args.update(q_dtype=torch.float16, kv_dtype=torch.float16, o_dtype=torch.float16)
    assert fp._blackwell_paged_auto_cudnn(**args) == bf16
    for field in ("q_dtype", "kv_dtype", "o_dtype"):
        mixed = {**args, field: torch.bfloat16}
        assert not fp._blackwell_paged_auto_cudnn(**mixed)


class DeviceOnly:
    device = SimpleNamespace(type="cuda")

    def tolist(self):
        raise AssertionError("unexpected admission readback")

    def flatten(self):
        raise AssertionError("unexpected device metadata inspection")


@pytest.mark.parametrize("field", ["qo_indptr_host", "kv_lens", "seq_lens_q"])
def test_fp16_d256_metadata_stays_on_cpu(monkeypatch, field):
    args = kwargs(256, [128] * 4, [8192] * 4)
    args.update(q_dtype=torch.float16, kv_dtype=torch.float16, o_dtype=torch.float16)
    args[field] = DeviceOnly()
    assert not fp._blackwell_paged_auto_cudnn(**args)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("group", [2, 4, 8, 16])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("page", [16, 64])
def test_fp16_includes_entire_d256_gqa_full_region(monkeypatch, cc, group, batch, page):
    monkeypatch.setattr(fp, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(
        fp, "_cudnn_supports_paged_full_prefill_auto", lambda dim: dim == 256
    )
    qs = [1024] + [768] * (batch - 1)
    args = kwargs(256, qs, qs, group=group, page=page)
    args.update(q_dtype=torch.float16, kv_dtype=torch.float16, o_dtype=torch.float16)
    assert fp._blackwell_paged_auto_cudnn(**args)
    # Equal maxima do not prove full-prefill semantics for every request.
    if batch > 1:
        args["kv_lens"][1] += 1
        assert not fp._blackwell_paged_auto_cudnn(**args)
