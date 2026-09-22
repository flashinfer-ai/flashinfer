# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
from bench_attention_ts_sparse_mla import Fixture
from bench_attention_ts_sparse_mla_models import cases
from sparse_mla_model_fixtures import (
    model_fixture,
    chunked_reference,
    streaming_fingerprint,
)
from flashinfer.testing.sparse_mla import (
    sparse_mla_reference,
    sparse_mla_input_fingerprint,
)


def test_requested_model_matrix():
    matrix = cases()
    assert len(matrix) == 480
    assert sum(c["phase"] == "prefill" for c in matrix) == 30
    assert sum(c["phase"] == "decode" for c in matrix) == 450
    assert (
        len(
            {
                (
                    c["phase"],
                    c["batch"],
                    c["queries"],
                    c["heads"],
                    c["topk"],
                    c["dtype"],
                )
                for c in matrix
            }
        )
        == 480
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for native FP8 indexing"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("ratio", [1, 128])
def test_causal_model_fixture_and_chunked_oracle(dtype, ratio):
    data, layout = model_fixture(
        2,
        8,
        4,
        16,
        dtype,
        raw_tokens=256,
        compression_ratio=ratio,
        score_chunk=2,
        debug_layout=True,
    )
    f = Fixture(**data)
    inverse = layout["primary_table"].reshape(-1).argsort()
    logical = inverse[f.ci.clamp_min(0).long()]
    per_request = layout["candidate_rows_per_request"]
    valid = f.ci >= 0
    request = torch.arange(2, device=f.query.device)[:, None, None].expand_as(f.ci)
    available = (256 - 4 + torch.arange(4, device=f.query.device) + 1) // ratio
    assert torch.equal((logical // per_request)[valid], request[valid])
    assert ((logical % per_request < available[None, :, None]) | ~valid).all()
    for ids in f.ci.reshape(-1, 16):
        ids = ids[ids >= 0]
        assert ids.unique().numel() == ids.numel()
    inverse_swa = layout["swa_table"].reshape(-1).argsort()
    swa_local = inverse_swa[(f.si // 256).long()] * 256 + f.si % 256
    expected_swa = request[..., :1] * layout["swa_resident_rows_per_request"] + (
        torch.arange(4, device=f.query.device)[None, :, None]
        + torch.arange(128, device=f.query.device)[None, None, :]
    )
    assert torch.equal(swa_local, expected_swa)
    assert streaming_fingerprint(f, chunk_bytes=137) == sparse_mla_input_fingerprint(
        **vars(f)
    )

    # Check empty rows as well as padding and infinite sinks.
    f.sl[0, 0] = 0
    f.cl[0, 0] = 0
    ref, lse, budget = chunked_reference(f, chunk_rows=3)
    expected, expected_lse, old_budget = sparse_mla_reference(
        f.query,
        f.swa,
        f.compressed,
        f.si,
        f.ci,
        swa_topk_lens=f.sl,
        compressed_topk_lens=f.cl,
        sinks=f.sinks,
        softmax_scale=f.softmax_scale,
        return_fp8_error_bound=True,
    )
    torch.testing.assert_close(ref, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(lse, expected_lse, atol=1e-12, rtol=1e-12)
    if dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(budget, old_budget, atol=1e-12, rtol=1e-12)
