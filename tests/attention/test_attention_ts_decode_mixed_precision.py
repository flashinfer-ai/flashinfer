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

"""Mixed-precision PrimTS FMHA decode grid.

Kept apart from test_attention_ts_decode.py because CI runs each test file
under one time budget, and this grid alone takes about an hour on GB200.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl==4.7.0",
)

from tests.attention.test_attention_ts_decode import (  # noqa: E402
    _REQUIRES_PRIMTS_GPU,
    _MixedPrecisionKind,
    run_mixed_precision_decode_case,
)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda value: f"b{value}")
@pytest.mark.parametrize(
    "seq_len_kv",
    [128, 2048, 4096, 8192],
    ids=lambda value: f"kv{value}",
)
@pytest.mark.parametrize(
    "seq_len_q",
    [1, 4, 8],
    ids=lambda value: f"q{value}",
)
@pytest.mark.parametrize(
    ("num_qo_heads", "num_kv_heads"),
    [(32, 4)],
    ids=["gqa32x4"],
)
@pytest.mark.parametrize(
    "head_dim,page_size,qkv_dtype",
    [
        pytest.param(d, p, kind, id=f"d{d}-p{p}-{kind}")
        for d in (64, 128, 256, 512)
        for p in (16, 64)
        for kind in (
            "bf16q-fp8kv",
            "bf16q-nvfp4kv",
            "fp8q-nvfp4kv",
            "fp8q-nvfp4kv-bf16o",
        )
        if not (d == 64 and p == 16 and "nvfp4" in kind)
    ],
)
def test_attention_ts_decode_mixed_precision(
    batch_size: int,
    seq_len_kv: int,
    seq_len_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    qkv_dtype: _MixedPrecisionKind,
):
    run_mixed_precision_decode_case(
        batch_size=batch_size,
        seq_len_kv=seq_len_kv,
        seq_len_q=seq_len_q,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        qkv_dtype=qkv_dtype,
    )
