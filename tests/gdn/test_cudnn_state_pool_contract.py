# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metadata rejection must precede native execution or state mutation."""

import pytest
import torch

from flashinfer.cudnn.linear_attention import _validate_la_state_pool


def _check(indices, state, output, return_state=True):
    return _validate_la_state_pool(
        indices, state, output, return_state, 2, 4, 64, 64, torch.device("cpu")
    )


@pytest.mark.parametrize(
    "problem",
    [
        "int64",
        "shape",
        "strided_indices",
        "no_input",
        "no_output",
        "fp16",
        "inner_stride",
        "partial_alias",
        "dlpack_alias",
        "grad",
    ],
)
def test_rejects_invalid_pool_before_execution(problem):
    state = torch.empty(6, 4, 64, 64)
    indices = torch.tensor([4, 1], dtype=torch.int32)
    output = state
    error = ValueError
    match = ""
    if problem == "int64":
        indices = indices.long()
        match = "int32"
    elif problem == "shape":
        indices = indices[:1]
        match = "num_seqs"
    elif problem == "strided_indices":
        indices = torch.empty(4, dtype=torch.int32)[::2]
        match = "contiguous"
    elif problem == "no_input":
        state = None
        match = "initial_state"
    elif problem == "no_output":
        output = None
        match = "explicit output_state"
    elif problem == "fp16":
        state = state.half()
        match = "float32 or bfloat16"
    elif problem == "inner_stride":
        state = torch.empty(6, 4, 64, 128)[..., ::2]
        match = "dense"
    elif problem == "partial_alias":
        output, state = state[1:], state[:-1]
        match = "identical views"
    elif problem == "dlpack_alias":
        output, state = torch.from_dlpack(state[1:]), state[:-1]
        assert not torch._C._overlaps(output, state)
        match = "identical views"
    elif problem == "grad":
        state.requires_grad_(True)
        error, match = RuntimeError, "requires gradients"
    with pytest.raises(error, match=match):
        _check(indices, state, output)


def test_pool_check_does_not_read_indices(monkeypatch):
    state = torch.empty(6, 4, 64, 64)
    # Values deliberately outside the caller precondition: only metadata is
    # validated here; executing this table would be invalid.
    indices = torch.tensor([99999, -5], dtype=torch.int32)

    def forbidden(*args, **kwargs):
        raise AssertionError("device indices were read on the host")

    for name in ("tolist", "item", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    assert _check(indices, state, state)
    assert not _check(indices, state, torch.empty_like(state))
    assert not _check(indices, state, None, return_state=False)


def test_inference_pool_is_checked_before_native_write():
    with torch.inference_mode():
        state = torch.empty(6, 4, 64, 64)
    indices = torch.tensor([4, 1], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="outside inference_mode"):
        _check(indices, state, state)
    with torch.inference_mode():
        assert _check(indices, state, state)


@pytest.mark.parametrize("family", ["gdn2", "gdp"])
def test_new_family_pool_requires_frontend_131_before_mutation(monkeypatch, family):
    from flashinfer.cudnn import linear_attention as adapter

    if not adapter.CUDNN_AVAILABLE:
        pytest.skip("requires the cuDNN Python package")
    monkeypatch.setattr(adapter.cudnn, "__version__", "1.30.0")
    q = torch.ones(4, 4, 64)
    pool = torch.randn(6, 4, 64, 64)
    before = pool.clone()
    kwargs = dict(
        initial_state=pool,
        output_state=pool,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 1, 4], dtype=torch.int32),
        state_indices=torch.tensor([4, 1], dtype=torch.int32),
    )
    fn = (
        adapter.cudnn_chunk_gated_delta_rule2
        if family == "gdn2"
        else adapter.cudnn_chunk_gated_delta_product
    )
    with pytest.raises(RuntimeError, match="state pools.*1.31"):
        fn(q, q, q, **kwargs)
    assert torch.equal(pool, before)
