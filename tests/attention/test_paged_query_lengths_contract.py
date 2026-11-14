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

"""A declined paged provider must not silently change independent Q masking."""

import pytest
import torch

import flashinfer
from flashinfer.cudnn.prefill import _cudnn_supports_paged_auto
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("backend", ["fa2", "auto"])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("metadata_device", ["cpu", "cuda"])
def test_rejected_query_mask_preserves_accepted_plan(
    backend, captured, metadata_device
):
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 8:
        pytest.skip("Requires BF16 query support")
    if backend == "auto" and cc != (10, 0):
        pytest.skip("Exercises the SM100 paged auto admission boundary")
    if backend == "auto" and not _cudnn_supports_paged_auto():
        pytest.skip("Requires a qualified cuDNN paged auto runtime")

    qo = torch.tensor([0, 128], dtype=torch.int32)
    kp = torch.tensor([0, 512], dtype=torch.int32)
    indices = torch.arange(512, device="cuda", dtype=torch.int32)
    last = torch.tensor([16], dtype=torch.int32)
    kvlen = torch.tensor([8192], dtype=torch.int32)
    q = torch.zeros(128, 16, 128, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(512, 2, 2, 16, 128, dtype=q.dtype, device=q.device)
    kv[-8:, 1].fill_(128)
    opts = {}
    if captured:
        opts = dict(
            use_cuda_graph=True,
            qo_indptr_buf=qo.cuda(),
            paged_kv_indptr_buf=kp.cuda(),
            paged_kv_indices_buf=indices.clone(),
            paged_kv_last_page_len_buf=last.cuda(),
        )
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
        "HND",
        backend=backend,
        **opts,
    )

    def plan(lengths, dtype):
        w.plan(
            qo,
            kp,
            indices,
            last,
            16,
            2,
            128,
            16,
            causal=True,
            q_data_type=dtype,
            seq_lens=kvlen,
            seq_lens_q=lengths,
            max_token_per_sequence=128,
            max_sequence_kv=8192,
            block_tables=indices.view(1, -1),
        )

    valid_q = 1 if backend == "auto" else 128
    plan(torch.tensor([valid_q], dtype=torch.int32), q.dtype)
    assert w._backend == ("cudnn" if backend == "auto" else "fa2")
    for _ in range(3):
        out, lse = w.run(q, kv, return_lse=True, lse_base="ln")
    torch.cuda.synchronize()
    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = w.run(q, kv, return_lse=True, lse_base="ln")
        graph.replay()
        torch.cuda.synchronize()
    expected_first = 2.0 if valid_q == 1 else 128 / 8065
    torch.testing.assert_close(
        out[0], torch.full_like(out[0], expected_first), rtol=0, atol=0
    )
    baseline_out, baseline_lse = out[:valid_q].clone(), lse[:valid_q].clone()
    state = w.__dict__.copy()
    buffers = {
        key: value.contiguous().view(torch.uint8).clone()
        for key, value in state.items()
        if torch.is_tensor(value) and key != "_float_workspace_buffer"
    }
    lengths = torch.tensor([1], dtype=torch.int32, device=metadata_device)
    torch.cuda.synchronize()
    prior_debug = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        # FP16 declines the BF16-only cuDNN auto rule. It must not silently use
        # FA2 with Q128 masking, or mutate the previously accepted BF16 plan.
        with pytest.raises(ValueError, match="seq_lens_q"):
            plan(lengths, torch.float16)
    finally:
        torch.cuda.set_sync_debug_mode(prior_debug)
    assert w.__dict__.keys() == state.keys()
    assert all(w.__dict__[key] is value for key, value in state.items())
    for key, value in buffers.items():
        assert torch.equal(w.__dict__[key].contiguous().view(torch.uint8), value)
    if graph is not None:
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out[:valid_q], baseline_out, rtol=0, atol=0)
        torch.testing.assert_close(lse[:valid_q], baseline_lse, rtol=0, atol=0)
    out, lse = w.run(q, kv, return_lse=True, lse_base="ln")
    torch.testing.assert_close(out[:valid_q], baseline_out, rtol=0, atol=0)
    torch.testing.assert_close(lse[:valid_q], baseline_lse, rtol=0, atol=0)
