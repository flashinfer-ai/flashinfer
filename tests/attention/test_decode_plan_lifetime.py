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

import flashinfer


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_fast_decode_replan_preserves_queued_outputs(use_tensor_cores):
    """A later plan must not overwrite metadata whose H2D copy is still queued."""
    dev = torch.device("cuda:0")
    if torch.cuda.get_device_capability(dev)[0] < 8:
        pytest.skip("BF16 decode requires SM80 or later")
    torch.manual_seed(17)
    batch_size, num_qo_heads, num_kv_heads, head_dim, page_size = 16, 16, 8, 128, 16
    num_pages = batch_size * 8192 // page_size
    kv_cache = torch.randn(
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=dev,
        dtype=torch.bfloat16,
    )
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device=dev, dtype=torch.bfloat16
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    indptr_buf = torch.empty(batch_size + 1, dtype=torch.int32, device=dev)
    indices_buf = torch.empty(num_pages, dtype=torch.int32, device=dev)
    last_page_buf = torch.empty(batch_size, dtype=torch.int32, device=dev)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=use_tensor_cores,
        backend="fa2",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_page_buf,
    )
    metadata = []
    for kv_len in (8192, 16):
        pages_per_request = kv_len // page_size
        metadata.append(
            (
                torch.tensor(
                    [i * pages_per_request for i in range(batch_size + 1)],
                    dtype=torch.int32,
                    pin_memory=True,
                ),
                torch.arange(
                    batch_size * pages_per_request, dtype=torch.int32, device=dev
                ),
                torch.full(
                    (batch_size,), page_size, dtype=torch.int32, pin_memory=True
                ),
            )
        )
    options = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=head_dim**-0.5,
    )

    def plan(meta, *, initial=False):
        indptr, indices, last_page_len = meta
        # The fast API borrows pre-staged graph metadata, as serving callers do.
        indptr_buf.copy_(indptr, non_blocking=True)
        indices_buf[: indices.numel()].copy_(indices)
        last_page_buf.copy_(last_page_len, non_blocking=True)
        if initial:
            wrapper.plan(indptr, indices, last_page_len, **options)
        else:
            flashinfer.fast_decode_plan(
                wrapper, indptr, indices, last_page_len, **options
            )

    references = []
    for i, meta in enumerate(metadata):
        plan(meta, initial=i == 0)
        references.append(wrapper.run(q, kv_cache).clone())
        torch.cuda.synchronize(dev)
    outputs = [torch.empty_like(q) for _ in range(3)]
    torch.cuda.synchronize(dev)
    with torch.cuda.device(dev):
        torch.cuda._sleep(50_000_000)
    # Force reuse of slot zero, leaving short metadata last so it cannot mask
    # corruption of the first, longer plan by restoring its metadata.
    order = (0, 1, 1)
    for batch, output in zip(order, outputs, strict=True):
        plan(metadata[batch])
        wrapper.run(q, kv_cache, out=output)
    torch.cuda.synchronize(dev)
    for batch, output in zip(order, outputs, strict=True):
        torch.testing.assert_close(output, references[batch], atol=1e-3, rtol=1e-3)
