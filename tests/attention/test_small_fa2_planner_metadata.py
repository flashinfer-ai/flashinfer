# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import flashinfer


def _make_case(
    requests,
    *,
    dtype=torch.float16,
    group=1,
    layout="NHD",
    page=16,
    query_tokens=None,
    include_prefix=False,
    causal_control=False,
    dim=128,
    window_left=-1,
    logits_soft_cap=0.0,
):
    """requests = [(prefix length, content lengths excluding delimiters), ...]."""
    torch.manual_seed(20261019)
    kv_heads = 4
    query_lengths = []
    kv_lengths = []
    positions = []
    for prefix, lengths in requests:
        pos = [p for length in lengths for p in range(length + 1)]
        positions.append(pos)
        kv_lengths.append(prefix + len(pos))
        qlen = kv_lengths[-1] if include_prefix else len(pos)
        query_lengths.append(min(qlen, query_tokens) if query_tokens else qlen)
    page_counts = [math.ceil(length / page) for length in kv_lengths]
    pages = sum(page_counts)
    kv = torch.randn(pages, 2, page, kv_heads, dim, device="cuda", dtype=dtype) / 5
    q = (
        torch.randn(
            sum(query_lengths), kv_heads * group, dim, device="cuda", dtype=dtype
        )
        / 5
    )
    indices = torch.randperm(pages, device="cuda", dtype=torch.int32)
    qptr, kptr = [0], [0]
    for qlen, npages in zip(query_lengths, page_counts, strict=True):
        qptr.append(qptr[-1] + qlen)
        kptr.append(kptr[-1] + npages)
    refs, lses = [], []
    for ri, ((prefix, lengths), qlen, klen) in enumerate(
        zip(requests, query_lengths, kv_lengths, strict=True)
    ):
        cache = kv[indices[kptr[ri] : kptr[ri + 1]].long()]
        k = (
            cache[:, 0]
            .reshape(-1, kv_heads, dim)[:klen]
            .repeat_interleave(group, 1)
            .float()
        )
        v = (
            cache[:, 1]
            .reshape(-1, kv_heads, dim)[:klen]
            .repeat_interleave(group, 1)
            .float()
        )
        item_ids = [-1] * prefix + [
            i for i, length in enumerate(lengths) for _ in range(length + 1)
        ]
        content = [False] * prefix + [
            p > 0 for length in lengths for p in range(length + 1)
        ]
        ids = torch.tensor(item_ids, device="cuda")
        content = torch.tensor(content, device="cuda")
        key_ids = torch.arange(klen, device="cuda")
        output, lse = [], []
        for start in range(0, qlen, 128):
            end = min(start + 128, qlen)
            query_ids = torch.arange(
                klen - qlen + start, klen - qlen + end, device="cuda"
            )
            mask = key_ids[None, :] <= query_ids[:, None]
            if window_left >= 0:
                mask &= key_ids[None, :] >= query_ids[:, None] - window_left
            if not causal_control:
                same_item = ids[query_ids, None] == ids[None, :]
                both_content = content[query_ids, None] & content[None, :]
                mask &= (key_ids[None, :] < prefix) | (same_item & both_content)
            scores = torch.einsum(
                "qhd,khd->hqk", q[qptr[ri] + start : qptr[ri] + end].float(), k
            ) / math.sqrt(dim)
            if logits_soft_cap:
                scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
            scores.masked_fill_(~mask.unsqueeze(0), -torch.inf)
            output.append(
                torch.einsum("hqk,khd->qhd", scores.softmax(-1).nan_to_num(0), v)
            )
            lse.append(scores.logsumexp(-1).transpose(0, 1) / math.log(2))
        refs.append(torch.cat(output))
        lses.append(torch.cat(lse))
    max_positions = max(map(len, positions))
    pos_tensor = torch.zeros(
        len(requests), max_positions, dtype=torch.int32, device="cuda"
    )
    for ri, pos in enumerate(positions):
        pos_tensor[ri, : len(pos)] = torch.tensor(pos, dtype=torch.int32, device="cuda")
    metadata = {
        "qo_indptr": torch.tensor(qptr, dtype=torch.int32, device="cuda"),
        "paged_kv_indptr": torch.tensor(kptr, dtype=torch.int32, device="cuda"),
        "paged_kv_indices": indices,
        "paged_kv_last_page_len": torch.tensor(
            [(length - 1) % page + 1 for length in kv_lengths],
            dtype=torch.int32,
            device="cuda",
        ),
        "num_qo_heads": kv_heads * group,
        "num_kv_heads": kv_heads,
        "head_dim_qk": dim,
        "page_size": page,
        "causal": True,
        "q_data_type": dtype,
        "kv_data_type": dtype,
        "window_left": window_left,
        "logits_soft_cap": logits_soft_cap,
    }
    if not causal_control:
        metadata.update(
            prefix_len_ptr=torch.tensor(
                [r[0] for r in requests], dtype=torch.uint32, device="cuda"
            ),
            token_pos_in_items_ptr=pos_tensor.to(torch.uint16),
            token_pos_in_items_len=max_positions,
            max_item_len_ptr=torch.tensor(
                [max(r[1]) for r in requests], dtype=torch.uint16, device="cuda"
            ),
        )
    if layout == "HND":
        kv = kv.permute(0, 1, 3, 2, 4).contiguous()
    return q, kv, metadata, torch.cat(refs).to(dtype), torch.cat(lses)


def _inputs(batch, iteration, dtype, mask):
    """Create changing metadata and independent output/LSE references."""
    q, kv, metadata, ref, lse = _make_case(
        [(17 + iteration + i, [3 + i, 7, 2]) for i in range(batch)],
        dtype=dtype,
        causal_control=mask != "mis",
    )
    if mask == "none":
        metadata["causal"] = False
        # Compute a dense CPU FP32 reference before any native execution.
        qh, kh = q.cpu().float(), kv.cpu().float()
        qptr = metadata["qo_indptr"].cpu().tolist()
        kptr = metadata["paged_kv_indptr"].cpu().tolist()
        idx = metadata["paged_kv_indices"].cpu().long()
        tail = metadata["paged_kv_last_page_len"].cpu().tolist()
        outputs, lses = [], []
        for i in range(batch):
            cache = kh[idx[kptr[i] : kptr[i + 1]]]
            length = (kptr[i + 1] - kptr[i] - 1) * 16 + tail[i]
            k = cache[:, 0].reshape(-1, 4, 128)[:length]
            v = cache[:, 1].reshape(-1, 4, 128)[:length]
            scores = torch.einsum("qhd,khd->hqk", qh[qptr[i] : qptr[i + 1]], k)
            scores /= math.sqrt(128)
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
            lses.append(scores.logsumexp(-1).T / math.log(2))
        ref, lse = torch.cat(outputs).to(dtype), torch.cat(lses)
    return q, kv, metadata, ref.cpu(), lse.cpu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", ["none", "causal", "mis"])
@pytest.mark.parametrize("location", ["cuda", "cpu", "mixed"])
def test_small_fa2_metadata_mutation_and_batch_resize(dtype, mask, location):
    """Replanning must observe in-place changes and resized batches on another stream."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="auto",
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    reusable = {}
    with torch.cuda.stream(stream):
        # Repeated sizes change values in place; 5 exercises tensor length arithmetic.
        for iteration, batch in enumerate((1, 1, 2, 2, 4, 4, 2, 1, 5)):
            q, kv, metadata, ref, lse_ref = _inputs(batch, iteration, dtype, mask)
            for key in ("qo_indptr", "paged_kv_indptr", "paged_kv_last_page_len"):
                value = metadata[key]
                if location == "cpu" or (location == "mixed" and key == "qo_indptr"):
                    value = value.cpu()
                cache_key = (key, batch)
                if cache_key not in reusable:
                    reusable[cache_key] = value.clone()
                else:
                    reusable[cache_key].copy_(value)
                metadata[key] = reusable[cache_key]
            with torch.device("cuda"):
                wrapper.plan(**metadata)
            qptr = metadata["qo_indptr"].cpu().tolist()
            kptr = metadata["paged_kv_indptr"].cpu().tolist()
            tail = metadata["paged_kv_last_page_len"].cpu().tolist()
            lengths = [
                max(b - a - 1, 0) * 16 + t
                for a, b, t in zip(kptr[:-1], kptr[1:], tail, strict=True)
            ]
            assert wrapper._max_q_len == max(
                b - a for a, b in zip(qptr[:-1], qptr[1:], strict=True)
            )
            assert wrapper._max_kv_len == max(lengths)
            torch.testing.assert_close(
                wrapper._kv_lens_buffer[:batch].cpu(),
                torch.tensor(lengths, dtype=torch.int32, device="cpu"),
            )
            out, lse = wrapper.run(q, kv, return_lse=True)
            torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(lse.cpu(), lse_ref, rtol=1e-2, atol=1e-2)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("option", ["seq_lens", "maxima", "fixed", "disable"])
def test_small_fa2_metadata_plan_overrides(dtype, option):
    """Explicit lengths, bounds and split options must retain their semantics."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
    )
    for iteration, batch in enumerate((2, 2, 1)):
        q, kv, metadata, ref, lse_ref = _inputs(batch, iteration, dtype, "causal")
        if option == "seq_lens":
            metadata["seq_lens"] = torch.tensor(
                [17 + iteration + i + 15 + i for i in range(batch)],
                dtype=torch.int32,
                device="cuda",
            )
        elif option == "maxima":
            metadata.update(max_token_per_sequence=32, max_sequence_kv=128)
        elif option == "fixed":
            metadata["fixed_split_size"] = 8
        else:
            metadata["disable_split_kv"] = True
        wrapper.plan(**metadata)
        if option == "maxima":
            assert wrapper._max_q_len == 32 and wrapper._max_kv_len == 128
        out, lse = wrapper.run(q, kv, return_lse=True)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse.cpu(), lse_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [8, 16, 32, 64])
def test_fa2_large_batch_pinned_metadata_refresh(dtype, batch):
    """Larger batches must refresh the same pinned storage on each plan."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
    )
    pointers = None
    for iteration in (0, 1):
        q, kv, metadata, ref, ref_lse = _inputs(batch, iteration, dtype, "causal")
        wrapper.plan(**metadata)
        buffers = wrapper._fa2_host_metadata[1]
        current = tuple(value.data_ptr() for value in buffers)
        if pointers is not None:
            assert pointers == current
        pointers = current
        for host, key in zip(
            buffers,
            ("qo_indptr", "paged_kv_indptr", "paged_kv_last_page_len"),
            strict=True,
        ):
            assert host.is_pinned()
            assert torch.equal(host, metadata[key].cpu())
        out, lse = wrapper.run(q, kv, return_lse=True)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_fa2_explicit_kv_bound_keeps_existing_host_path(dtype, backend):
    """A KV bound must not cause eager KV staging before auto backend selection."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend=backend,
    )
    q, kv, metadata, _, _ = _inputs(1, 0, dtype, "causal")
    # Resolve auto, then prime the same pinned cache used by explicit FA2.
    wrapper.plan(**metadata)
    wrapper.plan(**metadata)
    if wrapper._backend != "fa2":
        pytest.skip("requires auto to resolve to FA2")
    buffers = wrapper._fa2_host_metadata[1]
    saved = [value.clone() for value in buffers]
    q, kv, metadata, ref, ref_lse = _inputs(1, 1, dtype, "causal")
    metadata.update(max_token_per_sequence=32, max_sequence_kv=128)
    wrapper.plan(**metadata)
    assert wrapper._max_q_len == 32 and wrapper._max_kv_len == 128
    for host, previous in zip(buffers, saved, strict=True):
        assert torch.equal(host, previous)
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fa2_graph_wrapper_keeps_generic_host_metadata(dtype):
    """Explicit graph wrappers must keep their existing metadata planning path."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(2, dtype=torch.int32, device="cuda"),
        paged_kv_indptr_buf=torch.empty(2, dtype=torch.int32, device="cuda"),
        paged_kv_indices_buf=torch.empty(128, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len_buf=torch.empty(1, dtype=torch.int32, device="cuda"),
    )
    for iteration in (0, 1):
        q, kv, metadata, ref, ref_lse = _inputs(1, iteration, dtype, "causal")
        wrapper.plan(**metadata)
        assert not hasattr(wrapper, "_fa2_host_metadata")
        out, lse = wrapper.run(q, kv, return_lse=True)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
