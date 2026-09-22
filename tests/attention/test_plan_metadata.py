"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.mla import MLAPlanMetadata
from flashinfer.utils import _copy_to_cpu


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_copy_to_cpu_stream_and_strided_metadata(dtype):
    cpu = torch.arange(9, dtype=dtype)[1::2]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Produce the metadata on a non-default stream immediately before staging.
        device = torch.arange(19, device="cuda", dtype=dtype)
        device.add_(7)
        a, b, c, empty = _copy_to_cpu(device[1::3], cpu, device[::2], device[:0])
    assert torch.equal(a, torch.arange(19, dtype=dtype)[1::3] + 7)
    assert torch.equal(c, torch.arange(19, dtype=dtype)[::2] + 7)
    assert b is cpu
    assert a.is_pinned() and c.is_pinned()
    assert a.dtype == dtype and a.is_contiguous()
    assert empty.shape == (0,)
    # A later call must not overwrite metadata retained by an earlier caller.
    with torch.cuda.stream(stream):
        device.add_(100)
        (updated,) = _copy_to_cpu(device[1::3])
    assert torch.equal(updated, a + 100)


def _wrapper(kind, graph, batch, dtype=torch.int32):
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    qo = torch.empty(batch + 1, dtype=dtype, device="cuda")
    kv = torch.empty_like(qo)
    indices = torch.empty(64, dtype=dtype, device="cuda")
    lengths = torch.empty(batch, dtype=dtype, device="cuda")
    kwargs = dict(backend="fa2", use_cuda_graph=graph)
    if kind == "mla":
        return flashinfer.mla.BatchMLAPagedAttentionWrapper(
            workspace,
            qo_indptr=qo,
            kv_indptr=kv,
            kv_indices=indices,
            kv_len_arr=lengths,
            **kwargs,
        )
    if kind.startswith("decode"):
        return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            use_tensor_cores=kind == "decode_tc",
            paged_kv_indptr_buffer=kv,
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=lengths,
            **kwargs,
        )
    if kind == "paged":
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            qo_indptr_buf=qo,
            paged_kv_indptr_buf=kv,
            paged_kv_indices_buf=indices,
            paged_kv_last_page_len_buf=lengths,
            **kwargs,
        )
    return flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        qo_indptr_buf=qo,
        kv_indptr_buf=kv,
        **kwargs,
    )


def _metadata(kind, mode, changed=False, dtype=torch.int32):
    lengths = torch.tensor([30, 15, 47] if changed else [17, 31, 48], dtype=dtype)
    qo = torch.tensor([0, 1, 4, 6] if changed else [0, 2, 3, 6], dtype=dtype)
    if kind.startswith("decode"):
        qo = torch.arange(4, dtype=dtype)
    page_counts = (lengths + 15) // 16
    kv = torch.cat([torch.zeros(1, dtype=dtype), page_counts.cumsum(0).to(dtype)])
    indices = torch.tensor([6, 2, 5, 0, 1, 4, 3], dtype=dtype)[: int(kv[-1])]
    last = (lengths - 1) % 16 + 1
    if kind == "ragged":
        kv = torch.cat([torch.zeros(1, dtype=dtype), lengths.cumsum(0).to(dtype)])
    host = (qo, kv, indices, last, lengths)
    # Page mappings stay on the GPU even with CPU offsets and lengths.
    tensors = [
        x.cuda() if mode == "gpu" or i == 2 or (mode == "mixed" and i % 2 == 0) else x
        for i, x in enumerate(host)
    ]
    return host, tensors


def _plan(wrapper, kind, tensors, explicit_lengths=False):
    qo, kv, indices, last, lengths = tensors
    kwargs = dict(q_data_type=torch.float16, kv_data_type=torch.float16)
    if kind == "mla":
        wrapper.plan(
            metadata=MLAPlanMetadata.csr(qo, kv, indices, lengths),
            num_heads=32,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=16,
            causal=True,
            sm_scale=576**-0.5,
            lse_mode="basee",
            **kwargs,
        )
    elif kind == "ragged":
        wrapper.plan(qo, kv, 32, 8, 128, causal=True, **kwargs)
    else:
        if explicit_lengths:
            kwargs["seq_lens"] = lengths
        if kind == "paged":
            wrapper.plan(qo, kv, indices, last, 32, 8, 128, 16, causal=True, **kwargs)
        else:
            wrapper.plan(kv, indices, last, 32, 8, 128, 16, **kwargs)


def _inputs(kind):
    rows = 3 if kind.startswith("decode") else 6
    dim = 576 if kind == "mla" else 128
    q = torch.randn(rows, 32, dim, device="cuda", dtype=torch.float16)
    if kind == "mla":
        k = torch.randn(7, 16, dim, device="cuda", dtype=torch.float16)
        v = k[..., :512]
    elif kind == "ragged":
        k = torch.randn(96, 8, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
    else:
        k = torch.randn(7, 16, 8, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
    return q, k, v


def _run(wrapper, kind, q, k, v):
    if kind == "mla":
        return wrapper.run(
            query=q, kv_cache=k, return_lse=True, return_lse_base_on_e=True
        )
    if kind == "ragged":
        return wrapper.run(q, k, v, return_lse=True)
    return wrapper.run(q, (k, v), return_lse=True)


def _reference(kind, host, q, k, v):
    qo, kv, indices, _, lengths = host
    outputs, lses = [], []
    for i in range(len(lengths)):
        start, end, length = int(qo[i]), int(qo[i + 1]), int(lengths[i])
        qi = q[start:end].float().transpose(0, 1)
        if kind == "ragged":
            ki = k[int(kv[i]) : int(kv[i + 1])]
            vi = v[int(kv[i]) : int(kv[i + 1])]
        else:
            pages = indices[int(kv[i]) : int(kv[i + 1])].long().cuda()
            ki = k[pages].flatten(0, 1)[:length]
            vi = v[pages].flatten(0, 1)[:length]
        if kind == "mla":
            ki = ki[:, None, :].expand(-1, 32, -1)
            vi = vi[:, None, :].expand(-1, 32, -1)
        else:
            ki, vi = (x.repeat_interleave(4, dim=1) for x in (ki, vi))
        scores = qi @ ki.float().permute(1, 2, 0) * q.shape[-1] ** -0.5
        mask = torch.arange(length, device="cuda")[None, :] > (
            length - (end - start) + torch.arange(end - start, device="cuda")[:, None]
        )
        scores.masked_fill_(mask, -float("inf"))
        outputs.append(
            (scores.softmax(-1) @ vi.float().transpose(0, 1)).transpose(0, 1)
        )
        lse = scores.logsumexp(-1).transpose(0, 1)
        lses.append(lse if kind == "mla" else lse / math.log(2))
    return torch.cat(outputs), torch.cat(lses)


@pytest.mark.parametrize("kind", ["paged", "ragged", "decode", "decode_tc", "mla"])
@pytest.mark.parametrize("mode", ["cpu", "gpu", "mixed"])
@pytest.mark.parametrize("graph", [False, True])
def test_plan_metadata_updates_and_graph_replay(kind, mode, graph):
    torch.manual_seed(3338)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper = _wrapper(kind, graph, 3)
        q, k, v = _inputs(kind)
        host, tensors = _metadata(kind, mode)
        _plan(wrapper, kind, tensors)
        _run(wrapper, kind, q, k, v)
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                actual = _run(wrapper, kind, q, k, v)
        # Replan twice, including changing values in the same input tensors.
        for changed in (False, True, False):
            host, new_tensors = _metadata(kind, mode, changed)
            for i, new in enumerate(new_tensors):
                if tensors[i].shape == new.shape:
                    tensors[i].copy_(new)
                else:
                    tensors[i] = new
            _plan(wrapper, kind, tensors, explicit_lengths=changed)
            if graph:
                g.replay()
            else:
                actual = _run(wrapper, kind, q, k, v)
            expected = _reference(kind, host, q, k, v)
            for got, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(got.float(), want, atol=3e-3, rtol=3e-3)
    stream.synchronize()


@pytest.mark.parametrize(
    "kind,dtype",
    [
        ("paged", torch.int32),
        ("paged", torch.int64),
        ("ragged", torch.int32),
        ("ragged", torch.int64),
        ("decode", torch.int32),
        ("decode_tc", torch.int32),
        ("mla", torch.int32),
    ],
)
def test_plan_metadata_dtype_and_batch_growth(kind, dtype):
    wrapper = _wrapper(kind, False, 3, dtype)
    host, tensors = _metadata(kind, "gpu", dtype=dtype)
    _plan(wrapper, kind, tensors)
    # Reuse the same wrapper at a larger batch size; no staging-size assumption
    # may survive from the preceding plan.
    qo, kv, indices, last, lengths = tensors
    tensors = [
        torch.cat([qo, qo[1:] + qo[-1]]),
        torch.cat([kv, kv[1:] + kv[-1]]),
        indices.repeat(2),
        last.repeat(2),
        lengths.repeat(2),
    ]
    _plan(wrapper, kind, tensors, explicit_lengths=True)
    q, k, v = _inputs(kind)
    q = q.repeat(2, 1, 1)
    if kind == "ragged":
        k, v = k.repeat(2, 1, 1), v.repeat(2, 1, 1)
    actual = _run(wrapper, kind, q, k, v)
    host = tuple(x.cpu() for x in tensors)
    for got, want in zip(actual, _reference(kind, host, q, k, v), strict=True):
        torch.testing.assert_close(got.float(), want, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("kind", ["paged", "ragged", "decode", "mla"])
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_plan_metadata_synchronizes_at_most_once(kind, mode):
    wrapper = _wrapper(kind, False, 3)
    _, tensors = _metadata(kind, mode)
    for _ in range(3):
        _plan(wrapper, kind, tensors, explicit_lengths=True)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        _plan(wrapper, kind, tensors, explicit_lengths=True)
    waits = sum(event.name == "cudaStreamSynchronize" for event in prof.events())
    assert waits == (1 if mode == "gpu" else 0)
