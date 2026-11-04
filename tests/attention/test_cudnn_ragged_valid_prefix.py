"""Valid-length prefixes must remain independent of physical ragged offsets."""

import pytest
import torch

import flashinfer
from flashinfer.cudnn import prefill as cudnn_prefill


def _require_prefix_path():
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 3)
        or not cudnn_prefill.CUDNN_AVAILABLE
        or cudnn_prefill.cudnn.backend_version() < 92700
        or tuple(map(int, cudnn_prefill.cudnn.__version__.split(".")[:2])) < (1, 25)
    ):
        pytest.skip("requires SM103, cuDNN >= 9.27 and cudnn-frontend >= 1.25")
    assert cudnn_prefill._cudnn_prefill_prefers_cumulative_lengths(
        torch.device("cuda"),
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        64,
        8,
        64,
        64,
        64,
        causal=True,
    )


def _indptr(lengths):
    return torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int()


def _wrapper(batch=2, graph=True):
    kwargs = {}
    if graph:
        kwargs = dict(
            qo_indptr_buf=torch.empty(batch + 1, device="cuda", dtype=torch.int32),
            kv_indptr_buf=torch.empty(batch + 1, device="cuda", dtype=torch.int32),
        )
    return flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 << 20, device="cuda", dtype=torch.uint8),
        backend="cudnn",
        use_cuda_graph=graph,
        **kwargs,
    )


def _plan(
    wrapper,
    qslots,
    kslots,
    qlens,
    klens,
    *,
    mode="both",
    length_device="cpu",
    length_dtype=torch.int32,
    causal=True,
    dim=64,
):
    kwargs = {}
    for name, included, lengths in (
        ("seq_lens_q", mode in ("q", "both"), qlens),
        ("seq_lens", mode in ("kv", "both"), klens),
    ):
        if included:
            # Deliberately replace caller tensors on every plan.
            kwargs[name] = torch.tensor(
                lengths, device=length_device, dtype=length_dtype
            )
    wrapper.plan(
        _indptr(qslots),
        _indptr(kslots),
        64,
        8,
        dim,
        q_data_type=torch.bfloat16,
        causal=causal,
        max_token_per_sequence=64,
        max_sequence_kv=160,
        **kwargs,
    )
    return (
        qlens if mode in ("q", "both") else qslots,
        klens if mode in ("kv", "both") else kslots,
    )


def _inputs(dim=64, qtotal=96, ktotal=256):
    return (
        torch.empty(qtotal, 64, dim, device="cuda", dtype=torch.bfloat16),
        torch.empty(ktotal, 8, dim, device="cuda", dtype=torch.bfloat16),
        torch.empty(ktotal, 8, dim, device="cuda", dtype=torch.bfloat16),
    )


def _fill_valid(inputs, qslots, kslots, qlens, klens):
    for tensor, slots, lengths in zip(
        inputs, (qslots, kslots, kslots), (qlens, klens, klens), strict=True
    ):
        tensor.fill_(float("nan"))
        start = 0
        for slot, length in zip(slots, lengths, strict=True):
            tensor[start : start + length].normal_()
            start += slot


def _check(inputs, out, lse, qslots, kslots, qlens, klens, causal=True):
    q, k, v = inputs
    qs = ks = 0
    for qslot, kslot, nq, nk in zip(qslots, kslots, qlens, klens, strict=True):
        qq = q[qs : qs + nq].double().transpose(0, 1)
        kk = k[ks : ks + nk].double().transpose(0, 1).repeat_interleave(8, 0)
        vv = v[ks : ks + nk].double().transpose(0, 1).repeat_interleave(8, 0)
        scores = qq @ kk.transpose(1, 2) * (q.shape[-1] ** -0.5)
        if causal:
            mask = torch.arange(nk, device="cuda")[None, :] > (
                torch.arange(nq, device="cuda")[:, None] + nk - nq
            )
            scores.masked_fill_(mask, -torch.inf)
        expected = (scores.softmax(-1) @ vv).transpose(0, 1)
        torch.testing.assert_close(
            out[qs : qs + nq].double(), expected, atol=0.004, rtol=0.004
        )
        if lse is not None:
            torch.testing.assert_close(
                lse[qs : qs + nq].double(), scores.logsumexp(-1).T, atol=4e-4, rtol=0
            )
        qs += qslot
        ks += kslot


def _capture(wrapper, inputs, return_lse=True):
    out = torch.empty_like(inputs[0])
    lse = (
        torch.empty(inputs[0].shape[:2], device="cuda", dtype=torch.float32)
        if return_lse
        else None
    )

    def run():
        wrapper.run(*inputs, out=out, lse=lse, return_lse=return_lse, lse_base="ln")

    run()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        run()
    return capture, out, lse


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("backend_only", [False, True])
def test_valid_prefix_excludes_padding(monkeypatch, causal, return_lse, backend_only):
    _require_prefix_path()
    # An isolated real cache ensures the provider control actually builds a graph.
    build = cudnn_prefill.cudnn.graph_cache(key_fn=cudnn_prefill._sdpa_prefill_key_fn)(
        cudnn_prefill._build_prefill_graph.__wrapped__
    )
    graphs = []

    def record_graph(*args, **kwargs):
        result = build(*args, **kwargs)
        graphs.append(result[0])
        return result

    monkeypatch.setattr(cudnn_prefill, "_build_prefill_graph", record_graph)
    if backend_only:
        if not hasattr(cudnn_prefill.cudnn.pygraph, "_candidate_engines"):
            pytest.skip("frontend does not expose provider selection")
        monkeypatch.setattr(
            cudnn_prefill.cudnn.pygraph, "_candidate_engines", lambda _: []
        )
    torch.manual_seed(7005868)
    qslots, kslots, qlens, klens = [64, 32], [160, 96], [33, 17], [129, 83]
    inputs = _inputs()
    _fill_valid(inputs, qslots, kslots, qlens, klens)
    wrapper = _wrapper()
    _plan(wrapper, qslots, kslots, qlens, klens, causal=causal)
    metadata = wrapper._cudnn_plan.metadata
    if causal:
        assert metadata.actual_seq_lens_q is metadata.actual_seq_lens_kv is None
        torch.testing.assert_close(metadata.cu_seq_lens_q.cpu(), _indptr(qlens))
        torch.testing.assert_close(metadata.cu_seq_lens_kv.cpu(), _indptr(klens))
    else:
        assert metadata.cu_seq_lens_q is metadata.cu_seq_lens_kv is None
        torch.testing.assert_close(
            metadata.actual_seq_lens_q.flatten().cpu(),
            torch.tensor(qlens, dtype=torch.int32),
        )
        torch.testing.assert_close(
            metadata.actual_seq_lens_kv.flatten().cpu(),
            torch.tensor(klens, dtype=torch.int32),
        )
    torch.testing.assert_close(metadata.batch_offsets_q.cpu(), _indptr(qslots))
    torch.testing.assert_close(metadata.batch_offsets_k.cpu(), _indptr(kslots))
    capture, out, lse = _capture(wrapper, inputs, return_lse)
    try:
        out.fill_(float("nan"))
        if lse is not None:
            lse.fill_(float("nan"))
        capture.replay()
        _check(inputs, out, lse, qslots, kslots, qlens, klens, causal)
        if backend_only:
            from cudnn.engines import is_backend_engine

            assert graphs and all(
                is_backend_engine(g.get_engine_and_knobs_at_index(g._plan_index)[0])
                for g in graphs
            )
    finally:
        capture.reset()


@pytest.mark.parametrize("initial_mode", ["both", "neither"])
@pytest.mark.parametrize("length_device", ["cpu", "cuda"])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_valid_prefix_capture_survives_replan(
    initial_mode, length_device, length_dtype
):
    _require_prefix_path()
    torch.manual_seed(7438)
    inputs = _inputs()
    wrapper = _wrapper()
    qslots, kslots = [64, 32], [160, 96]
    _plan(wrapper, qslots, kslots, qslots, kslots, mode=initial_mode)
    _fill_valid(inputs, qslots, kslots, qslots, kslots)
    capture, out, lse = _capture(wrapper, inputs)
    pointers = tuple(t.data_ptr() for t in wrapper._cudnn_mask_prefix_buffers)
    try:
        for mode in ("q", "kv", "both", "neither", "both"):
            qslots, kslots = [48, 40], [144, 96]
            qlens, klens = _plan(
                wrapper,
                qslots,
                kslots,
                [21, 13],
                [91, 65],
                mode=mode,
                length_device=length_device,
                length_dtype=length_dtype,
            )
            assert pointers == tuple(
                t.data_ptr() for t in wrapper._cudnn_mask_prefix_buffers
            )
            _fill_valid(inputs, qslots, kslots, qlens, klens)
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            capture.replay()
            _check(inputs, out, lse, qslots, kslots, qlens, klens)
    finally:
        capture.reset()


def test_valid_prefix_and_per_request_captures_survive_eligibility_change():
    _require_prefix_path()
    torch.manual_seed(5868)
    wrapper = _wrapper()
    qslots, kslots = [64, 32], [160, 96]
    captures = []
    try:
        for dim in (128, 64):
            inputs = _inputs(dim)
            _fill_valid(inputs, qslots, kslots, qslots, kslots)
            _plan(wrapper, qslots, kslots, qslots, kslots, dim=dim)
            metadata = wrapper._cudnn_plan.metadata
            assert (metadata.cu_seq_lens_q is not None) == (dim == 64)
            capture, out, lse = _capture(wrapper, inputs)
            captures.append((capture, inputs, out, lse))
        # Re-enter the unqualified family after both representations were captured.
        for dim in (128, 64):
            qlens, klens = [29, 19], [119, 79]
            _plan(wrapper, qslots, kslots, qlens, klens, dim=dim)
            for capture, inputs, out, lse in captures:
                _fill_valid(inputs, qslots, kslots, qlens, klens)
                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                capture.replay()
                _check(inputs, out, lse, qslots, kslots, qlens, klens)
    finally:
        for capture, *_ in captures:
            capture.reset()


def test_valid_prefix_captures_survive_causal_transition():
    _require_prefix_path()
    torch.manual_seed(7005868)
    wrapper = _wrapper()
    inputs = _inputs()
    qslots, kslots = [64, 32], [160, 96]
    captures = []
    try:
        for step, (causal, qlens, klens) in enumerate(
            (
                (True, [33, 17], [129, 83]),
                (False, [29, 19], [119, 79]),
                (True, [21, 13], [91, 65]),
            )
        ):
            _plan(
                wrapper,
                qslots,
                kslots,
                qlens,
                klens,
                causal=causal,
                length_device="cuda" if step == 1 else "cpu",
                length_dtype=torch.int32 if step == 0 else torch.int64,
            )
            metadata = wrapper._cudnn_plan.metadata
            assert (metadata.cu_seq_lens_q is not None) == causal
            _fill_valid(inputs, qslots, kslots, qlens, klens)
            if step < 2:
                capture, out, lse = _capture(wrapper, inputs)
                captures.append((capture, out, lse, causal))
            for capture, out, lse, captured_causal in captures:
                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                capture.replay()
                _check(inputs, out, lse, qslots, kslots, qlens, klens, captured_causal)
    finally:
        for capture, *_ in captures:
            capture.reset()


def test_valid_prefix_eager_batch_resize():
    _require_prefix_path()
    wrapper = _wrapper(graph=False)
    for qslots, kslots, qlens, klens in (
        ([64, 32], [160, 96], [33, 17], [129, 83]),
        ([48, 32, 16], [128, 80, 48], [29, 23, 11], [97, 61, 37]),
        ([64], [160], [31], [111]),
    ):
        inputs = _inputs(qtotal=sum(qslots), ktotal=sum(kslots))
        _fill_valid(inputs, qslots, kslots, qlens, klens)
        _plan(wrapper, qslots, kslots, qlens, klens)
        out, lse = wrapper.run(*inputs, return_lse=True, lse_base="ln")
        assert all(
            prefix.numel() == len(qslots) + 1
            for prefix in wrapper._cudnn_mask_prefix_buffers
        )
        _check(inputs, out, lse, qslots, kslots, qlens, klens)


def test_valid_prefix_pointer_alignment():
    _require_prefix_path()
    wrapper = _wrapper(graph=False)
    for batch in (1, 2, 3, 16):
        qlens = [1 + i % 3 for i in range(batch)]
        klens = [9 + i % 5 for i in range(batch)]
        _plan(wrapper, [8] * batch, [16] * batch, qlens, klens)
        metadata = wrapper._cudnn_plan.metadata
        for prefix, lengths in (
            (metadata.cu_seq_lens_q, qlens),
            (metadata.cu_seq_lens_kv, klens),
        ):
            # indptr_tensor declares 16-byte alignment for both mask operands.
            assert prefix.data_ptr() % 16 == 0
            assert prefix.is_contiguous()
            torch.testing.assert_close(prefix.cpu(), _indptr(lengths))
