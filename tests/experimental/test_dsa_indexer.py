# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team
# Adapted from vLLM PR #65920, commit 833266447780a74713c574e67f3b3bf9caa868eb.
"""LiteTopK public API: ragged selection, GPU recovery, and workspace replay."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _require_sm100_and_deep_gemm():
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    from flashinfer.utils import is_sm100f_supported

    if not is_sm100f_supported(torch.device("cuda")):
        pytest.skip("Requires an SM100-family GPU and CUDA >= 12.9")
    from flashinfer.experimental.dsa_indexer import deep_gemm_include

    try:
        deep_gemm_include()
    except ImportError as e:
        pytest.skip(str(e))
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _inputs(rows, keys, heads=32, seed=913):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, heads, 128, device="cuda", generator=g).to(
        torch.float8_e4m3fn
    )
    kv = torch.randn(keys, 128, device="cuda", generator=g).to(torch.float8_e4m3fn)
    scales = torch.ones(keys, device="cuda")
    weights = torch.randn(rows, heads, device="cuda", generator=g) / heads
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.full((rows,), keys, device="cuda", dtype=torch.int32)
    return q, kv, scales, weights, starts, ends


def _run(inputs, topk=2048, **kwargs):
    from flashinfer.dsa_indexer import dsa_indexer_topk

    q, kv, scales, weights, starts, ends = inputs
    return dsa_indexer_topk(
        q, kv, scales, weights, ends, topk, cu_start=starts, **kwargs
    )


def _workspace(inputs, **kwargs):
    from flashinfer.dsa_indexer import dsa_indexer_topk_workspace

    q, kv = inputs[:2]
    return dsa_indexer_topk_workspace(
        q.device, q.shape[0], q.shape[1], kv.shape[0], **kwargs
    )


def _check(out, inputs, *, scores=True):
    q, kv, scales, weights, starts, ends = inputs
    rows, topk = out.shape
    assert out.dtype == torch.int32 and out.device == q.device
    starts, ends = starts.clamp(0, len(kv)), ends.clamp(0, len(kv))
    visible = (ends - starts).clamp(min=0)
    valid = out >= 0
    assert (valid.sum(1) == visible.clamp(max=topk)).all(), "valid count"
    assert (out[~valid] == -1).all(), "padding must be -1"
    absolute = out.long() + starts[:, None].long()
    assert (
        (~valid) | ((absolute >= starts[:, None]) & (absolute < ends[:, None]))
    ).all()
    ordered = out.sort(dim=1).values
    assert ((ordered[:, 1:] < 0) | (ordered.diff(dim=1) > 0)).all(), "duplicate index"
    if not scores:
        return
    # Keep the independent fp32 reference under 64 MiB, including long contexts.
    step = max(1, (16 * 1024**2) // len(kv))
    kf = kv.float()
    positions = torch.arange(len(kv), device=q.device)[None]
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        logits = torch.zeros(hi - lo, len(kv), device=q.device)
        for head in range(q.shape[1]):
            logits += weights[lo:hi, head, None] * torch.relu(
                q[lo:hi, head].float() @ kf.T
            )
        logits *= scales[None]
        logits.masked_fill_(
            (positions < starts[lo:hi, None]) | (positions >= ends[lo:hi, None]),
            -torch.inf,
        )
        ref = logits.topk(min(topk, len(kv)), dim=1).values
        if len(kv) < topk:
            ref = torch.nn.functional.pad(ref, (0, topk - len(kv)), value=-torch.inf)
        selected = logits.gather(1, absolute[lo:hi].clamp(0, len(kv) - 1))
        selected.masked_fill_(~valid[lo:hi], -torch.inf)
        # Linear-term decomposition differs from direct head reduction near ties.
        torch.testing.assert_close(
            selected.sort(descending=True).values, ref, rtol=2e-4, atol=1e-4
        )


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("keys", [4097, 65536, 1048576])
def test_selection_matches_fp32_reference(heads, topk, keys):
    inputs = _inputs(9, keys, heads)
    inputs[2].copy_(torch.linspace(0.5, 1.5, keys, device="cuda"))
    inputs[-1].sub_(torch.arange(9, device="cuda", dtype=torch.int32))
    _check(_run(inputs, topk), inputs)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_ragged_ranges_clamping_and_padding(heads, topk):
    inputs = _inputs(13, 8193, heads)
    inputs[-2].copy_(
        torch.tensor(
            [-9, 0, 0, 0, 7, 333, 1024, 4096, 8193, 8199, 9, 0, 33], device="cuda"
        )
    )
    inputs[-1].copy_(
        torch.tensor(
            [0, 1, 511, 512, 1030, 2381, 4096, 8193, 8200, 8200, 2, -7, 9000],
            device="cuda",
        )
    )
    _check(_run(inputs, topk), inputs)


@pytest.mark.parametrize("heads", [32, 64])
def test_default_start_and_fewer_keys_than_topk(heads):
    from flashinfer.dsa_indexer import dsa_indexer_topk

    inputs = _inputs(5, 7, heads)
    q, kv, scales, weights, _, ends = inputs
    _check(dsa_indexer_topk(q, kv, scales, weights, ends), inputs)


def test_empty_batch():
    inputs = _inputs(0, 4096)
    out = _run(inputs, 512)
    assert out.shape == (0, 512) and out.dtype == torch.int32


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_period_aligned_keys_trigger_exact_gpu_recovery(heads, topk):
    # The second packed request repeats at the sample stride (65), making its
    # sampled threshold miss high keys or admit too many candidates.
    rows, keys = 128, 16384
    inputs = _inputs(2 * rows, 2 * keys, heads, seed=929)
    q, kv, scales, weights, starts, ends = inputs
    kf = kv.float()
    kf[keys:] = kf[keys : keys + 65].repeat((keys + 64) // 65, 1)[:keys]
    kv.copy_(kf.to(kv.dtype))
    starts[rows:] = keys
    causal = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    ends.copy_(starts + causal.repeat(2))
    workspace = _workspace(inputs)
    out = _run(inputs, topk, workspace=workspace)
    _check(out, inputs)
    counts, _, overflow = workspace[4][:, rows:]
    recovered = (counts < topk) | (counts > workspace[2].shape[1]) | (overflow != 0)
    assert recovered.any(), "adversarial sample did not exercise recovery"
    assert torch.count_nonzero(workspace[0]) == 0


@pytest.mark.parametrize("heads", [32, 64])
def test_zero_weight_ties_recover_without_invalid_rows(heads):
    inputs = _inputs(9, 16384, heads)
    inputs[3].zero_()
    workspace = _workspace(inputs, cand_cap=4096)
    out = _run(inputs, 2048, cand_cap=4096, workspace=workspace)
    _check(out, inputs)
    assert (workspace[4][0] > 4096).all()
    assert torch.count_nonzero(workspace[0]) == 0


@pytest.mark.parametrize("shared", [8.0, 256.0])
def test_shifted_threshold_with_common_score_offset(shared):
    inputs = _inputs(128, 65536, seed=5)
    q, kv, _, weights, _, _ = inputs
    qf, kf = q.float(), kv.float()
    qf[:, :, 0], kf[:, 0] = shared, shared
    q.copy_(qf.to(q.dtype))
    kv.copy_(kf.to(kv.dtype))
    weights.fill_(1.0 / 32)
    workspace = _workspace(inputs)
    out = _run(inputs, workspace=workspace)
    _check(out, inputs)
    counts, _, overflow = workspace[4]
    assert ((counts >= 2048) & (counts <= workspace[2].shape[1])).all()
    assert (overflow == 0).all()
    assert counts.float().median() < 3 * 2048


@pytest.mark.parametrize("heads", [32, 64])
def test_row_windows_reuse_workspace_and_cover_partial_tail(heads):
    inputs = _inputs(4099, 2051, heads)
    workspace = _workspace(inputs)
    assert workspace[2].shape[0] == 4096
    out = _run(inputs, 512, workspace=workspace)
    _check(out, inputs, scores=False)
    # Check both sides of the window boundary and the ragged final block.
    rows = torch.tensor([0, 4095, 4096, 4098], device="cuda")
    sample = tuple(t[rows] if i in (0, 3, 4, 5) else t for i, t in enumerate(inputs))
    _check(out[rows], sample)
    assert torch.count_nonzero(workspace[0]) == 0


@pytest.mark.parametrize("heads", [32, 64])
def test_workspace_reuse_for_smaller_shapes(heads):
    inputs = _inputs(16, 65536, heads)
    workspace = _workspace(inputs)
    _check(_run(inputs, 512, workspace=workspace), inputs)
    smaller = _inputs(9, 4097, heads, seed=919)
    _check(_run(smaller, 1024, workspace=workspace), smaller)
    _check(_run(inputs, 2048, workspace=workspace), inputs)
    assert torch.count_nonzero(workspace[0]) == 0


@pytest.mark.parametrize("heads", [32, 64])
def test_cuda_graph_replay_with_changed_inputs_and_recovery(heads):
    inputs = _inputs(16, 16384, heads)
    workspace = _workspace(inputs, cand_cap=4096)
    out = torch.empty(16, 512, device="cuda", dtype=torch.int32)

    def run():
        return _run(inputs, 512, out=out, cand_cap=4096, workspace=workspace)

    assert run() is out  # Warm compilation and per-device launch setup before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(3):
        if iteration == 1:
            inputs[3].zero_()  # All rows overflow and must recover during replay.
        elif iteration == 2:
            inputs[3].copy_(_inputs(16, 16384, heads, seed=929)[3])
            inputs[-1][::2] = 17  # Change valid counts without recapturing.
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        _check(out, inputs)
        assert torch.count_nonzero(workspace[0]) == 0


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "strides", "alignment", "capacity", "topk"]
)
def test_rejects_invalid_inputs(bad):
    inputs = list(_inputs(8, 4096))
    kwargs = {}
    topk = 512
    if bad == "dtype":
        inputs[2] = inputs[2].half()
    elif bad == "shape":
        inputs[2] = inputs[2][:-1]
    elif bad == "strides":
        inputs[3] = torch.empty(8, 64, device="cuda")[:, ::2]
    elif bad == "alignment":
        inputs[2] = torch.empty(4100, device="cuda")[1:4097]
    elif bad == "capacity":
        kwargs["cand_cap"] = 4097
    else:
        topk = 768
    with pytest.raises((ValueError, RuntimeError)):
        _run(inputs, topk, **kwargs)


def test_rejects_misaligned_caller_workspace():
    inputs = _inputs(8, 4096)
    workspace = list(_workspace(inputs))
    rows, cap = workspace[3].shape
    workspace[3] = torch.empty(rows * cap + 4, device="cuda", dtype=torch.int32)[
        1 : 1 + rows * cap
    ].view(rows, cap)
    with pytest.raises((ValueError, RuntimeError), match="align"):
        _run(inputs, workspace=tuple(workspace))
