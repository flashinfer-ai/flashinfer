import pytest
import torch

from flashinfer.utils import get_compute_capability, is_sm100a_supported


@pytest.fixture(autouse=True)
def _require_sm100_and_deep_gemm():
    dev = torch.device("cuda")
    if not is_sm100a_supported(dev) or get_compute_capability(dev) != (10, 0):
        pytest.skip("dsa_indexer_topk is built for sm_100a (compute capability 10.0)")
    from flashinfer.experimental.dsa_indexer import deep_gemm_include

    try:
        deep_gemm_include()
    except ImportError as e:
        pytest.skip(str(e))


def _run(num_q, seq_kv, prefix_len, row_stride=None, positive=False, seed=0, edit=None):
    """Random causal-prefill inputs -> (indices, status, cu_end, dense reference logits)."""
    from flashinfer.dsa_indexer import dsa_indexer_topk

    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        x = torch.randn(*shape, device="cuda", generator=g)
        return x.abs_() if positive else x

    q = (rand(num_q, 32, 128) * 0.5).to(torch.float8_e4m3fn)
    kv = (rand(seq_kv, 128) * 0.5).to(torch.float8_e4m3fn)
    kv_scales = torch.rand((seq_kv + 3) // 4 * 4, device="cuda", generator=g) + 0.5
    weights = rand(num_q, 32) * 0.1
    cu_end = torch.arange(
        seq_kv - num_q + 1, seq_kv + 1, device="cuda", dtype=torch.int32
    )
    if edit is not None:
        kv, kv_scales = edit(kv, kv_scales, cu_end)
    ref = torch.zeros(num_q, seq_kv, device="cuda")
    for h in range(32):
        ref += weights[:, h, None] * torch.relu(q[:, h].float() @ kv.float().T)
    ref *= kv_scales[None, :seq_kv]
    ref.masked_fill_(
        torch.arange(seq_kv, device="cuda")[None] >= cu_end[:, None], float("-inf")
    )
    prefix = torch.empty(num_q, row_stride or prefix_len, device="cuda")[:, :prefix_len]
    prefix.copy_(ref[:, :prefix_len])
    idx, status = dsa_indexer_topk(q, kv, kv_scales, weights, prefix, cu_end)
    return idx, status, cu_end, ref


def _check(idx, status, cu_end, ref):
    assert idx.shape == (cu_end.numel(), 2048) and int(status.abs().max()) == 0
    assert bool(((idx >= 0) & (idx < cu_end[:, None])).all())
    assert bool((idx.sort(dim=1).values.diff(dim=1) > 0).all()), "duplicate index"
    hits = torch.zeros_like(ref, dtype=torch.bool).scatter_(
        1, ref.topk(2048, dim=1).indices, True
    )
    recall = hits.gather(1, idx.long()).float().mean(dim=1)
    assert recall.mean().item() > 0.999 and recall.min().item() > 0.99, (
        recall.min().item()
    )


@pytest.mark.parametrize("seq_kv", [65536, 262144])
@pytest.mark.parametrize("num_q", [128, 1024])
@pytest.mark.parametrize("prefix_len", [8192, 12288])
def test_recall(seq_kv, num_q, prefix_len):
    _check(*_run(num_q, seq_kv, prefix_len))


@pytest.mark.parametrize(
    "num_q,seq_kv,prefix_len,row_stride",
    [(13, 70001, 6001, 6004), (1021, 70001, 8192, 8192), (7, 9000, 2048, 2304)],
)
def test_shape_edges(num_q, seq_kv, prefix_len, row_stride):
    # Ragged q-block, KV tail, partial prefix vectors, padded prefix rows, empty suffixes.
    def empty_suffix(kv, kv_scales, cu_end):
        cu_end[:2] = prefix_len
        return kv, kv_scales

    _check(*_run(num_q, seq_kv, prefix_len, row_stride, seed=1, edit=empty_suffix))


def test_boundary_ties():
    # 3000 identical top scores: the selector must still return 2048 of them.
    def ties(kv, kv_scales, cu_end):
        kv[8192:11192] = kv[8192]
        kv_scales[8192:11192] = 100.0
        return kv, kv_scales

    idx, status, cu_end, _ = _run(128, 65536, 8192, positive=True, seed=2, edit=ties)
    assert int(status.abs().max()) == 0 and bool(
        (idx.sort(dim=1).values.diff(dim=1) > 0).all()
    )
    assert bool(((idx >= 8192) & (idx < 11192)).all())


def test_candidate_overflow_sets_status():
    # Every suffix score beats all earlier ones, so more than cand_cap positions pass.
    def rising(kv, kv_scales, cu_end):
        return kv[:1].expand_as(kv).contiguous(), torch.linspace(
            0.5, 1.5, kv_scales.numel(), device="cuda"
        )

    idx, status, _, _ = _run(64, 65536, 8192, positive=True, seed=3, edit=rising)
    assert bool((status != 0).all()) and bool((idx == -1).all())


def test_empty_batch():
    idx, status, _, _ = _run(0, 16384, 8192)
    assert idx.shape == (0, 2048) and status.shape == (0,)
