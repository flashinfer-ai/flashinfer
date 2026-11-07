"""Batch invariance of SM120 sparse-MLA decode (FP8 DSv3.2 / GLM-NSA).

``trtllm_batch_decode_with_kv_cache_mla(..., batch_invariant=True)`` must give
each output row the same bits whatever else shares the call: the other
requests, their order, and the total token count (including counts above the
64-token decode cutoff, which run as several launches).
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm12x_supported

from tests.attention.sparse_mla_test_utils import (
    _ref_sparse_attn,
    dequantize_kv_dsv3_2,
    quantize_kv_dsv3_2,
    quantize_kv_glm_nsa,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="Sparse-MLA SM120 requires SM12x.",
)

_D_QK = 576
_D_V = 512
_PAGE = 64
_NUM_BLOCKS = 64
_TOPK = 2048
_SM_SCALE = _D_QK**-0.5
_QUANTIZE = {"dsv3_2": quantize_kv_dsv3_2, "glm_nsa": quantize_kv_glm_nsa}
_KV_SCALE_FORMAT = {"dsv3_2": "pow2_fp32", "glm_nsa": "arbitrary_fp32"}


@pytest.fixture(scope="module")
def workspace() -> torch.Tensor:
    # One 64-token decode launch at 64 heads needs ~128 MiB of split scratch.
    return torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def _make_cache(fmt: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = (
        torch.randn(
            _NUM_BLOCKS,
            _PAGE,
            1,
            _D_QK,
            device="cuda",
            dtype=torch.bfloat16,
            generator=g,
        )
        / 10.0
    ).clamp(-1, 1)
    packed = _QUANTIZE[fmt](kv)
    return packed.transpose(1, 2), dequantize_kv_dsv3_2(packed)


def _make_rows(
    num_tokens: int, num_heads: int, seed: int, with_lengths: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn(
            num_tokens,
            num_heads,
            _D_QK,
            device="cuda",
            dtype=torch.bfloat16,
            generator=g,
        )
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0,
        _NUM_BLOCKS * _PAGE,
        (num_tokens, _TOPK),
        device="cuda",
        dtype=torch.int32,
        generator=g,
    )
    lengths = (
        torch.randint(
            1,
            _TOPK + 1,
            (num_tokens,),
            device="cuda",
            dtype=torch.int32,
            generator=g,
        )
        if with_lengths
        else None
    )
    return q, indices, lengths


def _decode(
    fmt: str,
    kv_cache: torch.Tensor,
    workspace: torch.Tensor,
    q: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    q_len: int,
    batch_invariant: bool = True,
    backend: str = "sparse",
    return_lse: bool = True,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run flat rows ``[T, H, D]`` as ``T // q_len`` requests of ``q_len``."""
    num_tokens, num_heads, _ = q.shape
    batch = num_tokens // q_len
    result = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=q.view(batch, q_len, num_heads, _D_QK),
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=indices.view(batch, q_len, _TOPK),
        seq_lens=None if lengths is None else lengths.view(batch, q_len),
        max_seq_len=_TOPK,
        sparse_mla_top_k=_TOPK,
        out=out,
        bmm1_scale=_SM_SCALE,
        bmm2_scale=1.0,
        backend=backend,
        lse=lse,
        return_lse=return_lse,
        kv_scale_format=_KV_SCALE_FORMAT[fmt],
        batch_invariant=batch_invariant,
    )
    if not return_lse:
        return result.view(num_tokens, num_heads, _D_V), None
    result_out, result_lse = result
    return (
        result_out.view(num_tokens, num_heads, _D_V),
        result_lse.view(num_tokens, num_heads),
    )


def _take(rows, perm: torch.Tensor):
    return tuple(None if x is None else x[perm].contiguous() for x in rows)


@pytest.mark.parametrize("fmt", ["dsv3_2", "glm_nsa"])
@pytest.mark.parametrize("num_heads", [16, 32, 64])
@pytest.mark.parametrize("q_len", [1, 2, 4])
@pytest.mark.parametrize("with_lengths", [False, True])
def test_request_rows_ignore_other_requests(
    fmt: str, num_heads: int, q_len: int, with_lengths: bool, workspace
) -> None:
    """A request's rows are identical alone, joined by others, and reordered.

    MTP-shaped steps from the issue: one request alone, then with 1, 3 and 19
    others (q_len=4 with 19 others is 80 tokens, past the decode cutoff), with
    the request first and last.
    """
    kv_cache, _ = _make_cache(fmt)
    rows = _make_rows(20 * q_len, num_heads, seed=1, with_lengths=with_lengths)
    target = torch.arange(q_len, device="cuda")
    alone_out, alone_lse = _decode(
        fmt, kv_cache, workspace, *_take(rows, target), q_len=q_len
    )

    for others in (1, 3, 19):
        rest = torch.arange(q_len, (1 + others) * q_len, device="cuda")
        for perm, sl in (
            (torch.cat([target, rest]), slice(0, q_len)),
            (torch.cat([rest, target]), slice(-q_len, None)),
        ):
            out, lse = _decode(
                fmt, kv_cache, workspace, *_take(rows, perm), q_len=q_len
            )
            assert torch.equal(out[sl], alone_out), (others, perm[0].item())
            assert torch.equal(lse[sl], alone_lse), (others, perm[0].item())


@pytest.mark.parametrize("fmt", ["dsv3_2", "glm_nsa"])
def test_flattened_rows_ignore_call_size(fmt: str, workspace) -> None:
    """vLLM-style ``[T, 1]`` rows keep their bits for any T, across launches."""
    num_heads, max_tokens = 32, 200
    kv_cache, kv_dequant = _make_cache(fmt)
    q, indices, lengths = _make_rows(max_tokens, num_heads, seed=2, with_lengths=True)
    full_out, full_lse = _decode(fmt, kv_cache, workspace, q, indices, lengths, q_len=1)

    for n in (1, 2, 8, 63, 64, 65, 128, 129):
        out, lse = _decode(
            fmt, kv_cache, workspace, q[:n], indices[:n], lengths[:n], q_len=1
        )
        assert torch.equal(out, full_out[:n]), n
        assert torch.equal(lse, full_lse[:n]), n

    # Rows on both sides of each 64-token launch boundary, computed alone.
    boundary_rows = (0, 63, 64, 127, 128, 199)
    for r in boundary_rows:
        out, lse = _decode(
            fmt,
            kv_cache,
            workspace,
            q[r : r + 1],
            indices[r : r + 1],
            lengths[r : r + 1],
            q_len=1,
        )
        assert torch.equal(out[0], full_out[r]), r
        assert torch.equal(lse[0], full_lse[r]), r

    no_lse_out, _ = _decode(
        fmt, kv_cache, workspace, q, indices, lengths, q_len=1, return_lse=False
    )
    assert torch.equal(no_lse_out, full_out)

    picked = torch.tensor(boundary_rows, device="cuda")
    ref_out, _ = _ref_sparse_attn(
        q[picked],
        kv_dequant,
        indices[picked],
        _SM_SCALE,
        _D_V,
        topk_length=lengths[picked],
    )
    torch.testing.assert_close(full_out[picked], ref_out, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("fmt", ["dsv3_2", "glm_nsa"])
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("q_len", [1, 2, 4, 8, 64])
def test_single_request_matches_default_route(
    fmt: str, num_heads: int, q_len: int, workspace, monkeypatch
) -> None:
    """Without calibration, a request alone keeps the default route's bits."""
    from flashinfer.mla._sparse_mla_sm120 import _policy, _prepared

    monkeypatch.setattr(_policy, "profile_selection", lambda *args, **kwargs: None)
    monkeypatch.setattr(_policy, "_resolve_cpb", lambda *args, **kwargs: -1)
    monkeypatch.setattr(_prepared, "_functional_plans", {})

    kv_cache, _ = _make_cache(fmt)
    rows = _make_rows(q_len, num_heads, seed=3, with_lengths=True)
    default_out, default_lse = _decode(
        fmt, kv_cache, workspace, *rows, q_len=q_len, batch_invariant=False
    )
    out, lse = _decode(fmt, kv_cache, workspace, *rows, q_len=q_len)
    assert torch.equal(out, default_out)
    assert torch.equal(lse, default_lse)


def _batch_invariant_plans(q_len: int) -> list:
    from flashinfer.mla._sparse_mla_sm120 import _prepared

    # Key slot 6 is batch_invariant_q_len (see _prepared._functional_plan).
    return [p for key, p in _prepared._functional_plans.items() if key[6] == q_len]


@pytest.mark.parametrize("fmt", ["dsv3_2", "glm_nsa"])
@pytest.mark.parametrize("q_len", [65, 128])
def test_long_request_rows_ignore_other_requests(
    fmt: str, q_len: int, workspace, monkeypatch
) -> None:
    """A request past the decode cutoff uses the 64-token split and keeps its bits.

    Its own rows straddle launch boundaries (64 + 1 alone; offset by the other
    requests when joined), so this also covers boundaries inside one request.
    """
    from flashinfer.mla._sparse_mla_sm120 import _prepared

    monkeypatch.setattr(_prepared, "_functional_plans", {})
    num_heads = 32
    kv_cache, kv_dequant = _make_cache(fmt)
    rows = _make_rows(3 * q_len, num_heads, seed=6, with_lengths=True)
    target = torch.arange(q_len, device="cuda")
    alone_out, alone_lse = _decode(
        fmt, kv_cache, workspace, *_take(rows, target), q_len=q_len
    )

    rest = torch.arange(q_len, 3 * q_len, device="cuda")
    for perm, sl in (
        (torch.cat([target, rest]), slice(0, q_len)),
        (torch.cat([rest, target]), slice(-q_len, None)),
    ):
        out, lse = _decode(fmt, kv_cache, workspace, *_take(rows, perm), q_len=q_len)
        assert torch.equal(out[sl], alone_out), perm[0].item()
        assert torch.equal(lse[sl], alone_lse), perm[0].item()

    # Every launch of the long request resolved the clamped 64-token cpb.
    _decode(fmt, kv_cache, workspace, *_take(rows, target[:64]), q_len=64)
    cutoff_cpbs = {int(p.plan.inspect()["cpb"]) for p in _batch_invariant_plans(64)}
    long_cpbs = {int(p.plan.inspect()["cpb"]) for p in _batch_invariant_plans(q_len)}
    assert len(cutoff_cpbs) == 1
    assert long_cpbs == cutoff_cpbs

    q, indices, lengths = _take(rows, target)
    picked = torch.tensor([0, 63, 64, q_len - 1], device="cuda")
    ref_out, _ = _ref_sparse_attn(
        q[picked],
        kv_dequant,
        indices[picked],
        _SM_SCALE,
        _D_V,
        topk_length=lengths[picked],
    )
    torch.testing.assert_close(alone_out[picked], ref_out, atol=5e-2, rtol=5e-2)


_MODEL_TYPE = {"dsv3_2": 0, "glm_nsa": 2}


@pytest.mark.parametrize("fmt", ["dsv3_2", "glm_nsa"])
@pytest.mark.parametrize("forced_route", ["decode_cpb", "prefill"])
def test_calibration_does_not_change_batch_invariant_rows(
    fmt: str, forced_route: str, workspace, monkeypatch
) -> None:
    """A conflicting profile (other cpb, or prefill) steers the default route only."""
    from flashinfer.mla._sparse_mla_sm120 import _policy, _prepared

    q_len, num_heads = 4, 32
    kv_cache, _ = _make_cache(fmt)
    rows = _make_rows(20 * q_len, num_heads, seed=7, with_lengths=True)

    monkeypatch.setattr(_prepared, "_functional_plans", {})
    base_out, base_lse = _decode(fmt, kv_cache, workspace, *rows, q_len=q_len)
    base_cpbs = {int(p.plan.inspect()["cpb"]) for p in _batch_invariant_plans(q_len)}
    assert len(base_cpbs) == 1

    # cpb == chunk count is one split per row, the coarsest legal choice and far
    # from what the wave heuristic picks for a 4-token request.
    forced_cpb = _TOPK // 64
    if forced_route == "decode_cpb":
        forced = _policy.PlannedCall(_policy.KernelVariant.DECODE_SPLITK, forced_cpb)
    else:
        prefill = _policy.prefill_variant(
            _MODEL_TYPE[fmt], num_heads, _TOPK, _PAGE, False, 0
        )
        if prefill is None:
            pytest.skip("no prefill kernel serves this shape")
        forced = _policy.PlannedCall(prefill, -1)
    monkeypatch.setattr(_policy, "profile_selection", lambda *a, **k: forced)
    monkeypatch.setattr(_policy, "_resolve_cpb", lambda *a, **k: forced_cpb)
    monkeypatch.setattr(_prepared, "_functional_plans", {})

    # The patches are live: the default route for one request now uses them.
    _decode(
        fmt,
        kv_cache,
        workspace,
        *_take(rows, torch.arange(q_len, device="cuda")),
        q_len=q_len,
        batch_invariant=False,
    )
    (default_plan,) = _batch_invariant_plans(0)
    assert int(default_plan.plan.inspect()["variant"]) == int(forced.variant)
    if forced_route == "decode_cpb":
        assert int(default_plan.plan.inspect()["cpb"]) == forced_cpb

    out, lse = _decode(fmt, kv_cache, workspace, *rows, q_len=q_len)
    assert torch.equal(out, base_out)
    assert torch.equal(lse, base_lse)
    plans = _batch_invariant_plans(q_len)
    assert {int(p.plan.inspect()["variant"]) for p in plans} == {0}
    assert {int(p.plan.inspect()["cpb"]) for p in plans} == base_cpbs


def test_batch_invariant_plan_skips_calibration_refresh(workspace, monkeypatch) -> None:
    """Warm batch-invariant plans survive calibration epochs and tuning mode."""
    from flashinfer.autotuner import autotune
    from flashinfer.mla._sparse_mla_sm120 import _calibration, _prepared

    fmt, num_heads, num_tokens = "dsv3_2", 32, 130
    monkeypatch.setattr(_prepared, "_functional_plans", {})
    kv_cache, _ = _make_cache(fmt)
    rows = _make_rows(num_tokens, num_heads, seed=8, with_lengths=True)
    warm_out, _ = _decode(fmt, kv_cache, workspace, *rows, q_len=1)
    warm_plans = _batch_invariant_plans(1)
    assert len(warm_plans) == 2  # a 64-row launch and the 2-row tail

    def no_refresh() -> None:
        raise AssertionError("batch-invariant call refreshed the calibration store")

    monkeypatch.setattr(
        _calibration, "_constants_version", _calibration._constants_version + 1
    )
    with autotune(True):
        monkeypatch.setattr(_calibration, "refresh_store", no_refresh)
        out, _ = _decode(fmt, kv_cache, workspace, *rows, q_len=1)
    assert torch.equal(out, warm_out)
    assert all(
        a is b for a, b in zip(_batch_invariant_plans(1), warm_plans, strict=True)
    )


def test_batch_invariant_cuda_graph(workspace) -> None:
    """A multi-launch batch-invariant call captures and replays."""
    fmt, num_heads, num_tokens = "glm_nsa", 32, 130
    kv_cache, _ = _make_cache(fmt)
    q, indices, lengths = _make_rows(num_tokens, num_heads, seed=4, with_lengths=True)
    out = torch.empty(
        num_tokens, 1, num_heads, _D_V, device="cuda", dtype=torch.bfloat16
    )
    lse = torch.empty(num_tokens, num_heads, device="cuda", dtype=torch.float32)

    def run() -> None:
        _decode(
            fmt,
            kv_cache,
            workspace,
            q,
            indices,
            lengths,
            q_len=1,
            out=out,
            lse=lse,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(0.75)
    graph.replay()
    torch.cuda.synchronize()

    eager_out, eager_lse = _decode(
        fmt, kv_cache, workspace, q, indices, lengths, q_len=1
    )
    assert torch.equal(out.view_as(eager_out), eager_out)
    assert torch.equal(lse, eager_lse)


def test_batch_invariant_auto_backend_resolves_to_sparse(workspace) -> None:
    """``backend="auto"`` with sparse top-k reaches the batch-invariant route."""
    fmt, q_len = "glm_nsa", 4
    kv_cache, _ = _make_cache(fmt)
    rows = _make_rows(20 * q_len, 32, seed=9, with_lengths=True)
    sparse_out, sparse_lse = _decode(fmt, kv_cache, workspace, *rows, q_len=q_len)
    auto_out, auto_lse = _decode(
        fmt, kv_cache, workspace, *rows, q_len=q_len, backend="auto"
    )
    assert torch.equal(auto_out, sparse_out)
    assert torch.equal(auto_lse, sparse_lse)


def test_batch_invariant_rejects_other_backends(workspace) -> None:
    kv_cache, _ = _make_cache("dsv3_2")
    q, indices, _ = _make_rows(4, 16, seed=5, with_lengths=False)
    for backend in ("xqa", "cake"):
        with pytest.raises(ValueError, match="batch_invariant"):
            flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(1),
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                qk_nope_head_dim=512,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=indices.unsqueeze(1),
                seq_lens=None,
                max_seq_len=_TOPK,
                sparse_mla_top_k=_TOPK,
                bmm1_scale=_SM_SCALE,
                bmm2_scale=1.0,
                backend=backend,
                batch_invariant=True,
            )


def test_batch_invariant_rejects_glm53_nope(workspace) -> None:
    num_tokens, num_heads = 4, 32
    kv_cache = torch.zeros(4, 1, _PAGE, 656, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="batch_invariant"):
        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=torch.zeros(
                num_tokens, 1, num_heads, 512, dtype=torch.bfloat16, device="cuda"
            ),
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=512,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            block_tables=torch.zeros(
                num_tokens, 1, 256, dtype=torch.int32, device="cuda"
            ),
            seq_lens=None,
            max_seq_len=256,
            sparse_mla_top_k=256,
            bmm1_scale=512**-0.5,
            bmm2_scale=1.0,
            backend="sparse",
            kv_scale_format="arbitrary_fp32",
            batch_invariant=True,
        )
