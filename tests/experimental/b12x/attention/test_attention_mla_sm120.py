"""Dispatch smoke tests for the SM120 sparse-MLA backend.

These tests do NOT exercise the SM120 sparse MLA kernel itself. They prove the
active SM120 path is wired into the MLA APIs:

  * a tiny GLM (q_head_dim==576) decode routes into the promoted SM120
    run_unified_decode entrypoint; and
  * backend="legacy" is rejected because the legacy sparse MLA kernels are no
    longer wired through public dispatch.

Scope decisions (.sm120port/scope_decisions.md): DSV3.2 is dropped; GLM_NSA is the
uncompressed q=576 contract.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.testing.reference import dsv4_extra_ref, dsv4_ref, glm_ref, prefill_ref

import b12x.attention._shared.mla.api as mla_api
from b12x.attention._shared.mla.api import (
    sparse_mla_decode_forward as _sparse_mla_decode_forward,
    sparse_mla_extend_forward as _sparse_mla_extend_forward,
)
from b12x.attention._shared.mla.compressed_api import (
    compressed_sparse_mla_decode_forward as _compressed_sparse_mla_decode_forward,
)
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_page_nbytes,
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
)
from b12x._lib.intrinsics import get_sm_version
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLAScratchCaps,
    _compressed_sparse_mla_scratch_layout,
    _materialize_compressed_sparse_mla_scratch,
)

from b12x.testing.reference.helpers import require_b12x as _require_b12x


# GLM_NSA uncompressed decode contract: q_head_dim = d_nope + d_rope = 512 + 64 = 576,
# v_head_dim = 512, packed KV cache = 656 bytes/token (verified_traits.md).
_GLM_Q_HEAD_DIM = 576
_GLM_V_HEAD_DIM = 512
_GLM_KV_BYTES_PER_TOKEN = 656
_NUM_Q_HEADS = 8
_SM_SCALE = 1.0 / math.sqrt(_GLM_Q_HEAD_DIM)


def test_cache_block_stride_distinguishes_flat_contiguous_and_packed_views() -> None:
    import b12x.attention._shared.mla.kernel as decode_kernel
    import b12x.attention._shared.mla.prefill as prefill_dispatch
    import b12x.attention._shared.mla.prefill_mg as prefill_mg
    from b12x.attention._shared.mla.traits import ModelType

    page_size = 64
    payload = page_size * _GLM_KV_BYTES_PER_TOKEN
    # This is the token-major rank-3 shape used by the GLM reference and
    # benchmark. Its stride(0) is one token, not one physical cache block.
    contiguous = torch.empty(
        (2 * page_size, 1, _GLM_KV_BYTES_PER_TOKEN), dtype=torch.uint8
    )
    physical_stride = payload + 256
    packed_storage = torch.empty((2, physical_stride), dtype=torch.uint8)
    packed_view = packed_storage[:, :payload]
    packed_rank3 = torch.as_strided(
        packed_storage,
        size=(2, page_size, _GLM_KV_BYTES_PER_TOKEN),
        stride=(physical_stride, _GLM_KV_BYTES_PER_TOKEN, 1),
    )
    assert contiguous.is_contiguous()
    assert not packed_view.is_contiguous()
    assert not packed_rank3.is_contiguous()
    assert mla_api._is_supported_packed_kv_cache_view(
        packed_rank3,
        page_size=page_size,
    )

    helpers = (
        lambda cache: decode_kernel._cache_block_stride_bytes(
            cache,
            page_size=page_size,
            model_type=int(ModelType.GLM_NSA),
        ),
        lambda cache: prefill_dispatch._cache_block_stride_bytes(
            cache,
            page_size=page_size,
            model_type=ModelType.GLM_NSA,
        ),
        lambda cache: prefill_mg._cache_block_stride_bytes(
            cache,
            page_size=page_size,
            is_glm=True,
        ),
    )
    for resolve in helpers:
        assert resolve(contiguous) == payload
        assert resolve(packed_view) == physical_stride

    expected_span = physical_stride + payload
    assert decode_kernel._cache_base_tensor(packed_rank3).numel() == expected_span
    assert prefill_mg._cache_base_tensor(packed_rank3).numel() == expected_span


def sparse_mla_decode_forward(
    *,
    workspace=None,
    q_all=None,
    page_table_1=None,
    cache_seqlens_int32=None,
    nsa_cache_seqlens_int32=None,
    **kwargs,
):
    if workspace is not None:
        binding = workspace.bind_sparse_mla(
            q=q_all,
            selected_indices=page_table_1,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )
        return _sparse_mla_decode_forward(binding=binding, **kwargs)
    return _sparse_mla_decode_forward(
        q_all=q_all,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        **kwargs,
    )


def sparse_mla_extend_forward(
    *,
    workspace=None,
    q_all=None,
    selected_token_offsets=None,
    cache_seqlens_int32=None,
    nsa_cache_seqlens_int32=None,
    **kwargs,
):
    if workspace is not None:
        binding = workspace.bind_sparse_mla(
            q=q_all,
            selected_indices=selected_token_offsets,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )
        return _sparse_mla_extend_forward(binding=binding, **kwargs)
    return _sparse_mla_extend_forward(
        q_all=q_all,
        selected_token_offsets=selected_token_offsets,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        **kwargs,
    )


def compressed_sparse_mla_decode_forward(
    *,
    workspace=None,
    q_all=None,
    swa_indices=None,
    swa_topk_lengths=None,
    indexed_indices=None,
    indexed_topk_lengths=None,
    indexed_page_table=None,
    **kwargs,
):
    if workspace is not None:
        binding = workspace.bind(
            q=q_all,
            swa_indices=swa_indices,
            swa_lengths=swa_topk_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_topk_lengths,
            indexed_page_table=indexed_page_table,
        )
        return _compressed_sparse_mla_decode_forward(binding=binding, **kwargs)
    return _compressed_sparse_mla_decode_forward(
        q_all=q_all,
        swa_indices=swa_indices,
        swa_topk_lengths=swa_topk_lengths,
        indexed_indices=indexed_indices,
        indexed_topk_lengths=indexed_topk_lengths,
        indexed_page_table=indexed_page_table,
        **kwargs,
    )


def require_b12x_sparse_mla() -> torch.device:
    """Skip unless a real SM120+ device is present (mirrors the compressed-MLA pattern).

    The routing branch under test is only reachable on SM120 hardware.
    """
    device = _require_b12x()
    if get_sm_version(device) < 120:
        pytest.skip("SM120 sparse MLA dispatch requires an SM120+ device")
    return device


def _make_glm_decode_inputs(device: torch.device, num_q_heads: int = _NUM_Q_HEADS):
    """Build a tiny GLM (q_head_dim==576) sparse-decode call and its legacy workspace."""
    rows = 1
    width = 4
    q_all = torch.zeros(
        (rows, num_q_heads, _GLM_Q_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv_cache = torch.zeros(
        (16, 1, _GLM_KV_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
    )
    page_table_1 = torch.zeros((rows, width), dtype=torch.int32, device=device)
    cache_seqlens = torch.full((rows,), width, dtype=torch.int32, device=device)
    from b12x.attention._shared.workspace import B12XAttentionWorkspace

    workspace = B12XAttentionWorkspace.for_contract(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_q_heads,
        head_dim=_GLM_Q_HEAD_DIM,
        v_head_dim=_GLM_V_HEAD_DIM,
        topk=width,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=rows * width,
        use_cuda_graph=False,
    )
    return q_all, kv_cache, page_table_1, cache_seqlens, workspace


def _force_legacy_reference(monkeypatch) -> dict[str, int]:
    """Count any unexpected PyTorch reference fallback use."""
    counters = {"reference_calls": 0}

    def fake_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts,
        sm_scale,
        v_head_dim,
        **kwargs,
    ):
        del kv_cache, page_table_1, active_token_counts, sm_scale, kwargs
        counters["reference_calls"] += 1
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(mla_api, "sparse_mla_reference", fake_reference)
    return counters


@torch.inference_mode()
def test_glm_decode_default_routes_to_sm120(monkeypatch) -> None:
    """No backend kwarg: a tiny GLM decode routes to the active SM120 path.

    SM120+ CUDA dispatch routes into SM120 sparse MLA.run_unified_decode (NOT the
    legacy reference).
    We monkeypatch run_unified_decode to a sentinel so the routing decision is observed
    without compiling a kernel for tiny inputs, and pin the legacy reference so any
    accidental legacy fallthrough would be counted (it must stay zero).
    """
    device = require_b12x_sparse_mla()
    counters = _force_legacy_reference(monkeypatch)

    routed = {"calls": 0}

    def fake_run_unified_decode(*, q_all, **kwargs):
        del kwargs
        routed["calls"] += 1
        return q_all[:, :, :_GLM_V_HEAD_DIM].clone()

    import b12x.attention._shared.mla.kernel as unified_launch

    monkeypatch.setattr(unified_launch, "run_unified_decode", fake_run_unified_decode)

    # HPB=16 contract: heads must be divisible by 16 for the unified GLM route.
    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(
        device, num_q_heads=16
    )
    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
    )

    assert output.shape == (1, 16, _GLM_V_HEAD_DIM)
    # The unified GLM launcher intercepted; the legacy reference did NOT run.
    assert routed["calls"] == 1
    assert counters["reference_calls"] == 0


@torch.inference_mode()
def test_glm_decode_backend_legacy_is_retired() -> None:
    device = require_b12x_sparse_mla()

    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(
        device
    )
    with pytest.raises(ValueError, match="legacy sparse MLA kernels have been retired"):
        sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=page_table_1,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=cache_seqlens,
            workspace=workspace,
            sm_scale=_SM_SCALE,
            v_head_dim=_GLM_V_HEAD_DIM,
            backend="legacy",
        )


@torch.inference_mode()
def test_glm_decode_valid_contract_routes_to_sm120(monkeypatch) -> None:
    """A valid GLM contract routes into SM120 sparse MLA.run_unified_decode.

    The launcher itself is exercised end-to-end by .sm120port/probes/
    glm_decode_e2e_check.py vs glm_ref; here we only prove the GATE intercepts a
    valid GLM contract (heads % HPB == 0, single row, no LSE/sink) and that the
    legacy reference does NOT run. We monkeypatch run_unified_decode to a sentinel
    so the routing decision is observed without compiling a kernel for tiny inputs.
    """
    device = require_b12x_sparse_mla()
    counters = _force_legacy_reference(monkeypatch)

    routed = {"calls": 0}

    def fake_run_unified_decode(*, q_all, **kwargs):
        del kwargs
        routed["calls"] += 1
        return q_all[:, :, :_GLM_V_HEAD_DIM].clone()

    import b12x.attention._shared.mla.kernel as unified_launch

    monkeypatch.setattr(unified_launch, "run_unified_decode", fake_run_unified_decode)

    # HPB=16 contract: heads must be divisible by 16 for the unified GLM route.
    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(
        device, num_q_heads=16
    )
    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
    )

    assert output.shape == (1, 16, _GLM_V_HEAD_DIM)
    # The unified GLM launcher intercepted; the legacy reference did NOT run.
    assert routed["calls"] == 1
    assert counters["reference_calls"] == 0


# ── P10 (3a) PREFILL routing: extend/verify/draft_extend -> run_unified_prefill ──
def _make_glm_extend_inputs(
    device: torch.device, num_q_heads: int = _NUM_Q_HEADS, mode: str = "extend"
):
    """A tiny GLM (q=576) prefill-like (extend) call + its workspace."""
    rows = 1
    width = 4
    q_all = torch.zeros(
        (rows, num_q_heads, _GLM_Q_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv_cache = torch.zeros(
        (16, 1, _GLM_KV_BYTES_PER_TOKEN), dtype=torch.uint8, device=device
    )
    selected = torch.zeros((rows, width), dtype=torch.int32, device=device)
    cache_seqlens = torch.full((rows,), width, dtype=torch.int32, device=device)
    from b12x.attention._shared.workspace import B12XAttentionWorkspace

    workspace = B12XAttentionWorkspace.for_contract(
        mode=mode,
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=num_q_heads,
        head_dim=_GLM_Q_HEAD_DIM,
        v_head_dim=_GLM_V_HEAD_DIM,
        topk=width,
        max_total_q=rows,
        max_batch=rows,
        max_kv_rows=rows * width,
        use_cuda_graph=False,
    )
    return q_all, kv_cache, selected, cache_seqlens, workspace


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["extend", "verify", "draft_extend"])
def test_glm_prefill_mode_routes_to_unified_prefill(monkeypatch, mode) -> None:
    """(3a) GLM extend/verify/draft_extend + flag on: the prefill-like modes route
    into SM120 sparse MLA run_unified_prefill (NOT run_unified_decode, NOT legacy).

    The launcher is exercised end-to-end by .sm120port/probes/glm_prefill_e2e_check.py
    vs glm_prefill_ref; here we prove ONLY the GATE intercepts a prefill-like GLM
    contract and routes to prefill (not decode / not legacy split)."""
    device = require_b12x_sparse_mla()
    counters = _force_legacy_reference(monkeypatch)

    routed = {"prefill": 0, "decode": 0}

    def fake_run_unified_prefill(*, q, output=None, **kwargs):
        del kwargs
        routed["prefill"] += 1
        if output is not None:
            output.zero_()
            out = output
        else:
            out = q[:, :, :_GLM_V_HEAD_DIM].clone()
        lse = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
        return out, lse

    def fake_run_unified_decode(*, q_all, **kwargs):
        del kwargs
        routed["decode"] += 1
        return q_all[:, :, :_GLM_V_HEAD_DIM].clone()

    import b12x.attention._shared.mla.kernel as unified_launch

    monkeypatch.setattr(unified_launch, "run_unified_prefill", fake_run_unified_prefill)
    monkeypatch.setattr(unified_launch, "run_unified_decode", fake_run_unified_decode)

    q_all, kv_cache, selected, cache_seqlens, workspace = _make_glm_extend_inputs(
        device, num_q_heads=16, mode=mode
    )
    output = sparse_mla_extend_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        selected_token_offsets=selected,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
    )

    assert output.shape == (1, 16, _GLM_V_HEAD_DIM)
    assert routed["prefill"] == 1  # prefill intercepted
    assert routed["decode"] == 0  # NOT decode
    assert counters["reference_calls"] == 0  # NOT legacy


@torch.inference_mode()
def test_glm_decode_mode_still_routes_to_unified_decode(monkeypatch) -> None:
    """(3a) The decode mode still routes to run_unified_decode (prefill routing is
    gated on workspace.mode -- decode must NOT regress to prefill)."""
    device = require_b12x_sparse_mla()
    _force_legacy_reference(monkeypatch)

    routed = {"prefill": 0, "decode": 0}

    def fake_run_unified_prefill(*, q, **kwargs):
        del q, kwargs
        routed["prefill"] += 1
        raise AssertionError("decode mode must not route to prefill")

    def fake_run_unified_decode(*, q_all, **kwargs):
        del kwargs
        routed["decode"] += 1
        return q_all[:, :, :_GLM_V_HEAD_DIM].clone()

    import b12x.attention._shared.mla.kernel as unified_launch

    monkeypatch.setattr(unified_launch, "run_unified_prefill", fake_run_unified_prefill)
    monkeypatch.setattr(unified_launch, "run_unified_decode", fake_run_unified_decode)

    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(
        device, num_q_heads=16
    )
    sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
    )
    assert routed["decode"] == 1
    assert routed["prefill"] == 0


# ── DSV4 MAIN-CACHE compressed decode (P7): real kernel + split-K + merge ──────
_DSV4_HEADS = 32  # local q heads (4 native head blocks of HPB=8)
_DSV4_HEAD_DIM = 512
_DSV4_PAGE = 64  # compressed page_size for the test cache
_DSV4_SM_SCALE = 1.0 / math.sqrt(_DSV4_HEAD_DIM)


def _make_dsv4_compressed_case(device, *, topk, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    num_blocks = 16
    n_tokens = num_blocks * _DSV4_PAGE
    k_nope = (
        torch.randn((n_tokens, 448), generator=gen, dtype=torch.float32, device=device)
        / 10
    ).clamp(-1, 1)
    k_rope = (
        torch.randn((n_tokens, 64), generator=gen, dtype=torch.float32, device=device)
        / 10
    ).clamp(-1, 1)
    cache = pack_compressed_sparse_mla_kv_cache_reference(
        k_nope, k_rope.to(torch.bfloat16), page_size=_DSV4_PAGE, num_pages=num_blocks
    )
    q = (
        (
            torch.randn(
                (1, _DSV4_HEADS, _DSV4_HEAD_DIM),
                generator=gen,
                dtype=torch.float32,
                device=device,
            )
            / 10
        )
        .clamp(-1, 1)
        .to(torch.bfloat16)
    )
    idx = torch.randint(
        0, n_tokens, (1, topk), generator=gen, dtype=torch.int32, device=device
    )
    idx[:, topk // 2 :] = -1  # invalidate the back half (matches dsv4_ref cases)
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    return q, cache, idx, lengths


def _make_dsv4_scratch(device, *, topk, max_chunks):
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=_DSV4_HEADS,
        max_q_rows=1,
        max_width=topk,
        head_dim=_DSV4_HEAD_DIM,
        v_head_dim=_DSV4_HEAD_DIM,
        max_chunks_per_row=max_chunks,
        page_size=_DSV4_PAGE,
    )
    layout = _compressed_sparse_mla_scratch_layout(caps)
    storage = torch.zeros(int(layout.nbytes), dtype=torch.uint8, device=device)
    return _materialize_compressed_sparse_mla_scratch(caps, storage, layout)


@torch.inference_mode()
@pytest.mark.parametrize("topk", [64, 128, 512])
def test_dsv4_compressed_decode_routes_to_sm120_and_matches_reference(
    monkeypatch, topk
) -> None:
    """DSV4 main-cache contract: compressed_sparse_mla_decode_forward routes to
    SM120 sparse MLA.run_unified_decode (kernel split-K partials + reused base-2 merge)
    and matches compressed_sparse_mla_reference."""
    device = require_b12x_sparse_mla()
    del monkeypatch

    q, cache, idx, lengths = _make_dsv4_compressed_case(device, topk=topk, seed=topk)
    scratch = _make_dsv4_scratch(device, topk=topk, max_chunks=8)

    out = compressed_sparse_mla_decode_forward(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=_DSV4_SM_SCALE,
        swa_page_size=_DSV4_PAGE,
    )
    torch.cuda.synchronize()
    assert out.shape == (1, _DSV4_HEADS, _DSV4_HEAD_DIM)

    import b12x.attention._shared.mla.kernel as launch

    plan = launch.LAST_DECODE_PLAN
    assert plan.get("native_dsv4_h8") is True
    assert plan.get("heads_per_block") == 8
    assert plan.get("math_warps") == 4
    assert plan.get("block_threads") == 160
    assert plan.get("kv_stage_packed") is True
    assert plan.get("kv_smem_stride") == 592
    assert plan.get("qk_candidates_per_warp") == 16
    assert plan.get("qk_swap_ab") is True

    exp = compressed_sparse_mla_reference(
        q, cache, idx, lengths, sm_scale=_DSV4_SM_SCALE, swa_page_size=_DSV4_PAGE
    )[0].float()
    got = out[0].float()
    cos = float(
        (got.flatten().double() @ exp.flatten().double())
        / (got.flatten().double().norm() * exp.flatten().double().norm())
    )
    assert cos > 0.999, f"DSV4 unified decode cos={cos}"
    assert (got - exp).abs().max().item() < 2e-2


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["extend", "verify", "draft_extend"])
def test_dsv4_compressed_prefill_mode_routes_to_unified_prefill(mode) -> None:
    device = require_b12x_sparse_mla()
    topk = 512
    q, cache, indices, lengths = _make_dsv4_compressed_case(
        device, topk=topk, seed=topk
    )
    scratch = _make_dsv4_scratch(
        device, topk=topk, max_chunks=8
    )
    scratch.mode = mode
    output = compressed_sparse_mla_decode_forward(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=indices,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=_DSV4_SM_SCALE,
        swa_page_size=_DSV4_PAGE,
    )
    expected = compressed_sparse_mla_reference(
        q, cache, indices, lengths, sm_scale=_DSV4_SM_SCALE,
        swa_page_size=_DSV4_PAGE,
    )
    torch.testing.assert_close(output.float(), expected.float(), atol=2e-2, rtol=2e-2)




@torch.inference_mode()
def test_dsv4_compressed_decode_extra_cache_routes_to_unified(monkeypatch) -> None:
    """has_extra_cache (indexed/extra-tokens, P7c): the gate routes the DSV4
    dual-cache to SM120 and the result matches dsv4_extra_ref.dsv4_extra_decode_reference
    over the UNION of the main + extra topk rows.

    A mapped indexed_page_table raises because the active gather addresses the
    extra cache by raw slot id; that guard is checked separately."""
    device = require_b12x_sparse_mla()
    del monkeypatch

    topk, extra_topk, pbs_extra = 64, 128, 2
    main_blocks = 16
    case = dsv4_extra_ref.make_dsv4_extra_decode_case(
        num_heads=_DSV4_HEADS,
        topk=topk,
        extra_topk=extra_topk,
        num_tokens=1,
        num_blocks=main_blocks,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        invalidate_half=True,
        with_sink=False,
        device=device,
        seed=11,
    )
    q = case["q"].contiguous()
    swa_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, main_blocks)
    extra_blocks = case["extra_kv_cache"].shape[0]
    idx_cache = _repack_dsv4_to_compressed(
        case["extra_kv_cache"], pbs_extra, extra_blocks
    )
    main_idx = case["topk_indices"].contiguous()
    extra_idx = case["extra_indices"].contiguous()
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    extra_lengths = torch.full((1,), extra_topk, dtype=torch.int32, device=device)
    exp_O = case["expected_O"][0].float()

    n_chunks = (topk + 64 - 1) // 64 + (extra_topk + 64 - 1) // 64
    scratch = _make_dsv4_scratch(
        device, topk=topk + extra_topk, max_chunks=max(8, n_chunks)
    )

    out = compressed_sparse_mla_decode_forward(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=main_idx,
        swa_topk_lengths=lengths,
        indexed_k_cache=idx_cache,
        indexed_indices=extra_idx,
        indexed_topk_lengths=extra_lengths,
        indexed_page_size=pbs_extra,
        workspace=scratch,
        sm_scale=case["sm_scale"],
        swa_page_size=_DSV4_PAGE,
    )
    torch.cuda.synchronize()
    assert out.shape == (1, _DSV4_HEADS, _DSV4_HEAD_DIM)

    got = out[0].float()
    cos = float(
        (got.flatten().double() @ exp_O.flatten().double())
        / (got.flatten().double().norm() * exp_O.flatten().double().norm())
    )
    assert cos > 0.999, f"DSV4 dual-cache unified decode cos={cos}"
    assert (got - exp_O).abs().max().item() < 2e-2


@torch.inference_mode()
def test_dsv4_compressed_decode_mapped_extra_page_table_raises() -> None:
    """(d') has_extra_cache WITH a mapped indexed_page_table is GENUINELY-UPSTREAM-
    UNSUPPORTED (upstream FlashInfer addresses the extra cache by raw slot id only).
    Per the no-legacy-fallback directive the dispatch must RAISE a clear error, not
    silently route to legacy."""
    device = require_b12x_sparse_mla()
    topk = 64
    q, cache, idx, lengths = _make_dsv4_compressed_case(device, topk=topk, seed=7)
    scratch = _make_dsv4_scratch(device, topk=topk * 2, max_chunks=8)
    scratch.fixed_capacity = False
    page_table = torch.zeros((1, topk), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="mapped"):
        compressed_sparse_mla_decode_forward(
            q_all=q,
            swa_k_cache=cache,
            swa_indices=idx,
            swa_topk_lengths=lengths,
            indexed_k_cache=cache,
            indexed_indices=idx,
            indexed_topk_lengths=lengths,
            indexed_page_size=_DSV4_PAGE,
            indexed_page_table=page_table,
            workspace=scratch,
            sm_scale=_DSV4_SM_SCALE,
            swa_page_size=_DSV4_PAGE,
        )


@torch.inference_mode()
def test_dsv4_compressed_decode_partial_extra_trio_raises() -> None:
    """A partial dual-cache trio (some-but-not-all indexed_* args) is a HARD ERROR
    matching upstream's required-together ICHECK (sparse_mla_sm120.cu:171-174)."""
    device = require_b12x_sparse_mla()
    topk = 64
    q, cache, idx, lengths = _make_dsv4_compressed_case(device, topk=topk, seed=11)
    scratch = _make_dsv4_scratch(device, topk=topk * 2, max_chunks=8)
    scratch.fixed_capacity = False

    with pytest.raises(ValueError, match="(?i)dual-cache|together|trio"):
        compressed_sparse_mla_decode_forward(
            q_all=q,
            swa_k_cache=cache,
            swa_indices=idx,
            swa_topk_lengths=lengths,
            indexed_k_cache=cache,  # extra cache provided
            indexed_indices=idx,  # extra indices provided
            indexed_topk_lengths=None,  # MISSING -> partial trio
            indexed_page_size=_DSV4_PAGE,
            workspace=scratch,
            sm_scale=_DSV4_SM_SCALE,
            swa_page_size=_DSV4_PAGE,
        )


@torch.inference_mode()
def test_glm_decode_backend_kwarg_routes_to_unified(monkeypatch) -> None:
    """(b') backend="sm120" routes the GLM decode to SM120 sparse MLA even with
    an explicit backend kwarg (GLM_NSA implemented; the gate routes via run_unified_decode)."""
    device = require_b12x_sparse_mla()
    counters = _force_legacy_reference(monkeypatch)

    routed = {"calls": 0}

    def fake_run_unified_decode(*, q_all, **kwargs):
        del kwargs
        routed["calls"] += 1
        return q_all[:, :, :_GLM_V_HEAD_DIM].clone()

    import b12x.attention._shared.mla.kernel as unified_launch

    monkeypatch.setattr(unified_launch, "run_unified_decode", fake_run_unified_decode)

    q_all, kv_cache, page_table_1, cache_seqlens, workspace = _make_glm_decode_inputs(
        device, num_q_heads=16
    )
    output = sparse_mla_decode_forward(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        workspace=workspace,
        sm_scale=_SM_SCALE,
        v_head_dim=_GLM_V_HEAD_DIM,
        backend="sm120",
    )

    assert output.shape == (1, 16, _GLM_V_HEAD_DIM)
    assert routed["calls"] == 1
    assert counters["reference_calls"] == 0


# ── DSV4 LAUNCHER NUMERICS (P7): run_unified_decode vs dsv4_ref ────────────────
# These exercise the REAL launcher (b12x.attention._shared.mla.kernel.
# run_unified_decode = warp-specialized 288-thread DSV4 decode -> per-split
# NORMALIZED partials in mid_out/mid_lse -> the REUSED split.py base-2 merge) for
# num_heads=128, topk in {64,128,512}, and BOTH num_splits=1 and a forced>1
# split, comparing final O + base-2 LSE to the pure-PyTorch dsv4_ref oracle at
# the validated P5 gate (cos > 0.999, O atol 2e-2, lse atol 5e-2).
_UNIFIED_NUM_HEADS = 128  # full DSV4 head count (8 head blocks of HPB=16)
_UNIFIED_NUM_BLOCKS = 64
_LN2 = math.log(2.0)


def _cosine(got: torch.Tensor, exp: torch.Tensor) -> float:
    a = got.flatten().double()
    b = exp.flatten().double()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else float((a @ b).item() / denom)


def _repack_dsv4_to_compressed(
    packed_dsv4: torch.Tensor, page_size: int, num_blocks: int
) -> torch.Tensor:
    """Re-lay the dsv4_ref (nb,bs,1,584) cache into the compressed flat
    [pages, page_nbytes] layout the launcher reads (swa_k_cache.reshape(-1) with
    per-page stride = compressed_sparse_mla_page_nbytes(page_size)).

    The data (pbs*576) + footer (pbs*8) = pbs*584 bytes are byte-identical
    between the two packings (the verified P7 layout finding); only the per-page
    byte stride is padded to a 576-multiple, so we copy the dsv4 page bytes into
    the leading (pbs*584) of each compressed (padded) page.
    """
    bs = page_size
    bpt = dsv4_ref.DSV4_KV_GMEM_STRIDE  # 584
    page_nbytes = compressed_sparse_mla_page_nbytes(page_size)
    flat = packed_dsv4.reshape(num_blocks, bs * bpt)
    out = torch.zeros(
        num_blocks, page_nbytes, dtype=torch.uint8, device=packed_dsv4.device
    )
    out[:, : bs * bpt] = flat
    return out


def _merge_base2_lse(mid_lse: torch.Tensor) -> torch.Tensor:
    """Reproduce split.py's base-2 merge of per-split LSEs -> the final base-2
    LSE. ``mid_lse`` is [heads, num_splits] base-2 (log2(sum)+max) with -inf for
    empty splits. The merge computes merged_m + log2(merged_d) =
    log2(sum_i 2^lse_i), i.e. a base-2 logsumexp over the split axis (all-empty
    rows stay -inf). torch.logsumexp handles the -inf sentinel correctly."""
    return torch.logsumexp(mid_lse.float() * _LN2, dim=-1) / _LN2


def _make_dsv4_scratch_heads(device, *, topk, max_chunks, num_heads):
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=num_heads,
        max_q_rows=1,
        max_width=topk,
        head_dim=_DSV4_HEAD_DIM,
        v_head_dim=_DSV4_HEAD_DIM,
        max_chunks_per_row=max_chunks,
        page_size=_DSV4_PAGE,
    )
    layout = _compressed_sparse_mla_scratch_layout(caps)
    storage = torch.zeros(int(layout.nbytes), dtype=torch.uint8, device=device)
    return _materialize_compressed_sparse_mla_scratch(caps, storage, layout)


def _run_unified_dsv4(device, *, topk, forced_num_splits, seed):
    """Build a dsv4_ref DSV4 decode case (num_heads=128), repack the KV into the
    compressed page layout, run the REAL launcher, and return
    (got_O, got_lse, exp_O, exp_lse, eff_splits)."""
    from b12x.attention._shared.mla.kernel import run_unified_decode

    case = dsv4_ref.make_dsv4_decode_case(
        num_heads=_UNIFIED_NUM_HEADS,
        topk=topk,
        num_tokens=1,
        num_blocks=_UNIFIED_NUM_BLOCKS,
        page_block_size=_DSV4_PAGE,
        invalidate_half=True,
        with_sink=False,
        device=device,
        seed=seed,
    )
    q = case["q"].contiguous()  # [1, 128, 512] bf16
    swa_cache = _repack_dsv4_to_compressed(
        case["kv_cache"], _DSV4_PAGE, _UNIFIED_NUM_BLOCKS
    )
    idx = case["topk_indices"].contiguous()  # [1, topk] int32
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    exp_O = case["expected_O"][0].float()  # [128, 512]
    exp_lse = case["expected_lse"][0].float()  # [128]
    sm_scale = case["sm_scale"]

    n_chunks = (topk + 64 - 1) // 64
    max_chunks = max(8, forced_num_splits)
    scratch = _make_dsv4_scratch_heads(
        device,
        topk=topk,
        max_chunks=max_chunks,
        num_heads=_UNIFIED_NUM_HEADS,
    )

    out = run_unified_decode(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=sm_scale,
        swa_page_size=_DSV4_PAGE,
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()

    eff_splits = min(forced_num_splits, n_chunks)
    got_O = out[0].float()  # [128, 512]
    # Per-split base-2 LSE partials live in the workspace mid_lse after the call;
    # the base-2 merge reconstructs the final LSE (= mid_lse[:, 0] when 1 split).
    mid_lse = scratch.tmp_lse[:1, :_UNIFIED_NUM_HEADS, :eff_splits][
        0
    ]  # [128, eff_splits]
    got_lse = _merge_base2_lse(mid_lse)  # [128]
    return got_O, got_lse, exp_O, exp_lse, eff_splits


@torch.inference_mode()
@pytest.mark.parametrize(
    "topk,forced_num_splits",
    [
        (64, 1),
        (128, 1),
        (512, 1),  # single split (trivial 1-split merge)
        (64, 2),
        (128, 2),
        (512, 4),  # forced multi-split (chunk-aligned ranges)
    ],
)
def test_unified_decode_launcher_matches_dsv4_ref(topk, forced_num_splits) -> None:
    """run_unified_decode (kernel split-K partials + reused base-2 merge) matches
    the dsv4_ref oracle for num_heads=128 at the validated P5 gate, for BOTH
    num_splits=1 and a forced>1 split (the multi-split chunk-aligned partition is
    numerically identical to single-split)."""
    device = require_b12x_sparse_mla()
    got_O, got_lse, exp_O, exp_lse, eff_splits = _run_unified_dsv4(
        device,
        topk=topk,
        forced_num_splits=forced_num_splits,
        seed=topk,
    )

    if forced_num_splits > 1:
        # The forced split must actually partition into the expected number of
        # chunk-aligned ranges (topk=64 -> 1 chunk, so a forced 2 collapses to
        # 1; the numerics below still validate the path either way).
        expected_splits = min(forced_num_splits, (topk + 64 - 1) // 64)
        assert eff_splits == expected_splits

    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"topk={topk} splits={forced_num_splits} O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 2e-2, (
        f"topk={topk} splits={forced_num_splits} O atol exceeded"
    )
    # base-2 LSE: all reference heads here are finite (back half invalidated, not
    # the whole row), so compare directly at atol 5e-2.
    assert torch.isfinite(exp_lse).all()
    assert torch.isfinite(got_lse).all()
    assert (got_lse - exp_lse).abs().max().item() < 5e-2, (
        f"topk={topk} splits={forced_num_splits} lse atol exceeded: "
        f"max|delta|={(got_lse - exp_lse).abs().max().item()}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [(128, 2), (512, 2), (512, 4)])
def test_unified_decode_multi_split_equals_single_split(
    topk, forced_num_splits
) -> None:
    """A forced multi-split decode must equal the single-split decode (same
    chunk-aligned candidate partition, merged base-2): each candidate is owned by
    exactly one split, so the reduction is exact, not just within-tolerance."""
    device = require_b12x_sparse_mla()
    got_single, _, _, _, _ = _run_unified_dsv4(
        device,
        topk=topk,
        forced_num_splits=1,
        seed=topk,
    )
    got_multi, _, _, _, eff_splits = _run_unified_dsv4(
        device,
        topk=topk,
        forced_num_splits=forced_num_splits,
        seed=topk,
    )
    assert eff_splits > 1, "forced multi-split did not partition into >1 split"
    delta = (got_multi - got_single).abs().max().item()
    assert delta < 5e-3, (
        f"topk={topk} splits={forced_num_splits} multi vs single max|delta|={delta}"
    )


# ── DSV4 DUAL-CACHE LAUNCHER NUMERICS (P7c): run_unified_decode (main + extra) ──
# Exercise the REAL launcher with the has_extra (extra-tokens) second KV pool:
# ceil(topk/BI) main chunks (gather from swa_k_cache) then ceil(extra_topk/BI)
# extra chunks (gather from indexed_k_cache with its own page size) folded into
# ONE online softmax + the reused base-2 merge, vs dsv4_extra_decode_reference.
def _run_unified_dsv4_extra(device, *, topk, extra_topk, forced_num_splits, seed):
    from b12x.attention._shared.mla.kernel import run_unified_decode

    pbs_extra = 2
    main_blocks = _UNIFIED_NUM_BLOCKS
    case = dsv4_extra_ref.make_dsv4_extra_decode_case(
        num_heads=_UNIFIED_NUM_HEADS,
        topk=topk,
        extra_topk=extra_topk,
        num_tokens=1,
        num_blocks=main_blocks,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        invalidate_half=True,
        with_sink=False,
        device=device,
        seed=seed,
    )
    q = case["q"].contiguous()
    swa_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, main_blocks)
    extra_blocks = case["extra_kv_cache"].shape[0]
    idx_cache = _repack_dsv4_to_compressed(
        case["extra_kv_cache"], pbs_extra, extra_blocks
    )
    main_idx = case["topk_indices"].contiguous()
    extra_idx = case["extra_indices"].contiguous()
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    extra_lengths = torch.full((1,), extra_topk, dtype=torch.int32, device=device)
    exp_O = case["expected_O"][0].float()

    n_chunks = (topk + 64 - 1) // 64 + (extra_topk + 64 - 1) // 64
    scratch = _make_dsv4_scratch_heads(
        device,
        topk=topk + extra_topk,
        max_chunks=max(8, forced_num_splits, n_chunks),
        num_heads=_UNIFIED_NUM_HEADS,
    )
    out = run_unified_decode(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=main_idx,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=case["sm_scale"],
        swa_page_size=_DSV4_PAGE,
        indexed_k_cache=idx_cache,
        indexed_indices=extra_idx,
        indexed_topk_lengths=extra_lengths,
        indexed_page_size=pbs_extra,
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()
    return out[0].float(), exp_O


@torch.inference_mode()
@pytest.mark.parametrize(
    "topk,extra_topk,forced_num_splits",
    [
        (512, 64, 1),
        (512, 128, 1),  # single split spanning both sections
        (512, 64, 4),
        (512, 128, 6),  # forced multi-split (chunk-aligned, both sections)
    ],
)
def test_unified_decode_dual_cache_matches_extra_ref(
    topk, extra_topk, forced_num_splits
) -> None:
    """run_unified_decode DSV4 dual-cache (main + extra, ONE online softmax over the
    union) matches dsv4_extra_decode_reference for num_heads=128, topk=512 x
    extra_topk in {64,128}, single AND forced>1 split, at the P5 gate."""
    device = require_b12x_sparse_mla()
    got_O, exp_O = _run_unified_dsv4_extra(
        device,
        topk=topk,
        extra_topk=extra_topk,
        forced_num_splits=forced_num_splits,
        seed=topk + extra_topk,
    )
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, (
        f"dual topk={topk} extra={extra_topk} splits={forced_num_splits} O cos={cos}"
    )
    assert (got_O - exp_O).abs().max().item() < 2e-2, (
        f"dual topk={topk} extra={extra_topk} splits={forced_num_splits} O atol exceeded"
    )


@torch.inference_mode()
@pytest.mark.parametrize("topk", [128, 512])
def test_unified_prefill_dual_cache_80_heads_split_tail_matches_extra_ref(
    topk: int,
) -> None:
    """DSV4 dual-cache prefill heads=80 uses the split MG path (64-head paired
    prefix + 16-head tail) and matches the PyTorch extra-cache oracle."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_prefill

    num_heads = 80
    extra_topk, pbs_extra = 128, 2
    main_blocks = 16
    case = dsv4_extra_ref.make_dsv4_extra_decode_case(
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        num_tokens=1,
        num_blocks=main_blocks,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        invalidate_half=False,
        with_sink=False,
        device=device,
        seed=8080,
    )
    q = case["q"].contiguous()
    swa_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, main_blocks)
    extra_blocks = case["extra_kv_cache"].shape[0]
    idx_cache = _repack_dsv4_to_compressed(
        case["extra_kv_cache"], pbs_extra, extra_blocks
    )
    main_idx = case["topk_indices"].contiguous()
    extra_idx = case["extra_indices"].contiguous()
    main_len = torch.full((1,), topk, dtype=torch.int32, device=device)
    extra_len = torch.full((1,), extra_topk, dtype=torch.int32, device=device)

    exp_O, _ = dsv4_extra_ref.dsv4_extra_decode_reference(
        q,
        case["kv_cache"],
        main_idx,
        case["sm_scale"],
        case["extra_kv_cache"],
        extra_idx,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        topk_length=main_len,
        extra_topk_length=extra_len,
        main_kv_dequant=case["kv_dequant"],
        extra_kv_dequant=case["extra_kv_dequant"],
    )

    O, _ = run_unified_prefill(
        q=q,
        kv_cache=swa_cache,
        topk_indices=main_idx,
        sm_scale=case["sm_scale"],
        page_block_size=_DSV4_PAGE,
        topk_length=main_len,
        extra_kv_cache=idx_cache,
        extra_indices=extra_idx,
        extra_topk_length=extra_len,
        extra_page_block_size=pbs_extra,
    )
    torch.cuda.synchronize()
    got = O[0].float()
    exp = exp_O[0].float()
    cos = _cosine(got, exp)
    assert cos > 0.999, f"DSV4 dual-cache prefill heads=80 O cos={cos}"
    assert (got - exp).abs().max().item() < 2e-2


@torch.inference_mode()
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("pbs_extra", [2, 64])
def test_unified_prefill_dual_cache_matches_extra_ref(
    num_heads: int, topk: int, pbs_extra: int
) -> None:
    """DSV4 BF16/FP8 dual-cache prefill uses the extra cache for K-RoPE."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_prefill

    extra_topk = 128
    main_blocks = (topk + _DSV4_PAGE - 1) // _DSV4_PAGE
    case = dsv4_extra_ref.make_dsv4_extra_decode_case(
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        num_tokens=1,
        num_blocks=main_blocks,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        invalidate_half=False,
        with_sink=False,
        device=device,
        seed=9_200 + num_heads,
    )
    q = case["q"].contiguous()
    main_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, main_blocks)
    extra_blocks = case["extra_kv_cache"].shape[0]
    extra_cache = _repack_dsv4_to_compressed(
        case["extra_kv_cache"], pbs_extra, extra_blocks
    )
    main_indices = case["topk_indices"].contiguous()
    extra_indices = case["extra_indices"].contiguous()
    main_lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    extra_lengths = torch.full((1,), extra_topk, dtype=torch.int32, device=device)

    expected, expected_lse = dsv4_extra_ref.dsv4_extra_decode_reference(
        q,
        case["kv_cache"],
        main_indices,
        case["sm_scale"],
        case["extra_kv_cache"],
        extra_indices,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        topk_length=main_lengths,
        extra_topk_length=extra_lengths,
        main_kv_dequant=case["kv_dequant"],
        extra_kv_dequant=case["extra_kv_dequant"],
    )
    output, lse = run_unified_prefill(
        q=q,
        kv_cache=main_cache,
        topk_indices=main_indices,
        topk_length=main_lengths,
        sm_scale=case["sm_scale"],
        page_block_size=_DSV4_PAGE,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        extra_page_block_size=pbs_extra,
    )
    torch.cuda.synchronize()
    got = output[0].float()
    expected = expected[0].float()
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got).item() > 0
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(
        lse.float(),
        expected_lse.float(),
        atol=6.0e-2,
        rtol=2.0e-2,
    )
    cos = _cosine(got, expected)
    assert cos > 0.999, (
        f"DSV4 dual-cache prefill topk={topk} heads={num_heads} O cos={cos}"
    )
    assert (got - expected).abs().max().item() < 2e-2


@torch.inference_mode()
def test_unified_prefill_dsv4_valid_hpb_8_matches_prefill_ref() -> None:
    """DSV4 prefill heads=8 uses a single MG group with VALID_HPB=8 and must not
    read or reduce the zero-padded upper half of the HPB=16 tile."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_prefill

    num_tokens, num_heads, topk = 16, 8, 128
    num_blocks = 8
    case = prefill_ref.make_dsv4_prefill_case(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        num_blocks=num_blocks,
        page_block_size=_DSV4_PAGE,
        with_sink=False,
        invalidate_half=True,
        device=device,
        seed=8128,
    )
    q = case["q"].contiguous()
    swa_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, num_blocks)
    idx = case["topk_indices"].contiguous()
    lengths = case["topk_lengths"].contiguous()

    O, lse = run_unified_prefill(
        q=q,
        kv_cache=swa_cache,
        topk_indices=idx,
        topk_length=lengths,
        sm_scale=case["sm_scale"],
        page_block_size=_DSV4_PAGE,
    )
    torch.cuda.synchronize()

    got = O.float()
    exp = case["expected_O"].float()
    got_lse = lse.float()
    exp_lse = case["expected_lse"].float()
    assert got.shape == (num_tokens, num_heads, _DSV4_HEAD_DIM)
    assert torch.isfinite(got).all()
    assert torch.isfinite(got_lse).all()
    assert (O != 0).any()
    cos = _cosine(got, exp)
    assert cos > 0.999, f"DSV4 prefill heads=8 O cos={cos}"
    assert (got - exp).abs().max().item() < 2e-2
    assert (got_lse - exp_lse).abs().max().item() < 5e-2


@torch.inference_mode()
def test_unified_prefill_glm_tp8_topk2048_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLM 5.x TP8 has eight local attention heads and a 2048-token sparse
    selection.  Exercise that serving shape directly, including the
    VALID_HPB=8 tail and both contiguous and vLLM packed page-strided caches."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_prefill

    num_tokens, num_heads, topk = 2, 8, 2048
    case = glm_ref.make_glm_decode_case(
        num_heads=num_heads,
        topk=topk,
        num_tokens=num_tokens,
        num_blocks=topk // _GLM_PAGE,
        page_block_size=_GLM_PAGE,
        invalidate_half=False,
        seed=52_820_488,
        device=device,
    )

    contiguous_cache = case["kv_cache"].contiguous()
    num_blocks = topk // _GLM_PAGE
    page_bytes = _GLM_PAGE * _GLM_KV_BYTES_PER_TOKEN
    packed_stride = page_bytes + 4096
    packed_storage = torch.empty(
        (num_blocks - 1) * packed_stride + page_bytes,
        dtype=torch.uint8,
        device=device,
    )
    packed_cache = torch.as_strided(
        packed_storage,
        size=(num_blocks, _GLM_PAGE, _GLM_KV_BYTES_PER_TOKEN),
        stride=(packed_stride, _GLM_KV_BYTES_PER_TOKEN, 1),
    )
    packed_cache.copy_(
        contiguous_cache.view(num_blocks, _GLM_PAGE, _GLM_KV_BYTES_PER_TOKEN)
    )
    assert not packed_cache.is_contiguous()

    def run(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = run_unified_prefill(
            q=case["q"].contiguous(),
            kv_cache=kv_cache,
            topk_indices=case["topk_indices"].contiguous(),
            sm_scale=case["sm_scale"],
            page_block_size=_GLM_PAGE,
        )
        torch.cuda.synchronize()
        return result

    # The TP8 packed-row path is an exact reorganization of the accurate
    # two-pass PV math: HIGH occupies rows 0..7 and LOW rows 8..15 of one m16
    # tile. Gate its serving default against the former path before comparing
    # either result to the independent reference.
    monkeypatch.setenv("B12X_MLA_SM120_PREFILL_PACK_HILO_ROWS", "0")
    legacy_output, legacy_lse = run(contiguous_cache)
    monkeypatch.setenv("B12X_MLA_SM120_PREFILL_PACK_HILO_ROWS", "1")
    optimized_output, optimized_lse = run(contiguous_cache)
    assert torch.equal(optimized_output, legacy_output)
    assert torch.equal(optimized_lse, legacy_lse)

    expected = case["expected_O"].float()
    packed_output, packed_lse = run(packed_cache)
    for output, lse in (
        (optimized_output, optimized_lse),
        (packed_output, packed_lse),
    ):
        got = output.float()
        assert torch.isfinite(got).all()
        assert torch.isfinite(lse).all()
        assert (output != 0).any()
        assert _cosine(got, expected) > 0.995
        assert (got - expected).abs().max().item() < 3e-2
        assert (lse.float() - case["expected_lse"].float()).abs().max().item() < 5e-2


@torch.inference_mode()
def test_unified_decode_dual_cache_extra_zero_equals_main_only() -> None:
    """extra_topk=0 (no extra cache) must reduce to the single-cache decode: the
    dual reference's concat is a no-op, so the unified main-only path matches."""
    device = require_b12x_sparse_mla()
    # main-only via the single-cache launcher path (extra args omitted) vs the
    # dual reference with extra_topk=0 (== dsv4_decode_reference).
    got_O, _, exp_O, _, _ = _run_unified_dsv4(
        device,
        topk=512,
        forced_num_splits=1,
        seed=512,
    )
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"extra=0 main-only cos={cos}"


@torch.inference_mode()
@pytest.mark.parametrize(
    "extra_topk,forced_num_splits",
    [
        (64, 1),  # single extra chunk, single split (zero-width main only)
        (128, 1),  # two extra chunks, single split spanning both
        (128, 2),  # forced multi-split over the extra-only chunk range
    ],
)
def test_unified_decode_dual_cache_zero_width_main_extra_only(
    extra_topk, forced_num_splits
) -> None:
    """ZERO-WIDTH MAIN (DSV4 dual-cache, swa_indices shape (1,0)): all KV is in
    the EXTRA cache. num_main_chunks==0 -> the launcher must NOT build a 0-extent
    main topk layout; the kernel attends ONLY the extra section and matches
    dsv4_extra_decode_reference (the main concat is empty).

    This is the P10h zero-width-main fix: cute.make_layout(0) is illegal, so the
    main topk_row is elided to a degenerate 1-extent view that is never read."""
    device = require_b12x_sparse_mla()
    got_O, exp_O = _run_unified_dsv4_extra(
        device,
        topk=0,
        extra_topk=extra_topk,
        forced_num_splits=forced_num_splits,
        seed=extra_topk + 13,
    )
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, (
        f"zero-width-main extra={extra_topk} splits={forced_num_splits} O cos={cos}"
    )
    assert (got_O - exp_O).abs().max().item() < 2e-2, (
        f"zero-width-main extra={extra_topk} splits={forced_num_splits} O atol exceeded"
    )


# ── GLM_NSA LAUNCHER NUMERICS (P7b): run_unified_decode vs glm_ref ─────────────
# GLM is rs-1's own model (no FlashInfer PTX), so the bar is NUMERICAL vs
# glm_ref.glm_decode_reference (sparse_mla_reference). The GLM K dequant->requant
# e4m3 + unit-sfb path is lossier than DSV4, so the tolerance is the looser GLM
# band (cos > 0.995, O atol 3e-2), matching glm_ref's own brute-force self-test
# tolerance. The unified GLM kernel reuses the SAME split.py base-2 merge as DSV4.
_GLM_NUM_HEADS = 128  # full GLM head count (8 head blocks of HPB=16)
_GLM_PAGE = 64  # GLM decode page_block_size (stride = idx*656; pbs-invariant)


def _make_glm_sparse_scratch(device, *, topk, max_chunks, num_heads, s_kv):
    from b12x.attention.sparse_mla._scratch import (
        B12XSparseMLAScratchCaps,
        plan_sparse_mla_scratch,
    )

    caps = B12XSparseMLAScratchCaps(
        softmax_scale=1.0,
        device=device,
        num_q_heads=num_heads,
        max_q_rows=1,
        max_batch=1,
        max_width=max(topk, 1),
        max_kv_rows=s_kv,
        head_dim=glm_ref.GLM_Q_HEAD_DIM,
        v_head_dim=glm_ref.GLM_D_V,
        max_chunks_per_row=max_chunks,
        page_size=_GLM_PAGE,
    )
    plan = plan_sparse_mla_scratch(caps)
    (spec,) = plan.scratch_specs()
    storage = torch.zeros(spec.shape, dtype=spec.dtype, device=device)
    return plan, storage


def _run_unified_glm(
    device,
    *,
    topk,
    forced_num_splits,
    seed,
    num_heads=_GLM_NUM_HEADS,
    use_length_tensor=True,
):
    """Build a glm_ref GLM decode case and run the real unified launcher."""
    from b12x.attention._shared.mla.kernel import run_unified_decode

    nblk = max(1, (topk + _GLM_PAGE - 1) // _GLM_PAGE)
    case = glm_ref.make_glm_decode_case(
        num_heads=num_heads,
        topk=topk,
        num_blocks=nblk,
        page_block_size=_GLM_PAGE,
        invalidate_half=True,
        seed=seed,
        device=device,
    )
    q = case["q"].contiguous()  # [1, 128, 576] bf16
    kv_cache = case["kv_cache"].contiguous()  # (nblk*64, 1, 656) uint8 GLM
    idx = case["topk_indices"].contiguous()  # [1, topk] int32
    exp_O = case["expected_O"][0].float()  # [128, 512]
    sm_scale = case["sm_scale"]
    s_kv = kv_cache.shape[0]

    n_chunks = (topk + 64 - 1) // 64
    max_chunks = max(8, forced_num_splits)
    plan, storage = _make_glm_sparse_scratch(
        device,
        topk=topk,
        max_chunks=max_chunks,
        num_heads=num_heads,
        s_kv=s_kv,
    )
    cache_seqlens = torch.full((1,), s_kv, dtype=torch.int32, device=device)
    nsa_seqlens = torch.full((1,), topk, dtype=torch.int32, device=device)
    binding = plan.bind(
        scratch=storage,
        q=q,
        selected_indices=idx,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_seqlens,
    )

    out = run_unified_decode(
        q_all=q,
        swa_k_cache=kv_cache,
        swa_indices=idx,
        swa_topk_lengths=(nsa_seqlens if use_length_tensor else None),
        workspace=binding.scratch,
        sm_scale=sm_scale,
        swa_page_size=_GLM_PAGE,
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()
    return out[0].float(), exp_O, min(forced_num_splits, n_chunks)


@torch.inference_mode()
@pytest.mark.parametrize("forced_num_splits", [1, 4])
def test_unified_decode_glm_tp8_native_swap_ab_matches_reference(
    forced_num_splits,
) -> None:
    """GLM TP8 uses the native four-warp swapped-QK decode specialization."""
    device = require_b12x_sparse_mla()
    import b12x.attention._shared.mla.kernel as launch

    got_O, exp_O, _ = _run_unified_glm(
        device,
        topk=256,
        forced_num_splits=forced_num_splits,
        seed=52_008 + forced_num_splits,
        num_heads=8,
    )
    plan = launch.LAST_DECODE_PLAN
    assert plan.get("native_glm_h8") is True
    assert plan.get("qk_swap_ab") is True
    assert plan.get("math_warps") == 4
    assert plan.get("block_threads") == 160
    assert plan.get("io_warps") == 1
    assert plan.get("kv_stage_packed") is True
    assert plan.get("kv_smem_stride") == 656
    assert plan.get("qk_candidates_per_warp") == 16
    assert torch.isfinite(got_O).all()
    assert (got_O != 0).any()
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"GLM TP8 splits={forced_num_splits} O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 3e-2


@torch.inference_mode()
def test_unified_decode_glm_tp8_native_matches_padded_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native TP8 path preserves the former padded decode numerics."""
    device = require_b12x_sparse_mla()
    kwargs = dict(
        topk=256,
        forced_num_splits=4,
        seed=52_016,
        num_heads=8,
    )
    monkeypatch.setenv("B12X_MLA_SM120_GLM_H8_NATIVE", "0")
    padded, _, _ = _run_unified_glm(device, **kwargs)
    monkeypatch.setenv("B12X_MLA_SM120_GLM_H8_NATIVE", "1")
    native, _, _ = _run_unified_glm(device, **kwargs)
    assert _cosine(native, padded) > 0.999999
    assert (native - padded).abs().max().item() < 1e-3


@torch.inference_mode()
def test_unified_decode_glm_tp8_native_scalar_length_matches_reference() -> None:
    """The uniform scalar-length entry uses the same native TP8 math path."""
    device = require_b12x_sparse_mla()
    import b12x.attention._shared.mla.kernel as launch

    got_O, exp_O, _ = _run_unified_glm(
        device,
        topk=128,
        forced_num_splits=1,
        seed=52_128,
        num_heads=8,
        use_length_tensor=False,
    )
    assert launch.LAST_DECODE_PLAN.get("native_glm_h8") is True
    assert launch.LAST_DECODE_PLAN.get("per_token_len") is False
    assert _cosine(got_O, exp_O) > 0.999
    assert (got_O - exp_O).abs().max().item() < 3e-2


@torch.inference_mode()
@pytest.mark.parametrize(
    "topk,forced_num_splits",
    [
        (64, 1),
        (128, 1),
        (512, 1),  # single split (trivial 1-split merge)
        (128, 2),
        (512, 4),  # forced multi-split (chunk-aligned ranges)
    ],
)
def test_unified_decode_launcher_matches_glm_ref(topk, forced_num_splits) -> None:
    """run_unified_decode GLM_NSA branch (ARBITRARY_FP32 inline scales,
    V_HAS_ROPE=false, 512/128/4) matches the glm_ref oracle for num_heads=128 at
    the looser GLM gate (cos > 0.995, O atol 3e-2), for single AND forced>1 split."""
    device = require_b12x_sparse_mla()
    got_O, exp_O, eff_splits = _run_unified_glm(
        device,
        topk=topk,
        forced_num_splits=forced_num_splits,
        seed=topk,
    )
    cos = _cosine(got_O, exp_O)
    assert cos > 0.995, f"GLM topk={topk} splits={forced_num_splits} O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 3e-2, (
        f"GLM topk={topk} splits={forced_num_splits} O atol exceeded: "
        f"max|delta|={(got_O - exp_O).abs().max().item()}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [(128, 2), (512, 4)])
def test_unified_decode_glm_multi_split_equals_single_split(
    topk, forced_num_splits
) -> None:
    """A forced GLM multi-split decode must match the single-split decode (same
    chunk-aligned candidate partition, merged base-2)."""
    device = require_b12x_sparse_mla()
    got_single, _, _ = _run_unified_glm(
        device, topk=topk, forced_num_splits=1, seed=topk
    )
    got_multi, _, eff_splits = _run_unified_glm(
        device,
        topk=topk,
        forced_num_splits=forced_num_splits,
        seed=topk,
    )
    assert eff_splits > 1, "forced GLM multi-split did not partition into >1 split"
    delta = (got_multi - got_single).abs().max().item()
    assert delta < 5e-3, (
        f"GLM topk={topk} splits={forced_num_splits} multi vs single max|delta|={delta}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_unified_decode_glm_multitoken_per_token_length(num_tokens) -> None:
    """GLM_NSA decode with num_tokens in {1,4,16}, num_heads=128, and MIXED
    per-token active_token_counts (the GLM topk_length) matches glm_decode_reference
    per token at the GLM gate (cos > 0.995)."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_decode
    from b12x.attention.sparse_mla._scratch import (
        B12XSparseMLAScratchCaps,
        plan_sparse_mla_scratch,
    )

    topk = 512
    nblk = max(1, (topk + _GLM_PAGE - 1) // _GLM_PAGE)
    case = glm_ref.make_glm_decode_case(
        num_heads=_GLM_NUM_HEADS,
        topk=topk,
        num_tokens=num_tokens,
        num_blocks=nblk,
        page_block_size=_GLM_PAGE,
        invalidate_half=False,
        seed=3000 + num_tokens,
        device=device,
    )
    q = case["q"].contiguous()  # [T, 128, 576]
    kv_cache = case["kv_cache"].contiguous()
    idx = case["topk_indices"].contiguous()  # [T, topk] (all valid)
    sm_scale = case["sm_scale"]
    s_kv = kv_cache.shape[0]
    lengths = _mixed_lengths(num_tokens, topk, device)

    exp_O = glm_ref.glm_decode_reference(
        q,
        kv_cache,
        idx,
        sm_scale,
        active_token_counts=lengths,
    ).float()

    n_chunks = (topk + 64 - 1) // 64
    caps = B12XSparseMLAScratchCaps(
        softmax_scale=1.0,
        device=device,
        num_q_heads=_GLM_NUM_HEADS,
        max_q_rows=num_tokens,
        max_batch=num_tokens,
        max_width=topk,
        max_kv_rows=s_kv,
        head_dim=glm_ref.GLM_Q_HEAD_DIM,
        v_head_dim=glm_ref.GLM_D_V,
        max_chunks_per_row=max(8, n_chunks),
        page_size=_GLM_PAGE,
    )
    plan = plan_sparse_mla_scratch(caps)
    (spec,) = plan.scratch_specs()
    storage = torch.zeros(spec.shape, dtype=spec.dtype, device=device)
    cache_seqlens = torch.full((num_tokens,), s_kv, dtype=torch.int32, device=device)
    binding = plan.bind(
        scratch=storage,
        q=q,
        selected_indices=idx,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=lengths,
    )
    out = run_unified_decode(
        q_all=q,
        swa_k_cache=kv_cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        workspace=binding.scratch,
        sm_scale=sm_scale,
        swa_page_size=_GLM_PAGE,
        forced_num_splits=2,
    )
    torch.cuda.synchronize()
    got = out.float()
    for t in range(num_tokens):
        cos = _cosine(got[t], exp_O[t])
        assert cos > 0.995, (
            f"GLM T={num_tokens} token {t} (len={int(lengths[t])}) O cos={cos}"
        )
        assert (got[t] - exp_O[t]).abs().max().item() < 3e-2


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize(
    "high_page_ids", [False, True], ids=["low-pages", "high-pages"]
)
def test_unified_decode_glm_serial_chunks_rescale_late_maximum(
    num_tokens, high_page_ids
) -> None:
    """Rescale accumulated values when every row-0 head peaks in the last chunk.

    Compare a serial ten-chunk walk and one chunk per split with the reference,
    using both ordinary page IDs and page offsets beyond signed 32-bit range.
    """
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_decode
    from b12x.attention.sparse_mla._scratch import (
        B12XSparseMLAScratchCaps,
        plan_sparse_mla_scratch,
    )

    topk = 640
    n_chunks = topk // 64
    boost = 60.0
    nblk = max(1, (topk + _GLM_PAGE - 1) // _GLM_PAGE)
    case = glm_ref.make_glm_decode_case(
        num_heads=_GLM_NUM_HEADS,
        topk=topk,
        num_tokens=num_tokens,
        num_blocks=nblk,
        page_block_size=_GLM_PAGE,
        invalidate_half=False,
        seed=7900 + num_tokens,
        device=device,
    )
    q = case["q"].contiguous()
    kv_cache = case["kv_cache"].contiguous()
    # Distinct slots keep boosted keys out of the first nine chunks.
    idx = torch.arange(topk, dtype=torch.int32, device=device)
    idx = idx.expand(num_tokens, -1).contiguous()
    sm_scale = case["sm_scale"]
    s_kv = kv_cache.shape[0]
    # Heads h and h + 64 of row 0 share key h in the last chunk; the key's rope
    # part points along the mean of their (unit) query rope parts.
    rope = q[0, :, 512:].float()
    rope = rope / rope.norm(dim=-1, keepdim=True)
    pair_mean = rope.view(-1, 64, rope.shape[-1]).sum(dim=0)
    pair_mean = boost * pair_mean / pair_mean.norm(dim=-1, keepdim=True)
    for key in range(64):
        slot = int(idx[0, topk - 64 + key])
        kv_cache[slot, 0, 528:656] = pair_mean[key].to(torch.bfloat16).view(torch.uint8)
    kv = glm_ref.unpack_mla_kv_cache_reference(kv_cache).squeeze(1).float()
    scores = q[0].float() @ kv[idx[0].long()].T * sm_scale
    assert (scores.argmax(dim=-1) >= topk - 64).all(), "maximum must arrive late"
    lengths = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    exp_O = glm_ref.glm_decode_reference(
        q, kv_cache, idx, sm_scale, active_token_counts=lengths
    ).float()
    assert torch.isfinite(exp_O).all() and exp_O.norm() > 0

    if high_page_ids:
        page_stride_bytes = _GLM_PAGE * _GLM_KV_BYTES_PER_TOKEN
        int32_max = torch.iinfo(torch.int32).max
        high_page = int32_max // page_stride_bytes + 2
        slot_bias = high_page * _GLM_PAGE
        required_bytes = (slot_bias + s_kv) * _GLM_KV_BYTES_PER_TOKEN
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < required_bytes + 2 * 1024**3:
            pytest.skip(
                "GLM high-page rescale test requires "
                f"{required_bytes + 2 * 1024**3} bytes free, found {free_bytes}"
            )
        pool = torch.empty(
            (slot_bias + s_kv, 1, _GLM_KV_BYTES_PER_TOKEN),
            dtype=torch.uint8,
            device=device,
        )
        # Relocate the oracle's initialized keys without reading the unused pool.
        pool[slot_bias:].copy_(kv_cache)
        kv_cache = pool
        idx = idx + slot_bias
        s_kv = pool.shape[0]
        assert high_page * page_stride_bytes > int32_max
        assert s_kv < int32_max

    def run(forced_num_splits: int) -> torch.Tensor:
        caps = B12XSparseMLAScratchCaps(
            softmax_scale=1.0,
            device=device,
            num_q_heads=_GLM_NUM_HEADS,
            max_q_rows=num_tokens,
            max_batch=num_tokens,
            max_width=topk,
            max_kv_rows=s_kv,
            head_dim=glm_ref.GLM_Q_HEAD_DIM,
            v_head_dim=glm_ref.GLM_D_V,
            max_chunks_per_row=max(8, n_chunks),
            page_size=_GLM_PAGE,
        )
        plan = plan_sparse_mla_scratch(caps)
        (spec,) = plan.scratch_specs()
        storage = torch.zeros(spec.shape, dtype=spec.dtype, device=device)
        cache_seqlens = torch.full(
            (num_tokens,), s_kv, dtype=torch.int32, device=device
        )
        binding = plan.bind(
            scratch=storage,
            q=q,
            selected_indices=idx,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=lengths,
        )
        out = run_unified_decode(
            q_all=q,
            swa_k_cache=kv_cache,
            swa_indices=idx,
            swa_topk_lengths=lengths,
            workspace=binding.scratch,
            sm_scale=sm_scale,
            swa_page_size=_GLM_PAGE,
            forced_num_splits=forced_num_splits,
        )
        torch.cuda.synchronize()
        return out.float().clone()

    serial = run(1)
    per_chunk = run(n_chunks)
    for label, got in (("serial", serial), ("per_chunk", per_chunk)):
        rel = ((got - exp_O).norm() / exp_O.norm()).item()
        assert rel < 2e-2, f"{label} rel-L2 vs reference {rel}"
    rel = ((serial - per_chunk).norm() / exp_O.norm()).item()
    assert rel < 1e-2, f"serial vs per-chunk rel-L2 {rel}"


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [(128, 1), (512, 4)])
def test_unified_decode_glm_return_lse_matches_reference(
    topk, forced_num_splits
) -> None:
    """GLM_NSA decode return_lse: the FINAL base-2 LSE reconstructed from mid_lse
    matches the glm_ref oracle base-2 LSE (the GLM branch shares the merge + LSE
    reconstruction with DSV4)."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_decode

    nblk = max(1, (topk + _GLM_PAGE - 1) // _GLM_PAGE)
    case = glm_ref.make_glm_decode_case(
        num_heads=_GLM_NUM_HEADS,
        topk=topk,
        num_blocks=nblk,
        page_block_size=_GLM_PAGE,
        invalidate_half=True,
        seed=topk + 5,
        device=device,
    )
    q = case["q"].contiguous()
    kv_cache = case["kv_cache"].contiguous()
    idx = case["topk_indices"].contiguous()
    exp_O = case["expected_O"][0].float()
    exp_lse_log2 = case["expected_lse"][0].float()
    sm_scale = case["sm_scale"]
    s_kv = kv_cache.shape[0]

    max_chunks = max(8, forced_num_splits)
    plan, storage = _make_glm_sparse_scratch(
        device,
        topk=topk,
        max_chunks=max_chunks,
        num_heads=_GLM_NUM_HEADS,
        s_kv=s_kv,
    )
    cache_seqlens = torch.full((1,), s_kv, dtype=torch.int32, device=device)
    nsa_seqlens = torch.full((1,), topk, dtype=torch.int32, device=device)
    binding = plan.bind(
        scratch=storage,
        q=q,
        selected_indices=idx,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_seqlens,
    )
    out_o, out_lse = run_unified_decode(
        q_all=q,
        swa_k_cache=kv_cache,
        swa_indices=idx,
        swa_topk_lengths=nsa_seqlens,
        workspace=binding.scratch,
        sm_scale=sm_scale,
        swa_page_size=_GLM_PAGE,
        return_lse=True,
        lse_scale="base2",
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()
    got_O = out_o[0].float()
    got_lse = out_lse[0].float()
    assert _cosine(got_O, exp_O) > 0.995
    assert torch.isfinite(got_lse).all()
    assert (got_lse - exp_lse_log2).abs().max().item() < 5e-2, (
        f"GLM return_lse topk={topk} splits={forced_num_splits} base-2 LSE atol exceeded: "
        f"max|delta|={(got_lse - exp_lse_log2).abs().max().item()}"
    )


# ── P10a DECODE PARITY: attn_sink fold, return_lse, VALID_HPB<16 ───────────────
# These exercise the four P10a decode features (attn_sink in the merge, returned
# final LSE, non-multiple-of-16 head shards) against the SAME dsv4_ref oracle that
# folds sink in its LSE/O and computes per-head independently.


def _run_unified_dsv4_feature(
    device,
    *,
    topk,
    num_heads,
    with_sink,
    return_lse,
    lse_scale,
    forced_num_splits,
    seed,
):
    """Run the REAL unified DSV4 decode launcher for an arbitrary head count, with
    optional attn_sink fold + return_lse, and return
    (got_O, got_lse_or_None, exp_O, exp_lse_log2, attn_sink)."""
    from b12x.attention._shared.mla.kernel import run_unified_decode

    case = dsv4_ref.make_dsv4_decode_case(
        num_heads=num_heads,
        topk=topk,
        num_tokens=1,
        num_blocks=_UNIFIED_NUM_BLOCKS,
        page_block_size=_DSV4_PAGE,
        invalidate_half=True,
        with_sink=with_sink,
        device=device,
        seed=seed,
    )
    q = case["q"].contiguous()  # [1, H, 512] bf16
    swa_cache = _repack_dsv4_to_compressed(
        case["kv_cache"], _DSV4_PAGE, _UNIFIED_NUM_BLOCKS
    )
    idx = case["topk_indices"].contiguous()  # [1, topk] int32
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    exp_O = case["expected_O"][0].float()  # [H, 512]
    exp_lse_log2 = case["expected_lse"][
        0
    ].float()  # [H] base-2 (sink-folded if with_sink)
    sm_scale = case["sm_scale"]
    attn_sink = case["attn_sink"]  # [H] f32 or None

    max_chunks = max(8, forced_num_splits)
    scratch = _make_dsv4_scratch_heads(
        device,
        topk=topk,
        max_chunks=max_chunks,
        num_heads=num_heads,
    )

    out = run_unified_decode(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=sm_scale,
        swa_page_size=_DSV4_PAGE,
        attn_sink=attn_sink if with_sink else None,
        return_lse=return_lse,
        lse_scale=lse_scale,
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()
    if return_lse:
        out_o, out_lse = out
        return out_o[0].float(), out_lse[0].float(), exp_O, exp_lse_log2, attn_sink
    return out[0].float(), None, exp_O, exp_lse_log2, attn_sink


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [(128, 1), (512, 1), (512, 4)])
def test_unified_decode_attn_sink_matches_reference(topk, forced_num_splits) -> None:
    """attn_sink fold (wired into the split.py sink-merge, upstream's sink-in-merge
    design) matches the dsv4_ref oracle (which folds sink as output *= sigmoid(lse_e
    - sink))."""
    device = require_b12x_sparse_mla()
    got_O, _, exp_O, _, sink = _run_unified_dsv4_feature(
        device,
        topk=topk,
        num_heads=_UNIFIED_NUM_HEADS,
        with_sink=True,
        return_lse=False,
        lse_scale="base2",
        forced_num_splits=forced_num_splits,
        seed=topk + 1,
    )
    assert sink is not None
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"sink topk={topk} splits={forced_num_splits} O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 2e-2, (
        f"sink topk={topk} splits={forced_num_splits} O atol exceeded: "
        f"max|delta|={(got_O - exp_O).abs().max().item()}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("topk,forced_num_splits", [(128, 1), (512, 1), (512, 4)])
def test_unified_decode_return_lse_matches_reference(topk, forced_num_splits) -> None:
    """return_lse returns the FINAL base-2 LSE reconstructed from the per-split
    mid_lse; it matches the dsv4_ref base-2 LSE (no sink)."""
    device = require_b12x_sparse_mla()
    got_O, got_lse, exp_O, exp_lse_log2, _ = _run_unified_dsv4_feature(
        device,
        topk=topk,
        num_heads=_UNIFIED_NUM_HEADS,
        with_sink=False,
        return_lse=True,
        lse_scale="base2",
        forced_num_splits=forced_num_splits,
        seed=topk + 2,
    )
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"lse topk={topk} splits={forced_num_splits} O cos={cos}"
    assert torch.isfinite(exp_lse_log2).all()
    assert torch.isfinite(got_lse).all()
    assert (got_lse - exp_lse_log2).abs().max().item() < 5e-2, (
        f"return_lse topk={topk} splits={forced_num_splits} base-2 LSE atol exceeded: "
        f"max|delta|={(got_lse - exp_lse_log2).abs().max().item()}"
    )


@torch.inference_mode()
def test_unified_decode_return_lse_natural_scale() -> None:
    """return_lse with lse_scale='natural' returns the natural-log LSE (= base-2 *
    ln2)."""
    device = require_b12x_sparse_mla()
    _, got_lse_nat, _, exp_lse_log2, _ = _run_unified_dsv4_feature(
        device,
        topk=256,
        num_heads=_UNIFIED_NUM_HEADS,
        with_sink=False,
        return_lse=True,
        lse_scale="natural",
        forced_num_splits=1,
        seed=99,
    )
    exp_lse_nat = exp_lse_log2 * _LN2
    assert (got_lse_nat - exp_lse_nat).abs().max().item() < 5e-2, (
        "natural-scale LSE atol exceeded: "
        f"max|delta|={(got_lse_nat - exp_lse_nat).abs().max().item()}"
    )


@torch.inference_mode()
@pytest.mark.parametrize("num_heads", [8, 24, 48])
@pytest.mark.parametrize("forced_num_splits", [1, 4])
def test_unified_decode_valid_hpb_small_and_nonmult16(
    num_heads, forced_num_splits
) -> None:
    """VALID_HPB<16 / non-multiple-of-16 head shards: num_heads in {8 (<16), 24, 48
    (not multiples of 16)} match the dsv4_ref oracle (which attends per-head over
    only the valid heads). The kernel zero-pads the HPB=16 tile and gates writes to
    the valid head rows via the (up to two) per-head-block grids."""
    device = require_b12x_sparse_mla()
    topk = 256
    got_O, _, exp_O, _, _ = _run_unified_dsv4_feature(
        device,
        topk=topk,
        num_heads=num_heads,
        with_sink=False,
        return_lse=False,
        lse_scale="base2",
        forced_num_splits=forced_num_splits,
        seed=num_heads * 7 + 3,
    )
    assert got_O.shape == (num_heads, _DSV4_HEAD_DIM)
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"heads={num_heads} splits={forced_num_splits} O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 2e-2, (
        f"heads={num_heads} splits={forced_num_splits} O atol exceeded: "
        f"max|delta|={(got_O - exp_O).abs().max().item()}"
    )


@torch.inference_mode()
def test_unified_decode_valid_hpb_with_lse_and_sink() -> None:
    """A non-multiple-of-16 head shard (24) WITH both attn_sink + return_lse: O and
    the sink-folded LSE both match the reference on exactly the valid heads."""
    device = require_b12x_sparse_mla()
    got_O, got_lse, exp_O, exp_lse_log2, sink = _run_unified_dsv4_feature(
        device,
        topk=256,
        num_heads=24,
        with_sink=True,
        return_lse=True,
        lse_scale="base2",
        forced_num_splits=2,
        seed=4242,
    )
    assert sink is not None
    assert got_O.shape == (24, _DSV4_HEAD_DIM)
    assert got_lse.shape == (24,)
    cos = _cosine(got_O, exp_O)
    assert cos > 0.999, f"heads=24 sink+lse O cos={cos}"
    assert (got_O - exp_O).abs().max().item() < 2e-2
    assert (got_lse - exp_lse_log2).abs().max().item() < 5e-2, (
        "heads=24 sink+lse base-2 LSE atol exceeded: "
        f"max|delta|={(got_lse - exp_lse_log2).abs().max().item()}"
    )


# ── P10b MULTI-TOKEN + PER-TOKEN topk_length DECODE ────────────────────────────
# Exercise run_unified_decode with num_tokens in {1,4,16}, num_heads=128, and
# MIXED per-token topk_length (deliberately non-multiples of 64, plus a near-zero
# and a full-length token), vs the multi-token dsv4_decode_reference (which masks
# each token's candidates past topk_length[t]). The per-token kernel reads
# section_len = clamp(topk_length[t], 0, topk) at t=blockIdx.x; over-allocated
# chunks for short tokens become fully masked (-> mid_lse=-inf -> merge ignores).
#
# SETTLE the open question two ways:
#   (a) the caller passes REAL per-token lengths and leaves the indices VALID past
#       the length (only the kernel's per-token section_len masks them). This is
#       the case the OLD uniform-length kernel got LATENTLY WRONG (it used the full
#       topk section, so it attended to candidates past topk_length[t]).
#   (b) the caller -1-pads the indices past each token's length AND passes the
#       lengths. Here BOTH the section bound and the S3 idx<0 mask agree, so the
#       OLD uniform kernel was already correct -- the new path must match too.
_MT_HEADS = 128


def _mixed_lengths(num_tokens: int, topk: int, device: torch.device) -> torch.Tensor:
    """Per-token MIXED lengths: a near-zero token, a full-length token, and the
    rest deliberately off-64-boundary and spread across [1, topk]."""
    base = [37, 200, 5, topk, 64 + 13, topk - 7, 128 + 1, 1]
    vals = [base[i % len(base)] for i in range(num_tokens)]
    # token 0 is always full topk (so a uniform check on token 0 alone can't hide
    # the mixing), token 1 near-zero to stress the all-masked-chunk path.
    if num_tokens >= 1:
        vals[0] = topk
    if num_tokens >= 2:
        vals[1] = 3
    vals = [max(1, min(int(v), topk)) for v in vals]
    return torch.tensor(vals, dtype=torch.int32, device=device)


def _run_unified_dsv4_multitoken(
    device,
    *,
    num_tokens,
    topk,
    lengths,
    neg_pad_past_len,
    forced_num_splits,
    seed,
):
    """Build a multi-token DSV4 case, optionally -1-pad indices past each token's
    length, run the REAL launcher with per-token swa_topk_lengths, and compare per
    token against dsv4_decode_reference(topk_length=lengths)."""
    from b12x.attention._shared.mla.kernel import run_unified_decode

    case = dsv4_ref.make_dsv4_decode_case(
        num_heads=_MT_HEADS,
        topk=topk,
        num_tokens=num_tokens,
        num_blocks=_UNIFIED_NUM_BLOCKS,
        page_block_size=_DSV4_PAGE,
        invalidate_half=False,
        with_sink=False,
        device=device,
        seed=seed,
    )
    q = case["q"].contiguous()  # [T, 128, 512]
    swa_cache = _repack_dsv4_to_compressed(
        case["kv_cache"], _DSV4_PAGE, _UNIFIED_NUM_BLOCKS
    )
    idx = case["topk_indices"].contiguous()  # [T, topk] (all valid)

    if neg_pad_past_len:
        # Case (b): force indices at position >= length[t] to the -1 sentinel.
        ar = torch.arange(topk, device=device).unsqueeze(0)
        idx = idx.clone()
        idx[ar >= lengths.unsqueeze(-1)] = -1

    # Reference always masks per-token by length (and by idx<0 for case (b)).
    exp_O, _ = dsv4_ref.dsv4_decode_reference(
        q,
        case["kv_cache"],
        idx,
        case["sm_scale"],
        page_block_size=_DSV4_PAGE,
        topk_length=lengths,
        kv_dequant=case["kv_dequant"],
    )

    n_chunks = (topk + 64 - 1) // 64
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=_MT_HEADS,
        max_q_rows=num_tokens,
        max_width=topk,
        head_dim=_DSV4_HEAD_DIM,
        v_head_dim=_DSV4_HEAD_DIM,
        max_chunks_per_row=max(8, forced_num_splits, n_chunks),
        page_size=_DSV4_PAGE,
    )
    layout = _compressed_sparse_mla_scratch_layout(caps)
    storage = torch.zeros(int(layout.nbytes), dtype=torch.uint8, device=device)
    scratch = _materialize_compressed_sparse_mla_scratch(caps, storage, layout)

    out = run_unified_decode(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=idx,
        swa_topk_lengths=lengths,
        workspace=scratch,
        sm_scale=case["sm_scale"],
        swa_page_size=_DSV4_PAGE,
        forced_num_splits=forced_num_splits,
    )
    torch.cuda.synchronize()
    return out.float(), exp_O.float()


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@pytest.mark.parametrize("neg_pad_past_len", [False, True])
def test_unified_decode_multitoken_per_token_length(
    num_tokens, neg_pad_past_len
) -> None:
    """run_unified_decode honours MIXED per-token topk_length for num_tokens in
    {1,4,16}, num_heads=128, matching the multi-token reference per token --
    BOTH when the caller passes real lengths with valid indices (a; the OLD uniform
    kernel was latently wrong) AND when the caller -1-pads indices past the length
    (b; the OLD uniform kernel was already correct via idx<0)."""
    device = require_b12x_sparse_mla()
    topk = 512
    lengths = _mixed_lengths(num_tokens, topk, device)
    got_O, exp_O = _run_unified_dsv4_multitoken(
        device,
        num_tokens=num_tokens,
        topk=topk,
        lengths=lengths,
        neg_pad_past_len=neg_pad_past_len,
        forced_num_splits=2,
        seed=1000 + num_tokens,
    )
    for t in range(num_tokens):
        cos = _cosine(got_O[t], exp_O[t])
        assert cos > 0.999, (
            f"T={num_tokens} pad={neg_pad_past_len} token {t} "
            f"(len={int(lengths[t])}) O cos={cos}"
        )
        assert (got_O[t] - exp_O[t]).abs().max().item() < 2e-2, (
            f"T={num_tokens} pad={neg_pad_past_len} token {t} O atol exceeded"
        )


@torch.inference_mode()
def test_unified_decode_multitoken_old_uniform_was_wrong_for_valid_indices() -> None:
    """Settle the open question: with REAL per-token lengths but VALID indices past
    the length (no -1 padding), the per-token kernel must DIFFER from a uniform
    full-topk decode (proving the per-token masking actually changes the result --
    i.e. the old uniform-length kernel was latently wrong for this contract)."""
    device = require_b12x_sparse_mla()
    topk, num_tokens = 512, 4
    # Real mixed lengths < topk for the short tokens.
    lengths = _mixed_lengths(num_tokens, topk, device)
    # Per-token correct result (valid indices, masked only by length).
    got_pertok, exp_pertok = _run_unified_dsv4_multitoken(
        device,
        num_tokens=num_tokens,
        topk=topk,
        lengths=lengths,
        neg_pad_past_len=False,
        forced_num_splits=1,
        seed=2024,
    )
    # Uniform full-topk decode of the SAME inputs (lengths == topk -> scalar path).
    full = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    got_uniform, _ = _run_unified_dsv4_multitoken(
        device,
        num_tokens=num_tokens,
        topk=topk,
        lengths=full,
        neg_pad_past_len=False,
        forced_num_splits=1,
        seed=2024,
    )
    # The per-token path matches the length-masked reference.
    for t in range(num_tokens):
        assert _cosine(got_pertok[t], exp_pertok[t]) > 0.999
    # For a SHORT token (len < topk), the uniform decode must differ from the
    # length-masked per-token decode -- the latent bug the per-token threading fixes.
    short = [t for t in range(num_tokens) if int(lengths[t]) < topk]
    assert short, "test setup must include a short token"
    t = short[0]
    delta = (got_uniform[t] - got_pertok[t]).abs().max().item()
    assert delta > 1e-3, (
        f"uniform vs per-token token {t} (len={int(lengths[t])}) delta={delta} "
        "-- per-token length masking had no effect (unexpected)"
    )


@torch.inference_mode()
def test_unified_decode_uniform_length_batch_matches_reference() -> None:
    """A multi-token batch whose every topk_length[t] == topk must match the
    reference. CUDA dispatch avoids a host read of the length tensor, so uniform
    batches can still use the per-token entrypoint."""
    device = require_b12x_sparse_mla()
    topk, num_tokens = 512, 4
    full = torch.full((num_tokens,), topk, dtype=torch.int32, device=device)
    got_O, exp_O = _run_unified_dsv4_multitoken(
        device,
        num_tokens=num_tokens,
        topk=topk,
        lengths=full,
        neg_pad_past_len=False,
        forced_num_splits=1,
        seed=99,
    )
    for t in range(num_tokens):
        assert _cosine(got_O[t], exp_O[t]) > 0.999, f"uniform batch token {t}"


@torch.inference_mode()
def test_unified_decode_mixed_length_routes_to_per_token_kernel() -> None:
    """A genuinely-mixed-length multi-token batch must route to the per-token
    kernel (per_token_len=True in the plan side-channel)."""
    device = require_b12x_sparse_mla()
    import b12x.attention._shared.mla.kernel as L

    topk, num_tokens = 512, 4
    lengths = _mixed_lengths(num_tokens, topk, device)
    _run_unified_dsv4_multitoken(
        device,
        num_tokens=num_tokens,
        topk=topk,
        lengths=lengths,
        neg_pad_past_len=False,
        forced_num_splits=1,
        seed=99,
    )
    assert L.LAST_DECODE_PLAN.get("per_token_len") is True, (
        "mixed-length batch did not route to the per-token kernel"
    )


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_unified_decode_dual_cache_multitoken_per_token_length(num_tokens) -> None:
    """DSV4 dual-cache decode with MIXED per-token MAIN topk_length AND per-token
    EXTRA extra_topk_length (num_tokens in {1,4,16}, num_heads=128) matches
    dsv4_extra_decode_reference over the masked union, per token."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_decode

    topk, extra_topk, pbs_extra = 512, 128, 2
    main_blocks = _UNIFIED_NUM_BLOCKS
    case = dsv4_extra_ref.make_dsv4_extra_decode_case(
        num_heads=_MT_HEADS,
        topk=topk,
        extra_topk=extra_topk,
        num_tokens=num_tokens,
        num_blocks=main_blocks,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        invalidate_half=False,
        with_sink=False,
        device=device,
        seed=7000 + num_tokens,
    )
    q = case["q"].contiguous()
    swa_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, main_blocks)
    extra_blocks = case["extra_kv_cache"].shape[0]
    idx_cache = _repack_dsv4_to_compressed(
        case["extra_kv_cache"], pbs_extra, extra_blocks
    )
    main_idx = case["topk_indices"].contiguous()
    extra_idx = case["extra_indices"].contiguous()

    main_len = _mixed_lengths(num_tokens, topk, device)
    extra_len = _mixed_lengths(num_tokens, extra_topk, device)

    exp_O, _ = dsv4_extra_ref.dsv4_extra_decode_reference(
        q,
        case["kv_cache"],
        main_idx,
        case["sm_scale"],
        case["extra_kv_cache"],
        extra_idx,
        page_block_size=_DSV4_PAGE,
        pbs_extra=pbs_extra,
        topk_length=main_len,
        extra_topk_length=extra_len,
        main_kv_dequant=case["kv_dequant"],
        extra_kv_dequant=case["extra_kv_dequant"],
    )
    exp_O = exp_O.float()

    n_chunks = (topk + 64 - 1) // 64 + (extra_topk + 64 - 1) // 64
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=_MT_HEADS,
        max_q_rows=num_tokens,
        max_width=topk + extra_topk,
        head_dim=_DSV4_HEAD_DIM,
        v_head_dim=_DSV4_HEAD_DIM,
        max_chunks_per_row=max(8, n_chunks),
        page_size=_DSV4_PAGE,
    )
    layout = _compressed_sparse_mla_scratch_layout(caps)
    storage = torch.zeros(int(layout.nbytes), dtype=torch.uint8, device=device)
    scratch = _materialize_compressed_sparse_mla_scratch(caps, storage, layout)

    out = run_unified_decode(
        q_all=q,
        swa_k_cache=swa_cache,
        swa_indices=main_idx,
        swa_topk_lengths=main_len,
        workspace=scratch,
        sm_scale=case["sm_scale"],
        swa_page_size=_DSV4_PAGE,
        indexed_k_cache=idx_cache,
        indexed_indices=extra_idx,
        indexed_topk_lengths=extra_len,
        indexed_page_size=pbs_extra,
        forced_num_splits=2,
    )
    torch.cuda.synchronize()
    got = out.float()
    for t in range(num_tokens):
        cos = _cosine(got[t], exp_O[t])
        assert cos > 0.999, (
            f"dual T={num_tokens} token {t} (main_len={int(main_len[t])} "
            f"extra_len={int(extra_len[t])}) O cos={cos}"
        )
        assert (got[t] - exp_O[t]).abs().max().item() < 2e-2


# ── P10e GLM PREFILL (extend) MIXED per-token topk_length incl a ZERO-LENGTH row ──
# The extend API (sparse_mla_extend_forward) threads its per-token active_token_counts
# into run_unified_prefill as the per-token topk_length; prefill masks each token's
# candidates past section_len=topk_length[t]. The catastrophic gap this guards: a
# token with topk_length[t]==0 (a TRUE zero-length row) AND NON--1-padded indices.
# With section_len==0 every candidate is masked, but the all-masked online softmax
# leaves a SPURIOUS positive global_sum (exp2(qk-local_max)=exp2(0)=1 for the FINITE
# _QK_MASK sentinel), so without the empty-row guard S7 normalizes garbage instead of
# writing O=0 / LSE=-inf. This exercises the REAL launcher and compares per token vs
# glm_decode_reference (the single-pass prefill shares the GLM s0-s7 math), settling
# both contracts: real per-token lengths with VALID indices past the length (the case
# the masking bug got catastrophically wrong) AND -1-padded indices.
def _glm_prefill_mixed_lengths(
    num_tokens: int, topk: int, device: torch.device
) -> torch.Tensor:
    """Per-token MIXED lengths with a TRUE zero-length row at token 1 (the
    extend-API zero active_token_counts contract). Other tokens are off-64-boundary
    and spread across [1, topk]; token 0 is full topk so a uniform check can't hide
    mixing."""
    base = [topk, 0, topk - 7, 64 + 13, 1, 128 + 5, topk // 2, 3]
    vals = [base[i % len(base)] for i in range(num_tokens)]
    if num_tokens >= 1:
        vals[0] = topk  # full-length token
    if num_tokens >= 2:
        vals[1] = 0  # TRUE zero-length row (the catastrophic case)
    vals = [max(0, min(int(v), topk)) for v in vals]
    return torch.tensor(vals, dtype=torch.int32, device=device)


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [2, 5, 16])
@pytest.mark.parametrize("neg_pad_past_len", [False, True])
def test_unified_prefill_glm_mixed_per_token_length_with_zero_row(
    num_tokens, neg_pad_past_len
) -> None:
    """run_unified_prefill (the GLM extend route) honours MIXED per-token topk_length
    INCLUDING a true zero-length row + non--1-padded valid indices, matching
    glm_decode_reference(active_token_counts=lengths) per token. The zero-length row
    must produce O==0 (cos==1 via _cosine's 0-norm guard, plus an explicit norm
    check) -- without the empty-row guard it normalized a spurious all-masked softmax
    sum to garbage (the cos~0.20 extend failure)."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_prefill

    # MG-eligible GLM width (topk in {512,1024,2048}); the off-64-boundary
    # partial-last-tile + zero-row coverage now comes from the MIXED per-token
    # lengths below (not multiples of 64), not from topk itself. (The decode-reuse
    # fallback that handled arbitrary topk was removed -- unsupported topk RAISEs.)
    topk = 512
    nblk = max(1, (topk + _GLM_PAGE - 1) // _GLM_PAGE)
    case = glm_ref.make_glm_decode_case(
        num_heads=_GLM_NUM_HEADS,
        topk=topk,
        num_tokens=num_tokens,
        num_blocks=nblk,
        page_block_size=_GLM_PAGE,
        invalidate_half=False,
        seed=9000 + num_tokens,
        device=device,
    )
    q = case["q"].contiguous()  # [T, 128, 576]
    kv_cache = case["kv_cache"].contiguous()
    idx = case["topk_indices"].contiguous()  # [T, topk] (ALL valid -> non -1-padded)
    sm_scale = case["sm_scale"]
    lengths = _glm_prefill_mixed_lengths(num_tokens, topk, device)
    assert int((lengths == 0).sum()) >= 1, "test must include a true zero-length row"

    if neg_pad_past_len:
        # Also settle the -1-padded contract: force indices past each length to -1
        # (a zero-length row becomes a fully -1 row). Both the section bound and the
        # S3 idx<0 mask then agree.
        ar = torch.arange(topk, device=device).unsqueeze(0)
        idx = idx.clone()
        idx[ar >= lengths.unsqueeze(-1)] = -1

    # Reference masks per token by length (and idx<0 for the padded case).
    exp_O = glm_ref.glm_decode_reference(
        q,
        kv_cache,
        idx,
        sm_scale,
        active_token_counts=lengths,
    ).float()

    O, lse = run_unified_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_indices=idx,
        sm_scale=sm_scale,
        page_block_size=_GLM_PAGE,
        topk_length=lengths,
    )
    torch.cuda.synchronize()
    got = O.float()
    for t in range(num_tokens):
        if int(lengths[t]) == 0:
            # Zero-length row: O must be exactly zero and LSE the -inf empty sentinel.
            assert got[t].abs().max().item() == 0.0, (
                f"GLM prefill T={num_tokens} zero-length token {t} O not zero: "
                f"norm={got[t].norm().item()}"
            )
            assert torch.isinf(lse[t]).all() and (lse[t] < 0).all(), (
                f"GLM prefill zero-length token {t} LSE must be -inf, got {lse[t]}"
            )
            continue
        cos = _cosine(got[t], exp_O[t])
        assert cos > 0.995, (
            f"GLM prefill T={num_tokens} token {t} (len={int(lengths[t])}, "
            f"neg_pad={neg_pad_past_len}) O cos={cos}"
        )
        assert (got[t] - exp_O[t]).abs().max().item() < 3e-2


@torch.inference_mode()
@pytest.mark.parametrize("topk", [192, 100])
def test_unified_prefill_dsv4_sub_container_topk_matches_ref(topk: int) -> None:
    """A short prefill clamps the runtime top-k below index_topk (192 for a
    192-token sequence); the entry must widen the rows to the 512 container and
    match both the reference and a caller-padded 512-wide call."""
    device = require_b12x_sparse_mla()
    from b12x.attention._shared.mla.kernel import run_unified_prefill

    num_tokens, num_heads = 16, _DSV4_HEADS
    num_blocks = 8
    case = prefill_ref.make_dsv4_prefill_case(
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        num_blocks=num_blocks,
        page_block_size=_DSV4_PAGE,
        with_sink=False,
        invalidate_half=True,
        device=device,
        seed=192,
    )
    q = case["q"].contiguous()
    swa_cache = _repack_dsv4_to_compressed(case["kv_cache"], _DSV4_PAGE, num_blocks)
    idx = case["topk_indices"].contiguous()
    lengths = case["topk_lengths"].contiguous()

    O, lse = run_unified_prefill(
        q=q,
        kv_cache=swa_cache,
        topk_indices=idx,
        topk_length=lengths,
        sm_scale=case["sm_scale"],
        page_block_size=_DSV4_PAGE,
    )
    padded_idx = torch.nn.functional.pad(idx, (0, 512 - topk), value=-1).contiguous()
    O_padded, lse_padded = run_unified_prefill(
        q=q,
        kv_cache=swa_cache,
        topk_indices=padded_idx,
        topk_length=lengths,
        sm_scale=case["sm_scale"],
        page_block_size=_DSV4_PAGE,
    )
    torch.cuda.synchronize()

    got = O.float()
    exp = case["expected_O"].float()
    assert got.shape == (num_tokens, num_heads, _DSV4_HEAD_DIM)
    assert torch.isfinite(got).all()
    assert torch.equal(O, O_padded)
    assert torch.equal(lse, lse_padded)
    cos = _cosine(got, exp)
    assert cos > 0.999, f"DSV4 prefill topk={topk} O cos={cos}"
    assert (got - exp).abs().max().item() < 2e-2
    assert (lse.float() - case["expected_lse"].float()).abs().max().item() < 5e-2
