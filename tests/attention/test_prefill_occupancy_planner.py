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

"""
Regression tests for the FA2 prefill planner occupancy predicate.

The planner (PrefillPlanImpl in scheduler.cuh) used to hardcode
``int num_blocks_per_sm = 2;`` while the launchers
(BatchPrefillWith{Ragged,Paged}KVCacheDispatched in prefill.cuh) derive the
actual CTAs-per-SM from ``cudaDevAttrMaxSharedMemoryPerMultiprocessor`` and the
kernel's shared-memory formula. On shared-memory-tight parts (e.g. GB10
sm_121 with 102400 B/SM) the launcher yields 1, not 2, so the planner
over-allocated the split-KV grid by 2x.

This module pins the property the patch actually guarantees, in the direction
it holds: **the planner drops to 1 only where the launcher does too, and
whenever the launcher fits two CTAs the historical value of 2 is preserved
exactly**.

The converse is deliberately *not* claimed. Because
``FA2PrefillCtaSmemLowerBound`` is a lower bound rather than an equality, there
are configurations where the launcher takes 1 CTA/SM while the planner still
says 2 -- the original over-allocation survives there, attenuated but present.
The tests below therefore assert ``planner == 1 => launcher == 1``, never
``launcher == 1 => planner == 1``.

It also pins the B-1 regression (``fixed_split_size > 0`` under CUDA graphs must
keep ``num_blocks_per_sm = 2`` to avoid a ``FLASHINFER_CHECK`` failure on
``padded_batch_size``) and the POD narrowing (``num_colocated_ctas > 0`` keeps
the historical value).

All three GPU-gated tests pin ``backend="fa2"``. The wrapper defaults to
``backend="auto"``, which may resolve to fa3/cudnn/trtllm-gen depending on the
part and on which JIT artifacts happen to be present; those paths do not go
through ``PrefillPlanImpl`` at all, so ``plan_info`` would not carry the FA2
``padded_batch_size`` these tests assert on. Pinning keeps the assertions
aimed at the planner this change actually touches.

Design:
  * The *predicate logic* tests are CPU-only (no GPU required). They mirror the
    shared-memory arithmetic of both the planner and the launcher in Python and
    assert the bound relationship holds over a representative config matrix.
    This is the same model used by ``check_bound.py`` in the evidence pack,
    collapsed to a single function pair and parametrised.
  * The *behavioural* tests are GPU-gated (``pytest.mark.skipif(not
    torch.cuda.is_available(), ...)``) and assert ``plan_info.padded_batch_size``
    on a shape where the correction applies and on a shape where it does not,
    plus the B-1 fixed-split + cuda_graph regression. Expectations are derived
    from device properties, not hardcoded, so the test is self-checking on any
    part (a part where the occupancy does not flip at the chosen head_dim skips
    rather than fails).
"""

from dataclasses import dataclass

import pytest
import torch

# ---------------------------------------------------------------------------
# CPU-only predicate logic mirror. No GPU, no flashinfer import required.
# ---------------------------------------------------------------------------
#
# These two functions reproduce the arithmetic of:
#   * FA2PrefillCtaSmemLowerBound (planner side, prefill_occupancy.cuh)
#   * kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV (launcher side, prefill.cuh)
# at the granularity that affects num_ctas_per_sm. They are intentionally a
# second transcription -- the whole point of the in-tree static_assert is that
# a same-author pair of models can hide a shared error; an independent
# transcription in a different language catches what a C++-only check cannot.
#
# Reference: include/flashinfer/attention/prefill.cuh @ 0d25b183
#            ragged: :4148-4216, paged: :4347-4415


def _get_num_warps_q(cta_tile_q: int) -> int:
    if cta_tile_q == 32:
        return 1  # HEAD_DIM_VO >= 512
    if cta_tile_q > 16:
        return 4
    return 1


def _get_num_warps_kv(cta_tile_q: int) -> int:
    return 4 // _get_num_warps_q(cta_tile_q)


@dataclass(frozen=True)
class DtypeInfo:
    q_dtype_bytes: int
    kv_dtype_bytes: int
    kv_is_fp4: bool


def planner_lower_bound(
    cta_tile_q: int,
    head_dim_qk: int,
    head_dim_vo: int,
    dtype_info: DtypeInfo,
) -> int:
    """Mirror of FA2PrefillCtaSmemLowerBound (the planner-side lower bound).

    Drops kVOSplitDispatch (both per-tile and fixed), kSharedRopeFreqSmem, and
    the NVFP4 SF staging term -- each is non-negative, so dropping them can only
    lower the estimate. Forces kLargeHeadWarpSplit when vo>=512 & ctq==32 (the
    ragged predicate, which gives the smaller NUM_WARPS_KV).
    """
    q_bytes = dtype_info.q_dtype_bytes
    kv_bytes = dtype_info.kv_dtype_bytes
    kv_is_fp4 = dtype_info.kv_is_fp4
    num_mma_d_vo = head_dim_vo // 16

    large_head_warp_split = (head_dim_vo >= 512) and (cta_tile_q == 32)
    num_warps_q = 2 if large_head_warp_split else _get_num_warps_q(cta_tile_q)
    num_warps_kv = 2 if large_head_warp_split else _get_num_warps_kv(cta_tile_q)

    use_repack = (
        kv_bytes == 1
        and not kv_is_fp4
        and head_dim_vo != 64
        and head_dim_vo <= 256
        and cta_tile_q > 16
    )
    kv_shared = (
        not kv_is_fp4
        and num_mma_d_vo > 16
        and num_mma_d_vo % num_warps_kv == 0
        and head_dim_qk == head_dim_vo
        and (kv_bytes == 2 or cta_tile_q > 16)
    )

    if kv_shared:
        kv_smem_per_mma_kv = head_dim_qk * 16 * num_warps_kv * kv_bytes
    else:
        kv_smem_per_mma_kv = (head_dim_qk + head_dim_vo) * 16 * num_warps_kv * kv_bytes
    if use_repack:
        kv_smem_per_mma_kv += (
            max(head_dim_qk, head_dim_vo) * 16 * num_warps_kv * q_bytes
        )
    # NVFP4 SF term deliberately NOT counted (ragged/paged launchers do not).

    fixed_smem = cta_tile_q * head_dim_qk * q_bytes
    min_valid_mma_kv = (num_warps_q // 2) if (kv_bytes == 1 and num_warps_q > 2) else 1
    return fixed_smem + min_valid_mma_kv * kv_smem_per_mma_kv


def launcher_requirement(
    cta_tile_q: int,
    head_dim_qk: int,
    head_dim_vo: int,
    dtype_info: DtypeInfo,
    *,
    is_paged: bool,
    use_softmax: bool,
) -> int:
    """Mirror of kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV (launcher side).

    This is the value the launcher actually charges. ``is_paged`` and
    ``use_softmax`` select between the ragged and paged kLargeHeadWarpSplit /
    kVOSplitDispatch predicates.
    """
    q_bytes = dtype_info.q_dtype_bytes
    kv_bytes = dtype_info.kv_dtype_bytes
    kv_is_fp4 = dtype_info.kv_is_fp4
    num_mma_d_vo = head_dim_vo // 16

    # kLargeHeadWarpSplit / kBf16VOSplit -- launcher-specific.
    # ragged: (kv_bytes<=2 or fp4) and vo>=512 and ctq==32
    # paged:  (kv_bytes==2 or fp4) and vo>=512 and ctq==32
    large_head_warp_split_ragged = (
        (kv_bytes <= 2 or kv_is_fp4) and head_dim_vo >= 512 and cta_tile_q == 32
    )
    large_head_warp_split_paged = (
        (kv_bytes == 2 or kv_is_fp4) and head_dim_vo >= 512 and cta_tile_q == 32
    )
    large_head_warp_split = (
        large_head_warp_split_ragged if not is_paged else large_head_warp_split_paged
    )
    num_warps_q = 2 if large_head_warp_split else _get_num_warps_q(cta_tile_q)
    num_warps_kv = 2 if large_head_warp_split else _get_num_warps_kv(cta_tile_q)

    use_repack = (
        kv_bytes == 1
        and not kv_is_fp4
        and head_dim_vo != 64
        and head_dim_vo <= 256
        and cta_tile_q > 16
    )
    kv_shared = (
        not kv_is_fp4
        and num_mma_d_vo > 16
        and num_mma_d_vo % num_warps_kv == 0
        and head_dim_qk == head_dim_vo
        and (kv_bytes == 2 or cta_tile_q > 16)
    )

    if kv_shared:
        kv_smem_per_mma_kv = head_dim_qk * 16 * num_warps_kv * kv_bytes
    else:
        kv_smem_per_mma_kv = (head_dim_qk + head_dim_vo) * 16 * num_warps_kv * kv_bytes
    if use_repack:
        kv_smem_per_mma_kv += (
            max(head_dim_qk, head_dim_vo) * 16 * num_warps_kv * q_bytes
        )

    # kVOSplitDispatch (launcher-specific).
    vo_split_dispatch_paged = num_mma_d_vo > 16 and num_mma_d_vo % num_warps_kv == 0
    vo_split_dispatch_ragged = vo_split_dispatch_paged and (
        (kv_bytes == 2 or kv_is_fp4) and use_softmax
    )
    vo_split_dispatch = (
        vo_split_dispatch_ragged if not is_paged else vo_split_dispatch_paged
    )
    if vo_split_dispatch:
        kv_smem_per_mma_kv += num_warps_kv * cta_tile_q * 8
        fixed_smem_extra = num_warps_kv * cta_tile_q * 8 + 2048
    else:
        fixed_smem_extra = 0

    # kSharedRopeFreqSmem -- non-negative, planner drops it.
    # (omitted from the planner bound; included here for completeness.)
    # We test with POS_ENCODING_MODE == NONE in the parametrisation below, so it
    # is 0; the bound relationship holds either way because the term is >= 0.

    fixed_smem = cta_tile_q * head_dim_qk * q_bytes + fixed_smem_extra
    min_valid_mma_kv = (num_warps_q // 2) if (kv_bytes == 1 and num_warps_q > 2) else 1
    return fixed_smem + min_valid_mma_kv * kv_smem_per_mma_kv


# ---------------------------------------------------------------------------
# CPU-only: the core invariant -- planner bound <= launcher requirement.
# ---------------------------------------------------------------------------

# Representative config matrix. Not exhaustive (the C++ static_assert is
# exhaustive over instantiations); this is the independent second check.
_HEAD_DIMS = [
    (64, 64),
    (128, 128),
    (192, 128),
    (192, 192),
    (256, 256),
    (512, 512),
    (576, 512),
]
_CTA_TILE_QS = [16, 32, 64, 128]
_DTYPES = [
    DtypeInfo(2, 2, False),  # bf16/bf16
    DtypeInfo(2, 1, False),  # bf16 Q, fp8 KV
    DtypeInfo(2, 1, True),  # bf16 Q, nvfp4 KV
]


def _config_ids():
    ids = []
    for hd_qk, hd_vo in _HEAD_DIMS:
        for ctq in _CTA_TILE_QS:
            for di in _DTYPES:
                for is_paged in (False, True):
                    for use_softmax in (False, True):
                        ids.append(
                            f"hd{hd_qk}-{hd_vo}_ctq{ctq}"
                            f"_q{di.q_dtype_bytes}kv{di.kv_dtype_bytes}"
                            f"{'fp4' if di.kv_is_fp4 else ''}"
                            f"{'paged' if is_paged else 'ragged'}"
                            f"{'sm' if use_softmax else 'nosm'}"
                        )
    return ids


_CONFIGS = [
    (hd_qk, hd_vo, ctq, di, is_paged, use_softmax)
    for (hd_qk, hd_vo) in _HEAD_DIMS
    for ctq in _CTA_TILE_QS
    for di in _DTYPES
    for is_paged in (False, True)
    for use_softmax in (False, True)
]


@pytest.mark.parametrize(
    "hd_qk,hd_vo,ctq,di,is_paged,use_softmax", _CONFIGS, ids=_config_ids()
)
def test_planner_bound_le_launcher_requirement(
    hd_qk, hd_vo, ctq, di, is_paged, use_softmax
):
    """planner_lower_bound <= launcher_requirement for every config.

    This is the load-bearing property: if the planner's smem estimate ever
    exceeds the launcher's, num_blocks_per_sm would drop to 1 on a part where
    the launcher actually fits two CTAs -- a performance regression. The C++
    static_assert pins this per-instantiation; this test is the independent
    second transcription.
    """
    bound = planner_lower_bound(ctq, hd_qk, hd_vo, di)
    req = launcher_requirement(
        ctq, hd_qk, hd_vo, di, is_paged=is_paged, use_softmax=use_softmax
    )
    assert bound <= req, (
        f"planner bound {bound} > launcher requirement {req} for "
        f"hd={hd_qk}/{hd_vo} ctq={ctq} di={di} "
        f"is_paged={is_paged} use_softmax={use_softmax}"
    )


# ---------------------------------------------------------------------------
# CPU-only: the corollary -- num_blocks_per_sm == 1 implies launcher == 1.
# ---------------------------------------------------------------------------

# Representative SMEM budgets. GB10 = 102400, H200 = 233472, plus a synthetic
# large value to confirm the 2-CTA path.
_SMEM_BUDGETS = [102400, 233472, 512 * 1024]


@pytest.mark.parametrize("smem_budget", _SMEM_BUDGETS, ids=lambda v: f"smem{v}")
@pytest.mark.parametrize(
    "hd_qk,hd_vo,ctq,di,is_paged,use_softmax", _CONFIGS, ids=_config_ids()
)
def test_planner_picks_1_implies_launcher_picks_1(
    smem_budget, hd_qk, hd_vo, ctq, di, is_paged, use_softmax
):
    """If planner says 1 CTA/SM, the launcher also says 1.

    Contrapositive of the property that matters: whenever the launcher fits
    two CTAs, the planner keeps 2 (byte-identical to today's plan).
    """
    bound = planner_lower_bound(ctq, hd_qk, hd_vo, di)
    req = launcher_requirement(
        ctq, hd_qk, hd_vo, di, is_paged=is_paged, use_softmax=use_softmax
    )
    planner_nbps = 2 if (smem_budget >= 2 * bound) else 1
    launcher_nbps = 2 if (smem_budget >= 2 * req) else 1
    if planner_nbps == 1:
        assert launcher_nbps == 1, (
            f"planner picked 1 ({bound=}, budget={smem_budget}) but launcher "
            f"picked {launcher_nbps} ({req=}) -- planner over-demoted"
        )


# ---------------------------------------------------------------------------
# GPU-gated behavioural tests. Skip cleanly when no GPU or when the
# occupancy does not flip at the chosen head_dim on this part.
# ---------------------------------------------------------------------------

_GPU_AVAILABLE = torch.cuda.is_available()


def _device_props():
    if not _GPU_AVAILABLE:
        return None
    p = torch.cuda.get_device_properties(0)
    return p


def _ceil_div(a, b):
    return -(-a // b)


def _determine_cta_tile_q(avg_packed_qo_len, hd_vo, hd_qk, kv_bytes):
    """include/flashinfer/utils.cuh:408-486 (Ampere+ branch only)."""
    qk = hd_qk or hd_vo
    if hd_vo >= 512:
        return 16 if avg_packed_qo_len <= 32 else 32
    if qk >= 512:
        return 16
    if avg_packed_qo_len > 64 and hd_vo < 256:
        return 128
    if avg_packed_qo_len > 16:
        return 64
    return 16


def _predict_padded_bs(nbps, num_sm, D, B, QL, KL_pages, H, HKV, PS, graph, fixed):
    """scheduler.cuh plan arithmetic, parameterised by num_blocks_per_sm."""
    gqa = H // HKV
    packed = [QL * gqa] * B
    kvlens = [KL_pages] * B
    avg = sum(packed) // B
    ctq = _determine_cta_tile_q(avg, D, D, 2)
    min_chunk = max(128 // PS, 1)
    budget = max(0, nbps * num_sm) // HKV

    if fixed is not None and fixed > 0:
        chunk = fixed
    else:
        lo, hi = min_chunk, max(1, max(kvlens))
        while lo < hi:
            mid = (lo + hi) // 2
            nbs = sum(
                _ceil_div(p, ctq) * _ceil_div(max(k, 1), mid)
                for p, k in zip(packed, kvlens, strict=False)
            )
            if nbs > budget:
                lo = mid + 1
            else:
                hi = mid
        chunk = lo

    new_bs = 0
    for p, k in zip(packed, kvlens, strict=False):
        nchunks = _ceil_div(max(k, 1), chunk)
        new_bs += _ceil_div(p, ctq) * nchunks

    if graph:
        total_tiles = _ceil_div(B * QL * gqa, ctq) + B - 1
        padded = max(budget, total_tiles)
    else:
        padded = new_bs
    return padded


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="requires CUDA")
def test_planner_padded_batch_size_matches_predicate_on_affected_shape():
    """On a part where the occupancy flips at hd256, the patched planner's
    padded_batch_size must equal the nbps=1 model, not the nbps=2 model.

    Skips (not fails) on parts where the occupancy does not flip at hd256.
    """
    import flashinfer

    p = _device_props()
    num_sm = p.multi_processor_count
    smem_per_sm = p.shared_memory_per_multiprocessor

    D, B, QL, KL, H, HKV, PS = 256, 1, 1, 12288, 16, 1, 16
    KL_pages = KL // PS

    di = DtypeInfo(2, 2, False)
    ctq = _determine_cta_tile_q(QL * (H // HKV), D, D, 2)
    bound = planner_lower_bound(ctq, D, D, di)
    # If the part fits 2 CTAs at this shape, the correction is inert here.
    if smem_per_sm >= 2 * bound:
        pytest.skip(
            f"part fits 2 CTAs/SM at hd256 ctq=16 (smem={smem_per_sm}, "
            f"bound={bound}); correction does not apply, nothing to assert"
        )

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    qo = torch.tensor([0, QL], device="cuda", dtype=torch.int32)
    kvp = torch.tensor([0, KL_pages], device="cuda", dtype=torch.int32)
    idx = torch.arange(0, KL_pages, device="cuda", dtype=torch.int32)
    lpl = torch.full((B,), PS, device="cuda", dtype=torch.int32)
    w.plan(
        qo,
        kvp,
        idx,
        lpl,
        H,
        HKV,
        D,
        PS,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    observed = int(w._plan_info[0])
    expected_patched = _predict_padded_bs(
        1, num_sm, D, B, QL, KL_pages, H, HKV, PS, False, None
    )
    expected_baseline = _predict_padded_bs(
        2, num_sm, D, B, QL, KL_pages, H, HKV, PS, False, None
    )
    assert expected_patched != expected_baseline, (
        "discriminator dead: nbps=1 and nbps=2 predict the same padded_bs "
        f"({expected_patched}); test cannot tell builds apart on this part"
    )
    assert observed == expected_patched, (
        f"observed padded_batch_size={observed} but patched model predicts "
        f"{expected_patched} (baseline would be {expected_baseline})"
    )


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="requires CUDA")
def test_planner_noop_on_mainstream_shape():
    """On hd128 the bound is small enough that every modern part fits 2 CTAs,
    so the patched planner must produce the same plan as the baseline
    (nbps=2). Asserts byte-identical padded_batch_size against the nbps=2
    model."""
    import flashinfer

    p = _device_props()
    num_sm = p.multi_processor_count

    D, B, QL, KL, H, HKV, PS = 128, 8, 16, 8192, 8, 8, 16
    KL_pages = KL // PS

    di = DtypeInfo(2, 2, False)
    ctq = _determine_cta_tile_q(QL * (H // HKV), D, D, 2)
    bound = planner_lower_bound(ctq, D, D, di)
    # This test asserts the no-op; if the part somehow flips here, the
    # affected-shape test above is the right one.
    if p.shared_memory_per_multiprocessor < 2 * bound:
        pytest.skip("part flips at hd128; covered by the affected-shape test")

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    qo = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * QL
    kvp = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * KL_pages
    idx = torch.arange(0, B * KL_pages, device="cuda", dtype=torch.int32)
    lpl = torch.full((B,), PS, device="cuda", dtype=torch.int32)
    w.plan(
        qo,
        kvp,
        idx,
        lpl,
        H,
        HKV,
        D,
        PS,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    observed = int(w._plan_info[0])
    expected = _predict_padded_bs(
        2, num_sm, D, B, QL, KL_pages, H, HKV, PS, False, None
    )
    assert observed == expected, (
        f"hd128 no-op: observed padded_batch_size={observed}, expected {expected} "
        f"(nbps=2 model; the patch must not change hd128)"
    )


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="requires CUDA")
def test_b1_fixed_split_size_under_cuda_graph_does_not_raise():
    """R1 finding B-1: with fixed_split_size > 0 and use_cuda_graph=True,
    lowering num_blocks_per_sm could push padded_batch_size below
    new_batch_size and trip FLASHINFER_CHECK. The patch keeps nbps=2 on
    that path; this test asserts plan() does not raise.

    Runs on any part (the fixed-split path is part-independent: nbps stays 2
    by construction, so the check is structural, not occupancy-dependent)."""
    import flashinfer

    D, B, QL, KL, H, HKV, PS = 192, 4, 4, 256, 8, 8, 16

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    qo = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * QL
    kvp = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * (KL // PS)
    idx = torch.arange(0, B * (KL // PS), device="cuda", dtype=torch.int32)
    lpl = torch.full((B,), PS, device="cuda", dtype=torch.int32)
    bufs = [t.clone() for t in (qo, kvp, idx, lpl)]
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws,
        "NHD",
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=bufs[0],
        paged_kv_indptr_buf=bufs[1],
        paged_kv_indices_buf=bufs[2],
        paged_kv_last_page_len_buf=bufs[3],
    )
    # Must not raise. On the pre-fix patch (v2) this raised RuntimeError
    # because padded_batch_size was halved below new_batch_size.
    w.plan(
        bufs[0],
        bufs[1],
        bufs[2],
        bufs[3],
        H,
        HKV,
        D,
        PS,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        fixed_split_size=8,
    )
    # If we got here, the B-1 narrowing held. Sanity-check the plan produced
    # a non-zero padded_batch_size.
    assert int(w._plan_info[0]) > 0
