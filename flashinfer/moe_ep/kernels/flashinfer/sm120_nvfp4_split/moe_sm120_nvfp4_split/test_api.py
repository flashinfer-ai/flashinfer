"""Host-only tests for the FlashInfer-facing configuration API."""

from dataclasses import replace

from .api import MegaMoEProblemSpec, select_compile_spec
from .jit_config import Sm120JitConfig
from .config import (
    RANK_COMBINE_MAX_GROUPS,
    RANK_COMBINE_PLAN_COUNT_SHIFT,
    RANK_COMBINE_PLAN_MAX_TOPK,
    RANK_COMBINE_PLAN_SLOT_BITS,
    _rank_local_vector_reduce_supported,
)


def _problem() -> MegaMoEProblemSpec:
    return MegaMoEProblemSpec(
        tokens_per_rank=8192,
        num_topk=6,
        num_total_experts=256,
        hidden=4096,
        intermediate=4096,
        expert_parallel_size=4,
        expert_parallel_rank=0,
    )


def _spec(jit: Sm120JitConfig = Sm120JitConfig()):
    return select_compile_spec(
        problem=_problem(),
        numerical_contract="owner_bf16",
        ep_same_numa_peer_count=3,
        ep_cross_numa_peer_count=0,
        num_sms=110,
        sm_min_partition=8,
        sm_partition_alignment=8,
        jit=jit,
    )


def test_cache_key_is_stable_for_equal_specs() -> None:
    lhs = _spec()
    rhs = _spec()
    assert lhs.canonical_dict() == rhs.canonical_dict()
    assert lhs.cache_key == rhs.cache_key


def test_cache_key_covers_codegen_options() -> None:
    baseline = _spec()
    modified = _spec(replace(baseline.jit, k2_token_stripe=3))
    assert baseline.cache_key != modified.cache_key


def test_cache_key_distinguishes_q8_from_q16() -> None:
    baseline = _spec()
    assert baseline.kernel.ready_queue_bundle == 8
    q16 = replace(
        baseline, kernel=replace(baseline.kernel, ready_queue_bundle=16)
    )
    assert baseline.cache_key != q16.cache_key


def test_cache_key_covers_gate_up_clamp() -> None:
    baseline = _spec()
    clamped = replace(
        baseline,
        problem=replace(baseline.problem, gate_up_clamp=10.0),
    )
    assert baseline.cache_key != clamped.cache_key


def test_cache_key_covers_tile_and_profiling_specializations() -> None:
    baseline = _spec()
    different_tile = replace(
        baseline,
        kernel=replace(baseline.kernel, k2_tile=(64, 64, 128)),
    )
    instrumented = replace(
        baseline,
        jit=replace(baseline.jit, enable_globaltimer=True),
    )
    assert baseline.cache_key != different_tile.cache_key
    assert baseline.cache_key != instrumented.cache_key


def test_cache_key_covers_rank_specialization() -> None:
    baseline = _spec()
    rank_one = replace(
        baseline,
        problem=replace(baseline.problem, expert_parallel_rank=1),
    )
    assert baseline.cache_key != rank_one.cache_key


def test_compile_spec_has_only_supported_dispatch_and_jit_options() -> None:
    spec = _spec()
    serialized = spec.canonical_dict()
    assert "dispatch_pull_mode" not in serialized["kernel"]
    assert "dispatch_warps_per_tile" not in serialized["kernel"]
    assert "fc12_stream_tiles" not in serialized["jit"]
    # The two real handoff controls remain distinct and cache-visible.
    assert serialized["jit"]["fc2_ready_bundle_k_tiles"] == 12
    assert serialized["kernel"]["ready_queue_bundle"] > 0
    assert "rank_local_vector_reduce" not in serialized["kernel"]
    assert "rank_local_vector_reduce" not in serialized["jit"]


def test_rank_local_vector_reduce_layout_guard() -> None:
    for hidden in (2048, 4096, 6144, 8192):
        for topk in range(1, 9):
            assert _rank_local_vector_reduce_supported(hidden, topk)
    for hidden in (0, -2048, 512, 1536, 2560, 3072, 7168):
        assert not _rank_local_vector_reduce_supported(hidden, 6)
    for topk in (0, 9, 16, 32):
        assert not _rank_local_vector_reduce_supported(4096, topk)


def test_rank_local_plan_encoding_preserves_slot_order() -> None:
    assert RANK_COMBINE_PLAN_MAX_TOPK * RANK_COMBINE_PLAN_SLOT_BITS == RANK_COMBINE_PLAN_COUNT_SHIFT
    mask = (1 << RANK_COMBINE_PLAN_SLOT_BITS) - 1
    for topk in range(1, RANK_COMBINE_PLAN_MAX_TOPK + 1):
        for owners in range(1 << topk):
            slots = [s for s in range(topk) if owners & (1 << s)]
            word = len(slots) << RANK_COMBINE_PLAN_COUNT_SHIFT
            for index, slot in enumerate(slots):
                word |= slot << (RANK_COMBINE_PLAN_SLOT_BITS * index)
            count = word >> RANK_COMBINE_PLAN_COUNT_SHIFT
            decoded = [(word >> (RANK_COMBINE_PLAN_SLOT_BITS * i)) & mask
                       for i in range(count)]
            assert decoded == slots
            assert word < (1 << 31)
            if not slots:
                assert word == 0


def test_rank_local_vector_lanes_cover_each_channel_once() -> None:
    for hidden in (2048, 4096, 6144, 8192):
        for groups in (1, RANK_COMBINE_MAX_GROUPS):
            chunk = hidden // groups
            channels = []
            for offset in range(0, chunk, 512):
                for lane in range(16):
                    for buffer in range(2):
                        for part in range(2):
                            h = offset + buffer * 256 + lane * 8 + part * 128
                            assert (h * 2) % 16 == 0
                            channels.extend(range(h, h + 8))
            assert sorted(channels) == list(range(chunk))


def test_cache_key_covers_workspace_abi(monkeypatch) -> None:
    from . import api

    old = _spec().cache_key
    monkeypatch.setattr(api, "KERNEL_CACHE_ABI", api.KERNEL_CACHE_ABI + 1)
    assert _spec().cache_key != old


def test_p2p_dispatch_admission_does_not_depend_on_rank_local_combine() -> None:
    """Both combine modes publish counts into a peer-reset control region.

    Guard the source-level admission placement without importing CUDA kernels.
    The dynamic GPU reset-skew regression tests the cross-rank ordering itself.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).with_name("token_comm.py").read_text())
    dispatch = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "dispatch_warp_body"
    )
    backend = next(
        node for node in dispatch.body
        if isinstance(node, ast.If) and "gin" in ast.unparse(node.test)
    )
    # These must be direct statements in the P2P branch, not RLC-conditional.
    admission, preparation = backend.orelse[:2]
    assert isinstance(admission, ast.Expr) and isinstance(admission.value, ast.Call)
    assert admission.value.func.attr == "nvlink_barrier"
    assert isinstance(preparation, ast.Expr) and isinstance(preparation.value, ast.Call)
    assert preparation.value.func.attr == "dispatch_prep"
    options = {kw.arg: ast.literal_eval(kw.value) for kw in admission.value.keywords
               if kw.arg in ("prologue_grid_sync", "epilogue_grid_sync")}
    assert options == {"prologue_grid_sync": True, "epilogue_grid_sync": True}
