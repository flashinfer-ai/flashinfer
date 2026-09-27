"""Cake SM90 VSA CuTe DSL build (``backend="cake_cute"`` on Hopper).

The route runs the same fifteen kernels and the same planner as ``backend="cake"``;
these tests pin (1) every stage against the FP32 reference at the CUDA route's
tolerances, (2) bit-exact agreement with the CUDA build per stage, (3) the split
and cluster replay contracts (two runs bit-exact, CUDA-graph replay) and (4) the
wrapper API on the new backend name.
"""

import pytest
import torch

from flashinfer.cake_vsa_sm90 import CakeVsaSm90Plan, cluster_capacity, small_route
from flashinfer.experimental.cake_vsa_sm90_cute import STAGES, is_available, load_stage

requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="Cake SM90 VSA requires an SM90 GPU",
)
requires_cute = pytest.mark.skipif(
    not is_available(), reason="the CuTe DSL (nvidia-cutlass-dsl) is not installed"
)

TOL = dict(atol=1e-2, rtol=1e-2)
MAX_ABS = 0.03


def _random_mask(h, mb, nb, capacity, seed=0, ragged=True, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros((h, mb, nb), dtype=torch.bool)
    for head in range(h):
        for row in range(mb):
            count = (
                capacity
                if (row == 0 or not ragged)
                else 1 + (row * 7 + head) % capacity
            )
            mask[head, row, torch.randperm(nb, generator=g)[:count]] = True
    return mask.to(device)


def _inputs(h, mb, nb, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            (h, length * 64, 128), device="cuda", dtype=torch.bfloat16, generator=g
        )
        for length in (mb, nb, nb)
    )


def _reference(q, k, v, mask, scale):
    scores = torch.einsum("hmd,hnd->hmn", q.float(), k.float()) * float(scale)
    dense = mask.to(q.device).repeat_interleave(64, dim=1).repeat_interleave(64, dim=2)
    scores.masked_fill_(~dense, float("-inf"))
    return torch.einsum("hmn,hnd->hmd", torch.softmax(scores, dim=-1), v.float())


def _descriptors(mask):
    h, mb, nb = mask.shape
    rows = torch.full((h, mb), 64, dtype=torch.int32, device=mask.device)
    cols = torch.full((h, nb), 64, dtype=torch.int32, device=mask.device)
    return rows, cols


def _stage_of(plan):
    if plan.small_kmax is None:
        return "attention"
    if plan.small_cluster:
        return f"small_k{plan.small_kmax}c{plan.small_cluster}"
    return f"small_k{plan.small_kmax}{'s' if plan.small_split else ''}"


# One problem per planner-selectable stage: (route, h, mb, nb, capacity,
# ragged, seed).  The selection sizes steer ``small_kmax_for`` / ``split_kmax``
# / ``cluster_variant_for`` to the named variant; the test asserts the stage.
# ``small_k3s`` is compiled and exported for parity with the CUDA route but
# ``split_kmax`` never selects it (its modelled chain cost ties KMAX 4, which
# wins on ties with fewer slices); it is checked by a direct launch below.
STAGE_CASES = {
    "attention": ("persistent", 2, 8, 32, 6, True, 11),
    "small_k1": ("small", 4, 4, 4, 1, False, 11),
    "small_k3": ("small", 8, 4, 4, 3, True, 11),
    "small_k4": ("small", 8, 16, 16, 4, True, 11),
    "small_k6": ("small", 2, 8, 32, 6, True, 11),
    "small_k1s": ("smallsplit", 2, 4, 8, 3, True, 13),
    "small_k4s": ("smallsplit", 1, 1, 8, 6, False, 13),
    "small_k6s": ("smallsplit", 1, 1, 64, 64, False, 13),
    "small_k2c4": ("smallcluster", 1, 1, 8, 8, False, 19),
    "small_k3c3": ("smallcluster", 3, 5, 10, 9, False, 19),
    "small_k3c6": ("smallcluster", 1, 16, 32, 18, False, 19),
    "small_k4c2": ("smallcluster", 4, 16, 4, 1, True, 13),
    "small_k4c4": ("smallcluster", 1, 16, 16, 16, False, 19),
    "small_k6c2": ("smallcluster", 2, 8, 64, 12, False, 19),
}


def _plans(stage, scale=None):
    route, h, mb, nb, capacity, ragged, seed = STAGE_CASES[stage]
    mask = _random_mask(h, mb, nb, capacity, seed=seed, ragged=ragged, device="cuda")
    rows, cols = _descriptors(mask)
    plans = {
        engine: CakeVsaSm90Plan(
            "cuda",
            mask,
            rows,
            cols,
            h,
            h,
            128,
            sm_scale=scale,
            route=route,
            engine=engine,
        )
        for engine in ("cuda", "cute")
    }
    for plan in plans.values():
        assert _stage_of(plan) == stage, (_stage_of(plan), stage)
    return mask, plans


def test_manifest_lists_every_stage():
    for stage in STAGES:
        loaded = load_stage(stage)
        names = [name for kind, name in loaded.arg_plan if kind == "buffer"]
        assert names[:4] == ["Q", "K", "Vt", "O"]
        assert set(loaded.tma) == {"Q", "K", "Vt"}
        assert loaded.threads % 128 == 0 and loaded.dynamic_smem_bytes > 0
        if "c" in stage.split("_k")[-1]:
            assert loaded.cluster_dims[0] == int(stage.split("c")[-1])
        else:
            assert loaded.cluster_dims == (1, 1, 1)


@requires_hopper
@requires_cute
@pytest.mark.parametrize("stage", list(STAGE_CASES))
@pytest.mark.parametrize("scale", [None, -0.125])
def test_every_stage_matches_reference_and_cuda_build(stage, scale):
    """Route C vs the FP32 reference (CUDA-route tolerances) and vs route A, bit-exact."""
    mask, plans = _plans(stage, scale)
    h, mb, nb = mask.shape
    q, k, v = _inputs(h, mb, nb)
    reference = _reference(q, k, v, mask, 128**-0.5 if scale is None else scale)
    outputs = {engine: plan.run(q, k, v).clone() for engine, plan in plans.items()}
    torch.cuda.synchronize()
    for engine, out in outputs.items():
        torch.testing.assert_close(out.float(), reference, **TOL)
        assert float((out.float() - reference).abs().max()) <= MAX_ABS, engine
    diff = (outputs["cute"].float() - outputs["cuda"].float()).abs().max().item()
    assert torch.equal(outputs["cute"], outputs["cuda"]), (
        f"{stage}: CuTe build differs from the CUDA build, max |diff| = {diff}"
    )


@requires_hopper
@requires_cute
@pytest.mark.parametrize(
    "stage", [s for s in STAGE_CASES if s.endswith("s") or "c" in s.split("_k")[-1]]
)
def test_split_and_cluster_stages_replay_bit_exactly(stage):
    """Two runs and a CUDA-graph replay of the CuTe build agree at atol=0."""
    mask, plans = _plans(stage)
    plan = plans["cute"]
    h, mb, nb = mask.shape
    q, k, v = _inputs(h, mb, nb)
    first = plan.run(q, k, v).clone()
    second = plan.run(q, k, v).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    if plan.small_split:
        assert int(plan.counters.abs().sum()) == 0
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    plan.run(q, k, v, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, k, v, out=out)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.view_as(q), first, atol=0, rtol=0)
    q.normal_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.view_as(q).float(), _reference(q, k, v, mask, 128**-0.5), **TOL
    )


@requires_hopper
@requires_cute
def test_plan_built_on_another_stream():
    """The plan upload (device plan buffer) is ordered before a run on a different stream."""
    h, mb, nb = 1, 16, 16
    mask = _random_mask(h, mb, nb, 4, seed=17, ragged=True, device="cuda")
    rows, cols = _descriptors(mask)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        plan = CakeVsaSm90Plan(
            "cuda", mask, rows, cols, h, h, 128, route="small", engine="cute"
        )
    q, k, v = _inputs(h, mb, nb)
    out = plan.run(q, k, v)
    torch.testing.assert_close(out.float(), _reference(q, k, v, mask, 128**-0.5), **TOL)


@requires_hopper
@requires_cute
def test_wrapper_backend_cake_cute_auto_route_matches_cake():
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    h, mb, nb = 2, 16, 16
    mask = _random_mask(h, mb, nb, 12, seed=23, ragged=True, device="cuda")
    rows, cols = _descriptors(mask)
    wrappers = {}
    for backend in ("cake", "cake_cute"):
        wrapper = VariableBlockSparseAttentionWrapper(
            torch.empty(0, device="cuda", dtype=torch.uint8), backend=backend
        )
        wrapper.plan(
            mask, rows, cols, h, h, 128, q_data_type=torch.bfloat16, non_blocking=False
        )
        wrappers[backend] = wrapper
    plan_a = wrappers["cake"]._cake_vsa_sm90_plan
    plan_c = wrappers["cake_cute"]._cake_vsa_sm90_plan
    assert (plan_a.engine, plan_c.engine) == ("cuda", "cute")
    assert _stage_of(plan_a) == _stage_of(plan_c)
    rule = small_route(
        mask,
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        cluster_capacity=cluster_capacity(torch.cuda.current_device()),
    )
    assert (plan_c.small_kmax is None) == (rule is None)
    q, k, v = _inputs(h, mb, nb)
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    expected = wrappers["cake"].run(q, k, v, enable_pdl=False)
    actual = wrappers["cake_cute"].run(q, k, v, out=out, enable_pdl=False)
    assert actual.data_ptr() == out.data_ptr() and actual.shape == q.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="log-sum-exp"):
        wrappers["cake_cute"].run(q, k, v, return_lse=True)
    with pytest.raises(ValueError, match="PDL"):
        wrappers["cake_cute"].run(q, k, v, enable_pdl=True)


@requires_hopper
def test_unknown_engine_is_rejected():
    mask = torch.ones((1, 2, 3), dtype=torch.bool, device="cuda")
    rows, cols = _descriptors(mask)
    with pytest.raises(ValueError, match="engine"):
        CakeVsaSm90Plan("cuda", mask, rows, cols, 1, 1, 128, engine="ptx")


@requires_hopper
@requires_cute
def test_planner_unreachable_k3s_stage_matches_reference():
    """``small_k3s`` is not selectable through ``split_kmax``; launch it directly."""
    from flashinfer.cake_vsa_sm90 import (
        BLOCK,
        HEAD_DIM,
        LOG2E,
        SMALL_ITEM_ELEMS,
        SMALL_STATS_FLOATS,
        plan_small,
    )

    h, mb, nb, capacity = 2, 4, 8, 6
    mask = _random_mask(h, mb, nb, capacity, seed=29, ragged=True, device="cuda")
    plan = plan_small(mask, kmax=3, split=True, cluster=0)
    assert plan["split"] and plan["num_items"] > plan["num_tiles"]
    q, k, v = _inputs(h, mb, nb)
    out = torch.empty_like(q)
    stage = load_stage("small_k3s")
    bindings = {
        "Q": q,
        "K": k,
        "Vt": v,
        "O": out,
        "plan": plan["plan"].contiguous().to("cuda"),
        "seqlen_q": mb * BLOCK,
        "seqlen_k": nb * BLOCK,
        "scale_log2": HEAD_DIM**-0.5 * LOG2E,
        "Wo": torch.empty(
            (plan["num_items"] * SMALL_ITEM_ELEMS,), dtype=torch.float32, device="cuda"
        ),
        "Ws": torch.empty(
            (plan["num_items"] * SMALL_STATS_FLOATS,),
            dtype=torch.float32,
            device="cuda",
        ),
        "Wc": torch.zeros((plan["num_tiles"],), dtype=torch.int32, device="cuda").view(
            torch.uint32
        ),
    }
    stage.run(bindings, (int(plan["num_items"]), 1, 1))
    torch.cuda.synchronize()
    reference = _reference(q, k, v, mask, HEAD_DIM**-0.5)
    torch.testing.assert_close(out.float(), reference, **TOL)
    assert float((out.float() - reference).abs().max()) <= MAX_ABS
    assert int(bindings["Wc"].view(torch.int32).abs().sum()) == 0
