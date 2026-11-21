import math
import pytest
from ..conftest import require_b12x

import torch
from b12x.attention import sparse_mla as smla
from b12x.preparation import PreparationSession, PreparedCall

dev = torch.device("cuda", 0)
D, PAGE, WIDTH = 512, 256, 2051
SCALE = 256**-0.5
_keep = []


def caps(heads, rows, mode, lse_scale="natural"):
    return smla.Caps(
        device=dev,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=WIDTH,
        softmax_scale=SCALE,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=D,
        v_head_dim=D,
        mode=mode,
        max_batch=rows,
        max_chunks_per_row=(WIDTH + 63) // 64,
        page_size=PAGE,
        return_lse=True,
        lse_scale=lse_scale,
        model_type=int(smla.ModelType.GLM_NEXT),
    )


def prepare(plan, prep):
    req = plan.request(
        name=f"repro.{id(plan):x}", prepare_call=prep, benchmark_call=prep
    )
    sess = PreparationSession(device=dev, autotune=False)
    _keep.append((sess, sess.prepare((req,))))


def write_cache(kv_c, record_bytes):
    n = kv_c.shape[0]
    kv = torch.zeros(
        (n + PAGE - 1) // PAGE + 1, PAGE, record_bytes, dtype=torch.uint8, device=dev
    )
    slots = torch.arange(
        PAGE, PAGE + n, dtype=torch.int64, device=dev
    )  # page 0 stays empty
    plan = smla.plan_cache_writer(kv_c, kv, slots)
    prepare(
        plan,
        lambda st: PreparedCall(
            run=lambda: st.run(kv_c, st.kv_cache, slots), produce=lambda: None
        ),
    )
    smla.concat_and_cache_glm_next_mla_fp8(kv_c, kv, slots, plan=plan)
    return kv


def change_replay_inputs(q, sel, lens, step, tokens):
    q.mul_(0.5)
    sel[:, :tokens] = (
        PAGE
        + (torch.arange(tokens, device=dev, dtype=torch.int32) + step * 37) % tokens
    )
    lengths = torch.tensor(
        [0, 1, 63, 129, 511, 1024, 1537, tokens], device=dev, dtype=torch.int32
    )
    lens.copy_(lengths.roll(step))


def attend(mode, q, kv, sel, lens, lse_scale="natural", graph=False):
    if graph:
        q, sel, lens = q.clone(), sel.clone(), lens.clone()
    rows, heads = q.shape[:2]
    plan = smla.plan(caps(heads, rows, mode, lse_scale))
    cache_lens = torch.full((rows,), kv.shape[0] * PAGE, dtype=torch.int32, device=dev)

    def prep(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=dev)
        b = state.bind(
            scratch=scratch,
            q=q,
            selected_indices=sel,
            cache_seqlens_int32=cache_lens,
            nsa_cache_seqlens_int32=lens,
            kv_cache=kv,
        )
        state.prime(b, kv_cache=kv)
        return PreparedCall(
            run=lambda: state.run(b, kv_cache=kv),
            produce=lambda: None,
            owners=(scratch,),
        )

    prepare(plan, prep)
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=dev)
    binding = smla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=kv,
        selected_indices=sel,
        cache_lengths=cache_lens,
        selected_lengths=lens,
    )
    out, lse = smla.run(binding)
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            out, lse = smla.run(binding)
        for step in range(1, 4):
            change_replay_inputs(q, sel, lens, step, 2048)
            capture.replay()
    return out.float().clone(), lse.float().clone()


def merge_natural(outs, lses):
    # Every shard of an empty row has LSE=-inf and contributes zero output.
    w = torch.softmax(torch.stack(lses), dim=0).nan_to_num(0)
    return (w.unsqueeze(-1) * torch.stack(outs)).sum(0)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("heads", [8, 24, 32, 40, 64])
@pytest.mark.parametrize("graph", [False, True])
def test_prepared_extend_lse_units(heads, graph):
    require_b12x()
    torch.manual_seed(7)
    rows, tokens = 8, 2048
    latent = (torch.randn(tokens, D, device=dev) * 0.5).to(torch.bfloat16)
    cache = write_cache(latent, int(caps(heads, rows, "extend").cache_record_bytes))
    q = (torch.randn(rows, heads, D, device=dev) * 2).to(torch.bfloat16)
    sel = torch.full((rows, WIDTH), -1, device=dev, dtype=torch.int32)
    sel[:, :tokens] = PAGE + torch.arange(tokens, device=dev, dtype=torch.int32)
    lens = torch.tensor(
        [0, 1, 63, 129, 511, 1024, 1537, tokens], device=dev, dtype=torch.int32
    )
    try:
        out2, lse2 = attend("extend", q, cache, sel, lens, "base2", graph)
        oute, lsee = attend("extend", q, cache, sel, lens, "natural", graph)
        if graph:
            for step in range(1, 4):
                change_replay_inputs(q, sel, lens, step, tokens)
        outd, lsed = attend("decode", q, cache, sel, lens, "natural")
        torch.testing.assert_close(oute, out2, atol=0, rtol=0)
        torch.testing.assert_close(lsee, lse2 * math.log(2), atol=0, rtol=0)
        assert torch.isneginf(lsee[lens == 0]).all()
        assert torch.isneginf(lse2[lens == 0]).all()
        torch.testing.assert_close(lsee, lsed, atol=1e-4, rtol=0)
        torch.testing.assert_close(oute, outd, atol=0.02, rtol=0.02)
    finally:
        _keep.clear()


@pytest.mark.parametrize("shards", [2, 4])
def test_prepared_extend_natural_shard_merge(shards):
    require_b12x()
    torch.manual_seed(7)
    rows, heads, tokens = 8, 8, 2048
    latent = (torch.randn(tokens, D, device=dev) * 0.5).to(torch.bfloat16)
    cache = write_cache(latent, int(caps(heads, rows, "extend").cache_record_bytes))
    q = (torch.randn(rows, heads, D, device=dev) * 2).to(torch.bfloat16)
    sel = torch.full((rows, WIDTH), -1, device=dev, dtype=torch.int32)
    sel[:, :tokens] = PAGE + torch.arange(tokens, device=dev, dtype=torch.int32)
    lens = torch.tensor(
        [0, 1, 63, 129, 511, 1024, 1537, tokens], device=dev, dtype=torch.int32
    )
    outs, lses = [], []
    try:
        reference, full_lse = attend("extend", q, cache, sel, lens)
        for rank in range(shards):
            shard_sel = torch.full_like(sel, -1)
            shard_sel[:, : tokens // shards] = sel[:, rank:tokens:shards]
            shard_lens = ((lens - rank + shards - 1) // shards).clamp_min(0)
            out, lse = attend("extend", q, cache, shard_sel, shard_lens)
            outs.append(out)
            lses.append(lse)
        merged = merge_natural(outs, lses)
        assert rel(merged, reference) < 0.005
        torch.testing.assert_close(
            torch.logsumexp(torch.stack(lses), dim=0), full_lse, atol=0.01, rtol=0
        )
        assert torch.isneginf(full_lse[lens == 0]).all()
        assert torch.count_nonzero(merged[lens == 0]) == 0
    finally:
        _keep.clear()


@pytest.mark.parametrize("lse_scale", ["base2", "natural"])
def test_prepared_extend_rejects_lse_unit_mismatch(lse_scale):
    require_b12x()
    from b12x._lib.compile_plan import compile_only_launches
    from b12x.attention._shared.mla.prefill import run_unified_prefill

    rows, heads, tokens = 1, 8, 2048
    latent = torch.randn(tokens, D, device=dev).to(torch.bfloat16)
    cache = write_cache(latent, int(caps(heads, rows, "extend").cache_record_bytes))
    q = torch.randn(rows, heads, D, device=dev, dtype=torch.bfloat16)
    sel = torch.full((rows, WIDTH), -1, device=dev, dtype=torch.int32)
    sel[:, :tokens] = PAGE + torch.arange(tokens, device=dev, dtype=torch.int32)
    kwargs = dict(
        q=q,
        kv_cache=cache,
        topk_indices=sel,
        topk_length=torch.full((rows,), tokens, device=dev, dtype=torch.int32),
        page_block_size=PAGE,
        sm_scale=SCALE,
        model_type=int(smla.ModelType.GLM_NEXT),
    )
    try:
        with compile_only_launches():
            prepared = run_unified_prefill(**kwargs, lse_scale=lse_scale)
        run_unified_prefill(**kwargs, prepared=prepared, lse_scale=lse_scale)
        other_scale = "base2" if lse_scale == "natural" else "natural"
        with pytest.raises(ValueError, match="different lse_scale"):
            run_unified_prefill(**kwargs, prepared=prepared, lse_scale=other_scale)
    finally:
        _keep.clear()


@pytest.mark.cpu_only
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("lse_scale", ["base2", "natural"])
def test_prepared_extend_lse_units_preserve_buffer(monkeypatch, return_lse, lse_scale):
    from types import SimpleNamespace
    from b12x.attention.sparse_mla._preparation import _SparseMlaState
    from b12x.attention._shared.mla import prefill

    output = torch.ones(1, 8, 512)
    lse = torch.full((1, 8), 2.0)
    calls = []

    def run_prefill(**kwargs):
        calls.append(kwargs)
        return output, lse

    monkeypatch.setattr(prefill, "run_unified_prefill", run_prefill)
    c = SimpleNamespace(
        mode="extend",
        return_lse=return_lse,
        lse_scale=lse_scale,
        softmax_scale=1.0,
        page_size=256,
        model_type=0,
        scale_format=0,
        fp8_rope=False,
        latent_scale_per_token=False,
        cache_traits=None,
    )
    state = _SparseMlaState(
        c, None, SimpleNamespace(prefill_mg_enabled=True), 84, object()
    )
    binding = SimpleNamespace(
        q=output,
        selected_indices=None,
        nsa_cache_seqlens_int32=None,
        scratch=SimpleNamespace(output_buffer=None, final_lse=None),
    )
    actual, actual_lse = state.run(binding, kv_cache=None)
    assert actual is output
    assert actual_lse is lse
    assert calls[0]["lse_scale"] == (lse_scale if return_lse else "base2")
    torch.testing.assert_close(lse, torch.full_like(lse, 2.0))
