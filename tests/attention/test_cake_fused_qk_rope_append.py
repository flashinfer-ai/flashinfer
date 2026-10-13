"""Tests for the Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16) kernel.

Independent torch reference (per-head RMSNorm in f32, NeoX rotation from the cos/sin table, NHD paged
append, last-page clearing).  Tolerances follow the Cake BF16 standard: Q/K ``atol = rtol = 1e-2``,
V bit-exact, every cache / output element outside the appended rows and the cleared range untouched.
"""

import pytest
import torch

from flashinfer.cake_fused_qk_rope_append import (
    cake_fused_qk_rmsnorm_rope_append_paged_kv_cache,
)
from flashinfer.utils import get_compute_capability

HEAD_DIM = 128
POISON = 1234.0


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda:0"))
    if cc not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip(
            f"cake_fused_qk_rope_append has no cubin for compute capability {cc}"
        )


def rotary_cos_sin_table(
    max_positions: int, device, base: float = 10000.0
) -> torch.Tensor:
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    pos = torch.arange(max_positions, dtype=torch.float64, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)


def reference(
    qkv,
    cos_sin,
    seq_lens,
    q_indptr,
    page_indices,
    key_cache,
    value_cache,
    *,
    num_q_heads,
    num_kv_heads,
    qk_norm_policy,
    q_norm_weight,
    k_norm_weight,
    eps=1e-6,
    out_q=None,
    out_k=None,
    out_v=None,
    clear_last_page=True,
):
    T = qkv.shape[0]
    hq, hkv = num_q_heads, num_kv_heads
    x = qkv.float()
    q = x[:, : hq * HEAD_DIM].reshape(T, hq, HEAD_DIM)
    k = x[:, hq * HEAD_DIM : (hq + hkv) * HEAD_DIM].reshape(T, hkv, HEAD_DIM)
    v = qkv[:, (hq + hkv) * HEAD_DIM :].reshape(
        T, hkv, HEAD_DIM
    )  # bf16, copied bit-exactly
    B = seq_lens.shape[0]
    qi = q_indptr.tolist()
    sl = seq_lens.tolist()
    page_size = key_cache.shape[1]
    if out_q is None:
        out_q = torch.empty((T, hq, HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)

    def rms(t, w):
        var = (t * t).mean(dim=-1, keepdim=True)
        return t * torch.rsqrt(var + eps) * w

    def rope(t, cs):
        half = HEAD_DIM // 2
        c = cs[:, None, :half]
        s = cs[:, None, half:]
        x1, x2 = t[..., :half], t[..., half:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    if clear_last_page:
        for b in range(B):
            if sl[b] <= 0:
                continue
            last = sl[b] - 1
            page = int(page_indices[b, last // page_size])
            first_unused = last % page_size + 1
            key_cache[page, first_unused:] = 0
            value_cache[page, first_unused:] = 0
    for b in range(B):
        rows = list(range(qi[b], qi[b + 1]))
        if not rows:
            continue
        positions = torch.tensor(
            [r + sl[b] - qi[b + 1] for r in rows], device=qkv.device
        )
        valid = positions >= 0
        rows_t = torch.tensor(rows, device=qkv.device)[valid]
        positions = positions[valid]
        if rows_t.numel() == 0:
            continue
        cs = cos_sin[positions]
        qb, kb = q[rows_t], k[rows_t]
        if qk_norm_policy == 2:
            qb, kb = rms(qb, q_norm_weight), rms(kb, k_norm_weight)
        qb, kb = rope(qb, cs), rope(kb, cs)
        if qk_norm_policy == 1:
            qb, kb = rms(qb, q_norm_weight), rms(kb, k_norm_weight)
        out_q[rows_t] = qb.to(torch.bfloat16)
        kb16 = kb.to(torch.bfloat16)
        vb = v[rows_t]
        if out_k is not None:
            out_k[rows_t] = kb16
        if out_v is not None:
            out_v[rows_t] = vb
        if out_k is None or out_v is None:
            pages = page_indices[b][positions // page_size].long()
            slots = positions % page_size
            if out_k is None:
                key_cache[pages, slots] = kb16
            if out_v is None:
                value_cache[pages, slots] = vb
    return out_q


def make_case(
    *,
    q_lens,
    ctx_lens,
    num_q_heads,
    num_kv_heads,
    page_size,
    policy,
    seed,
    device="cuda",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = len(q_lens)
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens, strict=True)]
    pages_per_req = [(s + page_size - 1) // page_size for s in seq_lens]
    max_pages = max(max(pages_per_req), 1)
    total_pages = sum(pages_per_req) + 3
    perm = torch.randperm(total_pages, generator=g)
    page_indices = torch.zeros((B, max_pages), dtype=torch.int32)
    cursor = 0
    for b in range(B):
        n = pages_per_req[b]
        page_indices[b, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
    q_indptr = torch.zeros(B + 1, dtype=torch.int32)
    q_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    T = int(q_indptr[-1])
    width = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    qkv = (
        torch.randn((T, width), generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    cos_sin = rotary_cos_sin_table(max(seq_lens) + 1, device)
    q_w = (
        (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16).float().to(device)
    )
    k_w = (
        (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16).float().to(device)
    )
    key_cache = torch.full(
        (total_pages, page_size, num_kv_heads, HEAD_DIM),
        POISON,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.full_like(key_cache, POISON)
    out_q = torch.full(
        (T, num_q_heads, HEAD_DIM), POISON, dtype=torch.bfloat16, device=device
    )
    return dict(
        qkv=qkv,
        cos_sin=cos_sin,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr.to(device),
        page_indices=page_indices.to(device),
        key_cache=key_cache,
        value_cache=value_cache,
        out_q=out_q,
        q_norm_weight=q_w,
        k_norm_weight=k_w,
        policy=policy,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def run_and_compare(case, *, caller_owned_kv=False, clear=True):
    T = case["qkv"].shape[0]
    hkv = case["num_kv_heads"]
    dev = case["qkv"].device
    kw = dict(
        num_q_heads=case["num_q_heads"],
        num_kv_heads=hkv,
        qk_norm_policy=case["policy"],
        q_norm_weight=case["q_norm_weight"] if case["policy"] else None,
        k_norm_weight=case["k_norm_weight"] if case["policy"] else None,
    )
    ok = ov = rok = rov = None
    if caller_owned_kv:
        ok = torch.full((T, hkv, HEAD_DIM), POISON, dtype=torch.bfloat16, device=dev)
        ov = torch.full_like(ok, POISON)
        rok, rov = ok.clone(), ov.clone()
    kc, vc, oq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    ret = cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        kc,
        vc,
        out_q=oq,
        out_k=ok,
        out_v=ov,
        clear_unused_last_page_rows=clear,
        **kw,
    )
    assert ret is oq
    torch.cuda.synchronize()
    rkc, rvc, roq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    reference(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        rkc,
        rvc,
        num_q_heads=case["num_q_heads"],
        num_kv_heads=hkv,
        qk_norm_policy=case["policy"],
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
        out_q=roq,
        out_k=rok,
        out_v=rov,
        clear_last_page=clear,
    )
    torch.testing.assert_close(oq.float(), roq.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(kc.float(), rkc.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(vc.view(torch.int16), rvc.view(torch.int16)), (
        "V must be copied bit-exactly"
    )
    if caller_owned_kv:
        torch.testing.assert_close(ok.float(), rok.float(), atol=1e-2, rtol=1e-2)
        assert torch.equal(ov.view(torch.int16), rov.view(torch.int16))
    # untouched regions: everything the reference left at POISON must still be POISON
    for mine, ref in ((kc, rkc), (vc, rvc), (oq, roq)):
        mask = ref == POISON
        assert torch.equal(mine[mask], ref[mask]), (
            "kernel wrote outside the appended / cleared range"
        )


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("policy", [0, 1, 2])
@pytest.mark.parametrize("page_size", [16, 64])
def test_decode_batch(heads, policy, page_size):
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 32,
        ctx_lens=[2048 + 7 * i for i in range(32)],
        num_q_heads=heads[0],
        num_kv_heads=heads[1],
        page_size=page_size,
        policy=policy,
        seed=892,
    )
    run_and_compare(case)


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("policy", [1, 2])
def test_prefill_chunks_cross_pages(heads, policy):
    _skip_unless_supported()
    case = make_case(
        q_lens=[16, 64, 128, 3],
        ctx_lens=[2047, 0, 2000, 63],
        num_q_heads=heads[0],
        num_kv_heads=heads[1],
        page_size=64,
        policy=policy,
        seed=893,
    )
    run_and_compare(case)


def test_ragged_with_empty_requests_and_caller_owned_kv():
    _skip_unless_supported()
    case = make_case(
        q_lens=[5, 0, 1, 33, 0, 2],
        ctx_lens=[100, 64, 4095, 0, 1, 127],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=894,
    )
    run_and_compare(case)
    run_and_compare(case, caller_owned_kv=True)


def test_no_clear_leaves_last_page_tail():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 4,
        ctx_lens=[10, 70, 128, 5],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=895,
    )
    run_and_compare(case, clear=False)


def test_large_batch_lookup_fallback():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 300,
        ctx_lens=[64 + (i % 200) for i in range(300)],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=896,
    )
    run_and_compare(case)


def test_cuda_graph_replay_matches_eager():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 8,
        ctx_lens=[2048] * 8,
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=897,
    )
    kw = dict(
        num_q_heads=8,
        num_kv_heads=1,
        qk_norm_policy=2,
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
    )
    kc, vc, oq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        kc,
        vc,
        out_q=oq,
        **kw,
    )
    torch.cuda.synchronize()
    gkc, gvc, goq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(
            2
        ):  # warm the JIT module / allocator on the side stream before capture
            cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
                case["qkv"],
                case["cos_sin"],
                case["seq_lens"],
                case["q_indptr"],
                case["page_indices"],
                gkc,
                gvc,
                out_q=goq,
                **kw,
            )
        gkc.fill_(POISON)
        gvc.fill_(POISON)
        goq.fill_(POISON)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
                case["qkv"],
                case["cos_sin"],
                case["seq_lens"],
                case["q_indptr"],
                case["page_indices"],
                gkc,
                gvc,
                out_q=goq,
                **kw,
            )
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(goq.view(torch.int16), oq.view(torch.int16))
    assert torch.equal(gkc.view(torch.int16), kc.view(torch.int16))
    assert torch.equal(gvc.view(torch.int16), vc.view(torch.int16))


def test_rejects_bad_arguments():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1],
        ctx_lens=[5],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=0,
        seed=1,
    )
    with pytest.raises(ValueError):
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            case["key_cache"],
            case["value_cache"],
            num_q_heads=4,
            num_kv_heads=1,
        )
    with pytest.raises(ValueError):
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            case["key_cache"],
            case["value_cache"],
            num_q_heads=8,
            num_kv_heads=1,
            qk_norm_policy=2,
        )
    with pytest.raises(ValueError):
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            case["key_cache"],
            case["value_cache"],
            num_q_heads=8,
            num_kv_heads=1,
            out_q=torch.empty((1, 8, 64), dtype=torch.bfloat16, device="cuda"),
        )


# ----------------------------------------------------------------------------------------------
# Stage policy (CPU): the host mirrors the Cake v4 policy and must only select shipped builds.

_POLICY_TABLE = {
    # (arch, Hq, Hkv, num_rows, num_requests, page_size, num_sms) -> stage
    ("sm_90a", 8, 1, 32, 32, 64, 132): "hq8_hkv1_vsub3_d",
    ("sm_90a", 8, 1, 128, 1, 64, 132): "hq8_hkv1_vsub3_d",
    ("sm_90a", 8, 1, 512, 512, 64, 132): "hq8_hkv1_vsub3_d",
    ("sm_90a", 64, 8, 32, 32, 64, 132): "hq64_hkv8_w20_vsub18_d",
    ("sm_90a", 64, 8, 128, 8, 64, 132): "hq64_hkv8",
    ("sm_90a", 64, 8, 128, 128, 64, 132): "hq64_hkv8",
    ("sm_90a", 64, 8, 128, 1, 64, 132): "hq64_hkv8_w20_vsub18_d",
    ("sm_90a", 64, 8, 512, 512, 64, 132): "hq64_hkv8_d",
    ("sm_100a", 8, 1, 32, 32, 64, 148): "hq8_hkv1",
    ("sm_100a", 8, 1, 128, 128, 64, 148): "hq8_hkv1_vsub3_d",
    ("sm_100a", 8, 1, 256, 32, 64, 148): "hq8_hkv1",
    ("sm_100a", 8, 1, 512, 512, 64, 148): "hq8_hkv1_vsub3_d",
    ("sm_100a", 64, 8, 32, 32, 64, 148): "hq64_hkv8_w20_vsub18_d",
    ("sm_100a", 64, 8, 128, 8, 64, 148): "hq64_hkv8_w20_vsub18",
    ("sm_100a", 64, 8, 128, 128, 64, 148): "hq64_hkv8_d_u2",
    ("sm_100a", 64, 8, 128, 1, 64, 148): "hq64_hkv8_w20_vsub18",
    ("sm_103a", 64, 8, 128, 128, 64, 148): "hq64_hkv8_d_u2",
}


@pytest.mark.parametrize("key,expected", sorted(_POLICY_TABLE.items()))
def test_stage_policy_table(key, expected):
    from flashinfer.jit.cake_fused_qk_rope_append import stage_for

    arch, hq, hkv, rows, reqs, page, sms = key
    assert (
        stage_for(
            hq,
            hkv,
            arch=arch,
            num_rows=rows,
            num_requests=reqs,
            page_size=page,
            num_sms=sms,
        )
        == expected
    )


@pytest.mark.parametrize("arch", ["sm_90a", "sm_100a", "sm_103a"])
def test_stage_policy_only_selects_shipped_builds(arch):
    from flashinfer.jit.cake_fused_qk_rope_append import stage_for, stages_for

    shipped = set(stages_for(arch))
    for hq, hkv in ((8, 1), (64, 8)):
        for page in (16, 32, 64, 128):
            for reqs in (1, 2, 8, 32, 64, 128, 256, 257, 512, 1024):
                for rows in sorted({reqs, 1, 32, 128, 148, 256, 512, 4096}):
                    if rows < reqs:
                        continue
                    for sms in (132, 148):
                        stage = stage_for(
                            hq,
                            hkv,
                            arch=arch,
                            num_rows=rows,
                            num_requests=reqs,
                            page_size=page,
                            num_sms=sms,
                        )
                        assert stage in shipped, (
                            arch,
                            hq,
                            hkv,
                            rows,
                            reqs,
                            page,
                            sms,
                            stage,
                        )


def test_stage_for_without_shape_is_the_base_build():
    from flashinfer.jit.cake_fused_qk_rope_append import stage_for

    assert stage_for(8, 1) == "hq8_hkv1"
    assert stage_for(64, 8) == "hq64_hkv8"
    with pytest.raises(ValueError):
        stage_for(16, 2)
