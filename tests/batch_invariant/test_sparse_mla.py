# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The sparse MLA classes ``mla.sparse.<bf16|e4m3>.d512[r64]`` behind
``trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k > 0, batch_invariant=True)``.

Bits only. The device-free tests (specification, exact CPU model, generated reduction code) run everywhere; the
device tests need SM100 / SM103 / SM107. ``FLASHINFER_BATCH_INVARIANT_FULL=1`` widens the grids (every configuration
of the space on every member, more shapes): the nightly tier.
"""

import os

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant import canon
from flashinfer.batch_invariant.fp_interface import exact
from flashinfer.batch_invariant.fp_interface.term import TermAlgebra
from flashinfer.mla.batch_invariant import api, codegen, reference, spec
from flashinfer.mla.batch_invariant import config as config_module
from flashinfer.mla.batch_invariant.config import (
    SparseMlaConfig,
    all_valid_configs,
    default_config,
)

FULL = os.environ.get("FLASHINFER_BATCH_INVARIANT_FULL") == "1"
E4M3 = torch.float8_e4m3fn
SCALE = 1.0 / 16
MEMBERS = [(64, "bf16"), (0, "bf16"), (64, "e4m3"), (0, "e4m3")]
MEMBER_IDS = [f"{'rope' if r else 'nope'}-{f}" for r, f in MEMBERS]

needs_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or api.compute_capability(torch.device("cuda")) not in api.MLA_ARCHS,
    reason="needs a CUDA device of SM100 / SM103 / SM107",
)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(bits(a), bits(b))


def cast(x: torch.Tensor, fmt: str) -> torch.Tensor:
    return x.to(torch.bfloat16 if fmt == "bf16" else E4M3)


class Pool:
    """A cache of ``slots`` rows and per-row ascending index lists drawn from it."""

    def __init__(
        self, rope, fmt, heads, lens, width, slots=4096, seed=0, device="cuda"
    ):
        g = torch.Generator().manual_seed(seed)
        self.rope, self.fmt, self.D, self.H = rope, fmt, 512 + rope, heads
        self.lens, self.width = list(lens), width
        self.kv = cast(torch.randn(slots, self.D, generator=g) * 0.7, fmt).to(device)
        self.q = cast(torch.randn(len(lens), heads, self.D, generator=g) * 0.7, fmt).to(
            device
        )
        idx = torch.full((len(lens), width), -1, dtype=torch.int32)
        for r, n in enumerate(self.lens):
            idx[r, :n] = torch.sort(torch.randperm(slots, generator=g)[:n]).values.int()
        self.idx = idx.to(device)
        self.lens_t = torch.tensor(self.lens, dtype=torch.int32, device=device)

    def run(self, rows=None, config=None, lens=True, q=None, idx=None):
        rows = list(range(len(self.lens))) if rows is None else list(rows)
        sel = torch.tensor(rows, device=self.q.device)
        q = (self.q if q is None else q)[sel].contiguous()
        ix = (self.idx if idx is None else idx)[sel].contiguous()
        return api.sparse_mla(
            q,
            self.kv,
            ix,
            self.lens_t[sel].contiguous() if lens else None,
            bmm1_scale=SCALE,
            config=config,
        )

    def raw(self, t):
        if self.fmt == "bf16":
            return t.float().cpu().tolist()
        return t.view(torch.uint8).cpu().tolist()


# ------------------------------------------------------------------------------------------------ device-free tier


def test_fingerprints_are_pinned():
    op = "trtllm_batch_decode_with_kv_cache_mla"
    assert bi.class_name(op) == "mla.sparse"
    got = {(r, f): bi.fingerprint(op, qk_rope_head_dim=r, dtype=f) for r, f in MEMBERS}
    assert got == {
        (64, "bf16"): "mla-sparse-bf16-d512r64-da94d889257d348d",
        (0, "bf16"): "mla-sparse-bf16-d512-576b087b91b54520",
        (64, "e4m3"): "mla-sparse-e4m3-d512r64-385bd0db8096a279",
        (0, "e4m3"): "mla-sparse-e4m3-d512-0d66be64e6c16464",
    }
    assert bi.fingerprint(op, qk_rope_head_dim=64, dtype=E4M3) == got[(64, "e4m3")]
    with pytest.raises(ValueError):
        bi.fingerprint(op, qk_rope_head_dim=32, dtype="bf16")


def test_key_reductions_build_the_canonical_tree_under_every_placement():
    alg = TermAlgebra()
    spec.verify_key_plans(alg)
    red = spec.rowsum_reduction(lambda fp, k: fp.op("p", k))
    # position k = 8 b + 2 c + e: e first, then b, then c
    t = red.canonical(alg)
    p = [alg.op("p", k) for k in range(spec.TILE_KEYS)]

    def tree(vals):
        while len(vals) > 1:
            vals = [alg.add(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
        return vals[0]

    per_c = [
        tree([alg.add(p[8 * b + 2 * c], p[8 * b + 2 * c + 1]) for b in range(16)])
        for c in range(4)
    ]
    assert t is tree(per_c)
    for split in spec.KEY_SPLITS:
        plan = canon.plan(red, spec.key_placement(split))
        assert plan.units["warp"] == split and plan.units["lane"] == 4


def test_generated_reduction_code_is_current():
    with open(codegen.OUT, encoding="utf-8") as f:
        assert f.read() == codegen.render(), (
            "regenerate with: python -m flashinfer.mla.batch_invariant.codegen"
        )


def test_e4m3_values_are_bf16_values():
    table = codegen.e4m3_table()
    assert len(table) == 256 and table[0x38] == 0x3F80 and table[0x80] == 0x8000
    assert table[0x01] == 0x3B00 and table[0x7E] == 0x43E0
    assert table[0x7F] == table[0xFF] == exact.CANONICAL_NAN_16


def test_configuration_space_and_dispatcher():
    space = all_valid_configs()
    assert len(space) == len(set(space)) == 80
    for rows in (1, 8, 64, 256, 8192):
        for heads in (1, 8, 16, 17, 64, 128):
            for sm in (148, 216):
                cfg = default_config(rows, heads, 2048, sm)
                assert cfg in space
                assert cfg.head_block == (
                    16 if heads <= 16 else 32 if heads <= 32 else 64
                )
                assert (cfg.mma_split == 4) == (cfg.head_block == 16)
    assert not SparseMlaConfig(32, 8, "fixup", 1, 128, 4).well_formed()
    assert not SparseMlaConfig(16, 8, "inkernel", 1, 128, 1).well_formed()
    assert not SparseMlaConfig(16, 8, "fixup", 0, 128, 1).well_formed()


def test_exact_mma_step():
    one = [1.0] + [0.0] * 15
    assert reference.mma16_exact(0.0, one, one) == 1.0
    # one rounding of the exact 17-term sum: 2^24 + 1 + 1 rounds to 2^24 + 2, not 2^24
    a = [1.0, 1.0] + [0.0] * 14
    assert reference.mma16_exact(float(2**24), a, a) == float(2**24 + 2)
    assert reference.mma16_exact(-1.0, one, one) == 0.0
    assert str(reference.mma16_exact(0.0, [-0.0] + [0.0] * 15, one)) == "0.0"


def test_cpu_model_is_the_canonical_term():
    g = torch.Generator().manual_seed(3)
    D, n = 512, 131

    def vals(k):
        return (
            (torch.randn(k, D, generator=g) * 0.7).to(torch.bfloat16).float().tolist()
        )

    q, keys = vals(1), vals(n)
    res = reference.reference_rows(q, [keys], SCALE, 0.75, cols=[0, 300])[0]
    fp = reference.ExactFP(reference.ex2_stand_in).bind(q, keys)
    c = spec.scale_log2(SCALE)
    for i, d in enumerate((0, 300)):
        red = spec.row_reduction(spec.SparseRow(n), D, d, c, -c)
        assert res.out[i] == spec.output_elem(
            fp, red.canonical(fp), spec.scale_out(0.75)
        )
    empty = reference.reference_rows(q, [[]], SCALE, cols=[0])[0]
    assert empty.out == [0.0] and str(empty.out[0]) == "0.0"


# ------------------------------------------------------------------------------------------------------ device tier


@needs_device
@pytest.mark.parametrize("rope,fmt", MEMBERS, ids=MEMBER_IDS)
def test_matches_cpu_model(rope, fmt):
    lens = [0, 1, 129, 300] if FULL else [0, 1, 129]
    heads = 18 if FULL else 2
    p = Pool(rope, fmt, heads, lens, 300, slots=600, seed=1)
    # a dead entry in the middle of a list: the position is absent, the tile is not
    p.idx[2, 5] = -1
    cols = [0, 1, 127, 128, 300, 511] if FULL else [0, 129, 511]
    q_rows, lists = [], []
    for r, n in enumerate(lens):
        for h in range(p.H):
            q_rows.append(reference.decode_cache_row(p.raw(p.q[r, h]), fmt))
            live = p.idx[r, :n].long()
            lists.append([p.raw(p.kv[j]) if j >= 0 else None for j in live.tolist()])
    ref = reference.reference_rows(
        q_rows, lists, SCALE, 1.0, fmt, ex2_device=api.ex2_device, cols=cols
    )
    want = torch.tensor([r.out for r in ref]).view(len(lens), p.H, len(cols))
    for cfg in (
        SparseMlaConfig(16, 8, "fixup", 1, 128, 4),
        SparseMlaConfig(16, 4, "inkernel", 0, 64, 1),
        SparseMlaConfig(64, 8, "fixup", 2, 128, 1),
        SparseMlaConfig(32, 4, "fixup", 4, 64, 1),
    ):
        got = p.run(config=cfg).float().cpu()[:, :, cols]
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), cfg


@needs_device
@pytest.mark.parametrize("rope", [64, 0])
def test_e4m3_member_is_the_bf16_member_on_the_converted_bytes(rope):
    # every finite e4m3 byte, through the kernel's conversion, against the table of the values the bytes denote
    D, H, n = 512 + rope, 3, 256 + 71
    g = torch.Generator().manual_seed(5)
    finite = torch.tensor(
        [b for b in range(256) if b not in (0x7F, 0xFF)], dtype=torch.uint8
    )
    pick = lambda shape: finite[torch.randint(len(finite), shape, generator=g)]  # noqa: E731
    kv8 = pick((512, D))
    kv8[:254, 0] = finite
    kv8[:254, D - 1] = finite.flip(0)
    q8 = pick((2, H, D))
    q8[0, 0, :254] = finite
    table = torch.tensor(codegen.e4m3_table(), dtype=torch.int32).to(torch.int16)
    conv = lambda t: table[t.long()].view(torch.bfloat16).cuda()  # noqa: E731
    idx = (
        torch.sort(torch.randperm(512, generator=g)[:n])
        .values.int()
        .repeat(2, 1)
        .cuda()
    )
    a = api.sparse_mla(
        q8.cuda().view(E4M3), kv8.cuda().view(E4M3), idx, bmm1_scale=SCALE
    )
    b = api.sparse_mla(conv(q8), conv(kv8), idx, bmm1_scale=SCALE)
    assert same(a, b)
    assert same(a, api.sparse_mla(q8.cuda(), kv8.cuda(), idx, bmm1_scale=SCALE))


@needs_device
@pytest.mark.parametrize("rope,fmt", MEMBERS, ids=MEMBER_IDS)
@pytest.mark.parametrize(
    "width", [2048, 2049, 2050, 2051, 2052] if FULL else [2048, 2051]
)
def test_batch_invariance(rope, fmt, width):
    lens = [width, 0, 1, 2047, width - 3, 128, 2048, 300, width, 5]
    lens = [min(n, width) for n in lens]
    p = Pool(rope, fmt, 16 if rope == 0 else 8, lens, width, seed=width)
    alone = [p.run([r]) for r in range(len(lens))]
    assert all(not torch.isnan(a.float()).any() for a in alone)
    assert bits(alone[1]).abs().max() == 0
    full = p.run()
    for r in range(len(lens)):
        assert same(full[r], alone[r][0]), r
    perm = [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]
    got = p.run(perm)
    for i, r in enumerate(perm):
        assert same(got[i], alone[r][0]), r
    sub = [3, 3, 0, 3]
    got = p.run(sub)
    for i, r in enumerate(sub):
        assert same(got[i], alone[r][0]), r
    # the live count comes from the lengths, from the -1 padding, or from both
    assert same(p.run(lens=False), full)


@needs_device
@pytest.mark.parametrize(
    "rope,fmt", [(64, "bf16"), (0, "e4m3")], ids=["rope-bf16", "nope-e4m3"]
)
def test_adversarial_neighbours_do_not_reach_a_row(rope, fmt):
    lens = [700, 300, 2048, 64, 1500, 129]
    p = Pool(rope, fmt, 8, lens, 2048, seed=11)
    clean = p.run()
    q = p.q.clone()
    bad_rows = [1, 3, 4]
    if fmt == "bf16":
        q[1] = float("nan")
        q[3, :, ::2] = float("inf")
        q[4] = -0.0
        tiny = torch.tensor(1e-40).to(torch.bfloat16)
        q[4, 0] = tiny
    else:
        q.view(torch.uint8)[1] = 0x7F
        q.view(torch.uint8)[3] = 0xFE
        q.view(torch.uint8)[4] = 0x81
    got = p.run(q=q)
    for r in range(len(lens)):
        if r not in bad_rows:
            assert same(got[r], clean[r]), r
    # entries past a row's live count are never read, whatever they hold
    idx = p.idx.clone()
    for r, n in enumerate(lens):
        idx[r, n:] = torch.tensor([2**31 - 1, -7, 0, 12345], dtype=torch.int32).repeat(
            600
        )[: 2048 - n]
    assert same(p.run(idx=idx), clean)


@needs_device
@pytest.mark.parametrize("rope,fmt", MEMBERS, ids=MEMBER_IDS)
def test_configuration_agreement(rope, fmt):
    lens = [0, 1, 130, 2051, 700, 2048]
    p = Pool(rope, fmt, 20, lens, 2051, seed=2)
    ref = p.run(config=SparseMlaConfig(16, 8, "fixup", 1, 128, 1))
    space = all_valid_configs()
    if not FULL and (rope, fmt) != (64, "e4m3"):
        # every value of every knob, and the direct (64 heads per CTA) gather with 4 and 8 warps
        space = [
            SparseMlaConfig(16, 4, "fixup", 2, 64, 4),
            SparseMlaConfig(16, 8, "inkernel", 0, 128, 4),
            SparseMlaConfig(16, 8, "fixup", 8, 128, 1),
            SparseMlaConfig(32, 4, "inkernel", 0, 64, 1),
            SparseMlaConfig(32, 8, "fixup", 4, 128, 1),
            SparseMlaConfig(64, 4, "fixup", 1, 64, 1),
            SparseMlaConfig(64, 8, "inkernel", 0, 128, 1),
        ]
    for cfg in space:
        assert same(p.run(config=cfg), ref), cfg
    assert same(p.run(), ref)


@needs_device
def test_sm_count_and_dispatcher_rules_do_not_change_bits(monkeypatch):
    p = Pool(64, "e4m3", 8, [2048] * 6 + [100, 0], 2048, seed=4)
    ref = p.run()
    picks = set()
    for sm in (1, 2, 4, 148, 216, 10**6):
        monkeypatch.setattr(api, "_sm_count", lambda index, sm=sm: sm)
        picks.add(default_config(8, 8, 2048, sm))
        assert same(p.run(), ref), sm
    assert {c.tiles_per_unit for c in picks} == {1, 4}
    # a prefill-sized call takes the in-kernel fold
    assert default_config(8192, 8, 2051, 148).fold == "inkernel"
    monkeypatch.setattr(config_module, "FIXUP_WORKSPACE_FLOATS", 1)
    assert default_config(8, 8, 2048, 148).fold == "inkernel"
    assert same(p.run(), ref)


@needs_device
@pytest.mark.parametrize("heads", [1, 8, 16, 17, 64, 128] if FULL else [8, 17, 64])
def test_head_subset_invariance(heads):
    p = Pool(64, "bf16", heads, [2048, 333], 2048, seed=6)
    full = p.run()
    for lo, hi in ((0, 1), (heads // 2, heads), (heads - 1, heads)):
        part = api.sparse_mla(
            p.q[:, lo:hi].contiguous(), p.kv, p.idx, p.lens_t, bmm1_scale=SCALE
        )
        assert same(part, full[:, lo:hi])


@needs_device
@pytest.mark.parametrize("rope,heads", [(64, 64), (0, 128), (64, 49)])
def test_e4m3_with_wide_head_blocks(rope, heads):
    # 64 heads per CTA: no staging area, the e4m3 rows are converted from direct loads
    lens = [2052, 1, 2049, 127, 0]
    p = Pool(rope, "e4m3", heads, lens, 2052, seed=7)
    ref = p.run(config=SparseMlaConfig(16, 8, "fixup", 1, 128, 4))
    assert same(p.run(), ref)
    for cfg in (
        SparseMlaConfig(64, 4, "fixup", 2, 128, 1),
        SparseMlaConfig(64, 8, "inkernel", 0, 64, 1),
        SparseMlaConfig(32, 8, "fixup", 1, 128, 1),
    ):
        assert same(p.run(config=cfg), ref), cfg
    for r in range(len(lens)):
        assert same(p.run([r])[0], ref[r]), r


@needs_device
@pytest.mark.parametrize("fold", ["fixup", "inkernel"])
def test_calls_on_two_streams_do_not_share_state(fold):
    cfg = SparseMlaConfig(16, 8, fold, 0 if fold == "inkernel" else 1, 128, 4)
    a = Pool(64, "e4m3", 16, [2048] * 48, 2048, seed=21)
    b = Pool(64, "e4m3", 16, [2048] * 48, 2048, seed=22)
    ref_a, ref_b = a.run(config=cfg), b.run(config=cfg)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(10):
        with torch.cuda.stream(s1):
            got_a = a.run(config=cfg)
        with torch.cuda.stream(s2):
            got_b = b.run(config=cfg)
        s1.synchronize()
        s2.synchronize()
        assert same(got_a, ref_a) and same(got_b, ref_b)


def _decode(p, q, idx, lens=None, seq=None, cum=None, **kw):
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=p.kv.view(-1, 1, 64, p.D),
        workspace_buffer=torch.empty(0, device=q.device),
        qk_nope_head_dim=64 if p.rope else 512,
        kv_lora_rank=512,
        qk_rope_head_dim=p.rope,
        block_tables=idx,
        seq_lens=seq,
        max_seq_len=0,
        sparse_mla_top_k=p.width,
        bmm1_scale=kw.pop("bmm1_scale", SCALE),
        sparse_mla_top_k_lens=lens,
        cum_seq_lens_q=cum,
        batch_invariant=True,
        **kw,
    )


@needs_device
@pytest.mark.parametrize("rope,fmt", MEMBERS, ids=MEMBER_IDS)
def test_prefill_equals_chunked_prefill_equals_decode(rope, fmt):
    # 2 requests of 12 and 5 tokens; every token has its own list (short rows early in the prompt)
    T = [12, 5]
    lens = [1, 2, 40, 128, 129, 500, 2048, 2049, 2050, 2051, 2051, 2051] + [
        3,
        700,
        2051,
        2051,
        9,
    ]
    W = 2051
    p = Pool(rope, fmt, 8, lens, W, seed=9)
    rows = sum(T)
    decode = torch.cat(
        [
            _decode(
                p,
                p.q[r : r + 1].view(1, 1, 8, p.D),
                p.idx[r : r + 1].view(1, 1, W),
                lens=p.lens_t[r : r + 1],
            )
            for r in range(rows)
        ]
    ).view(rows, 8, 512)
    cum = torch.tensor([0, T[0], rows], dtype=torch.int32, device="cuda")
    prefill = _decode(p, p.q, p.idx, lens=p.lens_t, cum=cum)
    assert same(prefill, decode)
    for cuts in ([0, 1, 4, 11, 12, 13, 17], [0, 7, 17]):
        pieces = []
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            c = torch.tensor([0, b - a], dtype=torch.int32, device="cuda")
            pieces.append(
                _decode(
                    p,
                    p.q[a:b].contiguous(),
                    p.idx[a:b].contiguous(),
                    lens=p.lens_t[a:b].contiguous(),
                    cum=c,
                )
            )
        assert same(torch.cat(pieces), decode), cuts
    # a fixed-shape multi-token call (MTP verify): 4 tokens per request
    mtp = _decode(
        p, p.q[:16].view(4, 4, 8, p.D), p.idx[:16].view(4, 4, W), lens=p.lens_t[:16]
    )
    assert same(mtp.view(16, 8, 512), decode[:16])
    assert same(api.sparse_mla(p.q, p.kv, p.idx, p.lens_t, bmm1_scale=SCALE), decode)


@needs_device
def test_seq_lens_give_the_live_count_when_no_lengths_are_passed():
    lens = [2048, 37, 900]
    p = Pool(64, "e4m3", 8, lens, 2048, seed=12)
    ref = p.run()
    seq = torch.tensor([40000, 37, 900], dtype=torch.int32, device="cuda")
    idx = p.idx.clone()
    idx[1, 37:] = 5
    got = _decode(p, p.q.view(3, 1, 8, p.D), idx.view(3, 1, 2048), seq=seq)
    assert same(got.view(3, 8, 512), ref)
    # the compact form takes each token's count from its request
    lens5 = [2048, 2048, 37, 900, 900]
    p5 = Pool(64, "e4m3", 8, lens5, 2048, seed=16)
    for cum_dtype in (torch.int32, torch.int64):
        cum = torch.tensor([0, 2, 3, 5], dtype=cum_dtype, device="cuda")
        got5 = _decode(p5, p5.q, p5.idx, seq=seq, cum=cum)
        assert same(got5, p5.run())
    got4 = _decode(
        p5,
        p5.q[:4].view(2, 2, 8, p.D),
        p5.idx[:4].view(2, 2, 2048),
        seq=torch.tensor([2048, 37], dtype=torch.int32, device="cuda"),
    )
    assert same(got4.view(4, 8, 512)[:3], p5.run([0, 1, 2]))
    out = torch.empty(3, 1, 8, 512, dtype=torch.bfloat16, device="cuda")
    assert (
        _decode(p, p.q.view(3, 1, 8, p.D), idx.view(3, 1, 2048), seq=seq, out=out)
        is out
    )
    assert same(out.view(3, 8, 512), ref)


@needs_device
def test_determinism_and_graph_replay():
    p = Pool(0, "e4m3", 16, [2051, 2048, 0, 77], 2051, seed=13)
    ref = p.run()
    for _ in range(5):
        assert same(p.run(), ref)
    q, idx, lens = p.q.clone(), p.idx.clone(), p.lens_t.clone()
    out = torch.empty_like(ref)
    api.sparse_mla(q, p.kv, idx, lens, bmm1_scale=SCALE, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.sparse_mla(q, p.kv, idx, lens, bmm1_scale=SCALE, out=out)
    out.zero_()
    graph.replay()
    assert same(out, ref)
    # new inputs in place: the replay equals an eager call on them
    order = [3, 0, 1, 2]
    q.copy_(p.q[order])
    idx.copy_(p.idx[order])
    lens.copy_(p.lens_t[order])
    graph.replay()
    assert same(out, ref[order])
    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        got = p.run()
    other.synchronize()
    assert same(got, ref)


@needs_device
def test_output_scale_is_one_multiply():
    p = Pool(64, "bf16", 2, [150], 150, slots=300, seed=14)
    cols = [0, 77, 511]
    q_rows = [reference.decode_cache_row(p.raw(p.q[0, h]), "bf16") for h in range(2)]
    lists = [p.raw(p.kv[p.idx[0, :150].long()])] * 2
    for s2 in (1.0, 0.37):
        ref = reference.reference_rows(
            q_rows, lists, SCALE, s2, ex2_device=api.ex2_device, cols=cols
        )
        got = api.sparse_mla(p.q, p.kv, p.idx, bmm1_scale=SCALE, bmm2_scale=s2)
        want = torch.tensor([r.out for r in ref])
        assert torch.equal(
            got.float().cpu()[0][:, cols].view(torch.int32), want.view(torch.int32)
        )


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    err = bi.BatchInvariantUnsupportedError
    p = Pool(64, "bf16", 8, [100, 100], 128, seed=15)
    q4, idx3 = p.q.view(2, 1, 8, 576), p.idx.view(2, 1, 128)
    ok = _decode(p, q4, idx3, lens=p.lens_t)
    assert same(ok.view(2, 8, 512), p.run())
    for kw in (
        dict(backend="trtllm-gen"),
        dict(backend="prims-ts"),
        dict(return_lse=True),
        dict(sinks=[torch.zeros(8, device="cuda")]),
        dict(enable_dcp=True),
        dict(enable_pdl=True),
        dict(bmm2_scale=torch.ones(1, device="cuda")),
        dict(skip_softmax_threshold_scale_factor=1.0),
        dict(cp_world=2),
        dict(cp_rank=1),
        dict(causal_seqlens_kv_global=p.lens_t),
        dict(multi_ctas_kv_counter_buffer=torch.zeros(4, device="cuda")),
        dict(cute_dsl_impl="modular"),
        dict(uses_shared_paged_kv_idx=False),
        dict(bmm1_scale=0.0),
        dict(bmm1_scale=-SCALE),
        dict(bmm1_scale=float("inf")),
        dict(bmm2_scale=float("nan")),
    ):
        with pytest.raises(err):
            _decode(p, q4, idx3, lens=p.lens_t, **kw)
    bad_cum = torch.tensor([0, 1], dtype=torch.int32)  # on the host
    with pytest.raises(err):
        _decode(p, p.q, p.idx, seq=p.lens_t[:1].contiguous(), cum=bad_cum)
    with pytest.raises(err):  # dense MLA has no class
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q4,
            kv_cache=p.kv.view(-1, 1, 64, 576),
            workspace_buffer=torch.empty(0),
            qk_nope_head_dim=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=idx3,
            seq_lens=p.lens_t,
            max_seq_len=128,
            batch_invariant=True,
        )
    with pytest.raises(err):  # one dtype for the query and the cache
        api.sparse_mla(p.q, p.kv.to(E4M3), p.idx, bmm1_scale=SCALE)
    with pytest.raises(err):
        api.sparse_mla(p.q.float(), p.kv.float(), p.idx, bmm1_scale=SCALE)
    with pytest.raises(err):
        api.sparse_mla(p.q[:, :, :500].contiguous(), p.kv, p.idx, bmm1_scale=SCALE)
    with pytest.raises(err):
        api.sparse_mla(p.q, p.kv, p.idx.long(), bmm1_scale=SCALE)
    with pytest.raises(err):
        api.sparse_mla(p.q, p.kv, p.idx[:, ::2], bmm1_scale=SCALE)
    with pytest.raises(err):
        api.sparse_mla(p.q.cpu(), p.kv.cpu(), p.idx.cpu(), bmm1_scale=SCALE)
    with pytest.raises(err):
        api.sparse_mla(
            p.q,
            p.kv,
            p.idx,
            bmm1_scale=SCALE,
            config=SparseMlaConfig(32, 8, "fixup", 1, 128, 4),
        )
    # the flag off: the regular backends, untouched
    assert (
        "batch_invariant"
        in flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla.__doc__
    )
