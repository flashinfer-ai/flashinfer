# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``flashinfer.sampling.sampling_from_probs(..., batch_invariant=True)`` and ``sampling_from_logits(...,
batch_invariant=True)``: the draw class ``sample.icdf.v1`` (the inverse CDF read off the canonical sum tree of the row's
weights, at the fp32 uniform of the request-keyed stream ``stream.philox4x32-10.splitmix-key.v1``).

Device-free: the pinned fingerprints (the sandbox's, byte for byte), the symbolic prefix terms equal the exact model
element by element, the tree-order address maps, the stream's known-answer vectors (Random123, SplitMix64) and
conversions, the model's draw follows the weights, the degenerate rows, dispatcher membership over the grid and four SM
counts (never a heavy share), the error contract, the kernels' PTX (only the declared forms). On an SM100 / SM103 /
SM107 device with the CuTe DSL: tokens and validity bitwise against the exact model for probability and logits inputs,
bf16 / f16 / f32, across single-CTA / cluster / staged members and the dispatcher's picks; the index gather; the
generator-only call shape (the pair reserved as the operator's own path reserves it) and per-row / scalar / uint64 keys;
batch invariance the way the operator's kernel was probed (a row alone == the row at slot 5 of 64 == at slot 5 of 8; a
batch permutation moves no token) and key-not-slot; invalid rows, ``check_nan`` and ``return_valid``; the drawn
distribution against the weights; R4 (the drawn token's log-probability is ``log_probs`` of it); determinism,
CUDA-graph replay and PDL; ``batch_invariant=False`` is the operator's kernel, byte for byte; 3-D inputs, views, M = 0;
the stream kernel's words against the specification.
"""

import glob
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant.sampling import draw_reference as DR
from flashinfer.batch_invariant.sampling import draw_schedule as DS
from flashinfer.batch_invariant.sampling import draw_spec as D
from flashinfer.batch_invariant.sampling import schedule as C
from flashinfer.batch_invariant.sampling import stream as S
from flashinfer.batch_invariant.sampling import sterm as term
from flashinfer.batch_invariant.sampling.draw_api import (
    DRAW_ARCHS,
    draw,
    sampling_from_logits_batch_invariant,
    sampling_from_probs_batch_invariant,
)
from flashinfer.batch_invariant.sampling.schedule import device_sm_count

DT = [torch.bfloat16, torch.float16, torch.float32]
_DS = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
#: One block, a partial block (an odd vocabulary), 16 blocks, an odd long one, a long one.
NS = [64, 1000, 4096, 4097, 32768]
N_BIG = 152064
SM_COUNTS = [148, 152, 204, 224]  # B200 / B300 (SM100 / SM103), Rubin (SM107) parts
#: The dispatcher's row counts the device tests use (the fixture precompiles their picks).
MS = (1, 2, 4, 5, 6, 8, 11, 16, 37, 64, 512, 20000)
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), DRAW_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)
_PDL = True  # the public entries follow the device (PDL on SM >= 9.0); the generic entry is called the same way

# Random123's known-answer tests for philox4x32 with 10 rounds (kat_vectors): counter, key -> output.
PHILOX_KATS = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    (
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


# ------------------------------------------------------------------------------------------------------ helpers
def logits(M, N, dtype, seed=0, scale=2.0**-10):
    """Rows that make reduction-order differences visible (the row class's suite): magnitudes over ~40 binades with
    random signs, near-cancelling pairs, signed zeros, subnormals, outliers, scaled into a logits-like range; finite."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    exp_span = 20.0 if dtype != torch.float16 else 6.0
    e = (torch.rand(M, N, generator=g) * 2 - 1) * exp_span
    mant = torch.rand(M, N, generator=g) + 1.0
    sign = torch.where(torch.rand(M, N, generator=g) < 0.5, -1.0, 1.0)
    x = sign * mant * torch.exp2(e)
    pairs = torch.rand(M, N // 2, generator=g) < 0.15
    idx = torch.arange(0, N - 1, 2)
    x[:, idx + 1] = torch.where(
        pairs,
        -x[:, idx] * (1 + 1e-3 * torch.randn(M, N // 2, generator=g)),
        x[:, idx + 1],
    )
    r = torch.rand(M, N, generator=g)
    outlier = 2.0**12 if dtype != torch.float16 else 2.0**4
    x = torch.where(r < 0.01, x * outlier, x)
    x = torch.where((r >= 0.01) & (r < 0.02), torch.zeros_like(x), x)
    x = torch.where((r >= 0.02) & (r < 0.03), -torch.zeros_like(x), x)
    x = torch.where((r >= 0.03) & (r < 0.035), x * 2.0**-130, x)
    x = (x.to(dtype).float() * scale).to(dtype)
    assert torch.isfinite(x).all()
    return x.cuda()


def weights(M, N, dtype, seed=0):
    """Weight rows: random, the first half of row 0 zero, row 2 one-hot at the last element."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(M, N, generator=g).to(dtype)
    p[0, : N // 2] = 0
    if M > 2:
        p[2] = 0
        p[2, N - 1] = 1
    return p.cuda()


def keys(M, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(-(2**62), 2**62, (M,), generator=g).cuda(),
        torch.randint(0, 2**40, (M,), generator=g).cuda(),
    )


def same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())


def _configs(N, eb, k=6):
    """Members whose per-thread share is at most 512 elements (the draw's crossing pass unrolls per block; a 32-thread
    single-CTA member of a 1024-block row, never a dispatcher pick, compiles for twenty minutes): the first two, two
    cluster members, a staged member, a single-CTA member of >= 512 threads; the whole (bounded) space for bf16 at
    N = 1000."""
    cfgs = [c for c in C.all_valid_configs(N, eb) if c.elems_per_thread(N) <= 512]
    if eb == 2 and N == 1000:
        return cfgs
    picks = (
        cfgs[:2]
        + [c for c in cfgs if c.cluster > 1][:2]
        + [c for c in cfgs if c.stage == "smem"][:1]
        + [c for c in cfgs if c.threads_per_row >= 512 and c.cluster == 1][:1]
    )
    return list(dict.fromkeys(picks))[:k]


def probs_draw(p, s, o, **kw):
    """The generic entry on weights with the public entries' PDL choice."""
    return draw(p, s, o, enable_pdl=_PDL, **kw)


def logits_draw(x, s, o, **kw):
    return draw(x, s, o, from_logits=True, enable_pdl=_PDL, **kw)


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprints_are_pinned():
    """The receipts: the draw's fingerprints are the sandbox's pins byte for byte (the dataflow is the same); one
    class with two weight forms behind the two entries; the input dtype enters nothing (exact widenings)."""
    assert D.spec_fingerprint(300) == "sample-icdf-48d4cb539bae34fb"
    assert D.spec_fingerprint(1000) == "sample-icdf-22e55f7a33a8e40e"
    assert D.spec_fingerprint(
        300, from_logits=True, in_fmt="bf16"
    ) != D.spec_fingerprint(300)
    assert bi.class_name("sampling_from_probs") == "sample.icdf.v1"
    assert bi.class_name("sampling_from_logits") == "sample.icdf.v1"
    assert (
        bi.fingerprint("sampling_from_probs", N=300, dtype="bf16")
        == "sample-icdf-48d4cb539bae34fb"
    )
    assert bi.fingerprint("sampling_from_probs", N=300, dtype=torch.float32) == (
        bi.fingerprint("sampling_from_probs", N=300, dtype=torch.bfloat16)
    )
    assert bi.fingerprint("sampling_from_probs", N=1000, dtype="f16") == (
        "sample-icdf-22e55f7a33a8e40e"
    )
    fl = bi.fingerprint("sampling_from_logits", N=300, dtype="bf16")
    assert fl.startswith("sample-icdf-") and fl != D.spec_fingerprint(300)
    assert fl == D.spec_fingerprint(300, from_logits=True, in_fmt="bf16")
    assert D.spec_fingerprint(301) != D.spec_fingerprint(300)
    with pytest.raises(TypeError):
        bi.fingerprint("sampling_from_probs", N=300)
    with pytest.raises(ValueError):
        bi.fingerprint("sampling_from_probs", N=300, dtype="f64")
    assert S.STREAM_NAME == "stream.philox4x32-10.splitmix-key.v1"
    assert (
        S.stream_fingerprint().startswith("stream-")
        and len(S.stream_fingerprint()) == 23
    )


@pytest.mark.parametrize("N", [8, 40, 300, 1000, 4097])
def test_symbolic_prefix_equals_the_exact_model(N):
    """The spec's inclusive-prefix terms, evaluated exactly, are the exact model's bits element by element; the total is
    the row class's canonical sum tree."""
    torch.manual_seed(N)
    sp = D.DrawSpec(N)
    p = torch.rand(N).to(torch.float32) * 0.01
    p[::7] = 0.0
    pl = p.double().tolist()
    folds, present, totals = DR.slot_folds(p, N)
    outer = DR.outer_prefix(*DR.node_tables(totals), sp.Kp)
    probe = sorted(
        {0, 1, 7, 8, 255, 256, 257, 8 * 32, 8 * 32 + 9, 8 * 33, N // 2, N - 2, N - 1}
        & set(range(N))
    )
    for i in probe:
        ref = term.evaluate(sp.prefix_incl(0, i), lambda r, j: pl[j], {}, {})
        slot, e = divmod(i, 8)
        block, lane = divmod(slot, 32)
        assert ref == float(torch.add(outer[lane, block], folds[i])), (N, i)
    assert term.evaluate(sp.total(0), lambda r, j: pl[j], {}, {}) == DR.row_sum(p, N)


def test_address_maps_are_inverse():
    for N in (8, 300, 1000, 32768):
        Kp = C.padded_blocks(N)
        for i in (0, 1, 8, 255, 256, N - 1):
            if i < Kp * 256:  # only the tree's own elements have an address
                assert D.index_of_address(D.address_of_index(i, Kp), Kp) == i
    assert D.prefix_nodes(0, 0, 8) == [] and D.prefix_nodes(1, 0, 8) == [("lane", 0, 0)]
    assert D.prefix_nodes(0, 5, 8) == [("block", 0, 4), ("block", 2, 0)]


def test_stream_known_answers_and_conversions():
    """The stream specification: Random123's Philox4x32-10 vectors, SplitMix64's vectors, the key separates seeds and
    offsets, ``word(i)`` indexes as ``curand`` does, ``uniform32`` is ``curand_uniform``'s conversion (in (0, 1]),
    ``uniform64`` is exact and complement-symmetric, one block per draw on separate lanes."""
    for ctr, key, out in PHILOX_KATS:
        assert S.philox4x32_10(ctr, key) == out, (ctr, key)
    assert S.mix64(0) == 0xE220A8397B1DCDAF
    assert S.mix64(S.SPLITMIX_GOLDEN) == 0x6E789E6AA1B965F4
    assert S.mix64((2 * S.SPLITMIX_GOLDEN) & S.MASK64) == 0x06C45D188009454F
    ks = {
        S.request_key(s, o)
        for s in (0, 1, 42, 2**63 + 7)
        for o in (0, 1, 2, 3, 1000, 2**40)
    }
    assert len(ks) == 24
    a, b = S.words(42, 5, 0), S.words(42, 6, 0)
    assert all(x != y for x, y in zip(a, b, strict=True))
    for i in range(16):
        assert S.word(7, 11, i) == S.block(7, 11, i // 4)[i % 4]
    assert S.block(7, 11, 2**32) == S.philox4x32_10(
        (0, 1, 0, 0), S.key_words(S.request_key(7, 11))
    )
    from flashinfer.batch_invariant.fp_interface import exact as fp32

    for w in (
        0,
        1,
        2,
        0x7FFFFFFF,
        0x80000000,
        0xFFFFFFFE,
        0xFFFFFFFF,
        0x00FFFFFF,
        0x01000001,
    ):
        u = S.uniform32(w)
        assert 0.0 < u <= 1.0 and u == fp32.to_f32(u)
        assert u == fp32.add(fp32.mul(fp32.to_f32(float(w)), 2.0**-32), 2.0**-33)
    assert S.uniform32(0) == 2.0**-33
    assert S.uniform32(0xFFFFFFFF) == 1.0 and S.uniform32(0xFFFFFF7F) < 1.0
    rng = random.Random(0)
    for _ in range(300):
        w1, w2 = rng.getrandbits(32), rng.getrandbits(32)
        u = S.uniform64(w1, w2)
        z = ((w2 << 32) | w1) >> 12
        assert u == (2 * z + 1) / 2.0**53 and 0.0 < u < 1.0
        assert S.uniform64(w1 ^ 0xFFFFFFFF, w2 ^ 0xFFFFFFFF) == 1.0 - u
    w = S.words(3, 4, 9)
    u32, u64 = S.uniforms(3, 4, 9)
    assert u32 == S.uniform32(w[0]) and u64 == S.uniform64(w[1], w[2])


def test_model_draw_follows_the_weights_and_degenerate_rows():
    """The exact model: the drawn distribution over many keys follows ``p / S`` (a sanity bound, not a proof); the
    declared corners (an all-zero row, ``u = 1`` picks the mass, the smallest ``u`` skips zeros, NaN / inf rows)."""
    N = 64
    torch.manual_seed(2)
    p = torch.rand(N).to(torch.float32)
    p[5] = 3.0
    counts = torch.zeros(N)
    for k in range(2000):
        tok, ok, _, _ = DR.draw_row(p, N, S.uniforms(7, k, 0)[0])
        assert ok and 0 <= tok < N
        counts[tok] += 1
    freq, pn = counts / counts.sum(), p / p.sum()
    assert float((freq - pn).abs().max()) < 0.025 and int(counts.argmax()) == 5
    assert DR.draw_row(torch.zeros(10), 10, 0.5)[:2] == (-1, False)
    assert DR.draw_row(torch.tensor([0.0, 0.0, 1.0, 0.0]), 4, 1.0)[:2] == (2, True)
    assert DR.draw_row(torch.tensor([0.0, 0.0, 1.0, 0.0]), 4, 2.0**-33)[:2] == (2, True)
    assert DR.draw_row(torch.tensor([1.0, float("nan")]), 2, 0.3)[:2] == (-1, False)
    assert DR.draw_row(torch.tensor([1.0, float("inf")]), 2, 0.3)[:2] == (-1, False)


def test_dispatcher_picks_are_members_of_the_space():
    """R3 membership: every pick of the draw's closed-form dispatcher, over the grid, both element widths and the SM
    counts of the SM100 / SM103 / SM107 parts, is a member of the enumerated space at one row per CTA, never a heavy
    share (above 512 elements per thread: the crossing pass unrolls per element and such members compile for minutes),
    and never a register-resident share above ``REGS_SHARE_MAX`` where the space has a member of the pick's cluster at
    or below it (the kernel stages a longer one in shared memory: ``STAGE_ABOVE_ELEMS``, the same number)."""
    try:
        from flashinfer.batch_invariant.sampling.draw_kernel import STAGE_ABOVE_ELEMS

        assert STAGE_ABOVE_ELEMS == DS.REGS_SHARE_MAX
    except (
        ImportError
    ):  # no CuTe DSL in this environment: the kernel module is not importable
        pass
    for N in (8, 64, 1000, 4097, 32000, 32768, 128256, 152064, 65537, 262145):
        for eb in (2, 4):
            space = C.all_valid_configs(N, eb)
            members = set(space)
            for M in (1, 4, 37, 256, 4096):
                for sm in SM_COUNTS:
                    cfg = DS.draw_config(N, M, eb, sm)
                    assert cfg in members and cfg.rows_per_cta == 1, (
                        N,
                        eb,
                        M,
                        sm,
                        cfg.describe(),
                    )
                    assert cfg.elems_per_thread(N) <= DS.SHARE_MAX == 512, (
                        N,
                        eb,
                        M,
                        sm,
                        cfg.describe(),
                    )
                    if (
                        cfg.stage == "regs"
                        and cfg.elems_per_thread(N) > DS.REGS_SHARE_MAX
                    ):
                        light = [
                            c
                            for c in space
                            if c.cluster == cfg.cluster
                            and c.elems_per_thread(N) <= DS.REGS_SHARE_MAX
                        ]
                        assert not light, (
                            N,
                            eb,
                            M,
                            sm,
                            cfg.describe(),
                            light[0].describe(),
                        )
    with pytest.raises(ValueError, match="no configuration"):
        DS.draw_config(C.VOCAB_COVERED + 1, 64, 4, 148)


def test_unsupported_inputs_raise_before_any_kernel():
    x = torch.rand(4, 64, dtype=torch.bfloat16)
    s = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sample.icdf"):
        flashinfer.sampling.sampling_from_probs(
            x, seed=s, offset=s, batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sample.icdf"):
        flashinfer.sampling.sampling_from_logits(
            x, seed=s, offset=s, batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sample.icdf"):
        sampling_from_probs_batch_invariant(x.double(), seed=s, offset=s)
    with pytest.raises(ValueError, match="log-probability"):
        draw(x, s, s, return_logprob=True)


# The admitted PTX forms (the row kernels' suite): the `.rn` arithmetic incl. the mixed-precision forms, the exact
# conversions and selects, the two declared MUFU instructions, `max.f32`; the stream's integer forms are not fp-like.
PTX_FP_ALLOWED = re.compile(
    r"^(fma\.rn\.f32(\.bf16|\.f16)?|add\.rn\.f32(\.bf16|\.f16)?|sub\.rn\.f32(\.bf16|\.f16)?|mul\.rn\.f32|div\.rn\.f32"
    r"|cvt\.f32\.(bf16|f16)|cvt\.rn\.(bf16|f16)\.f32|cvt\.rn\.f32\.s32|cvt\.rn\.f32\.s64|cvt\.rn\.f32\.u32|cvt\.f32\.f16"
    r"|cvt\.rn\.f64\.u64|mul\.rn\.f64|cvt\.u32\.u64|cvt\.u64\.u32|cvt\.s64\.s32|cvt\.u64\.s32|cvt\.s32\.s64"
    r"|max\.(f32|bf16|f16)|min\.f32|abs\.f32|neg\.f32|mov\.(b32|f32)|selp\.f32|setp\.[a-z]+\.(f32|bf16|f16)"
    r"|ex2\.approx\.ftz\.f32|lg2\.approx\.ftz\.f32)$"
)
PTX_FP_LIKE = re.compile(
    r"^(fma|add|sub|mul|div|sqrt|rsqrt|rcp|ex2|lg2|cvt|max|min|abs|neg|selp|setp|fns|mad|sin|cos|tanh)\."
)
IR_CASES = [
    ("f32", 4096, (128, 1, 4), False, False, False, False),  # probabilities, single CTA
    (
        "bf16",
        152064,
        (128, 1, 8, "regs", 8),
        True,
        True,
        False,
        True,
    ),  # logits, cluster, per-row T, the token's logp
    (
        "bf16",
        32768,
        (64, 1, 8, "regs", 4),
        True,
        False,
        True,
        False,
    ),  # logits, cluster, index gather
]


@pytest.mark.parametrize(
    "dtype,N,cfg,from_logits,trow,idx,lp", IR_CASES, ids=lambda v: str(v)
)
def test_draw_ir_uses_only_declared_forms(
    tmp_path, dtype, N, cfg, from_logits, trow, idx, lp
):
    """Compile one kernel per code path with no GPU visible and check the PTX: `.rn` arithmetic only, `ex2.approx`
    iff logits and `lg2.approx` iff the token's log-probability, no division except the per-row temperature's (the draw
    never divides), the stream's `cvt.rn.f32.u32` and Philox multiplies, the cluster forms only in cluster members, no
    fast-math anywhere."""
    pytest.importorskip("cutlass")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        CUTE_DSL_KEEP_IR="1",
        CUTE_DSL_KEEP="ptx",
        FLASHINFER_CUTE_DSL_DISABLE_CACHE="1",
        CUTE_DSL_ARCH=os.environ.get("BI_TEST_ARCH", "sm_100a"),
    )
    code = (
        "from flashinfer.batch_invariant.sampling.schedule import LSEConfig\n"
        "from flashinfer.batch_invariant.sampling.draw_kernel import get_compiled_kernel\n"
        f"get_compiled_kernel({dtype!r}, {N}, LSEConfig{cfg}, {from_logits}, {trow}, {idx}, {lp}, False)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    mlir = glob.glob(str(tmp_path / "*.mlir"))
    assert mlir, "no MLIR dumped"
    ir = Path(mlir[0]).read_text()
    assert not re.search(r"arith\.(addf|mulf|divf|subf|fmaf)\b", ir), (
        "arith float op present"
    )
    assert not re.search(r"\bmath\.", ir), "math-dialect op present"
    assert (
        "fastmath<" not in ir
        and "rsqrt" not in ir
        and "div.approx" not in ir
        and "rcp.approx" not in ir
    )
    ptxs = glob.glob(str(tmp_path / "*.ptx"))
    assert ptxs, "no PTX dumped"
    ptx = Path(ptxs[0]).read_text()
    ops = set()
    for line in ptx.splitlines():
        tok = line.strip().split()
        if not tok or tok[0].startswith(("//", ".", "@", "{", "}", "bra", "ret")):
            continue
        if PTX_FP_LIKE.match(tok[0].rstrip(";")):
            ops.add(tok[0].rstrip(";"))
    bad = sorted(
        o
        for o in ops
        if not PTX_FP_ALLOWED.match(o)
        and not re.search(r"\.(b|u|s)(8|16|32|64)(\.|$)", o)
        and not re.match(r"^(cvt\.rzi|mul\.(wide|lo|hi))", o)
    )
    assert not bad, f"PTX contains non-admitted floating-point forms: {bad}"
    assert not re.search(r"rcp\.|div\.approx|div\.full|sqrt\.", ptx)
    assert "ftz" not in ptx.replace("ex2.approx.ftz.f32", "").replace(
        "lg2.approx.ftz.f32", ""
    )
    assert ("ex2.approx.ftz.f32" in ptx) == from_logits and (
        "lg2.approx.ftz.f32" in ptx
    ) == lp
    assert ptx.count("div.rn.f32") == (
        1 if trow else 0
    )  # LOG2E / T_row only: the draw never divides
    assert "cvt.rn.f32.u32" in ptx and ("mul.wide.u32" in ptx or "mul.hi.u32" in ptx)
    Cc = cfg[4] if len(cfg) > 4 else 1
    assert ("barrier.cluster" in ptx) == (Cc > 1) and (
        "ld.shared::cluster.b32" in ptx
    ) == (Cc > 1)


# ----------------------------------------------------------------------------------------------- on the device

_WORKER_CODE = r"""
import ast, sys
from flashinfer.batch_invariant.sampling.schedule import LSEConfig
from flashinfer.batch_invariant.sampling.draw_kernel import get_compiled_kernel
from flashinfer.batch_invariant.sampling.stream_kernel import get_stream_words_kernel
bad = []
for s in ast.literal_eval(sys.argv[1]):
    try:
        if s == "stream":
            get_stream_words_kernel()
        else:
            ds, N, cfg, from_logits, trow, idx, lp, pdl = s
            get_compiled_kernel(ds, N, LSEConfig(*cfg), from_logits, trow, idx, lp, pdl)
    except Exception as e:  # noqa: BLE001
        bad.append((s, repr(e)[:300]))
print("BAD", repr(bad))
"""


def _cfg_tuple(c):
    return (c.threads_per_row, c.rows_per_cta, c.vec_elems, c.stage, c.cluster)


def _suite_specs(sm):
    """Every kernel the device tests below may ask for: the config samples (probabilities; logits with per-row
    temperatures and the token's log-probability), the dispatcher's picks for the tests' row counts (both forms, with
    and without the gather and the log-probability), the big vocabulary, the stream kernel."""
    specs = ["stream"]
    for dtype in DT:
        ds, eb = _DS[dtype], dtype.itemsize
        for N in NS:
            for c in _configs(N, eb):
                t = _cfg_tuple(c)
                specs.append((ds, N, t, False, False, False, False, _PDL))
                specs.append((ds, N, t, True, True, False, True, _PDL))
            for M in MS:
                t = _cfg_tuple(DS.draw_config(N, M, eb, sm))
                specs.append((ds, N, t, False, False, False, False, _PDL))
                specs.append((ds, N, t, True, False, False, False, _PDL))
                specs.append((ds, N, t, True, True, False, True, _PDL))
                specs.append((ds, N, t, False, False, True, False, _PDL))
                specs.append((ds, N, t, True, True, True, True, _PDL))
                specs.append((ds, N, t, True, False, True, False, _PDL))
    for c in _configs(N_BIG, 2, 3):
        specs.append(("bf16", N_BIG, _cfg_tuple(c), True, False, False, True, _PDL))
    for M in (2, 512):
        t = _cfg_tuple(DS.draw_config(N_BIG, M, 2, sm))
        specs.append(("bf16", N_BIG, t, True, False, False, True, _PDL))
        specs.append(("bf16", N_BIG, t, True, False, False, False, _PDL))
    for M in (
        64,
        1,
        8,
    ):  # the probe's shape (fp32, N = 32000) and the determinism test's pdl-off kernel
        t = _cfg_tuple(DS.draw_config(32000, M, 4, sm))
        specs.append(("f32", 32000, t, False, False, False, False, _PDL))
        specs.append(("f32", 32000, t, True, False, False, False, _PDL))
    t = _cfg_tuple(DS.draw_config(4096, 5, 2, sm))
    specs.append(("bf16", 4096, t, False, False, False, False, False))
    specs.append(("bf16", 4096, t, True, False, False, False, False))
    return list(dict.fromkeys(specs))


@pytest.fixture(scope="session", autouse=True)
def _precompile_draw_kernels():
    if not _supported or os.environ.get("BI_DRAW_NO_PRECOMPILE") == "1":
        yield
        return
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit import env as jit_env

    dev = torch.device("cuda")
    sm = device_sm_count(dev)
    # the workers compile without a device: they take the device's architecture explicitly, so the module directory
    # (FLASHINFER_CUDA_ARCH_LIST decides the workspace's architecture segment, CUTE_DSL_ARCH the kernel target) is the
    # one this process loads from
    archs = sorted(CompilationContext().TARGET_CUDA_ARCHS)
    major, minor = archs[0]
    specs = _suite_specs(sm)
    random.Random(0).shuffle(
        specs
    )  # spread the heavy (cluster) kernels over the workers
    workers = int(
        os.environ.get("BI_DRAW_PRECOMPILE_WORKERS", min(32, os.cpu_count() or 8))
    )
    workers = max(1, min(workers, len(specs)))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        CUTE_DSL_ARCH=f"sm_{major}{minor}",
        FLASHINFER_CUDA_ARCH_LIST=" ".join(f"{a}.{b}" for a, b in archs),
    )
    chunks = [specs[i::workers] for i in range(workers)]

    def run(chunk):
        r = subprocess.run(
            [sys.executable, "-c", _WORKER_CODE, repr(chunk)],
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if r.returncode != 0:
            return [(("worker",), r.stderr[-1500:])]
        tail = [ln for ln in r.stdout.splitlines() if ln.startswith("BAD ")]
        import ast

        return ast.literal_eval(tail[-1][4:]) if tail else [(("worker",), "no report")]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        bad = [b for res in ex.map(run, chunks) for b in res]
    print(
        f"[batch_invariant.sampling.draw] precompiled {len(specs)} kernels on {workers} workers in {time.time() - t0:.0f} s",
        flush=True,
    )
    assert not bad, bad[:3]
    module_dir = (
        jit_env.FLASHINFER_JIT_DIR
        / f"batch_invariant_sampling_draw_sm{major}{minor}_cute_dsl"
    )
    n_cached = len(list(module_dir.glob("*.o"))) if module_dir.exists() else 0
    print(
        f"[batch_invariant.sampling.draw] {n_cached} kernels in {module_dir}",
        flush=True,
    )
    assert n_cached >= len(specs) // 2, (
        "the workers did not fill this process's module cache"
    )
    yield


@needs_device
def test_stream_words_match_the_specification():
    """The stream kernel's words and both uniforms equal the CPU specification for every key and draw (64 keys x 37
    draws), the same key gives the same words at any row of any batch, and the kernel is deterministic."""
    from flashinfer.batch_invariant.sampling.stream_kernel import stream_words

    rng = random.Random(1)
    seeds = [0, 1, 42, 2**63 - 1, -1, -(2**63)] + [
        rng.getrandbits(64) - 2**63 for _ in range(58)
    ]
    offsets = [0, 1, 2, 3, 1000, 2**40] + [rng.getrandbits(63) for _ in range(58)]
    draws = 37
    s = torch.tensor(seeds, dtype=torch.int64, device="cuda")
    o = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    w, u32, u64 = stream_words(s, o, draws)
    torch.cuda.synchronize()
    w, u32, u64 = w.cpu(), u32.cpu(), u64.cpu()
    ref = S.reference_words(seeds, offsets, draws)
    for b in range(len(seeds)):
        for d in range(draws):
            got = tuple(int(x) & 0xFFFFFFFF for x in w[b, d].tolist())
            assert got == ref[b][d], (b, d)
            r32, r64 = S.uniforms(seeds[b], offsets[b], d)
            assert u32[b, d].item() == r32 and u64[b, d].item() == r64, (b, d)
    w2, u2, _ = stream_words(
        torch.tensor([5, 9, 5, 7], device="cuda"),
        torch.tensor([1, 1, 1, 2], device="cuda"),
        4,
    )
    assert torch.equal(w2[0], w2[2]) and torch.equal(u2[0], u2[2])
    w1, _, _ = stream_words(
        torch.tensor([5], device="cuda"), torch.tensor([1], device="cuda"), 4
    )
    assert torch.equal(w1[0], w2[0]) and not torch.equal(w2[1], w2[0])
    for _ in range(3):
        w3, u3, _ = stream_words(s, o, draws)
        assert torch.equal(w3.cpu(), w) and torch.equal(u3.cpu(), u32)


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
@pytest.mark.parametrize("N", NS)
def test_probs_match_exact_model_across_configs(dtype, N):
    """Tokens and validity equal the exact model bit for bit for every configuration sampled (single-CTA, cluster,
    staged) and for the dispatcher's pick through the public entry."""
    M = 37
    p = weights(M, N, dtype)
    s, o = keys(M)
    ref = DR.draw_reference(p, s.tolist(), o.tolist())
    for cfg in _configs(N, dtype.itemsize):
        got = probs_draw(p, s, o, config=cfg)
        assert torch.equal(got["samples"].cpu().long(), ref["samples"]), cfg.describe()
        assert torch.equal(got["valid"].cpu(), ref["valid"]), cfg.describe()
    tok, ok = flashinfer.sampling.sampling_from_probs(
        p, seed=s, offset=o, return_valid=True, batch_invariant=True
    )
    assert tok.dtype == torch.int32 and ok.dtype == torch.bool
    assert torch.equal(tok.cpu().long(), ref["samples"]) and torch.equal(
        ok.cpu(), ref["valid"]
    )


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
@pytest.mark.parametrize("N", NS)
def test_logits_match_exact_model_and_logprob(dtype, N):
    """The logits form: tokens and validity equal the exact model (the ``lse.row`` weights with per-row temperatures)
    for every configuration sampled, and R4: the drawn token's log-probability is ``log_probs`` of it, bit for bit; the
    public entry (no temperature) equals the model."""
    M = 11
    x = logits(M, N, dtype)
    s, o = keys(M, 3)
    temps = torch.rand(M, device="cuda") + 0.5
    ref = DR.draw_reference(
        x, s.tolist(), o.tolist(), from_logits=True, temperature=temps
    )
    assert bool(ref["valid"].all())
    for cfg in _configs(N, dtype.itemsize):
        got = logits_draw(x, s, o, temperature=temps, return_logprob=True, config=cfg)
        assert torch.equal(got["samples"].cpu().long(), ref["samples"]), cfg.describe()
        assert torch.equal(got["valid"].cpu(), ref["valid"]), cfg.describe()
        lp_ref = flashinfer.sampling.log_probs(x, got["samples"].long(), temps)
        assert same(got["logp"], lp_ref), (
            "the drawn token's log-prob is log_probs of it"
        )
    ref0 = DR.draw_reference(x, s.tolist(), o.tolist(), from_logits=True)
    tok = flashinfer.sampling.sampling_from_logits(
        x, seed=s, offset=o, batch_invariant=True
    )
    assert tok.dtype == torch.int32 and torch.equal(tok.cpu().long(), ref0["samples"])


@needs_device
def test_big_vocabulary():
    """N = 152064 (594 blocks padded to 1024): the exact model, three members and the dispatcher's picks of both
    regimes agree, and a row alone equals the row inside a batch of 512."""
    x = logits(2, N_BIG, torch.bfloat16)
    s, o = keys(2, 5)
    ref = DR.draw_reference(x, s.tolist(), o.tolist(), from_logits=True)
    for cfg in _configs(N_BIG, 2, 3) + [None]:
        got = logits_draw(x, s, o, return_logprob=True, config=cfg)
        assert torch.equal(got["samples"].cpu().long(), ref["samples"]) and bool(
            got["valid"].all()
        )
        assert same(
            got["logp"], flashinfer.sampling.log_probs(x, got["samples"].long())
        )
    xb = logits(512, N_BIG, torch.bfloat16, seed=2)
    xb[100] = x[0]
    sb, ob = keys(512, 6)
    sb[100], ob[100] = s[0], o[0]
    big = flashinfer.sampling.sampling_from_logits(
        xb, seed=sb, offset=ob, batch_invariant=True
    )
    assert int(big[100]) == int(ref["samples"][0])


@needs_device
def test_batch_invariance_the_way_the_operator_was_probed():
    """The probe that found the operator's default path slot-dependent, run on the flag: row 5's token alone == at slot
    5 of a batch of 64 == at slot 5 of a batch of 8 (same per-row seed / offset); a batch permutation moves no token;
    for probabilities and for logits. The default path's count of moved tokens is printed for contrast, not asserted."""
    torch.manual_seed(0)
    N, M = 32000, 64
    probs = torch.softmax(torch.randn(M, N, device="cuda") * 2, dim=-1)
    lg = torch.randn(M, N, device="cuda") * 2
    seed = torch.full((M,), 1234, dtype=torch.int64, device="cuda")
    off = torch.full((M,), 7, dtype=torch.int64, device="cuda")
    perm = torch.roll(torch.arange(M, device="cuda"), 1)
    for fn, x in (
        (flashinfer.sampling.sampling_from_probs, probs),
        (flashinfer.sampling.sampling_from_logits, lg),
    ):
        a = fn(x, seed=seed, offset=off, batch_invariant=True)
        assert torch.equal(a, fn(x, seed=seed, offset=off, batch_invariant=True))
        alone = fn(x[5:6].clone(), seed=seed[:1], offset=off[:1], batch_invariant=True)
        a8 = fn(x[:8], seed=seed[:8], offset=off[:8], batch_invariant=True)
        assert int(alone[0]) == int(a[5]) == int(a8[5])
        ap = fn(x[perm], seed=seed[perm], offset=off[perm], batch_invariant=True)
        assert torch.equal(ap, a[perm]), "a batch permutation moved a token"
        d = fn(x, seed=seed, offset=off, deterministic=True)
        dp = fn(x[perm], seed=seed[perm], offset=off[perm], deterministic=True)
        print(
            f"[probe] {fn.__name__}: default path moves {int((dp != d[perm]).sum())} of {M} tokens under the permutation; "
            f"the flag moves 0",
            flush=True,
        )


@needs_device
def test_batch_invariance_and_key_not_slot():
    """R2 and the key: rows alone equal their rows in the batch; a permuted batch permutes the tokens; the same key at
    another slot draws the same token from the same row; other keys draw (almost surely) other tokens."""
    N, M = 4096, 64
    p = weights(M, N, torch.float32)
    s, o = keys(M, 9)
    full = flashinfer.sampling.sampling_from_probs(
        p, seed=s, offset=o, batch_invariant=True
    )
    for i in (0, 17, 63):
        one = flashinfer.sampling.sampling_from_probs(
            p[i : i + 1].clone(),
            seed=s[i : i + 1],
            offset=o[i : i + 1],
            batch_invariant=True,
        )
        assert torch.equal(one, full[i : i + 1])
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1)).cuda()
    got = flashinfer.sampling.sampling_from_probs(
        p[perm].contiguous(), seed=s[perm], offset=o[perm], batch_invariant=True
    )
    assert torch.equal(got, full[perm])
    for k in (1, 3, 16):
        idx = torch.randperm(M, generator=torch.Generator().manual_seed(k))[:k].cuda()
        got = flashinfer.sampling.sampling_from_probs(
            p[idx].contiguous(), seed=s[idx], offset=o[idx], batch_invariant=True
        )
        assert torch.equal(got, full[idx]), k
    same_key = flashinfer.sampling.sampling_from_probs(
        p[17].repeat(4, 1),
        seed=s[17].repeat(4),
        offset=o[17].repeat(4),
        batch_invariant=True,
    )
    assert bool((same_key == full[17]).all())
    other = flashinfer.sampling.sampling_from_probs(
        p[17].repeat(64, 1),
        seed=s[17].repeat(64),
        offset=o[17] + torch.arange(64, device="cuda"),
        batch_invariant=True,
    )
    assert len(set(other.tolist())) > 8


@needs_device
def test_generator_only_call_shape_and_key_forms():
    """A call with only a ``generator`` reserves ``(seed, offset)`` exactly as the operator's path does (the same
    increment: the rows for probabilities, rows x vocabulary for logits), so a fresh generator seeded alike reproduces
    the draw through the exact model; scalar keys broadcast to the rows; a length-one tensor broadcasts; uint64 keys
    are read by their bits; a seed without an offset takes the generator's pair (the operator's rule)."""
    N, M = 1000, 6
    p = weights(M, N, torch.float32, seed=4)
    x = logits(M, N, torch.bfloat16)
    dev = torch.device("cuda", torch.cuda.current_device())
    for fn, inp, inc, fl in (
        (flashinfer.sampling.sampling_from_probs, p, M, False),
        (flashinfer.sampling.sampling_from_logits, x, M * N, True),
    ):
        g = torch.Generator(device="cuda").manual_seed(11)
        tok = fn(inp, generator=g, batch_invariant=True)
        g2 = torch.Generator(device="cuda").manual_seed(11)
        seed, offset = flashinfer.sampling.get_seed_and_offset(inc, g2, dev)
        ref = DR.draw_reference(inp, [seed] * M, [offset] * M, from_logits=fl)
        assert torch.equal(tok.cpu().long(), ref["samples"]) and bool(
            ref["valid"].all()
        )
        assert torch.equal(fn(inp, seed=seed, offset=offset, batch_invariant=True), tok)
        assert torch.equal(
            fn(
                inp,
                seed=torch.tensor([seed], device="cuda"),
                offset=torch.tensor([offset], device="cuda"),
                batch_invariant=True,
            ),
            tok,
        )
        g3 = torch.Generator(device="cuda").manual_seed(11)
        assert torch.equal(fn(inp, seed=5, generator=g3, batch_invariant=True), tok)
        tok2 = fn(
            inp, generator=g, batch_invariant=True
        )  # the generator advanced: another pair
        seed2, offset2 = flashinfer.sampling.get_seed_and_offset(inc, g2, dev)
        assert (seed2, offset2) != (seed, offset)
        ref2 = DR.draw_reference(inp, [seed2] * M, [offset2] * M, from_logits=fl)
        assert torch.equal(tok2.cpu().long(), ref2["samples"])
    big = 2**63 + 5
    su = torch.full((M,), big, dtype=torch.uint64, device="cuda")
    ou = torch.full((M,), 2**64 - 3, dtype=torch.uint64, device="cuda")
    tok = flashinfer.sampling.sampling_from_probs(
        p, seed=su, offset=ou, batch_invariant=True
    )
    ref = DR.draw_reference(p, [big] * M, [2**64 - 3] * M)
    assert torch.equal(tok.cpu().long(), ref["samples"])
    with pytest.raises(ValueError, match="length"):
        flashinfer.sampling.sampling_from_probs(
            p,
            seed=torch.zeros(5, dtype=torch.int64, device="cuda"),
            offset=ou,
            batch_invariant=True,
        )
    with pytest.raises(ValueError, match="both"):
        flashinfer.sampling.sampling_from_probs(
            p, seed=su, offset=3, batch_invariant=True
        )


@needs_device
def test_index_gather():
    """``indices[i]`` selects the source row of output ``i`` (one row per CTA); the output takes the indices' dtype, as
    the operator documents; logits with per-row temperatures and the token's log-probability through the gather."""
    N = 1000
    p = weights(5, N, torch.float32)
    ids = torch.tensor([4, 0, 2, 2, 1, 3], device="cuda")
    s, o = keys(6, 11)
    tok = flashinfer.sampling.sampling_from_probs(
        p, indices=ids, seed=s, offset=o, batch_invariant=True
    )
    ref = DR.draw_reference(p[ids], s.tolist(), o.tolist())
    assert tok.dtype == torch.int64 and torch.equal(tok.cpu(), ref["samples"])
    tok32 = flashinfer.sampling.sampling_from_probs(
        p, indices=ids.to(torch.int32), seed=s, offset=o, batch_invariant=True
    )
    assert tok32.dtype == torch.int32 and torch.equal(tok32.long(), tok)
    x = logits(5, N, torch.bfloat16)
    temps = torch.rand(6, device="cuda") + 0.5
    got = logits_draw(x, s, o, temperature=temps, indices=ids, return_logprob=True)
    ref = DR.draw_reference(
        x[ids], s.tolist(), o.tolist(), from_logits=True, temperature=temps
    )
    assert torch.equal(got["samples"].cpu().long(), ref["samples"])
    assert same(
        got["logp"], flashinfer.sampling.log_probs(x[ids], got["samples"].long(), temps)
    )
    assert torch.equal(
        flashinfer.sampling.sampling_from_logits(
            x, indices=ids, seed=s, offset=o, batch_invariant=True
        ).cpu(),
        DR.draw_reference(x[ids], s.tolist(), o.tolist(), from_logits=True)["samples"],
    )
    with pytest.raises(ValueError, match="one row per CTA"):
        probs_draw(p, s, o, indices=ids, config=C.LSEConfig(32, 2, 4))


@needs_device
def test_invalid_rows_check_nan_and_validity():
    """A row whose weights do not sum to a positive finite number (all zero, a NaN, an infinite weight, a row of -inf
    logits) is invalid: token -1 and ``valid`` False; ``check_nan`` is the operator's own host check."""
    p = torch.rand(5, 300, device="cuda")
    p[1] = 0
    p[3, 7] = float("nan")
    p[4, 9] = float("inf")
    s, o = keys(5, 13)
    tok, ok = flashinfer.sampling.sampling_from_probs(
        p, seed=s, offset=o, return_valid=True, batch_invariant=True
    )
    assert ok.tolist() == [True, False, True, False, False]
    assert tok[1] == -1 and tok[3] == -1 and tok[4] == -1
    ref = DR.draw_reference(p, s.tolist(), o.tolist())
    assert torch.equal(tok.cpu().long(), ref["samples"]) and torch.equal(
        ok.cpu(), ref["valid"]
    )
    with pytest.raises(ValueError, match="NaN"):
        flashinfer.sampling.sampling_from_probs(
            p, seed=s, offset=o, check_nan=True, batch_invariant=True
        )
    x = torch.randn(4, 300, device="cuda")
    x[1] = float("-inf")
    x[2, 5] = float("nan")
    tok = flashinfer.sampling.sampling_from_logits(
        x, seed=s[:4], offset=o[:4], batch_invariant=True
    )
    assert (
        tok[1] == -1
        and tok[2] == -1
        and 0 <= int(tok[0]) < 300
        and 0 <= int(tok[3]) < 300
    )
    got = logits_draw(x, s[:4], o[:4], return_logprob=True)
    assert got["valid"].tolist() == [True, False, False, True]
    assert torch.isnan(got["logp"][1]) and torch.isnan(got["logp"][2])
    with pytest.raises(ValueError, match="NaN"):
        flashinfer.sampling.sampling_from_logits(
            x, seed=s[:4], offset=o[:4], check_nan=True, batch_invariant=True
        )
    x[1, 0] = 0.0  # a masked row with one live element draws it
    assert (
        int(
            flashinfer.sampling.sampling_from_logits(
                x[1:2], seed=s[:1], offset=o[:1], batch_invariant=True
            )[0]
        )
        == 0
    )


@needs_device
def test_distribution_follows_the_weights():
    """Many keys over one row: the drawn frequencies follow ``p / S`` (a sanity bound, not a promise), for the weights
    and for their logarithms through the logits entry."""
    N, M = 32, 20000
    p = torch.rand(N, device="cuda")
    p[3] = 4.0
    seeds = torch.full((M,), 5, dtype=torch.int64, device="cuda")
    offs = torch.arange(M, device="cuda")
    tok = flashinfer.sampling.sampling_from_probs(
        p.expand(M, N).contiguous(), seed=seeds, offset=offs, batch_invariant=True
    )
    freq = torch.bincount(tok.long(), minlength=N).float() / M
    assert float((freq - p / p.sum()).abs().max()) < 0.015 and int(freq.argmax()) == 3
    tok = flashinfer.sampling.sampling_from_logits(
        p.log().expand(M, N).contiguous(), seed=seeds, offset=offs, batch_invariant=True
    )
    freq = torch.bincount(tok.long(), minlength=N).float() / M
    assert float((freq - p / p.sum()).abs().max()) < 0.015 and int(freq.argmax()) == 3


@needs_device
def test_sm_count_independence():
    """The dispatcher's pick depends on the SM count; the bits do not."""
    N, M = 4097, 37
    p = weights(M, N, torch.bfloat16)
    x = logits(M, N, torch.bfloat16)
    s, o = keys(M, 2)
    ref_p = sampling_from_probs_batch_invariant(p, seed=s, offset=o)
    ref_x = sampling_from_logits_batch_invariant(x, seed=s, offset=o)
    for sm in SM_COUNTS:
        assert torch.equal(
            sampling_from_probs_batch_invariant(p, seed=s, offset=o, sm_count=sm), ref_p
        )
        assert torch.equal(
            sampling_from_logits_batch_invariant(x, seed=s, offset=o, sm_count=sm),
            ref_x,
        )
    p = weights(512, 32768, torch.bfloat16, seed=3)
    s, o = keys(512, 4)
    ref = sampling_from_probs_batch_invariant(p, seed=s, offset=o)
    for sm in SM_COUNTS:
        assert torch.equal(
            sampling_from_probs_batch_invariant(p, seed=s, offset=o, sm_count=sm), ref
        )


@needs_device
def test_determinism_graph_replay_and_pdl():
    """R1: repeated launches, PDL off and on, and CUDA-graph replay give the same tokens."""
    N, M = 4096, 5
    p = weights(M, N, torch.bfloat16)
    x = logits(M, N, torch.bfloat16)
    s, o = keys(M, 7)
    ref_p = flashinfer.sampling.sampling_from_probs(
        p, seed=s, offset=o, batch_invariant=True
    )
    ref_x = flashinfer.sampling.sampling_from_logits(
        x, seed=s, offset=o, batch_invariant=True
    )
    for _ in range(3):
        assert torch.equal(
            flashinfer.sampling.sampling_from_probs(
                p, seed=s, offset=o, batch_invariant=True
            ),
            ref_p,
        )
        assert torch.equal(
            flashinfer.sampling.sampling_from_logits(
                x, seed=s, offset=o, batch_invariant=True
            ),
            ref_x,
        )
    for pdl in (False, True):
        assert torch.equal(draw(p, s, o, enable_pdl=pdl)["samples"], ref_p)
        assert torch.equal(
            draw(x, s, o, from_logits=True, enable_pdl=pdl)["samples"], ref_x
        )
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        tp = flashinfer.sampling.sampling_from_probs(
            p, seed=s, offset=o, batch_invariant=True
        )
        tx = flashinfer.sampling.sampling_from_logits(
            x, seed=s, offset=o, batch_invariant=True
        )
        torch.cuda.current_stream().synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            tp = flashinfer.sampling.sampling_from_probs(
                p, seed=s, offset=o, batch_invariant=True
            )
            tx = flashinfer.sampling.sampling_from_logits(
                x, seed=s, offset=o, batch_invariant=True
            )
    torch.cuda.synchronize()
    tp.fill_(-7)
    tx.fill_(-7)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(tp, ref_p) and torch.equal(tx, ref_x)


@needs_device
def test_default_path_is_the_operator_kernel(monkeypatch):
    """``batch_invariant=False`` (the default) never enters the class and is byte for byte the operator's own kernel
    (the registered custom op the unmodified entries called), for both entries and with ``return_valid``."""
    import flashinfer.batch_invariant.sampling.draw_api as dapi

    def boom(*a, **k):
        raise AssertionError("the batch-invariant class ran without the flag")

    monkeypatch.setattr(dapi, "sampling_from_probs_batch_invariant", boom)
    monkeypatch.setattr(dapi, "sampling_from_logits_batch_invariant", boom)
    torch.manual_seed(0)
    N, M = 32000, 16
    probs = torch.softmax(torch.randn(M, N, device="cuda") * 2, dim=-1)
    lg = torch.randn(M, N, device="cuda") * 2
    seed = torch.full((M,), 1234, dtype=torch.int64, device="cuda")
    off = torch.full((M,), 7, dtype=torch.int64, device="cuda")
    mod = flashinfer.sampling.get_sampling_module()
    a, v = flashinfer.sampling.sampling_from_probs(
        probs, seed=seed, offset=off, return_valid=True
    )
    b, w = mod.sampling_from_probs(probs, None, True, None, seed, off, True)
    assert torch.equal(a, b) and torch.equal(v, w)
    assert torch.equal(
        flashinfer.sampling.sampling_from_logits(lg, seed=seed, offset=off),
        mod.sampling_from_logits(lg, None, True, None, seed, off),
    )
    with pytest.raises(AssertionError, match="without the flag"):
        flashinfer.sampling.sampling_from_probs(
            probs, seed=seed, offset=off, batch_invariant=True
        )
    with pytest.raises(AssertionError, match="without the flag"):
        flashinfer.sampling.sampling_from_logits(
            lg, seed=seed, offset=off, batch_invariant=True
        )


@needs_device
def test_shapes_views_and_empty_batches():
    """3-D inputs (tokens of the leading shape), a view into a wider tensor and a strided view (copied: a placement),
    M = 0, the output dtypes, the unsupported dtypes."""
    N = 4096
    s, o = keys(6, 8)
    x3 = weights(6, N, torch.bfloat16).view(2, 3, N)
    t3, v3 = flashinfer.sampling.sampling_from_probs(
        x3, seed=s, offset=o, return_valid=True, batch_invariant=True
    )
    assert (
        t3.shape == (2, 3)
        and t3.dtype == torch.int32
        and v3.shape == (2, 3)
        and v3.dtype == torch.bool
    )
    assert torch.equal(
        t3.view(6),
        flashinfer.sampling.sampling_from_probs(
            x3.view(6, N), seed=s, offset=o, batch_invariant=True
        ),
    )
    assert torch.equal(
        flashinfer.sampling.sampling_from_logits(
            x3, seed=s, offset=o, batch_invariant=True
        ).view(6),
        flashinfer.sampling.sampling_from_logits(
            x3.view(6, N), seed=s, offset=o, batch_invariant=True
        ),
    )
    wide = weights(6, 2 * N, torch.bfloat16, seed=1)
    for v in (wide[:, :N], wide[:, ::2]):
        assert torch.equal(
            flashinfer.sampling.sampling_from_probs(
                v, seed=s, offset=o, batch_invariant=True
            ),
            flashinfer.sampling.sampling_from_probs(
                v.contiguous(), seed=s, offset=o, batch_invariant=True
            ),
        )
    x0 = torch.empty(0, N, device="cuda", dtype=torch.bfloat16)
    e = torch.empty(0, dtype=torch.int64, device="cuda")
    t0, v0 = flashinfer.sampling.sampling_from_probs(
        x0, seed=e, offset=e, return_valid=True, batch_invariant=True
    )
    assert t0.shape == (0,) and v0.shape == (0,)
    assert flashinfer.sampling.sampling_from_logits(
        x0, seed=e, offset=e, batch_invariant=True
    ).shape == (0,)
    for dt in (torch.float64, torch.int32):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="sample.icdf"):
            flashinfer.sampling.sampling_from_probs(
                torch.zeros(2, N, device="cuda", dtype=dt),
                seed=s[:2],
                offset=o[:2],
                batch_invariant=True,
            )
    with pytest.raises(
        ValueError, match="vector width"
    ):  # N = 4100: a 4-element alignment, an 8-element vector
        probs_draw(
            weights(2, 4100, torch.bfloat16), s[:2], o[:2], config=C.LSEConfig(32, 1, 8)
        )
