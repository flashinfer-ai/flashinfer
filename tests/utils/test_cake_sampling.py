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

Tests for :mod:`flashinfer.cake_sampling`.  Every case is compared *bitwise* (sorted slab,
``renorm_out``, sample) against an independent numpy reference of the documented semantics and
additionally carries a hand-derived expectation.  The adversarial section covers ties at the
k / p boundaries, uniform rows, fewer than k non-zero entries, all-zero / NaN / +inf / mixed rows,
denormals, a probability near one, extreme p and k, odd vocabularies, Philox offsets that are
not multiples of 4, identical rows, per-request tensors, bitwise replay across launches /
CUDA graphs / every frozen kernel variant, and parity with ``top_k_first`` sampling.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest
import subprocess
import torch

from flashinfer.cake_sampling import (
    cake_sampling_route,
    choose_stage1,
    choose_stage23,
    top_k_probs_to_slab,
    top_k_top_p_sampling_from_probs,
)
import flashinfer.jit.cake_sampling as cake_sampling_jit
from flashinfer.jit.cake_sampling import (
    load_cake_sampling_module,
    load_manifest,
    parse_gpu_arch_list,
    supported_capabilities,
    supported_capability,
)

SLAB = 1024
INF_KEY = 0x7F800000
_PHILOX_M0, _PHILOX_M1, _PHILOX_W0, _PHILOX_W1 = (
    0xD2511F53,
    0xCD9E8D57,
    0x9E3779B9,
    0xBB67AE85,
)


def _require_supported_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    capability = supported_capability(torch.cuda.get_device_capability())
    if capability is None:
        pytest.skip(
            "frozen radix sampling kernels need compute capability 9.0/10.0/10.3/10.7/11.0"
        )
    return capability


def _probs(batch, vocab, seed=536, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.softmax(
        torch.randn(batch, vocab, device="cuda", generator=g) * scale, dim=-1
    )


def _rows(*rows):
    return torch.tensor(
        np.stack([np.ascontiguousarray(r, dtype=np.float32) for r in rows]),
        device="cuda",
    )


# ------------------------------------------------------------------ independent host reference


def _curand_uniform(seed: int, subsequence: int, offset: int) -> np.float32:
    """First curand_uniform() after curand_init(seed, subsequence, offset) (Philox4x32-10)."""
    mask = 0xFFFFFFFF
    c = [
        (offset >> 2) & mask,
        (offset >> 34) & mask,
        subsequence & mask,
        (subsequence >> 32) & mask,
    ]
    k = [seed & mask, (seed >> 32) & mask]
    for r in range(10):
        p0, p1 = _PHILOX_M0 * c[0], _PHILOX_M1 * c[2]
        c = [
            ((p1 >> 32) & mask) ^ c[1] ^ k[0],
            p1 & mask,
            ((p0 >> 32) & mask) ^ c[3] ^ k[1],
            p0 & mask,
        ]
        if r < 9:
            k = [(k[0] + _PHILOX_W0) & mask, (k[1] + _PHILOX_W1) & mask]
    word = c[offset & 3]
    return np.float32(
        np.float32(word) * np.float32(2.3283064e-10) + np.float32(1.1641532e-10)
    )


def _keys(row: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(row, dtype=np.float32).view(np.uint32)
    return np.where(bits <= np.uint32(INF_KEY), bits, np.uint32(0)).astype(np.uint32)


@dataclass
class Ref:
    idx: np.ndarray
    vals: np.ndarray
    renorm: np.ndarray
    sample: int
    kept: list


def _reference(row: np.ndarray, k: int, p: float, uniform: np.float32) -> Ref:
    """Documented semantics: sanitized keys, lexsort(-key, index) support, exact fixed-point
    prefix sums, p clamped to (0, 1], +inf rows, zero-mass rows, float64 renormalization."""
    vocab = row.shape[0]
    k = max(1, min(int(k), vocab, SLAB))
    keys = _keys(row)
    idx = np.lexsort((np.arange(vocab), -keys.astype(np.int64)))[:k].astype(np.int32)
    kk = [int(x) for x in keys[idx]]
    vals = keys[idx].view(np.float32).copy()
    is_inf = kk[0] == INF_KEY
    if is_inf:
        ints = [1 if x == INF_KEY else 0 for x in kk]
        vint0 = 1
    else:
        e0 = kk[0] >> 23
        eff0 = e0 if e0 else 1
        vint0 = ((kk[0] & 0x7FFFFF) | (0x800000 if e0 else 0)) << 29
        ints = []
        for key in kk:
            e = key >> 23
            m = (key & 0x7FFFFF) | (0x800000 if e else 0)
            shift = eff0 - (e if e else 1)
            ints.append(((m << 29) >> shift) if shift < 64 else 0)
    incl, run = [], 0
    for v in ints:
        run += v
        incl.append(run)
    excl = [0] + incl[:-1]
    total_f = float(incl[-1])
    p32 = np.float32(p)
    p_eff = np.float32(1.0) if not (p32 < np.float32(1.0)) else p32
    target = float(np.float64(p_eff) * np.float64(total_f))
    if is_inf:
        target = total_f
    if not (target > 0.0):
        target = 1.0
    kept = [float(excl[j]) < target and ints[j] > 0 for j in range(k)]
    cutoff, kept_int = 0, 0
    for j in range(k):
        if kept[j] and float(incl[j]) >= target:
            cutoff, kept_int = j, incl[j]
    kept_f = float(kept_int)
    u_f = float(np.float64(uniform) * np.float64(kept_f))
    pos = cutoff
    if u_f < kept_f:
        for j in range(k):
            if kept[j] and float(excl[j]) <= u_f < float(incl[j]):
                pos = j
                break
    renorm = np.zeros(k, dtype=np.float32)
    if kept_int > 0:
        if is_inf:
            r = np.float32(np.float64(1.0) / np.float64(kept_f))
            renorm[[j for j in range(k) if kept[j]]] = r
        else:
            kept_mass = (np.float64(kept_f) / np.float64(float(vint0))) * np.float64(
                vals[0]
            )
            for j in range(k):
                if kept[j]:
                    renorm[j] = np.float32(np.float64(vals[j]) / kept_mass)
    return Ref(
        idx, vals, renorm, int(idx[pos]), [int(idx[j]) for j in range(k) if kept[j]]
    )


# ----------------------------------------------------------------------------- run helpers


@dataclass
class Run:
    samples: np.ndarray
    renorm: np.ndarray
    vals: np.ndarray
    idx: np.ndarray
    count: np.ndarray


def _ws(batch):
    return (
        torch.empty(batch, SLAB, device="cuda", dtype=torch.float32),
        torch.empty(batch, SLAB, device="cuda", dtype=torch.int32),
        torch.empty(batch, device="cuda", dtype=torch.int32),
    )


def _run(probs, k, p, seed, offset, *, variant=None, out=None, pdl=True) -> Run:
    batch = probs.shape[0]
    ws = _ws(batch)
    renorm = torch.full((batch, SLAB), float("nan"), device="cuda")
    kmax = k if isinstance(k, int) else int(k.max().item())
    if variant is None:
        out = top_k_top_p_sampling_from_probs(
            probs,
            k,
            p,
            top_k_max=kmax,
            philox_seed=seed,
            philox_offset=offset,
            out=out,
            renorm_out=renorm,
            workspace=ws,
            enable_pdl=pdl,
        )
        assert cake_sampling_route(probs, k, kmax) == "pipeline"
    else:
        s1, (threads, items) = variant
        cluster, ept = s1[0], s1[1]
        stream_variant = 1 if len(s1) > 2 and s1[2] else 0
        capability = _require_supported_device()
        module = load_cake_sampling_module(capability)
        vals, idxs, cnt = ws
        if out is None:
            out = torch.empty(batch, device="cuda", dtype=torch.int32)
        k_args = (cnt, int(k), 1) if isinstance(k, int) else (k, 0, 2)
        p_args = (probs, float(p), 1) if isinstance(p, (int, float)) else (p, 0.0, 2)
        stream = torch.cuda.current_stream().cuda_stream
        module.radix_topk(
            probs,
            k_args[0],
            k_args[1],
            k_args[2],
            vals,
            idxs,
            cnt,
            cluster,
            ept,
            stream_variant,
            stream,
        )
        module.sparse_topp_sample(
            vals,
            idxs,
            cnt,
            p_args[0],
            p_args[1],
            p_args[2],
            out,
            renorm,
            seed,
            offset,
            1,
            threads,
            items,
            1 if pdl else 0,
            stream,
        )
    torch.cuda.synchronize()
    vals, idxs, cnt = ws
    return Run(
        out.cpu().numpy().copy(),
        renorm.cpu().numpy(),
        vals.cpu().numpy(),
        idxs.cpu().numpy(),
        cnt.cpu().numpy(),
    )


def _check(run: Run, pn: np.ndarray, k, p, seed, offset):
    batch, vocab = pn.shape
    refs = []
    for r in range(batch):
        kr = int(k if isinstance(k, int) else k[r])
        pr = float(p if isinstance(p, (int, float)) else p[r])
        ref = _reference(pn[r], kr, pr, _curand_uniform(seed, r, offset))
        n = int(run.count[r])
        assert n == max(1, min(kr, vocab, SLAB)) == len(ref.idx), f"row {r}: count {n}"
        got_idx, got_vals, got_ren = run.idx[r, :n], run.vals[r, :n], run.renorm[r, :n]
        assert (
            got_idx.min() >= 0
            and got_idx.max() < vocab
            and len(np.unique(got_idx)) == n
        )
        assert (
            np.isfinite(got_ren).all() and (got_ren >= 0).all() and (got_ren <= 1).all()
        )
        assert (
            not np.isnan(got_vals).any()
            and (got_vals[np.isfinite(got_vals)] >= 0).all()
        )
        assert int(run.samples[r]) in set(got_idx.tolist())
        assert np.array_equal(got_idx, ref.idx), (
            f"row {r}: slab order != lexsort(-prob, index)"
        )
        assert np.array_equal(got_vals.view(np.uint32), ref.vals.view(np.uint32)), (
            f"row {r}: slab values"
        )
        assert np.array_equal(got_ren.view(np.uint32), ref.renorm.view(np.uint32)), (
            f"row {r}: renorm"
        )
        assert int(run.samples[r]) == ref.sample, (
            f"row {r}: sample {run.samples[r]} != {ref.sample}"
        )
        refs.append(ref)
    return refs


def _run_and_check(probs, k, p, seed, offset, **kw):
    run = _run(probs, k, p, seed, offset, **kw)
    kk = k if isinstance(k, int) else k.cpu().numpy()
    pp = p if isinstance(p, (int, float)) else p.cpu().numpy()
    return run, _check(run, probs.cpu().numpy(), kk, pp, seed, offset)


def _kept(run: Run, r: int):
    n = int(run.count[r])
    return run.idx[r, :n][run.renorm[r, :n] > 0].tolist()


# ------------------------------------------------------------------------------- baseline


@pytest.mark.parametrize(
    ("batch", "vocab", "k", "p"),
    [
        (1, 32768, 50, 0.9),
        (8, 128256, 50, 0.9),
        (2, 151936, 10, 0.5),
        (4, 262144, 1000, 0.95),
        (3, 32003, 64, 0.9),
        (1, 128256, 1, 0.9),
        (2, 128256, 50, 1e-6),
        (16, 128256, 50, 0.9),
    ],
)
def test_support_matches_top_k_first_semantics(batch, vocab, k, p):
    _require_supported_device()
    probs = _probs(batch, vocab)
    run, _ = _run_and_check(probs, k, p, 0xC0FFEE, 3)
    pn = probs.cpu().numpy()
    for r in range(batch):
        top = torch.topk(probs[r], k).indices.cpu().numpy()
        assert sorted(run.idx[r, :k].tolist()) == sorted(top.tolist())
        if k == 1 or p <= 1e-6:
            assert int(run.samples[r]) == int(np.argmax(pn[r]))


def test_deterministic_replay_and_offset_sensitivity():
    _require_supported_device()
    probs = _probs(16, 128256)
    a = _run(probs, 50, 0.9, 11, 5)
    for _ in range(5):
        b = _run(probs, 50, 0.9, 11, 5)
        assert np.array_equal(a.samples, b.samples) and np.array_equal(
            a.idx[:, :50], b.idx[:, :50]
        )
        assert np.array_equal(
            a.renorm[:, :50].view(np.uint32), b.renorm[:, :50].view(np.uint32)
        )
    c = _run(probs, 50, 0.9, 11, 6)
    assert not np.array_equal(a.samples, c.samples)
    assert np.array_equal(a.idx[:, :50], c.idx[:, :50])


def test_generator_advances_like_flashinfer_sampling():
    _require_supported_device()
    from flashinfer.sampling import get_seed_and_offset

    probs = _probs(6, 32768)
    g1 = torch.Generator(device="cuda").manual_seed(4242)
    g2 = torch.Generator(device="cuda").manual_seed(4242)
    top_k_top_p_sampling_from_probs(probs, 50, 0.9, generator=g1)
    get_seed_and_offset(6 * 32, g2, torch.device("cuda"))
    assert torch.equal(g1.get_state(), g2.get_state())


def test_statistical_total_variation():
    _require_supported_device()
    batch, vocab, k, p = 4, 32768, 50, 0.9
    probs = _probs(batch, vocab, scale=3.0)
    _, refs = _run_and_check(probs, k, p, 2024, 0)
    draws = 20000
    counts = [dict() for _ in range(batch)]
    for t in range(draws):
        out = top_k_top_p_sampling_from_probs(
            probs, k, p, philox_seed=2024, philox_offset=t
        )
        for r, tok in enumerate(out.cpu().tolist()):
            counts[r][tok] = counts[r].get(tok, 0) + 1
    for r in range(batch):
        ref = {
            int(i): float(v)
            for i, v in zip(refs[r].idx, refs[r].renorm, strict=False)
            if v > 0
        }
        assert set(counts[r]) <= set(ref)
        tv = 0.5 * sum(abs(counts[r].get(i, 0) / draws - ref[i]) for i in ref)
        assert tv < 3.0 * math.sqrt(len(ref) / (2 * math.pi * draws))


def test_cuda_graph_capture_and_replay():
    _require_supported_device()
    probs = _probs(8, 128256)
    eager = _run(probs, 50, 0.9, 7, 1)
    out = torch.empty(8, device="cuda", dtype=torch.int32)
    renorm = torch.empty(8, SLAB, device="cuda")
    ws = _ws(8)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            top_k_top_p_sampling_from_probs(
                probs,
                50,
                0.9,
                philox_seed=7,
                philox_offset=1,
                out=out,
                renorm_out=renorm,
                workspace=ws,
            )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        top_k_top_p_sampling_from_probs(
            probs,
            50,
            0.9,
            philox_seed=7,
            philox_offset=1,
            out=out,
            renorm_out=renorm,
            workspace=ws,
        )
    for _ in range(3):
        out.zero_()
        renorm.fill_(-1.0)
        g.replay()
        torch.cuda.synchronize()
        assert np.array_equal(out.cpu().numpy(), eager.samples)
        assert np.array_equal(
            renorm[:, :50].cpu().numpy().view(np.uint32),
            eager.renorm[:, :50].view(np.uint32),
        )
        assert np.array_equal(ws[1][:, :50].cpu().numpy(), eager.idx[:, :50])


def test_per_request_tensors_and_routes():
    _require_supported_device()
    batch, vocab = 7, 128256
    probs = _probs(batch, vocab)
    pn = probs.cpu().numpy()
    pn[6] = 0
    probs.copy_(torch.tensor(pn, device="cuda"))
    ks = torch.tensor([1, 2, 50, 1023, 1024, 7, 50], device="cuda", dtype=torch.int32)
    ps = torch.tensor(
        [0.0, 1e-9, 0.5, 1.0, 2.0, float("nan"), 0.9],
        device="cuda",
        dtype=torch.float32,
    )
    run, _ = _run_and_check(probs, ks, ps, 1, 0)
    assert int(run.samples[0]) == int(np.argmax(pn[0])) and int(run.samples[1]) == int(
        np.argmax(pn[1])
    )
    assert (
        len(_kept(run, 3)) == 1023
        and len(_kept(run, 4)) == 1024
        and len(_kept(run, 5)) == 7
    )
    assert int(run.samples[6]) == 0 and not run.renorm[6, :50].any()
    assert cake_sampling_route(probs, 50) == "pipeline"
    assert cake_sampling_route(probs, None) == "fallback:no_top_k"
    assert cake_sampling_route(probs, vocab) == "fallback:top_k_disabled"
    assert cake_sampling_route(probs, 1025) == "fallback:top_k_gt_slab"
    assert cake_sampling_route(probs.half(), 50) == "fallback:dtype"
    assert (
        cake_sampling_route(torch.empty(256, 262144, device="cuda"), 50) == "pipeline"
    )
    assert choose_stage1(1, 128256) == (8, 32, False)
    assert choose_stage1(8, 65536) == (8, 16, False)
    assert choose_stage1(16, 128256) == (4, 16, True)
    assert choose_stage1(64, 128256) == (2, 16, True)
    assert choose_stage1(16, 262144) == (4, 16, True)
    assert choose_stage1(64, 262144)[2] and choose_stage1(128, 151936)[2]
    assert choose_stage23(50) == (32, 2)
    res = top_k_top_p_sampling_from_probs(probs, vocab, 0.9)
    assert res.dtype == torch.int32 and res.shape == (batch,)


# --------------------------------------------------------------------------- toolkit guard

_TABLE = ((9, 0), (10, 0), (10, 3), (10, 7), (11, 0))
# `nvcc --list-gpu-arch` of CUDA 13.3 (public toolkit): no compute_107.
_CUDA_13_3_ARCH_LIST = (
    "compute_75\ncompute_80\ncompute_86\ncompute_87\ncompute_88\ncompute_89\ncompute_90\n"
    "compute_100\ncompute_110\ncompute_103\ncompute_120\ncompute_121\n"
)


def test_toolkit_guard_parses_nvcc_arch_list():
    archs = parse_gpu_arch_list(_CUDA_13_3_ARCH_LIST)
    assert {"compute_90", "compute_100", "compute_103", "compute_110"} <= archs
    assert "compute_107" not in archs
    # Suffixed spellings collapse onto the base virtual architecture.
    assert parse_gpu_arch_list("compute_90a compute_100f") == {
        "compute_90",
        "compute_100",
    }
    assert parse_gpu_arch_list("") == frozenset()
    assert parse_gpu_arch_list("sm_90a\nnot-an-arch") == frozenset()


def test_toolkit_guard_rejects_capabilities_nvcc_cannot_target(monkeypatch):
    archs = parse_gpu_arch_list(_CUDA_13_3_ARCH_LIST)
    monkeypatch.setattr(cake_sampling_jit, "_nvcc_gpu_archs", lambda: archs)
    assert supported_capability((9, 0)) == (9, 0)
    assert supported_capability((10, 0)) == (10, 0)
    assert supported_capability((10, 3)) == (10, 3)
    assert supported_capability((11, 0)) == (11, 0)
    # compute_107 is only known to the internal Rubin toolchain: a public toolkit must not
    # attempt the JIT build for an SM107 device.
    assert supported_capability((10, 7)) is None
    assert (
        supported_capability((8, 0)) is None and supported_capability((12, 0)) is None
    )
    assert supported_capabilities() == ((9, 0), (10, 0), (10, 3), (11, 0))
    with pytest.raises(ValueError, match="not compiled for compute capability 10.7"):
        cake_sampling_jit.gen_cake_sampling_module((10, 7))

    hopper_only = frozenset({"compute_80", "compute_90"})
    monkeypatch.setattr(cake_sampling_jit, "_nvcc_gpu_archs", lambda: hopper_only)
    assert supported_capability((9, 0)) == (9, 0)
    assert all(supported_capability(cap) is None for cap in _TABLE[1:])
    assert supported_capabilities() == ((9, 0),)


def test_toolkit_guard_keeps_table_when_nvcc_cannot_be_queried(monkeypatch):
    monkeypatch.setattr(cake_sampling_jit, "_nvcc_gpu_archs", lambda: None)
    assert all(supported_capability(cap) == cap for cap in _TABLE)
    assert supported_capabilities() == _TABLE
    assert supported_capability((8, 9)) is None


def test_toolkit_guard_queries_nvcc_once(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[-1] == "--list-gpu-arch"
        return subprocess.CompletedProcess(cmd, 0, _CUDA_13_3_ARCH_LIST, "")

    monkeypatch.setattr(cake_sampling_jit.subprocess, "run", fake_run)
    monkeypatch.setattr(cake_sampling_jit, "_nvcc_executable", lambda: "nvcc")
    cake_sampling_jit._nvcc_gpu_archs.cache_clear()
    try:
        for _ in range(3):
            assert supported_capability((10, 7)) is None
            assert supported_capability((10, 3)) == (10, 3)
        assert len(calls) == 1
        monkeypatch.setattr(
            cake_sampling_jit.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "unknown option"),
        )
        cake_sampling_jit._nvcc_gpu_archs.cache_clear()
        assert cake_sampling_jit._nvcc_gpu_archs() is None
    finally:
        cake_sampling_jit._nvcc_gpu_archs.cache_clear()


def test_toolkit_guard_routes_untargetable_device_to_top_k_first(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flashinfer.sampling import top_k_top_p_sampling_from_probs as reference

    batch, vocab = 5, 32768
    probs = _probs(batch, vocab)
    expected = reference(
        probs,
        50,
        0.9,
        filter_apply_order="top_k_first",
        deterministic=True,
        seed=7,
        offset=8,
    )
    monkeypatch.setattr(
        cake_sampling_jit, "_nvcc_gpu_archs", lambda: frozenset({"compute_50"})
    )
    assert supported_capability(torch.cuda.get_device_capability()) is None
    assert cake_sampling_route(probs, 50) == "fallback:arch"
    got = top_k_top_p_sampling_from_probs(
        probs, 50, 0.9, philox_seed=7, philox_offset=8
    )
    assert got.dtype == torch.int32 and torch.equal(got.to(expected.dtype), expected)
    with pytest.raises(ValueError, match="fallback:arch"):
        top_k_top_p_sampling_from_probs(
            probs, 50, 0.9, renorm_out=torch.empty(batch, SLAB, device="cuda")
        )


# --------------------------------------------------------------------------- adversarial


def test_adv_equal_probs_straddling_k_boundary():
    _require_supported_device()
    vocab = 4096
    where = (np.arange(7, 7 + 100 * 37, 37) % vocab).astype(np.int64)
    row = np.zeros(vocab, np.float32)
    row[where] = np.float32(2**-7)
    probs = _rows(row, row)
    for offset in (0, 1, 2, 3):
        run, _ = _run_and_check(probs, 50, 1.0, 0xABCDEF, offset)
        for r in range(2):
            assert run.idx[r, :50].tolist() == sorted(where.tolist())[:50]
            assert np.all(run.renorm[r, :50] == np.float32(0.02))
            u = float(_curand_uniform(0xABCDEF, r, offset))
            assert (
                int(run.samples[r])
                == sorted(where.tolist())[min(int(math.floor(u * 50)), 49)]
            )


def test_adv_equal_probs_straddling_p_boundary():
    _require_supported_device()
    row = np.zeros(32768, np.float32)
    row[[400, 300, 200, 100]] = 0.25
    probs = _rows(row, row)
    run, _ = _run_and_check(probs, 10, 0.5, 9, 4)
    for r in range(2):
        assert run.idx[r, :10].tolist() == [100, 200, 300, 400, 0, 1, 2, 3, 4, 5]
        assert run.renorm[r, :10].tolist() == [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]
        assert int(run.samples[r]) in (100, 200)
    run, _ = _run_and_check(
        probs, 10, float(np.nextafter(np.float32(0.5), np.float32(1))), 9, 4
    )
    for r in range(2):
        assert _kept(run, r) == [100, 200, 300]
        assert np.all(
            run.renorm[r, :3] == np.float32(np.float64(0.25) / np.float64(0.75))
        )


def test_adv_fully_uniform_row():
    _require_supported_device()
    probs = _rows(np.full(4096, np.float32(2**-12)), np.full(4096, np.float32(2**-12)))
    for k, p, expect in ((50, 0.9, 45), (1024, 0.9, 922), (7, 1.0, 7), (1, 0.5, 1)):
        run, _ = _run_and_check(probs, k, p, 77, 1)
        for r in range(2):
            assert run.idx[r, :k].tolist() == list(range(k)) and _kept(run, r) == list(
                range(expect)
            )
            assert np.all(
                run.renorm[r, :expect] == np.float32(np.float64(1.0) / expect)
            )
            u = float(_curand_uniform(77, r, 1))
            assert int(run.samples[r]) == min(int(math.floor(u * expect)), expect - 1)


def test_adv_fewer_than_k_nonzero_entries():
    _require_supported_device()
    row = np.zeros(128256, np.float32)
    row[1000], row[2000], row[3000] = 0.5, 0.25, 0.25
    probs = _rows(row)
    for p in (1.0, 0.9, 0.5, 0.75):
        run, _ = _run_and_check(probs, 10, p, 3, 7)
        assert run.idx[0, :10].tolist() == [1000, 2000, 3000, 0, 1, 2, 3, 4, 5, 6]
        assert run.vals[0, :10].tolist() == [0.5, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0]
        assert not run.renorm[0, 3:10].any()
    run, _ = _run_and_check(probs, 10, 1.0, 3, 7)
    assert run.renorm[0, :3].tolist() == [0.5, 0.25, 0.25]
    u = float(_curand_uniform(3, 0, 7))
    assert int(run.samples[0]) == (1000 if u < 0.5 else (2000 if u < 0.75 else 3000))
    run, _ = _run_and_check(probs, 10, 0.5, 3, 7)
    assert (
        _kept(run, 0) == [1000]
        and run.renorm[0, 0] == 1.0
        and int(run.samples[0]) == 1000
    )


def test_adv_all_zero_rows():
    _require_supported_device()
    for vocab in (4096, 32003, 128256):
        run, _ = _run_and_check(
            _rows(np.zeros(vocab, np.float32), np.zeros(vocab, np.float32)),
            50,
            0.9,
            1,
            1,
        )
        for r in range(2):
            assert run.idx[r, :50].tolist() == list(range(50))
            assert (
                not run.vals[r, :50].any()
                and not run.renorm[r, :50].any()
                and int(run.samples[r]) == 0
            )


def test_adv_nan_rows():
    _require_supported_device()
    vocab = 32768
    base = _probs(1, vocab, seed=5)[0].cpu().numpy()
    poisoned = base.copy()
    nan_at = np.argsort(-base)[:10]
    poisoned[nan_at] = np.nan
    poisoned[[17, 4242]] = np.nan
    probs = _rows(np.full(vocab, np.nan, np.float32), poisoned, base)
    run, _ = _run_and_check(probs, 50, 0.9, 12, 2)
    assert (
        run.idx[0, :50].tolist() == list(range(50))
        and int(run.samples[0]) == 0
        and not run.renorm[0, :50].any()
    )
    nan_set = set(nan_at.tolist()) | {17, 4242}
    assert (
        not (set(run.idx[1, :50].tolist()) & nan_set)
        and int(run.samples[1]) not in nan_set
    )
    expected = [
        int(i) for i in np.argsort(-base, kind="stable") if int(i) not in nan_set
    ][:50]
    assert sorted(run.idx[1, :50].tolist()) == sorted(expected)


def test_adv_inf_rows():
    _require_supported_device()
    vocab = 32768
    base = _probs(1, vocab, seed=6)[0].cpu().numpy()
    three, one, many = base.copy(), base.copy(), base.copy()
    three[[4000, 5, 12345]] = np.inf
    one[31000] = np.inf
    many_at = np.arange(64) * 500 + 3
    many[many_at] = np.inf
    probs = _rows(three, one, many)
    for p in (0.1, 1.0):
        run, _ = _run_and_check(probs, 50, p, 21, 3)
        assert run.idx[0, :3].tolist() == [5, 4000, 12345] and _kept(run, 0) == [
            5,
            4000,
            12345,
        ]
        assert (
            np.all(run.renorm[0, :3] == np.float32(np.float64(1) / 3))
            and not run.renorm[0, 3:50].any()
        )
        u = float(_curand_uniform(21, 0, 3))
        assert int(run.samples[0]) == [5, 4000, 12345][min(int(math.floor(u * 3)), 2)]
        assert (
            _kept(run, 1) == [31000]
            and run.renorm[1, 0] == 1.0
            and int(run.samples[1]) == 31000
        )
        assert run.idx[2, :50].tolist() == many_at[:50].tolist() and np.all(
            run.renorm[2, :50] == np.float32(0.02)
        )
        assert np.isfinite(run.renorm[:, :50]).all()


def test_adv_mixed_nan_inf_negative_rows():
    _require_supported_device()
    row = np.zeros(4096, np.float32)
    row[10], row[20], row[30], row[40] = np.nan, -np.inf, -0.5, -0.0
    row[50], row[60], row[70] = 0.3, 0.7, np.inf
    no_inf = row.copy()
    no_inf[70] = 0.0
    run, _ = _run_and_check(_rows(row, no_inf), 5, 0.9, 8, 5)
    assert (
        run.idx[0, :5].tolist() == [70, 60, 50, 0, 1]
        and _kept(run, 0) == [70]
        and int(run.samples[0]) == 70
    )
    assert run.idx[1, :5].tolist() == [60, 50, 0, 1, 2] and _kept(run, 1) == [60, 50]
    assert np.all(np.isfinite(run.vals[1, :5])) and np.all(run.vals[1, :5] >= 0)


def test_adv_denormal_probabilities():
    _require_supported_device()
    den = np.zeros(4096, np.float32)
    for m in range(1, 21):
        den[100 * m] = np.float32(m) * np.float32(2.0**-149)
    mixed = den.copy()
    mixed[7] = 0.5
    run, _ = _run_and_check(_rows(den, mixed), 10, 1.0, 4, 9)
    assert run.idx[0, :10].tolist() == [100 * m for m in range(20, 10, -1)]
    expect = np.array([m / 155 for m in range(20, 10, -1)])
    assert (
        np.isfinite(run.renorm[0, :10]).all()
        and np.abs(run.renorm[0, :10] / expect - 1).max() < 1e-6
    )
    assert (
        run.idx[1, 0] == 7
        and _kept(run, 1) == [7]
        and run.renorm[1, 0] == 1.0
        and int(run.samples[1]) == 7
    )


def test_adv_one_probability_near_one():
    _require_supported_device()
    row = np.zeros(32768, np.float32)
    row[777], row[778] = np.float32(1 - 2**-24), np.float32(2**-24)
    probs = _rows(row)
    run, _ = _run_and_check(probs, 10, 0.5, 2, 1)
    assert (
        _kept(run, 0) == [777]
        and run.renorm[0, 0] == 1.0
        and int(run.samples[0]) == 777
    )
    run, _ = _run_and_check(probs, 10, 1.0, 2, 1)
    assert (
        _kept(run, 0) == [777, 778]
        and abs(float(run.renorm[0, 0]) + float(run.renorm[0, 1]) - 1.0) <= 2**-23
    )


def test_adv_extreme_p_values():
    _require_supported_device()
    probs = _probs(4, 128256, seed=13)
    pn = probs.cpu().numpy()
    for p in (1e-30, 1e-45, float(np.finfo(np.float32).tiny)):
        run, _ = _run_and_check(probs, 50, p, 31, 2)
        for r in range(4):
            am = int(np.argmax(pn[r]))
            assert (
                int(run.samples[r]) == am
                and _kept(run, r) == [am]
                and run.renorm[r, 0] == 1.0
            )
    run, _ = _run_and_check(probs, 50, 1.0, 32, 3)
    for r in range(4):
        assert _kept(run, r) == run.idx[r, :50].tolist()
        assert abs(float(run.renorm[r, :50].astype(np.float64).sum()) - 1.0) < 1e-5
    # Out-of-range scalar p is rejected by the binding (per-row tensors are clamped in-kernel).
    with pytest.raises((ValueError, RuntimeError, Exception)):  # noqa: B017 - tvm-ffi error type varies
        top_k_top_p_sampling_from_probs(probs, 50, 0.0, philox_seed=1, philox_offset=0)


def test_adv_k_one_and_k_vocab_minus_one():
    _require_supported_device()
    probs = _probs(3, 128256, seed=15)
    pn = probs.cpu().numpy()
    run, _ = _run_and_check(probs, 1, 0.9, 41, 0)
    for r in range(3):
        assert (
            run.idx[r, 0] == int(np.argmax(pn[r]))
            and run.renorm[r, 0] == 1.0
            and int(run.samples[r]) == run.idx[r, 0]
        )
    vocab = 1025
    row = np.random.default_rng(7).random(vocab).astype(np.float32)
    row /= row.sum()
    row[[3, 900, 1010]] = np.float32(1e-9)
    run, _ = _run_and_check(_rows(row), vocab - 1, 0.9, 41, 0)
    assert 1010 not in run.idx[0, :1024].tolist() and {3, 900} <= set(
        run.idx[0, :1024].tolist()
    )


def test_adv_k_at_slab_cap_with_ties():
    _require_supported_device()
    vocab = 128256
    row = np.zeros(vocab, np.float32)
    big = np.arange(24) * 5000 + 11
    row[big] = np.float32(1e-3) * (np.arange(24) + 1)
    tied = np.setdiff1d((np.arange(2000) * 61 + 7) % vocab, big)
    row[tied] = np.float32(2**-20)
    run, _ = _run_and_check(_rows(row), 1024, 0.999, 51, 6)
    idx = run.idx[0, :1024].tolist()
    assert idx[:24] == sorted(big.tolist(), key=lambda i: -row[i])
    assert idx[24:] == sorted(tied.tolist())[: 1024 - 24]


@pytest.mark.parametrize("vocab", [32003, 100003, 128256, 151936])
def test_adv_vocab_not_multiple_of_chunk(vocab):
    _require_supported_device()
    probs = _probs(2, vocab, seed=vocab)
    pn = probs.cpu().numpy()
    pn[1] = 0
    pn[1, vocab - 1] = 1.0
    probs.copy_(torch.tensor(pn, device="cuda"))
    variants = [
        (v["cluster"], v["ept"])
        for v in load_manifest()["stage1"]
        if 512 * v["cluster"] * v["ept"] >= vocab
    ][:4]
    first = None
    for v in variants:
        run, _ = _run_and_check(probs, 50, 0.9, 61, 1, variant=(v, (128, 2)))
        assert run.idx[1, 0] == vocab - 1 and int(run.samples[1]) == vocab - 1
        assert run.idx[1, 1:50].tolist() == list(range(49))
        if first is None:
            first = run
        else:
            assert np.array_equal(first.idx[:, :50], run.idx[:, :50])
            assert np.array_equal(first.samples, run.samples)


def test_adv_philox_offset_not_multiple_of_four():
    _require_supported_device()
    probs = _probs(5, 32768, seed=16)
    seen = set()
    for offset in (0, 1, 2, 3, 5, 6, 7, 4097, (1 << 32) + 3):
        run, _ = _run_and_check(probs, 50, 0.95, 0x5EED, offset)
        seen.add(tuple(run.samples.tolist()))
    assert len(seen) > 1


def test_adv_identical_rows_identical_results():
    _require_supported_device()
    vocab = 128256
    content = _probs(1, vocab, seed=17)[0]
    a = torch.stack([content, content, _probs(1, vocab, seed=18)[0], content])
    b = torch.stack([_probs(1, vocab, seed=19)[0], content, content, content])
    ra, _ = _run_and_check(a, 50, 0.9, 71, 2)
    rb, _ = _run_and_check(b, 50, 0.9, 71, 2)
    for r in (1, 3):
        assert np.array_equal(ra.idx[r, :50], rb.idx[r, :50]) and np.array_equal(
            ra.renorm[r, :50], rb.renorm[r, :50]
        )
        assert int(ra.samples[r]) == int(rb.samples[r])
    assert np.array_equal(ra.idx[0, :50], ra.idx[1, :50]) and np.array_equal(
        ra.renorm[0, :50], ra.renorm[1, :50]
    )


def test_adv_bitwise_across_launch_graph_and_every_variant():
    _require_supported_device()
    vocab, k, p = 32768, 200, 0.9
    probs = _probs(4, vocab, seed=23)
    pn = probs.cpu().numpy()
    pn[2, (np.arange(500) * 17) % vocab] = np.float32(2**-12)
    pn[3, [1, 999]] = np.inf
    probs.copy_(torch.tensor(pn, device="cuda"))
    base, _ = _run_and_check(probs, k, p, 0xFEED, 5)
    for _ in range(3):
        again = _run(probs, k, p, 0xFEED, 5)
        assert np.array_equal(again.samples, base.samples) and np.array_equal(
            again.idx[:, :k], base.idx[:, :k]
        )
    man = load_manifest()
    s1 = [
        (v["cluster"], v["ept"])
        for v in man["stage1"]
        if 512 * v["cluster"] * v["ept"] >= vocab
    ]
    s23 = [
        (v["threads"], v["items"])
        for v in man["stage23"]
        if v["threads"] * v["items"] >= k
    ]
    assert len(s1) >= 4 and len(s23) >= 2
    for v1 in s1:
        for v23 in s23:
            for pdl in (True, False):
                run = _run(probs, k, p, 0xFEED, 5, variant=(v1, v23), pdl=pdl)
                assert np.array_equal(run.samples, base.samples), (v1, v23, pdl)
                assert np.array_equal(run.count, base.count) and np.array_equal(
                    run.idx[:, :k], base.idx[:, :k]
                ), (v1, v23)
                assert np.array_equal(
                    run.vals[:, :k].view(np.uint32), base.vals[:, :k].view(np.uint32)
                ), (v1, v23)
                assert np.array_equal(
                    run.renorm[:, :k].view(np.uint32),
                    base.renorm[:, :k].view(np.uint32),
                ), (v1, v23)
    out = torch.empty(4, device="cuda", dtype=torch.int32)
    renorm = torch.empty(4, SLAB, device="cuda")
    ws = _ws(4)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        top_k_top_p_sampling_from_probs(
            probs,
            k,
            p,
            philox_seed=0xFEED,
            philox_offset=5,
            out=out,
            renorm_out=renorm,
            workspace=ws,
        )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        top_k_top_p_sampling_from_probs(
            probs,
            k,
            p,
            philox_seed=0xFEED,
            philox_offset=5,
            out=out,
            renorm_out=renorm,
            workspace=ws,
        )
    for _ in range(3):
        out.zero_()
        renorm.fill_(-1.0)
        g.replay()
        torch.cuda.synchronize()
        assert np.array_equal(out.cpu().numpy(), base.samples)
        assert np.array_equal(
            renorm[:, :k].cpu().numpy().view(np.uint32),
            base.renorm[:, :k].view(np.uint32),
        )
        assert np.array_equal(ws[1][:, :k].cpu().numpy(), base.idx[:, :k])


def test_adv_stage1_slab_is_deterministic_and_exact():
    _require_supported_device()
    vocab = 128256
    probs = _probs(3, vocab, seed=29)
    pn = probs.cpu().numpy()
    pn[1, (np.arange(300) * 401 + 5) % vocab] = np.float32(2**-11)  # boundary ties
    pn[2, [8, 88, 888]] = np.nan
    probs.copy_(torch.tensor(pn, device="cuda"))
    vals, idxs, cnt = top_k_probs_to_slab(probs, 100)
    torch.cuda.synchronize()
    for _ in range(3):
        v2, i2, c2 = top_k_probs_to_slab(probs, 100)
        torch.cuda.synchronize()
        assert (
            torch.equal(i2[:, :100], idxs[:, :100])
            and torch.equal(v2[:, :100], vals[:, :100])
            and torch.equal(c2, cnt)
        )
    for r in range(3):
        expect = _reference(pn[r], 100, 1.0, np.float32(0.5)).idx
        got = idxs[r, :100].cpu().numpy()
        assert sorted(got.tolist()) == sorted(expect.tolist())
        assert np.array_equal(
            vals[r, :100].cpu().numpy().view(np.uint32), _keys(pn[r])[got]
        )
    assert not ({8, 88, 888} & set(idxs[2, :100].cpu().numpy().tolist()))


def test_adv_top_k_first_parity():
    """Support parity with the ``top_k_first`` route: the top-k set equals ``top_k_renorm_probs``'s support
    exactly, the top-p kept set equals ``top_p_renorm_probs`` (on the renormalized top-k) up to the single
    float32 boundary element, and every ``top_k_first`` draw on a shared generator falls inside that
    support.  Per-draw equality is not expected: that route's top-p sampler is a rejection sampler that
    consumes the Philox stream differently from the pipeline's inverse CDF."""
    _require_supported_device()
    from flashinfer.sampling import (
        get_seed_and_offset,
        top_k_renorm_probs,
        top_p_renorm_probs,
    )
    from flashinfer.sampling import top_k_top_p_sampling_from_probs as reference

    batch, vocab, k, p = 32, 128256, 50, 0.9
    probs = _probs(batch, vocab, seed=24, scale=2.0)
    run, _ = _run_and_check(probs, k, p, 0xBEEF, 0)
    topk_ren = top_k_renorm_probs(probs, k).cpu().numpy()
    topp_ren = top_p_renorm_probs(top_k_renorm_probs(probs, k), p).cpu().numpy()
    boundary_diffs = 0
    for r in range(batch):
        assert sorted(np.nonzero(topk_ren[r])[0].tolist()) == sorted(
            run.idx[r, :k].tolist()
        )
        ours = _kept(run, r)
        diff = set(np.nonzero(topp_ren[r])[0].tolist()) ^ set(ours)
        if diff:
            boundary_diffs += 1
            nxt = run.idx[r, len(ours)] if len(ours) < k else None
            assert diff <= {ours[-1], nxt}, (
                f"row {r}: kept sets differ beyond the boundary: {diff}"
            )
    assert boundary_diffs <= 2
    g1 = torch.Generator(device="cuda").manual_seed(0xBEEF)
    g2 = torch.Generator(device="cuda").manual_seed(0xBEEF)
    for _ in range(4):
        fi = (
            reference(
                probs,
                k,
                p,
                filter_apply_order="top_k_first",
                deterministic=True,
                generator=g1,
            )
            .cpu()
            .numpy()
        )
        seed, offset = get_seed_and_offset(batch * 32, g2, probs.device)
        ours_run, _ = _run_and_check(probs, k, p, int(seed), int(offset))
        for r in range(batch):
            support = set(_kept(ours_run, r)) | set(np.nonzero(topp_ren[r])[0].tolist())
            assert int(fi[r]) in support and int(ours_run.samples[r]) in support
    assert torch.equal(g1.get_state(), g2.get_state())
