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

Correctness tests for the Cake fused QK RMSNorm + NeoX RoPE + FP8 quantize +
paged KV append kernel.

Every case is checked against an independent fp32 reference (per-head RMSNorm,
NeoX RoPE, ``amax / upper_max`` or static Q scale, ``x / scale`` K/V payloads,
last-page tail clear, device-side ``q_indptr`` validation).  FP8 payloads are
compared against the *unrounded* fp32 payload with ``atol = rtol = 0.1``;
``q_scale`` with ``rtol = 1e-5``; flags, cleared tails and every untouched byte
bitwise.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_fused_qk_rope_fp8_append`` facade is covered by
``test_cake_fused_qk_rope_fp8_append_api.py``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_fused_qk_rope_fp8_append")
cake_build = cake_package("cake_fused_qk_rope_fp8_append", "cake_build")
loader = cake_package("cake_fused_qk_rope_fp8_append", "cake_jit")
fused_fp8 = pkg.cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache

HEAD_DIM = 128
FP8_MAX = 448.0
AMAX_FLOOR = 1e-6
EPS = 1e-6
POISON_BYTE = 0xFF  # NaN in float8_e4m3fn: never produced by the kernel
PACKAGE_DIR = Path(pkg.__file__).resolve().parent
GRID_RULES = (
    "rows_plus_requests_by_kv_heads",
    "rows_by_kv_heads",
    "kv_heads_by_rows",
    "kv_heads_by_rows_plus_requests",
)
FOLD_GRID_RULES = ("rows_by_kv_heads", "kv_heads_by_rows")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _arch():
    if not torch.cuda.is_available():
        return None
    return loader.arch_for_capability(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return _arch() is not None


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if _arch() is None:
        pytest.skip(
            "cake_fused_qk_rope_fp8_append ships no build for compute capability "
            f"{torch.cuda.get_device_capability(_device())}"
        )


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


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


def _rmsnorm(x, w):
    inv = torch.rsqrt(x.square().sum(dim=-1, keepdim=True) / x.shape[-1] + EPS)
    return x * inv * w.float()


def _rope(x, cos_sin, positions):
    half = HEAD_DIM // 2
    cos = cos_sin[positions, :half].float().unsqueeze(1)
    sin = cos_sin[positions, half:].float().unsqueeze(1)
    left, right = x[..., :half], x[..., half:]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)


def reference(case, *, caller_owned_kv=False):
    """fp32 reference; returns the expected unrounded payloads in output layout.

    ``key_cache`` / ``value_cache`` expectations are fp32 tensors of the cache shape
    holding NaN where the kernel must not write, 0 on the cleared tails and the
    unrounded ``x / scale`` payload on the appended rows.
    """
    qkv, dev = case["qkv"], case["qkv"].device
    T = qkv.shape[0]
    hq, hkv = case["num_q_heads"], case["num_kv_heads"]
    B = case["seq_lens"].shape[0]
    page_size = case["page_size"]
    qi = case["q_indptr"].tolist()
    sl = case["seq_lens"].tolist()
    pi = case["page_indices"].cpu()
    quant, norm = case["quant_policy"], case["qk_norm_policy"]
    dyn_prefill = quant == 1 and case["is_prefill"]
    aligned = (case["max_seqlen"] + 127) // 128 * 128 if dyn_prefill else 0

    valid = qi[0] == 0 and qi[-1] == T
    lengths = [qi[b + 1] - qi[b] for b in range(B)]
    valid = valid and all(n >= 0 for n in lengths)
    if dyn_prefill:
        valid = valid and all(n <= case["max_seqlen"] for n in lengths)
    flags = torch.full((B, hkv), 0 if valid else -1, dtype=torch.int32, device=dev)
    nan_cache = torch.full(
        case["cache_shape"], float("nan"), dtype=torch.float32, device=dev
    )
    exp = dict(
        valid=valid,
        flags=flags,
        out_q=None,
        q_scale=None,
        key_cache=nan_cache.clone(),
        value_cache=nan_cache.clone(),
        out_k=None,
        out_v=None,
    )
    if not valid:
        return exp

    batch_ids = torch.empty(T, dtype=torch.int64)
    positions = torch.empty(T, dtype=torch.int64)
    for b in range(B):
        lo, hi = qi[b], qi[b + 1]
        if hi > lo:
            batch_ids[lo:hi] = b
            positions[lo:hi] = torch.arange(lo, hi) + sl[b] - hi
    assert T == 0 or bool((positions >= 0).all()), (
        "test cases keep every position valid"
    )

    x = qkv.float()
    q = x[:, : hq * HEAD_DIM].reshape(T, hq, HEAD_DIM)
    k = x[:, hq * HEAD_DIM : (hq + hkv) * HEAD_DIM].reshape(T, hkv, HEAD_DIM)
    v = x[:, (hq + hkv) * HEAD_DIM :].reshape(T, hkv, HEAD_DIM)
    qw, kw = case["q_norm_weight"], case["k_norm_weight"]
    if norm == 2:
        q, k = _rmsnorm(q, qw), _rmsnorm(k, kw)
    if T > 0:
        pos_dev = positions.to(dev)
        q, k = _rope(q, case["cos_sin"], pos_dev), _rope(k, case["cos_sin"], pos_dev)
    if norm == 1:
        q, k = _rmsnorm(q, qw), _rmsnorm(k, kw)
    if quant == 1:
        scale = torch.clamp(q.abs().amax(dim=-1), min=AMAX_FLOOR) / case["upper_max"]
        q_payload = q / scale.unsqueeze(-1)
        if dyn_prefill:
            q_scale = torch.full((B, hq, aligned), float("nan"), device=dev)
            for b in range(B):
                lo, hi = qi[b], qi[b + 1]
                if hi > lo:
                    q_scale[b, :, : hi - lo] = scale[lo:hi].t()
        else:
            q_scale = scale
    else:
        q_payload = q * case["q_scale_inv"].float().reshape(())
        q_scale = torch.empty(0, device=dev)
    k_payload = k / case["k_scale"].float().reshape(())
    v_payload = v / case["v_scale"].float().reshape(())
    exp["out_q"], exp["q_scale"] = q_payload, q_scale
    if caller_owned_kv:
        exp["out_k"], exp["out_v"] = k_payload, v_payload
    elif T > 0:
        phys = pi[batch_ids, positions // page_size].to(dev)
        slot = (positions % page_size).to(dev)
        exp["key_cache"][phys, slot] = k_payload
        exp["value_cache"][phys, slot] = v_payload
    for b in range(B):
        last = sl[b] - 1
        if last < 0:
            continue
        page = int(pi[b, last // page_size])
        zero_from = last % page_size + 1
        if zero_from < page_size:
            exp["key_cache"][page, zero_from:] = 0.0
            exp["value_cache"][page, zero_from:] = 0.0
    return exp


def make_case(
    *,
    q_lens,
    ctx_lens,
    num_q_heads,
    num_kv_heads,
    page_size,
    qk_norm_policy,
    quant_policy,
    seed,
    is_prefill=None,
    max_seqlen=None,
    upper_max=FP8_MAX,
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
    q_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16).float()
    k_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16).float()
    if is_prefill is None:
        is_prefill = max(q_lens) > 1
    if max_seqlen is None:
        max_seqlen = max(q_lens) if quant_policy == 1 and is_prefill else 0
    cache_shape = (total_pages, page_size, num_kv_heads, HEAD_DIM)
    return dict(
        qkv=qkv,
        cos_sin=rotary_cos_sin_table(max(seq_lens) + 1, device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr.to(device),
        page_indices=page_indices.to(device),
        cache_shape=cache_shape,
        page_size=page_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qk_norm_policy=qk_norm_policy,
        quant_policy=quant_policy,
        is_prefill=is_prefill,
        max_seqlen=max_seqlen,
        upper_max=upper_max,
        q_norm_weight=q_w.to(device),
        k_norm_weight=k_w.to(device),
        k_scale=torch.tensor([0.02], dtype=torch.float32, device=device),
        v_scale=torch.tensor([0.03], dtype=torch.float32, device=device),
        q_scale_inv=torch.tensor([1.0 / 0.02], dtype=torch.float32, device=device),
    )


def _poisoned_fp8(shape, device):
    t = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    t.view(torch.uint8).fill_(POISON_BYTE)
    return t


def _q_scale_buffer(case):
    hq, B, T = case["num_q_heads"], case["seq_lens"].shape[0], case["qkv"].shape[0]
    dev = case["qkv"].device
    if case["quant_policy"] == 2:
        return torch.empty(0, dtype=torch.float32, device=dev)
    if case["is_prefill"]:
        aligned = (case["max_seqlen"] + 127) // 128 * 128
        return torch.full((B, hq, aligned), float("nan"), device=dev)
    return torch.full((T, hq), float("nan"), device=dev)


def launch(case, *, out_k=None, out_v=None, q_indptr=None, max_seqlen=None):
    """Run the fused entry on poisoned caller-owned buffers; return (buffers, returned tuple)."""
    dev = case["qkv"].device
    T, hq, hkv = case["qkv"].shape[0], case["num_q_heads"], case["num_kv_heads"]
    bufs = dict(
        key_cache=_poisoned_fp8(case["cache_shape"], dev),
        value_cache=_poisoned_fp8(case["cache_shape"], dev),
        out_q=_poisoned_fp8((T, hq, HEAD_DIM), dev),
        q_scale=_q_scale_buffer(case),
        flags=torch.full(
            (case["seq_lens"].shape[0], hkv), 7, dtype=torch.int32, device=dev
        ),
    )
    norm = case["qk_norm_policy"]
    ret = fused_fp8(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"] if q_indptr is None else q_indptr,
        case["page_indices"],
        (bufs["key_cache"], bufs["value_cache"]),
        case["is_prefill"],
        case["k_scale"],
        case["v_scale"],
        case["quant_policy"],
        max_seqlen=case["max_seqlen"] if max_seqlen is None else max_seqlen,
        upper_max=case["upper_max"],
        q_scale_inv=case["q_scale_inv"] if case["quant_policy"] == 2 else None,
        q_norm_weight=case["q_norm_weight"] if norm else None,
        k_norm_weight=case["k_norm_weight"] if norm else None,
        qk_norm_policy=norm,
        out_q=bufs["out_q"],
        out_k=out_k,
        out_v=out_v,
        q_scale=bufs["q_scale"],
        split_k_flag=bufs["flags"],
    )
    torch.cuda.synchronize()
    return bufs, ret


def _assert_fp8_matches(got_fp8, expected_f32, label):
    """Written elements within the FP8 tolerance; NaN in ``expected`` means untouched poison."""
    written = ~torch.isnan(expected_f32)
    torch.testing.assert_close(
        got_fp8.float()[written],
        expected_f32[written],
        atol=0.1,
        rtol=0.1,
        msg=lambda m: f"{label}: {m}",
    )
    untouched = got_fp8.view(torch.uint8)[~written]
    assert bool((untouched == POISON_BYTE).all()), f"{label}: wrote outside its range"


def run_and_compare(case, *, caller_owned_kv=False):
    dev = case["qkv"].device
    T, hkv = case["qkv"].shape[0], case["num_kv_heads"]
    ok = ov = None
    if caller_owned_kv:
        ok = _poisoned_fp8((T, hkv, HEAD_DIM), dev)
        ov = _poisoned_fp8((T, hkv, HEAD_DIM), dev)
    bufs, ret = launch(case, out_k=ok, out_v=ov)
    assert ret[0] is bufs["out_q"] and ret[1] is bufs["q_scale"]
    assert ret[2] is bufs["flags"]
    exp = reference(case, caller_owned_kv=caller_owned_kv)
    assert exp["valid"]
    assert torch.equal(bufs["flags"], exp["flags"])
    _assert_fp8_matches(bufs["out_q"], exp["out_q"], "out_q")
    if case["quant_policy"] == 1:
        valid = ~torch.isnan(exp["q_scale"])
        torch.testing.assert_close(
            bufs["q_scale"][valid], exp["q_scale"][valid], rtol=1e-5, atol=1e-6
        )
        # the padding of the dynamic-prefill layout is never written
        assert bool(torch.isnan(bufs["q_scale"][~valid]).all())
    else:
        assert bufs["q_scale"].numel() == 0
    _assert_fp8_matches(bufs["key_cache"], exp["key_cache"], "key_cache")
    _assert_fp8_matches(bufs["value_cache"], exp["value_cache"], "value_cache")
    if caller_owned_kv:
        _assert_fp8_matches(ok, exp["out_k"], "out_k")
        _assert_fp8_matches(ov, exp["out_v"], "out_v")
    return bufs


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("quant_policy", [1, 2])
@pytest.mark.parametrize("page_size", [16, 64])
def test_decode_batch(heads, qk_norm_policy, quant_policy, page_size):
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 6,
        ctx_lens=[0, 5, 63, 64, 130, 2047],
        num_q_heads=heads[0],
        num_kv_heads=heads[1],
        page_size=page_size,
        qk_norm_policy=qk_norm_policy,
        quant_policy=quant_policy,
        seed=893 + page_size,
    )
    run_and_compare(case)


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("quant_policy", [1, 2])
def test_prefill_chunks_cross_pages(heads, qk_norm_policy, quant_policy):
    _skip_unless_supported()
    case = make_case(
        q_lens=[16, 0, 7, 70],
        ctx_lens=[60, 3, 0, 125],
        num_q_heads=heads[0],
        num_kv_heads=heads[1],
        page_size=64,
        qk_norm_policy=qk_norm_policy,
        quant_policy=quant_policy,
        seed=71,
    )
    run_and_compare(case)


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("q_len", [127, 128, 129])
def test_dynamic_prefill_scale_padding_boundary(heads, q_len):
    """``q_scale`` padding (``round_up(max_seqlen, 128)``) stays untouched at the boundary."""
    _skip_unless_supported()
    case = make_case(
        q_lens=[q_len] * 3,
        ctx_lens=[0, 64, 1000],
        num_q_heads=heads[0],
        num_kv_heads=heads[1],
        page_size=64,
        qk_norm_policy=2,
        quant_policy=1,
        seed=q_len,
        max_seqlen=q_len,
    )
    run_and_compare(case)


@pytest.mark.parametrize("quant_policy", [1, 2])
def test_ragged_with_empty_requests_and_caller_owned_kv(quant_policy):
    _skip_unless_supported()
    case = make_case(
        q_lens=[3, 0, 2, 0],
        ctx_lens=[2, 5, 2, 0],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=4,
        qk_norm_policy=2,
        quant_policy=quant_policy,
        seed=5,
    )
    run_and_compare(case, caller_owned_kv=True)
    run_and_compare(case, caller_owned_kv=False)


def test_small_upper_max_and_small_page():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1, 1, 1],
        ctx_lens=[7, 8, 9],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=4,
        qk_norm_policy=1,
        quant_policy=1,
        seed=11,
        upper_max=240.0,
    )
    run_and_compare(case)


@pytest.mark.parametrize(
    "kind", ["first_nonzero", "non_monotone", "last_mismatch", "overflow_max_seqlen"]
)
def test_invalid_metadata_leaves_outputs_untouched(kind):
    _skip_unless_supported()
    case = make_case(
        q_lens=[4, 1, 3],
        ctx_lens=[1, 3, 60],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=16,
        qk_norm_policy=2,
        quant_policy=1,
        seed=9,
        max_seqlen=4,
    )
    dev = case["qkv"].device
    q_indptr, max_seqlen = case["q_indptr"], None
    if kind == "first_nonzero":
        q_indptr = torch.tensor([1, 4, 5, 8], dtype=torch.int32, device=dev)
    elif kind == "non_monotone":
        q_indptr = torch.tensor([0, 5, 4, 8], dtype=torch.int32, device=dev)
    elif kind == "last_mismatch":
        q_indptr = torch.tensor([0, 4, 5, 9], dtype=torch.int32, device=dev)
    else:
        max_seqlen = 3  # request 0 has 4 rows
    bufs, _ = launch(case, q_indptr=q_indptr, max_seqlen=max_seqlen)
    assert bool((bufs["flags"] == -1).all())
    for name in ("out_q", "key_cache", "value_cache"):
        assert bool((bufs[name].view(torch.uint8) == POISON_BYTE).all()), name
    assert bool(torch.isnan(bufs["q_scale"]).all())


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_empty_batch_with_rows_leaves_outputs_untouched(heads, is_prefill):
    """No request (B = 0) with rows is a malformed table (q_indptr = [0]): the call
    writes nothing and has no flag to set."""
    _skip_unless_supported()
    hq, hkv = heads
    case = make_case(
        q_lens=[1],
        ctx_lens=[5],
        num_q_heads=hq,
        num_kv_heads=hkv,
        page_size=16,
        qk_norm_policy=2,
        quant_policy=1,
        seed=13,
        is_prefill=is_prefill,
    )
    dev = case["qkv"].device
    g = torch.Generator(device="cpu").manual_seed(14)
    width = (hq + 2 * hkv) * HEAD_DIM
    case.update(
        qkv=torch.randn((8, width), generator=g).to(torch.bfloat16).to(dev),
        seq_lens=torch.zeros(0, dtype=torch.int32, device=dev),
        q_indptr=torch.zeros(1, dtype=torch.int32, device=dev),
        page_indices=torch.zeros((0, 1), dtype=torch.int32, device=dev),
        max_seqlen=8 if is_prefill else 0,
    )
    bufs, _ = launch(case)
    assert tuple(bufs["flags"].shape) == (0, hkv)
    for name in ("out_q", "key_cache", "value_cache"):
        assert bool((bufs[name].view(torch.uint8) == POISON_BYTE).all()), name
    assert bool(torch.isnan(bufs["q_scale"]).all())


def test_cuda_graph_replay_matches_eager():
    _skip_unless_supported()
    case = make_case(
        q_lens=[16] * 4,
        ctx_lens=[2048, 100, 0, 63],
        num_q_heads=64,
        num_kv_heads=8,
        page_size=64,
        qk_norm_policy=2,
        quant_policy=1,
        seed=897,
    )
    dev = case["qkv"].device
    T = case["qkv"].shape[0]
    kc = _poisoned_fp8(case["cache_shape"], dev)
    vc = _poisoned_fp8(case["cache_shape"], dev)
    oq = _poisoned_fp8((T, 64, HEAD_DIM), dev)
    qs = _q_scale_buffer(case)
    flags = torch.full((4, 8), 7, dtype=torch.int32, device=dev)

    def run():
        fused_fp8(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            (kc, vc),
            True,
            case["k_scale"],
            case["v_scale"],
            1,
            max_seqlen=case["max_seqlen"],
            q_norm_weight=case["q_norm_weight"],
            k_norm_weight=case["k_norm_weight"],
            qk_norm_policy=2,
            out_q=oq,
            q_scale=qs,
            split_k_flag=flags,
        )

    def poison():
        for t in (kc, vc, oq):
            t.view(torch.uint8).fill_(POISON_BYTE)
        qs.fill_(float("nan"))
        flags.fill_(7)

    def snapshot():
        torch.cuda.synchronize()
        return [
            t.view(torch.uint8).clone() if t.dtype == torch.float8_e4m3fn else t.clone()
            for t in (oq, qs, flags, kc, vc)
        ]

    poison()
    run()
    eager = snapshot()
    assert bool((flags == 0).all())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # warm the JIT module / allocator on the capture stream
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        poison()
        graph.replay()
        for got, want in zip(snapshot(), eager, strict=True):
            assert torch.equal(
                got.nan_to_num(-1.0) if got.is_floating_point() else got,
                want.nan_to_num(-1.0) if want.is_floating_point() else want,
            )


def test_repeated_launches_and_buffer_reuse_are_bitwise_identical():
    """Three launches of one problem on fresh poisoned buffers write the same
    bytes, and a launch into the already written caller-owned buffers writes
    them again."""
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 8 + [16, 0, 3],
        ctx_lens=[2048 + 5 * i for i in range(8)] + [2047, 5, 0],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        qk_norm_policy=2,
        quant_policy=1,
        seed=898,
        is_prefill=True,
        max_seqlen=16,
    )

    def bytes_of(bufs):
        return [
            t.view(torch.uint8).clone()
            if t.dtype == torch.float8_e4m3fn
            else t.view(torch.int32).clone()
            for t in (
                bufs["out_q"],
                bufs["q_scale"],
                bufs["flags"],
                bufs["key_cache"],
                bufs["value_cache"],
            )
        ]

    snapshots = []
    for _ in range(3):
        bufs, ret = launch(case)
        assert ret[0] is bufs["out_q"]
        snapshots.append(bytes_of(bufs))
    for later in snapshots[1:]:
        for got, want in zip(later, snapshots[0], strict=True):
            assert torch.equal(got, want)
    # reuse: launch again into the buffers of the last launch without re-poisoning
    fused_fp8(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        (bufs["key_cache"], bufs["value_cache"]),
        case["is_prefill"],
        case["k_scale"],
        case["v_scale"],
        case["quant_policy"],
        max_seqlen=case["max_seqlen"],
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
        qk_norm_policy=case["qk_norm_policy"],
        out_q=bufs["out_q"],
        q_scale=bufs["q_scale"],
        split_k_flag=bufs["flags"],
    )
    torch.cuda.synchronize()
    for got, want in zip(bytes_of(bufs), snapshots[0], strict=True):
        assert torch.equal(got, want)


def test_changing_shapes_in_one_process_does_not_rebuild():
    """Shapes the policy maps onto one stage share one loaded library: no
    second build, no second load."""
    _skip_unless_supported()
    arch = _arch()
    num_sms = loader.sm_count(_device().index)

    def pages(requests: int, page_size: int) -> int:
        return max(1, -(-(2048 + requests) // page_size))

    def stage_of(requests: int, page_size: int) -> str:
        return loader.stage_for(
            8,
            1,
            arch=arch,
            num_rows=requests,
            num_requests=requests,
            page_size=page_size,
            num_sms=num_sms,
            is_prefill=False,
            max_pages_per_request=pages(requests, page_size),
            has_out_kv=False,
        )

    def decode_case(requests: int, page_size: int, seed: int):
        return make_case(
            q_lens=[1] * requests,
            ctx_lens=[2048 + i for i in range(requests)],
            num_q_heads=8,
            num_kv_heads=1,
            page_size=page_size,
            qk_norm_policy=2,
            quant_policy=1,
            seed=seed,
            is_prefill=False,
        )

    base = stage_of(32, 64)
    same_stage = [
        (requests, page_size)
        for requests, page_size in ((40, 64), (16, 64), (33, 64), (8, 64), (48, 128))
        if stage_of(requests, page_size) == base
    ]
    assert same_stage, "the policy left no second decode shape on the B32 stage"
    run_and_compare(decode_case(32, 64, 899))
    misses = loader.load.cache_info().misses
    for requests, page_size in same_stage:
        run_and_compare(decode_case(requests, page_size, 900 + requests))
    assert loader.load.cache_info().misses == misses, (
        "a changed shape on the same stage loaded another library"
    )


def test_rejects_bad_arguments():
    _skip_unless_supported()
    case = make_case(
        q_lens=[2, 1],
        ctx_lens=[5, 70],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        qk_norm_policy=2,
        quant_policy=1,
        seed=1,
    )
    dev = case["qkv"].device
    T = case["qkv"].shape[0]
    kc = _poisoned_fp8(case["cache_shape"], dev)
    vc = _poisoned_fp8(case["cache_shape"], dev)

    def call(**over):
        kw = dict(
            max_seqlen=2,
            q_norm_weight=case["q_norm_weight"],
            k_norm_weight=case["k_norm_weight"],
            qk_norm_policy=2,
        )
        pos = dict(
            qkv=case["qkv"],
            cos_sin_cache=case["cos_sin"],
            seq_lens=case["seq_lens"],
            q_indptr=case["q_indptr"],
            page_indices=case["page_indices"],
            paged_kv_cache=(kc, vc),
            is_prefill=True,
            k_scale=case["k_scale"],
            v_scale=case["v_scale"],
            quant_policy=1,
        )
        pos.update({k: v for k, v in over.items() if k in pos})
        kw.update({k: v for k, v in over.items() if k not in pos})
        return fused_fp8(**pos, **kw)

    for upper_max in (0.0, -1.0, 448.01, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            call(upper_max=upper_max)
    with pytest.raises(ValueError):
        call(quant_policy=3)
    with pytest.raises(ValueError):
        call(quant_policy=2)  # q_scale_inv missing
    with pytest.raises(ValueError):
        call(qk_norm_policy=1, q_norm_weight=None)
    with pytest.raises(ValueError):
        call(max_seqlen=0)  # dynamic prefill needs a positive max_seqlen
    with pytest.raises(ValueError):
        call(out_q=_poisoned_fp8((T, 4, HEAD_DIM), dev))
    with pytest.raises(ValueError):
        call(q_scale=torch.empty((T, 8), device=dev))  # decode layout on prefill
    with pytest.raises(ValueError):
        call(split_k_flag=torch.empty((2, 2), dtype=torch.int32, device=dev))
    with pytest.raises(ValueError):
        call(out_k=_poisoned_fp8((T, 1, HEAD_DIM), dev))  # out_v missing
    with pytest.raises(ValueError):
        call(qkv=case["qkv"][:, : 9 * HEAD_DIM].contiguous())  # (7, 1) heads
    # the valid call still works after the rejections
    out_q, q_scale, flags = call()
    torch.cuda.synchronize()
    assert out_q.shape == (T, 8, HEAD_DIM) and q_scale.shape == (2, 8, 128)
    assert bool((flags == 0).all())


# ----------------------------------------------------------------------------------------------
# Registry and build manifest (CPU): every module record names its ``build.json`` library
# and the host plan only; the binding's argument order is one contract for every stage.

_ARG_NAMES = [
    "qkv",
    "cos_sin",
    "seq_lens",
    "q_indptr",
    "page_indices",
    "q_norm_weight",
    "k_norm_weight",
    "k_scale",
    "v_scale",
    "q_scale_inv",
    "out_q",
    "key_cache",
    "value_cache",
    "out_k",
    "out_v",
    "q_scale",
    "split_k_flag",
    "num_rows",
    "num_requests",
    "num_q_heads",
    "num_kv_heads",
    "page_size",
    "max_pages_per_request",
    "max_seqlen",
    "max_seqlen_aligned",
    "quant_policy",
    "norm_policy",
    "is_prefill",
    "has_out_kv",
    "upper_max",
    "grid_x",
    "grid_y",
    "grid_z",
]


def test_every_stage_binds_the_public_argument_plan():
    """The binding's argument order (seventeen buffers, thirteen scalars, the grid)
    is the same for every stage; the host binds it through the record's
    ``logical`` names."""
    for name, record in loader.MODULES.items():
        plan = record["arg_plan"]
        assert [key for _kind, key in plan] == _ARG_NAMES, name
        assert [kind for kind, _key in plan] == (
            ["buffer"] * 17 + ["parameter"] * 13 + ["grid"] * 3
        ), name


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only (stage, knobs, launch record,
    grid rule and the kernel-parameter -> logical-argument map).
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    logical_names = set(loader.LOGICAL_TENSORS) | set(loader.LOGICAL_SCALARS)
    referenced = set()
    assert loader.STAGE in loader.MODULES
    assert list(loader.MODULES[loader.STAGE]["arches"]) == list(loader.ARCHES)
    assert (
        loader.MODULES[loader.STAGE]["grid_rule"]["kind"]
        == "rows_plus_requests_by_kv_heads"
    )
    for name, record in loader.MODULES.items():
        assert record["stage"] == name
        assert name.startswith(loader.STAGE), name
        assert record["library"] == record["kernel"], name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        assert record["ffi_entry"] == "run", name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["knobs"]) == set(loader.KNOB_KEYS), name
        assert tuple(record["launch"]["block"]) == (loader.THREADS, 1, 1), name
        kind = record["grid_rule"]["kind"]
        assert kind in GRID_RULES, name
        assert bool(record["knobs"]["fold"]) == (kind in FOLD_GRID_RULES), name
        assert {key for kind_, key in record["arg_plan"] if kind_ != "grid"} == set(
            record["logical"]
        ), name
        assert {key for kind_, key in record["arg_plan"] if kind_ == "grid"} == {
            "grid_x",
            "grid_y",
            "grid_z",
        }, name
        assert set(record["logical"].values()) <= logical_names, name
        assert record["arches"], name
        assert set(record["arches"]) <= set(loader.ARCHES), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (PACKAGE_DIR / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert not any("use_fast_math" in flag for flag in entry["nvcc_flags"])
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    for arch in loader.ARCHES:
        shipped = {
            name for name, record in loader.MODULES.items() if arch in record["arches"]
        }
        assert set(loader.stages_for(arch)) == shipped
        assert loader.STAGE in shipped, arch


def test_stage_policy_table_is_well_formed():
    policy = loader.STAGE_POLICY
    assert set(policy) == {"constants", "rule", "version"}
    assert isinstance(policy["rule"], str) and isinstance(policy["version"], str)
    assert {"ctas_per_sm", "fold_waves", "fold_min_kv_heads"} <= set(
        policy["constants"]
    )


def test_default_launch_plan_is_the_default_builds_grid():
    """The public ``launch_plan`` describes the default build: one CTA per (row,
    kv head) plus one per (request, kv head)."""
    assert pkg.launch_plan(40, 32, 8) == {"grid": (72, 8, 1)}
    assert pkg.launch_plan(1, 1, 1) == {"grid": (2, 1, 1)}


# ----------------------------------------------------------------------------------------------
# Stage policy (CPU): the host mirrors the Cake policy and must only select shipped builds.
# ``_POLICY_TABLE`` pins the policy's choice at representative problems (rows change when the
# registry's policy changes; the structure does not); the scan covers the problem space.

_POLICY_TABLE: dict[tuple[str, int, int, int, int, int, int, bool, int, bool], str] = {
    # (arch, Hq, Hkv, num_rows, num_requests, page_size, num_sms, is_prefill,
    #  max_pages_per_request, has_out_kv) -> stage
    # no request (B = 0; with rows, a malformed table): the default build everywhere
    ("sm_90a", 8, 1, 8, 0, 64, 132, False, 1, False): "fp8_e4m3",
    ("sm_90a", 8, 1, 8, 0, 64, 132, True, 1, False): "fp8_e4m3",
    ("sm_100a", 8, 1, 8, 0, 64, 148, False, 1, False): "fp8_e4m3",
    ("sm_100a", 8, 1, 8, 0, 64, 148, True, 1, False): "fp8_e4m3",
    ("sm_103a", 8, 1, 8, 0, 64, 148, False, 1, False): "fp8_e4m3",
    ("sm_103a", 8, 1, 8, 0, 64, 148, True, 1, False): "fp8_e4m3",
    ("sm_103a", 8, 1, 8, 0, 64, 152, False, 1, False): "fp8_e4m3",
    ("sm_103a", 8, 1, 8, 0, 64, 152, True, 1, False): "fp8_e4m3",
    # B300 single-KV-head decode from 512 CTAs: the nine-CTA prefetch
    (
        "sm_103a",
        8,
        1,
        256,
        256,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_103a",
        8,
        1,
        224,
        224,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_100a",
        8,
        1,
        256,
        256,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_90a",
        8,
        1,
        32,
        32,
        64,
        132,
        False,
        33,
        False,
    ): "fp8_e4m3_scan_rounds3_clear_tie1",
    (
        "sm_90a",
        64,
        8,
        32,
        32,
        64,
        132,
        False,
        33,
        False,
    ): "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_grid_order2_clear_tie1",
    (
        "sm_90a",
        8,
        1,
        128,
        8,
        64,
        132,
        True,
        33,
        False,
    ): "fp8_e4m3_scan_rounds3_clear_tie1",
    (
        "sm_90a",
        64,
        8,
        128,
        8,
        64,
        132,
        True,
        33,
        False,
    ): "fp8_e4m3_scan_rounds3_clear_tie1",
    (
        "sm_90a",
        64,
        8,
        256,
        256,
        64,
        132,
        False,
        33,
        False,
    ): "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_90a",
        8,
        1,
        1024,
        1,
        64,
        132,
        True,
        48,
        False,
    ): "fp8_e4m3_scan_rounds3_clear_tie1",
    (
        "sm_90a",
        8,
        1,
        32,
        32,
        16,
        132,
        False,
        129,
        False,
    ): "fp8_e4m3_scan_rounds3_clear_tie1",
    (
        "sm_100a",
        8,
        1,
        32,
        32,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_100a",
        64,
        8,
        32,
        32,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_100a",
        8,
        1,
        128,
        8,
        64,
        148,
        True,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    ("sm_100a", 64, 8, 128, 8, 64, 148, True, 33, False): "fp8_e4m3_grid_order2",
    (
        "sm_100a",
        64,
        8,
        256,
        256,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_scan_rounds3_clear_tie1",
    (
        "sm_100a",
        8,
        1,
        32,
        32,
        64,
        148,
        False,
        33,
        True,
    ): "fp8_e4m3_grid_order3_zero_store_width256",
    (
        "sm_100a",
        8,
        1,
        32,
        32,
        16,
        148,
        False,
        129,
        False,
    ): "fp8_e4m3_grid_order3_zero_store_width256",
    (
        "sm_103a",
        8,
        1,
        32,
        32,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_103a",
        64,
        8,
        32,
        32,
        64,
        148,
        False,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_103a",
        8,
        1,
        128,
        8,
        64,
        148,
        True,
        33,
        False,
    ): "fp8_e4m3_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_103a",
        8,
        1,
        32,
        32,
        64,
        152,
        False,
        33,
        False,
    ): "fp8_e4m3_clear_heads_per_cta2_zero_store_width256",
    (
        "sm_103a",
        64,
        8,
        32,
        32,
        64,
        152,
        False,
        33,
        False,
    ): "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_clear_tie1",
    (
        "sm_103a",
        8,
        1,
        128,
        8,
        64,
        152,
        True,
        33,
        False,
    ): "fp8_e4m3_min_blocks9_early_rt21_page_prefetch1_clear_tie1",
    ("sm_103a", 64, 8, 128, 8, 64, 152, True, 33, False): "fp8_e4m3_grid_order2",
}


def _stage(key):
    arch, hq, hkv, rows, reqs, page, sms, prefill, pages, out_kv = key
    return loader.stage_for(
        hq,
        hkv,
        arch=arch,
        num_rows=rows,
        num_requests=reqs,
        page_size=page,
        num_sms=sms,
        is_prefill=prefill,
        max_pages_per_request=pages,
        has_out_kv=out_kv,
    )


@pytest.mark.parametrize("key,expected", sorted(_POLICY_TABLE.items()))
def test_stage_policy_table(key, expected):
    stage = _stage(key)
    assert stage == expected
    assert stage in loader.stages_for(key[0])


@pytest.mark.parametrize("arch", loader.ARCHES)
def test_stage_policy_only_selects_shipped_builds(arch):
    shipped = set(loader.stages_for(arch))
    for hq, hkv in loader.SUPPORTED_HEADS:
        for page in (4, 16, 64, 128):
            for reqs in (0, 1, 4, 8, 32, 96, 97, 128, 256, 512):
                # rows < reqs: requests without rows in this step
                for rows in sorted({reqs, 1, 128, 1024, 4096}):
                    for prefill in (False, True):
                        for pages in (1, 33, 48, 129, 513):
                            for out_kv in (False, True):
                                for sms in (132, 148, 152):
                                    key = (
                                        arch,
                                        hq,
                                        hkv,
                                        rows,
                                        reqs,
                                        page,
                                        sms,
                                        prefill,
                                        pages,
                                        out_kv,
                                    )
                                    stage = _stage(key)
                                    assert stage in shipped, (key, stage)
                                    # a stage's grid serves the problem it was selected for
                                    loader.launch_grid(
                                        loader.MODULES[stage],
                                        num_rows=rows,
                                        num_requests=reqs,
                                        num_kv_heads=hkv,
                                    )


def test_stage_policy_rejects_unsupported_heads():
    with pytest.raises(ValueError):
        _stage(("sm_100a", 16, 2, 1, 1, 64, 148, False, 1, False))


# ----------------------------------------------------------------------------------------------
# Package hygiene (CPU): the vendorable package imports itself, torch and tvm_ffi only at
# module level and reads no environment variables.  Function-level lazy imports are the
# documented exceptions: ``torch`` inside the registry's device helpers and FlashInfer's JIT
# builder inside the byte-pinned ``cake_build.py`` JIT path (the shared-loader exception of
# the Cake package form).


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.module is None:
                    names.update(f".{alias.name}" for alias in node.names)
                else:
                    names.add(f".{node.module.split('.')[0]}")
            else:
                assert node.module is not None
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "filename,allowed",
    [
        (
            "cake_jit.py",
            {"__future__", "functools", "typing", "tvm_ffi", ".cake_build"},
        ),
        (
            "__init__.py",
            {"__future__", "functools", "typing", "math", "torch", ".cake_jit"},
        ),
    ],
)
def test_package_module_level_imports_are_whitelisted(filename, allowed):
    assert _module_level_imports(PACKAGE_DIR / filename) <= allowed


def test_cake_build_imports_only_the_standard_library():
    names = _module_level_imports(PACKAGE_DIR / "cake_build.py")
    assert names <= set(sys.stdlib_module_names) | {"__future__"}, names


@pytest.mark.parametrize("filename", ["cake_jit.py", "__init__.py", "cake_build.py"])
def test_package_reads_no_environment_variables(filename):
    text = (PACKAGE_DIR / filename).read_text(encoding="utf-8")
    assert "os.environ" not in text
    assert "getenv" not in text
