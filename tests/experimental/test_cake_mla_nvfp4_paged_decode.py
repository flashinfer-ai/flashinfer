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

# Tests for the experimental Cake dense MLA decode over the NVFP4 paged latent
# cache (flashinfer-ai/flashinfer#4676 layout): FP32 reference over the decoded
# NVFP4 / FP8 operands, dense and packed variable-length queries, natural-log
# LSE, decode context parallelism, page sizes 32 / 64 / 128, the single-buffer
# 352-byte cache layout, CUDA-Graph replay with a changed page table, the BF16
# query quantizer against its torch reference, and the host plan.

import math

import pytest
import torch

from flashinfer.experimental.cake_mla_nvfp4_paged_decode import cake_jit
from flashinfer.jit.cpp_ext import get_cuda_version
from flashinfer.experimental.cake_mla_nvfp4_paged_decode.cake_backend import (
    CKV_BYTES,
    LSE_BIAS,
    MAX_SPLITS,
    QK_DIM,
    ROPE,
    ROW_TILES,
    SF_BYTES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    V_DIM,
    WIDE_BLOCK_M,
    WIDE_CLUSTER,
    WIDE_LSE_BIAS,
    CakeMlaNvfp4PagedDecode,
    CakeMlaNvfp4QueryQuantize,
    cake_mla_nvfp4_paged_decode,
    max_workspace_bytes,
    mla_nvfp4_query_buffers,
    plan_mla_nvfp4_paged_decode,
    quantize_mla_nvfp4_query,
    rt_for_rows,
    use_wide_route,
    wide_tiles,
    workspace_bytes,
)

LATENT = V_DIM
SF_VEC = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
SM_SCALE = 1.0 / math.sqrt(
    128 + 64
)  # pre-absorption 192-dim scale of Kimi-K3 / DeepSeek-V3
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _arch():
    major, minor = torch.cuda.get_device_capability(torch.device("cuda"))
    return SUPPORTED_COMPUTE_CAPABILITIES.get((major, minor))


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    arch = _arch()
    if arch is None or not cake_jit.program_available(arch):
        pytest.skip("Cake NVFP4 MLA decode has no generated programs for this device")
    unsupported = [p for p in cake_jit.PROGRAMS if not cake_jit.toolkit_supports(p)]
    if unsupported:
        # The attention programs spell the Blackwell QMUL4 in PTX ISA 9.4 (CUDA 13.4).
        pytest.skip(
            f"Cake NVFP4 MLA decode needs CUDA 13.4 or newer, nvcc is {get_cuda_version()}"
        )


# ---------------------------------------------------------------------------
# NVFP4 reference arithmetic (the #4676 cache recipe)
# ---------------------------------------------------------------------------


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code (ties to even) of ``x`` (|x| <= 6 after scaling)."""
    mag = x.abs().float()
    values = _E2M1_VALUES.to(x.device)
    mids = (values[1:] + values[:-1]) / 2
    idx = torch.searchsorted(mids, mag.reshape(-1), right=True).reshape(mag.shape)
    tie = (idx > 0) & (mag == mids[(idx - 1).clamp_min(0)])
    idx = torch.where(tie & ((idx - 1) % 2 == 0), idx - 1, idx)
    codes = idx.to(torch.uint8)
    # The sign bit is kept on zero codes too (negative values rounding to 0 encode as -0, like the kernel).
    return codes | torch.signbit(x).to(torch.uint8) << 3


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    values = _E2M1_VALUES.to(packed.device)
    low = packed & 0x7
    high = (packed >> 4) & 0x7
    sign_low = ((packed >> 3) & 1).bool()
    sign_high = ((packed >> 7) & 1).bool()
    out = torch.stack(
        [
            torch.where(sign_low, -values[low.long()], values[low.long()]),
            torch.where(sign_high, -values[high.long()], values[high.long()]),
        ],
        dim=-1,
    )
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def _quantize_blocks(x: torch.Tensor, scale: float):
    """``(packed_u8[..., D/2], sf_e4m3[..., D/16])``: ``sf = e4m3(amax_block / (6 scale))``,
    ``codes = e2m1(x / (sf scale))`` (zero for a zero block)."""
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // SF_VEC, SF_VEC)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    sf = _fp8(amax / (E2M1_MAX * scale))
    sf_f = sf.float()
    inv = torch.where(sf_f > 0, 1.0 / (sf_f * scale), torch.zeros_like(sf_f))
    codes = _e2m1_codes(blocks * inv).reshape(x.shape)
    return _pack_e2m1(codes), sf.reshape(*x.shape[:-1], x.shape[-1] // SF_VEC)


def _quantize_query_reference(q: torch.Tensor, ckv_scale: float, kpe_scale: float):
    """The quantizer the ``quantize`` program reproduces bit for bit (float32 arithmetic)."""
    f32 = torch.float32
    qf = q.float()
    nope, rope = qf[..., :LATENT], qf[..., LATENT:]
    c_nope = torch.tensor(1.0 / (E2M1_MAX * E4M3_MAX), dtype=f32, device=q.device)
    c_rope = torch.tensor(kpe_scale, dtype=f32, device=q.device) * torch.reciprocal(
        torch.tensor(E4M3_MAX * ckv_scale, dtype=f32, device=q.device)
    )
    q_scale = torch.maximum(
        nope.abs().amax(dim=-1) * c_nope, rope.abs().amax(dim=-1) * c_rope
    )
    q_scale = torch.where(q_scale > 0, q_scale, torch.ones_like(q_scale))
    blocks = nope.reshape(*nope.shape[:-1], LATENT // SF_VEC, SF_VEC)
    amax = blocks.abs().amax(dim=-1)
    sf = _fp8(amax * torch.reciprocal(6.0 * q_scale)[..., None])
    sf_f = sf.float()
    inv = torch.where(
        sf_f > 0, torch.reciprocal(sf_f * q_scale[..., None]), torch.zeros_like(sf_f)
    )
    codes = _e2m1_codes(blocks * inv[..., None]).reshape(nope.shape)
    rope_mul = torch.tensor(kpe_scale, dtype=f32, device=q.device) * torch.reciprocal(
        q_scale * torch.tensor(ckv_scale, dtype=f32, device=q.device)
    )
    q_rope = _fp8(rope * rope_mul[..., None])
    return _pack_e2m1(codes), sf, q_rope, q_scale


def _decode_query(q_nope, q_sf, q_rope, q_scale, ckv_scale, kpe_scale):
    nope = (
        _decode_e2m1(q_nope)
        * q_sf.float().repeat_interleave(SF_VEC, dim=-1)
        * q_scale[..., None]
    )
    rope = q_rope.float() * (q_scale[..., None] * (ckv_scale / kpe_scale))
    return torch.cat([nope, rope], dim=-1)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _make_case(
    q_lens,
    kv_lens,
    num_heads,
    *,
    seed,
    device,
    page_size=64,
    pool_pages=None,
    kv_lens_global=None,
    cp_world=1,
    cp_rank=0,
    packed_cache=False,
    return_lse=False,
):
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(kv_lens)
    pages_per_seq = [max(1, (k + page_size - 1) // page_size) for k in kv_lens]
    width = max(pages_per_seq)
    total_pages = sum(pages_per_seq)
    pool_pages = pool_pages or total_pages + 4
    ckv = (
        torch.randn((pool_pages, page_size, LATENT), generator=gen, device=device)
        .to(torch.bfloat16)
        .float()
    )
    kpe = (
        (torch.randn((pool_pages, page_size, ROPE), generator=gen, device=device) * 2.5)
        .to(torch.bfloat16)
        .float()
    )
    ckv_scale = float(ckv.abs().amax()) / (E2M1_MAX * E4M3_MAX)
    kpe_scale = float(kpe.abs().amax()) / E4M3_MAX
    ckv_packed, ckv_sf = _quantize_blocks(ckv, ckv_scale)
    kpe_fp8 = _fp8(kpe / kpe_scale)
    if packed_cache:
        # vLLM-style single allocation: 256 FP4 bytes | 64 rope bytes | 32 scale bytes per token.
        buffer = torch.empty(
            (pool_pages, page_size, CKV_BYTES + ROPE + SF_BYTES),
            dtype=torch.uint8,
            device=device,
        )
        buffer[..., :CKV_BYTES] = ckv_packed
        buffer[..., CKV_BYTES : CKV_BYTES + ROPE] = kpe_fp8.view(torch.uint8)
        buffer[..., CKV_BYTES + ROPE :] = ckv_sf.view(torch.uint8)
        ckv_cache = buffer[..., :CKV_BYTES]
        kpe_cache = buffer[..., CKV_BYTES : CKV_BYTES + ROPE].view(torch.float8_e4m3fn)
        ckv_sf_cache = buffer[..., CKV_BYTES + ROPE :].view(torch.float8_e4m3fn)
    else:
        ckv_cache, ckv_sf_cache, kpe_cache = (
            ckv_packed.contiguous(),
            ckv_sf.contiguous(),
            kpe_fp8.contiguous(),
        )
    perm = torch.randperm(pool_pages, generator=gen, device=device)[:total_pages].to(
        torch.int32
    )
    block_tables = torch.full((batch, width), -1, dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(pages_per_seq):
        block_tables[b, :n] = perm[off : off + n]
        off += n
    total_q = sum(q_lens)
    q = torch.randn((total_q, num_heads, QK_DIM), generator=gen, device=device).to(
        torch.bfloat16
    )
    q_nope, q_sf, q_rope, q_scale = _quantize_query_reference(q, ckv_scale, kpe_scale)
    q_indptr = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    return dict(
        q_bf16=q,
        q_nope=q_nope,
        q_sf=q_sf,
        q_rope=q_rope,
        q_scale=q_scale,
        ckv_cache=ckv_cache,
        ckv_sf_cache=ckv_sf_cache,
        kpe_cache=kpe_cache,
        ckv_scale=ckv_scale,
        kpe_scale=kpe_scale,
        block_tables=block_tables,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr,
        q_lens=list(q_lens),
        kv_lens=list(kv_lens),
        kv_lens_global=list(kv_lens_global)
        if kv_lens_global is not None
        else list(kv_lens),
        cp_world=cp_world,
        cp_rank=cp_rank,
        num_heads=num_heads,
        page_size=page_size,
        return_lse=return_lse,
    )


def _visible_keys(kv_len, q_len, t, *, cp_world, cp_rank, kv_len_global):
    if cp_world == 1:
        return max(0, min(kv_len, kv_len - q_len + t + 1))
    num = kv_len_global - q_len + t - cp_rank
    if num < 0:
        return 0
    return min(kv_len, num // cp_world + 1)


def _reference(case):
    """FP32 attention over the decoded NVFP4 query and cache; ``(O bf16, LSE natural log)``."""
    device = case["q_nope"].device
    page_size, num_heads = case["page_size"], case["num_heads"]
    nope = _decode_e2m1(case["ckv_cache"]) * case[
        "ckv_sf_cache"
    ].float().repeat_interleave(SF_VEC, dim=-1)
    cache = torch.cat(
        [nope * case["ckv_scale"], case["kpe_cache"].float() * case["kpe_scale"]],
        dim=-1,
    )
    q_rows = _decode_query(
        case["q_nope"],
        case["q_sf"],
        case["q_rope"],
        case["q_scale"],
        case["ckv_scale"],
        case["kpe_scale"],
    )
    total_q = q_rows.shape[0]
    out = torch.zeros((total_q, num_heads, LATENT), dtype=torch.float32, device=device)
    lse = torch.full(
        (total_q, num_heads), float("-inf"), dtype=torch.float32, device=device
    )
    q_indptr = case["q_indptr"].tolist()
    for b, (q_len, kv_len) in enumerate(
        zip(case["q_lens"], case["kv_lens"], strict=True)
    ):
        n_pages = (kv_len + page_size - 1) // page_size
        pages = case["block_tables"][b, :n_pages].long()
        keys = cache[pages].reshape(-1, QK_DIM)[:kv_len]
        for t in range(q_len):
            visible = _visible_keys(
                kv_len,
                q_len,
                t,
                cp_world=case["cp_world"],
                cp_rank=case["cp_rank"],
                kv_len_global=case["kv_lens_global"][b],
            )
            row = q_indptr[b] + t
            if visible == 0:
                out[row] = 0.0
                continue
            logits = (q_rows[row] @ keys[:visible].T) * SM_SCALE
            m = logits.amax(dim=-1, keepdim=True)
            p = torch.exp(logits - m)
            s = p.sum(dim=-1, keepdim=True)
            out[row] = (p / s) @ keys[:visible, :LATENT]
            lse[row] = (m + torch.log(s)).squeeze(-1)
    return out.to(torch.bfloat16), lse


def _workspace(case, device):
    rows = case["q_nope"].shape[0] * case["num_heads"]
    return torch.zeros(max_workspace_bytes(rows), dtype=torch.uint8, device=device)


def _run(case, *, dense_q_len=None, graph=False, num_split=None):
    device = case["q_nope"].device
    num_heads = case["num_heads"]
    total_q = case["q_nope"].shape[0]
    workspace = _workspace(case, device)
    lead = (total_q, num_heads)
    out = torch.full((*lead, LATENT), float("nan"), dtype=torch.bfloat16, device=device)
    lse = (
        torch.full(lead, float("nan"), dtype=torch.float32, device=device)
        if case["return_lse"]
        else None
    )
    kwargs = dict(
        ckv_cache=case["ckv_cache"],
        ckv_sf_cache=case["ckv_sf_cache"],
        kpe_cache=case["kpe_cache"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        workspace_buffer=workspace,
        sm_scale=SM_SCALE,
        ckv_scale=case["ckv_scale"],
        max_seq_len=int(case["block_tables"].shape[1]) * case["page_size"],
        cp_world=case["cp_world"],
        cp_rank=case["cp_rank"],
        kv_len_global=(
            torch.tensor(case["kv_lens_global"], dtype=torch.int32, device=device)
            if case["cp_world"] > 1
            else None
        ),
        num_split=num_split,
    )
    operands = ("q_nope", "q_sf", "q_rope", "q_scale")
    if dense_q_len is not None:
        batch = len(case["q_lens"])
        shaped = {
            k: case[k].reshape(batch, dense_q_len, *case[k].shape[1:]) for k in operands
        }
        runner = CakeMlaNvfp4PagedDecode(
            **shaped,
            out=out.view(batch, dense_q_len, num_heads, LATENT),
            lse=None if lse is None else lse.view(batch, dense_q_len, num_heads),
            **kwargs,
        )
    else:
        runner = CakeMlaNvfp4PagedDecode(
            **{k: case[k] for k in operands},
            out=out,
            lse=lse,
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=max(case["q_lens"]),
            **kwargs,
        )
    if not graph:
        runner.launch()
        torch.cuda.synchronize()
        return out, lse, runner
    runner.launch()  # warm outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
        runner.launch()
    torch.cuda.synchronize()
    return out, lse, runner, g


def _check(out, ref, *, atol=0.1, rtol=0.1, rel_l2=0.06):
    # rel_l2 0.06 (the contract gates 0.05): a one-key request's output is one E4M3-rounded V row, whose
    # relative L2 error against the FP32 decode of the cache reaches ~0.05 on its own.
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
    rel = (out.float() - ref.float()).norm(dim=-1) / ref.float().norm(dim=-1).clamp_min(
        1e-6
    )
    assert float(rel.max()) <= rel_l2


def _check_lse(lse, ref_lse, *, atol=0.01):
    both_inf = torch.isneginf(ref_lse) & torch.isneginf(lse)
    err = torch.where(both_inf, torch.zeros_like(ref_lse), (lse - ref_lse).abs())
    assert float(err.nan_to_num(float("inf")).max()) <= atol


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [12, 96])
@pytest.mark.parametrize(
    "kv_lens", [[1, 65, 700], [4095, 777, 129], [20000, 16384, 17001]]
)
def test_decode_q1(num_heads, kv_lens):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case(
        [1] * len(kv_lens), kv_lens, num_heads, seed=625001, device=device
    )
    out, _, runner = _run(case, dense_q_len=1)
    assert runner.plan.main_kind == (
        "main_wide"
        if use_wide_route(num_heads)
        else f"main_rt{rt_for_rows(num_heads)[0]}"
    )
    ref, _ = _reference(case)
    _check(out, ref)


@pytest.mark.parametrize(
    "num_heads,q_lens,kv_lens",
    [
        (12, [1, 4, 8], [300, 4096, 777]),  # 96 packed rows: two 48-row tiles
        (24, [2, 1], [4100, 999]),  # 48 rows per request
        (96, [4, 4], [2048, 3001]),
        (128, [1, 1, 1], [100, 3000, 6001]),  # DeepSeek-V3 TP1 head count
    ],
)
def test_packed_variable_query(num_heads, q_lens, kv_lens):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case(
        q_lens, kv_lens, num_heads, seed=625003, device=device, return_lse=True
    )
    out, lse, _ = _run(case)
    ref, ref_lse = _reference(case)
    _check(out, ref)
    _check_lse(lse, ref_lse)


@pytest.mark.parametrize("page_size", [32, 128])
def test_page_sizes(page_size):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case(
        [2, 1, 1], [31, 640, 3333], 12, seed=625010, device=device, page_size=page_size
    )
    out, _, _ = _run(case)
    ref, _ = _reference(case)
    _check(out, ref)


def test_packed_352_byte_cache_layout_matches_separate_tensors():
    _skip_unless_supported()
    device = torch.device("cuda")
    kv_lens = [700, 2000, 4097]
    separate = _make_case([1] * 3, kv_lens, 12, seed=625012, device=device)
    packed = _make_case(
        [1] * 3, kv_lens, 12, seed=625012, device=device, packed_cache=True
    )
    assert packed["ckv_cache"].stride(-2) == CKV_BYTES + ROPE + SF_BYTES
    out_a, _, _ = _run(separate, dense_q_len=1)
    out_b, _, _ = _run(packed, dense_q_len=1)
    assert torch.equal(out_a, out_b)
    ref, _ = _reference(separate)
    _check(out_a, ref)


@pytest.mark.parametrize(
    "num_heads,q_len,kv_lens,kv_lens_global,cp_world,cp_rank",
    [
        (12, 4, [250, 1001, 2], [1000, 4005, 9], 4, 1),
        (96, 1, [120, 4], [967, 37], 8, 7),
    ],
)
def test_decode_context_parallel_rank(
    num_heads, q_len, kv_lens, kv_lens_global, cp_world, cp_rank
):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case(
        [q_len] * len(kv_lens),
        kv_lens,
        num_heads,
        seed=625020,
        device=device,
        kv_lens_global=kv_lens_global,
        cp_world=cp_world,
        cp_rank=cp_rank,
        return_lse=True,
    )
    out, lse, _ = _run(case, dense_q_len=q_len)
    ref, ref_lse = _reference(case)
    _check(out, ref)
    _check_lse(lse, ref_lse)
    # Rows without a visible key on this rank are zero with LSE -inf (cross-rank merge contract).
    empty = torch.isneginf(ref_lse)
    if bool(empty.any()):
        assert torch.isneginf(lse[empty]).all()
        assert (out[empty].float() == 0).all()


def test_long_kv_split_merge_and_lse():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case(
        [1, 1], [65536, 40000], 12, seed=625030, device=device, return_lse=True
    )
    out, lse, runner = _run(case, dense_q_len=1)
    assert runner.num_split > 1
    ref, ref_lse = _reference(case)
    _check(out, ref)
    _check_lse(lse, ref_lse)


def test_cta_reducer_many_splits():
    _skip_unless_supported()
    device = torch.device("cuda")
    # One request: the planner takes one split per SM, above the warp reducer's 32-split range.
    case = _make_case([1], [40000], 12, seed=625031, device=device, return_lse=True)
    out, lse, runner = _run(case, dense_q_len=1)
    assert runner.plan.reduce_kind == "reduce_cta"
    ref, ref_lse = _reference(case)
    _check(out, ref)
    _check_lse(lse, ref_lse)


def test_query_quantizer_matches_reference_bitwise():
    _skip_unless_supported()
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(625040)
    for rows, ckv_scale, kpe_scale in ((37, 0.0021, 0.0057), (1024, 1.0, 0.03)):
        q = (torch.randn((rows, 12, QK_DIM), generator=gen, device=device) * 1.7).to(
            torch.bfloat16
        )
        q[3, 2] = 0.0  # all-zero row: q_scale 1, zero codes / scales
        got = quantize_mla_nvfp4_query(q, ckv_scale, kpe_scale)
        want = _quantize_query_reference(q, ckv_scale, kpe_scale)
        for g, w in zip(got, want, strict=True):
            assert g.shape == w.shape and g.dtype == w.dtype
            assert torch.equal(
                g.view(torch.uint8) if g.dtype == torch.float8_e4m3fn else g,
                w.view(torch.uint8) if w.dtype == torch.float8_e4m3fn else w,
            )


def test_complete_call_bf16_query():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case([1, 1], [1000, 2000], 12, seed=625041, device=device)
    batch = 2
    q = case["q_bf16"].reshape(batch, 1, 12, QK_DIM)
    buffers = mla_nvfp4_query_buffers((batch, 1, 12), device)
    quantizer = CakeMlaNvfp4QueryQuantize(
        query=q, ckv_scale=case["ckv_scale"], kpe_scale=case["kpe_scale"], out=buffers
    )
    q_nope, q_sf, q_rope, q_scale = quantizer.launch()
    out = cake_mla_nvfp4_paged_decode(
        q_nope,
        q_sf,
        q_rope,
        q_scale,
        case["ckv_cache"],
        case["ckv_sf_cache"],
        case["kpe_cache"],
        case["block_tables"],
        case["seq_lens"],
        _workspace(case, device),
        sm_scale=SM_SCALE,
        ckv_scale=case["ckv_scale"],
    )
    torch.cuda.synchronize()
    ref, _ = _reference(case)
    _check(out.view(batch, 12, LATENT), ref)


def test_cuda_graph_replay_changing_page_table():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case(
        [1] * 4, [900, 1800, 2700, 3600], 12, seed=625050, device=device, pool_pages=192
    )
    out, _, _, g = _run(case, dense_q_len=1, graph=True)
    ref, _ = _reference(case)
    _check(out, ref)
    # New page assignment and cache contents in place; replay must follow the tables.
    gen = torch.Generator(device=device).manual_seed(7)
    pool_pages, page_size = case["ckv_cache"].shape[:2]
    ckv = (
        torch.randn((pool_pages, page_size, LATENT), generator=gen, device=device)
        .to(torch.bfloat16)
        .float()
    )
    ckv_packed, ckv_sf = _quantize_blocks(ckv, case["ckv_scale"])
    case["ckv_cache"].copy_(ckv_packed)
    case["ckv_sf_cache"].copy_(ckv_sf)
    perm = torch.randperm(pool_pages, generator=gen, device=device).to(torch.int32)
    off = 0
    for b in range(4):
        n = (case["kv_lens"][b] + page_size - 1) // page_size
        case["block_tables"][b, :n] = perm[off : off + n]
        off += n
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ref, _ = _reference(case)
    _check(out, ref)


def test_launch_allocates_nothing():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case([1, 1], [5000, 300], 12, seed=625060, device=device)
    _, _, runner = _run(case, dense_q_len=1)
    runner.launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]


def test_rejects_unsupported_inputs():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _make_case([1], [128], 12, seed=1, device=device)
    with pytest.raises(ValueError, match="page_size"):
        _run(_make_case([1], [100], 12, seed=1, device=device, page_size=16))
    operands = {k: case[k] for k in ("q_nope", "q_sf", "q_rope", "q_scale")}
    common = dict(
        ckv_cache=case["ckv_cache"],
        ckv_sf_cache=case["ckv_sf_cache"],
        kpe_cache=case["kpe_cache"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        workspace_buffer=_workspace(case, device),
        sm_scale=SM_SCALE,
        ckv_scale=case["ckv_scale"],
        cum_seq_lens_q=case["q_indptr"],
        max_q_len=1,
    )
    out = torch.empty((1, 12, LATENT), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="kv_len_global"):
        CakeMlaNvfp4PagedDecode(**operands, out=out, cp_world=2, cp_rank=1, **common)
    with pytest.raises(ValueError, match="q_nope must be uint8"):
        CakeMlaNvfp4PagedDecode(
            **dict(operands, q_nope=case["q_nope"].view(torch.int8)), out=out, **common
        )


def test_every_kernel_key_resolves_on_the_running_architecture():
    _skip_unless_supported()
    arch = _arch()
    for key in cake_jit.KERNELS:
        program = cake_jit.select_program(key, arch)
        record = cake_jit.PROGRAMS[program]
        assert arch in record["arches"]
        assert set(record) >= {"role", "sources", "arches"}
        # Only the attention programs (hardware QMUL4 through PTX ISA 9.4) carry the toolkit floor.
        assert (record.get("min_cuda_version") == "13.4") == key.startswith("main_")
    assert {f"main_rt{rt}" for rt in ROW_TILES} | {"main_wide"} <= set(cake_jit.KERNELS)
    assert {"reduce_w1", "reduce_w2", "reduce_w4", "reduce_cta", "quantize"} <= set(
        cake_jit.KERNELS
    )


# ---------------------------------------------------------------------------
# CPU tests: host plan and workspace
# ---------------------------------------------------------------------------


def test_row_tiles():
    assert rt_for_rows(6) == (16, 1)
    assert rt_for_rows(12) == (16, 1)
    assert rt_for_rows(24) == (32, 1)
    assert rt_for_rows(48) == (48, 1)
    assert rt_for_rows(96) == (48, 2)
    assert rt_for_rows(128) == (48, 3)
    assert rt_for_rows(12 * 5) == (
        32,
        2,
    )  # 60 rows: two 48-row tiles, covered by two 32-row tiles


@pytest.mark.parametrize("sm_count", [148, 152, 160])
def test_plan_invariants(sm_count):
    for batch in (1, 2, 8, 32):
        for max_q_len, num_heads in (
            (1, 6),
            (1, 12),
            (1, 96),
            (4, 12),
            (8, 12),
            (1, 128),
        ):
            for max_seq_len in (1, 700, 8192, 131072, 1048576):
                plan = plan_mla_nvfp4_paged_decode(
                    batch=batch,
                    max_q_len=max_q_len,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    sm_count=sm_count,
                )
                if use_wide_route(max_q_len * num_heads):
                    assert plan.wide and plan.main_kind == "main_wide"
                    rt, m_tiles, ctas = (
                        WIDE_BLOCK_M,
                        wide_tiles(max_q_len * num_heads),
                        WIDE_CLUSTER,
                    )
                else:
                    assert not plan.wide and plan.main_kind == f"main_rt{plan.rt}"
                    (rt, m_tiles), ctas = rt_for_rows(max_q_len * num_heads), 1
                assert (plan.rt, plan.m_tiles) == (rt, m_tiles)
                assert plan.rows_max == batch * max_q_len * num_heads
                assert plan.grid_main == (ctas * plan.num_split, m_tiles, batch)
                assert 1 <= plan.num_split <= MAX_SPLITS
                assert ctas * plan.num_split * batch * m_tiles <= max(
                    sm_count, ctas * batch * m_tiles
                )
                if plan.num_split >= 33:
                    assert plan.reduce_kind == "reduce_cta" and plan.grid_reduce == (
                        plan.rows_max,
                        4,
                        1,
                    )
                else:
                    assert plan.reduce_kind == f"reduce_w{plan.reduce_warps}"
                    assert (
                        plan.grid_reduce[0] * (8 // plan.reduce_warps) >= plan.rows_max
                    )
                assert workspace_bytes(
                    plan.rows_max, plan.num_split
                ) <= max_workspace_bytes(plan.rows_max)
    cached = plan_mla_nvfp4_paged_decode(
        batch=8, max_q_len=1, num_heads=12, max_seq_len=342305, sm_count=152
    )
    assert cached is plan_mla_nvfp4_paged_decode(
        batch=8, max_q_len=1, num_heads=12, max_seq_len=342305, sm_count=152
    )
    assert cached.num_split == 19 and cached.reduce_kind == "reduce_w4"
    # Packed variable-length queries: the partials and the merge grid follow the rows the query holds.
    ragged = plan_mla_nvfp4_paged_decode(
        batch=3, max_q_len=8, num_heads=12, max_seq_len=4096, sm_count=152, rows=13 * 12
    )
    # 8 x 12 = 96 rows per request: the wide route, one 128-row cluster per (split, request).
    assert ragged.wide and ragged.rows_max == 156
    assert ragged.grid_main == (WIDE_CLUSTER * ragged.num_split, 1, 3)
    with pytest.raises(ValueError, match="rows must be"):
        plan_mla_nvfp4_paged_decode(
            batch=1, max_q_len=1, num_heads=12, max_seq_len=64, sm_count=152, rows=13
        )


def test_lse_bias_is_the_e4m3_probability_budget():
    assert pytest.approx(math.log2(448.0) - 5.0) == LSE_BIAS
    assert WIDE_LSE_BIAS == 6.0


def test_wide_route_selection():
    assert not use_wide_route(48) and use_wide_route(49) and use_wide_route(128)
    assert (
        wide_tiles(49) == 1
        and wide_tiles(128) == 1
        and wide_tiles(129) == 2
        and wide_tiles(8 * 48) == 3
    )
    plan = plan_mla_nvfp4_paged_decode(
        batch=8, max_q_len=1, num_heads=96, max_seq_len=342305, sm_count=152
    )
    assert (
        plan.wide
        and plan.main_kind == "main_wide"
        and plan.rt == WIDE_BLOCK_M
        and plan.m_tiles == 1
    )
    assert (
        plan.grid_main == (WIDE_CLUSTER * plan.num_split, 1, 8)
        and plan.lse_bias == WIDE_LSE_BIAS
    )
    assert plan.num_split * WIDE_CLUSTER * 8 <= 152
    narrow = plan_mla_nvfp4_paged_decode(
        batch=8, max_q_len=1, num_heads=48, max_seq_len=342305, sm_count=152
    )
    assert not narrow.wide and narrow.lse_bias == LSE_BIAS
