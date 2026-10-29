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

Backend suite of the Cake dense NVFP4 MLA decode on Rubin
(``flashinfer.experimental.cake_rubin_mla_nvfp4_decode``): FP32 reference over
the decoded NVFP4 / FP8 operands, dense and packed variable-length queries,
natural-log LSE, decode context parallelism (the ranks merged on the host),
the single-buffer 352-byte cache layout, split / merge rows up to and past 32
splits, CUDA-graph replay with a changed page table, the BF16 query quantizer
bit for bit against its torch reference, and the host plan (no device).

The suite imports only the package, ``flashinfer.experimental.cake_rubin_mla_nvfp4_decode``.
"""

import math
import re
from pathlib import Path

import pytest
import torch

from flashinfer.experimental.cake_rubin_mla_nvfp4_decode import cake_backend, cake_jit

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent

LATENT = cake_backend.LATENT
ROPE = cake_backend.ROPE
QK_DIM = cake_backend.QK_DIM
SF_VEC = cake_backend.SF_VEC
CKV_BYTES = cake_backend.CKV_BYTES
SF_BYTES = cake_backend.SF_BYTES
PAGE = cake_backend.PAGE
ROWS_TILE = cake_backend.ROWS_TILE
MAX_SPLITS = cake_backend.MAX_SPLITS
E2M1_MAX = 6.0
E4M3_MAX = 448.0
SM_SCALE = 1.0 / math.sqrt(
    128 + 64
)  # pre-absorption 192-dim scale of Kimi-K3 / DeepSeek-V3

# The contract: O atol = rtol = 0.1 (FP8-class), LSE 0.05 absolute, a per-row relative L2 error of O under 0.08
# (P is E4M3, whose rounding is up to 2^-4 = 6.25 % of a value, and a row one key dominates carries that key's
# rounding whole); one element in a thousand may cross the element gate where a large V entry meets a small
# expected value.
O_ATOL = O_RTOL = 0.1
LSE_ATOL = 0.05
ROW_REL_L2 = 0.08
MAX_MISMATCH = 1e-3

# The owner's shape list: H96, Q in {4, 8}, B in {1, 4, 16, 64}, KV in {64K, 128K}.
OWNER_CONFIGS = [
    (96, q, b, kv) for q in (4, 8) for b in (1, 4, 16, 64) for kv in (65536, 131072)
]

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {cake_jit.ARCHES}"
        )
    if not (PACKAGE_DIR / "build.json").exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", index)


# ---------------------------------------------------------------------------
# NVFP4 reference arithmetic (the #4676 cache recipe and the query quantizer)
# ---------------------------------------------------------------------------


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    mag = x.abs()
    code = torch.zeros_like(mag, dtype=torch.uint8)
    # Round to nearest even on the E2M1 grid 0, .5, 1, 1.5, 2, 3, 4, 6: ties go to the even code.
    for thresh, value, strict in (
        (0.25, 1, True),
        (0.75, 2, False),
        (1.25, 3, True),
        (1.75, 4, False),
        (2.5, 5, True),
        (3.5, 6, False),
        (5.0, 7, True),
    ):
        code = torch.where(
            (mag > thresh) if strict else (mag >= thresh),
            torch.tensor(value, dtype=torch.uint8, device=x.device),
            code,
        )
    return code | (torch.signbit(x).to(torch.uint8) << 3)


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.cat((_E2M1, -_E2M1)).to(packed.device)
    return torch.stack(
        (lut[(packed & 0xF).long()], lut[(packed >> 4).long()]), dim=-1
    ).flatten(-2)


def _nvfp4_quantize_blocks(x: torch.Tensor, global_scale: torch.Tensor):
    """Packed E2M1 codes and E4M3 block scales of ``x`` under the per-row ``global_scale``."""
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // SF_VEC, SF_VEC)
    gs_b = global_scale.reshape(*global_scale.shape, 1, 1)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    ok = (amax > 0) & (gs_b > 0)
    sf_val = torch.where(
        ok, amax * torch.reciprocal(6.0 * gs_b), torch.zeros_like(amax)
    )
    sf = sf_val.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    out_scale = torch.where(
        ok & (sf_f > 0), torch.reciprocal(sf_f * gs_b), torch.zeros_like(sf_f)
    )
    codes = _e2m1_codes(blocks * out_scale).reshape(shape)
    return _pack_e2m1(codes), sf.reshape(*shape[:-1], shape[-1] // SF_VEC)


def _quantize_query_reference(q: torch.Tensor, ckv_scale: float, kpe_scale: float):
    """The reference quantizer the ``quantize`` program reproduces bit for bit (float32 arithmetic):
    ``(q_nope u8, q_sf e4m3, q_rope e4m3, q_scale f32)`` of a BF16 ``[.., 576]`` query."""
    f32 = torch.float32
    qf = q.float()
    nope, rope = qf[..., :LATENT], qf[..., LATENT:]
    c_nope = torch.tensor(1.0 / (E2M1_MAX * E4M3_MAX), dtype=f32, device=qf.device)
    c_rope = torch.tensor(kpe_scale, dtype=f32, device=qf.device) * torch.reciprocal(
        torch.tensor(E4M3_MAX * ckv_scale, dtype=f32, device=qf.device)
    )
    q_scale = torch.maximum(nope.abs().amax(-1) * c_nope, rope.abs().amax(-1) * c_rope)
    q_scale = torch.where(q_scale > 0, q_scale, torch.ones_like(q_scale))
    packed, sf = _nvfp4_quantize_blocks(nope, q_scale)
    rope_mul = torch.tensor(kpe_scale, dtype=f32, device=qf.device) * torch.reciprocal(
        q_scale * torch.tensor(ckv_scale, dtype=f32, device=qf.device)
    )
    rope_q = (
        (rope * rope_mul[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    )
    return (
        packed.contiguous(),
        sf.contiguous(),
        rope_q.contiguous(),
        q_scale.contiguous(),
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _make_case(
    kv_lens,
    q_lens,
    num_heads,
    *,
    seed,
    device,
    packed_cache=False,
    values="normal",
    o_scale=1.0,
    cp_world=1,
    cp_rank=0,
    cp_interleave=1,
    kv_lens_global=None,
    return_lse=False,
    pool_pages=None,
):
    """One problem: the paged cache (``values="normal"``: a quantized normal latent and rope with global scales;
    ``"bytes"``: random E2M1 codes, block scales from {0.5 .. 2}, FP8 rope codes 0..119, unit scales, whose
    log2-domain logits spread over hundreds of units within a row), a quantized normal BF16 query and the tables.
    Unused ``block_tables`` entries are 0, a valid page: the decode reads entry ``[b, 0]`` even for a request
    without local keys."""
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(kv_lens)
    q_lens = (
        [int(q_lens)] * batch if isinstance(q_lens, int) else [int(q) for q in q_lens]
    )
    pages_per = [-(-kv // PAGE) for kv in kv_lens]
    total_pages = sum(pages_per) + 1
    pool_pages = pool_pages or total_pages
    assert pool_pages >= total_pages
    max_pages = max(1, max(pages_per))
    perm = torch.randperm(pool_pages, generator=gen, device=device)[:total_pages].to(
        torch.int32
    )
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(pages_per):
        block_tables[b, :n] = perm[off : off + n]
        off += n
    if values == "normal":
        ckv = (
            torch.randn((pool_pages, PAGE, LATENT), generator=gen, device=device)
            .to(torch.bfloat16)
            .float()
        )
        kpe = (
            (torch.randn((pool_pages, PAGE, ROPE), generator=gen, device=device) * 2.5)
            .to(torch.bfloat16)
            .float()
        )
        ckv_scale = float(ckv.abs().amax()) / (E2M1_MAX * E4M3_MAX)
        kpe_scale = float(kpe.abs().amax()) / E4M3_MAX
        ckv_packed, ckv_sf = _nvfp4_quantize_blocks(
            ckv, torch.full((pool_pages, PAGE), ckv_scale, device=device)
        )
        kpe_fp8 = (kpe / kpe_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    elif values == "bytes":
        scale_bytes = torch.tensor(
            [0x30, 0x34, 0x38, 0x3C, 0x40], dtype=torch.uint8, device=device
        )
        ckv_packed = torch.randint(
            0,
            256,
            (pool_pages, PAGE, CKV_BYTES),
            dtype=torch.int32,
            device=device,
            generator=gen,
        ).to(torch.uint8)
        ckv_sf = scale_bytes[
            torch.randint(
                0, 5, (pool_pages, PAGE, SF_BYTES), device=device, generator=gen
            )
        ].view(torch.float8_e4m3fn)
        kpe_fp8 = (
            torch.randint(
                0,
                120,
                (pool_pages, PAGE, ROPE),
                dtype=torch.int32,
                device=device,
                generator=gen,
            )
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
        )
        ckv_scale, kpe_scale = 1.0, 1.0
    else:
        raise ValueError(values)
    if packed_cache:
        # vLLM-style single allocation: 256 FP4 bytes | 64 rope bytes | 32 scale bytes per token.
        buffer = torch.zeros(
            (pool_pages, PAGE, CKV_BYTES + ROPE + SF_BYTES),
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
    total_q = sum(q_lens)
    q = torch.randn((total_q, num_heads, QK_DIM), generator=gen, device=device).to(
        torch.bfloat16
    )
    q_nope, q_sf, q_rope, q_scale = _quantize_query_reference(q, ckv_scale, kpe_scale)
    q_indptr = [0]
    for ql in q_lens:
        q_indptr.append(q_indptr[-1] + ql)
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
        q_lens=q_lens,
        kv_lens=[int(k) for k in kv_lens],
        kv_lens_global=[
            int(k) for k in (kv_lens if kv_lens_global is None else kv_lens_global)
        ],
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
        num_heads=num_heads,
        o_scale=o_scale,
        return_lse=return_lse,
    )


def _global_positions(
    n: int, world: int, rank: int, interleave: int, device
) -> torch.Tensor:
    """The global positions of rank ``rank``'s local keys ``0 .. n`` under the cyclic sharding."""
    k = torch.arange(n, device=device)
    return (
        torch.div(k, interleave, rounding_mode="floor") * world + rank
    ) * interleave + k % interleave


def _local_count(kv_global: int, world: int, rank: int, interleave: int) -> int:
    """How many of a request's ``kv_global`` keys rank ``rank`` holds (brute force)."""
    if kv_global == 0:
        return 0
    gpos = _global_positions(kv_global, world, rank, interleave, "cpu")
    return int((gpos < kv_global).sum())


def _row_limits(case) -> list[int]:
    """The keys every row sees, by brute force over the global positions (independent of the package's rule)."""
    world, rank, inter = case["cp_world"], case["cp_rank"], case["cp_interleave"]
    limits = []
    for kv, ql, g in zip(
        case["kv_lens"], case["q_lens"], case["kv_lens_global"], strict=True
    ):
        if world == 1:
            counts = [max(0, min(kv, g - ql + t + 1)) for t in range(ql)]
        else:
            gpos = _global_positions(kv, world, rank, inter, "cpu")
            counts = [int((gpos <= g - ql + t).sum()) for t in range(ql)]
        for n in counts:
            limits.extend([n] * case["num_heads"])
    return limits


def _dequantized_query(case) -> torch.Tensor:
    qn = (
        _decode_e2m1(case["q_nope"])
        * case["q_sf"].float().repeat_interleave(SF_VEC, -1)
        * case["q_scale"][..., None]
    )
    qr = case["q_rope"].float() * (
        case["q_scale"][..., None] * (case["ckv_scale"] / case["kpe_scale"])
    )
    return torch.cat((qn, qr), -1).reshape(-1, QK_DIM)


def _dequantized_keys(case, tok: torch.Tensor) -> torch.Tensor:
    """The FP32 ``[len(tok), 576]`` keys (``[nope, rope]``, global scales applied) of the cache tokens ``tok``."""
    pages, offs = torch.div(tok, PAGE, rounding_mode="floor"), tok % PAGE
    ckv = case["ckv_cache"][pages, offs]
    sf = case["ckv_sf_cache"][pages, offs].float().repeat_interleave(SF_VEC, -1)
    kpe = case["kpe_cache"][pages, offs].float()
    return torch.cat(
        (_decode_e2m1(ckv) * sf * case["ckv_scale"], kpe * case["kpe_scale"]), -1
    )


def _reference(case):
    """FP32 attention over the decoded query and cache under the row limits: ``(O bf16 [rows, 512], LSE f32
    [rows])`` in natural log, ``O = 0`` / ``LSE = -inf`` for a row no key is visible to."""
    device = case["q_nope"].device
    H = case["num_heads"]
    q = _dequantized_query(case)
    rows_total = q.shape[0]
    out = torch.zeros((rows_total, LATENT), dtype=torch.float32, device=device)
    lse = torch.full((rows_total,), -math.inf, dtype=torch.float32, device=device)
    limits = torch.tensor(_row_limits(case), device=device)
    r0 = 0
    for b, (kv, ql) in enumerate(zip(case["kv_lens"], case["q_lens"], strict=True)):
        rows = ql * H
        if kv > 0:
            n_pages = -(-kv // PAGE)
            pages = case["block_tables"][b, :n_pages].long()
            tok = (
                pages[:, None] * PAGE + torch.arange(PAGE, device=device)[None, :]
            ).reshape(-1)[:kv]
            keys = _dequantized_keys(case, tok)
            logits = (q[r0 : r0 + rows] @ keys.T) * SM_SCALE
            limit = limits[r0 : r0 + rows]
            logits = logits.masked_fill(
                torch.arange(kv, device=device)[None, :] >= limit[:, None], -math.inf
            )
            mx = logits.amax(-1, keepdim=True)
            mx_safe = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
            p = torch.exp(logits - mx_safe)
            total = p.sum(-1)
            out[r0 : r0 + rows] = (
                (p @ keys[:, :LATENT])
                / total.clamp(min=1e-30)[:, None]
                * case["o_scale"]
            )
            lse[r0 : r0 + rows] = torch.where(
                total > 0, mx_safe.squeeze(-1) + torch.log(total), lse[r0 : r0 + rows]
            )
        r0 += rows
    return out.to(torch.bfloat16), lse


def _units_pairs(device):
    return cake_backend.device_units_pairs(
        device.index if device.index is not None else 0
    )


def _plan_of(case, device, *, split=None, units=None, pairs=None):
    dev_units, dev_pairs = _units_pairs(device)
    dense = len(set(case["q_lens"])) == 1
    return cake_backend.plan(
        seq_lens_host=case["kv_lens"],
        num_heads=case["num_heads"],
        q_len=case["q_lens"][0] if dense else None,
        cum_seq_lens_q_host=None if dense else case["q_indptr"],
        units=dev_units if units is None else units,
        pairs=dev_pairs if pairs is None else pairs,
        cp_world=case["cp_world"],
        cp_rank=case["cp_rank"],
        cp_interleave=case["cp_interleave"],
        kv_len_global_host=case["kv_lens_global"] if case["cp_world"] > 1 else None,
        split=split,
    )


def _run(
    case,
    *,
    dense=False,
    graph=False,
    split=None,
    units=None,
    pairs=None,
    host_lists=True,
):
    """Prepare and launch the decode on ``case``; ``dense`` reshapes the query to ``[B, q_len, H, .]`` (equal query
    lengths), ``graph`` captures the launch in a CUDA graph after a warm launch.  Returns ``(out [rows, 512], lse
    [rows] or None, runner[, graph])``."""
    device = case["q_nope"].device
    H = case["num_heads"]
    total_q = case["q_nope"].shape[0]
    batch = len(case["kv_lens"])
    plan = _plan_of(case, device, split=split, units=units, pairs=pairs)
    workspace = torch.zeros(
        cake_backend.workspace_bytes(plan), dtype=torch.uint8, device=device
    )
    lead = (total_q, H)
    out = torch.full((*lead, LATENT), math.nan, dtype=torch.bfloat16, device=device)
    lse = (
        torch.full(lead, math.nan, dtype=torch.float32, device=device)
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
        o_scale=case["o_scale"],
        cp_world=case["cp_world"],
        cp_rank=case["cp_rank"],
        cp_interleave=case["cp_interleave"],
        kv_len_global=(
            torch.tensor(case["kv_lens_global"], dtype=torch.int32, device=device)
            if case["cp_world"] > 1
            else None
        ),
        seq_lens_host=case["kv_lens"] if host_lists else None,
        kv_len_global_host=case["kv_lens_global"]
        if (host_lists and case["cp_world"] > 1)
        else None,
        units=units,
        pairs=pairs,
        split=split,
    )
    operands = ("q_nope", "q_sf", "q_rope", "q_scale")
    if dense:
        q_len = case["q_lens"][0]
        assert all(q == q_len for q in case["q_lens"])
        shaped = {
            k: case[k].reshape(batch, q_len, *case[k].shape[1:]) for k in operands
        }
        runner = cake_backend.CakeMlaNvfp4PagedDecode(
            **shaped,
            out=out.view(batch, q_len, H, LATENT),
            lse=None if lse is None else lse.view(batch, q_len, H),
            **kwargs,
        )
    else:
        runner = cake_backend.CakeMlaNvfp4PagedDecode(
            **{k: case[k] for k in operands},
            out=out,
            lse=lse,
            cum_seq_lens_q=torch.tensor(
                case["q_indptr"], dtype=torch.int32, device=device
            ),
            cum_seq_lens_q_host=case["q_indptr"] if host_lists else None,
            max_q_len=max(case["q_lens"]),
            **kwargs,
        )
    assert (
        runner.workspace_bytes
        == workspace.numel()
        == cake_backend.workspace_bytes(runner.plan)
    )
    assert runner.plan == plan
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


def _check_out(out, ref):
    got, want = out.float().reshape(-1, LATENT), ref.float().reshape(-1, LATENT)
    assert torch.isfinite(got).all()
    err = (got - want).abs()
    gate = O_ATOL + O_RTOL * want.abs()
    mismatch = float((err > gate).float().mean())
    assert mismatch <= MAX_MISMATCH, (
        f"{mismatch:.2e} of the elements cross atol {O_ATOL} + rtol {O_RTOL}"
    )
    rel = (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)
    assert float(rel.max()) <= ROW_REL_L2, (
        f"row relative L2 error {float(rel.max()):.4f} > {ROW_REL_L2}"
    )


def _check_lse(lse, ref_lse):
    got, want = lse.reshape(-1), ref_lse.reshape(-1)
    both_inf = torch.isneginf(want) & torch.isneginf(got)
    err = torch.where(both_inf, torch.zeros_like(want), (got - want).abs())
    worst = float(err.nan_to_num(math.inf).max())
    assert worst <= LSE_ATOL, f"LSE error {worst:.4f} > {LSE_ATOL}"


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [16, 96])
@pytest.mark.parametrize(
    "kv_lens", [[1, 64, 65, 200], [513, 1000, 4096, 777], [65536, 40000]]
)
def test_decode_dense_q1(num_heads, kv_lens):
    device = _device_or_skip()
    case = _make_case(
        kv_lens, 1, num_heads, seed=625001, device=device, return_lse=True
    )
    out, lse, runner = _run(case, dense=True)
    assert runner.plan.rows_total == len(kv_lens) * num_heads
    ref, ref_lse = _reference(case)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)


@pytest.mark.parametrize(
    "q_len,kv_lens,values",
    [
        (
            4,
            [65536],
            "normal",
        ),  # Q4 B1 KV64K: one full tile and one half tile per request, split over the clusters
        (
            8,
            [65536],
            "bytes",
        ),  # Q8 B1 KV64K: three full tiles, no half tile (the decode_full build)
        (4, [65536, 40000, 1000, 65], "normal"),
        (
            8,
            [8192] * 16,
            "bytes",
        ),  # more items than clusters: cluster launch control hands out the rest
        (8, [256] * 16, "bytes"),  # short requests, split
    ],
)
def test_decode_dense_h96(q_len, kv_lens, values):
    device = _device_or_skip()
    case = _make_case(
        kv_lens, q_len, 96, seed=625002, device=device, values=values, return_lse=True
    )
    out, lse, runner = _run(case, dense=True)
    rows = q_len * 96
    half = rows - (-(-rows // ROWS_TILE) - 1) * ROWS_TILE <= ROWS_TILE // 2
    assert runner.plan.decode_record == ("decode" if half else "decode_full")
    ref, ref_lse = _reference(case)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)


@pytest.mark.parametrize(
    "num_heads,q_lens,kv_lens",
    [
        (
            96,
            [1, 4, 8, 3, 8],
            [513, 1000, 64, 65, 1],
        ),  # half tiles, a partial page, a 1-token and a 1-key request
        (16, [2, 1], [4100, 999]),
        (
            128,
            [1, 1, 1],
            [100, 3000, 6001],
        ),  # DeepSeek-V3 TP1 head count: half tiles of 128 rows
        (96, [4, 2, 8], [2000, 300, 1500]),
    ],
)
def test_packed_variable_query(num_heads, q_lens, kv_lens):
    device = _device_or_skip()
    case = _make_case(
        kv_lens, q_lens, num_heads, seed=625003, device=device, return_lse=True
    )
    out, lse, _ = _run(case)
    ref, ref_lse = _reference(case)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)


def test_packed_352_byte_cache_layout_matches_separate_tensors():
    device = _device_or_skip()
    kv_lens, q_lens = [777, 2048], [8, 4]
    separate = _make_case(kv_lens, q_lens, 96, seed=625012, device=device)
    packed = _make_case(
        kv_lens, q_lens, 96, seed=625012, device=device, packed_cache=True
    )
    assert packed["ckv_cache"].stride(-2) == CKV_BYTES + ROPE + SF_BYTES
    out_a, _, _ = _run(separate)
    out_b, _, _ = _run(packed)
    assert torch.equal(out_a, out_b)
    ref, _ = _reference(separate)
    _check_out(out_a, ref)


def test_o_scale_scales_the_output():
    device = _device_or_skip()
    case = _make_case([3000, 129], 1, 16, seed=625013, device=device, o_scale=0.5)
    out, _, _ = _run(case, dense=True)
    ref, _ = _reference(case)
    _check_out(out, ref)


def _shard(full, world, rank, interleave):
    """Rank ``rank``'s problem of the full-context ``full``: its local key counts, a fresh page table and the
    full cache's rows at its global positions; the query and the scales are the full problem's."""
    device = full["q_nope"].device
    local = [_local_count(g, world, rank, interleave) for g in full["kv_lens"]]
    batch = len(local)
    pages_per = [-(-kv // PAGE) for kv in local]
    pool = sum(pages_per) + 1
    max_pages = max(1, max(pages_per))
    gen = torch.Generator(device=device).manual_seed(7000 + rank)
    perm = torch.randperm(pool, generator=gen, device=device).to(torch.int32)
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    ckv = torch.zeros((pool, PAGE, CKV_BYTES), dtype=torch.uint8, device=device)
    sf = torch.zeros((pool, PAGE, SF_BYTES), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    kpe = torch.zeros((pool, PAGE, ROPE), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    off = 0
    for b, n in enumerate(pages_per):
        block_tables[b, :n] = perm[off : off + n]
        off += n
        if local[b] == 0:
            continue
        k = torch.arange(local[b], device=device)
        gpos = _global_positions(local[b], world, rank, interleave, device)
        assert int(gpos.max()) < full["kv_lens"][b]
        src_p, src_o = (
            full["block_tables"][
                b, torch.div(gpos, PAGE, rounding_mode="floor")
            ].long(),
            gpos % PAGE,
        )
        dst_p, dst_o = (
            block_tables[b, torch.div(k, PAGE, rounding_mode="floor")].long(),
            k % PAGE,
        )
        ckv[dst_p, dst_o] = full["ckv_cache"][src_p, src_o]
        sf[dst_p, dst_o] = full["ckv_sf_cache"][src_p, src_o]
        kpe[dst_p, dst_o] = full["kpe_cache"][src_p, src_o]
    case = dict(full)
    case.update(
        ckv_cache=ckv,
        ckv_sf_cache=sf,
        kpe_cache=kpe,
        block_tables=block_tables,
        seq_lens=torch.tensor(local, dtype=torch.int32, device=device),
        kv_lens=local,
        kv_lens_global=list(full["kv_lens"]),
        cp_world=world,
        cp_rank=rank,
        cp_interleave=interleave,
        return_lse=True,
    )
    return case


@pytest.mark.parametrize(
    "kv_lens,q_lens,num_heads,world,interleave",
    [
        (
            [1, 130, 3, 4000],
            [4, 1, 1, 3],
            16,
            4,
            1,
        ),  # tokens no key is visible to; ranks without a key
        ([2, 1000, 517], [3, 2, 5], 128, 2, 1),
        ([5, 333, 4097], [8, 1, 6], 64, 4, 4),  # blocks of 4 keys per rank
        ([3, 700, 20000, 64], [2, 8, 4, 1], 96, 8, 1),
        ([8192] * 4, [1] * 4, 128, 8, 1),  # the serving shape at world 8
    ],
)
def test_decode_context_parallel_ranks_merge_on_host(
    kv_lens, q_lens, num_heads, world, interleave
):
    device = _device_or_skip()
    full = _make_case(
        kv_lens, q_lens, num_heads, seed=625020, device=device, return_lse=True
    )
    outs, lses = [], []
    for rank in range(world):
        case = _shard(full, world, rank, interleave)
        out, lse, _ = _run(case)
        ref, ref_lse = _reference(case)  # each rank against its own shard's softmax
        _check_out(out, ref)
        _check_lse(lse, ref_lse)
        # Rows without a visible key on this rank are zero with LSE -inf (the cross-rank merge contract).
        empty = torch.isneginf(ref_lse)
        if bool(empty.any()):
            assert torch.isneginf(lse.reshape(-1)[empty]).all()
            assert (out.reshape(-1, LATENT)[empty].float() == 0).all()
        outs.append(out.float().reshape(-1, LATENT))
        lses.append(lse.reshape(-1))
    # The serving stack's merge: LSE = ln sum_r exp LSE_r, O = sum_r exp(LSE_r - LSE) O_r.
    stacked = torch.stack(lses)
    lse = torch.logsumexp(stacked, 0)
    lse_safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    w = torch.where(
        torch.isfinite(stacked),
        torch.exp(stacked - lse_safe),
        torch.zeros_like(stacked),
    )
    merged = sum(w[r][:, None] * outs[r] for r in range(world))
    ref, ref_lse = _reference(full)
    _check_out(merged.to(torch.bfloat16), ref)
    _check_lse(lse, ref_lse)


def test_split_merge_up_to_and_past_32_splits():
    device = _device_or_skip()
    # A 1-token H96 request (a half tile, 96 rows: the eight-warp merge) at KV64K: the planner's own split count,
    # then 20 double ranges = 40 partial slots per row (the eight-warp form's 48-slot program), a 3072-row problem at two
    # partials (the one-warp merge above 2048 rows), a 1536-row one at three (two warps: that grid is one wave
    # up to 1664 rows), 28 one-token H16 requests (448 rows) at 24 partials (the four-warp column merge of the
    # 385-448-row box) and a 768-row problem at 24 partials (the four-warp, two-rows-per-CTA merge above it).
    case = _make_case([65536], 1, 96, seed=625030, device=device, return_lse=True)
    out, lse, runner = _run(case, dense=True)
    assert runner.plan.max_splits > 1 and runner.plan.merge_record == "merge_w8_pal"
    ref, ref_lse = _reference(case)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)
    out, lse, runner = _run(case, dense=True, split=(1, 20))
    assert (
        runner.plan.max_splits == 40 and runner.plan.merge_record == "merge_w8_b48_pal"
    )
    _check_out(out, ref)
    _check_lse(lse, ref_lse)
    wide = _make_case(
        [20000, 30000, 4096, 777], 8, 96, seed=625031, device=device, return_lse=True
    )
    out, lse, runner = _run(wide, dense=True, split=2)
    assert runner.plan.max_splits == 2 and runner.plan.merge_record == "merge_w1"
    ref, ref_lse = _reference(wide)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)
    mid = _make_case(
        [9000, 4096], 8, 96, seed=625032, device=device, return_lse=True
    )  # 1536 rows
    out, lse, runner = _run(mid, dense=True, split=3)
    assert runner.plan.max_splits == 3 and runner.plan.merge_record == "merge_w2_pal"
    ref, ref_lse = _reference(mid)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)
    col = _make_case([8192] * 28, 1, 16, seed=625033, device=device, return_lse=True)
    out, lse, runner = _run(col, dense=True, split=(1, 12))  # 448 rows
    assert (
        runner.plan.max_splits == 24
        and runner.plan.merge_record == "merge_w4_c4_col_pal"
    )
    ref, ref_lse = _reference(col)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)
    four = _make_case([65536], 8, 96, seed=625034, device=device, return_lse=True)
    out, lse, runner = _run(four, dense=True, split=24)  # 768 rows
    assert runner.plan.max_splits == 24 and runner.plan.merge_record == "merge_w4"
    ref, ref_lse = _reference(four)
    _check_out(out, ref)
    _check_lse(lse, ref_lse)


def test_query_quantizer_matches_reference_bitwise():
    device = _device_or_skip()
    gen = torch.Generator(device=device).manual_seed(625040)
    # A partial last CTA; the Kimi-K3 H96 rows of Q4 and Q8; zero rows and zero blocks (the degenerate scale
    # paths); rows whose scale the rope sets; large values that saturate the block scale and the rope.
    for rows, scale, ckv_scale, kpe_scale, zero_rows, zero_blocks in (
        (7, 1.0, 1.0, 1.0, 0.0, 0.0),
        (384, 1.0, 1.0, 1.0, 0.1, 0.05),
        (768, 40.0, 0.5, 2.0, 0.0, 0.0),
        (1000, 1e-3, 1e-3, 1.0, 0.0, 0.0),
        (258, 3000.0, 1.0, 1.0, 0.0, 0.0),
    ):
        q = torch.randn((rows, QK_DIM), generator=gen, device=device) * scale
        if zero_blocks:
            blocks = (
                torch.rand((rows, QK_DIM // SF_VEC), generator=gen, device=device)
                < zero_blocks
            )
            q.view(rows, -1, SF_VEC).masked_fill_(blocks[..., None], 0.0)
        if zero_rows:
            q.masked_fill_(
                (torch.rand((rows,), generator=gen, device=device) < zero_rows)[
                    :, None
                ],
                0.0,
            )
        q = q.to(torch.bfloat16)
        workspace = torch.zeros(
            cake_backend.query_workspace_bytes(rows), dtype=torch.uint8, device=device
        )
        got = cake_backend.quantize_mla_nvfp4_query(
            q,
            ckv_scale,
            kpe_scale,
            out=cake_backend.mla_nvfp4_query_buffers(workspace, (rows,)),
        )
        torch.cuda.synchronize()
        want = _quantize_query_reference(q, ckv_scale, kpe_scale)
        for g, w in zip(got, want, strict=True):
            assert g.shape == w.shape and g.dtype == w.dtype
            assert torch.equal(g.view(torch.uint8), w.view(torch.uint8))


def test_complete_call_bf16_query_through_the_one_shot_entry():
    device = _device_or_skip()
    case = _make_case([1000, 2000], 1, 16, seed=625041, device=device)
    batch = 2
    q = case["q_bf16"].reshape(batch, 1, 16, QK_DIM)
    query_workspace = torch.zeros(
        cake_backend.query_workspace_bytes(batch * 16), dtype=torch.uint8, device=device
    )
    quantizer = cake_backend.CakeMlaNvfp4QueryQuantize(
        query=q,
        ckv_scale=case["ckv_scale"],
        kpe_scale=case["kpe_scale"],
        out=cake_backend.mla_nvfp4_query_buffers(query_workspace, (batch, 1, 16)),
    )
    q_nope, q_sf, q_rope, q_scale = quantizer.launch()
    out = torch.empty((batch, 1, 16, LATENT), dtype=torch.bfloat16, device=device)
    # No host lists: the one-shot call reads the device lengths (the documented synchronization).
    got = cake_backend.cake_mla_nvfp4_paged_decode(
        q_nope,
        q_sf,
        q_rope,
        q_scale,
        case["ckv_cache"],
        case["ckv_sf_cache"],
        case["kpe_cache"],
        case["block_tables"],
        case["seq_lens"],
        torch.zeros(
            cake_backend.max_workspace_bytes(batch * 16, batch),
            dtype=torch.uint8,
            device=device,
        ),
        sm_scale=SM_SCALE,
        ckv_scale=case["ckv_scale"],
        out=out,
    )
    torch.cuda.synchronize()
    assert got is out
    ref, _ = _reference(case)
    _check_out(out, ref)


def test_cuda_graph_replay_changing_page_table():
    device = _device_or_skip()
    case = _make_case(
        [900, 1800, 2700, 3600], 1, 16, seed=625050, device=device, pool_pages=192
    )
    # The runner outlives the graph: its workspace (partials, plan block) is what the replays read and write.
    out, _, runner, g = _run(case, dense=True, graph=True)
    ref, _ = _reference(case)
    _check_out(out, ref)
    # New page assignment and cache contents in place; replay must follow the tables.
    gen = torch.Generator(device=device).manual_seed(7)
    pool_pages = case["ckv_cache"].shape[0]
    ckv = (
        torch.randn((pool_pages, PAGE, LATENT), generator=gen, device=device)
        .to(torch.bfloat16)
        .float()
    )
    ckv_packed, ckv_sf = _nvfp4_quantize_blocks(
        ckv, torch.full((pool_pages, PAGE), case["ckv_scale"], device=device)
    )
    case["ckv_cache"].copy_(ckv_packed)
    case["ckv_sf_cache"].copy_(ckv_sf)
    perm = torch.randperm(pool_pages, generator=gen, device=device).to(torch.int32)
    off = 0
    for b in range(4):
        n = -(-case["kv_lens"][b] // PAGE)
        case["block_tables"][b, :n] = perm[off : off + n]
        off += n
    out.fill_(math.nan)
    g.replay()
    torch.cuda.synchronize()
    ref, _ = _reference(case)
    _check_out(out, ref)
    del runner


def test_launch_allocates_nothing_and_repeats_bitwise():
    device = _device_or_skip()
    case = _make_case([5000, 300], 4, 96, seed=625060, device=device, return_lse=True)
    out, lse, runner = _run(case, dense=True)
    first, first_lse = out.clone(), lse.clone()
    out.fill_(math.nan)
    lse.fill_(math.nan)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert torch.equal(out, first) and torch.equal(lse, first_lse)


def test_result_and_query_buffers_are_caller_owned():
    device = _device_or_skip()
    case = _make_case([128], 1, 16, seed=625036, device=device)
    with pytest.raises(ValueError, match="out is required"):
        cake_backend.quantize_mla_nvfp4_query(
            case["q_bf16"], case["ckv_scale"], case["kpe_scale"]
        )
    workspace = torch.zeros(
        cake_backend.query_workspace_bytes(16) - 1, dtype=torch.uint8, device=device
    )
    with pytest.raises(ValueError, match="query workspace needs at least"):
        cake_backend.mla_nvfp4_query_buffers(workspace, (1, 16))
    with pytest.raises(ValueError, match="out .* required"):
        cake_backend.cake_mla_nvfp4_paged_decode(
            case["q_nope"],
            case["q_sf"],
            case["q_rope"],
            case["q_scale"],
            case["ckv_cache"],
            case["ckv_sf_cache"],
            case["kpe_cache"],
            case["block_tables"],
            case["seq_lens"],
            torch.zeros(
                cake_backend.max_workspace_bytes(16, 1),
                dtype=torch.uint8,
                device=device,
            ),
            sm_scale=SM_SCALE,
            ckv_scale=case["ckv_scale"],
        )


def test_rejects_unsupported_inputs():
    device = _device_or_skip()
    case = _make_case([128], 1, 16, seed=1, device=device)
    operands = {k: case[k] for k in ("q_nope", "q_sf", "q_rope", "q_scale")}
    common = dict(
        ckv_cache=case["ckv_cache"],
        ckv_sf_cache=case["ckv_sf_cache"],
        kpe_cache=case["kpe_cache"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        workspace_buffer=torch.zeros(
            cake_backend.max_workspace_bytes(16, 1), dtype=torch.uint8, device=device
        ),
        sm_scale=SM_SCALE,
        ckv_scale=case["ckv_scale"],
        seq_lens_host=case["kv_lens"],
    )
    out = torch.empty((1, 16, LATENT), dtype=torch.bfloat16, device=device)
    packed = dict(cum_seq_lens_q_host=case["q_indptr"])
    # Page size: the decode's TMA box is one 64-token page.
    with pytest.raises(ValueError, match="page_size must be 64"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **operands,
            out=out,
            **dict(
                common,
                ckv_cache=case["ckv_cache"].reshape(-1, 32, CKV_BYTES),
                ckv_sf_cache=case["ckv_sf_cache"].reshape(-1, 32, SF_BYTES),
                kpe_cache=case["kpe_cache"].reshape(-1, 32, ROPE),
            ),
            **packed,
        )
    with pytest.raises(ValueError, match="kv_len_global"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **operands, out=out, cp_world=2, cp_rank=1, **common, **packed
        )
    with pytest.raises(ValueError, match="q_nope must be uint8"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **dict(operands, q_nope=case["q_nope"].view(torch.int8)),
            out=out,
            **common,
            **packed,
        )
    with pytest.raises(ValueError, match="q_sf / q_rope must be float8"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **dict(operands, q_sf=case["q_sf"].view(torch.uint8)),
            out=out,
            **common,
            **packed,
        )
    # A strided query view is refused, never copied.
    wide = torch.zeros((1, 16, 2 * CKV_BYTES), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="must be contiguous"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **dict(operands, q_nope=wide[..., :CKV_BYTES]), out=out, **common, **packed
        )
    # A packed query needs its prefix sums.
    with pytest.raises(ValueError, match="cum_seq_lens_q"):
        cake_backend.CakeMlaNvfp4PagedDecode(**operands, out=out, **common)
    # Every request's rows are a multiple of 16: 8 heads x 1 token are not.
    small = _make_case([128], 1, 8, seed=2, device=device)
    with pytest.raises(ValueError, match="multiple of 16"):
        _run(small, dense=True)
    with pytest.raises(ValueError, match="cp_interleave"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **operands, out=out, cp_interleave=0, **common, **packed
        )
    with pytest.raises(ValueError, match="workspace_buffer needs at least"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **operands,
            out=out,
            **dict(
                common,
                workspace_buffer=torch.zeros(1024, dtype=torch.uint8, device=device),
            ),
            **packed,
        )
    with pytest.raises(ValueError, match="seq_len exceeds"):
        cake_backend.CakeMlaNvfp4PagedDecode(
            **operands, out=out, **dict(common, seq_lens_host=[65 * PAGE]), **packed
        )


def test_every_record_resolves_on_the_running_architecture():
    _device_or_skip()
    import json

    manifest = json.loads((PACKAGE_DIR / "build.json").read_text())
    libraries = {entry["name"] for entry in manifest["libraries"]}
    arch = cake_jit.device_arch(torch.cuda.current_device())
    parameters = {
        "decode": cake_jit.DECODE_PARAMETERS,
        "merge": cake_jit.MERGE_PARAMETERS,
        "quantize": cake_jit.QUANTIZE_PARAMETERS,
    }
    for name, record in cake_jit.MODULES.items():
        assert arch in record["arches"]
        assert cake_jit.library_name(name, arch) in libraries
        # The exporter filled the derived leaves: the argument plan names every kernel parameter in order, the
        # per-call PDL switch of a decode or merge record right before the grid (none for the quantizer) and the
        # three grid arguments; the launch record is a real one.
        assert record["kernel_symbol"]
        plan = record["arg_plan"]
        assert [key for kind, key in plan if kind not in ("grid", "launch")] == list(
            parameters[record["stage"]]
        )
        assert [key for kind, key in plan if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        launch = record["launch"]
        launches = [i for i, (kind, _) in enumerate(plan) if kind == "launch"]
        if launch["use_pdl"] == "runtime":
            assert launches == [len(plan) - 4]
            assert plan[len(plan) - 4][1] == cake_jit.LAUNCH_ARGUMENT
        else:
            assert launches == [] and launch["use_pdl"] is False
        assert len(launch["block"]) == 3 and launch["block"][0] > 0
        assert len(launch["cluster"]) == 3 and all(c >= 1 for c in launch["cluster"])
        assert launch["dynamic_smem_bytes"] >= 0
    # The decode launches clusters (a required cluster of two, a preferred one of four); the others are plain CTAs.
    assert cake_jit.MODULES["decode"]["launch"]["cluster"][0] > 1
    assert cake_jit.MODULES["decode_full"]["launch"]["cluster"][0] > 1
    assert cake_jit.MODULES["quantize"]["launch"]["cluster"] == (1, 1, 1) or list(
        cake_jit.MODULES["quantize"]["launch"]["cluster"]
    ) == [1, 1, 1]


# ---------------------------------------------------------------------------
# CPU tests: registry, rules, the masking rule, the plan and the workspace
# ---------------------------------------------------------------------------


def test_registry_records_follow_the_plan_vocabulary():
    import re

    parameters = {
        "decode": cake_jit.DECODE_PARAMETERS,
        "merge": cake_jit.MERGE_PARAMETERS,
        "quantize": cake_jit.QUANTIZE_PARAMETERS,
    }
    select_kinds = {"always", "all_full", "plan_within", "rows_splits_within"}
    grid_kinds = {"ctas_per_item", "rows_per_cta"}
    assert cake_jit.ARCHES == ("sm_107a",) and cake_jit.ARCH_BY_CAPABILITY == {
        (10, 7): "sm_107a"
    }
    assert cake_jit.STAGES == ("quantize", "decode", "merge")
    for name, record in cake_jit.MODULES.items():
        assert record["arches"] == ["sm_107a"], name
        assert record["ffi_entry"] == "run"
        assert record["stage"] in cake_jit.STAGES
        assert record["select_rule"]["kind"] in select_kinds
        assert record["grid_rule"]["kind"] in grid_kinds
        assert re.fullmatch(
            re.escape(cake_jit.PREFIX) + r"_[0-9a-f]{20}", record["library"]
        ), name
        params = parameters[record["stage"]]
        assert set(record["logical"]) == set(params), name
        assert all(k == v for k, v in record["logical"].items()), name
        # The decode and the merge take the per-call PDL switch; the quantizer launches plain.
        assert record["launch"]["use_pdl"] == (
            "runtime" if record["stage"] in ("decode", "merge") else False
        ), name
        for key in params:
            assert key in cake_jit.LOGICAL_TENSORS or key in cake_jit.LOGICAL_SCALARS, (
                key
            )
    assert cake_jit.records_of_stage("decode") == (
        "decode",
        "decode_full",
        "decode_full_pairpipe",
    )
    assert cake_jit.records_of_stage("merge") == (
        "merge_w1",
        "merge_w4",
        "merge_w2_pal",
        "merge_w8_pal",
        "merge_w8_b48_pal",
        "merge_w4_c4_col_pal",
    )
    assert cake_jit.records_of_stage("quantize") == ("quantize",)
    assert len({r["library"] for r in cake_jit.MODULES.values()}) == len(
        cake_jit.MODULES
    )
    assert set(cake_jit.LOGICAL_TENSORS).isdisjoint(cake_jit.LOGICAL_SCALARS)


def test_select_rules_and_grids():
    assert (
        cake_jit.select_record("decode", rows=768, n_full=3, items=3) == "decode_full"
    )
    assert cake_jit.select_record("decode", rows=384, n_full=1, items=3) == "decode"
    assert cake_jit.select_record("decode", rows=96, n_full=0, items=4) == "decode"
    # The pair-body converter pipeline program takes the one plan box of its rule (64 requests of
    # 8 query tokens over 96 heads, 64K keys: 618 full-tile items, 4 splits); its neighbours keep
    # the all_full / half-tile selection.
    assert (
        cake_jit.select_record("decode", rows=49152, n_full=618, items=618, splits=4)
        == "decode_full_pairpipe"
    )
    assert (
        cake_jit.select_record("decode", rows=49152, n_full=618, items=618, splits=2)
        == "decode_full"
    )
    assert (
        cake_jit.select_record("decode", rows=49152, n_full=617, items=618, splits=4)
        == "decode"
    )
    # (rows, partials per row, record, rows per merge CTA): eight warps per row up to 384 rows or over 32
    # partials (``merge_w8_pal``; up to 384 rows at 33-48 partials its 48-slot program, ``merge_w8_b48_pal``);
    # two warps up to 16 partials up to 1664 rows, and over 8 up to 2048 rows
    # (``merge_w2_pal``); four warps over 17-32 partials (the four-warp column form up to 448 rows, ``merge_w4``
    # above); one warp above 2048 rows or up to 8 partials above 1664 rows.
    cases = (
        (96, 24, "merge_w8_pal", 1),
        (192, 52, "merge_w8_pal", 1),
        (384, 24, "merge_w8_pal", 1),
        (384, 33, "merge_w8_b48_pal", 1),
        (384, 46, "merge_w8_b48_pal", 1),
        (384, 48, "merge_w8_b48_pal", 1),
        (384, 49, "merge_w8_pal", 1),
        (96, 40, "merge_w8_b48_pal", 1),
        (385, 33, "merge_w8_pal", 1),
        (385, 16, "merge_w2_pal", 4),
        (385, 17, "merge_w4_c4_col_pal", 1),
        (385, 24, "merge_w4_c4_col_pal", 1),
        (448, 25, "merge_w4_c4_col_pal", 1),
        (448, 32, "merge_w4_c4_col_pal", 1),
        (448, 33, "merge_w8_pal", 1),
        (449, 24, "merge_w4", 2),
        (768, 12, "merge_w2_pal", 4),
        (768, 16, "merge_w2_pal", 4),
        (768, 20, "merge_w4", 2),
        (768, 33, "merge_w8_pal", 1),
        (1536, 8, "merge_w2_pal", 4),
        (1664, 8, "merge_w2_pal", 4),
        (1665, 8, "merge_w1", 8),
        (1792, 6, "merge_w1", 8),
        (1792, 12, "merge_w2_pal", 4),
        (2048, 33, "merge_w8_pal", 1),
        (2049, 33, "merge_w1", 8),
        (3072, 10, "merge_w1", 8),
        (24576, 4, "merge_w1", 8),
    )
    for r, splits, w, rows_per_cta in cases:
        assert (
            cake_jit.select_record("merge", rows=r, n_full=0, items=1, splits=splits)
            == w
        )
        assert cake_jit.merge_warps_per_row(r, splits) == _merge_warps(w)
        assert cake_jit.MODULES[w]["grid_rule"] == {
            "kind": "rows_per_cta",
            "rows": rows_per_cta,
        }
        assert cake_jit.launch_grid(w, items=1, rows=r) == (-(-r // rows_per_cta), 1, 1)
    assert cake_jit.select_record("quantize", rows=1, n_full=0, items=0) == "quantize"
    assert cake_jit.launch_grid("decode", items=7, rows=96) == (28, 1, 1)
    assert cake_jit.launch_grid("decode_full", items=46, rows=768) == (184, 1, 1)
    assert cake_jit.launch_grid("quantize", items=0, rows=7) == (2, 1, 1)
    assert cake_jit.launch_grid("quantize", items=0, rows=49152) == (12288, 1, 1)
    with pytest.raises(ValueError):
        cake_jit.launch_grid("decode", items=0, rows=96)
    with pytest.raises(ValueError):
        cake_jit.select_record("merge", rows=0, n_full=0, items=1, splits=2)
    with pytest.raises(ValueError):
        cake_jit.records_of_stage("reduce")


def test_visible_keys_matches_brute_force():
    for world in (1, 2, 4, 8):
        for interleave in (1, 4, 16):
            for g in (1, 2, 3, 63, 64, 65, 200, 1000):
                for rank in range(world):
                    n = _local_count(g, world, rank, interleave)
                    gpos = (
                        _global_positions(n, world, rank, interleave, "cpu")
                        if n
                        else None
                    )
                    for bound in (
                        -2,
                        -1,
                        0,
                        1,
                        2,
                        62,
                        63,
                        64,
                        65,
                        g - 2,
                        g - 1,
                        g,
                        g + 5,
                    ):
                        want = int((gpos <= bound).sum()) if n else 0
                        assert (
                            cake_backend.visible_keys(bound, n, world, rank, interleave)
                            == want
                        )
    # The plain case: clamp(bound + 1, 0, kv_len).
    assert cake_backend.visible_keys(-1, 10, 1, 0, 1) == 0
    assert cake_backend.visible_keys(3, 10, 1, 0, 1) == 4
    assert cake_backend.visible_keys(100, 10, 1, 0, 1) == 10
    limits = cake_backend.row_limits([5, 0, 3], [2, 1, 4], 2, [5, 0, 3])
    assert limits == [4, 4, 5, 5, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3]


def test_units_pairs_table_and_fallback():
    assert cake_backend.units_pairs_for_sm_count(212) == (46, 14)
    assert cake_backend.units_pairs_for_sm_count(208) == (50, 4)
    assert cake_backend.units_pairs_for_sm_count(216) == (54, 0)
    assert cake_backend.units_pairs_for_sm_count(3) == (1, 0)
    with pytest.raises(ValueError):
        cake_backend.units_pairs_for_sm_count(0)
    with pytest.raises(ValueError):
        cake_backend.plan(seq_lens_host=[64], num_heads=16, q_len=1, units=0, pairs=0)


def _merge_warps(record):
    """Warps per row of a merge record, from its name (``merge_w<warps>...``)."""
    return int(re.match(r"merge_w(\d)", record).group(1))


@pytest.mark.parametrize("units, pairs", [(50, 4), (49, 6)])
@pytest.mark.parametrize("kv_len", [65536, 131072])
def test_one_full_half_request_fills_the_wave(units, pairs, kv_len):
    """One request whose rows make one full tile and one half tile (H96 x Q4 = 384 rows at B1) plans the pair of
    split counts that fills the wave (``units + pairs // 2`` items: 52 on both 208-SM tables) at the half-tile
    factor the forced plans fitted (29 full and 23 half splits at 64K keys, 29 full and 23 half splits at 128K),
    which the half tile's grid of counts misses; the plan keeps every invariant."""
    p = cake_backend.plan(
        seq_lens_host=[kv_len], num_heads=96, q_len=4, units=units, pairs=pairs
    )
    assert (p.items, p.n_full, p.max_splits) == {
        65536: (52, 29, 46),
        131072: (52, 29, 46),
    }[kv_len]
    assert p.merge_record == "merge_w8_b48_pal" and p.grid_decode == (208, 1, 1)
    _check_plan(p, [kv_len], [4], 96)
    if (units, pairs) == (50, 4):
        # Two such requests are not in the class (their wave holds both): the plan keeps the grid's counts.
        q = cake_backend.plan(
            seq_lens_host=[kv_len, kv_len],
            num_heads=96,
            q_len=4,
            units=units,
            pairs=pairs,
        )
        assert (q.items, q.n_full, q.max_splits) == (52, 32, 20)
        assert q.merge_record == "merge_w4"
        _check_plan(q, [kv_len, kv_len], [4, 4], 96)


def _check_plan(p, kv_lens, q_lens, heads):
    """The plan's invariants against its problem: every (request, row tile) covered by items whose key ranges
    partition its tiles, split indices unique, the full-tile items first, the records and grids of the rules."""
    batch = len(kv_lens)
    rows_total = sum(q_lens) * heads
    assert p.batch == batch and p.num_heads == heads and p.rows_total == rows_total
    assert p.q_lens == tuple(q_lens) and p.kv_lens == tuple(kv_lens)
    assert (
        len(p.item_table) == p.items * 6 and 0 <= p.n_full <= p.items and p.items >= 1
    )
    assert 1 <= p.max_splits <= MAX_SPLITS
    assert p.decode_record == ("decode_full" if p.n_full == p.items else "decode")
    if p.max_splits == 1:
        assert p.merge_record is None and p.grid_merge is None
    else:
        assert p.merge_record == cake_jit.select_record(
            "merge",
            rows=rows_total,
            n_full=p.n_full,
            items=p.items,
            splits=p.max_splits,
        )
        assert _merge_warps(p.merge_record) == cake_jit.merge_warps_per_row(
            rows_total, p.max_splits
        )
        assert p.grid_merge == cake_jit.launch_grid(
            p.merge_record, items=p.items, rows=rows_total
        )
    assert p.grid_decode == (4 * p.items, 1, 1)
    assert p.q_indptr == tuple([0] + [sum(q_lens[: b + 1]) for b in range(batch)])
    assert len(p.row_limit) == rows_total and all(0 <= lim <= kv for lim, kv in zip(
        p.row_limit, [kv for kv, ql in zip(kv_lens, q_lens, strict=True) for _ in range(ql * heads)], strict=True))  # fmt: skip
    tiles_of = [max(1, -(-kv // (2 * PAGE))) for kv in kv_lens]
    rows = [tuple(p.item_table[6 * i : 6 * i + 6]) for i in range(p.items)]
    by_tile: dict[tuple[int, int], list[tuple[int, tuple[int, ...]]]] = {}
    for i, (b, m, lo, n, z, s) in enumerate(rows):
        assert 0 <= b < batch and 0 <= m < -(-q_lens[b] * heads // ROWS_TILE)
        assert 0 <= lo <= lo + n <= tiles_of[b] and n >= 0 and 1 <= s <= p.max_splits
        by_tile.setdefault((b, m), []).append((i, (lo, n, z, s)))
    expected = {
        (b, m) for b in range(batch) for m in range(-(-q_lens[b] * heads // ROWS_TILE))
    }
    assert set(by_tile) == expected
    for (b, m), entries in by_tile.items():
        half = q_lens[b] * heads - m * ROWS_TILE <= ROWS_TILE // 2
        indices = [i for i, _ in entries]
        assert (
            all(i >= p.n_full for i in indices)
            if half
            else all(i < p.n_full for i in indices)
        )
        splits = {s for _, (_, _, _, s) in entries}
        assert len(splits) == 1
        s = splits.pop()
        zs = sorted(z for _, (_, _, z, _) in entries)
        if half:
            assert s % 2 == 0 and zs == list(
                range(s // 2)
            )  # one table row per double item, two ranges each
        else:
            # A range may hold no tile (a short request cut by the batch's split count): the kernel writes -inf
            # for it and the merge weighs it nothing.
            assert zs == list(range(s))
        ranges = sorted((lo, n) for _, (lo, n, _, _) in entries)
        lo_expect = 0
        for lo, n in ranges:
            assert lo == lo_expect
            lo_expect += n
        assert lo_expect == tiles_of[b]
    assert p.items <= cake_backend.max_items(rows_total, batch)
    layout = p.layout()
    assert [r.name for r in layout] == list(cake_backend.WORKSPACE_SLOTS)
    assert all(r.offset % cake_backend.ALIGN == 0 for r in layout)
    assert layout[0].shape == (rows_total, p.max_splits, LATENT) and layout[
        1
    ].shape == (rows_total, p.max_splits)
    assert layout[2].shape == (rows_total,) and layout[3].shape == (len(p.plan_block),)
    assert cake_backend.workspace_bytes(p) <= cake_backend.max_workspace_bytes(
        rows_total, batch
    )
    assert cake_backend.workspace_bytes(p) == p.workspace_bytes


@pytest.mark.parametrize("sm_count", [208, 212, 216])
def test_plan_invariants(sm_count):
    units, pairs = cake_backend.units_pairs_for_sm_count(sm_count)
    configs = [c for c in OWNER_CONFIGS if sm_count == 212 or c[2] in (1, 16)]
    for heads, q_len, batch, kv in configs:
        kv_lens, q_lens = [kv] * batch, [q_len] * batch
        p = cake_backend.plan(
            seq_lens_host=kv_lens,
            num_heads=heads,
            q_len=q_len,
            units=units,
            pairs=pairs,
        )
        _check_plan(p, kv_lens, q_lens, heads)
        if q_len == 8:
            assert (
                p.n_full == p.items and p.decode_record == "decode_full"
            )  # 768 rows: three full tiles
        else:
            assert (
                p.n_full < p.items and p.decode_record == "decode"
            )  # 384 rows: one full and one half tile
    # One-token requests (a half tile at H96): the planner keeps the merge's 64 partial slots per row.
    for kv, batch in ((65536, 1), (131072, 1), (65536, 4), (32768, 16)):
        p = cake_backend.plan(
            seq_lens_host=[kv] * batch, num_heads=96, q_len=1, units=units, pairs=pairs
        )
        _check_plan(p, [kv] * batch, [1] * batch, 96)
        assert (
            p.n_full == 0 and p.decode_record == "decode" and p.max_splits <= MAX_SPLITS
        )
    # Short and odd lengths, a request without keys, packed query lengths, a small head count.
    for kv_lens, q_lens, heads in (
        ([1, 64, 65, 200, 513, 1000], [1] * 6, 16),
        ([513, 1000, 64, 65, 1], [1, 4, 8, 3, 8], 96),
        ([0, 130, 3, 4000], [4, 1, 1, 3], 16),
        ([2, 1000, 517], [3, 2, 5], 128),
        ([65536], [1], 96),
    ):
        p = cake_backend.plan(
            seq_lens_host=kv_lens,
            num_heads=heads,
            cum_seq_lens_q_host=[0]
            + [sum(q_lens[: b + 1]) for b in range(len(q_lens))],
            units=units,
            pairs=pairs,
        )
        _check_plan(p, kv_lens, q_lens, heads)
    # Cached per distinct argument tuple; a forced split is honoured.
    a = cake_backend.plan(
        seq_lens_host=[65536], num_heads=96, q_len=1, units=units, pairs=pairs
    )
    assert a is cake_backend.plan(
        seq_lens_host=[65536], num_heads=96, q_len=1, units=units, pairs=pairs
    )
    forced = cake_backend.plan(
        seq_lens_host=[65536],
        num_heads=96,
        q_len=1,
        units=units,
        pairs=pairs,
        split=(1, 20),
    )
    assert forced.max_splits == 40 and forced.merge_record == "merge_w8_b48_pal"
    _check_plan(forced, [65536], [1], 96)
    with pytest.raises(ValueError, match="at most 64"):
        cake_backend.plan(
            seq_lens_host=[65536],
            num_heads=96,
            q_len=1,
            units=units,
            pairs=pairs,
            split=(1, 40),
        )


def test_pdl_follows_the_plan():
    """The decode and the merge launch with programmatic dependent launch when the plan has a split merge and its
    decode is one hardware wave (``items <= units + pairs``: the clusters of four plus the pairs the device holds
    resident); the switch is a pure function of the plan."""
    import dataclasses

    units, pairs = cake_backend.units_pairs_for_sm_count(208)
    p = cake_backend.plan(
        seq_lens_host=[65536], num_heads=96, q_len=1, units=units, pairs=pairs
    )
    assert cake_backend.pdl_for(p) is (
        p.max_splits > 1 and p.items <= p.units + p.pairs
    )
    one_wave = dataclasses.replace(p, max_splits=2, items=units)
    assert cake_backend.pdl_for(one_wave) is True
    # The pairs still fit the hardware wave (204-208 CTAs on a 208-SM board).
    assert (
        cake_backend.pdl_for(dataclasses.replace(one_wave, items=units + pairs)) is True
    )
    # A second wave: the early-launched merge would take its SMs.
    assert (
        cake_backend.pdl_for(dataclasses.replace(one_wave, items=units + pairs + 1))
        is False
    )
    # No merge: nothing behind the decode to overlap.
    assert cake_backend.pdl_for(dataclasses.replace(one_wave, max_splits=1)) is False
    # The owner's largest row (Q8, B64) has more items than clusters on every board: plain launches.
    wide = cake_backend.plan(
        seq_lens_host=[65536] * 64, num_heads=96, q_len=8, units=units, pairs=pairs
    )
    assert wide.items > wide.units + wide.pairs and cake_backend.pdl_for(wide) is False


def test_plan_shaves_one_wave_of_long_full_tiles():
    """One wave of nearly as many full tiles as clusters over 1024 key tiles takes the shaved plan: every tile a
    long range and a short one of ``MLA2_SHAVE_TILES`` tiles, the long ranges first, two partials per row behind a
    ``merge_w1`` merge and no direct item; the same shapes over 64K keys and a second wave of tiles keep the
    equal-length plans."""
    units, pairs = cake_backend.units_pairs_for_sm_count(208)
    shave = cake_backend.MLA2_SHAVE_TILES
    tiles = 131072 // (2 * PAGE)
    for heads, q_len, batch in ((96, 8, 16), (192, 1, 48), (448, 1, 24)):
        kv_lens, q_lens = [131072] * batch, [q_len] * batch
        n_tiles = batch * -(
            -q_len * heads // ROWS_TILE
        )  # 192, 448 and 768 rows: one, two and three full tiles
        assert 0.8 * units <= n_tiles <= units
        p = cake_backend.plan(
            seq_lens_host=kv_lens,
            num_heads=heads,
            q_len=q_len,
            units=units,
            pairs=pairs,
        )
        _check_plan(p, kv_lens, q_lens, heads)
        assert p.n_full == p.items == 2 * n_tiles and p.max_splits == 2
        assert p.decode_record == "decode_full" and p.merge_record == "merge_w1"
        rows = [tuple(p.item_table[6 * i : 6 * i + 6]) for i in range(p.items)]
        assert [r[2:] for r in rows[:n_tiles]] == [(0, tiles - shave, 0, 2)] * n_tiles
        assert [r[2:] for r in rows[n_tiles:]] == [
            (tiles - shave, shave, 1, 2)
        ] * n_tiles
        assert [r[:2] for r in rows[:n_tiles]] == [r[:2] for r in rows[n_tiles:]]
        # Two items per cluster: more items than resident clusters, so the launches are plain.
        assert p.items > units and cake_backend.pdl_for(p) is False
        q = cake_backend.plan(
            seq_lens_host=[65536] * batch,
            num_heads=heads,
            q_len=q_len,
            units=units,
            pairs=pairs,
        )
        assert (
            q.items == q.n_full == n_tiles
            and q.max_splits == 1
            and q.merge_record is None
        )
    # More full tiles than clusters (the 212-SM table): a second wave, no shaved range.
    units, pairs = cake_backend.units_pairs_for_sm_count(212)
    assert units < 48
    p = cake_backend.plan(
        seq_lens_host=[131072] * 16, num_heads=96, q_len=8, units=units, pairs=pairs
    )
    _check_plan(p, [131072] * 16, [8] * 16, 96)
    rows = [tuple(p.item_table[6 * i : 6 * i + 6]) for i in range(p.items)]
    assert not any(r[2:4] == (tiles - shave, shave) for r in rows)


def test_plan_rejects_malformed_problems():
    with pytest.raises(ValueError, match="multiple of 16"):
        cake_backend.plan(seq_lens_host=[64], num_heads=8, q_len=1, units=46, pairs=14)
    with pytest.raises(ValueError, match="exactly one of"):
        cake_backend.plan(seq_lens_host=[64], num_heads=16, units=46, pairs=14)
    with pytest.raises(ValueError, match="non-decreasing"):
        cake_backend.plan(
            seq_lens_host=[64, 64],
            num_heads=16,
            cum_seq_lens_q_host=[0, 2, 1],
            units=46,
            pairs=14,
        )
    with pytest.raises(ValueError, match="at least one query token"):
        cake_backend.plan(
            seq_lens_host=[64, 64],
            num_heads=16,
            cum_seq_lens_q_host=[0, 1, 1],
            units=46,
            pairs=14,
        )
    with pytest.raises(ValueError, match="cp_rank"):
        cake_backend.plan(
            seq_lens_host=[64],
            num_heads=16,
            q_len=1,
            units=46,
            pairs=14,
            cp_world=2,
            cp_rank=2,
        )
    with pytest.raises(ValueError, match="kv_len_global_host"):
        cake_backend.plan(
            seq_lens_host=[64],
            num_heads=16,
            q_len=1,
            units=46,
            pairs=14,
            cp_world=2,
            kv_len_global_host=[1, 2],
        )
    with pytest.raises(ValueError, match="non-negative"):
        cake_backend.plan(seq_lens_host=[-1], num_heads=16, q_len=1, units=46, pairs=14)


def test_dcp_plan_limits_follow_the_sharding():
    # Rank 3 of 4 holds the global positions 3, 7, 11, ...: a 3-key request gives it none, a 130-key one 32.
    kv_global = [3, 130]
    local = [_local_count(g, 4, 3, 1) for g in kv_global]
    assert local == [0, 32]
    p = cake_backend.plan(
        seq_lens_host=local,
        num_heads=16,
        cum_seq_lens_q_host=[0, 1, 2],
        units=4,
        pairs=0,
        cp_world=4,
        cp_rank=3,
        kv_len_global_host=kv_global,
    )
    _check_plan(p, local, [1, 1], 16)
    assert set(p.row_limit[:16]) == {0}  # no key of request 0 on this rank
    assert set(p.row_limit[16:]) == {
        32
    }  # every local key of request 1 is within the causal bound 129


def test_workspace_bytes_bounds():
    rows, batch = 768, 1
    bound = cake_backend.max_workspace_bytes(rows, batch)
    assert bound >= rows * MAX_SPLITS * (LATENT * 2 + 4) + rows * 4
    assert cake_backend.max_workspace_bytes(rows, batch, max_splits=4) < bound
    assert cake_backend.max_workspace_bytes(rows, batch, items_max=3) < bound
    with pytest.raises(ValueError):
        cake_backend.max_workspace_bytes(rows, batch, max_splits=65)
    assert cake_backend.query_workspace_bytes(10) == 10 * (
        CKV_BYTES + SF_BYTES + ROPE + 4
    )
    c_nope, c_rope = cake_backend.query_scale_constants(0.0021, 0.0057)
    assert c_nope == pytest.approx(1.0 / (6 * 448)) and c_rope == pytest.approx(
        0.0057 / (448 * 0.0021), rel=1e-6
    )
