"""SM8x ``mm_bf16_fp4`` on the W4A16 records GEMM (one expert, top-1, no activation).

Operands come from the torch quantizer below (not FlashInfer's), the expected value from a
torch matmul on the dequantized operands, and every case checks through the CUDA profiler
that the records GEMM ran and no other GEMM did.
"""

import pytest
import torch

from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] != 8,
    reason="SM8x dense FP4",
)

from flashinfer.gemm import sm8x_fp4_dense as dense  # noqa: E402
from flashinfer.gemm.gemm_bf16_fp4 import (  # noqa: E402
    mm_bf16_fp4,
    prepare_bf16_fp4_weights,
)

KERNEL_SYMBOL = "w4a16_moe_sm80::gemm"
FOREIGN_SYMBOLS = ("cutlass", "cudnn", "cute", "sm90", "sm100", "sm120", "gemv")
GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BACKENDS = (
    "cudnn",
    "cute-dsl",
    "cute-dsl-native",
    "blackwell-native",
    "blackwell-tiled",
)


def _quantize(x: torch.Tensor):
    """NVFP4: packed uint8 [R, K/2] (low nibble = even K), E4M3 bytes [R, K/16], dequantized."""
    r, k = x.shape
    xb = x.float().reshape(r, k // 16, 16)
    scale = (xb.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale.float().clamp_min(1e-30)
    grid = torch.tensor(GRID, device=x.device)
    idx = ((xb / scale_f).abs().clamp(max=6.0).unsqueeze(-1) - grid).abs().argmin(-1)
    neg = (xb < 0) & (idx != 0)
    code = (idx + 8 * neg.to(idx.dtype)).reshape(r, k).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    deq = grid[idx] * torch.where(neg, -1.0, 1.0) * scale.float()
    sf = scale.view(torch.uint8).reshape(r, k // 16)
    return packed.contiguous(), sf.contiguous(), deq.reshape(r, k)


def _swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """[R, C] scale bytes to the 128x4 layout, flat, rows/cols padded to 128/4."""
    r, c = sf.shape
    rp, cp = -(-r // 128) * 128, -(-c // 4) * 4
    ri = torch.arange(rp, device=sf.device)[:, None]
    ci = torch.arange(cp, device=sf.device)[None, :]
    off = (
        (ri // 128) * (cp // 4) * 512
        + (ci // 4) * 512
        + (ri % 32) * 16
        + ((ri % 128) // 32) * 4
        + ci % 4
    )
    padded = torch.zeros(rp, cp, dtype=torch.uint8, device=sf.device)
    padded[:r, :c] = sf
    out = torch.zeros(rp * cp, dtype=torch.uint8, device=sf.device)
    out[off.reshape(-1)] = padded.reshape(-1)
    return out


def _rand(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), device="cuda", generator=g).to(torch.bfloat16)


def _weights(n, k, seed):
    codes, sf, deq = _quantize(_rand(n, k, seed))
    alpha = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    return codes, _swizzle_128x4(sf), alpha, deq


def _check(out, ref, what):
    ref = ref.float()
    err = (out.float() - ref).abs().max().item()
    assert err <= 2e-2 * ref.abs().max().item() + 1e-6, f"{what}: max abs err {err}"


def _run_profiled(fn):
    fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    return result, [e.key for e in prof.key_averages() if e.device_type.name == "CUDA"]


def _run_checked(fn, ref, what):
    out, names = _run_profiled(fn)
    assert any(KERNEL_SYMBOL in s for s in names), (
        f"{what}: no {KERNEL_SYMBOL}: {names}"
    )
    others = [s for s in names if "w4a16_moe_sm80" not in s]
    foreign = [s for s in others if any(f in s.lower() for f in FOREIGN_SYMBOLS)]
    assert not foreign, f"{what}: another GEMM ran: {foreign}"
    _check(out, ref, what)
    return out


@pytest.mark.parametrize(
    "backend,out_dtype",
    [(b, torch.bfloat16) for b in BACKENDS]
    + [(b, torch.float16) for b in ("cudnn", "cute-dsl-native", "blackwell-native")],
)
@pytest.mark.parametrize("m", [1, 16, 130])
def test_backends(backend, out_dtype, m):
    """Each backend name through its own prepare: the records, then the GEMM."""
    n, k = 512, 1024
    codes, sf, alpha, deq = _weights(n, k, 11)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(codes, sf, alpha, backend=backend)
    assert b_p.dtype == torch.uint8 and b_p.shape[0] == n and sf_p.numel() == 0
    assert alpha_p is alpha
    a = _rand(m, k, 12)
    _run_checked(
        lambda: mm_bf16_fp4(
            a, b_p, sf_p, alpha_p, backend=backend, out_dtype=out_dtype
        ),
        (a.float() @ deq.T) * 0.75,
        f"{backend} {out_dtype} m={m}",
    )


def _sm100_format(fmt, codes, sf, alpha):
    """The weight as an SM100+ ``prepare_bf16_fp4_weights`` returns it."""
    from flashinfer.gemm import gemm_bf16_fp4_blackwell as bw
    from flashinfer.gemm import gemm_bf16_fp4_cute_dsl as cd

    if fmt == "linear":
        return bw._prepare_blackwell_bf16_fp4(codes, sf, alpha, 16, "blackwell-native")
    if fmt == "tiled":
        return bw._prepare_blackwell_bf16_fp4(codes, sf, alpha, 16, "blackwell-tiled")
    if fmt == "int32":
        return cd._prepare_cute_dsl_sm12x(codes, sf, alpha, 16)
    if fmt == "swz6d":
        return cd._prepare_cute_dsl_sm100(codes, sf, alpha, 16)
    return codes, sf, alpha  # canonical (cute-dsl-native)


@pytest.mark.parametrize(
    "backend,fmt",
    [
        ("cudnn", "linear"),
        ("blackwell-native", "linear"),
        ("blackwell-tiled", "tiled"),
        ("cute-dsl", "int32"),
        ("cute-dsl", "swz6d"),
        ("cute-dsl-native", "canonical"),
    ],
)
@pytest.mark.parametrize("m", [1, 40])
def test_sm100_formats_match_records(backend, fmt, m, monkeypatch):
    """An SM100+-prepared weight gives the records' output bit for bit (same tactic)."""
    from flashinfer.gemm import gemm_bf16_fp4_blackwell as bw

    # Build the SM100 / SM103 layouts with the upstream prepare body on this device.
    monkeypatch.setattr(bw, "_require_blackwell_source_arch", lambda device: None)
    n, k = 256, 512
    codes, sf, alpha, deq = _weights(n, k, 21)
    a = _rand(m, k, 22)
    b_r, sf_r, _ = prepare_bf16_fp4_weights(codes, sf, alpha, backend=backend)
    expected = mm_bf16_fp4(a, b_r, sf_r, alpha, backend=backend)
    b_f, sf_f, alpha_f = _sm100_format(fmt, codes, sf, alpha)
    out = _run_checked(
        lambda: mm_bf16_fp4(a, b_f, sf_f, alpha_f, backend=backend),
        (a.float() @ deq.T) * 0.75,
        f"{backend} {fmt} m={m}",
    )
    assert torch.equal(out, expected)


@pytest.mark.parametrize("tactic", dense._TACTICS)
@pytest.mark.parametrize("m", [1, 33, 200])
def test_every_tactic(tactic, m):
    n, k = 256, 512
    codes, sf, alpha, deq = _weights(n, k, 31)
    b_p, sf_p, _ = prepare_bf16_fp4_weights(codes, sf, alpha, backend="cudnn")
    a = _rand(m, k, 32)
    rec, n_rec = dense._call_records(a, b_p, sf_p, "cudnn")
    assert n_rec == n

    def run():
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        dense._run(a, rec, n, alpha, out, tactic)
        return out

    _run_checked(run, (a.float() @ deq.T) * 0.75, f"tactic {tactic} m={m}")


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,k", [(72, 160), (100, 256), (8, 16), (200, 4096)])
@pytest.mark.parametrize("m", [3, 70])
def test_unaligned_shapes(n, k, m, out_dtype):
    """N % 128 != 0, N % 8 != 0 (through the BF16 buffer) and K % 128 != 0."""
    codes, sf, alpha, deq = _weights(n, k, 41)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(codes, sf, alpha, backend="cudnn")
    a = _rand(m, k, 42)
    out = _run_checked(
        lambda: mm_bf16_fp4(
            a, b_p, sf_p, alpha_p, backend="cudnn", out_dtype=out_dtype
        ),
        (a.float() @ deq.T) * 0.75,
        f"n={n} k={k} m={m} {out_dtype}",
    )
    assert out.shape == (m, n) and out.dtype == out_dtype


def test_out_in_place_and_unaligned():
    """A preallocated out is written in place, also at a 2 B offset (BF16 buffer, copy)."""
    n, k, m = 256, 512, 9
    codes, sf, alpha, deq = _weights(n, k, 51)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(codes, sf, alpha, backend="cudnn")
    a = _rand(m, k, 52)
    ref = (a.float() @ deq.T) * 0.75
    for offset in (0, 1):
        store = torch.empty(m * n + 1, dtype=torch.bfloat16, device="cuda")
        out = store[offset : offset + m * n].view(m, n)
        res = mm_bf16_fp4(a, b_p, sf_p, alpha_p, out=out, backend="cudnn")
        assert res.data_ptr() == out.data_ptr()
        _check(out, ref, f"out offset {offset}")


def test_alpha_none_and_errors():
    n, k, m = 256, 512, 4
    codes, sf, alpha, deq = _weights(n, k, 61)
    b_p, sf_p, _ = prepare_bf16_fp4_weights(codes, sf, None, backend="cudnn")
    a = _rand(m, k, 62)
    _check(mm_bf16_fp4(a, b_p, sf_p, None, backend="cudnn"), a.float() @ deq.T, "None")
    with pytest.raises(ValueError):  # a's K does not match the records
        mm_bf16_fp4(_rand(m, 2 * k, 63), b_p, sf_p, None, backend="cudnn")
    with pytest.raises(ValueError):
        out = torch.empty((m, n + 8), dtype=torch.bfloat16, device="cuda")
        mm_bf16_fp4(a, b_p, sf_p, None, backend="cudnn", out=out)
    with pytest.raises(TypeError):
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        mm_bf16_fp4(a, b_p, sf_p, None, backend="cudnn", out=out)
    with pytest.raises(NotImplementedError):
        mm_bf16_fp4(a, b_p, sf_p, None, backend="cute-dsl", out_dtype=torch.float16)


def test_cuda_graph_replay():
    n, k, m = 256, 512, 24
    codes, sf, alpha, deq = _weights(n, k, 71)
    b_p, sf_p, alpha_p = prepare_bf16_fp4_weights(codes, sf, alpha, backend="cute-dsl")
    a = _rand(m, k, 72)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    mm_bf16_fp4(a, b_p, sf_p, alpha_p, out=out, backend="cute-dsl")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mm_bf16_fp4(a, b_p, sf_p, alpha_p, out=out, backend="cute-dsl")
    a.copy_(_rand(m, k, 73))
    graph.replay()
    torch.cuda.synchronize()
    _check(out, (a.float() @ deq.T) * 0.75, "graph replay")


def test_vllm_padded_weight():
    """vLLM's cute-dsl linear: N and K padded to 64, scales swizzled as float8 [pad128 N,
    pad4 K/16], prepare, K-padded activations and the output sliced back to N."""
    n, k, m = 100, 208, 5
    np_, kp = 128, 256
    codes, sf, deq = _quantize(_rand(n, k, 81))
    codes_p = torch.zeros((np_, kp // 2), dtype=torch.uint8, device="cuda")
    codes_p[:n, : k // 2] = codes
    sf_p = torch.zeros((np_, kp // 16), dtype=torch.uint8, device="cuda")
    sf_p[:n, : k // 16] = sf
    sf_swz = _swizzle_128x4(sf_p).view(torch.float8_e4m3fn).view(np_, kp // 16)
    alpha = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    b_q, sf_q, alpha_q = prepare_bf16_fp4_weights(
        codes_p, sf_swz, alpha, backend="cute-dsl"
    )
    assert alpha_q is not None and b_q.shape[0] == np_
    a = _rand(m, k, 82)
    a_p = torch.nn.functional.pad(a, (0, kp - k))
    out = _run_checked(
        lambda: mm_bf16_fp4(a_p, b_q, sf_q, alpha_q, backend="cute-dsl")[:, :n],
        (a.float() @ deq.T) * 0.5,
        "vllm padded",
    )
    assert out.shape == (m, n)
