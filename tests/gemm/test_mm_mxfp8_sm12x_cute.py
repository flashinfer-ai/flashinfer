"""SM120 / SM121 tests for ``mm_mxfp8(..., backend="cute-dsl")``.

Operands are drawn from a small set of E4M3 values with power-of-two block
scales, so every product and partial sum is exact in FP32 and the result is
independent of the summation order. Every kernel, tactic and split of K must
then reproduce the FP64 reference bit for bit after one rounding to the
output dtype.
"""

import pytest
import torch

from flashinfer import autotune, mm_mxfp8, mxfp8_quantize
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability, version_at_least


@pytest.fixture(scope="module", autouse=True)
def _require_sm12x_cuda13():
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM120 or SM121 GPU")
    cc = get_compute_capability(torch.device("cuda"))
    if cc not in ((12, 0), (12, 1)):
        pytest.skip("Requires an SM120 or SM121 GPU")
    if torch.version.cuda is None or not version_at_least(torch.version.cuda, "13.0"):
        pytest.skip("SM12x cute-dsl mm_mxfp8 requires CUDA 13 or newer")
    pytest.importorskip("cutlass.cute")
    assert mm_mxfp8.is_backend_supported("cute-dsl", cc[0] * 10 + cc[1])


def _runner_and_device():
    from flashinfer.gemm.kernels.sm12x_mxfp8 import runner

    return runner.get_runner(), runner._device(torch.cuda.current_device())


# E4M3 encodings of 0, +-0.5, +-1, +-1.5, +-2, +-3.
_E4M3 = torch.tensor([0x00, 0x30, 0xB0, 0x38, 0xB8, 0x3C, 0xBC, 0x40, 0xC0, 0x44, 0xC4])
# UE8M0 encodings of 2^-1, 2^0, 2^1.
_UE8M0 = torch.tensor([126, 127, 128])


def _swizzle(sf, rows, kb):
    """[rows, kb] UE8M0 -> flat F8_128x4 layout (rows padded to 128, kb to 4)."""
    kt = (kb + 3) // 4
    out = torch.full(((rows + 127) // 128 * 128 * kt * 4,), 127, dtype=torch.uint8)
    r = torch.arange(rows).view(-1, 1)
    c = torch.arange(kb).view(1, -1)
    idx = (
        (r >> 7) * (kt * 512)
        + (c >> 2) * 512
        + (r & 31) * 16
        + ((r >> 5) & 3) * 4
        + (c & 3)
    )
    out[idx.reshape(-1)] = sf.reshape(-1)
    return out


def _operands(m, n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    a = _E4M3[torch.randint(len(_E4M3), (m, k), generator=gen)].to(torch.uint8)
    w = _E4M3[torch.randint(len(_E4M3), (n, k), generator=gen)].to(torch.uint8)
    sfa = _UE8M0[torch.randint(3, (m, k // 32), generator=gen)].to(torch.uint8)
    sfw = _UE8M0[torch.randint(3, (n, k // 32), generator=gen)].to(torch.uint8)
    da = a.view(torch.float8_e4m3fn).double() * torch.pow(
        2.0, sfa.double() - 127
    ).repeat_interleave(32, dim=1)
    dw = w.view(torch.float8_e4m3fn).double() * torch.pow(
        2.0, sfw.double() - 127
    ).repeat_interleave(32, dim=1)
    operands = (
        a.view(torch.float8_e4m3fn).cuda(),
        w.view(torch.float8_e4m3fn).cuda().T,
        _swizzle(sfa, m, k // 32).cuda(),
        _swizzle(sfw, n, k // 32).cuda(),
    )
    return operands, (da @ dw.T).cuda()


def _assert_exact(out, ref):
    expected = ref.to(out.dtype)
    assert torch.isfinite(out).all()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def _bucket_cases():
    # (m, n, k): every M range of the policy, N and K tails, N < 128.
    return [
        (1, 1, 2080),
        (1, 256, 512),
        (2, 129, 384),
        (3, 7, 160),
        (3, 96, 2560),
        (4, 1024, 3200),
        (7, 640, 2560),
        (8, 15, 1056),
        (8, 2688, 1856),
        (13, 1280, 2944),
        (16, 33, 2592),
        (16, 2880, 1024),
        (17, 1856, 2688),
        (24, 320, 4096),
        (24, 640, 2080),
        (32, 3072, 1536),
        (48, 1280, 2560),
        (33, 1024, 1024),
        (100, 129, 384),
        (128, 2560, 640),
        (200, 96, 2560),
        (256, 1280, 2560),
        (300, 2688, 1856),
        (512, 1024, 3200),
        (600, 129, 384),
        (1031, 2560, 640),
        (2149, 1024, 1024),
    ]


@pytest.mark.parametrize("m,n,k", _bucket_cases())
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_default_tactic_exact(m, n, k, out_dtype):
    (a, b, sfa, sfb), ref = _operands(m, n, k, seed=m + n + k)
    out = mm_mxfp8(a, b, sfa, sfb, out_dtype=out_dtype, backend="cute-dsl")
    _assert_exact(out, ref)
    provided = torch.full((m, n), float("nan"), dtype=out_dtype, device="cuda")
    assert (
        mm_mxfp8(a, b, sfa, sfb, out=provided, out_dtype=out_dtype, backend="cute-dsl")
        is provided
    )
    _assert_exact(provided, ref)


_BF16, _FP16 = torch.bfloat16, torch.float16


@pytest.mark.parametrize(
    "m,n,k,out_dtype",
    [(m, n, k, _BF16) for m, n, k in [(1, 640, 2560), (4, 129, 384), (8, 129, 1056)]]
    + [(m, n, k, _BF16) for m, n, k in [(8, 2688, 1856), (16, 96, 2560)]]
    + [(m, n, k, _BF16) for m, n, k in [(32, 1856, 2688), (32, 640, 2080)]]
    + [(m, n, k, _BF16) for m, n, k in [(64, 1024, 3200), (64, 256, 160)]]
    + [(m, n, k, _BF16) for m, n, k in [(128, 2560, 640), (128, 129, 416)]]
    + [(m, n, k, _BF16) for m, n, k in [(256, 129, 384), (512, 3072, 1536)]]
    + [(1024, 1280, 2560, _BF16)]
    # FP16 output: GEMV and persistent, stream-K, ping-pong (direct and TMA store).
    + [(4, 129, 384, _FP16), (16, 96, 2560, _FP16), (256, 129, 384, _FP16)]
    + [(1024, 1280, 2560, _FP16)],
)
def test_every_tactic_exact(m, n, k, out_dtype):
    """Every advertised tactic of an autotuner bucket, at the bucket's M and at
    the smallest M of the bucket (runtime M tails). Calls the runner directly,
    so K < 128 (rejected by ``mm_mxfp8``) is covered too."""
    runner, dev = _runner_and_device()
    from flashinfer.gemm.kernels.sm12x_mxfp8 import policy

    tactics = policy.valid_tactics(m, n, k, dev)
    assert tactics
    for mm in sorted({m, m // 2 + 1}):
        (a, b, sfa, sfb), ref = _operands(mm, n, k, seed=mm * 7 + n)
        out = torch.empty((mm, n), dtype=out_dtype, device="cuda")
        inputs = [a, b, sfa, sfb, out_dtype, out, None]
        for tactic in tactics:
            if not policy.supports_m(tactic, mm):
                continue
            out.fill_(float("nan"))
            runner(inputs, tactic=tactic)
            _assert_exact(out, ref)


@pytest.mark.parametrize(
    "m,n,k", [(1900, 1280, 2720), (951, 1280, 2720), (300, 129, 160)]
)
def test_pingpong_tails_exact(m, n, k):
    """Every ping-pong candidate (including the cooperative 256-row tile, run
    without the runtime fallback) on M tails and a 16-byte-unaligned N."""
    _, dev = _runner_and_device()
    from flashinfer.gemm.kernels.sm12x_mxfp8 import policy, runner

    (a, b, sfa, sfb), ref = _operands(m, n, k, seed=m + k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    for tactic in policy._pingpong_candidates(m, n, k, dev):
        out.fill_(float("nan"))
        runner.launch(tactic, a, b, sfa, sfb, out)
        _assert_exact(out, ref)


@pytest.mark.parametrize("extra", [1, -1])
def test_streamk_sparse_split(extra):
    """Persistent stream-K with sms + 1 or 2 * sms - 1 tiles: the split region
    has 1 or sms - 1 tiles over all SMs, so many CTAs get no K iterations."""
    _, dev = _runner_and_device()
    from flashinfer.gemm.kernels.sm12x_mxfp8 import policy, runner

    tiles = dev.sms + 1 if extra == 1 else 2 * dev.sms - 1
    m, n, k = tiles * 128 - 5, 128, 256
    tactic = policy.persistent(128, 128, 2, 4, sched="streamk")
    grid, sk_tiles = policy.persistent_schedule(tactic, m, n, k, dev)
    assert (grid, sk_tiles) == (dev.sms, tiles - dev.sms)
    (a, b, sfa, sfb), ref = _operands(m, n, k, seed=tiles)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    runner.launch(tactic, a, b, sfa, sfb, out)
    _assert_exact(out, ref)


@pytest.mark.parametrize("m,n,k", [(2, 256, 512), (2, 4096, 2048), (24, 640, 2560)])
def test_nan_scale_propagates(m, n, k):
    """A UE8M0 scale of 0xFF (NaN) makes its output row NaN in every kernel."""
    runner, dev = _runner_and_device()
    from flashinfer.gemm.kernels.sm12x_mxfp8 import policy

    (a, b, sfa, sfb), ref = _operands(m, n, k, seed=4)
    sfa[0] = 255  # row 0, first k32 block
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    inputs = [a, b, sfa, sfb, torch.bfloat16, out, None]
    for tactic in policy.valid_tactics(m, n, k, dev):
        out.zero_()
        runner(inputs, tactic=tactic)
        assert torch.isnan(out[0]).all(), tactic
        _assert_exact(out[1:], ref[1:])


@pytest.mark.parametrize("m,n,k", [(1, 2560, 6144), (24, 640, 2560), (96, 2688, 1856)])
def test_streamk_deterministic(m, n, k):
    """Stream-K tactics reduce in a fixed order: repeated launches agree bitwise
    on random (inexact) data."""
    runner, dev = _runner_and_device()
    from flashinfer.gemm.kernels.sm12x_mxfp8 import policy

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a, sfa = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, sfb = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    inputs = [a, wq.T, sfa, sfb, torch.bfloat16, out, None]
    for tactic in policy.valid_tactics(m, n, k, dev):
        if "streamk" not in tactic:
            continue
        runner(inputs, tactic=tactic)
        first = out.clone()
        for _ in range(3):
            runner(inputs, tactic=tactic)
            assert torch.equal(out.view(torch.int16), first.view(torch.int16)), tactic


@pytest.mark.parametrize("m,n,k", [(1, 2560, 640), (5, 2688, 1856), (24, 1024, 3200)])
@pytest.mark.parametrize("m2", [0, 1])
def test_cuda_graph_replay(m, n, k, m2):
    """Capture after one eager call, change the inputs in place, replay."""
    m = m + m2 * 300
    (a, b, sfa, sfb), _ = _operands(m, n, k, seed=1)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    mm_mxfp8(a, b, sfa, sfb, out=out, backend="cute-dsl")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mm_mxfp8(a, b, sfa, sfb, out=out, backend="cute-dsl")
    for seed in (2, 3):
        (a2, b2, sfa2, sfb2), ref = _operands(m, n, k, seed=seed)
        for live, new in zip((a, b, sfa, sfb), (a2, b2, sfa2, sfb2), strict=True):
            live.copy_(new)
        out.fill_(float("nan"))
        graph.replay()
        _assert_exact(out, ref)


@pytest.mark.parametrize("n,k", [(1856, 2688), (129, 384)])
def test_autotune_then_bucket_tails(n, k):
    """Tune at the largest M, then run every M of the tuned buckets untuned."""
    AutoTuner.get().clear_cache()
    tune_m = 600
    (a, b, sfa, sfb), ref = _operands(tune_m, n, k, seed=11)
    with autotune(True):
        out = mm_mxfp8(a, b, sfa, sfb, backend="cute-dsl")
    _assert_exact(out, ref)
    for m in (1, 3, 5, 9, 17, 33, 65, 129, 257, 513, 600):
        (a, b, sfa, sfb), ref = _operands(m, n, k, seed=m)
        _assert_exact(mm_mxfp8(a, b, sfa, sfb, backend="cute-dsl"), ref)


def test_autotune_cache_file_roundtrip(tmp_path):
    """Tactics persisted by autotune(cache=...) are reloaded and replayed."""
    path = str(tmp_path / "mxfp8_sm12x.json")
    (a, b, sfa, sfb), ref = _operands(64, 1024, 1024, seed=5)
    AutoTuner.get().clear_cache()
    with autotune(True, cache=path):
        _assert_exact(mm_mxfp8(a, b, sfa, sfb, backend="cute-dsl"), ref)
    AutoTuner.get().clear_cache()
    with autotune(False, cache=path):
        _assert_exact(mm_mxfp8(a, b, sfa, sfb, backend="cute-dsl"), ref)


def test_foreign_tactic_falls_back_to_default():
    """A tactic that is not one of this backend's runs the default instead."""
    runner, _ = _runner_and_device()
    (a, b, sfa, sfb), ref = _operands(40, 256, 2048, seed=6)
    out = torch.empty((40, 256), dtype=torch.bfloat16, device="cuda")
    inputs = [a, b, sfa, sfb, torch.bfloat16, out, None]
    runner(inputs, tactic=((128, 8), (1, 1), True, False, 1))
    _assert_exact(out, ref)


def test_requirement_rejections():
    m, n, k = 16, 256, 512
    (a, b, sfa, sfb), _ = _operands(m, n, k, seed=0)
    with pytest.raises(ValueError):
        mm_mxfp8(a, b, sfa, sfb, out_dtype=torch.float32, backend="cute-dsl")
    from flashinfer.gemm.gemm_base import _cute_dsl_gemm_mxfp8_requirement

    with pytest.raises(ValueError, match="128x4"):
        _cute_dsl_gemm_mxfp8_requirement(
            a, b, sfa, sfb, use_8x4_sf_layout=True, backend="cute-dsl"
        )
    wide = torch.empty((m, k + 32), dtype=torch.float8_e4m3fn, device="cuda")
    wide[:, :k] = a
    with pytest.raises(ValueError, match="row-major"):
        mm_mxfp8(wide[:, :k], b, sfa, sfb, backend="cute-dsl")
    offset = torch.empty(m * k + 8, dtype=torch.uint8, device="cuda")
    unaligned = offset[8:].view(torch.float8_e4m3fn).view(m, k)
    unaligned.copy_(a)
    with pytest.raises(ValueError, match="16-byte aligned"):
        mm_mxfp8(unaligned, b, sfa, sfb, backend="cute-dsl")
    # Automatic selection skips the backend instead of raising.
    assert (
        _cute_dsl_gemm_mxfp8_requirement(
            a,
            b,
            sfa,
            sfb,
            out_dtype=torch.float32,
            use_8x4_sf_layout=False,
            backend="auto",
        )
        is False
    )


def test_k_not_multiple_of_32_rejected():
    from flashinfer.gemm.kernels.sm12x_mxfp8 import policy

    with pytest.raises(ValueError, match="K % 32"):
        policy.check_shape(16, 256, 144)
