# NOTE for future contributors (incl. AI agents): keep this file a SMALL curated
# smoke set. Randomized breadth lives in tests/gemm/test_unified_gemm_fuzz.py (the
# mm_tf32 adapter); the CUDA-graph / PDL synchronization stress of the cluster
# split-K kernel in tests/gemm/test_mm_tf32_cute_dsl_splitk_sync.py.

import contextlib

import pytest
import torch

from flashinfer import autotune, mm_tf32
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability
from tests.utils.test_helpers.pdl import late_store


def _require_cute_dsl():
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("The TF32 cluster split-K kernel requires SM100/SM103/SM107.")
    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available.")


@contextlib.contextmanager
def _allow_tf32(enabled: bool):
    """Set torch.backends.cuda.matmul.allow_tf32 and restore it on exit."""
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


def _inputs(m: int, n: int, k: int, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((m, k), device="cuda", generator=generator)
    w = torch.randn((n, k), device="cuda", generator=generator) / k**0.5
    return a, w.T  # b: (k, n) column-major


def _check(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Within TF32 error of the FP64 product and of torch.matmul under TF32."""
    ref = a.double() @ b.double()
    scale = ref.abs().max().item()
    err = (out.double() - ref).abs().max().item() / scale
    assert err < 2e-3, f"max |out - fp64| / max |fp64| = {err:.2e}"
    with _allow_tf32(True):
        torch_tf32 = (a @ b).double()
    err = (out.double() - torch_tf32).abs().max().item() / scale
    assert err < 2e-3, f"max |out - torch TF32| / max |fp64| = {err:.2e}"


# Curated smoke set: one plain case per backend, PDL and autotune on each that has
# them, N tails and a short K.
_SMOKE_CASES = [
    # m, n, k, pdl, backend, auto_tuning
    (1, 1024, 256, False, "cute-dsl", False),
    (8, 256, 8192, True, "cute-dsl", False),
    (24, 200, 4096, False, "cute-dsl", True),
    (130, 384, 2048, True, "cute-dsl", False),
    (8, 250, 4096, True, "cute-dsl", False),  # N % 4 != 0: 8-byte-aligned output rows
    (64, 256, 8192, False, "cublas", False),
    (16, 256, 4096, True, "auto", True),
    (1024, 2048, 1024, False, "auto", False),
]


@pytest.mark.parametrize("m,n,k,pdl,backend,auto_tuning", _SMOKE_CASES)
def test_mm_tf32(m, n, k, pdl, backend, auto_tuning):
    if backend == "cute-dsl":
        _require_cute_dsl()
    AutoTuner.get().clear_cache()
    a, b = _inputs(m, n, k, seed=m + n)
    out = torch.full((m, n), float("nan"), device="cuda")
    # The cuBLAS backend follows the caller's TF32 setting.
    with _allow_tf32(True):
        with autotune(auto_tuning):
            mm_tf32(a, b, pdl=pdl, out=out, backend=backend)
        again = mm_tf32(a, b, pdl=pdl, backend=backend)
    torch.cuda.synchronize()
    _check(out, a, b)
    if backend == "cute-dsl" and not auto_tuning:
        assert torch.equal(out, again), "two runs differ"


# One explicit tactic per kernel variant (m, n, k, default_tactic overrides): a
# segment shorter than one output row, split 16 with deep K tiles, a partial last
# M tile, the 128-row tile (32dp32b TMEM loads), an N tail, and a single K tile
# per rank.
_TACTIC_CASES = [
    (8, 256, 8192, dict(split_k=16, cta_k=128, mma_n=8)),
    (64, 256, 8192, dict(split_k=16, cta_k=128, mma_n=32)),
    (200, 256, 8192, dict(split_k=4, cta_k=64, mma_n=64)),
    (512, 256, 8192, dict(split_k=4, cta_k=32, mma_m=128, mma_n=64)),
    (24, 200, 4096, dict(split_k=8, cta_k=32, mma_n=32)),
    (1, 1024, 256, dict(split_k=2, cta_k=128, mma_n=8)),
]


@pytest.mark.parametrize("m,n,k,overrides", _TACTIC_CASES)
def test_mm_tf32_cute_dsl_tactics(m, n, k, overrides):
    _require_cute_dsl()
    from flashinfer.gemm.kernels.dense_tf32_gemm_sm100_splitk import (
        default_tactic,
        run_tf32_splitk_dense,
    )

    tactic = default_tactic(m, n, k, **overrides)
    a, b = _inputs(m, n, k, seed=m + n)
    out = torch.full((m, n), float("nan"), device="cuda")
    run_tf32_splitk_dense(a, b, out, True, tactic)
    again = torch.full_like(out, float("nan"))
    run_tf32_splitk_dense(a, b, again, True, tactic)
    torch.cuda.synchronize()
    _check(out, a, b)
    assert torch.equal(out, again), f"{tactic}: two runs differ"


def test_mm_tf32_explicit_tactic():
    _require_cute_dsl()
    m, n, k = 256, 256, 8192
    a, b = _inputs(m, n, k)
    tactic = (64, 64, 2, 4, 64)
    out = mm_tf32(a, b, pdl=True, tactic=tactic)
    torch.cuda.synchronize()
    _check(out, a, b)
    assert torch.equal(out, mm_tf32(a, b, pdl=True, backend="auto", tactic=tactic))
    with pytest.raises(ValueError, match="cute-dsl backend only"):
        mm_tf32(a, b, backend="cublas", tactic=tactic)
    # Under autotune() the tactic still runs as given; nothing is profiled.
    with autotune(True):
        assert torch.equal(out, mm_tf32(a, b, pdl=True, tactic=tactic))
    for backend in ("auto", "cute-dsl"):
        with pytest.raises(ValueError, match="split_k"):
            mm_tf32(a, b, backend=backend, tactic=(64, 64, 3, 4, 64))
    with pytest.raises(TypeError):
        mm_tf32(a, b, True, None, "cute-dsl", tactic)  # keyword-only


def test_mm_tf32_cute_dsl_pdl_waits_for_a():
    _require_cute_dsl()
    m, n, k = 8, 256, 8192
    a, b = _inputs(m, n, k)
    expected = mm_tf32(a, b, pdl=True, backend="cute-dsl")

    # The first element of a is what the previous kernel is still producing.
    # No other kernel may run between that producer and the GEMM.
    word = a.view(-1)[:1]
    value = word.item()
    word.fill_(0.0)
    late_store(word, value)
    out = mm_tf32(a, b, pdl=True, backend="cute-dsl")

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_mm_tf32_autotune_tactics_are_valid():
    _require_cute_dsl()
    from flashinfer.gemm.kernels.dense_tf32_gemm_sm100_splitk import (
        autotune_tactics,
        validate_tactic,
    )

    for m in (1, 8, 32, 64, 128, 256, 512, 1024):
        for n, k in ((256, 8192), (2048, 4096), (64, 512)):
            tactics = autotune_tactics(m, n, k)
            assert tactics, f"no tactic for M={m}, N={n}, K={k}"
            for tactic in tactics:
                validate_tactic(tactic, m, n, k)


@pytest.mark.parametrize(
    "a_shape,n,dtype,backend,pdl,message",
    [
        ((8, 4096), 255, torch.float32, "cute-dsl", False, "multiple of 2"),
        (
            (8, 4000),
            256,
            torch.float32,
            "cute-dsl",
            False,
            "must be a positive multiple",
        ),
        ((8, 4096), 256, torch.bfloat16, "cute-dsl", False, "float32"),
        ((8, 4096), 256, torch.float32, "cublas", True, "no PDL"),
    ],
)
def test_mm_tf32_rejects(a_shape, n, dtype, backend, pdl, message):
    if backend != "cublas":
        _require_cute_dsl()
    a = torch.randn(a_shape, device="cuda").to(dtype)
    b = torch.randn((n, a_shape[1]), device="cuda").to(dtype).T
    with pytest.raises(ValueError, match=message):
        mm_tf32(a, b, pdl=pdl, backend=backend)


def test_mm_tf32_cublas_requires_allow_tf32():
    if get_compute_capability(torch.device("cuda")) < (8, 0):
        pytest.skip("The cuBLAS TF32 backend requires SM80+.")
    a, b = _inputs(8, 256, 4096)
    with _allow_tf32(False), pytest.raises(ValueError, match="allow_tf32"):
        mm_tf32(a, b, backend="cublas")


def test_mm_tf32_rejects_row_major_b():
    _require_cute_dsl()
    a = torch.randn((8, 4096), device="cuda")
    b = torch.randn((4096, 256), device="cuda")
    with pytest.raises(ValueError, match="column-major"):
        mm_tf32(a, b)
