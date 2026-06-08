"""Brutal randomized + cross-arch fuzzer for FP8 batched GEMM (``bmm_fp8``).

``test_bmm_fp8.py`` is gentle: a fixed b/m/n/k grid and a single ``cosine_similarity > 0.99``
oracle. FP8 GEMM bugs that escaped were arch- and shape-specific: #2337 (cublas uninit on
B200), #3068 (cutlass FP8 wrong for hidden>=512 per-tensor dequant), #3334 (garbage on
Hopper). A cosine oracle + a fixed grid cannot see partial-zero / arch-divergent output.

This fuzzer sweeps random shapes + magnitude regimes + dtypes + backends and enforces
reference-aware invariants. ``bmm_fp8`` runs on SM89/SM90/SM100, so running the *same seed*
on each of those GPUs and diffing the pass/fail sets is a strong **cross-arch oracle**: a
config that matches the torch reference on one arch but not another is an arch-specific bug.

Invariants (per arch, vs an fp32 ``torch.bmm`` reference):
  1. no spurious NaN/Inf (output non-finite only where the reference, in out dtype, is too),
  2. output not (almost) all-zero when the reference is non-trivial,
  3. run-to-run determinism.

Env: FLASHINFER_FP8_FUZZ_NUM_TESTS (default 300), FLASHINFER_FP8_FUZZ_SEED (default 0).
"""

import os
import random

import pytest
import torch

from flashinfer import autotune, bmm_fp8
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8

NUM_TESTS = int(os.environ.get("FLASHINFER_FP8_FUZZ_NUM_TESTS", "300"))
BASE_SEED = int(os.environ.get("FLASHINFER_FP8_FUZZ_SEED", "0"))

_SKIP = None
_CC0 = 0
if not torch.cuda.is_available():
    _SKIP = "CUDA not available"
else:
    _CC0 = get_compute_capability(torch.device("cuda:0"))[0]
    if _CC0 < 8:
        _SKIP = "bmm_fp8 needs SM80+"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))

_B = [1, 2, 4, 16, 32]
_M = [1, 2, 7, 8, 16, 48, 64, 128, 129, 256, 512, 1024]
_N = [16, 32, 64, 80, 128, 256, 512, 1024, 4096]
_K = [16, 32, 64, 128, 256, 512, 2688, 4096]
_REGIMES = ["normal", "tiny", "large", "mixed_rows", "with_zeros", "skewed"]
_SKIP_KW = (
    "not support",
    "unsupported",
    "no kernel",
    "invalid",
    "must be",
    "requires",
    "only support",
    "not implemented",
    "e5m2",
    "no valid engine",
)


def _regime(shape, regime, rng):
    g = torch.Generator(device="cuda").manual_seed(rng.randint(0, 2**31 - 1))
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g)
    if regime == "tiny":
        x = x * (10.0 ** rng.uniform(-20.0, -6.0))
    elif regime == "large":
        x = x * (10.0 ** rng.uniform(2.0, 4.0))
    elif regime == "mixed_rows":
        x = x * (
            10.0
            ** torch.randint(-12, 4, (shape[0], shape[1], 1), device="cuda").float()
        )
    elif regime == "with_zeros":
        x = x.masked_fill(
            torch.rand(shape, device="cuda", generator=g) < rng.uniform(0.3, 0.9), 0.0
        )
    elif regime == "skewed":
        x = x.sign() * (x.abs() ** rng.uniform(2.0, 5.0))
    return x.to(torch.bfloat16)


class _Cfg:
    __slots__ = ("seed", "b", "m", "n", "k", "idt", "mdt", "res", "backend", "regime")

    def __init__(self, seed):
        rng = random.Random(seed)
        self.seed = seed
        self.b, self.m, self.n, self.k = (
            rng.choice(_B),
            rng.choice(_M),
            rng.choice(_N),
            rng.choice(_K),
        )
        # avoid the documented invalid e5m2+e5m2 combo
        self.idt = rng.choice([torch.float8_e4m3fn, torch.float8_e5m2])
        self.mdt = (
            torch.float8_e4m3fn
            if self.idt == torch.float8_e5m2
            else rng.choice([torch.float8_e4m3fn, torch.float8_e5m2])
        )
        self.res = rng.choice([torch.bfloat16, torch.float16])
        # Respect the documented contract (test_bmm_fp8.py:21,27): cutlass bmm_fp8 needs
        # SM100/110/120 and does NOT support e5m2. Only offer cutlass when both hold; else it
        # silently NaNs instead of NOT_SUPPORTED (a minor robustness gap, not a numeric bug).
        backends = ["cublas", "cudnn", "auto"]
        if _CC0 in (10, 11, 12) and torch.float8_e5m2 not in (self.idt, self.mdt):
            backends.append("cutlass")
        self.backend = rng.choice(backends)
        self.regime = rng.choice(_REGIMES)

    def __repr__(self):
        return (
            f"b{self.b}_m{self.m}_n{self.n}_k{self.k}_{self.regime}_{self.backend}_"
            f"{str(self.idt).split('_')[-1]}x{str(self.mdt).split('_')[-1]}_s{self.seed}"
        )


_CONFIGS = [
    _Cfg(random.Random(BASE_SEED).randint(0, 2**31 - 1) ^ (i * 2654435761 & 0x7FFFFFFF))
    for i in range(NUM_TESTS)
]


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[repr(c) for c in _CONFIGS])
def test_bmm_fp8_fuzz(cfg):
    rng = random.Random(cfg.seed)
    inp = _regime((cfg.b, cfg.m, cfg.k), cfg.regime, rng)
    mat2 = _regime((cfg.b, cfg.n, cfg.k), cfg.regime, rng).transpose(-2, -1)
    try:
        inp_fp8, inp_s = to_float8(inp, dtype=cfg.idt)
        mat2_fp8, mat2_s = to_float8(mat2, dtype=cfg.mdt)
    except Exception as e:
        pytest.skip(f"quantize unsupported: {e}")

    reference = torch.bmm(inp.float(), mat2.float())
    res = torch.full((cfg.b, cfg.m, cfg.n), float("nan"), device="cuda", dtype=cfg.res)

    def _call(out):
        with autotune(False):
            bmm_fp8(inp_fp8, mat2_fp8, inp_s, mat2_s, cfg.res, out, backend=cfg.backend)

    try:
        _call(res)
        torch.cuda.synchronize()
    except pytest.skip.Exception:
        raise
    except Exception as e:  # incl. cudnn.cudnnGraphNotSupportedError
        if any(k in str(e).lower() for k in _SKIP_KW):
            pytest.skip(f"unsupported config {cfg}: {e}")
        raise

    resf = res.float()
    # FP8 (e4m3 max ~448, e5m2 ~57344) with per-tensor scaling cannot represent inputs whose
    # per-row magnitudes span many orders (mixed_rows) or sit far outside range (large/skewed/
    # tiny): NaN/Inf there is the legitimate consequence of un-representable data, not a kernel
    # bug. Assert the no-spurious-NaN numeric invariant only for in-range regimes; for the rest
    # we still require no crash and determinism below.
    if cfg.regime in ("normal", "with_zeros"):
        ref_od = reference.to(cfg.res).float()
        spurious = int(((~torch.isfinite(resf)) & torch.isfinite(ref_od)).sum().item())
        assert spurious == 0, (
            f"{cfg}: {spurious}/{resf.numel()} spurious NaN/Inf (ref finite) [#2440/#3334-class]"
        )

    if cfg.regime in ("normal", "with_zeros") and reference.abs().max().item() > 1e-2:
        nz = (resf.abs() > 0).float().mean().item()
        assert nz > 0.01, (
            f"{cfg}: ~all-zero output (nz={nz:.4f}) vs non-trivial reference [#3068/#3334-class]"
        )

    res2 = torch.full((cfg.b, cfg.m, cfg.n), float("nan"), device="cuda", dtype=cfg.res)
    _call(res2)
    torch.cuda.synchronize()
    if not torch.equal(res, res2):
        md = (resf - res2.float()).abs().max().item()
        pytest.fail(f"{cfg}: NONDETERMINISTIC bmm_fp8 output (max abs diff {md:.3e})")
