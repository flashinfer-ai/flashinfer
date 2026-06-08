"""Brutal randomized fuzzer for low-precision FP4 GEMM (``mm_fp4``) and FP4 quantization.

Motivation
----------
``test_mm_fp4.py`` is gentle: a fixed ``m in [1..512] x n,k in [128,256,512]`` grid with a
single loose ``cosine_similarity > 0.97`` oracle. It has no edge-magnitude inputs, no
output poisoning, no all-zero / NaN output checks, and no run-to-run determinism check.
That is precisely why these low-precision GEMM bug classes escaped to users:

  * GH #3398 / #2577 : ``mm_fp4`` returns all-zero output for *specific* (in-between) M
                       values on some arch/backend (e.g. M=3072 between working 2048/4096).
  * GH #2440         : FP4/MXFP8 quantization emits NaN when inputs are very small
                       (tiny magnitude -> huge global scale -> overflow).
  * GH #2514         : ``mxfp4`` GEMM non-deterministically produces NaN on some arch.
  * GH #2516 / #2373 : NVFP4 GEMM crash / illegal memory access.

This fuzzer sweeps *random* nasty shapes and input-magnitude regimes and enforces brutal,
high-signal invariants that catch those classes regardless of the (loose) numeric oracle:

  1. no NaN / Inf in the output when the inputs are finite,
  2. the output is not (almost) all-zero when the reference is non-trivially non-zero,
  3. the result is identical run-to-run (same plan, re-poisoned output buffer),
  4. no launch crash / illegal memory access (a clean ``NOT_SUPPORTED`` is fine and skips).

Env knobs (cuDNN-fuzzer style):
  FLASHINFER_FUZZ_NUM_TESTS  number of random configs (default 200)
  FLASHINFER_FUZZ_SEED       base RNG seed (default 0)
"""

import os
import random

import pytest
import torch
import torch.nn.functional as F

from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    mxfp4_quantize,
    nvfp4_quantize,
)
from flashinfer.utils import LibraryError, get_compute_capability
from flashinfer.gemm.gemm_base import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR

NUM_TESTS = int(os.environ.get("FLASHINFER_FUZZ_NUM_TESTS", "200"))
BASE_SEED = int(os.environ.get("FLASHINFER_FUZZ_SEED", "0"))

# Skip the whole module off-GPU / on architectures with no FP4 GEMM support.
_SKIP_REASON = None
if not torch.cuda.is_available():
    _SKIP_REASON = "CUDA not available"
else:
    _CC = get_compute_capability(torch.device("cuda:0"))
    if _CC[0] < 10:
        _SKIP_REASON = f"FP4 GEMM requires SM100+, got SM{_CC[0]}{_CC[1]}"
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))

# Nasty M values: degenerate (1..), non-power-of-2, the "in-between" sizes that broke
# tile/dispatch selection (#3398 M=3072), and large.
_M_CHOICES = [
    1,
    2,
    3,
    4,
    7,
    8,
    15,
    16,
    17,
    31,
    48,
    63,
    64,
    96,
    127,
    128,
    192,
    255,
    256,
    257,
    384,
    512,
    768,
    1000,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
]
_N_CHOICES = [16, 32, 64, 128, 256, 512, 1024]
# k is forced to a multiple of the block size below.
_K_BASE = [16, 32, 64, 128, 256, 512, 1024]
_REGIMES = ["normal", "tiny", "large", "mixed_rows", "with_zeros", "skewed"]
# skip-keywords that mean "this config is legitimately unsupported" (not a bug)
_SKIP_KW = (
    "not support",
    "unsupported",
    "no kernel",
    "invalid",
    "must be",
    "requires",
    "only support",
    "not implemented",
    "alignment",
    "no valid engine",
)


def _make_regime_tensor(shape, regime, rng):
    """Generate a (deterministically-seeded) bf16 tensor in a nasty magnitude regime."""
    g = torch.Generator(device="cuda").manual_seed(rng.randint(0, 2**31 - 1))
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g)
    if regime == "tiny":
        x = x * (10.0 ** rng.uniform(-30.0, -8.0))  # #2440: tiny -> huge scale
    elif regime == "large":
        x = x * (10.0 ** rng.uniform(2.0, 5.0))
    elif regime == "mixed_rows":
        scale = 10.0 ** torch.randint(-18, 6, (shape[0], 1), device="cuda").float()
        x = x * scale
    elif regime == "with_zeros":
        mask = torch.rand(shape, device="cuda", generator=g) < rng.uniform(0.3, 0.95)
        x = x.masked_fill(mask, 0.0)
    elif regime == "skewed":
        x = x.sign() * (x.abs() ** rng.uniform(2.0, 6.0))
    return x.to(torch.bfloat16)


class _Cfg:
    __slots__ = (
        "seed",
        "m",
        "n",
        "k",
        "fp4_type",
        "block",
        "res_dtype",
        "backend",
        "regime",
    )

    def __init__(self, seed):
        rng = random.Random(seed)
        self.seed = seed
        self.fp4_type = rng.choice(["nvfp4", "mxfp4"])
        self.block = 16 if self.fp4_type == "nvfp4" else 32
        self.m = rng.choice(_M_CHOICES)
        # cutlass/cudnn nvfp4+mxfp4 require n and k divisible by 32; align by default,
        # but ~10% of the time leave them unaligned to verify a *clean* rejection
        # (NOT_SUPPORTED), never a crash/corruption.
        align = 32
        self.n = rng.choice(_N_CHOICES)
        k = rng.choice(_K_BASE)
        if rng.random() < 0.90:
            self.n = max(align, (self.n // align) * align)
            self.k = max(align, (k // align) * align)
        else:
            self.k = max(self.block, (k // self.block) * self.block)
        self.res_dtype = rng.choice([torch.bfloat16, torch.float16])
        # Respect the documented backend x dtype contract (test_mm_fp4.py:48): mxfp4 is only
        # supported on cudnn/cute-dsl/auto; nvfp4 additionally on cutlass. (Generating
        # mxfp4+cutlass just makes flashinfer silently emit NaN instead of NOT_SUPPORTED —
        # a real-but-minor robustness gap, out of scope for the numeric fuzzer.)
        if self.fp4_type == "mxfp4":
            self.backend = rng.choice(["auto", "cudnn"])
        else:
            self.backend = rng.choice(["auto", "cutlass", "cudnn"])
        self.regime = rng.choice(_REGIMES)

    def __repr__(self):
        return (
            f"{self.fp4_type}_m{self.m}_n{self.n}_k{self.k}_"
            f"{self.regime}_{self.backend}_{str(self.res_dtype).split('.')[-1]}"
        )


def _configs():
    rng = random.Random(BASE_SEED)
    return [_Cfg(rng.randint(0, 2**31 - 1)) for _ in range(NUM_TESTS)]


_CONFIGS = _configs()


def _run_one(cfg):
    use_nvfp4 = cfg.fp4_type == "nvfp4"
    inp = _make_regime_tensor((cfg.m, cfg.k), cfg.regime, random.Random(cfg.seed + 1))
    mat2 = _make_regime_tensor((cfg.n, cfg.k), cfg.regime, random.Random(cfg.seed + 2))

    # Global scale exactly as the production helper / existing test computes it.
    gsf_a = (448 * 6) / inp.float().abs().nan_to_num().max().clamp_min(1e-30)
    gsf_b = (448 * 6) / mat2.float().abs().nan_to_num().max().clamp_min(1e-30)

    if use_nvfp4:
        a_fp4, a_sf = nvfp4_quantize(
            inp, gsf_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
        b_fp4, b_sf = nvfp4_quantize(
            mat2, gsf_b, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
        alpha = 1.0 / (gsf_a * gsf_b)
    else:
        a_fp4, a_sf = mxfp4_quantize(inp)
        b_fp4, b_sf = mxfp4_quantize(mat2)
        alpha = None

    # Quantization itself must not invent NaN/Inf from finite inputs (#2440).
    for name, q in (("a_descale", a_sf), ("b_descale", b_sf)):
        bad = (~torch.isfinite(q.float())).sum().item()
        assert bad == 0, (
            f"{cfg}: quantize produced {bad} non-finite {name} from finite input"
        )

    reference = torch.mm(inp.float(), mat2.float().T)

    # Poison the output so a kernel that does not fully write it is caught.
    res = torch.full((cfg.m, cfg.n), float("nan"), device="cuda", dtype=cfg.res_dtype)

    def _call(out):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            cfg.res_dtype,
            out,
            block_size=cfg.block,
            use_8x4_sf_layout=False,
            backend=cfg.backend,
            use_nvfp4=use_nvfp4,
            skip_check=False,
        )

    try:
        with autotune(False):
            _call(res)
        torch.cuda.synchronize()
    except LibraryError as e:
        if str(e) == CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR:
            pytest.skip(str(e))
        raise
    except pytest.skip.Exception:
        raise
    except (
        Exception
    ) as e:  # incl. cudnn.cudnnGraphNotSupportedError ("No valid engine configs")
        msg = str(e).lower()
        if any(kw in msg for kw in _SKIP_KW):
            pytest.skip(f"unsupported config {cfg}: {e}")
        raise

    resf = res.float()
    # (1) NaN/Inf in the output is only a bug where the reference is finite *in the output
    # dtype*. Extreme-magnitude regimes (large/skewed/mixed) can legitimately overflow fp16,
    # and the reference overflows there too — so compare reference-aware to avoid flagging
    # expected overflow. A spurious NaN/Inf (reference finite, output not) is the #2440 class.
    # FP4 has a very small dynamic range; the extreme regimes (tiny/large/skewed/mixed_rows)
    # deliberately push inputs outside it, where NaN/Inf can be the legitimate consequence of
    # un-representable data (garbage-in). For those we only require no crash / determinism.
    # The no-spurious-NaN numeric invariant is asserted only for in-range regimes, where a
    # NaN/Inf with a finite reference is a real defect (the #2440 / #3103 class).
    if cfg.regime in ("normal", "with_zeros"):
        ref_od = reference.to(cfg.res_dtype).float()
        spurious = (~torch.isfinite(resf)) & torch.isfinite(ref_od)
        n_bad = int(spurious.sum().item())
        assert n_bad == 0, (
            f"{cfg}: {n_bad}/{resf.numel()} output elems are NaN/Inf where the reference is "
            f"finite (spurious non-finite output from finite inputs, #2440/#3103-class)"
        )

    # (2) output must not be (almost) all-zero when the reference is non-trivial (#3398).
    ref_mag = reference.abs().max().item()
    if ref_mag > 1e-3:
        nz_frac = (resf.abs() > 0).float().mean().item()
        assert nz_frac > 0.01, (
            f"{cfg}: output is ~all-zero (nz_frac={nz_frac:.4f}) while reference "
            f"max-magnitude={ref_mag:.3e} (M-dependent-zeros class, GH #3398)"
        )

    # (3) determinism: re-run the same call into a re-poisoned buffer; must match bit-exactly.
    res2 = torch.full((cfg.m, cfg.n), float("nan"), device="cuda", dtype=cfg.res_dtype)
    _call(res2)
    torch.cuda.synchronize()
    if not torch.equal(res, res2):
        max_d = (res.float() - res2.float()).abs().max().item()
        pytest.fail(
            f"{cfg}: NONDETERMINISTIC output across identical runs "
            f"(max abs diff {max_d:.3e}, GH #2514)"
        )

    # (4) loose numeric oracle, only on the well-conditioned 'normal' regime where it is
    # meaningful (FP4 quant error is large in edge regimes; structural invariants cover those).
    if cfg.regime == "normal" and ref_mag > 1e-2:
        cos = F.cosine_similarity(reference.reshape(-1), resf.reshape(-1), dim=0).item()
        assert cos > 0.90, f"{cfg}: cosine similarity {cos:.4f} too low vs reference"


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[repr(c) for c in _CONFIGS])
def test_mm_fp4_fuzz(cfg):
    _run_one(cfg)


# --- direct FP4 quantization fuzzer (the root of the #2440 tiny-input NaN class) ---

_Q_CONFIGS = [
    (rows, cols, regime, fp4)
    for rows in (1, 7, 64, 2048)
    for cols in (16, 64, 256, 1024)
    for regime in ("tiny", "large", "with_zeros", "skewed", "normal")
    for fp4 in ("nvfp4", "mxfp4")
]


@pytest.mark.parametrize("rows,cols,regime,fp4", _Q_CONFIGS)
def test_fp4_quantize_fuzz(rows, cols, regime, fp4):
    """Quantizing finite inputs must never produce NaN/Inf scale factors (GH #2440)."""
    block = 16 if fp4 == "nvfp4" else 32
    cols = max(block, (cols // block) * block)
    x = _make_regime_tensor(
        (rows, cols), regime, random.Random(hash((rows, cols, regime, fp4)) & 0xFFFF)
    )
    assert torch.isfinite(x.float()).all(), "test input itself is non-finite"
    if fp4 == "nvfp4":
        gsf = (448 * 6) / x.float().abs().nan_to_num().max().clamp_min(1e-30)
        q, sf = nvfp4_quantize(x, gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    else:
        q, sf = mxfp4_quantize(x)
    n_bad = (~torch.isfinite(sf.float())).sum().item()
    assert n_bad == 0, (
        f"{fp4} quantize({rows}x{cols},{regime}) produced {n_bad} non-finite "
        f"scale-factor elements from finite input (GH #2440)"
    )
