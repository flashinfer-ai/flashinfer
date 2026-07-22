"""Unified fuzzer + convention auditor for flashinfer's scaled GEMM/BMM family.

ONE harness for the whole {op-shape} x {quant-mode} x {backend} cross-product instead of N
per-API files, built on the same best-practice oracle kit as the unified MoE fuzzer
(``tests/moe/test_unified_moe_fuzz.py``):
  * SPARSE (~75% zero) + EXACTLY-REPRESENTABLE inputs (snapped to each quant mode's grid), so input
    quantization is lossless and the gemm reductions are short -- a structural bug (wrong tile /
    dropped block / wrong scale) is a GROSS error, not one averaged away. This is the MoE harness's
    model (it dropped the old magnitude-regime axis for this), enabling a TIGHT numeric oracle
    instead of a loose cosine. Nasty shapes too (non-pow2, the "in-between" M that broke tile
    selection #3398, degenerate 1/2/3, occasional unaligned -> clean reject).
  * numeric vs an AUTHORITATIVE reference at a tight per-quant-mode tolerance atol=C*||ref||_inf
    (input quant is lossless, so C is the accumulation/requant floor, NOT the dense-random fp4/fp8
    quant error); + reference-aware no-spurious-NaN/Inf,
  * output-buffer POISON (NaN-fill so a kernel that doesn't fully write its output is caught),
  * run-to-run determinism (#2514),
  * device-state probe after each config (a context-corrupting IMA -> clean failure),
  * cross-arch oracle by construction: bmm_fp8 runs SM89/90/100 -> run the SAME seed on each GPU
    and diff the pass/fail sets (a config that matches ref on one arch but not another = arch bug).

WHY ALSO A CONVENTION AUDITOR
-----------------------------
Each scaled-GEMM API ships its OWN scale convention (per-tensor alpha vs block scale vs block+global
alpha vs none; A/B-scale roles; layout). That heterogeneity is the "convention mismatch" surface
where the fp4-vs-fp8 incompatibility was found. The existing APIs CANNOT be changed (would break
users), so this harness does NOT force them to agree. Instead each adapter **declares its convention
explicitly** (``CONVENTION`` dict), and:
  * the per-config oracle validates each backend against its OWN authoritative reference (a backend
    that deviates from its declared recipe is caught) -- NOT against a different convention (forcing
    cross-convention numeric equality is a false-positive trap, the lesson from the MoE
    intermediate-scale-policy finding);
  * ``test_convention_conformance`` cross-checks backends that DECLARE the same convention (e.g.
    cutlass vs cudnn nvfp4) -- those must agree numerically (a real cross-backend oracle) -- and
    prints a convention matrix, with a ``_CONVENTION_DIVERGENCES`` ledger documenting known
    cross-mode incompatibilities so they are tracked, not silently passing.

This is the enforcement hook for the future: if a unified GEMM API (or an incremental
convention-compat fix) makes two previously-divergent APIs share a convention, move them into the
same conformance group and the test will *enforce* they now agree -- and the ledger entry's removal
becomes the proof the convention was unified.

STATUS: framework + 7 grounded adapters (bf16 mm+bmm / fp8 bmm incl e4m3+e5m2 / nvfp4 mm / mxfp4 mm /
mxfp8 mm+bmm) + a standalone quantize-root fuzzer. This file REPLACES the earlier per-op
``test_mm_fp4_fuzz.py`` / ``test_bmm_fp8_fuzz.py`` (their logic + the #2440 quantize test are folded
in here). Validated on A100/SM80, L40S/SM89, H100/SM90, B200/SM100. The mxfp8 adapters surfaced two
real cuDNN findings (a ``_KNOWN_FAILURES`` ledger entry + the ``min_dim`` note record them). EXTEND
by folding in more adapters one ``GemmAdapter`` at a time: mm_fp8 (low-latency decode-shape regime,
needs ``prepare_low_latency_gemm_weights``) and the grouped ``group_*`` / ``*deepgemm*`` entries
(m_indptr op-shape).

Run (needs CUDA_HOME for JIT; SM100 for the fp4 adapters, SM89+ for fp8):
  CUDA_HOME=<cuda> CUDA_VISIBLE_DEVICES=<idx> pytest -q tests/gemm/test_unified_gemm_fuzz.py
Env: FLASHINFER_GEMM_FUZZ_NUM_TESTS (default 2000; autotune-OFF breadth ~0.4s/cfg + ~5% autotune-ON
     at ~5-8s/cfg), FLASHINFER_GEMM_FUZZ_SEED (default 0),
     FLASHINFER_GEMM_FUZZ_ONLY_SEED (comma-separated seeds -> run ONLY those configs; the
     perfect-repro hook printed on every failure).

Determinism / repro: every config is fully derived from its seed (shapes, modes, input data,
buffer poison, and the global RNG via torch.manual_seed(seed)), so a failing test reproduces
bit-for-bit from the REPRO command it prints. Each test prints its full config + repro command;
on a numeric mismatch it dumps output-vs-oracle stats + the worst <=100 elements (so the CI log
alone shows whether the output is all-zero / all-NaN / Inf, without rerunning).
"""

from __future__ import annotations

import contextlib
import os
import random
import zlib
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from flashinfer import (
    SfLayout,
    autotune,
    bmm_bf16,
    bmm_fp8,
    bmm_mxfp8,
    mm_bf16,
    mm_fp4,
    mm_mxfp8,
    mxfp4_quantize,
    mxfp8_dequantize_host,
    mxfp8_quantize,
    nvfp4_quantize,
)
from flashinfer.autotuner import AutoTuner
from flashinfer.quantization import e2m1_and_ufp8sf_scale_to_float, mxfp4_dequantize
from flashinfer.utils import LibraryError, get_compute_capability

try:
    from flashinfer.gemm.gemm_base import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR
except Exception:  # pragma: no cover - constant may move between versions
    CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR = "\0never-matches\0"
from tests.utils_fp8 import to_float8

# Default sized so a full sweep is a substantial fuzz (~10 min on a warm cache on one GPU). Tune
# down via the env var for a quick smoke; the per-config seed makes any failure reproducible.
NUM_TESTS = int(os.environ.get("FLASHINFER_GEMM_FUZZ_NUM_TESTS", "2000"))
BASE_SEED = int(os.environ.get("FLASHINFER_GEMM_FUZZ_SEED", "0"))
# Perfect-repro hook: if set (comma-separated seeds), the suite runs ONLY those configs --
# `_gen(seed)` is fully deterministic, so a single seed reproduces one config exactly. The
# repro command printed on every failure uses this.
_ONLY_SEEDS = os.environ.get("FLASHINFER_GEMM_FUZZ_ONLY_SEED", "")

_SKIP = None
_CC = (0, 0)
if not torch.cuda.is_available():
    _SKIP = "CUDA not available"
else:
    _CC = get_compute_capability(torch.device("cuda:0"))
    if _CC[0] < 8:
        _SKIP = f"scaled GEMM needs SM80+, got SM{_CC[0]}{_CC[1]}"
pytestmark = [
    # ~15-30 min/leg in CI (full sweep) -> front-load in the parallel queue.
    pytest.mark.long_running,
    pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP)),
]
_SM = _CC[0] * 10 + _CC[1]

# A clean rejection ("not supported on this shape/arch/dtype") must SKIP, never fail; a crash
# (CUDA error / illegal access / device assert) is always a finding and re-raises.
_SKIP_KW = (
    "not support",
    "unsupported",
    "no kernel",
    "no valid engine",
    "no suitable",  # BackendSupportedError: "No suitable auto backends found for ..."
    "invalid",
    "must be",
    "requires",
    "only support",
    "not implemented",
    "alignment",
    "e5m2",
)
_CRASH_KW = ("cuda error", "illegal memory", "misaligned", "device-side assert")

# Magnitude regimes are used ONLY by the standalone quantize-root fuzzer (test_gemm_quantize_fuzz),
# where extreme magnitudes are the point (#2440). The main fuzz uses sparse + exact-grid (below).
_REGIMES = ["normal", "tiny", "large", "mixed_rows", "with_zeros", "skewed"]


def _is_unsupported(e: Exception) -> bool:
    msg = str(e).lower()
    if any(k in msg for k in _CRASH_KW):
        return False
    return any(k in msg for k in _SKIP_KW)


def _regime_tensor(shape, regime, rng) -> torch.Tensor:
    """A deterministically-seeded bf16 tensor in a nasty magnitude regime (carried over from the
    validated GEMM fuzzers)."""
    g = torch.Generator(device="cuda").manual_seed(rng.randint(0, 2**31 - 1))
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g)
    if regime == "tiny":
        x = x * (
            10.0 ** rng.uniform(-30.0, -8.0)
        )  # #2440: tiny -> huge scale -> overflow
    elif regime == "large":
        x = x * (10.0 ** rng.uniform(2.0, 5.0))
    elif regime == "mixed_rows":
        s = 10.0 ** torch.randint(-18, 6, (*shape[:-1], 1), device="cuda").float()
        x = x * s
    elif regime == "with_zeros":
        m = torch.rand(shape, device="cuda", generator=g) < rng.uniform(0.3, 0.9)
        x = x.masked_fill(m, 0.0)
    elif regime == "skewed":
        x = x.sign() * (x.abs() ** rng.uniform(2.0, 6.0))
    return x.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Adapter protocol. Canonical NT convention everywhere: a = [(B,) M, K], b = [(B,) N, K];
# reference = a @ b^T. Each adapter quantizes/lays-out per its API's own convention and calls it,
# DECLARING that convention so the conformance auditor can group compatible backends.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GemmAdapter:
    key: str
    op_shape: str  # "mm" | "bmm"
    quant_mode: str  # "bf16" | "fp8" | "mxfp8" | "nvfp4" | "mxfp4"
    backends: tuple  # selectable `backend=` values
    convention: dict  # DECLARED scale convention (the audit substrate)
    supported: Callable[[int], bool]  # (sm) -> bool
    run: Callable  # (a_bf16, b_bf16, out, backend, cfg) -> None  (writes into `out`)
    deterministic: bool = True
    align: int = 1  # required N/K alignment (snap shapes to this)
    min_dim: int = (
        1  # minimum N/K (some kernels require a min tile, e.g. mxfp8 needs n,k >= 128)
    )


# --- bf16 mm: no quant convention -> the cross-backend EQUALITY oracle is valid here ----------
def _run_bf16_mm(a, b, out, backend, cfg):
    # mm_bf16 wants b as [K, N] column-major == our [N, K]^T.
    o = mm_bf16(a, b.transpose(0, 1), out=out, out_dtype=out.dtype, backend=backend)
    if out.data_ptr() != o.data_ptr():
        out.copy_(o)


# --- fp8 bmm: per-tensor A/B scales passed as separate args, no output scale -------------------
def _run_fp8_bmm(a, b, out, backend, cfg):
    idt = cfg.fp8_idt or torch.float8_e4m3fn
    mdt = cfg.fp8_mdt or torch.float8_e4m3fn
    a_fp8, a_s = to_float8(a, dtype=idt)
    # B must be the column-major [B,K,N] VIEW (stride [k*n,1,k]) -- the layout fp8 bmm requires.
    # Do NOT .contiguous() it: that makes B row-major, which cudnn/cutlass reject and cublas
    # SILENTLY computes garbage from (cosine~0). Matches the original test_bmm_fp8_fuzz.py.
    bt = b.transpose(-2, -1)
    b_fp8, b_s = to_float8(bt, dtype=mdt)
    bmm_fp8(a_fp8, b_fp8, a_s, b_s, out.dtype, out, backend=backend)


# --- nvfp4 mm: block(16) scales + a GLOBAL dequant scalar alpha = 1/(gsf_a*gsf_b) --------------
def _run_nvfp4_mm(a, b, out, backend, cfg):
    gsf_a = (448 * 6) / a.float().abs().nan_to_num().max().clamp_min(1e-30)
    gsf_b = (448 * 6) / b.float().abs().nan_to_num().max().clamp_min(1e-30)
    a_q, a_sf = nvfp4_quantize(
        a, gsf_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_q, b_sf = nvfp4_quantize(
        b, gsf_b, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = 1.0 / (gsf_a * gsf_b)
    mm_fp4(
        a_q,
        b_q.T,
        a_sf,
        b_sf.T,
        alpha,
        out.dtype,
        out,
        block_size=16,
        use_8x4_sf_layout=cfg.use_8x4,
        backend=backend,
        use_nvfp4=True,
        skip_check=False,
    )


# --- mxfp4 mm: block(32) scales only, NO global alpha (the convention that diverges from nvfp4) -
def _run_mxfp4_mm(a, b, out, backend, cfg):
    a_q, a_sf = mxfp4_quantize(a)
    b_q, b_sf = mxfp4_quantize(b)
    mm_fp4(
        a_q,
        b_q.T,
        a_sf,
        b_sf.T,
        None,
        out.dtype,
        out,
        block_size=32,
        use_8x4_sf_layout=cfg.use_8x4,
        backend=backend,
        use_nvfp4=False,
        skip_check=False,
    )


def _run_bf16_bmm(a, b, out, backend, cfg):
    # bmm_bf16: like bmm_fp8, mat2 must be column-major [B,K,N] -> pass the transposed VIEW of our
    # [B,N,K] (NO .contiguous(); a contiguous copy is row-major and the kernel reads it transposed).
    bmm_bf16(a, b.transpose(-2, -1), out=out, out_dtype=out.dtype, backend=backend)


def _run_mm_mxfp8(a, b, out, backend, cfg):
    # mm_mxfp8: per-block(32) mxfp8 with 128x4-swizzled e8m0 scales; a=[M,K], b=[N,K], pass mat2 as
    # b_q.T. reference = a @ b^T. (swizzle changes only scale STORAGE, not values -> matches snap.)
    a_q, a_s = mxfp8_quantize(a, sf_swizzle_layout=SfLayout.layout_128x4)
    b_q, b_s = mxfp8_quantize(b, sf_swizzle_layout=SfLayout.layout_128x4)
    res = mm_mxfp8(a_q, b_q.T, a_s, b_s, out=out, out_dtype=out.dtype, backend=backend)
    if res.data_ptr() != out.data_ptr():
        out.copy_(res)


def _run_mxfp8_bmm(a, b, out, backend, cfg):
    # bmm_mxfp8: per-block(32) mxfp8. A=[B,M,K] row-major; the weight is quantized as the contiguous
    # [B,N,K] (blocked along K, like every other adapter) and passed as B = weight_q.transpose(-2,-1)
    # -> a COLUMN-major [B,K,N] VIEW (the kernel rejects a row-major/contiguous B). reference = a@b^T.
    # cudnn here drives the override-shape (dynamic-M) path -- the #3455 regression surface.
    a_q, a_s = mxfp8_quantize(
        a, True
    )  # is_sf_swizzled_layout=True (values identical to linear)
    b_q, b_s = mxfp8_quantize(
        b, True
    )  # b is the [B,N,K] weight; do NOT .contiguous() the transpose
    bmm_mxfp8(a_q, b_q.transpose(-2, -1), a_s, b_s, out.dtype, out, backend=backend)


_ADAPTERS = [
    GemmAdapter(
        key="mm_bf16",
        op_shape="mm",
        quant_mode="bf16",
        backends=("cudnn", "cutlass", "cublaslt", "auto"),
        convention={"a_scale": "none", "b_scale": "none", "global": "none"},
        supported=lambda sm: sm >= 80,
        run=_run_bf16_mm,
        deterministic=True,
        align=1,
    ),
    GemmAdapter(
        key="bmm_fp8",
        op_shape="bmm",
        quant_mode="fp8",
        backends=("cublas", "cudnn", "auto")
        + (("cutlass",) if _CC[0] in (10, 11, 12) else ()),
        convention={
            "a_scale": "per-tensor (arg)",
            "b_scale": "per-tensor (arg)",
            "global": "none",
        },
        supported=lambda sm: sm >= 89,
        run=_run_fp8_bmm,
        deterministic=True,
        align=1,
    ),
    GemmAdapter(
        key="mm_nvfp4",
        op_shape="mm",
        quant_mode="nvfp4",
        backends=("cutlass", "cudnn", "auto"),
        convention={
            "a_scale": "block-16",
            "b_scale": "block-16",
            "global": "alpha = 1/(gsf_a*gsf_b)",
            "layout": "128x4",
        },
        supported=lambda sm: sm >= 100,
        run=_run_nvfp4_mm,
        deterministic=True,
        align=32,
    ),
    GemmAdapter(
        key="mm_mxfp4",
        op_shape="mm",
        quant_mode="mxfp4",
        backends=("cudnn", "auto"),
        convention={
            "a_scale": "block-32",
            "b_scale": "block-32",
            "global": "none",
            "layout": "mx",
        },
        supported=lambda sm: sm >= 100,
        run=_run_mxfp4_mm,
        deterministic=True,
        align=32,
    ),
    GemmAdapter(
        key="bmm_bf16",
        op_shape="bmm",
        quant_mode="bf16",
        backends=("cutlass", "cudnn", "auto"),
        convention={"a_scale": "none", "b_scale": "none", "global": "none"},
        supported=lambda sm: sm >= 80,
        run=_run_bf16_bmm,
        deterministic=True,
        align=8,
    ),
    GemmAdapter(
        key="mm_mxfp8",
        op_shape="mm",
        quant_mode="mxfp8",
        backends=("cudnn", "cutlass", "auto"),
        convention={
            "a_scale": "block-32",
            "b_scale": "block-32",
            "global": "none",
            "layout": "128x4 (swizzled e8m0)",
        },
        supported=lambda sm: sm >= 100,
        run=_run_mm_mxfp8,
        deterministic=True,
        align=32,
        min_dim=128,  # mxfp8 GEMM requires n,k >= 128 (cutlass rejects below; see _gen B2 note)
    ),
    GemmAdapter(
        key="bmm_mxfp8",
        op_shape="bmm",
        quant_mode="mxfp8",
        # cudnn -> SM10x (override-shape / dynamic-M path), cutlass -> SM12x; chosen from live GPU.
        backends=("cudnn",) if _CC[0] == 10 else (("cutlass",) if _CC[0] == 12 else ()),
        convention={
            "a_scale": "block-32",
            "b_scale": "block-32",
            "global": "none",
            "layout": "mx",
            # NB: B must be a COLUMN-major [b,k,n] VIEW (weight_q.transpose, no .contiguous());
            # the cudnn override-shape path (PR #3455) rejects a row-major/contiguous B.
            "b_memory": "column-major (transpose view)",
        },
        supported=lambda sm: (sm // 10) in (10, 12),
        run=_run_mxfp8_bmm,
        deterministic=True,
        align=32,
        min_dim=128,  # mxfp8 needs n,k >= 128; bmm_mxfp8 does NOT enforce it (silent garbage) -- see
        # _gen B2 note: the fuzzer stays in the supported regime, the missing guard is reported.
    ),
    # EXTEND:
    # * mm_fp8 (trtllm_low_latency): decode-only shape regime (M<=16, K>=8192, N in {2560,5120} +
    #   prepare_low_latency_gemm_weights) -> needs its OWN shape distribution, not the uniform _gen.
    # * group_gemm_*_nt_groupwise / *deepgemm*: grouped op-shape with m_indptr offsets.
    # Add one GemmAdapter each; unsupported (op_shape, arch) just skips.
]

# Known cross-mode convention incompatibilities (documented, not silently passing). The moment a
# unification makes a pair agree, move them into one conformance group and delete the entry.
_CONVENTION_DIVERGENCES = [
    (
        "mm_nvfp4",
        "mm_mxfp4",
        "nvfp4 carries a global dequant alpha=1/(gsf_a*gsf_b); mxfp4 has block scales only and NO "
        "global -> the two cannot be validated against one shared reference (fp4 vs fp4 mismatch).",
    ),
    (
        "bmm_fp8",
        "mm_nvfp4",
        "fp8 uses separate per-tensor A/B scale args; nvfp4 folds range into block+global alpha -> "
        "incompatible scale ABIs (the fp4-vs-fp8 convention mismatch).",
    ),
]

# Tracked findings the fuzzer surfaced (xfail-but-RUN: the case still executes, a correctness
# failure is tolerated to keep the suite green, and an unexpected PASS warns "fixed -> remove it").
# Each entry is (predicate, reason, crash_capable). crash_capable=True entries are xfailed UP FRONT
# (the kernel is never launched): the same root cause that yields garbage on one library stack can
# be an out-of-bounds access on another, and one IMA poisons the CUDA context for every later test
# in the session (seen live in CI, see the #3604 entry). crash_capable=False entries still RUN and
# xfail only if an invariant actually fails -> they keep the xpass "fixed -> remove me" signal.
_KNOWN_FAILURES = [
    (
        lambda cfg: cfg.adapter.key == "bmm_mxfp8" and cfg.b > 1 and (cfg.m % 128) != 0,
        "TRACKED: flashinfer-ai/flashinfer#3604. "
        "bmm_mxfp8 returns GARBAGE/inf for batch indices > 0 when b>1 AND M is not a multiple of 128 "
        "(batch 0 is always correct; b==1 is correct for any M; b>1 with M%128==0 is correct). This is "
        "BOTH-backend, not cuDNN-specific: verified on cuDNN/SM100 (9.23.0 AND 9.23.1) and CUTLASS/"
        "SM120 (RTX 5080) -- identical fingerprint, 30 xfail / 0 xpass. Shared root cause: the default "
        "mxfp8_quantize([b,m,k]) pads the FLATTENED b*m dim (round_up(b*m,128)), so the 128x4-swizzled "
        "scale-factor tile interleaves batch boundaries and is not per-batch-separable; both the cuDNN "
        "override-shape stride math and the CUTLASS kernel then index batch>0's scales wrong. The known "
        "#3455 'b>1 SF-padding' follow-up; real fix = per-batch / rank_preserving 3D SF (#3457), which "
        "is NOT in tree and is unfixed in 9.23.1. Keep this predicate arch/backend-agnostic. On the "
        "cu129 CI stack this root cause escalates from garbage to an ILLEGAL MEMORY ACCESS (GB200, "
        "b=16 m=7 n=512 k=2688). To probe fixed-ness, run the REPRO with this entry removed. "
        "See HANDOFF_SM120_MXFP8_BMM_VERIFY.md.",
        True,  # crash-capable: IMA'd on GB200/cu129
    ),
]


def _known_failure(cfg: Cfg, crash_only: bool = False) -> Optional[str]:
    for pred, reason, crash_capable in _KNOWN_FAILURES:
        if (crash_capable or not crash_only) and pred(cfg):
            return reason
    return None


_SHAPES_M = [
    1,
    2,
    3,
    7,
    8,
    16,
    17,
    48,
    63,
    64,
    127,
    128,
    129,
    256,
    257,
    512,
    1024,
    3072,
    4096,
]
_SHAPES_N = [16, 32, 64, 80, 128, 256, 512, 1024]
_SHAPES_K = [16, 32, 64, 128, 256, 512, 1024, 2688]
_BATCH = [1, 2, 4, 16, 32]


@dataclass
class Cfg:
    seed: int
    adapter: GemmAdapter
    backend: str
    b: int
    m: int
    n: int
    k: int
    out_dtype: torch.dtype
    fp8_idt: Optional[torch.dtype] = (
        None  # fp8 A-operand dtype (e4m3/e5m2); None -> e4m3
    )
    fp8_mdt: Optional[torch.dtype] = None  # fp8 B-operand dtype
    noncontig: bool = False  # feed non-contiguous input views
    use_8x4: bool = False  # mm_fp4 use_8x4_sf_layout (the #2861 padding-leak flag)

    @property
    def label(self):
        sh = f"b{self.b}_" if self.adapter.op_shape == "bmm" else ""
        dt = str(self.out_dtype).split(".")[-1]
        fp8 = (
            f"_{str(self.fp8_idt).split('_')[-1]}x{str(self.fp8_mdt).split('_')[-1]}"
            if self.fp8_idt is not None
            else ""
        )
        mode = ("nc" if self.noncontig else "") + ("8x4" if self.use_8x4 else "")
        mode = f"_{mode}" if mode else ""
        return (
            f"{self.adapter.key}_{self.backend}_{sh}m{self.m}_n{self.n}_k{self.k}_"
            f"{dt}{fp8}{mode}_s{self.seed}"
        )


def _gen(seed) -> Cfg:
    rng = random.Random(seed)
    wired = [a for a in _ADAPTERS if a.supported(_SM)]
    ad = rng.choice(wired)
    al = ad.align
    # K stays aligned (>= the per-mode floor) so fp4/mxfp8 input quant is FORMABLE -- an unaligned K
    # breaks the quantizer/snap itself, before the kernel, which is not a meaningful "rejects" test.
    lo = max(al, ad.min_dim)  # floor for N/K (mxfp8 kernels need n,k >= 128)
    k = max(lo, (rng.choice(_SHAPES_K) // al) * al)
    # B2: ~10% leave N unaligned to verify a clean kernel rejection (not a crash) -- only for modes
    # that DO reject (min_dim==1). mxfp8 (min_dim=128) is excluded: bmm_mxfp8 lacks the n>=128 guard
    # mm_mxfp8 has and silently returns garbage for 32<=n<128 (a real robustness gap, reported
    # separately), so an unaligned-N "clean reject" assertion would spuriously fail there.
    if ad.min_dim <= 1 and rng.random() < 0.10:
        n = rng.choice(_SHAPES_N)  # possibly unaligned
    else:
        n = max(lo, (rng.choice(_SHAPES_N) // al) * al)
    # B1 (DEFERRED): non-contiguous inputs. Needs the right oracle -- "non-contig result must MATCH
    # the contiguous result, else clean-reject" -- not a direct ref compare (some APIs legitimately
    # require contiguous and should raise). Re-enable with that consistency oracle. Kept False.
    noncontig = False
    # C1 (DEFERRED): mm_fp4 use_8x4_sf_layout=True is the #2861 padding-leak flag, BUT it is a
    # trtllm-backend feature and requires the scale factors in the *matching* 8x4 SF layout (not the
    # layout_128x4 / mxfp4 default this harness prepares). Toggling it on the cutlass/cudnn/auto
    # backends just yields "No suitable backends" rejections. Re-enable once a trtllm mm_fp4 adapter
    # + the matching SfLayout are wired (then the output-poison oracle catches #2861). Kept False.
    use_8x4 = False
    fp8_idt = fp8_mdt = None
    backends = ad.backends
    if ad.quant_mode == "fp8":
        fp8_idt = rng.choice([torch.float8_e4m3fn, torch.float8_e5m2])
        # documented invalid combo: e5m2 x e5m2 -> force the other operand to e4m3.
        fp8_mdt = (
            torch.float8_e4m3fn
            if fp8_idt == torch.float8_e5m2
            else rng.choice([torch.float8_e4m3fn, torch.float8_e5m2])
        )
        # cutlass bmm_fp8 does NOT support e5m2 (test_bmm_fp8.py contract) -> don't offer it then
        # (else it silently NaNs instead of NOT_SUPPORTED -- a minor robustness gap, out of scope).
        if torch.float8_e5m2 in (fp8_idt, fp8_mdt):
            backends = tuple(b for b in backends if b != "cutlass")
    return Cfg(
        seed=seed,
        adapter=ad,
        backend=rng.choice(backends),
        b=rng.choice(_BATCH) if ad.op_shape == "bmm" else 1,
        m=rng.choice(_SHAPES_M),
        n=n,
        k=k,
        out_dtype=rng.choice([torch.bfloat16, torch.float16]),
        fp8_idt=fp8_idt,
        fp8_mdt=fp8_mdt,
        noncontig=noncontig,
        use_8x4=use_8x4,
    )


if _SKIP is not None or not any(a.supported(_SM) for a in _ADAPTERS):
    _CONFIGS = []
elif _ONLY_SEEDS:  # perfect-repro: run only the named seed(s)
    _CONFIGS = [_gen(int(s)) for s in _ONLY_SEEDS.split(",") if s.strip()]
else:  # draw the per-config seeds ONCE from a BASE_SEED-seeded RNG (deterministic + a real
    # permutation; recreating Random(BASE_SEED) per iteration would draw the same value each time).
    _rng = random.Random(BASE_SEED)
    _CONFIGS = [_gen(_rng.randint(0, 2**31 - 1)) for _ in range(NUM_TESTS)]


# ---------------------------------------------------------------------------
# Per-quant-mode SNAP (round-trip a bf16 tensor onto that mode's exact grid -> input quant lossless)
# + tight tolerance. Same model as the MoE harness: with exact-grid inputs the oracle measures the
# accumulation/requant floor, not the (large) dense-random quant error, so C can be tight.
# ---------------------------------------------------------------------------
def _snap_bf16(t, dtype=None):
    return t  # bf16 is exactly representable; the gemm accumulates in fp32


def _snap_fp8(t, dtype):
    dt = dtype or torch.float8_e4m3fn
    q, s = to_float8(t, dtype=dt)
    return (q.float() * s).to(torch.bfloat16)


def _snap_mxfp8(t, dtype=None):
    # mxfp8 = e4m3 + per-32-element pow2 (e8m0) block scale along the LAST (K) dim -- for both the
    # [.,M,K] activation and the [.,N,K] weight (the kernel quantizes the contiguous [b,n,k] weight,
    # then passes its transpose), so a plain last-dim round-trip matches the kernel's own grid.
    # swizzled vs linear changes only scale STORAGE, not the quantized values -> linear (which is
    # host-dequantizable) here equals the kernel's swizzled quant.
    flat = t.reshape(-1, t.shape[-1])
    q, s = mxfp8_quantize(flat, False)  # linear (non-swizzled) scale layout
    # mxfp8_dequantize_host is a HOST function (like e2m1_..._to_float in _snap_nvfp4) -> move to CPU
    # first, else it dereferences device pointers on the host and segfaults. Its value-tensor must be
    # uint8-wrapped e4m3 (CHECK_INPUT_TYPE dl_uint8 -- not a kernel bug, q is e4m3) and 2D, and the
    # scale layout must MATCH the quantize layout (decoding linear scales as swizzled reads OOB).
    deq = mxfp8_dequantize_host(
        q.view(torch.uint8).cpu(), s.cpu(), is_sf_swizzled_layout=False
    )
    return deq.reshape(t.shape).to(t.device, torch.bfloat16)


def _snap_nvfp4(t, dtype=None):
    flat = t.reshape(-1, t.shape[-1])
    gsf = (448 * 6) / flat.float().abs().nan_to_num().max().clamp_min(1e-30)
    q, sf = nvfp4_quantize(flat, gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    deq = e2m1_and_ufp8sf_scale_to_float(
        q.cpu(), sf.cpu().view(torch.uint8).reshape(-1), (1.0 / gsf).cpu(), 16, 1, False
    )
    return deq.reshape(t.shape).to(t.device, torch.bfloat16)


def _snap_mxfp4(t, dtype=None):
    flat = t.reshape(-1, t.shape[-1])
    q, sf = mxfp4_quantize(flat)
    deq = mxfp4_dequantize(q, sf)
    return deq.reshape(t.shape).to(t.device, torch.bfloat16)


# (snap, atol_frac, rtol). atol = atol_frac*||ref||_inf. Tight because inputs are exact-grid; final
# values calibrated on SM100 (observed max-ratio per mode, ~2x margin) -- see the RATIO diagnostic.
_QMODE = {
    # Calibrated on SM100 (60-config sweep, observed max-ratio): bf16 .0027, fp8 .0092 (incl e5m2),
    # mxfp4 .0039, nvfp4 .053. atol_frac set ~3-5x that (extra margin on bf16/fp8 for cross-arch).
    "bf16": (_snap_bf16, 1.5e-2, 1.0e-2),
    "fp8": (_snap_fp8, 4.0e-2, 5.0e-2),
    "mxfp8": (_snap_mxfp8, 4.0e-2, 5.0e-2),
    "nvfp4": (_snap_nvfp4, 1.2e-1, 1.0e-1),
    "mxfp4": (_snap_mxfp4, 2.0e-2, 5.0e-2),
}


def _sparse(shape, rng):
    """~75% zeros, bf16, deterministically seeded -> short gemm reductions so structural bugs are
    gross, not averaged away (the MoE/cuDNN sparse trick)."""
    g = torch.Generator(device="cuda").manual_seed(rng.randint(0, 2**31 - 1))
    dense = torch.randn(shape, device="cuda", generator=g)
    keep = torch.rand(shape, device="cuda", generator=g) >= 0.75
    return (dense * keep).to(torch.bfloat16)


def _canonical(cfg: Cfg, salt: int):
    """Sparse + exact-grid NT inputs: a=[(B,)M,K], b=[(B,)N,K] snapped to the quant mode's grid;
    reference = snapped_a @ snapped_b^T (fp32) -- the authoritative oracle."""
    snap = _QMODE[cfg.adapter.quant_mode][0]
    pre = (cfg.b,) if cfg.adapter.op_shape == "bmm" else ()
    a = snap(_sparse((*pre, cfg.m, cfg.k), random.Random(cfg.seed + salt)), cfg.fp8_idt)
    b = snap(
        _sparse((*pre, cfg.n, cfg.k), random.Random(cfg.seed + salt + 1)), cfg.fp8_mdt
    )
    ref = torch.matmul(a.float(), b.float().transpose(-2, -1))
    return a, b, ref


def _out_buffer(cfg: Cfg):
    pre = (cfg.b,) if cfg.adapter.op_shape == "bmm" else ()
    # POISON: NaN-fill so a kernel that doesn't fully write its output leaks and is caught.
    return torch.full(
        (*pre, cfg.m, cfg.n), float("nan"), device="cuda", dtype=cfg.out_dtype
    )


# ---------------------------------------------------------------------------
# Diagnostics: every test prints its full config + a perfect-repro command; on a numeric
# mismatch we dump output-vs-oracle stats + the worst <=100 elements, so the CI log alone tells
# you whether the output is all-zero / all-NaN / Inf without having to rerun.
# ---------------------------------------------------------------------------
def _describe(cfg: Cfg) -> str:
    return (
        f"CONFIG {cfg.label}\n"
        f"  adapter={cfg.adapter.key} op={cfg.adapter.op_shape} quant={cfg.adapter.quant_mode} "
        f"backend={cfg.backend} convention={cfg.adapter.convention}\n"
        f"  shape: b={cfg.b} m={cfg.m} n={cfg.n} k={cfg.k}  out_dtype={cfg.out_dtype}\n"
        f"  modes: noncontig={cfg.noncontig} use_8x4={cfg.use_8x4} "
        f"fp8=({cfg.fp8_idt},{cfg.fp8_mdt})  seed={cfg.seed}"
    )


def _repro(cfg: Cfg) -> str:
    cuda = os.environ.get("CUDA_HOME", "<cuda>")
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "<sm100-idx>")
    return (
        f"REPRO: CUDA_HOME={cuda} CUDA_VISIBLE_DEVICES={dev} "
        f"FLASHINFER_GEMM_FUZZ_ONLY_SEED={cfg.seed} "
        f"pytest -s tests/gemm/test_unified_gemm_fuzz.py::test_unified_gemm_fuzz"
    )


def _stats(t: torch.Tensor) -> str:
    tf = t.float()
    n = tf.numel()
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} nan={int(torch.isnan(tf).sum())} "
        f"inf={int(torch.isinf(tf).sum())} zero={int((tf == 0).sum())}/{n} "
        f"max|.|={tf.abs().nan_to_num().max().item():.4g}"
    )


def _dump(out: torch.Tensor, ref: torch.Tensor, k: int = 100) -> str:
    of, rf = out.float().reshape(-1), ref.float().reshape(-1)
    diff = (of - rf).abs()
    # rank by |diff|, treating non-finite diffs as worst so NaN/Inf elems surface first.
    diffn = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
    idx = torch.topk(diffn, min(k, diffn.numel())).indices.tolist()
    lines = [
        f"  output: {_stats(out)}",
        f"  oracle: {_stats(ref)}",
        f"  worst {len(idx)} elems  [flat_idx]  output  vs  oracle:",
    ]
    lines += [f"    [{i}] {of[i].item():.6g}  vs  {rf[i].item():.6g}" for i in idx]
    return "\n".join(lines)


def _fail(cfg: Cfg, why: str, out=None, ref=None):
    parts = [why, _describe(cfg)]
    if out is not None and ref is not None:
        parts.append(_dump(out, ref))
    parts.append(_repro(cfg))
    pytest.fail("\n".join(parts))


def _as_noncontig(t: torch.Tensor) -> torch.Tensor:
    """Return a value-identical NON-contiguous view by stuffing values into a 2x-wide buffer at
    stride 2 and slicing them back -- robust even when a leading dim is size 1 (last-dim padding
    would still be contiguous there). Deterministic; no RNG. (B1 helper, currently unused.)"""
    pad = torch.empty((*t.shape[:-1], t.shape[-1] * 2), device=t.device, dtype=t.dtype)
    pad[..., ::2] = t
    return pad[..., ::2]


def _ratio(res: torch.Tensor, ref: torch.Tensor) -> float:
    """max|out-ref| / ||ref||_inf -- the per-config error ratio (used to calibrate atol_frac)."""
    denom = ref.abs().max().item()
    return (res.float() - ref).abs().max().item() / denom if denom > 0 else 0.0


def _assert_invariants(cfg: Cfg, res: torch.Tensor, ref: torch.Tensor):
    resf = res.float()
    # (1) reference-aware no-spurious-NaN/Inf. Inputs are exact-grid + well-conditioned (sparse
    #     randn), so the reference is finite and any non-finite output is a real defect.
    bad = int(((~torch.isfinite(resf)) & torch.isfinite(ref)).sum().item())
    if bad != 0:
        _fail(
            cfg,
            f"{bad}/{resf.numel()} spurious NaN/Inf where oracle is finite (#2440/#3103/#3334-class)",
            res,
            ref,
        )
    # (2) TIGHT numeric vs the authoritative exact-grid reference (replaces the loose cosine). With
    #     lossless input quant, atol_frac is the accumulation/requant floor -> catches structural
    #     bugs grossly AND sub-floor accuracy regressions a cosine oracle misses.
    _, atol_frac, rtol = _QMODE[cfg.adapter.quant_mode]
    atol = atol_frac * ref.abs().max().item() + 1e-3
    over = (resf - ref).abs() > (atol + rtol * ref.abs())
    if over.any():
        _fail(
            cfg,
            f"{int(over.sum())}/{resf.numel()} elems exceed tol "
            f"(atol_frac={atol_frac} rtol={rtol} atol={atol:.3g}, ratio={_ratio(res, ref):.4g})",
            res,
            ref,
        )
    # (no separate "all-zero output" invariant: an all-zero/all-NaN output necessarily fails (2),
    # and the failure dump -- output/oracle stats + the worst 100 elements -- makes the pattern
    # self-evident. A standalone predicate false-positived in CI on a tiny sparse config where the
    # ORACLE itself was 223/224 zero and the output correctly matched it.)


@pytest.fixture(autouse=True)
def _fresh_autotune_cache():
    # autotune(True) populates the AutoTuner's process-global profiling_cache, and autotune(False)
    # will REUSE a cached winner if one exists -- so without clearing, a config's result could
    # depend on which earlier configs tuned the same shape (breaks per-seed repro + order-independence).
    # Clear the *profiling* cache before each test (cheap). We deliberately do NOT clear the cuDNN
    # graph/plan cache (keeping it avoids per-config plan rebuilds; it doesn't affect winner choice).
    with contextlib.suppress(Exception):
        AutoTuner.get().clear_cache()
    yield


def _validate(cfg: Cfg, out: torch.Tensor, ref: torch.Tensor):
    """Assert correctness vs the authoritative reference; a tracked finding is tolerated (xfail)."""
    try:
        _assert_invariants(cfg, out, ref)
    except (AssertionError, pytest.fail.Exception):
        known = _known_failure(cfg)
        if known:
            pytest.xfail(f"KNOWN FINDING: {known}")
        raise


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_unified_gemm_fuzz(cfg: Cfg):
    # Determinism: pin the global RNG too (device-state probe + any global draw), so the run is
    # bit-reproducible from cfg.seed alone (inputs/buffer already use per-config seeded generators).
    torch.manual_seed(cfg.seed)
    # Every test prints its full config + the exact repro command (captured by pytest; shown on
    # failure, or always with `-s`) so a CI log is self-explanatory.
    print("\n" + _describe(cfg))
    print(_repro(cfg))

    # A crash-capable LEDGERED config must not run at all: the same root cause that produces
    # garbage output can be an out-of-bounds access on another library stack (seen live in CI:
    # bmm_mxfp8 b=16,m=7 IMA'd on GB200/cu129 and poisoned the CUDA context for all ~1000 later
    # tests in the session). xfail up front; to probe whether a finding is fixed, run its REPRO
    # with the ledger entry removed. Numeric-only entries still run (xfail at compare time).
    known = _known_failure(cfg, crash_only=True)
    if known:
        pytest.xfail(f"KNOWN FINDING (not run; crash-capable): {known}")

    a, b, ref = _canonical(cfg, salt=1)
    if cfg.noncontig:  # feed value-identical non-contiguous views (oracle unchanged)
        a, b = _as_noncontig(a), _as_noncontig(b)
    res = _out_buffer(cfg)

    def _call(out):
        with autotune(False):  # determinism: pin the tactic
            cfg.adapter.run(a, b, out, cfg.backend, cfg)
        torch.cuda.synchronize()

    try:
        _call(res)
    except LibraryError as e:
        if str(e) == CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR:
            pytest.skip(str(e))
        raise
    except pytest.skip.Exception:
        raise
    except Exception as e:
        if _is_unsupported(e):
            pytest.skip(f"unsupported {cfg.label}: {e}")
        raise  # a crash is a finding

    print(f"RATIO {cfg.adapter.quant_mode} {cfg.backend} ratio={_ratio(res, ref):.4g}")
    # autotune-OFF (default tactic) correctness vs the authoritative reference; tracked findings xfail.
    _validate(cfg, res, ref)

    # (3) determinism: re-run into a freshly-poisoned buffer; a deterministic backend must match
    #     bit-exactly.
    if cfg.adapter.deterministic:
        res2 = _out_buffer(cfg)
        _call(res2)
        if not torch.equal(res, res2):
            _fail(
                cfg,
                "NONDETERMINISTIC output across identical runs (#2514-class); "
                "first run = output, second run = oracle below",
                res,
                res2,
            )

    # (5) device-state probe: a context-corrupting IMA above surfaces here as a non-finite probe.
    probe = torch.randn(2048, device="cuda") * 2.0
    torch.cuda.synchronize()
    if not torch.isfinite(probe).all():
        _fail(cfg, "CUDA context corrupted after GEMM (device-state probe non-finite)")

    # (a') autotune-ON winner validation (gated ~20%) -- run LAST: it populates the profiling cache
    # (autotune(False) would then reuse that winner), so doing it after the determinism check above
    # keeps that check on the clean default tactic. Runs the REAL autotuner-selected winner and
    # validates it vs the SAME reference -> catches "autotuner picks a fast-but-wrong tactic"
    # (#3227/#3398/#2504-class) on exactly what users get. Backend-agnostic; a backend that rejects
    # under tuning just falls back to the OFF result already validated above. Gated to ~5% because
    # autotune profiling is ~5-8s/config (it times every tactic); the cheap autotune-OFF path above
    # carries the breadth, this adds bounded autotuner-correctness depth.
    if cfg.seed % 20 == 0:
        a_out = _out_buffer(cfg)
        try:
            with autotune(True):
                cfg.adapter.run(a, b, a_out, cfg.backend, cfg)
            torch.cuda.synchronize()
        except Exception as e:
            if not _is_unsupported(e):
                raise
        else:
            _validate(cfg, a_out, ref)


# ---------------------------------------------------------------------------
# Convention conformance auditor: cross-check backends that DECLARE the same convention (a real
# cross-backend oracle), document cross-mode divergences, and print the convention matrix. This is
# the hook a future unified GEMM API / convention-compat fix gets enforced through.
# ---------------------------------------------------------------------------
def _conformance_groups():
    """Group (adapter, backend) pairs by (op_shape, quant_mode, frozen convention). Within a group
    all backends share one declared recipe -> they MUST agree numerically."""
    groups = {}
    for ad in _ADAPTERS:
        if not ad.supported(_SM):
            continue
        gkey = (ad.op_shape, ad.quant_mode, tuple(sorted(ad.convention.items())))
        groups.setdefault(gkey, []).append(ad)
    return groups


def test_convention_conformance(capsys):
    if _SKIP is not None:
        pytest.skip(_SKIP)
    groups = _conformance_groups()
    if not groups:
        pytest.skip(f"no wired GEMM adapter on SM{_SM}")

    matrix_lines = ["", "=== GEMM convention matrix (declared scale conventions) ==="]
    checked_pairs = 0
    for ad in _ADAPTERS:
        mark = "" if ad.supported(_SM) else "  [unsupported on this arch]"
        matrix_lines.append(
            f"  {ad.key:12s} {ad.op_shape}/{ad.quant_mode:6s} {ad.convention}{mark}"
        )

    # Cross-backend agreement WITHIN each shared-convention group (the valid equality oracle).
    for (op_shape, quant_mode, _conv), ads in groups.items():
        # all adapters in a group share key fields; iterate every backend of every adapter
        runs = []
        cfg = Cfg(
            seed=20260611,
            adapter=ads[0],
            backend="",
            b=2,
            m=128,
            n=256,
            k=256,
            out_dtype=torch.bfloat16,
        )
        a, b, _ref = _canonical(cfg, salt=7)
        for ad in ads:
            for be in ad.backends:
                out = _out_buffer(cfg)
                try:
                    with autotune(False):
                        ad.run(a, b, out, be, cfg)
                    torch.cuda.synchronize()
                except Exception as e:
                    if _is_unsupported(e):
                        continue
                    raise
                runs.append((f"{ad.key}:{be}", out.float()))
        # All runs in a shared-convention group must agree (loosely; quant noise) with each other.
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                (n1, o1), (n2, o2) = runs[i], runs[j]
                cos = F.cosine_similarity(o1.reshape(-1), o2.reshape(-1), dim=0).item()
                checked_pairs += 1
                assert cos > 0.95, (
                    f"shared-convention {op_shape}/{quant_mode}: {n1} vs {n2} disagree "
                    f"(cos {cos:.4f}) -- a backend deviates from the DECLARED convention"
                )

    matrix_lines.append("")
    matrix_lines.append(
        "=== documented cross-mode convention divergences (NOT cross-compared) ==="
    )
    for k1, k2, why in _CONVENTION_DIVERGENCES:
        matrix_lines.append(f"  {k1} <-> {k2}: {why}")
    matrix_lines.append("")
    matrix_lines.append(f"cross-backend conformance pairs checked: {checked_pairs}")
    with capsys.disabled():
        print("\n".join(matrix_lines))


# ---------------------------------------------------------------------------
# Standalone quantize-root fuzzer (the #2440 class): quantizing FINITE inputs must never emit
# non-finite scale factors -- the root cause behind several low-precision GEMM NaN reports, and a
# class a GEMM-call-only fuzzer reaches only indirectly. Ported from test_mm_fp4_fuzz.py's
# test_fp4_quantize_fuzz and extended to fp8 to_float8.
# ---------------------------------------------------------------------------
_Q_CONFIGS = [
    (rows, cols, regime, qmode)
    for rows in (1, 7, 64, 2048)
    for cols in (16, 64, 256, 1024)
    for regime in ("tiny", "large", "with_zeros", "skewed", "normal")
    for qmode in ("nvfp4", "mxfp4", "fp8")
]


@pytest.mark.parametrize(
    "rows,cols,regime,qmode",
    _Q_CONFIGS,
    ids=[f"{q}_{r}x{c}_{g}" for r, c, g, q in _Q_CONFIGS],
)
def test_gemm_quantize_fuzz(rows, cols, regime, qmode):
    """Quantizing finite inputs must never produce NaN/Inf scale factors (GH #2440)."""
    if qmode in ("nvfp4", "mxfp4") and _SM < 100:
        pytest.skip(
            f"fp4 quantize requires SM100 (got SM{_SM})"
        )  # to_float8/fp8 runs on SM80+
    block = {"nvfp4": 16, "mxfp4": 32, "fp8": 1}[qmode]
    cols = max(block, (cols // block) * block)
    # Deterministic seed across processes: Python's hash() of str/tuple is randomized per-process
    # (PYTHONHASHSEED), which would break repro; crc32 of a stable string is stable.
    seed = zlib.crc32(f"{rows}_{cols}_{regime}_{qmode}".encode())
    x = _regime_tensor((rows, cols), regime, random.Random(seed))
    assert torch.isfinite(x.float()).all(), "test input itself is non-finite"
    if qmode == "nvfp4":
        gsf = (448 * 6) / x.float().abs().nan_to_num().max().clamp_min(1e-30)
        _q, sf = nvfp4_quantize(
            x, gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
    elif qmode == "mxfp4":
        _q, sf = mxfp4_quantize(x)
    else:  # fp8 per-tensor
        _q, sf = to_float8(x, dtype=torch.float8_e4m3fn)
    n_bad = int((~torch.isfinite(sf.float())).sum().item())
    assert n_bad == 0, (
        f"{qmode} quantize({rows}x{cols},{regime}) produced {n_bad} non-finite "
        f"scale-factor elements from finite input (GH #2440-class)"
    )


# ---------------------------------------------------------------------------
# Autotune-cache + dynamic-shape coherence (backend-agnostic). Under autotune(True), drive ONE
# backend over a SEQUENCE of M (the dynamic dim): the first M's build/profile the per-bucket winner
# + plan; later M's (incl odd / in-between / repeats) REUSE the cached winner and OVERRIDE the shape
# to the new M -- the cuDNN override-shape path (>=9.21) and the analogous cross-call cache reuse in
# cutlass/cublas/trtllm. A stale/mis-keyed cached winner or a bad shape-override -> wrong output,
# caught against the per-M reference. No backend special-casing: the cuDNN override-shape regression
# surface is exercised by the same generic sequence that exercises every other backend.
# ---------------------------------------------------------------------------
_DYNSHAPE_M_SEQ = [
    256,
    4096,
    17,
    4097,
    3,
    256,
    4096,
    129,
    256,
]  # buckets + boundaries + odd + repeats


@pytest.mark.parametrize(
    "akey", [a.key for a in _ADAPTERS], ids=[a.key for a in _ADAPTERS]
)
def test_autotune_cache_dynshape(akey):
    if _SKIP is not None:
        pytest.skip(_SKIP)
    ad = next(a for a in _ADAPTERS if a.key == akey)
    if not ad.supported(_SM):
        pytest.skip(f"{akey} unsupported on SM{_SM}")
    torch.manual_seed(0)
    al = ad.align
    N, K = max(al, (512 // al) * al), max(al, (512 // al) * al)
    fp8 = ad.quant_mode == "fp8"
    n_ran = 0
    with autotune(
        True
    ):  # fill the per-bucket cache on first sight; reuse + override on later M
        for i, m in enumerate(_DYNSHAPE_M_SEQ):
            cfg = Cfg(
                seed=70000 + i,
                adapter=ad,
                backend="auto",
                b=2 if ad.op_shape == "bmm" else 1,
                m=m,
                n=N,
                k=K,
                out_dtype=torch.bfloat16,
                fp8_idt=torch.float8_e4m3fn if fp8 else None,
                fp8_mdt=torch.float8_e4m3fn if fp8 else None,
            )
            if _known_failure(cfg, crash_only=True):
                # crash-capable ledgered configs may IMA, not just miscompute (bmm_mxfp8 b>1,
                # M%128!=0 on GB200/cu129) -> never run them; skip this M, keep the sequence going.
                continue
            a, b, ref = _canonical(cfg, salt=1)
            out = _out_buffer(cfg)
            try:
                ad.run(a, b, out, "auto", cfg)
                torch.cuda.synchronize()
            except Exception as e:
                if _is_unsupported(e):
                    continue  # this M unsupported (e.g. unaligned fp4) -> skip it, keep the sequence
                raise
            n_ran += 1
            _validate(cfg, out, ref)
    if n_ran == 0:
        pytest.skip(f"no M in the dyn-shape sequence ran for {akey} on SM{_SM}")
