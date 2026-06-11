"""Unified fuzzer + convention auditor for flashinfer's scaled GEMM/BMM family.

ONE harness for the whole {op-shape} x {quant-mode} x {backend} cross-product instead of N
per-API files, built on the same best-practice oracle kit as the unified MoE fuzzer
(``tests/moe/test_unified_moe_fuzz.py``):
  * adversarial input regimes (tiny / large / mixed-rows / with-zeros / skewed) + nasty shapes
    (non-pow2, the "in-between" M that broke tile selection #3398, degenerate 1/2/3),
  * reference-aware no-spurious-NaN/Inf (only flag non-finite where the fp32 ref is finite),
  * not-(almost)-all-zero vs a non-trivial reference (#3398/#3068 M-dependent / arch-divergent),
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

STATUS: framework + 4 grounded adapters (bf16 mm / fp8 bmm incl e4m3+e5m2 / nvfp4 mm / mxfp4 mm) +
a standalone quantize-root fuzzer. This file REPLACES the earlier per-op ``test_mm_fp4_fuzz.py`` /
``test_bmm_fp8_fuzz.py`` (their logic + the #2440 quantize test are folded in here). Validated on
SM100 (B200). EXTEND by folding in more adapters one ``GemmAdapter`` at a time: mm_fp8 (low-latency,
needs ``prepare_low_latency_gemm_weights`` -- a distinct convention), mm_mxfp8, bmm_mxfp8, and the
grouped ``group_*`` / ``*deepgemm*`` entries (m_indptr op-shape).

Run (needs CUDA_HOME for JIT; SM100 for the fp4 adapters, SM89+ for fp8):
  CUDA_HOME=<cuda> CUDA_VISIBLE_DEVICES=<idx> pytest -q tests/gemm/test_unified_gemm_fuzz.py
Env: FLASHINFER_GEMM_FUZZ_NUM_TESTS (default 250), FLASHINFER_GEMM_FUZZ_SEED (default 0).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from flashinfer import (
    SfLayout,
    autotune,
    bmm_fp8,
    mm_bf16,
    mm_fp4,
    mxfp4_quantize,
    nvfp4_quantize,
)
from flashinfer.utils import LibraryError, get_compute_capability

try:
    from flashinfer.gemm.gemm_base import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR
except Exception:  # pragma: no cover - constant may move between versions
    CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR = "\0never-matches\0"
from tests.utils_fp8 import to_float8

NUM_TESTS = int(os.environ.get("FLASHINFER_GEMM_FUZZ_NUM_TESTS", "250"))
BASE_SEED = int(os.environ.get("FLASHINFER_GEMM_FUZZ_SEED", "0"))

_SKIP = None
_CC = (0, 0)
if not torch.cuda.is_available():
    _SKIP = "CUDA not available"
else:
    _CC = get_compute_capability(torch.device("cuda:0"))
    if _CC[0] < 8:
        _SKIP = f"scaled GEMM needs SM80+, got SM{_CC[0]}{_CC[1]}"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
_SM = _CC[0] * 10 + _CC[1]

# A clean rejection ("not supported on this shape/arch/dtype") must SKIP, never fail; a crash
# (CUDA error / illegal access / device assert) is always a finding and re-raises.
_SKIP_KW = (
    "not support",
    "unsupported",
    "no kernel",
    "no valid engine",
    "invalid",
    "must be",
    "requires",
    "only support",
    "not implemented",
    "alignment",
    "e5m2",
)
_CRASH_KW = ("cuda error", "illegal memory", "misaligned", "device-side assert")

_REGIMES = ["normal", "tiny", "large", "mixed_rows", "with_zeros", "skewed"]
_INRANGE = (
    "normal",
    "with_zeros",
)  # where a spurious-NaN / all-zero assertion is meaningful


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
    quant_mode: str  # "bf16" | "fp8" | "nvfp4" | "mxfp4"
    backends: tuple  # selectable `backend=` values
    convention: dict  # DECLARED scale convention (the audit substrate)
    supported: Callable[[int], bool]  # (sm) -> bool
    run: Callable  # (a_bf16, b_bf16, out, backend, cfg) -> None  (writes into `out`)
    deterministic: bool = True
    align: int = 1  # required N/K alignment (snap shapes to this)


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
    bt = b.transpose(-2, -1).contiguous()  # [B, K, N]
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
        use_8x4_sf_layout=False,
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
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=False,
        skip_check=False,
    )


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
    # EXTEND: mm_fp8 (trtllm_low_latency, needs prepare_low_latency_gemm_weights -> a distinct
    # convention), mm_mxfp8, bmm_mxfp8, group_gemm_*_nt_groupwise / *deepgemm* (grouped op-shape
    # with m_indptr offsets). Add one GemmAdapter each; unsupported (op_shape, arch) just skips.
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
    regime: str
    out_dtype: torch.dtype
    fp8_idt: Optional[torch.dtype] = (
        None  # fp8 A-operand dtype (e4m3/e5m2); None -> e4m3
    )
    fp8_mdt: Optional[torch.dtype] = None  # fp8 B-operand dtype

    @property
    def label(self):
        sh = f"b{self.b}_" if self.adapter.op_shape == "bmm" else ""
        dt = str(self.out_dtype).split(".")[-1]
        fp8 = (
            f"_{str(self.fp8_idt).split('_')[-1]}x{str(self.fp8_mdt).split('_')[-1]}"
            if self.fp8_idt is not None
            else ""
        )
        return (
            f"{self.adapter.key}_{self.backend}_{sh}m{self.m}_n{self.n}_k{self.k}_"
            f"{self.regime}_{dt}{fp8}_s{self.seed}"
        )


def _gen(seed) -> Cfg:
    rng = random.Random(seed)
    wired = [a for a in _ADAPTERS if a.supported(_SM)]
    ad = rng.choice(wired)
    al = ad.align
    n = max(al, (rng.choice(_SHAPES_N) // al) * al)
    k = max(al, (rng.choice(_SHAPES_K) // al) * al)
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
        regime=rng.choice(_REGIMES),
        out_dtype=rng.choice([torch.bfloat16, torch.float16]),
        fp8_idt=fp8_idt,
        fp8_mdt=fp8_mdt,
    )


_CONFIGS = (
    [
        _gen(
            random.Random(BASE_SEED).randint(0, 2**31 - 1)
            ^ (i * 2654435761 & 0x7FFFFFFF)
        )
        for i in range(NUM_TESTS)
    ]
    if _SKIP is None and any(a.supported(_SM) for a in _ADAPTERS)
    else []
)


def _canonical(cfg: Cfg, salt: int):
    """NT-convention canonical bf16 inputs: a=[(B,)M,K], b=[(B,)N,K]; reference = a @ b^T (fp32)."""
    pre = (cfg.b,) if cfg.adapter.op_shape == "bmm" else ()
    a = _regime_tensor((*pre, cfg.m, cfg.k), cfg.regime, random.Random(cfg.seed + salt))
    b = _regime_tensor(
        (*pre, cfg.n, cfg.k), cfg.regime, random.Random(cfg.seed + salt + 1)
    )
    ref = torch.matmul(a.float(), b.float().transpose(-2, -1))
    return a, b, ref


def _out_buffer(cfg: Cfg):
    pre = (cfg.b,) if cfg.adapter.op_shape == "bmm" else ()
    # POISON: NaN-fill so a kernel that doesn't fully write its output leaks and is caught.
    return torch.full(
        (*pre, cfg.m, cfg.n), float("nan"), device="cuda", dtype=cfg.out_dtype
    )


def _assert_invariants(cfg: Cfg, res: torch.Tensor, ref: torch.Tensor):
    resf = res.float()
    # (1) reference-aware no-spurious-NaN/Inf (only meaningful in-range; edge regimes legitimately
    #     overflow the dtype and the reference overflows too).
    if cfg.regime in _INRANGE:
        ref_od = ref.to(cfg.out_dtype).float()
        bad = int(((~torch.isfinite(resf)) & torch.isfinite(ref_od)).sum().item())
        assert bad == 0, (
            f"{cfg.label}: {bad}/{resf.numel()} spurious NaN/Inf where ref finite "
            f"(#2440/#3103/#3334-class)"
        )
    # (2) not (almost) all-zero vs a non-trivial reference (#3398/#3068 M-dependent / arch-divergent).
    if cfg.regime in _INRANGE and ref.abs().max().item() > 1e-3:
        nz = (resf.abs() > 0).float().mean().item()
        assert nz > 0.01, (
            f"{cfg.label}: ~all-zero output (nz={nz:.4f}) vs non-trivial reference "
            f"(#3398/#3068-class)"
        )
    # (4) loose numeric oracle only on the well-conditioned regime (FP4/FP8 quant error is large in
    #     edge regimes; structural invariants cover those). A tight snapped-input reference is the
    #     planned upgrade (see module docstring / MoE harness check #2).
    if cfg.regime == "normal" and ref.abs().max().item() > 1e-2:
        cos = F.cosine_similarity(ref.reshape(-1), resf.reshape(-1), dim=0).item()
        assert cos > 0.90, f"{cfg.label}: cosine {cos:.4f} too low vs reference"


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_unified_gemm_fuzz(cfg: Cfg):
    a, b, ref = _canonical(cfg, salt=1)
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

    _assert_invariants(cfg, res, ref)

    # (3) determinism: re-run into a freshly-poisoned buffer; a deterministic backend must match
    #     bit-exactly.
    if cfg.adapter.deterministic:
        res2 = _out_buffer(cfg)
        _call(res2)
        if not torch.equal(res, res2):
            md = (res.float() - res2.float()).abs().max().item()
            pytest.fail(
                f"{cfg.label}: NONDETERMINISTIC output (max abs diff {md:.3e}, #2514-class)"
            )

    # (5) device-state probe: a context-corrupting IMA above surfaces here as a non-finite probe.
    probe = torch.randn(2048, device="cuda") * 2.0
    torch.cuda.synchronize()
    assert torch.isfinite(probe).all(), (
        f"{cfg.label}: CUDA context corrupted after GEMM"
    )


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
            regime="normal",
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
    block = {"nvfp4": 16, "mxfp4": 32, "fp8": 1}[qmode]
    cols = max(block, (cols // block) * block)
    x = _regime_tensor(
        (rows, cols), regime, random.Random(hash((rows, cols, regime, qmode)) & 0xFFFF)
    )
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
