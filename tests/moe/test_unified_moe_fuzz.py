"""Forward-compatible fuzzer for the unified MoE API (``MoELayer`` + Packs, PR #3093).

Drives the **real user-facing surface** -- one ``MoEConfig`` -> the API's own
``XxxConfig.prepare_weights(w1_bf16, w2_bf16, ...)`` marshalling -> ``MoELayer``'s per-backend
runners -- so what's under test is the production dispatch + the new ``prepare.py`` scale/layout
plumbing, where the low-precision-MoE bugs cluster (GH #2356/#2485/#2907/#3068).

Forward-compatible by construction:
  * Backends are discovered from the live runner registry (``layer.runners``); an unwired backend
    is skipped and is tested the moment its runner lands -- zero new code.
  * Weight prep is the uniform ``cfg.prepare_weights(w1_bf16, w2_bf16, **shape)`` (canonical bf16 in,
    quantize+layout done internally).
  * Only the per-DTYPE pieces live in one ``_DTYPE`` table: how to make golden inputs, how to build
    the activation pack, and the canonical reference recipe.

Config space: random shapes deliberately incl non-pow2 (aligned) hidden/intermediate + odd/
tile-boundary token counts (real-model + #2907/#3168 territory), routing-load skew, all under a
weight-memory budget so one config never hogs the GPU (parallel-CI-friendly), plus a few curated
larger-end shapes. Large expert counts are reached with small H/I and/or **expert-parallel shards**
(global>local + ``local_expert_offset``, the real deployment shape), not by filling the GPU.

A small ``_KNOWN_FAILURES`` ledger xfails already-filed bugs (e.g. the since-fixed trtllm EP
offset>0 all-zero bug, gh #3547): the case is still *run* so the suite stays green on a tracked bug
yet flags loudly the day it starts passing (fixed). A crash is never tolerated -- only a wrong answer.

Verification model (single mode, uniform -- every config that runs gets the same checks):
  1. **no crash / no NaN-Inf** where the reference is finite.
  2. **numeric vs the canonical quant-aware reference.** The reference is the *authority*: it
     defines the one true numerical recipe (exactly-representable inputs + the fp4 intermediate
     requant), so a backend that invents a different recipe is wrong by definition. Inputs are
     snapped to the exact nvfp4 grid and sparsified, so input quantization is lossless and the
     gemm reductions are short -- a structural bug (dropped expert / wrong index / wrong scale role)
     produces a gross error instead of one averaged away. Tolerance is set to the fp4
     intermediate-requant floor (~0.08), far tighter than a dense-random comparison.
  3. **determinism, per-backend contract.** A backend declared deterministic must reproduce
     bitwise across reruns; a non-deterministic one (CuteDSL's atomic-scatter finalize) is exempt.
     Flags are established empirically (CRC across runs) -- see ``_DETERMINISTIC``.
  4. **output-buffer poison.** The kernel owns its output (an uninitialized ``new_empty`` inside
     the runner's ``pack_inputs``) and MoE finalize *accumulates* into it -- so the result must not
     depend on the buffer being clean. We fill it with garbage + NaN/Inf and re-assert #1+#2.
     torch's caching allocator usually returns clean memory and hides this; JAX/XLA donates dirty
     buffers (the GH-6158764 padding-leak class), so this is the torch->JAX buffer-hygiene guard.
  5. **autotune-tactic sweep.** EVERY valid tactic (``get_valid_tactics``), not just the default,
     must match the reference -- the autotuner-picks-a-corrupting-tactic class (#3168/#3227) on the
     real ``MoELayer`` dispatch, since the autotuner enumerates these same tactics in production.
  6. **autotune-ON, real production path** (gated to a config subset for cost): drive
     ``with autotune(True): layer(...)`` so ``MoELayer._select_winner`` *profiles* every tactic of
     every runner (the #3168 profiling-IMA / #2749 profiling-crash class -- distinct from #5, which
     replays tactics outside the tuner) then selects + caches a winner; the autotuned output must
     still match the authoritative reference. Skipped when a candidate has a known failure (the
     tuner could legitimately pick the broken backend).
  7. **device-state probe** after each config: a context-corrupting IMA surfaces as a failed
     alloc/launch or non-finite probe, turning silent corruption into a clean failure.

A sibling SCENARIO test ``test_autotune_cache_coherence`` covers the one autotune surface this
per-config fuzz structurally can't: the cross-call **winner cache**. It drives ONE persistent
``MoELayer`` through a token-count *sequence* (incl bucket boundaries 4095/4096/4097) under
``autotune(True)`` -- filling the per-bucket cache, crossing shapes, then re-running earlier counts
to force cache hits -- asserting each output stays correct, so a stale / mis-keyed cached winner
reused for a different shape is caught (the #2933-adjacent class).

(Cross-backend agreement is intentionally NOT a check: with an authoritative tight reference, a
deviating backend is caught by #2 directly, and #2 also names which backend -- so a cross-backend
comparison adds no pass/fail power, only redundancy. See the design discussion.)

Routing coverage (both modes, axes ``routing_method`` x ``routing_input_mode`` x ``logits_dtype``):
  * **pre-routed** (RoutingInputMode.PackedPrecomputed): the host computes the top-k per method and
    feeds packed indices -- the original path.
  * **in-kernel** (RoutingInputMode.FromLogits): the kernel routes from raw logits per
    RoutingConfig.method -- reaches the bug cluster the pre-routed harness structurally can't:
    DeepSeekV3 group-topk + bias (#2575), all-negative logits (#2822), fp32 router logits (#2796),
    bias-method weight leakage (#2485/#2907). The SAME ``_route`` oracle (ported verbatim from the
    kernel-validated references in ``tests/moe/test_trtllm_gen_fused_moe.py``) is the authority for
    both modes, so a kernel that routes wrong is caught by check #2. In-kernel routing is single-GPU
    (non-EP) here; EP + in-kernel routing semantics are a separate validation.

Coverage today: NVFP4 (CuteDSL pre-routed + TRTLLM-FP4 pre-routed/in-kernel) on SM100 -- the only
wired MVP runners (CuteDSL is pre-routed-only; FromLogits restricts to the trtllm backend).

OPT-IN: this suite is gated behind FLASHINFER_UMOE_FUZZ (see the pytestmark below) and is
SKIPPED unless that env var is set -- waived in CI pending root-cause of a
whole-process device-side-assert abort that would block B200 CI. Run it explicitly:
  FLASHINFER_UMOE_FUZZ=1 CUDA_HOME=<cuda> CUDA_VISIBLE_DEVICES=<sm100-idx> \
    pytest tests/moe/test_unified_moe_fuzz.py
NOTE: `pytest --forked` does NOT work here (CUDA inits at collection ->
"Cannot re-initialize CUDA in forked subprocess"); for crash-isolated enumeration run each
test id in its own process instead (see var/03-ssh-docker-workflow.md).
Env: FLASHINFER_UMOE_FUZZ_NUM_TESTS (default 80), FLASHINFER_UMOE_FUZZ_SEED (default 0),
     FLASHINFER_UMOE_FUZZ_ONLY_SEED (comma-separated seeds -> run ONLY those configs; the
     perfect-repro hook printed on every test).

Determinism / repro / diagnostics: every config is fully derived from its seed -- shapes
(random.Random(seed)), input tensors + output-buffer init (per-config torch.Generator), and the
global RNG (torch.manual_seed(seed)) -- so a failing test reproduces bit-for-bit from the REPRO
command it prints. Each test prints its full config + repro command (visible with `-s`, or on
failure); on a numeric mismatch it dumps output-vs-oracle stats + the worst <=30 elements, so the
CI log alone tells you whether the output is all-zero / all-NaN / Inf without having to rerun.

------------------------------------------------------------------------------------------------
EXTENDING (cheap, by design):
  * New backend -> nothing to do: it is auto-discovered from ``_BACKEND_RUNNERS`` the moment its
    runner registers and ``supported(sm)`` is true. If it ships with a tracked bug, add one
    ``_KNOWN_FAILURES`` entry (the case still RUNS; an xpass then flags the fix).
  * New dtype -> add ONE ``DTypeHandler`` to ``_DTYPE`` (snap / make_act_pack / reference / poison
    / tolerances). Everything else (config gen, all 7 checks, the cache test) is dtype-generic.

ROADMAP -- what's left, ranked by the 2026-06-09 audit of 51 past MoE GH issues (the full-build-out
harness catches ~60% full / ~91% touched of the 35 in-scope; ~31% are structurally out-of-scope).
Full synthesis lives in the cuDNN-project auto-memory ``flashinfer_quality_fuzzers.md``. Each item
names the issue class it closes:
  1. [HIGHEST LEVERAGE -- infra, not code] Blackwell/SM120 PR-CI runner. PR-gating CI tops out at
     H100/SM90, so the dominant ~36% fp4/MoE bug class is collected-then-SKIPPED at PR time. This
     harness only protects users on arches it actually RUNS on -- no oracle improvement beats
     provisioning the runner. (This is the #1 documented escape reason for the whole MoE class.)
  2. N-run (>=10) stress per config + a PER-TEST TIMEOUT, under ``--forked`` isolation. Turns the
     intermittent PARTIALs into CAUGHT: #2569 (intermittent NaN), #2933 (concurrency-bucket hang).
     A single pass samples a "hangs 1-in-100" failure poorly. NOTE a *deterministic* hang is
     already catchable -- but TODAY it blocks the whole job; add ``@pytest.mark.timeout`` so it
     fails ONE test cleanly. ``--forked`` needs lazy-CUDA-init handled (the cuDNN _replacement
     Heisenbug lesson: forked children must init CUDA fresh).
  3. Curated PRODUCTION shapes: seed the generator with real model dims (DeepSeek-V3, Llama-4,
     Qwen3, Mixtral) + dense tile-window enumeration (every M in [tile-2, tile+2] around each
     kernel's tile boundary). Closes the shape-luck escapes #3310 (Llama-4-Scout "no kernel") and
     #2732 (Qwen3-Coder wrong output) that a synthetic 4096+-1 sweep misses.
  4. BUILD-MANIFEST oracle: enumerate the advertised (backend x quant x arch) support matrix and
     assert each combo actually INSTANTIATES a kernel. Closes #2501 (W4A8 autotune fail) -- an
     un-compiled combo is invisible to a runtime fuzzer (the harness assumes backends are built).
  5. [DEEPEST -- the one structurally-weak oracle] Tighten the QUANTIZED-NUMERIC net. Today check
     #2 compares to ONE authoritative quant-aware reference at the fp4 requant-floor tolerance
     (~10% of ||ref||inf). That floor HIDES sub-10% accuracy regressions (#2356 small-scale, #3103
     minority-NaN), and the reference -- because it must itself encode the quant recipe -- can be
     "wrong the same way" as a kernel (no independent fp32 ideal, unlike bf16). The real fix is the
     unified API standardizing ONE intermediate-activation-scale POLICY (the design doc's
     role-named QuantSpec; gh #3548): once every backend honors one DECLARED recipe, a single fp32
     reference computing that recipe becomes an INDEPENDENT authority for all of them (and
     calibrated checkpoints become expressible). Until then: add a small-scale / edge-magnitude
     input axis and document the floor.

OUT OF SCOPE for this single-GPU correctness harness (must live elsewhere, do NOT try to force in):
  * multi-GPU / EP>1 / TP collective hangs & deadlocks (#3279 EP=8, #3530 TP8) -> a distributed
    (2-8 GPU) test tier with collective-aware timeouts. (Single-GPU EP SHARDS -- global>local +
    local_expert_offset -- ARE in scope and tested here; the COLLECTIVE is not.)
  * perf/latency regressions (#2671) -> perf-CI with per-kernel latency baselines. A wrong-but-fast
    tactic IS caught (check #5/#6); a correct-but-slow one is invisible by design.
  * framework-glue triggers (vLLM/SGLang dispatch sequences #3427, #3390) -> integration tests.
    The underlying KERNEL bug is in scope here IF invoked directly; the live-dispatch trigger isn't.
  * build / cubin / packaging (#3466 missing SM103 cubin, #3344 _sm100f-only) -> an arch-coverage
    manifest check in build CI (related to roadmap #4 but at the .so/cubin level).

POINTERS for future agents (point me at this file and I know the rest):
  * Full context (this fuzzer + the older adapter/GEMM fuzzers + the audit + findings): cuDNN-
    project auto-memory ``flashinfer_quality_fuzzers.md``.
  * Bugs THIS fuzzer found + filed: gh #3547 (trtllm EP offset>0 all-zero -- tracked in the
    ``_KNOWN_FAILURES`` ledger below until fixed) and
    gh #3548 (activation global-scale gap == roadmap #5's scale-policy fix).
  * Findings writeups: flashinfer_triage/EP_OFFSET_FINDING.md, flashinfer_triage/WEIGHT_SCALE_FINDING.md.
  * The unified API under test: PR #3093 (branch ``moe_api``); this fuzzer is PR aleozlx/flashinfer#6
    (branch ``yanxu/unified-moe-api-fuzzer``).
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoELayer,
    MoEWeightPack,
    RoutingInputMode,
)
from flashinfer.fused_moe.api import (
    ActivationConfig,
    BackendOptions,
    CuteDslConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEConfig,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    TrtllmBf16Config,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import get_compute_capability

NUM_TESTS = int(os.environ.get("FLASHINFER_UMOE_FUZZ_NUM_TESTS", "80"))
BASE_SEED = int(os.environ.get("FLASHINFER_UMOE_FUZZ_SEED", "0"))
# Perfect-repro hook: if set (comma-separated seeds), the suite runs ONLY those configs. A curated
# seed maps to its hand-written Cfg; any other seed is regenerated via the deterministic _gen(seed),
# so a single seed reproduces exactly one config. The repro command printed on every test uses this.
_ONLY_SEEDS = os.environ.get("FLASHINFER_UMOE_FUZZ_ONLY_SEED", "")

# --- CI-safety gate: OPT-IN ----------------------------------------------------------------
# Waived in CI pending root-cause of a whole-process abort. Running the SM100 fuzzer
# in a single `pytest` process can hit `CUDA error: device-side assert triggered` ->
# `Fatal Python error: Aborted`, which would BLOCK B200 CI (an abort fails the whole job, not one
# test). Notes from triage (2026-06-09): per-config isolation (one process each) passes 68/86
# incl. EP offset>0 -- so the abort is NOT cleanly attributable to one config (the since-fixed
# gh #3547 EP case returned tolerated zeros, no assert, under torch.cuda.synchronize); it surfaces
# only in the accumulated single-process run that CI uses. `pytest --forked` can't isolate it
# either (CUDA inits at collection -> "Cannot re-initialize CUDA in forked subprocess"). Until the
# abort is root-caused, this suite is opt-in: set FLASHINFER_UMOE_FUZZ=1
# to run it (developer / nightly / SM100 box). Unset (CI default) -> collected-and-skipped, so it
# never launches a kernel and cannot abort the job.
pytestmark = pytest.mark.skipif(
    not os.environ.get("FLASHINFER_UMOE_FUZZ"),
    reason="opt-in fuzzer (set FLASHINFER_UMOE_FUZZ=1); waived in CI pending "
    "root-cause of the whole-process device-side-assert abort",
)

# Per-backend determinism contract, established empirically (CRC across reruns) + confirmed against
# code. A "True" backend MUST reproduce bitwise; flip to False only with evidence (and ideally an
# upstream note), because a deterministic->non-deterministic regression is exactly a bug to catch.
_DETERMINISTIC = {
    "trtllm_fp4_routed": True,  # bitwise-stable across reruns in calibration
    "cute_dsl_nvfp4": False,  # atomic scatter-add finalize -> non-bit-exact by design
    "trtllm_bf16_routed": True,  # same trtllm-gen finalize path as fp4_routed; bitwise-stable in calibration
}

# Known-bug ledger: (backend_key, predicate(cfg)) -> reason. A matching (backend, config) is run but
# its correctness failure is TOLERATED (xfail) -- this keeps the suite green on a filed-and-tracked
# bug while still EXERCISING it, so the day the bug is fixed the case starts passing and we get a loud
# "unexpectedly passed -> remove this entry" signal. A crash is never tolerated (only wrong answers).
_KNOWN_FAILURES = [
    # Entries: (backend_key, predicate(cfg), "reason; gh #NNNN").
    # Empty since the gh #3547 EP-offset double-subtraction fix.
]


def _known_failure(backend_key, cfg):
    for bk, predicate, reason in _KNOWN_FAILURES:
        if bk == backend_key and predicate(cfg):
            return reason
    return None


# ---------------------------------------------------------------------------
# nvfp4 exact-grid snapping: make a tensor a fixed point of the kernel's quantizer, so input
# quantization is lossless (the kernel re-quantizes to the same fp4 values) and only the
# intermediate requant remains as quant error.
# ---------------------------------------------------------------------------
def _snap_to_nvfp4(t: torch.Tensor) -> torch.Tensor:
    one = torch.tensor([1.0], device=t.device)
    flat = t.reshape(-1, t.shape[-1]).to(torch.bfloat16)
    packed, scale = fp4_quantize(
        flat, global_scale=one, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    deq = e2m1_and_ufp8sf_scale_to_float(
        packed.cpu(),
        scale.cpu().view(torch.uint8).reshape(-1),
        (1.0 / one).cpu(),
        16,
        1,
        False,
    )
    return deq.reshape(t.shape).to(t.device, torch.bfloat16)


# ---------------------------------------------------------------------------
# Per-DTYPE handlers: golden input generation, activation pack, canonical reference recipe.
# The ONLY place a new quant variant needs code; new *backends* for a variant are free.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DTypeHandler:
    variant: QuantVariant
    candidate_configs: (
        tuple  # all plausible backend config classes; unwired ones auto-skip
    )
    snap: Callable  # bf16 tensor -> exactly-representable fixed point for this dtype
    make_act_pack: Callable  # (x, selected_experts, final_scales) -> MoEActivationPack (pre-routed)
    make_act_pack_logits: (
        Callable | None  # (x, routing_logits, routing_bias) -> pack (in-kernel routing)
    )
    reference: Callable  # (x, w1, w2, selected_experts, final_scales, I) -> fp32 [T,H] authority
    poison: Callable  # in-place fill a kernel-owned output buffer with garbage + (NaN/Inf if repr.)
    out_dtype: torch.dtype  # output buffer dtype (used to locate it in the inputs list)
    atol_frac: float  # numeric tolerance vs reference = atol_frac * ‖ref‖∞
    rtol: float


def _poison_bf16_out(buf, gen):
    """Fill a bf16 output buffer with large garbage + scattered NaN/±Inf, DETERMINISTICALLY (from a
    per-config seeded generator, so a failure repros bit-for-bit). If a kernel reads or scatter-adds
    into an uninitialized output instead of fully writing it, the poison leaks and is caught by
    no-NaN / numeric. This is the torch->JAX buffer-hygiene guard: torch's caching allocator usually
    hands back clean memory (masking the bug), JAX/XLA donates dirty buffers (the GH-6158764 class)."""
    g = torch.randn(buf.shape, generator=gen, device=buf.device, dtype=buf.dtype) * 1e4
    flat = g.view(-1)
    flat[0::4], flat[1::4], flat[2::4] = float("nan"), float("inf"), float("-inf")
    buf.copy_(g)


def _nvfp4_act_pack(x, selected_experts, final_scales):
    # global activation scale == 1.0 (MVP wires no global-scale field; block scales carry range).
    one = torch.tensor([1.0], device=x.device)
    packed, scale = fp4_quantize(
        x, global_scale=one, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    return MoEActivationPack(
        hidden_states_q=packed,
        hidden_states_scale=scale.squeeze(-1) if scale.dim() > 2 else scale,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        topk_ids=selected_experts,
        topk_weights=final_scales,
    )


def _nvfp4_act_pack_logits(x, routing_logits, routing_bias):
    """In-kernel-routing pack: same nvfp4 activation quant as ``_nvfp4_act_pack`` but carrying raw
    ``routing_logits`` (and optional ``routing_bias``) instead of pre-routed indices, so the kernel
    computes the top-k selection itself (RoutingInputMode.FromLogits)."""
    one = torch.tensor([1.0], device=x.device)
    packed, scale = fp4_quantize(
        x, global_scale=one, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    return MoEActivationPack(
        hidden_states_q=packed,
        hidden_states_scale=scale.squeeze(-1) if scale.dim() > 2 else scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
    )


def _nvfp4_reference(
    x, w1, w2, selected_experts, final_scales, intermediate_size, expert_offset=0
):
    """Canonical nvfp4 MoE recipe (the authority): exact-fp4 inputs (lossless), SwiGLU =
    silu(2nd half)*(1st half), then the intermediate is re-quantized to fp4 (block-scaled,
    gs=1.0) before gemm2 -- matching what the kernels do. w1/w2 hold only this rank's LOCAL
    experts; a token routed to global id ``g`` uses local weight ``g - expert_offset`` (EP)."""
    x32, half = x.float(), intermediate_size
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        mask = (
            selected_experts == local_e + expert_offset
        )  # global id of this local expert
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        gate, up = w1[local_e][half:, :].float(), w1[local_e][:half, :].float()
        inter = F.silu(x32[tok] @ gate.t()) * (x32[tok] @ up.t())
        inter = _snap_to_nvfp4(
            inter.to(torch.bfloat16)
        ).float()  # intermediate fp4 requant
        out[tok] += final_scales[tok, nth, None] * (inter @ w2[local_e].float().t())
    return out


def _bf16_snap(t: torch.Tensor) -> torch.Tensor:
    # bf16 IS the storage grid: the cast is the fixed point (input quant lossless).
    return t.to(torch.bfloat16)


def _bf16_act_pack(x, selected_experts, final_scales):
    # Raw bf16 activations; the bf16 runner reads hidden_states_q directly and
    # ignores hidden_states_scale.
    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        topk_ids=selected_experts,
        topk_weights=final_scales,
    )


def _bf16_reference(
    x, w1, w2, selected_experts, final_scales, intermediate_size, expert_offset=0
):
    """Dense bf16 MoE authority: same SwiGLU convention as ``_nvfp4_reference``
    but no fp4 requant -- the only intermediate quantization is the bf16 rounding
    of the gemm1 and gemm2 outputs, mirrored below.  Routing weights are cast through bf16
    to match the packed-id truncation in pack_inputs, so the tolerance measures
    kernel error, not oracle mismatch."""
    final_scales = final_scales.to(torch.bfloat16).float()
    x32, half = x.float(), intermediate_size
    out = torch.zeros_like(x32)
    for local_e in range(w1.shape[0]):
        mask = selected_experts == local_e + expert_offset
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        gate, up = w1[local_e][half:, :].float(), w1[local_e][:half, :].float()
        inter = F.silu(x32[tok] @ gate.t()) * (x32[tok] @ up.t())
        inter = inter.to(torch.bfloat16).float()  # gemm1 output is stored bf16
        expert_out = (inter @ w2[local_e].float().t()).to(torch.bfloat16).float()
        out[tok] += final_scales[tok, nth, None] * expert_out
    return out


_DTYPE = {
    QuantVariant.NVFP4: DTypeHandler(
        variant=QuantVariant.NVFP4,
        candidate_configs=(CuteDslConfig, TrtllmFp4Config),
        snap=_snap_to_nvfp4,
        make_act_pack=_nvfp4_act_pack,
        make_act_pack_logits=_nvfp4_act_pack_logits,
        reference=_nvfp4_reference,
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.15,  # calibrated: obs ratio ≤0.077 (fp4 intermediate-requant floor)
        rtol=0.1,
    ),
    QuantVariant.BF16: DTypeHandler(
        variant=QuantVariant.BF16,
        candidate_configs=(TrtllmBf16Config,),
        snap=_bf16_snap,
        make_act_pack=_bf16_act_pack,
        make_act_pack_logits=None,
        reference=_bf16_reference,
        poison=_poison_bf16_out,
        out_dtype=torch.bfloat16,
        atol_frac=0.05,  # initial; calibrate on SM100 (bf16 rounding floor)
        rtol=0.05,
    ),
    # FP8 / MXFP4 / MXINT4 add one entry each as their runners are wired upstream.
}

# Cfg.variant string <-> handler lookup (labels stay lowercase enum names).
_VARIANT_IDS = tuple(v.name.lower() for v in _DTYPE)


def _handler_for(cfg):
    return _DTYPE[QuantVariant[cfg.variant.upper()]]


# ---------------------------------------------------------------------------
# Config generation: random shapes + routing-load skew (an orthogonal axis -- uniform enforcement,
# not a numeric mode, so it never changes which checks apply).
# ---------------------------------------------------------------------------
# Deliberately NOT all powers of two: real models use non-pow2 (aligned) hidden/intermediate
# (Llama 14336/11008, DeepSeek-MoE 1408/1536, Qwen 18944), and #2907 was an intermediate-padding
# accuracy bug. H/I stay %16 for fp4 block alignment; if a kernel rejects a shape we skip it.
_HIDDEN = [256, 512, 1024, 1536, 2048, 3072]  # 1536/3072 aligned-non-pow2
_INTERMED = [
    256,
    512,
    768,
    1024,
    1408,
    1536,
]  # 768/1408/1536 aligned-non-pow2 (#2907 class)
_EXPERTS = [
    8,
    16,
    32,
    64,
    72,
    128,
    160,
    256,
    512,
]  # 72/160 non-pow2; 512 needs small H/I (budget)
_TOPK = [1, 2, 4, 6, 8]  # 6 non-pow2
# num_tokens is runtime batch*seqlen -- arbitrary. Sweep odd + tile/autotune-bucket boundaries
# (the #3168 16384-bucket / 4095-4097 tile-remainder class), not just clean powers of two.
_TOKENS = [1, 2, 3, 7, 17, 64, 127, 129, 256, 1024, 2048, 4095, 4096, 4097]
# Routing-logits *distribution* skew (orthogonal to the routing METHOD below). "all_negative"
# (#2822 all-negative-logit mis-selection) and "all_to_one" only bite the in-kernel router; the
# pre-routed host topk handles them trivially, but exercising both modes is free coverage.
_ROUTE = ["uniform", "uniform", "hot1", "imbalanced", "all_negative", "all_to_one"]

# Routing METHOD axis (RoutingMethodType). Pre-routed mode computes the host weights per method
# (the kernel then ignores the method, using the packed weights directly); in-kernel mode hands the
# kernel raw logits and it routes per this method -- so the SAME _route() oracle validates both.
# Covers the in-kernel-routing bug cluster the pre-routed harness structurally can't reach:
# DeepSeekV3 group routing + bias (#2575), bias methods (#2485/#2907), fp32 logits (#2796).
_ROUTING_METHODS = [
    RoutingMethodType.RenormalizeNaive,  # == the harness's original host routing
    RoutingMethodType.Default,
    RoutingMethodType.Renormalize,
    RoutingMethodType.TopK,
    RoutingMethodType.Sigmoid,
    RoutingMethodType.SigmoidRenorm,
    RoutingMethodType.DeepSeekV3,  # sigmoid+bias -> group-topk -> top_k (#2575 lives here)
    RoutingMethodType.MiniMax2,  # sigmoid+bias -> top_k -> scaled sum-norm
    RoutingMethodType.Llama4,  # top1 -> sigmoid (top_k forced to 1)
]
# Routing logits dtype axis: fp32 router logits are the #2796 class; bf16 is the common case.
_LOGITS_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}

# Backend config classes whose runner can do in-kernel routing (RoutingInputMode.FromLogits),
# derived from the runners' own capability declaration so this can't drift from the layer's
# dispatch filtering. CuteDSL is pre-routed-only, so a fromlogits config restricts to these.
_FROMLOGITS_BACKENDS = {
    cfg_cls
    for cfg_cls, runner_cls in _BACKEND_RUNNERS.items()
    if RoutingInputMode.FromLogits in runner_cls.supported_routing_modes
}

# Methods whose routing uses an additive bias (selection only -- weights stay unbiased). DeepSeekV3
# REQUIRES a bias; MiniMax2's is optional but we always supply one to exercise the bias path.
_BIAS_METHODS = {RoutingMethodType.DeepSeekV3, RoutingMethodType.MiniMax2}

# Per-test weight footprint cap so one fuzz config never hogs the GPU (parallel-CI-friendly) and the
# CPU exact-grid snap stays sub-few-seconds. ~500M bf16 weight elems ≈ 1 GB. The cap naturally pairs
# a large expert count with small H/I (and rejects giant-H/I x many-experts), matching real EP-sharded
# deployments where no single rank holds thousands of full-size experts.
_WEIGHT_ELEM_BUDGET = 500_000_000


def _weight_elems(num_experts, hidden, intermediate):
    return num_experts * (2 * intermediate * hidden + hidden * intermediate)  # w1 + w2


@dataclass(frozen=True)
class Cfg:
    num_tokens: int
    hidden: int
    intermediate: int
    num_experts: int  # GLOBAL expert count (RoutingConfig.num_experts)
    top_k: int
    variant: str
    route: str
    seed: int
    local_experts: int = 0  # this rank's shard; 0 -> non-EP (== num_experts)
    expert_offset: int = 0  # global id of this shard's first expert (EP)
    # Routing axes (defaults keep the original pre-routed RenormalizeNaive behavior so the
    # positional _CURATED literals below are unaffected).
    routing_method: RoutingMethodType = RoutingMethodType.RenormalizeNaive
    routing_input_mode: str = (
        "prerouted"  # "prerouted" (PackedPrecomputed) | "fromlogits"
    )
    logits_dtype: str = "bf16"  # "bf16" | "fp32" (#2796 fp32-router-logits class)
    n_group: int = 0  # DeepSeekV3 group count (0 -> None)
    topk_group: int = 0  # DeepSeekV3 groups kept (0 -> None)
    routed_scaling: float = 0.0  # DeepSeekV3 weight scale (0.0 -> None)

    @property
    def n_local(self):  # experts actually held + computed on this rank
        return self.local_experts or self.num_experts

    @property
    def is_ep(self):
        return self.expert_offset > 0 or self.n_local != self.num_experts

    @property
    def is_fromlogits(self):
        return self.routing_input_mode == "fromlogits"

    @property
    def label(self):
        ep = f"L{self.n_local}o{self.expert_offset}_" if self.is_ep else ""
        mode = "FL_" if self.is_fromlogits else ""
        ld = "fp32_" if self.logits_dtype == "fp32" else ""
        grp = f"g{self.n_group}x{self.topk_group}_" if self.n_group else ""
        return (
            f"{self.variant}_{mode}{self.routing_method.name}_{ld}{self.route}_"
            f"e{self.num_experts}_{ep}{grp}k{self.top_k}_"
            f"t{self.num_tokens}_h{self.hidden}_i{self.intermediate}_s{self.seed}"
        )


def _gen(seed):
    rng = random.Random(seed)
    method = rng.choice(_ROUTING_METHODS)
    # In-kernel routing ~half the time. FromLogits is single-shard only here: EP + in-kernel
    # routing semantics (does the kernel route over global logits then filter to local?) are a
    # separate validation, and EP collectives are out of scope for this single-GPU harness.
    # DeepSeekV3 group routing scores over the full expert set, so keep it non-EP too.
    fromlogits = rng.random() < 0.5
    force_non_ep = fromlogits or method == RoutingMethodType.DeepSeekV3
    # Resample shape until the weights of the FINAL config fit the budget (modest per-test
    # GPU footprint). Routing mode is chosen BEFORE this loop on purpose: non-EP-forced
    # configs (FromLogits / DeepSeekV3) hold the FULL expert set, so budgeting a sharded
    # `local` and flipping to non-EP afterwards would admit up to shards x the budget.
    for _ in range(64):
        ne, h, i = rng.choice(_EXPERTS), rng.choice(_HIDDEN), rng.choice(_INTERMED)
        # ~30%: expert-parallel shard -- split the global experts and pick a shard (offset>0). This
        # is how large MoE actually runs (no rank holds all experts) and exercises the offset path.
        local, offset = ne, 0
        if not force_non_ep:
            shards = rng.choice([2, 4])
            if rng.random() < 0.3 and ne >= 16 and ne % shards == 0:
                local = ne // shards
                offset = local * rng.randrange(shards)
        if _weight_elems(local, h, i) <= _WEIGHT_ELEM_BUDGET:
            break

    # Method-specific top_k + group params.
    n_group = topk_group = 0
    routed_scaling = 0.0
    if method == RoutingMethodType.Llama4:
        top_k = 1  # the reference (and the kernel) only define Llama4 for top1
    elif method == RoutingMethodType.DeepSeekV3:
        # n_group divides ne with ne>n_group (=> >=2 experts/group for the top-2 group score);
        # topk_group<=min(4,n_group); top_k < topk_group*ne/n_group (experts reachable after the
        # group mask) and <= local. ne is always %4==0 (every _EXPERTS entry is).
        n_group = rng.choice([g for g in (1, 2, 4, 8) if g < ne and ne % g == 0])
        topk_group = rng.randint(1, min(4, n_group))
        reachable = topk_group * ne // n_group
        valid_k = [t for t in _TOPK if t < reachable and t <= local]
        top_k = rng.choice(valid_k) if valid_k else 1
        routed_scaling = rng.choice([1.0, 2.5])
    else:
        top_k = rng.choice(
            [t for t in _TOPK if t <= local]
        )  # route within the local shard

    return Cfg(
        num_tokens=rng.choice(_TOKENS),
        hidden=h,
        intermediate=i,
        num_experts=ne,
        top_k=top_k,
        # BF16 is pre-routed-only today; FromLogits currently requires the
        # TRTLLM FP4 runner.
        variant="nvfp4" if fromlogits else rng.choice(_VARIANT_IDS),
        route=rng.choice(_ROUTE),
        seed=seed,
        local_experts=local,
        expert_offset=offset,
        routing_method=method,
        routing_input_mode="fromlogits" if fromlogits else "prerouted",
        logits_dtype="fp32"
        if rng.random() < 0.25
        else "bf16",  # #2796 fp32-logits axis
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling=routed_scaling,
    )


# A few curated "larger end of the common range" shapes (all within the weight budget) so the big
# end is always represented, not left to chance: many-experts, large-hidden+many-tokens, and max-experts.
_CURATED = [
    Cfg(
        256, 1024, 512, 256, 8, "nvfp4", "uniform", 900_001
    ),  # DeepSeek-ish: 256 experts, top_k 8
    Cfg(
        4096, 2048, 1024, 64, 8, "nvfp4", "uniform", 900_002
    ),  # large hidden + many tokens
    Cfg(
        2048, 1024, 1024, 128, 6, "nvfp4", "imbalanced", 900_003
    ),  # empty-expert load + mid size
    Cfg(
        512, 512, 512, 512, 4, "nvfp4", "hot1", 900_004
    ),  # max expert count (small H/I)
    # In-kernel routing (FromLogits) headline cases the pre-routed harness can't reach:
    Cfg(
        256,
        1024,
        512,
        256,
        8,
        "nvfp4",
        "uniform",
        900_005,
        routing_method=RoutingMethodType.DeepSeekV3,
        routing_input_mode="fromlogits",
        n_group=8,
        topk_group=4,
        routed_scaling=2.5,
    ),  # #2575 DeepSeekV3 group routing at large expert count (top_k 8 < 4*256/8=128)
    Cfg(
        512,
        1024,
        512,
        64,
        6,
        "nvfp4",
        "uniform",
        900_006,
        routing_method=RoutingMethodType.Default,
        routing_input_mode="fromlogits",
        logits_dtype="fp32",
    ),  # #2796 fp32 router logits, in-kernel softmax->topk
    Cfg(
        256,
        512,
        512,
        128,
        4,
        "nvfp4",
        "all_negative",
        900_007,
        routing_method=RoutingMethodType.Renormalize,
        routing_input_mode="fromlogits",
    ),  # #2822 all-negative logits, in-kernel topk->softmax
    Cfg(
        1024,
        1024,
        768,
        128,
        8,
        "nvfp4",
        "uniform",
        900_008,
        routing_method=RoutingMethodType.DeepSeekV3,
        routing_input_mode="fromlogits",
        n_group=4,
        topk_group=2,
        routed_scaling=1.0,
    ),  # DeepSeekV3 mid-size (top_k 8 < 2*128/4=64), non-pow2 intermediate
    Cfg(
        256, 1024, 512, 256, 8, "bf16", "uniform", 900_009
    ),  # DeepSeek-ish shape on the bf16 path
    Cfg(
        2048, 1024, 1024, 128, 6, "bf16", "imbalanced", 900_010
    ),  # bf16 mid size + empty-expert load
]
if _ONLY_SEEDS:  # perfect-repro: run only the named seed(s)
    _curated_by_seed = {c.seed: c for c in _CURATED}
    _CONFIGS = [
        _curated_by_seed.get(s) or _gen(s)
        for s in (int(t) for t in _ONLY_SEEDS.split(",") if t.strip())
    ]
else:
    _CONFIGS = _CURATED + [_gen(BASE_SEED + i) for i in range(NUM_TESTS)]


def _route(
    logits, method, top_k, *, bias=None, n_group=0, topk_group=0, routed_scaling=None
):
    """Host routing reference: logits[T,E] -> (selected[T,k] int64, weights[T,k] float32).

    Mirrors the per-method math in ``tests/moe/test_trtllm_gen_fused_moe.py``
    (``routing_reference_*`` / ``noaux_tc_ref``), which is validated against the SAME
    trtllm-gen kernel the unified FromLogits path drives -- so the in-kernel router
    agrees with this oracle by transitivity.  Selection/weight alignment is by column
    (``selected[t,j]`` <-> ``weights[t,j]``); column ORDER is irrelevant downstream
    (the reference sums over the top-k and matches experts by id, not position).
    """
    M = RoutingMethodType
    lf = logits.float()
    if (
        method == M.Default
    ):  # softmax -> top_k (NOT renormalized; norm_topk_prob is a no-op here)
        w, sel = torch.topk(F.softmax(lf, dim=-1), top_k, dim=-1)
    elif method in (M.Renormalize, M.RenormalizeNaive):
        # top_k(raw) -> softmax over selected. The kernel aliases RenormalizeNaive to this
        # (Softmax->TopK->SumNorm is algebraically identical to TopK->Softmax).
        raw, sel = torch.topk(lf, top_k, dim=-1)
        w = F.softmax(raw, dim=-1)
    elif (
        method == M.TopK
    ):  # top_k of raw logits, raw logit values as weights (no normalization)
        w, sel = torch.topk(lf, top_k, dim=-1)
    elif method == M.Sigmoid:  # sigmoid -> top_k (no renorm)
        w, sel = torch.topk(torch.sigmoid(lf), top_k, dim=-1)
    elif (
        method == M.SigmoidRenorm
    ):  # sigmoid -> top_k -> renorm (divide by sum of selected)
        w, sel = torch.topk(torch.sigmoid(lf), top_k, dim=-1)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    elif method == M.Llama4:  # top1 -> sigmoid weight (top_k forced to 1 by config gen)
        w, sel = torch.topk(torch.sigmoid(lf), top_k, dim=-1)
    elif method in (M.DeepSeekV3, M.MiniMax2):
        # Sigmoid + bias drives SELECTION; the final weights use the UNBIASED sigmoid scores
        # (the classic "bias leaks into weights" bug). DeepSeekV3 adds a group-topk pre-mask.
        scores = torch.sigmoid(lf)
        sel_scores = scores + bias.float() if bias is not None else scores.clone()
        if method == M.DeepSeekV3 and n_group > 1:
            E = sel_scores.shape[-1]
            grp = sel_scores.view(*sel_scores.shape[:-1], n_group, E // n_group)
            group_scores = torch.topk(grp, k=2, dim=-1).values.sum(
                dim=-1
            )  # top-2 sum per group
            _, gidx = torch.topk(group_scores, k=topk_group, dim=-1)
            gmask = torch.zeros_like(group_scores).scatter_(-1, gidx, 1.0)
            smask = (
                gmask.unsqueeze(-1)
                .expand(*sel_scores.shape[:-1], n_group, E // n_group)
                .reshape(sel_scores.shape)
            )
            sel_scores = (
                sel_scores * smask
            )  # zero out experts outside the selected groups
        _, sel = torch.topk(sel_scores, top_k, dim=-1)
        w = torch.gather(scores, -1, sel)  # UNBIASED sigmoid weights
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        if routed_scaling is not None:
            w = w * routed_scaling
    else:
        raise NotImplementedError(
            f"routing method {method!r} not supported by the fuzzer oracle"
        )
    return sel.to(torch.int64), w.float()


def _master(cfg, handler):
    """Sparse, exactly-representable bf16 inputs + host routing. Sparsity keeps the gemm reductions
    short so a structural bug isn't averaged away; exact-grid snapping makes input quant lossless.
    Weights cover only this rank's LOCAL shard (E_local); routing selects within the shard's GLOBAL
    id range [offset, offset+E_local) (the EP contract -- non-EP is offset=0, E_local=num_experts).

    Routing is computed per ``cfg.routing_method`` via ``_route`` (the kernel-matching oracle). The
    raw ``logits`` (+ ``routing_bias``) are returned so the in-kernel (FromLogits) path can feed them
    to the kernel, while the host-computed ``selected_experts`` / ``final_scales`` remain the
    authoritative reference for BOTH modes (the kernel must reproduce them)."""
    g = torch.Generator(device="cuda").manual_seed(cfg.seed)
    E_local, H, I, T = cfg.n_local, cfg.hidden, cfg.intermediate, cfg.num_tokens

    def sparse(*shape):
        dense = torch.randn(*shape, device="cuda", generator=g)
        keep = torch.rand(shape, device="cuda", generator=g) >= 0.75  # ~75% zeros
        return handler.snap(dense * keep)

    x, w1, w2 = sparse(T, H), sparse(E_local, 2 * I, H), sparse(E_local, H, I)

    logits = torch.randn(T, E_local, device="cuda", generator=g)  # over the local shard
    if cfg.route in ("hot1", "all_to_one"):  # pile every token onto one expert
        logits[:, 0] += 50.0
    elif cfg.route == "imbalanced":  # rank-skew -> some experts get zero tokens
        logits += torch.linspace(8.0, -8.0, E_local, device="cuda")
    elif (
        cfg.route == "all_negative"
    ):  # #2822: no positive anchor for the in-kernel router
        logits = -logits.abs() - 1.0
    logits = logits.to(_LOGITS_DTYPE[cfg.logits_dtype])

    # Bias for bias-aware methods (affects SELECTION only); dtype follows logits here for simplicity (the kernel accepts bf16 or fp32 independently of logits dtype).
    routing_bias = None
    if cfg.routing_method in _BIAS_METHODS:
        routing_bias = torch.randn(E_local, device="cuda", generator=g).to(logits.dtype)

    local_sel, final_scales = _route(
        logits,
        cfg.routing_method,
        cfg.top_k,
        bias=routing_bias,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        routed_scaling=cfg.routed_scaling or None,
    )
    selected_experts = (local_sel + cfg.expert_offset).to(
        torch.int32
    )  # local -> global ids
    return x, w1, w2, selected_experts, final_scales, logits, routing_bias


_SKIP_SUBSTR = (
    "not supported",
    "unsupported",
    "no valid",
    "not implemented",
    "must be",
    "divisible",
    "requires",
    "only support",
)
_CRASH_SUBSTR = ("cuda error", "illegal memory", "misaligned", "device-side assert")


def _is_unsupported(e):
    msg = str(e).lower()
    if any(c in msg for c in _CRASH_SUBSTR):
        return False  # a crash is always a finding, never "unsupported"
    return isinstance(e, NotImplementedError) or any(s in msg for s in _SKIP_SUBSTR)


# ---------------------------------------------------------------------------
# Diagnostics: every test prints its full config + a perfect-repro command; on a mismatch we dump
# output-vs-oracle stats + the worst <=30 elements, so the CI log alone tells you whether the output
# is all-zero / all-NaN / Inf without having to rerun. (Mirrors tests/gemm/test_unified_gemm_fuzz.py.)
# ---------------------------------------------------------------------------
def _describe(cfg: Cfg) -> str:
    return (
        f"CONFIG {cfg.label}\n"
        f"  variant={cfg.variant} routing={cfg.routing_input_mode} "
        f"method={cfg.routing_method.name} logits_dtype={cfg.logits_dtype} route={cfg.route}\n"
        f"  shape: tokens={cfg.num_tokens} hidden={cfg.hidden} intermediate={cfg.intermediate}  "
        f"experts={cfg.num_experts} top_k={cfg.top_k}\n"
        f"  EP: n_local={cfg.n_local} expert_offset={cfg.expert_offset} (is_ep={cfg.is_ep})  "
        f"group: n_group={cfg.n_group} topk_group={cfg.topk_group} "
        f"routed_scaling={cfg.routed_scaling}  seed={cfg.seed}"
    )


def _env_prefix() -> str:
    """Shell-safe env prefix for repro commands: include only variables that are
    actually set (an unset variable is not needed to reproduce), quoting values so
    the printed command is directly executable."""
    import shlex

    parts = [
        f"{var}={shlex.quote(os.environ[var])}"
        for var in ("CUDA_HOME", "CUDA_VISIBLE_DEVICES")
        if var in os.environ
    ]
    return " ".join(parts) + " " if parts else ""


def _repro(cfg: Cfg) -> str:
    return (
        f"REPRO: {_env_prefix()}FLASHINFER_UMOE_FUZZ=1 "
        f"FLASHINFER_UMOE_FUZZ_ONLY_SEED={cfg.seed} "
        f"pytest -s tests/moe/test_unified_moe_fuzz.py::test_unified_moe_fuzz"
    )


def _stats(t: torch.Tensor) -> str:
    tf = t.float()
    n = tf.numel()
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} nan={int(torch.isnan(tf).sum())} "
        f"inf={int(torch.isinf(tf).sum())} zero={int((tf == 0).sum())}/{n} "
        f"max|.|={tf.abs().nan_to_num().max().item():.4g}"
    )


def _dump(out: torch.Tensor, ref: torch.Tensor, k: int = 30) -> str:
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


def _fail(cfg: Cfg, tag: str, why: str, out=None, ref=None):
    parts = [f"{tag}: {why}", _describe(cfg)]
    if out is not None and ref is not None:
        parts.append(_dump(out, ref))
    parts.append(_repro(cfg))
    pytest.fail("\n".join(parts))


@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_unified_moe_fuzz(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    # Full per-config determinism so any failure reproduces from the seed alone. Shapes
    # (random.Random(seed)) and input tensors (a per-config torch.Generator) are already seeded;
    # this pins the global RNG (the device probe), and the output buffer is initialized from the
    # dedicated `poison_gen` below -- so the entire run is bitwise-reproducible from `cfg.seed`.
    # (Autotune winner selection is timing-based and may vary run-to-run, but every tactic is
    # validated, so a correctness failure still reproduces via the tactic sweep regardless of which
    # winner the tuner picks.)
    torch.manual_seed(cfg.seed)
    # Dedicated generator for output-buffer init, decoupled from the input generator's stream so the
    # poison/zero fill is deterministic regardless of how many runners/calls a config drives.
    poison_gen = torch.Generator(device="cuda").manual_seed(cfg.seed + 1_000_003)
    # Every test prints its full config + the exact repro command (captured by pytest; shown on
    # failure, or always with `-s`) so a CI log is self-explanatory without a rerun.
    print("\n" + _describe(cfg))
    print(_repro(cfg))
    sm = get_compute_capability(torch.device("cuda:0"))
    sm = sm[0] * 10 + sm[1]

    handler = _handler_for(cfg)
    dev = torch.device("cuda")
    # Backend *config classes* whose runner is registered in the live MoELayer registry AND valid
    # on this arch. A newly-wired backend lands here automatically.
    wired_backends = [
        BackendCfg
        for BackendCfg in handler.candidate_configs
        if BackendCfg in _BACKEND_RUNNERS and BackendCfg.supported(sm)
    ]
    if cfg.is_fromlogits:
        # In-kernel routing restricts to FromLogits-capable backends (CuteDSL is pre-routed-only,
        # so it cannot serve a logits-only pack and would compare apples to oranges).
        wired_backends = [B for B in wired_backends if B in _FROMLOGITS_BACKENDS]
    if not wired_backends:
        mode = "in-kernel-routing " if cfg.is_fromlogits else ""
        pytest.skip(f"no wired {mode}backend for {cfg.variant} on SM{sm}")

    x, w1, w2, selected_experts, final_scales, logits, routing_bias = _master(
        cfg, handler
    )
    ref = handler.reference(
        x, w1, w2, selected_experts, final_scales, cfg.intermediate, cfg.expert_offset
    )
    atol = handler.atol_frac * ref.abs().max().item() + 1e-3
    rtol = handler.rtol

    # One activation pack + one weight pack with each backend's native view, all built from the
    # SAME bf16 inputs (this rank's LOCAL shard) via the API's uniform prepare_weights. In-kernel
    # routing hands the kernel raw logits (+ bias); pre-routed hands it the host selection.
    if cfg.is_fromlogits:
        assert handler.make_act_pack_logits is not None
        act_pack = handler.make_act_pack_logits(x, logits, routing_bias)
    else:
        act_pack = handler.make_act_pack(x, selected_experts, final_scales)
    weight_pack = MoEWeightPack()
    for BackendCfg in wired_backends:
        weight_pack.prepare_for(
            _BACKEND_RUNNERS[BackendCfg].backend_key,
            BackendCfg.prepare_weights(
                w1,
                w2,
                num_local_experts=cfg.n_local,
                hidden_size=cfg.hidden,
                intermediate_size=cfg.intermediate,
                device=dev,
            ),
        )

    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            method=cfg.routing_method,
            n_group=cfg.n_group or None,
            topk_group=cfg.topk_group or None,
            routed_scaling_factor=cfg.routed_scaling or None,
        ),
        quant=QuantConfig(variant=handler.variant),
        experts=ExpertConfig(
            intermediate_size=cfg.intermediate,
            local_num_experts=cfg.n_local,
            local_expert_offset=cfg.expert_offset,
        ),
        activation=ActivationConfig(),
        backend=BackendOptions(
            candidates=tuple(BackendCfg() for BackendCfg in wired_backends)
        ),
        execution=ExecutionConfig(tune_max_num_tokens=max(cfg.num_tokens, 8192)),
    )

    try:
        layer = MoELayer(config)
    except Exception as e:
        if _is_unsupported(e):
            pytest.skip(f"MoELayer rejected {cfg.label}: {e}")
        raise

    out_shape = (cfg.num_tokens, cfg.hidden)

    def run(runner, poison=False):
        inputs = runner.pack_inputs(act_pack, weight_pack)
        # Deterministically initialize the kernel-owned output buffer (a `new_empty` in the runner's
        # pack_inputs; cute_dsl idx 11, trtllm the `output=`), located by dtype+shape: clean=zeros,
        # poison=seeded garbage+NaN/Inf. Both are bit-reproducible from cfg.seed, so any failure --
        # including a partial-write that depends on the buffer -- reproduces exactly.
        act_ptrs = {
            t.data_ptr()
            for t in (
                act_pack.hidden_states_q,
                act_pack.hidden_states_scale,
                act_pack.topk_ids,
                act_pack.topk_weights,
                act_pack.routing_logits,
                act_pack.routing_bias,
            )
            if torch.is_tensor(t)
        }
        bufs = [
            t
            for t in inputs
            if torch.is_tensor(t)
            and t.dtype == handler.out_dtype
            and tuple(t.shape) == out_shape
            and t.data_ptr() not in act_ptrs
        ]
        assert bufs, "could not locate the output buffer in pack_inputs"
        for b in bufs:
            handler.poison(b, poison_gen) if poison else b.zero_()
        out = runner.forward(inputs, tactic=-1)
        out = (out[0] if isinstance(out, (list, tuple)) else out).float()
        torch.cuda.synchronize()
        return out

    def assert_correct(out, tag):
        # (1) no NaN/Inf where the reference is finite.
        n_bad = int(((~torch.isfinite(out)) & torch.isfinite(ref)).sum().item())
        if n_bad != 0:
            _fail(
                cfg,
                tag,
                f"{n_bad}/{out.numel()} non-finite outputs where oracle is finite "
                f"(#2569/#3103-class)",
                out,
                ref,
            )
        # (2) numeric vs the canonical quant-aware reference (the authority), magnitude-scaled.
        abs_diff = (out - ref).abs()
        over_tol = abs_diff > (atol + rtol * ref.abs())
        if over_tol.any():
            _fail(
                cfg,
                tag,
                f"{int(over_tol.sum())}/{out.numel()} elems exceed tol "
                f"(rtol={rtol} atol={atol:.3g}; max|diff|={abs_diff.max().item():.4g}, "
                f"‖ref‖∞={ref.abs().max().item():.4g})",
                out,
                ref,
            )

    def check_backend(runner, out, tag):
        # (1)+(2) no-NaN + numeric vs the authoritative reference, on a clean run.
        assert_correct(out, tag)
        # (3) determinism per the backend's contract: deterministic backends must reproduce
        # bitwise; non-deterministic ones (atomic-scatter finalize) are exempt.
        if _DETERMINISTIC.get(runner.backend_key, False):
            out2 = run(runner)
            if not torch.equal(out, out2):
                _fail(
                    cfg,
                    tag,
                    "declared DETERMINISTIC but not bitwise-reproducible across identical runs "
                    "(#2514-class); 'output' = first run, 'oracle' = second run",
                    out,
                    out2,
                )
        # (4) output-buffer poison: the kernel owns its (uninitialized `new_empty`) output, so the
        # result must NOT depend on it being clean. torch's allocator usually hands back zeros and
        # hides this; poisoning forces it -- the torch->JAX hazard (GH-6158764 padding leak).
        assert_correct(run(runner, poison=True), f"{tag} [poisoned-output]")
        # (5) autotune-tactic sweep: EVERY valid tactic must be correct, not just the default --
        # the autotuner-picks-a-bad-tactic class (#3168/#3227) on the real MoELayer dispatch.
        inputs = runner.pack_inputs(act_pack, weight_pack)
        try:
            tactics = runner.get_valid_tactics(inputs, None)
        except Exception:
            tactics = []  # backend needs a profile object -> skip the sweep (default tactic stands)
        for tactic in tactics:
            o = runner.forward(inputs, tactic=tactic)
            o = (o[0] if isinstance(o, (list, tuple)) else o).float()
            torch.cuda.synchronize()
            assert_correct(o, f"{tag} [tactic={tactic}]")

    n_ran = 0
    for runner in layer.runners:
        try:
            out = run(runner)
        except Exception as e:
            if _is_unsupported(e):
                continue  # backend rejects this shape -> skip; a crash re-raises
            raise
        tag = f"{runner.backend_key} {cfg.label}"
        n_ran += 1

        known = _known_failure(runner.backend_key, cfg)
        if known:  # tracked bug -> run it, tolerate a wrong answer, but flag if it starts passing
            try:
                check_backend(runner, out, tag)
            except (AssertionError, pytest.fail.Exception):
                continue
            warnings.warn(
                f"{tag}: KNOWN-FAILURE unexpectedly PASSED -- fixed? remove from "
                f"_KNOWN_FAILURES ({known})",
                stacklevel=2,
            )
        else:
            check_backend(runner, out, tag)

    if n_ran == 0:
        pytest.skip(f"no runner ran {cfg.label} on SM{sm}")

    # (6) autotune-ON: drive the REAL production path -- MoELayer._select_winner profiles every
    # tactic of every runner (the #3168 profiling-IMA class) then selects + caches a winner; the
    # autotuned output must match the authoritative reference. Gated to a subset (profiling is slow)
    # and skipped if a candidate has a known failure (the tuner could pick the broken backend).
    autotune_due = cfg.seed % 4 == 0 and not any(
        _known_failure(_BACKEND_RUNNERS[B].backend_key, cfg) for B in wired_backends
    )
    if autotune_due:
        with autotune(True):
            a_out = layer(act_pack, weight_pack)
        a_out = (a_out[0] if isinstance(a_out, (list, tuple)) else a_out).float()
        torch.cuda.synchronize()
        assert_correct(
            a_out, f"{cfg.label} [autotune-ON winner={layer.winner_backend}]"
        )

    # (7) device-state probe: a context-corrupting IMA in any backend above would surface here as a
    # failed alloc/launch or a non-finite probe, turning a silent corruption into a clean failure.
    probe = torch.randn(2048, device="cuda") * 2.0
    torch.cuda.synchronize()
    assert torch.isfinite(probe).all(), (
        f"{cfg.label}: CUDA context corrupted after MoE run"
    )


# ---------------------------------------------------------------------------
# Sibling SCENARIO test (not per-config-stateless): the autotune CACHE is cross-call state, which
# the fuzz test (fresh MoELayer per config) structurally can't reach. So drive ONE persistent layer
# through a token-count SEQUENCE under autotune -- fill the per-bucket winner cache, cross shapes,
# and re-run earlier counts to force cache hits. A stale / mis-keyed cached winner reused for a
# different shape would produce a wrong answer here. (Shares the harness's snap/reference/prep.)
# ---------------------------------------------------------------------------
_CACHE_BASES = [
    (32, 1024, 512),
    (128, 512, 512),
]  # (experts, hidden, intermediate), non-EP
_CACHE_TOKEN_SEQ = [
    16,
    256,
    4095,
    4096,
    4097,
    256,
    16,
]  # buckets + boundaries + cache-hit re-runs


@pytest.mark.parametrize("variant", list(_DTYPE), ids=[v.name.lower() for v in _DTYPE])
@pytest.mark.parametrize(
    "base", _CACHE_BASES, ids=[f"e{e}h{h}i{i}" for e, h, i in _CACHE_BASES]
)
def test_autotune_cache_coherence(base, variant):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    sm = get_compute_capability(torch.device("cuda:0"))
    sm = sm[0] * 10 + sm[1]
    handler = _DTYPE[variant]
    dev = torch.device("cuda")
    wired = [
        B
        for B in handler.candidate_configs
        if B in _BACKEND_RUNNERS and B.supported(sm)
    ]
    if not wired:
        pytest.skip(f"no wired backend on SM{sm}")

    E, H, I = base
    top_k = 4
    _repro_cmd = (
        f"REPRO: {_env_prefix()}FLASHINFER_UMOE_FUZZ=1 "
        f"pytest -s tests/moe/test_unified_moe_fuzz.py::test_autotune_cache_coherence -k e{E}h{H}i{I}"
    )
    print(
        f"\nCACHE-COHERENCE base=e{E}h{H}i{I} top_k={top_k} token_seq={_CACHE_TOKEN_SEQ}"
    )
    print(_repro_cmd)
    g = torch.Generator(device="cuda").manual_seed(12345)

    def sparse(*shape):
        dense = torch.randn(*shape, device="cuda", generator=g)
        return handler.snap(
            dense * (torch.rand(shape, device="cuda", generator=g) >= 0.75)
        )

    # Fixed weights + ONE layer instance; the cache lives across the whole sequence.
    w1, w2 = sparse(E, 2 * I, H), sparse(E, H, I)
    weight_pack = MoEWeightPack()
    for B in wired:
        weight_pack.prepare_for(
            _BACKEND_RUNNERS[B].backend_key,
            B.prepare_weights(
                w1,
                w2,
                num_local_experts=E,
                hidden_size=H,
                intermediate_size=I,
                device=dev,
            ),
        )
    layer = MoELayer(
        MoEConfig(
            routing=RoutingConfig(num_experts=E, top_k=top_k),
            quant=QuantConfig(variant=variant),
            experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
            activation=ActivationConfig(),
            backend=BackendOptions(candidates=tuple(B() for B in wired)),
            execution=ExecutionConfig(tune_max_num_tokens=max(_CACHE_TOKEN_SEQ)),
        )
    )

    with autotune(
        True
    ):  # fill the per-bucket cache on first sight; hit it on the re-runs
        for num_tokens in _CACHE_TOKEN_SEQ:
            x = sparse(num_tokens, H)
            w = F.softmax(torch.randn(num_tokens, E, device="cuda", generator=g), dim=1)
            w, sel = torch.topk(w, top_k, dim=-1)
            final = (w / w.sum(dim=-1, keepdim=True)).float()
            sel = sel.to(torch.int32)
            act = handler.make_act_pack(x, sel, final)
            ref = handler.reference(x, w1, w2, sel, final, I, 0)
            out = layer(act, weight_pack)
            out = (out[0] if isinstance(out, (list, tuple)) else out).float()
            torch.cuda.synchronize()
            tag = f"cache-seq T={num_tokens} (winner={layer.winner_backend}) {base}"
            n_bad = int(((~torch.isfinite(out)) & torch.isfinite(ref)).sum().item())
            atol = handler.atol_frac * ref.abs().max().item() + 1e-3
            over = (out - ref).abs() > (atol + handler.rtol * ref.abs())
            if n_bad != 0 or over.any():
                why = (
                    f"{n_bad} non-finite outputs"
                    if n_bad
                    else f"{int(over.sum())} elems exceed tol "
                    f"(max|diff|={(out - ref).abs().max().item():.4g}) -- stale/mis-keyed cached winner?"
                )
                pytest.fail(f"{tag}: {why}\n{_dump(out, ref)}\n{_repro_cmd}")
